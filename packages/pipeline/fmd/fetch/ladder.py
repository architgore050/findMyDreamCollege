"""The fetch ladder: httpx -> curl_cffi -> MCP browser -> Firecrawl Local -> quarantine.

Escalation happens **only** on a block signal (403/429/503 or a
``cf-mitigated`` response header). A 404/410 is terminal and is returned from
the first tier. Every result carries the tier, elapsed time, final URL and the
robots decision so the snapshot layer can record how a page was obtained.

The stealth tiers are external processes, not in-process engines: the MCP
browser tier (:mod:`fmd.fetch.mcp_browser`) drives ``invisible-playwright-mcp``
over stdio, and the Firecrawl Local tier (:mod:`fmd.fetch.firecrawl_local`)
talks to a locally served ``firecrawl-local`` with ``FIRECRAWL_CLOUD=off``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import urlencode

from fmd.fetch.failure_log import (
    FetchFailure,
    failure_kind_for,
    institution_hint,
    is_failure,
    record_failure,
)
from fmd.fetch.politeness import (
    PolitenessManager,
    SystemClock,
    parse_retry_after,
)
from fmd.fetch.robots import RobotsCache
from fmd.fetch.types import (
    BLOCK_STATUSES,
    FetchRequest,
    FetchResult,
    TierFetcher,
    TierUnavailable,
    UnsupportedReplay,
)
from fmd.net.policy import (
    DEFAULT_RETRY,
    DEFAULT_TIMEOUTS,
    NetworkFault,
    NetworkGuard,
    NetworkUnavailable,
    RetryPolicy,
    Timeouts,
    classify_exception,
    make_client,
    request_with_retries,
)

logger = logging.getLogger(__name__)

TIER_HTTPX = "httpx"
TIER_CURL_CFFI = "curl_cffi"
TIER_MCP_BROWSER = "invisible_playwright_mcp"
TIER_FIRECRAWL_LOCAL = "firecrawl_local"
TIER_QUARANTINE = "quarantine"
TIER_ROBOTS = "robots"


def _lower_headers(headers) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in dict(headers).items()}


def _set_header(headers: dict[str, str], name: str, value: str) -> None:
    """Case-insensitive overwrite; caller casing is otherwise preserved."""
    lowered = name.lower()
    for key in [key for key in headers if key.lower() == lowered]:
        del headers[key]
    headers[name] = value


def _merged_headers(request: FetchRequest) -> dict[str, str]:
    """Caller headers plus conditional validators.

    Conditional headers are *merged into* the replay shape: a POST's
    ``Content-Type``/``Referer``/custom headers survive revalidation, and an
    explicit validator already on the caller's headers is replaced by the
    stored (newer) one only when the request carries one.
    """
    headers = {str(key): str(value) for key, value in dict(request.headers or {}).items()}
    if request.etag:
        _set_header(headers, "If-None-Match", request.etag)
    if request.last_modified:
        _set_header(headers, "If-Modified-Since", request.last_modified)
    if request.referer:
        _set_header(headers, "Referer", request.referer)
    return headers


def _method(request: FetchRequest) -> str:
    return str(request.method or "GET").upper()


def _payload(request: FetchRequest) -> tuple[bytes | None, str | None]:
    """Encode a request body and its implied content type, if any.

    ``json_body`` wins over ``body``. A mapping body is form-urlencoded (the
    persisted NABH shape); bytes/str pass through untouched so a caller that
    already encoded a payload is not re-encoded.
    """
    if request.json_body is not None:
        return (
            json.dumps(request.json_body, separators=(",", ":")).encode("utf-8"),
            "application/json",
        )
    body = request.body
    if body is None:
        return None, None
    if isinstance(body, (bytes, bytearray, memoryview)):
        return bytes(body), None
    if isinstance(body, str):
        return body.encode("utf-8"), None
    if isinstance(body, Mapping):
        return urlencode(dict(body), doseq=True).encode("utf-8"), (
            "application/x-www-form-urlencoded"
        )
    raise TypeError(f"unsupported request body type {type(body).__name__}")


def _prepared(request: FetchRequest) -> tuple[str, dict[str, str], bytes | None]:
    """Method, merged headers and encoded payload for a replaying tier."""
    method = _method(request)
    headers = _merged_headers(request)
    payload, content_type = _payload(request)
    if content_type is not None and not any(key.lower() == "content-type" for key in headers):
        headers["Content-Type"] = content_type
    return method, headers, payload


class HttpxFetcher:
    """Primary tier. Pooled, HTTP/2 when the server offers it.

    Timeouts and retries come from :mod:`fmd.net.policy`; this tier never
    constructs a bare client. Status retries are disabled here on purpose: a
    429/503 is a *block signal* the ladder escalates on, not something to
    hammer with the same fingerprint.
    """

    name = TIER_HTTPX

    def __init__(
        self,
        client=None,
        *,
        timeout: float | Timeouts | None = None,
        transport=None,
        retry: RetryPolicy | None = None,
        guard: NetworkGuard | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        if timeout is None:
            self.timeouts = DEFAULT_TIMEOUTS
        elif isinstance(timeout, Timeouts):
            self.timeouts = timeout
        else:
            # Backwards-compatible scalar: it is the read timeout.
            self.timeouts = replace(DEFAULT_TIMEOUTS, read=float(timeout))
        self.timeout = self.timeouts.read
        self.transport = transport
        # Two attempts total: the ladder already has three tiers, so a long
        # internal retry budget would multiply into a hammer.
        self.retry = retry or replace(DEFAULT_RETRY, max_attempts=2)
        self.guard = guard
        self._sleeper = sleeper or time.sleep

    def _get_client(self):
        if self._client is None:
            from fmd.settings import get_settings

            self._client = make_client(
                timeouts=self.timeouts,
                transport=self.transport,
                headers={"User-Agent": get_settings().user_agent},
            )
        return self._client

    def fetch(self, request: FetchRequest) -> FetchResult:
        method, headers, payload = _prepared(request)
        response = request_with_retries(
            self._get_client(),
            method,
            request.url,
            retry=self.retry,
            retry_statuses=(),
            headers=headers,
            content=payload,
            guard=self.guard,
            sleep=self._sleeper,
        )
        return FetchResult(
            url=request.url,
            final_url=str(response.url),
            status=response.status_code,
            headers=_lower_headers(response.headers),
            body=response.content,
            fetcher_tier=self.name,
            method=method,
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None


class CurlCffiFetcher:
    """Second tier: browser TLS fingerprint impersonation."""

    name = TIER_CURL_CFFI

    def __init__(
        self,
        *,
        impersonate: str = "chrome",
        timeout: float = 25.0,
        guard: NetworkGuard | None = None,
    ) -> None:
        self.impersonate = impersonate
        self.timeout = timeout
        self.guard = guard

    def _guard_failure(self, exc: BaseException) -> None:
        """Map a curl_cffi transport error onto the shared outage taxonomy."""
        if self.guard is None:
            return
        text = str(exc).lower()
        if "name or service not known" in text or "could not resolve" in text:
            self.guard.record_failure(
                "curl_cffi", NetworkFault("dns_failure", "dns_failure", False, str(exc))
            )
        elif "timed out" in text or "timeout" in text:
            self.guard.record_failure(
                "curl_cffi", NetworkFault("connect_timeout", "connect_timeout", True, str(exc))
            )
        elif "network is unreachable" in text:
            self.guard.record_failure(
                "curl_cffi",
                NetworkFault("network_unreachable", "network_unavailable", True, str(exc)),
            )

    def fetch(self, request: FetchRequest) -> FetchResult:
        try:
            from curl_cffi import requests as curl_requests
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
            raise TierUnavailable(
                "curl_cffi tier is unavailable. Install the crawl extra: `uv sync --extra crawl`."
            ) from exc

        from fmd.settings import get_settings

        if self.guard is not None:
            self.guard.check()
        method, headers, payload = _prepared(request)
        if not any(key.lower() == "user-agent" for key in headers):
            headers["User-Agent"] = get_settings().user_agent
        try:
            response = curl_requests.request(
                method,
                request.url,
                data=payload,
                headers=headers,
                impersonate=self.impersonate,
                timeout=self.timeout,
                allow_redirects=True,
            )
        except Exception as exc:
            self._guard_failure(exc)
            raise
        if self.guard is not None:
            self.guard.record_success()
        return FetchResult(
            url=request.url,
            final_url=str(response.url),
            status=response.status_code,
            headers=_lower_headers(response.headers),
            body=response.content,
            fetcher_tier=self.name,
            method=method,
        )


class BrowserNavigationError(RuntimeError):
    """A browser-tier failure with an already-classified fault."""

    def __init__(self, fault: NetworkFault, detail: str) -> None:
        self.fault = fault
        super().__init__(f"{fault.kind}: {detail}")


def _attach_guard(fetchers: Iterable[TierFetcher], guard: NetworkGuard) -> None:
    """Share one outage breaker across every tier (and the robots cache)."""
    for fetcher in fetchers:
        if getattr(fetcher, "guard", None) is None and hasattr(fetcher, "guard"):
            fetcher.guard = guard  # type: ignore[attr-defined]


def default_fetchers() -> list[TierFetcher]:
    """The tier order. The stealth tiers are lazy: importing and spawning both
    happen on first use, so the base install still works without them."""
    from fmd.fetch.firecrawl_local import FirecrawlLocalFetcher
    from fmd.fetch.mcp_browser import InvisiblePlaywrightMCPFetcher

    return [
        HttpxFetcher(),
        CurlCffiFetcher(),
        InvisiblePlaywrightMCPFetcher(),
        FirecrawlLocalFetcher(),
    ]


class FetchLadder:
    """Runs one URL through the tier ladder with robots + politeness applied."""

    def __init__(
        self,
        *,
        fetchers: Iterable[TierFetcher] | None = None,
        robots: RobotsCache | None = None,
        politeness: PolitenessManager | None = None,
        clock: SystemClock | None = None,
        sleeper: Callable[[float], None] | None = None,
        guard: NetworkGuard | None = None,
        stage: str | None = None,
    ) -> None:
        self.fetchers = list(fetchers) if fetchers is not None else default_fetchers()
        self.clock = clock or SystemClock()
        self.politeness = politeness or PolitenessManager(clock=self.clock)
        # One breaker for the whole ladder: three DNS/connect failures in a row
        # abort the stage instead of grinding through every URL and tier.
        self.guard = guard or NetworkGuard()
        _attach_guard(self.fetchers, self.guard)
        # robots.txt uses the same truthful UA and the same tier ladder, so a
        # 403 (UA/appearance block) escalates instead of being read as a
        # `Disallow: /` policy. Sharing the instances keeps one client pool.
        self.robots = robots or RobotsCache(clock=self.clock, fetchers=self.fetchers)
        self.sleeper = sleeper or time.sleep
        # Recorded on every scrape-failure log entry so `fmd failures` can
        # attribute losses to the pipeline stage that asked for the URL.
        self.stage = stage

    # ------------------------------------------------------------------- helpers
    def _elapsed_ms(self, start: float) -> float:
        return round((self.clock.monotonic() - start) * 1000.0, 3)

    def _refused(
        self,
        url: str,
        *,
        tier: str,
        error: str,
        robots=None,
        status: int = 0,
        quarantined: bool = False,
        start: float | None = None,
        method: str = "GET",
    ) -> FetchResult:
        return FetchResult(
            url=url,
            final_url=url,
            status=status,
            fetcher_tier=tier,
            elapsed_ms=0.0 if start is None else self._elapsed_ms(start),
            robots=robots,
            error=error,
            quarantined=quarantined,
            method=method,
        )

    # -------------------------------------------------------------------- public
    def close(self) -> None:
        """Release every tier's pooled resources (clients, spawned servers).

        Best-effort and idempotent: injected test fetchers without ``close``
        are skipped; a failing close is logged and never raised. Long-running
        callers (the orchestrator worker, the CLI verdict-live stream) call
        this on shutdown so the MCP browser server does not outlive the run.
        """
        for fetcher in self.fetchers:
            closer = getattr(fetcher, "close", None)
            if not callable(closer):
                continue
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 - shutdown must never raise
                logger.info("tier %s close failed: %s", getattr(fetcher, "name", "?"), exc)

    def fetch(
        self,
        url: str | FetchRequest,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        referer: str | None = None,
    ) -> FetchResult:
        """One URL through the ladder, with any failure appended to the log.

        This wrapper is the *single* logging exit (:mod:`fmd.fetch.failure_log`):
        every failure outcome of :meth:`_fetch_impl` — transport exceptions,
        ``robots_disallowed``, ``domain_quarantined``, ``unsupported_replay``,
        404/410/4xx/5xx, ``all_tiers_blocked`` — is recorded best-effort, and a
        stage-level exception (no result to inspect) is recorded too, then
        re-raised unchanged. Logging never breaks the fetch path.
        """
        request = (
            url
            if isinstance(url, FetchRequest)
            else FetchRequest(url=url, etag=etag, last_modified=last_modified, referer=referer)
        )
        start = self.clock.monotonic()
        try:
            result = self._fetch_impl(request)
        except Exception as exc:  # stage-level exit: no result to inspect
            self._record_failure_if_any(None, request.url, start, exception=exc)
            raise
        self._record_failure_if_any(result, request.url, start)
        return result

    def _record_failure_if_any(
        self,
        result: FetchResult | None,
        url: str,
        start: float,
        *,
        exception: Exception | None = None,
    ) -> None:
        """Append one scrape-failure record if this fetch was a failure.

        Best-effort by contract: any error here is swallowed so the fetch path
        can never break on logging.
        """
        try:
            if result is None:
                if exception is None:
                    return
                record_failure(
                    FetchFailure(
                        ts=utcnow(),
                        url=url,
                        status=0,
                        error=str(exception),
                        kind=type(exception).__name__,
                        tier="none",
                        method="GET",
                        elapsed_ms=self._elapsed_ms(start),
                        institution=institution_hint(url),
                        stage=self.stage,
                    )
                )
                return
            if not is_failure(result):
                return
            record_failure(
                FetchFailure(
                    ts=utcnow(),
                    url=url,
                    final_url=result.final_url,
                    status=result.status,
                    error=result.error,
                    kind=failure_kind_for(result),
                    tier=result.fetcher_tier,
                    robots_reason=(result.robots.reason if result.robots else None),
                    method=result.method or "GET",
                    elapsed_ms=result.elapsed_ms or self._elapsed_ms(start),
                    institution=institution_hint(url),
                    stage=self.stage,
                )
            )
        except Exception:  # noqa: BLE001 - logging must never break the fetch path
            return

    def _fetch_impl(self, request: FetchRequest) -> FetchResult:
        url = request.url
        method = _method(request)
        self.guard.check()

        decision = self.robots.decision(url)
        if not decision.allowed:
            logger.info("fetch refused by robots: %s (%s)", url, decision.reason)
            return self._refused(
                url,
                tier=TIER_ROBOTS,
                error="robots_disallowed",
                robots=decision,
                method=method,
            )
        if decision.crawl_delay is not None:
            self.politeness.set_crawl_delay(url, decision.crawl_delay)
        if self.politeness.is_quarantined(url):
            return self._refused(
                url,
                tier=TIER_QUARANTINE,
                error="domain_quarantined",
                robots=decision,
                quarantined=True,
                method=method,
            )

        delay = self.politeness.note_request(url)
        if delay > 0:
            self.sleeper(delay)

        start = self.clock.monotonic()
        last_result: FetchResult | None = None
        unavailable: list[str] = []
        unsupported: list[tuple[str, str]] = []
        for index, fetcher in enumerate(self.fetchers):
            try:
                result = fetcher.fetch(request)
            except TierUnavailable as exc:
                logger.info("tier %s unavailable for %s: %s", fetcher.name, url, exc)
                unavailable.append(f"{fetcher.name}: {exc}")
                continue
            except UnsupportedReplay as exc:
                # Never downgrade a POST to GET. Record the honest refusal and
                # keep the ladder moving (a later tier may be able to replay).
                logger.info("tier %s cannot replay %s for %s: %s", fetcher.name, method, url, exc)
                unsupported.append((fetcher.name, str(exc)))
                continue
            except NetworkUnavailable:
                # The link is gone. Do not turn it into one status-0 result per
                # URL: abort the stage so the caller can stop with a reason.
                raise
            except Exception as exc:  # transport failure: do not escalate
                fault = getattr(exc, "fault", None) or classify_exception(exc)
                logger.warning("tier %s failed for %s (%s): %s", fetcher.name, url, fault.kind, exc)
                # Back off *this host* instead of raising global concurrency.
                if fault.retryable:
                    self.politeness.note_timeout(url)
                result = FetchResult(
                    url=url,
                    final_url=url,
                    status=0,
                    fetcher_tier=fetcher.name,
                    error=f"{fault.kind}: {exc}",
                    method=method,
                )
                self.politeness.record_response(url, 0)
                result.elapsed_ms = self._elapsed_ms(start)
                result.robots = decision
                return result

            result.fetcher_tier = fetcher.name
            result.robots = decision
            result.method = method
            retry_after = parse_retry_after(result.headers.get("retry-after"), now=self.clock.now())
            self.politeness.record_response(url, result.status, retry_after=retry_after)

            if result.blocked and index < len(self.fetchers) - 1:
                logger.info(
                    "block signal %s from tier %s for %s; escalating",
                    result.status,
                    fetcher.name,
                    url,
                )
                last_result = result
                continue
            if result.blocked:
                last_result = result
                break

            # 304, 404/410, 4xx and success all stop here.
            result.elapsed_ms = self._elapsed_ms(start)
            return result

        # Every tier was blocked (or no tier could answer).
        if last_result is None:
            if unsupported and not unavailable:
                tier, detail = unsupported[-1]
                return self._refused(
                    url,
                    tier=tier,
                    error=f"unsupported_replay: {detail}",
                    robots=decision,
                    start=start,
                    method=method,
                )
            if unavailable:
                raise TierUnavailable(
                    f"no fetch tier could run for {url}. Missing tiers: " + "; ".join(unavailable)
                )
            return self._refused(
                url,
                tier=TIER_QUARANTINE,
                error="no_fetch_tiers",
                robots=decision,
                start=start,
                method=method,
            )
        self.politeness.record_response(url, 0)
        if not self.politeness.is_quarantined(url):
            # A Retry-After already set a shorter/longer quarantine; keep it.
            self.politeness.quarantine(url)
        base = last_result or result_from_unavailable(url, TIER_QUARANTINE, "no tier available")
        return FetchResult(
            url=url,
            final_url=base.final_url,
            status=base.status if base.status in BLOCK_STATUSES else 0,
            headers=base.headers,
            body=b"",
            fetcher_tier=TIER_QUARANTINE,
            elapsed_ms=self._elapsed_ms(start),
            robots=decision,
            error="all_tiers_blocked",
            quarantined=True,
            method=method,
        )


def result_from_unavailable(url: str, tier: str, message: str) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=url,
        status=0,
        fetcher_tier=tier,
        error=f"tier_unavailable: {message}",
    )


def fetch(url: str, *, ladder: FetchLadder | None = None) -> FetchResult:
    """Convenience wrapper; the ladder instance is reusable/pooled."""
    return (ladder or FetchLadder()).fetch(url)


def utcnow() -> datetime:
    return datetime.now(UTC)
