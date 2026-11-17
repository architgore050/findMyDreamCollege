"""The fetch ladder: httpx -> curl_cffi -> patchright -> quarantine (brief §7).

Escalation happens **only** on a block signal (403/429/503 or a
``cf-mitigated`` response header). A 404/410 is terminal and is returned from
the first tier. Every result carries the tier, elapsed time, final URL and the
robots decision so the snapshot layer can record how a page was obtained.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import urlencode

from fmd.fetch.politeness import (
    PolitenessManager,
    SystemClock,
    parse_retry_after,
    registrable_domain,
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
    BrowserPolicy,
    NetworkFault,
    NetworkGuard,
    NetworkUnavailable,
    RetryPolicy,
    Timeouts,
    classify_browser_error,
    classify_exception,
    make_client,
    request_with_retries,
)

logger = logging.getLogger(__name__)

TIER_HTTPX = "httpx"
TIER_CURL_CFFI = "curl_cffi"
TIER_PATCHRIGHT = "patchright"
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


#: One in-flight browser navigation per registrable domain. Browser fetches are
#: hundreds of KB each; running two on a hostel link starves both.
_BROWSER_LOCKS: dict[str, threading.Lock] = {}
_BROWSER_LOCKS_GUARD = threading.Lock()


def _domain_lock(domain: str) -> threading.Lock:
    with _BROWSER_LOCKS_GUARD:
        lock = _BROWSER_LOCKS.get(domain)
        if lock is None:
            lock = threading.Lock()
            _BROWSER_LOCKS[domain] = lock
        return lock


def _browser_child_pids() -> set[int]:
    """Linux PIDs whose parent is this process and whose comm is a browser.

    Playwright does not expose the Chromium PID portably. Reading ``/proc`` is
    the honest fallback used only when a close deadline is exceeded: we kill
    *new* children rather than every Chromium on the host.
    """
    if sys.platform != "linux":  # pragma: no cover - the host is Linux
        return set()
    parent = os.getpid()
    found: set[int] = set()
    try:
        entries = os.listdir("/proc")
    except OSError:  # pragma: no cover - /proc is expected
        return set()
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as handle:
                stat = handle.read()
            with open(f"/proc/{entry}/comm", encoding="utf-8", errors="replace") as handle:
                comm = handle.read().strip().lower()
        except OSError:
            continue
        # `/proc/<pid>/stat`: fields after `comm` differ, so split on the last
        # closing paren and take the first field (state), then ppid.
        try:
            tail = stat.rsplit(")", 1)[1].split()
            ppid = int(tail[1])
        except (IndexError, ValueError):
            continue
        if ppid == parent and ("chrom" in comm or "headless" in comm):
            found.add(int(entry))
    return found


def _kill_new_browser_children(before: set[int], *, reason: str) -> list[int]:
    """SIGKILL Chromium children spawned during a timed-out navigation."""
    killed: list[int] = []
    for pid in sorted(_browser_child_pids() - before):
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError:  # pragma: no cover - already gone
            continue
    if killed:
        logger.warning("browser zombies killed after %s: pids %s", reason, killed)
    return killed


def _close_with_deadline(obj: object, deadline_s: float) -> bool:
    """Close a Playwright object in a daemon thread; False means it hung."""

    def _close() -> None:
        with contextlib.suppress(Exception):
            obj.close()  # type: ignore[attr-defined]

    thread = threading.Thread(target=_close, daemon=True)
    thread.start()
    thread.join(max(0.1, deadline_s))
    return not thread.is_alive()


class PatchrightFetcher:
    """Third tier: a real Chromium under Xvfb. Headful, never headless.

    Bounded by :class:`~fmd.net.policy.BrowserPolicy`: navigation, action,
    selector and network-idle timeouts; optional asset blocking; a hard page
    size cap; one navigation per domain; and a hard kill of Chromium children
    when a close exceeds its deadline. Browser failures are mapped onto the
    same taxonomy as HTTP failures (:func:`classify_browser_error`).
    """

    name = TIER_PATCHRIGHT

    def __init__(
        self,
        *,
        headless: bool = False,
        timeout_ms: int | None = None,
        policy: BrowserPolicy | None = None,
        guard: NetworkGuard | None = None,
    ) -> None:
        self.headless = headless
        self.policy = policy or BrowserPolicy.from_env()
        self.timeout_ms = int(timeout_ms or self.policy.navigation_timeout_ms)
        self.guard = guard

    def _route(self, route) -> None:
        """Block non-essential loads so one dead asset cannot stall the page."""
        try:
            if self.policy.block_assets and route.request.resource_type in {
                "image",
                "media",
                "font",
            }:
                route.abort()
                return
            route.continue_()
        except Exception:  # pragma: no cover - route can race a close
            pass

    def fetch(self, request: FetchRequest) -> FetchResult:
        method = _method(request)
        if method != "GET":
            # `page.goto` only issues GET. Faking a POST by submitting a form
            # would execute the page's own JavaScript and render whatever it
            # decides, not replay the persisted request/body. Downgrading would
            # also be invisible to the caller (HTTP 200, wrong page), so refuse
            # explicitly and let the ladder record `unsupported_replay`.
            raise UnsupportedReplay(
                f"patchright cannot faithfully replay {method} requests; a browser "
                "navigation only issues GET. Refusing rather than downgrading."
            )
        try:
            from patchright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
            raise TierUnavailable(
                "patchright tier is unavailable. Install the crawl extra and a "
                "Chromium build: `uv sync --extra crawl && patchright install chromium`."
            ) from exc

        from fmd.settings import get_settings

        if self.guard is not None:
            self.guard.check()
        headers = _merged_headers(request)
        domain = registrable_domain(request.url) or request.url
        lock = _domain_lock(domain)
        with lock:
            return self._navigate(
                sync_playwright=sync_playwright,
                request=request,
                headers=headers,
                user_agent=get_settings().user_agent,
                method=method,
            )

    def _navigate(
        self,
        *,
        sync_playwright,
        request: FetchRequest,
        headers: dict[str, str],
        user_agent: str,
        method: str,
    ) -> FetchResult:
        policy = self.policy
        before = _browser_child_pids()
        started = time.monotonic()
        browser = None
        context = None
        try:
            with sync_playwright() as pw:
                launch_options: dict[str, object] = {
                    "headless": self.headless,
                    "timeout": policy.launch_timeout_ms,
                    "args": ["--no-sandbox", "--disable-dev-shm-usage"],
                }
                if policy.profile_dir:
                    # Persistent profile: an isolated copy per task is safer
                    # than sharing one across concurrent fetches.
                    context = pw.chromium.launch_persistent_context(
                        policy.profile_dir, **launch_options
                    )
                    browser = None
                else:
                    browser = pw.chromium.launch(**launch_options)
                    context = browser.new_context(
                        user_agent=user_agent,
                        extra_http_headers=headers or None,
                    )
                context.set_default_timeout(policy.action_timeout_ms)
                context.set_default_navigation_timeout(self.timeout_ms)
                if policy.block_assets:
                    context.route("**/*", self._route)
                page = context.new_page()
                response = page.goto(
                    request.url, wait_until="domcontentloaded", timeout=self.timeout_ms
                )
                if policy.wait_network_idle:
                    try:
                        page.wait_for_load_state(
                            "networkidle", timeout=policy.network_idle_timeout_ms
                        )
                    except Exception as exc:
                        logger.info(
                            "network idle not reached for %s (%s); using domcontentloaded",
                            request.url,
                            exc,
                        )
                if response is None:
                    raise BrowserNavigationError(
                        NetworkFault(
                            "browser_no_response",
                            "browser_no_response",
                            False,
                            "browser: navigation produced no response",
                        ),
                        "navigation produced no response",
                    )
                status = response.status
                response_headers = _lower_headers(response.headers)
                body = page.content().encode("utf-8")
                if len(body) > policy.max_page_bytes:
                    raise BrowserNavigationError(
                        NetworkFault(
                            "page_too_large",
                            "page_too_large",
                            False,
                            f"browser: page exceeded {policy.max_page_bytes} bytes",
                        ),
                        f"page is {len(body)} bytes (cap {policy.max_page_bytes})",
                    )
                final_url = page.url
        except BrowserNavigationError as exc:
            self._record_failure(request.url, exc.fault)
            raise
        except Exception as exc:
            fault = classify_browser_error(str(exc))
            self._record_failure(request.url, fault)
            raise BrowserNavigationError(fault, str(exc)) from exc
        finally:
            # Never leave a Chromium tree behind: close with a deadline, then
            # SIGKILL any *new* child this navigation spawned.
            deadline_s = policy.hard_kill_grace_ms / 1000.0
            closed = True
            for obj in (context, browser):
                if obj is not None:
                    closed = _close_with_deadline(obj, deadline_s) and closed
            if (
                not closed
                or time.monotonic() - started
                > (self.timeout_ms + policy.hard_kill_grace_ms) / 1000.0
            ):
                _kill_new_browser_children(before, reason="navigation timeout")
        if self.guard is not None:
            self.guard.record_success()
        return FetchResult(
            url=request.url,
            final_url=final_url,
            status=status,
            headers=response_headers,
            body=body,
            fetcher_tier=self.name,
            method=method,
        )

    def _record_failure(self, url: str, fault: NetworkFault) -> None:
        if self.guard is None:
            return
        self.guard.record_failure(url, fault)
        # Raises NetworkUnavailable when the breaker trips; that propagates out
        # of the ladder so the stage aborts instead of opening more browsers.
        self.guard.check()


def _attach_guard(fetchers: Iterable[TierFetcher], guard: NetworkGuard) -> None:
    """Share one outage breaker across every tier (and the robots cache)."""
    for fetcher in fetchers:
        if getattr(fetcher, "guard", None) is None and hasattr(fetcher, "guard"):
            fetcher.guard = guard  # type: ignore[attr-defined]


def default_fetchers() -> list[TierFetcher]:
    return [HttpxFetcher(), CurlCffiFetcher(), PatchrightFetcher()]


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
    def fetch(
        self,
        url: str | FetchRequest,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        referer: str | None = None,
    ) -> FetchResult:
        request = (
            url
            if isinstance(url, FetchRequest)
            else FetchRequest(url=url, etag=etag, last_modified=last_modified, referer=referer)
        )
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
