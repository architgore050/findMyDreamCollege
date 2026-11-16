"""The fetch ladder: httpx -> curl_cffi -> patchright -> quarantine (brief §7).

Escalation happens **only** on a block signal (403/429/503 or a
``cf-mitigated`` response header). A 404/410 is terminal and is returned from
the first tier. Every result carries the tier, elapsed time, final URL and the
robots decision so the snapshot layer can record how a page was obtained.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

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
)

logger = logging.getLogger(__name__)

TIER_HTTPX = "httpx"
TIER_CURL_CFFI = "curl_cffi"
TIER_PATCHRIGHT = "patchright"
TIER_QUARANTINE = "quarantine"
TIER_ROBOTS = "robots"


def _lower_headers(headers) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in dict(headers).items()}


class HttpxFetcher:
    """Primary tier. Pooled, HTTP/2 when the server offers it."""

    name = TIER_HTTPX

    def __init__(self, client=None, *, timeout: float = 20.0, transport=None) -> None:
        self._client = client
        self._owns_client = client is None
        self.timeout = timeout
        self.transport = transport

    def _get_client(self):
        if self._client is None:
            import httpx

            from fmd.settings import get_settings

            common = {
                "follow_redirects": True,
                "timeout": self.timeout,
                "headers": {"User-Agent": get_settings().user_agent},
                "transport": self.transport,
            }
            try:
                self._client = httpx.Client(http2=True, **common)
            except ImportError:
                # The base install does not ship the optional `h2` package.
                logger.info("httpx http2 unavailable; falling back to HTTP/1.1")
                self._client = httpx.Client(http2=False, **common)
        return self._client

    def fetch(self, request: FetchRequest) -> FetchResult:
        headers: dict[str, str] = {}
        if request.etag:
            headers["If-None-Match"] = request.etag
        if request.last_modified:
            headers["If-Modified-Since"] = request.last_modified
        if request.referer:
            headers["Referer"] = request.referer
        response = self._get_client().get(request.url, headers=headers)
        return FetchResult(
            url=request.url,
            final_url=str(response.url),
            status=response.status_code,
            headers=_lower_headers(response.headers),
            body=response.content,
            fetcher_tier=self.name,
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None


class CurlCffiFetcher:
    """Second tier: browser TLS fingerprint impersonation."""

    name = TIER_CURL_CFFI

    def __init__(self, *, impersonate: str = "chrome", timeout: float = 25.0) -> None:
        self.impersonate = impersonate
        self.timeout = timeout

    def fetch(self, request: FetchRequest) -> FetchResult:
        try:
            from curl_cffi import requests as curl_requests
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
            raise TierUnavailable(
                "curl_cffi tier is unavailable. Install the crawl extra: `uv sync --extra crawl`."
            ) from exc

        from fmd.settings import get_settings

        headers = {"User-Agent": get_settings().user_agent}
        if request.etag:
            headers["If-None-Match"] = request.etag
        if request.last_modified:
            headers["If-Modified-Since"] = request.last_modified
        if request.referer:
            headers["Referer"] = request.referer
        response = curl_requests.get(
            request.url,
            headers=headers,
            impersonate=self.impersonate,
            timeout=self.timeout,
            allow_redirects=True,
        )
        return FetchResult(
            url=request.url,
            final_url=str(response.url),
            status=response.status_code,
            headers=_lower_headers(response.headers),
            body=response.content,
            fetcher_tier=self.name,
        )


class PatchrightFetcher:
    """Third tier: a real Chromium under Xvfb. Headful, never headless."""

    name = TIER_PATCHRIGHT

    def __init__(self, *, headless: bool = False, timeout_ms: int = 30_000) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms

    def fetch(self, request: FetchRequest) -> FetchResult:
        try:
            from patchright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
            raise TierUnavailable(
                "patchright tier is unavailable. Install the crawl extra and a "
                "Chromium build: `uv sync --extra crawl && patchright install chromium`."
            ) from exc

        from fmd.settings import get_settings

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=self.headless, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            try:
                context = browser.new_context(user_agent=get_settings().user_agent)
                page = context.new_page()
                response = page.goto(
                    request.url, wait_until="domcontentloaded", timeout=self.timeout_ms
                )
                status = response.status if response is not None else 0
                headers = _lower_headers(response.headers) if response is not None else {}
                body = page.content().encode("utf-8")
                final_url = page.url
            finally:
                browser.close()
        return FetchResult(
            url=request.url,
            final_url=final_url,
            status=status,
            headers=headers,
            body=body,
            fetcher_tier=self.name,
        )


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
    ) -> None:
        self.fetchers = list(fetchers) if fetchers is not None else default_fetchers()
        self.clock = clock or SystemClock()
        self.politeness = politeness or PolitenessManager(clock=self.clock)
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

        decision = self.robots.decision(url)
        if not decision.allowed:
            logger.info("fetch refused by robots: %s (%s)", url, decision.reason)
            return self._refused(
                url,
                tier=TIER_ROBOTS,
                error="robots_disallowed",
                robots=decision,
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
            )

        delay = self.politeness.note_request(url)
        if delay > 0:
            self.sleeper(delay)

        start = self.clock.monotonic()
        last_result: FetchResult | None = None
        unavailable: list[str] = []
        for index, fetcher in enumerate(self.fetchers):
            try:
                result = fetcher.fetch(request)
            except TierUnavailable as exc:
                logger.info("tier %s unavailable for %s: %s", fetcher.name, url, exc)
                unavailable.append(f"{fetcher.name}: {exc}")
                continue
            except Exception as exc:  # transport failure: do not escalate
                logger.warning("tier %s failed for %s: %s", fetcher.name, url, exc)
                result = FetchResult(
                    url=url,
                    final_url=url,
                    status=0,
                    fetcher_tier=fetcher.name,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.politeness.record_response(url, 0)
                result.elapsed_ms = self._elapsed_ms(start)
                result.robots = decision
                return result

            result.fetcher_tier = fetcher.name
            result.robots = decision
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
