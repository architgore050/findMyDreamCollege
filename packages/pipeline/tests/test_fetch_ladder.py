from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fmd.fetch.dedupe import hamming, near_duplicate, sha256_hex, simhash64
from fmd.fetch.ladder import FetchLadder
from fmd.fetch.politeness import (
    PolitenessManager,
    SystemClock,
    parse_retry_after,
    registrable_domain,
)
from fmd.fetch.robots import RobotsCache
from fmd.fetch.types import FetchRequest, FetchResult
from fmd.frontier.state import InMemoryDomainStateStore


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)
        self._mono += seconds


class FakeTier:
    def __init__(self, name: str, statuses: list[int], retry_after: str = "0") -> None:
        self.name = name
        self.statuses = list(statuses)
        self.retry_after = retry_after
        self.calls = 0
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResult:
        self.calls += 1
        self.requests.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        headers = {"content-type": "text/html"}
        if status == 429:
            headers["retry-after"] = self.retry_after
        return FetchResult(
            url=request.url,
            final_url=request.url,
            status=status,
            headers=headers,
            body=b"ok" if status == 200 else b"",
            fetcher_tier=self.name,
        )


class UnavailableTier:
    name = "curl_cffi"

    def fetch(self, request: FetchRequest) -> FetchResult:
        from fmd.fetch.types import TierUnavailable

        raise TierUnavailable("not installed")


def permissive_robots(clock: FakeClock) -> RobotsCache:
    return RobotsCache(fetch=lambda url: (404, ""), clock=clock)


def make_ladder(tiers, clock: FakeClock, *, politeness=None) -> FetchLadder:
    delays: list[float] = []

    def sleeper(seconds: float) -> None:
        delays.append(seconds)
        clock.advance(seconds)

    ladder = FetchLadder(
        fetchers=tiers,
        robots=permissive_robots(clock),
        politeness=politeness
        or PolitenessManager(clock=clock, jitter=0.0, store=InMemoryDomainStateStore()),
        clock=clock,
        sleeper=sleeper,
    )
    ladder.slept = delays  # type: ignore[attr-defined]
    return ladder


def test_403_escalates_to_next_tier() -> None:
    clock = FakeClock()
    httpx_tier = FakeTier("httpx", [403])
    curl_tier = FakeTier("curl_cffi", [200])
    ladder = make_ladder([httpx_tier, curl_tier], clock)

    result = ladder.fetch("https://college.ac.in/placements")

    assert result.status == 200
    assert result.fetcher_tier == "curl_cffi"
    assert httpx_tier.calls == 1
    assert curl_tier.calls == 1


def test_404_does_not_escalate() -> None:
    clock = FakeClock()
    httpx_tier = FakeTier("httpx", [404])
    curl_tier = FakeTier("curl_cffi", [200])
    ladder = make_ladder([httpx_tier, curl_tier], clock)

    result = ladder.fetch("https://college.ac.in/missing")

    assert result.status == 404
    assert result.fetcher_tier == "httpx"
    assert result.terminal
    assert curl_tier.calls == 0


def test_cf_mitigated_header_escalates() -> None:
    clock = FakeClock()
    httpx_tier = FakeTier("httpx", [200])
    httpx_tier.headers_override = {"cf-mitigated": "challenge"}  # type: ignore[attr-defined]
    # Patch the response directly to carry the header.
    original = httpx_tier.fetch

    def fetch_with_header(request):
        result = original(request)
        result.headers["cf-mitigated"] = "challenge"
        return result

    httpx_tier.fetch = fetch_with_header  # type: ignore[assignment]
    curl_tier = FakeTier("curl_cffi", [200])
    ladder = make_ladder([httpx_tier, curl_tier], clock)

    result = ladder.fetch("https://college.ac.in/cf")

    assert result.fetcher_tier == "curl_cffi"


def test_unavailable_tier_is_skipped() -> None:
    clock = FakeClock()
    httpx_tier = FakeTier("httpx", [403])
    patchright = FakeTier("patchright", [200])
    ladder = make_ladder([httpx_tier, UnavailableTier(), patchright], clock)

    result = ladder.fetch("https://college.ac.in/page")

    assert result.status == 200
    assert result.fetcher_tier == "patchright"


def test_all_tiers_unavailable_raises_clear_error() -> None:
    import pytest

    from fmd.fetch.types import TierUnavailable

    clock = FakeClock()
    ladder = make_ladder([UnavailableTier(), UnavailableTier()], clock)

    with pytest.raises(TierUnavailable, match="crawl extra|not installed"):
        ladder.fetch("https://college.ac.in/page")


def test_all_tiers_blocked_quarantines_domain() -> None:
    clock = FakeClock()
    tiers = [FakeTier("httpx", [403]), FakeTier("curl_cffi", [403])]
    ladder = make_ladder(tiers, clock)

    result = ladder.fetch("https://college.ac.in/blocked")

    assert result.fetcher_tier == "quarantine"
    assert result.quarantined
    assert ladder.politeness.is_quarantined("https://college.ac.in/blocked")


def test_robots_disallow_never_calls_a_tier() -> None:
    clock = FakeClock()
    robots = RobotsCache(
        fetch=lambda url: (200, "User-agent: *\nDisallow: /private\n"), clock=clock
    )
    tier = FakeTier("httpx", [200])
    ladder = FetchLadder(
        fetchers=[tier], robots=robots, clock=clock, sleeper=lambda s: clock.advance(s)
    )

    result = ladder.fetch("https://college.ac.in/private/report.pdf")

    assert result.fetcher_tier == "robots"
    assert result.error == "robots_disallowed"
    assert result.robots is not None and result.robots.allowed is False
    assert tier.calls == 0


def test_robots_allows_public_path_and_caches() -> None:
    clock = FakeClock()
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, str]:
        calls.append(url)
        return 200, "User-agent: *\nAllow: /\nCrawl-delay: 2\n"

    robots = RobotsCache(fetch=fetch, clock=clock)
    for _ in range(3):
        decision = robots.decision("https://college.ac.in/public")
        assert decision.allowed
        assert decision.crawl_delay == 2

    # One robots.txt fetch for three decisions (24h TTL).
    assert calls == ["https://college.ac.in/robots.txt"]


def test_robots_cache_expires_after_ttl() -> None:
    clock = FakeClock()
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, str]:
        calls.append(url)
        return 200, "User-agent: *\nAllow: /\n"

    robots = RobotsCache(fetch=fetch, clock=clock)
    robots.decision("https://college.ac.in/a")
    clock.advance(60 * 60 * 25)
    robots.decision("https://college.ac.in/b")

    assert len(calls) == 2


def test_missing_robots_is_allowed_conservatively_and_cached() -> None:
    clock = FakeClock()
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, str]:
        calls.append(url)
        return 404, ""

    robots = RobotsCache(fetch=fetch, clock=clock)
    first = robots.decision("https://college.ac.in/a")
    second = robots.decision("https://college.ac.in/b")

    assert first.allowed and first.reason == "unavailable_allow"
    assert first.crawl_delay == 1.0
    assert second.allowed
    assert len(calls) == 1


def test_ladder_keeps_retry_after_instead_of_default_quarantine() -> None:
    clock = FakeClock()
    politeness = PolitenessManager(clock=clock, jitter=0.0)
    tiers = [
        FakeTier("httpx", [429], retry_after="120"),
        FakeTier("curl_cffi", [429], retry_after="120"),
    ]
    ladder = make_ladder(tiers, clock, politeness=politeness)

    result = ladder.fetch("https://college.ac.in/limited")

    assert result.fetcher_tier == "quarantine"
    assert politeness.delay_for("https://college.ac.in/other") == 120.0


def test_crawl_delay_is_honoured() -> None:
    clock = FakeClock()
    store = InMemoryDomainStateStore()
    politeness = PolitenessManager(clock=clock, jitter=0.0, store=store)
    politeness.set_crawl_delay("college.ac.in", 5.0)

    first = politeness.delay_for("https://college.ac.in/a")
    assert first == 0.0
    politeness.note_request("https://college.ac.in/a", at=clock.now())
    clock.advance(1.0)
    second = politeness.delay_for("https://college.ac.in/b")
    assert second == 4.0


def test_retry_after_is_honoured() -> None:
    clock = FakeClock()
    politeness = PolitenessManager(clock=clock, jitter=0.0)
    politeness.record_response("https://college.ac.in/x", 429, retry_after=120)

    assert politeness.is_quarantined("https://college.ac.in/y")
    assert politeness.delay_for("https://college.ac.in/y") == 120

    assert parse_retry_after("90") == 90
    assert parse_retry_after(None) is None


def test_five_consecutive_5xx_quarantine_for_six_hours() -> None:
    clock = FakeClock()
    store = InMemoryDomainStateStore()
    politeness = PolitenessManager(clock=clock, jitter=0.0, store=store)
    url = "https://college.ac.in/page"

    for _ in range(4):
        assert not politeness.record_response(url, 503)
    assert not politeness.is_quarantined(url)
    assert politeness.record_response(url, 503)
    assert politeness.is_quarantined(url)

    state = store.load("college.ac.in")
    assert state.quarantined_until == clock.now() + timedelta(hours=6)

    clock.advance(6 * 60 * 60 + 1)
    assert not politeness.is_quarantined(url)


def test_success_resets_5xx_streak() -> None:
    clock = FakeClock()
    politeness = PolitenessManager(clock=clock, jitter=0.0)
    url = "https://college.ac.in/page"
    for _ in range(4):
        politeness.record_response(url, 503)
    politeness.record_response(url, 200)
    for _ in range(4):
        politeness.record_response(url, 503)
    assert not politeness.is_quarantined(url)


def test_ac_in_and_co_in_are_distinct_domains() -> None:
    ac = registrable_domain("https://www.college.ac.in/report.pdf")
    co = registrable_domain("https://www.college.co.in/report.pdf")
    gov = registrable_domain("https://portal.gov.in/x")

    assert ac == "college.ac.in"
    assert co == "college.co.in"
    assert gov == "portal.gov.in"
    assert ac != co


def test_304_is_conditional_and_costs_no_budget() -> None:
    clock = FakeClock()
    tier = FakeTier("httpx", [304])
    ladder = make_ladder([tier], clock)

    result = ladder.fetch(
        FetchRequest(
            url="https://college.ac.in/page",
            etag='"abc"',
            last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
        )
    )

    assert result.status == 304
    assert result.not_modified
    assert result.body == b""
    assert result.costs_budget is False
    assert tier.calls == 1
    sent = tier.requests[0]
    assert sent.etag == '"abc"'
    assert sent.last_modified == "Wed, 01 Jan 2026 00:00:00 GMT"


def test_revalidate_builds_conditionals_from_stored_document_fields() -> None:
    clock = FakeClock()
    tier = FakeTier("httpx", [304])
    ladder = make_ladder([tier], clock)

    # A documents row stores last_modified as a timestamptz; the ladder needs a
    # header value. `revalidate` is that seam, so WS-C never re-implements it.
    request = FetchRequest.revalidate(
        "https://college.ac.in/page",
        etag='"v1"',
        last_modified=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
    )
    result = ladder.fetch(request)

    assert request.last_modified == "Thu, 01 Jan 2026 08:00:00 GMT"
    assert tier.requests[0].etag == '"v1"'
    assert tier.requests[0].last_modified == "Thu, 01 Jan 2026 08:00:00 GMT"
    assert result.not_modified
    assert result.body == b""
    assert result.costs_budget is False


def test_httpx_tier_sends_conditionals_via_mock_transport() -> None:
    import httpx

    from fmd.fetch.ladder import HttpxFetcher

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["if-none-match"] = request.headers.get("if-none-match", "")
        seen["if-modified-since"] = request.headers.get("if-modified-since", "")
        return httpx.Response(304, headers={"etag": '"abc"'})

    clock = FakeClock()
    tier = HttpxFetcher(transport=httpx.MockTransport(handler))
    ladder = make_ladder([tier], clock)

    result = ladder.fetch(
        FetchRequest(
            url="https://college.ac.in/page",
            etag='"abc"',
            last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
        )
    )

    assert result.status == 304
    assert seen == {
        "if-none-match": '"abc"',
        "if-modified-since": "Wed, 01 Jan 2026 00:00:00 GMT",
    }
    tier.close()


def test_quarantined_domain_is_not_requested() -> None:
    clock = FakeClock()
    politeness = PolitenessManager(clock=clock, jitter=0.0)
    politeness.quarantine("college.ac.in", hours=6)
    tier = FakeTier("httpx", [200])
    ladder = make_ladder([tier], clock, politeness=politeness)

    result = ladder.fetch("https://college.ac.in/page")
    assert result.quarantined
    assert result.status == 0
    assert tier.calls == 0


def test_dedupe_hashes_and_simhash() -> None:
    assert sha256_hex(b"a\r\nb  \r\n") == sha256_hex(b"a\nb\n")
    one = simhash64("BSc Medical Laboratory Technology placement report 2025")
    two = simhash64("BSc Medical Laboratory Technology placement report 2025")
    other = simhash64("fee structure for diploma courses in another state")
    assert one == two
    assert hamming(one, other) > 3
    assert near_duplicate(one, two)
    assert not near_duplicate(one, other)


def test_default_system_clock_is_usable() -> None:
    assert isinstance(SystemClock().now(), datetime)
