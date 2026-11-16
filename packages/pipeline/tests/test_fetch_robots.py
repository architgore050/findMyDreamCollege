"""Offline tests for the robots.txt policy (RFC 9309 + ladder escalation).

Every test runs through ``httpx.MockTransport``; nothing touches the network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from fmd.fetch.ladder import HttpxFetcher
from fmd.fetch.robots import RobotsCache
from fmd.frontier.state import InMemoryDomainStateStore

UA = "findMyDreamCollege/0.1 (+https://example.org/contact; research crawler)"


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


def tier(status: int, text: str = "", *, calls: list[str] | None = None) -> HttpxFetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(status, text=text, headers={"content-type": "text/plain"})

    return HttpxFetcher(transport=httpx.MockTransport(handler))


def robots(fetchers, clock: FakeClock, *, store=None) -> RobotsCache:
    return RobotsCache(
        fetchers=fetchers,
        user_agent=UA,
        clock=clock,
        store=store or InMemoryDomainStateStore(),
        conservative_delay=2.0,
    )


def test_200_policy_allows() -> None:
    clock = FakeClock()
    cache = robots([tier(200, "User-agent: *\nAllow: /\n")], clock)

    decision = cache.decision("https://college.ac.in/report.pdf")

    assert decision.allowed
    assert decision.reason == "allowed"
    assert decision.status == 200


def test_200_policy_disallows_path() -> None:
    clock = FakeClock()
    cache = robots([tier(200, "User-agent: *\nDisallow: /private\n")], clock)

    decision = cache.decision("https://college.ac.in/private/report.pdf")

    assert not decision.allowed
    assert decision.reason == "disallowed_by_robots"
    assert cache.summary()["blocked_by_robots"] == 1


def test_404_is_unavailable_allow_with_conservative_delay_and_is_cached() -> None:
    clock = FakeClock()
    calls: list[str] = []
    store = InMemoryDomainStateStore()
    cache = robots([tier(404, calls=calls)], clock, store=store)

    first = cache.decision("https://college.ac.in/a")
    second = cache.decision("https://college.ac.in/b")

    assert first.allowed and first.reason == "unavailable_allow"
    assert first.crawl_delay == 2.0
    assert second.allowed
    assert len(calls) == 1, "the unavailable decision is cached with a TTL"
    persisted = store.load("college.ac.in").robots_json
    assert persisted is not None
    assert persisted["case"] == "unavailable_allow"
    assert persisted["reason"] == "unavailable_allow"
    assert persisted["transient"] is False


def test_403_escalates_to_the_next_tier() -> None:
    clock = FakeClock()
    blocked_calls: list[str] = []
    allowed_calls: list[str] = []
    cache = robots(
        [
            tier(403, calls=blocked_calls),
            tier(200, "User-agent: *\nAllow: /\n", calls=allowed_calls),
        ],
        clock,
    )

    decision = cache.decision("https://www.nirfindia.org/Rankings/2025/MedicalRanking.html")

    assert decision.allowed
    assert decision.reason == "allowed"
    assert blocked_calls == ["https://www.nirfindia.org/robots.txt"]
    assert allowed_calls == ["https://www.nirfindia.org/robots.txt"]


def test_403_after_escalation_abstains_with_reason_and_retries_next_run() -> None:
    clock = FakeClock()
    store = InMemoryDomainStateStore()
    first_run = robots([tier(403), tier(403)], clock, store=store)

    decision = first_run.decision("https://www.nirfindia.org/")

    assert not decision.allowed
    assert decision.reason == "unreachable_disallow"
    assert decision.status == 403
    assert first_run.case_for("https://www.nirfindia.org/") == "unreachable_disallow"
    assert first_run.summary()["blocked_by_robots"] == 0
    assert first_run.summary()["unreachable_disallow"] == 1

    persisted = store.load("nirfindia.org").robots_json
    assert persisted is not None
    assert persisted["case"] == "unreachable_disallow"
    assert persisted["reason"] == "unreachable_disallow"
    assert persisted["transient"] is True, "a block must not be cached as permanent"

    # A new process (fresh cache, same store) tries again; this time the site
    # answers. The transient failure must not short-circuit the retry.
    second_run = robots([tier(200, "User-agent: *\nAllow: /\n")], clock, store=store)
    retried = second_run.decision("https://www.nirfindia.org/")

    assert retried.allowed
    assert retried.reason == "allowed"


def test_5xx_abstains_and_is_retried_next_run() -> None:
    clock = FakeClock()
    store = InMemoryDomainStateStore()
    first_run = robots([tier(503)], clock, store=store)

    decision = first_run.decision("https://college.ac.in/a")

    assert not decision.allowed
    assert decision.reason == "unreachable_disallow"
    assert decision.status == 503
    assert first_run.summary()["unreachable_disallow"] == 1

    # Not usable across runs even though it is stored for reporting.
    second_run = robots([tier(200, "User-agent: *\nAllow: /\n")], clock, store=store)
    assert second_run.decision("https://college.ac.in/a").allowed


def test_429_after_escalation_never_becomes_allow() -> None:
    clock = FakeClock()
    cache = robots([tier(429)], clock)

    decision = cache.decision("https://college.ac.in/a")

    assert not decision.allowed
    assert decision.reason == "unreachable_disallow"
    assert decision.status == 429


def test_network_failure_abstains_instead_of_allowing() -> None:
    clock = FakeClock()

    def boom(_url: str) -> tuple[int, str]:
        raise httpx.ConnectError("no route")

    cache = RobotsCache(fetch=boom, user_agent=UA, clock=clock)

    decision = cache.decision("https://college.ac.in/a")

    assert not decision.allowed
    assert decision.reason == "unreachable_disallow"
    assert decision.status is None
    assert cache.case_for("https://college.ac.in/a") == "unreachable_disallow"


def test_transient_decision_is_reused_in_process_within_short_ttl() -> None:
    clock = FakeClock()
    calls: list[str] = []
    cache = robots([tier(403, calls=calls)], clock)

    assert not cache.decision("https://college.ac.in/a").allowed
    assert not cache.decision("https://college.ac.in/b").allowed
    assert len(calls) == 1, "one escalation per domain per short TTL, not per URL"

    clock.advance(60 * 11)
    cache.decision("https://college.ac.in/c")
    assert len(calls) == 2


def test_summary_separates_robots_blocks_from_failures() -> None:
    clock = FakeClock()
    store = InMemoryDomainStateStore()
    cache = robots([tier(200, "User-agent: *\nDisallow: /\n")], clock, store=store)
    cache.decision("https://blocked.ac.in/a")

    failing = robots([tier(503)], clock, store=store)
    failing.decision("https://broken.ac.in/a")

    assert cache.summary()["blocked_by_robots"] == 1
    assert failing.summary()["unreachable_disallow"] == 1
    assert cache.summary()["unreachable_disallow"] == 0
    assert cache.blocked_domains() == ("blocked.ac.in",)
    assert failing.unreachable_domains() == ("broken.ac.in",)
