from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fmd.frontier.queue import (
    Frontier,
    InMemoryCrawlStats,
    InMemoryFrontierRepo,
)
from fmd.frontier.state import InMemoryDomainStateStore


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


def make_frontier(**kwargs) -> Frontier:
    clock = FakeClock()
    return Frontier(
        InMemoryFrontierRepo(),
        stats=kwargs.pop("stats", InMemoryCrawlStats()),
        domain_state=InMemoryDomainStateStore(),
        clock=clock,
        **kwargs,
    )


def test_priority_order_and_domain_keying() -> None:
    frontier = make_frontier()
    frontier.enqueue("https://college.ac.in/low", priority=1)
    frontier.enqueue("https://college.ac.in/high", priority=9)

    record = frontier.claim()
    assert record is not None
    assert record.url == "https://college.ac.in/high"
    assert record.domain == "college.ac.in"
    assert record.state == "in_progress"


def test_per_domain_daily_budget_exhaustion() -> None:
    frontier = make_frontier(max_pages_per_domain_per_day=2)
    for index in range(3):
        frontier.enqueue(f"https://college.ac.in/page-{index}")

    for _ in range(2):
        record = frontier.claim()
        assert record is not None
        frontier.complete(record.url, valuable_pages=1)

    assert frontier.budget_remaining("college.ac.in") == 0
    assert frontier.claim() is None


def test_budget_resets_next_day() -> None:
    frontier = make_frontier(max_pages_per_domain_per_day=1)
    frontier.enqueue("https://college.ac.in/a")
    record = frontier.claim()
    assert record is not None
    frontier.complete(record.url)

    assert frontier.claim() is None
    frontier.clock.advance(24 * 60 * 60)
    frontier.enqueue("https://college.ac.in/b")
    assert frontier.claim() is not None


def test_stop_crawl_below_two_percent_after_200_pages() -> None:
    stats = InMemoryCrawlStats()
    frontier = make_frontier(stats=stats, max_pages_per_domain_per_day=1000)
    stats.record_page(
        "college.ac.in",
        "https://college.ac.in/bulk",
        frontier.clock.now(),
        valuable=1,
        pages=200,
    )
    frontier.enqueue("https://college.ac.in/page")

    assert frontier.yield_score("college.ac.in") == 0.005
    assert frontier.should_stop("college.ac.in")
    assert frontier.stop_reason("college.ac.in") == "yield_below_2%_after_200_pages"
    assert frontier.claim() is None


def test_stop_crawl_not_triggered_at_good_yield() -> None:
    stats = InMemoryCrawlStats()
    frontier = make_frontier(stats=stats, max_pages_per_domain_per_day=1000)
    stats.record_page(
        "college.ac.in",
        "https://college.ac.in/bulk",
        frontier.clock.now(),
        valuable=40,
        pages=200,
    )
    assert not frontier.should_stop("college.ac.in")
    frontier.enqueue("https://college.ac.in/page")
    assert frontier.claim() is not None


def test_quarantined_domain_is_skipped() -> None:
    clock = FakeClock()
    frontier = Frontier(
        InMemoryFrontierRepo(),
        stats=InMemoryCrawlStats(),
        domain_state=InMemoryDomainStateStore(),
        clock=clock,
    )
    frontier.enqueue("https://college.ac.in/page")
    state = frontier.domain_state.load("college.ac.in")
    state.quarantined_until = clock.now() + timedelta(hours=6)
    frontier.domain_state.save(state)

    assert frontier.claim() is None
    clock.advance(6 * 60 * 60 + 1)
    assert frontier.claim() is not None


def test_completion_persists_yield_and_attempts() -> None:
    frontier = make_frontier()
    frontier.enqueue("https://college.ac.in/report.pdf")
    record = frontier.claim()
    assert record is not None

    done = frontier.complete(record.url, valuable_pages=1, pages=1, tier="httpx")
    assert done is not None
    assert done.state == "done"
    assert done.attempts == 1
    assert done.tier_used == "httpx"
    assert frontier.yield_score("college.ac.in") == 1.0


def test_failure_retries_then_marks_failed() -> None:
    frontier = make_frontier(max_attempts=2)
    frontier.enqueue("https://college.ac.in/flaky")
    record = frontier.claim()
    assert record is not None

    retried = frontier.fail(record.url, retry_after=0)
    assert retried is not None
    assert retried.state == "queued"
    assert retried.attempts == 1

    record = frontier.claim()
    assert record is not None
    failed = frontier.fail(record.url, retry_after=0)
    assert failed is not None
    assert failed.state == "failed"
    assert failed.attempts == 2


def test_retry_backoff_moves_next_allowed_later() -> None:
    frontier = make_frontier()
    url = "https://college.ac.in/flaky"
    frontier.enqueue(url)
    claimed = frontier.claim()
    assert claimed is not None
    before = frontier.repo.get(url)
    assert before is not None

    retried = frontier.fail(url, retry_after=120)
    assert retried is not None
    after = frontier.repo.get(url)
    assert after is not None

    assert after.next_allowed_at == before.next_allowed_at + timedelta(seconds=120)
    assert after.next_allowed_at > before.next_allowed_at
    assert frontier.claim() is None

    frontier.clock.advance(121)
    assert frontier.claim() is not None


def test_duplicate_enqueue_does_not_push_next_allowed_later() -> None:
    frontier = make_frontier()
    url = "https://college.ac.in/repeat"
    scheduled = frontier.clock.now() + timedelta(seconds=600)

    first = frontier.enqueue(url, next_allowed_at=scheduled)
    duplicate = frontier.enqueue(url)

    assert first.next_allowed_at == scheduled
    assert duplicate.next_allowed_at == frontier.clock.now()
    assert duplicate.next_allowed_at <= scheduled


def test_enqueue_ref_honours_ref_priority() -> None:
    from fmd.discover.base import DocumentRef

    frontier = make_frontier()
    record = frontier.enqueue_ref(DocumentRef("https://college.ac.in/x", priority=9))
    assert record.priority == 9


def test_frontier_counts_are_not_in_memory_only() -> None:
    repo = InMemoryFrontierRepo()
    clock = FakeClock()
    frontier = Frontier(
        repo, stats=InMemoryCrawlStats(), domain_state=InMemoryDomainStateStore(), clock=clock
    )
    frontier.enqueue("https://college.ac.in/a")
    record = frontier.claim()
    assert record is not None
    frontier.complete(record.url)

    # A fresh worker sharing the same repo sees the completed state.
    restarted = Frontier(
        repo, stats=InMemoryCrawlStats(), domain_state=InMemoryDomainStateStore(), clock=clock
    )
    assert restarted.counts().get("done") == 1
