"""Postgres-backed crawl frontier (brief §7, §21).

The WS-A contract gives us ``CrawlFrontierRepo`` (enqueue/get/list_due/set_state/
count_by_state). Everything else needed for a restartable crawl — per-domain
daily budget, yield score, stop-crawl rule, quarantine — is kept in small
protocols with in-memory implementations for tests and Postgres implementations
in :mod:`fmd.frontier.state`. No crawler state lives only in a process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import Protocol

from fmd.db.repositories import CrawlFrontierRepo, FrontierRecord, FrontierUpsert
from fmd.fetch.politeness import SystemClock, registrable_domain
from fmd.frontier.state import (
    DomainStateStore,
    InMemoryDomainStateStore,
    PostgresDomainStateStore,
)

logger = logging.getLogger(__name__)

STOP_YIELD_THRESHOLD = 0.02
STOP_AFTER_PAGES = 200
DEFAULT_MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 60.0


class FrontierRepo(Protocol):
    def enqueue(self, item: FrontierUpsert) -> FrontierRecord:  # pragma: no cover
        ...

    def get(self, url: str) -> FrontierRecord | None:  # pragma: no cover
        ...

    def list_due(self, at: datetime, limit: int = 50) -> list[FrontierRecord]:  # pragma: no cover
        ...

    def set_next_allowed_at(
        self, url: str, when: datetime
    ) -> FrontierRecord | None:  # pragma: no cover
        ...

    def set_state(
        self,
        url: str,
        state,
        tier_used: str | None = None,
        attempts: int | None = None,
    ) -> FrontierRecord | None:  # pragma: no cover
        ...

    def count_by_state(self) -> dict[str, int]:  # pragma: no cover
        ...


class CrawlStatsStore(Protocol):
    def record_page(
        self, domain: str, url: str, when: datetime, *, valuable: float, pages: int
    ) -> None:  # pragma: no cover - protocol
        ...

    def pages_today(self, domain: str, since: datetime) -> int:  # pragma: no cover
        ...

    def domain_yield(self, domain: str) -> tuple[int, float]:  # pragma: no cover
        """Return ``(pages_fetched, valuable_pages)`` for the whole crawl."""
        ...


class Clock(Protocol):
    def now(self) -> datetime:  # pragma: no cover - protocol
        ...


@dataclass
class InMemoryFrontierRepo:
    """Deterministic stand-in for ``CrawlFrontierRepo`` in offline tests."""

    _items: dict[str, FrontierRecord] = field(default_factory=dict)

    def enqueue(self, item: FrontierUpsert) -> FrontierRecord:
        existing = self._items.get(item.url)
        next_allowed = item.next_allowed_at or datetime.now(UTC)
        if existing is None:
            data = item.model_dump()
            data["next_allowed_at"] = next_allowed
            data["updated_at"] = datetime.now(UTC)
            record = FrontierRecord(**data)
        else:
            record = existing.model_copy(
                update={
                    "domain": item.domain,
                    "priority": item.priority,
                    "next_allowed_at": min(existing.next_allowed_at, next_allowed),
                    "state": "done" if existing.state == "done" else item.state,
                    "updated_at": datetime.now(UTC),
                }
            )
        self._items[item.url] = record
        return record

    def get(self, url: str) -> FrontierRecord | None:
        return self._items.get(url)

    def set_next_allowed_at(self, url: str, when: datetime) -> FrontierRecord | None:
        record = self._items.get(url)
        if record is None:
            return None
        record = record.model_copy(
            update={"next_allowed_at": when, "updated_at": datetime.now(UTC)}
        )
        self._items[url] = record
        return record

    def list_due(self, at: datetime, limit: int = 50) -> list[FrontierRecord]:
        due = [
            record
            for record in self._items.values()
            if record.state == "queued" and record.next_allowed_at <= at
        ]
        due.sort(key=lambda r: (-r.priority, r.next_allowed_at))
        return due[:limit]

    def set_state(
        self,
        url: str,
        state,
        tier_used: str | None = None,
        attempts: int | None = None,
    ) -> FrontierRecord | None:
        record = self._items.get(url)
        if record is None:
            return None
        record = record.model_copy(
            update={
                "state": state,
                "tier_used": tier_used if tier_used is not None else record.tier_used,
                "attempts": attempts if attempts is not None else record.attempts,
                "updated_at": datetime.now(UTC),
            }
        )
        self._items[url] = record
        return record

    def count_by_state(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._items.values():
            counts[record.state] = counts.get(record.state, 0) + 1
        return counts


@dataclass
class InMemoryCrawlStats:
    pages: dict[str, int] = field(default_factory=dict)
    valuable: dict[str, float] = field(default_factory=dict)
    day: dict[str, tuple[object, int]] = field(default_factory=dict)

    def record_page(
        self, domain: str, url: str, when: datetime, *, valuable: float, pages: int
    ) -> None:
        self.pages[domain] = self.pages.get(domain, 0) + pages
        self.valuable[domain] = self.valuable.get(domain, 0.0) + valuable
        day_key = when.date()
        stored_day, count = self.day.get(domain, (day_key, 0))
        if stored_day != day_key:
            count = 0
        self.day[domain] = (day_key, count + pages)

    def pages_today(self, domain: str, since: datetime) -> int:
        day_key, count = self.day.get(domain, (since.date(), 0))
        return count if day_key == since.date() else 0

    def domain_yield(self, domain: str) -> tuple[int, float]:
        return self.pages.get(domain, 0), self.valuable.get(domain, 0.0)


class PostgresCrawlStats:
    """Budget/yield derived from the existing ``crawl_frontier`` rows."""

    def __init__(self, database) -> None:
        self.repo = CrawlFrontierRepo(database)

    def record_page(
        self, domain: str, url: str, when: datetime, *, valuable: float, pages: int
    ) -> None:
        if pages <= 0:
            return
        self.repo.update_yield(domain, pages, valuable / pages)

    def pages_today(self, domain: str, since: datetime) -> int:
        return self.repo.count_for_domain_since(domain, since)

    def domain_yield(self, domain: str) -> tuple[int, float]:
        return self.repo.yield_for_domain(domain)


def day_start(at: datetime) -> datetime:
    return datetime.combine(at.date(), time.min, tzinfo=UTC)


class Frontier:
    """Priority queue with per-domain budget, yield score and stop-crawl."""

    def __init__(
        self,
        repo: FrontierRepo,
        *,
        stats: CrawlStatsStore | None = None,
        domain_state: DomainStateStore | None = None,
        clock: Clock | None = None,
        max_pages_per_domain_per_day: int | None = None,
        stop_yield: float = STOP_YIELD_THRESHOLD,
        stop_after_pages: int = STOP_AFTER_PAGES,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.repo = repo
        self.stats = stats or InMemoryCrawlStats()
        self.domain_state = domain_state or InMemoryDomainStateStore()
        self.clock = clock or SystemClock()
        if max_pages_per_domain_per_day is None:
            from fmd.settings import get_settings

            max_pages_per_domain_per_day = get_settings().max_pages_per_domain_per_day
        self.max_pages_per_domain_per_day = max_pages_per_domain_per_day
        self.stop_yield = stop_yield
        self.stop_after_pages = stop_after_pages
        self.max_attempts = max_attempts

    @classmethod
    def from_database(cls, database) -> Frontier:
        """Production constructor: every queue/stat/breaker field is in Postgres."""
        return cls(
            CrawlFrontierRepo(database),
            stats=PostgresCrawlStats(database),
            domain_state=PostgresDomainStateStore(database),
        )

    # ------------------------------------------------------------------ enqueue
    def enqueue(
        self,
        url: str,
        *,
        priority: int = 0,
        next_allowed_at: datetime | None = None,
    ) -> FrontierRecord:
        domain = registrable_domain(url)
        return self.repo.enqueue(
            FrontierUpsert(
                url=url,
                domain=domain,
                priority=priority,
                next_allowed_at=next_allowed_at or self.clock.now(),
                state="queued",
            )
        )

    def enqueue_ref(self, ref, *, priority: int | None = None) -> FrontierRecord:
        """Enqueue any object exposing the ``DocumentRef`` protocol."""
        if priority is None:
            priority = getattr(ref, "priority", 0) or 0
        return self.enqueue(ref.url, priority=priority)

    # ------------------------------------------------------------------- budget
    def pages_today(self, domain: str, *, at: datetime | None = None) -> int:
        at = at or self.clock.now()
        return self.stats.pages_today(domain, day_start(at))

    def budget_remaining(self, domain: str, *, at: datetime | None = None) -> int:
        return max(0, self.max_pages_per_domain_per_day - self.pages_today(domain, at=at))

    def yield_score(self, domain: str) -> float:
        pages, valuable = self.stats.domain_yield(domain)
        return 0.0 if pages == 0 else min(1.0, valuable / pages)

    def should_stop(self, domain: str) -> bool:
        pages, _ = self.stats.domain_yield(domain)
        return pages >= self.stop_after_pages and self.yield_score(domain) < self.stop_yield

    def stop_reason(self, domain: str) -> str | None:
        if self.should_stop(domain):
            return f"yield_below_{self.stop_yield:.0%}_after_{self.stop_after_pages}_pages"
        return None

    def _quarantined(self, domain: str) -> bool:
        state = self.domain_state.load(domain)
        until = state.quarantined_until
        if until is None:
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=UTC)
        return until > self.clock.now()

    # -------------------------------------------------------------------- claim
    def claim(self, *, limit: int = 50, at: datetime | None = None) -> FrontierRecord | None:
        at = at or self.clock.now()
        for record in self.repo.list_due(at, limit):
            if self._quarantined(record.domain):
                continue
            if self.budget_remaining(record.domain, at=at) <= 0:
                continue
            if self.should_stop(record.domain):
                logger.info("stop-crawl for %s: %s", record.domain, self.stop_reason(record.domain))
                continue
            updated = self.repo.set_state(record.url, "in_progress")
            return updated or record
        return None

    # ------------------------------------------------------------------ outcomes
    def complete(
        self,
        url: str,
        *,
        valuable_pages: float = 0,
        pages: int = 1,
        tier: str | None = None,
    ) -> FrontierRecord | None:
        record = self.repo.get(url)
        if record is None:
            return None
        if pages > 0:
            self.stats.record_page(
                record.domain,
                url,
                self.clock.now(),
                valuable=valuable_pages,
                pages=pages,
            )
        attempts = record.attempts + 1
        return self.repo.set_state(url, "done", tier_used=tier, attempts=attempts)

    def fail(
        self,
        url: str,
        *,
        blocked: bool = False,
        retry_after: float | None = None,
    ) -> FrontierRecord | None:
        record = self.repo.get(url)
        if record is None:
            return None
        attempts = record.attempts + 1
        if blocked:
            return self.repo.set_state(url, "quarantined", attempts=attempts)
        if attempts >= self.max_attempts:
            return self.repo.set_state(url, "failed", attempts=attempts)
        backoff = (
            retry_after if retry_after is not None else BACKOFF_BASE_SECONDS * (2 ** (attempts - 1))
        )
        retry_at = self.clock.now() + timedelta(seconds=backoff)
        # Backoff must move the URL later, so it bypasses `enqueue` (whose
        # conflict clause keeps the earlier of two discovery times).
        self.repo.set_next_allowed_at(url, retry_at)
        return self.repo.set_state(url, "queued", attempts=attempts)

    def counts(self) -> dict[str, int]:
        return self.repo.count_by_state()
