"""Persisted per-domain crawl state (politeness, circuit breaker, robots).

The tables live in ``db/migrations`` (``domain_politeness``). SQL for them lives
in :class:`~fmd.db.repositories.CrawlFrontierRepo`; this module only maps the
in-memory :class:`DomainState` onto a repo row so the fetch package can stay
storage-agnostic. There is no SQL in ``fmd/frontier``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from fmd.db.repositories import CrawlFrontierRepo, DomainPolitenessRecord


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


@dataclass
class DomainState:
    """All mutable crawl state for one registrable domain."""

    domain: str
    rps: float | None = None
    crawl_delay: float | None = None
    consecutive_5xx: int = 0
    quarantined_until: datetime | None = None
    last_request_at: datetime | None = None
    robots_fetched_at: datetime | None = None
    robots_json: dict | None = None
    sitemaps: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)


class DomainStateStore(Protocol):
    def load(self, domain: str) -> DomainState:  # pragma: no cover - protocol
        ...

    def save(self, state: DomainState) -> None:  # pragma: no cover - protocol
        ...


class InMemoryDomainStateStore:
    """Deterministic default, used by tests and single-process runs."""

    def __init__(self) -> None:
        self._states: dict[str, DomainState] = {}

    def load(self, domain: str) -> DomainState:
        state = self._states.get(domain)
        if state is None:
            state = DomainState(domain=domain)
            self._states[domain] = state
        return state

    def save(self, state: DomainState) -> None:
        self._states[state.domain] = state


class PostgresDomainStateStore:
    """Persist domain politeness through the repository (no SQL here)."""

    def __init__(self, database) -> None:
        self.repo = CrawlFrontierRepo(database)

    def load(self, domain: str) -> DomainState:
        row = self.repo.get_politeness(domain)
        if row is None:
            return DomainState(domain=domain)
        return DomainState(
            domain=domain,
            rps=row.rps,
            crawl_delay=row.crawl_delay,
            quarantined_until=_as_utc(row.quarantined_until),
            robots_fetched_at=_as_utc(row.robots_fetched_at),
            robots_json=row.robots,
        )

    def save(self, state: DomainState) -> None:
        self.repo.upsert_politeness(
            DomainPolitenessRecord(
                domain=state.domain,
                rps=state.rps,
                crawl_delay=state.crawl_delay,
                robots=state.robots_json,
                robots_fetched_at=state.robots_fetched_at,
                quarantined_until=state.quarantined_until,
            )
        )
