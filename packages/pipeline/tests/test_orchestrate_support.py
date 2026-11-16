"""Offline fakes for orchestration tests (no network, no Postgres).

Kept in a ``test_orchestrate*.py`` module so it ships with the orchestration
tests. The fakes implement the repository methods the orchestrator calls, using
the real record models so a schema/type change breaks the fake too.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fmd.db.repositories import (
    ClaimCreate,
    ClaimRecord,
    DocumentRecord,
    DocumentUpsert,
    EntityRecord,
    ExtractionCreate,
    ExtractionRecord,
    FrontierRecord,
    ProgrammeFilter,
    ProgrammeRecord,
    ScoreCreate,
    ScoreRecord,
    SnapshotCreate,
    SnapshotRecord,
)
from fmd.discover.base import DocumentRef
from fmd.fetch.types import FetchRequest, FetchResult
from fmd.frontier.queue import Frontier, InMemoryCrawlStats, InMemoryFrontierRepo
from fmd.frontier.state import InMemoryDomainStateStore


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return 0.0

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


def make_frontier(*, clock: FakeClock | None = None, **kwargs: Any) -> Frontier:
    clock = clock or FakeClock()
    return Frontier(
        InMemoryFrontierRepo(),
        stats=InMemoryCrawlStats(),
        domain_state=InMemoryDomainStateStore(),
        clock=clock,
        **kwargs,
    )


@dataclass
class FakeSource:
    id: str
    adapter: str = "institution_site"
    kind: str = "institution"
    enabled: bool = True
    seeds: list[str] = field(default_factory=list)


class FakeAdapter:
    def __init__(
        self, refs: Iterable[DocumentRef] | None = None, *, error: Exception | None = None
    ):
        self.refs = list(refs or [])
        self.error = error
        self.calls = 0

    def fetch(self, *, domains: Iterable[str] | None = None) -> list[DocumentRef]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.refs)


class FakeFetcher:
    """Deterministic fetch ladder stand-in keyed by URL."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest | str) -> FetchResult:
        req = request if isinstance(request, FetchRequest) else FetchRequest(url=str(request))
        self.requests.append(req)
        value = self.responses.get(req.url)
        if value is None:
            return FetchResult(url=req.url, final_url=req.url, status=0, error="no_fake_response")
        if callable(value):
            value = value(req)
        if isinstance(value, bytes):
            return FetchResult(url=req.url, final_url=req.url, status=200, body=value)
        return value


def response(
    url: str,
    *,
    body: bytes | str = b"",
    status: int = 200,
    headers: dict[str, str] | None = None,
    tier: str = "httpx",
) -> FetchResult:
    payload = body.encode("utf-8") if isinstance(body, str) else body
    return FetchResult(
        url=url,
        final_url=url,
        status=status,
        headers=dict(headers or {}),
        body=payload,
        fetcher_tier=tier,
    )


class MemoryDocuments:
    def __init__(self) -> None:
        self.rows: dict[str, DocumentRecord] = {}
        self._next = 1

    def upsert(self, document: DocumentUpsert) -> DocumentRecord:
        existing = self.rows.get(document.url)
        if existing is None:
            data = document.model_dump()
            data["first_seen"] = data.get("first_seen") or datetime.now(UTC)
            data["last_seen"] = data.get("last_seen") or datetime.now(UTC)
            record = DocumentRecord(**data, id=self._next)
            self._next += 1
            self.rows[document.url] = record
            return record
        updates: dict[str, Any] = {}
        for key, value in document.model_dump().items():
            if value is not None:
                updates[key] = value
        merged = existing.model_copy(update=updates)
        self.rows[document.url] = merged
        return merged

    def get(self, document_id: int) -> DocumentRecord | None:
        return next((row for row in self.rows.values() if row.id == document_id), None)

    def get_by_url(self, url: str) -> DocumentRecord | None:
        return self.rows.get(url)


class MemorySnapshots:
    def __init__(self, extractions: MemoryExtractions | None = None) -> None:
        self.rows: list[SnapshotRecord] = []
        self.extractions = extractions
        self._next = 1

    def create(self, snapshot: SnapshotCreate) -> SnapshotRecord:
        for row in self.rows:
            if row.document_id == snapshot.document_id and row.sha256 == snapshot.sha256:
                return row
        record = SnapshotRecord(**snapshot.model_dump(), id=self._next)
        self._next += 1
        self.rows.append(record)
        return record

    def get(self, snapshot_id: int) -> SnapshotRecord | None:
        return next((row for row in self.rows if row.id == snapshot_id), None)

    def latest_for_document(self, document_id: int) -> SnapshotRecord | None:
        matches = [row for row in self.rows if row.document_id == document_id]
        return max(matches, key=lambda row: row.id) if matches else None

    def list_for_document(self, document_id: int) -> list[SnapshotRecord]:
        return [row for row in self.rows if row.document_id == document_id]

    def list_without_extraction(
        self, tool_version: str | None = None, limit: int = 100
    ) -> list[SnapshotRecord]:
        rows = self.rows
        if tool_version is not None and self.extractions is not None:
            done = {
                row.snapshot_id for row in self.extractions.rows if row.tool_version == tool_version
            }
            rows = [row for row in rows if row.id not in done]
        return sorted(rows, key=lambda row: row.id)[:limit]


class MemoryExtractions:
    def __init__(self) -> None:
        self.rows: list[ExtractionRecord] = []
        self._next = 1

    def create(self, extraction: ExtractionCreate) -> ExtractionRecord:
        record = ExtractionRecord(**extraction.model_dump(), id=self._next)
        self._next += 1
        self.rows.append(record)
        return record

    def get(self, extraction_id: int) -> ExtractionRecord | None:
        return next((row for row in self.rows if row.id == extraction_id), None)

    def list_for_snapshot(self, snapshot_id: int) -> list[ExtractionRecord]:
        return [row for row in self.rows if row.snapshot_id == snapshot_id]


class MemoryClaims:
    def __init__(self) -> None:
        self.rows: list[ClaimRecord] = []
        self._next = 1

    def create(self, claim: ClaimCreate) -> ClaimRecord:
        record = ClaimRecord(**claim.model_dump(), id=self._next, superseded_by=None)
        self._next += 1
        self.rows.append(record)
        return record

    def list_for_entity(self, entity_id: int, **_: Any) -> list[ClaimRecord]:
        return [row for row in self.rows if row.entity_id == entity_id]


class MemoryEntities:
    def __init__(self, rows: Iterable[EntityRecord] | None = None) -> None:
        self.rows = list(rows or [])

    def find(self, kind: str, canonical_name: str) -> EntityRecord | None:
        return next(
            (row for row in self.rows if row.kind == kind and row.canonical_name == canonical_name),
            None,
        )


class MemoryProgrammes:
    def __init__(self, rows: Iterable[ProgrammeRecord] | None = None) -> None:
        self.rows = list(rows or [])

    def list(self, filters: ProgrammeFilter) -> list[ProgrammeRecord]:
        return self.rows[filters.offset : filters.offset + filters.limit]


class MemoryScores:
    def __init__(self) -> None:
        self.rows: list[ScoreRecord] = []
        self._next = 1

    def create(self, score: ScoreCreate) -> ScoreRecord:
        record = ScoreRecord(**score.model_dump(), id=self._next)
        self._next += 1
        self.rows.append(record)
        return record

    def latest_for_entity(self, entity_id: int) -> ScoreRecord | None:
        matches = [row for row in self.rows if row.entity_id == entity_id]
        return max(matches, key=lambda row: row.id) if matches else None


def frontier_rows(frontier: Frontier) -> list[FrontierRecord]:
    repo = frontier.repo
    return list(getattr(repo, "_items", {}).values())
