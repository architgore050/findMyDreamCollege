from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from test_orchestrate_support import (
    FakeAdapter,
    FakeClock,
    FakeFetcher,
    FakeSource,
    MemoryClaims,
    MemoryDocuments,
    MemoryEntities,
    MemoryExtractions,
    MemoryProgrammes,
    MemoryScores,
    MemorySnapshots,
    frontier_rows,
    make_frontier,
    response,
)

from fmd.adapters.base import SourceUnavailable
from fmd.db.repositories import (
    DocumentUpsert,
    FrontierUpsert,
    ProgrammeRecord,
    ScoreCreate,
    SnapshotCreate,
)
from fmd.discover.base import DocumentRef
from fmd.er import EntityCandidate, ERConfig
from fmd.extract.pipeline import build_router
from fmd.orchestrate import JsonFileLedger, Orchestrator
from fmd.orchestrate.crawl import Crawler
from fmd.orchestrate.ledger import RunSummary
from fmd.plan import DocumentPlanner
from fmd.store import SnapshotStore

TESTS = Path(__file__).parent
EXTRACT_FIXTURES = TESTS / "fixtures" / "extract"
LLM_FIXTURES = EXTRACT_FIXTURES / "llm"
PLAN_FIXTURES = TESTS / "fixtures" / "plan"
PLAN_LLM_FIXTURES = PLAN_FIXTURES / "llm"


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        registry: Any,
        *,
        sources: list[FakeSource] | None = None,
        adapters: dict[str, Any] | None = None,
        fetcher: FakeFetcher | None = None,
        router: Any = None,
        frontend_limit: int = 10,
    ) -> None:
        self.clock = FakeClock()
        self.extractions = MemoryExtractions()
        self.documents = MemoryDocuments()
        self.snapshots = MemorySnapshots(self.extractions, self.documents)
        self.claims = MemoryClaims()
        self.entities = MemoryEntities()
        self.programmes = MemoryProgrammes()
        self.scores = MemoryScores()
        self.store = SnapshotStore(tmp_path / "snapshots")
        self.ledger = JsonFileLedger(tmp_path / "ledger")
        self.frontier = make_frontier(clock=self.clock, max_pages_per_domain_per_day=frontend_limit)
        self.sources = sources or []
        self.adapters = adapters or {}
        self.fetcher = fetcher or FakeFetcher({})
        self.run_ids = itertools.count(1)
        self.orchestrator = Orchestrator(
            registry=registry,
            database=None,
            store=self.store,
            frontier=self.frontier,
            documents=self.documents,
            snapshots=self.snapshots,
            extractions=self.extractions,
            claims=self.claims,
            programmes=self.programmes,
            scores=self.scores,
            entities=self.entities,
            sources=self.sources,
            adapter_factory=self._adapter,
            fetcher=self.fetcher,
            ledger=self.ledger,
            router=router,
            run_id_factory=lambda: f"run-{next(self.run_ids)}",
            score_persist=self._persist,
        )

    def _adapter(self, source: FakeSource) -> Any:
        return self.adapters[source.id]

    def _persist(self, _database: Any, result: Any) -> None:
        self.scores.create(ScoreCreate(**result.to_score_create()))


def _ref(url: str, *, source_id: str = "fixture", **metadata: Any) -> DocumentRef:
    return DocumentRef(url=url, source_id=source_id, metadata=dict(metadata))


def test_crawl_happy_path_stores_snapshot_and_document(tmp_path, registry) -> None:
    url = "https://college.ac.in/placement.txt"
    source = FakeSource(id="fixture")
    adapter = FakeAdapter([_ref(url, family="placement_report", entity_id=7)])
    fetcher = FakeFetcher(
        {
            url: response(
                url,
                body="Placement report 2024-25",
                headers={"content-type": "text/plain", "etag": '"v1"'},
            )
        }
    )
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={source.id: adapter},
        fetcher=fetcher,
    )

    summary = harness.orchestrator.crawl(limit=5)

    assert summary.failed is False
    assert summary.counts["enqueued"] == 1
    assert summary.counts["fetched"] == 1
    assert summary.counts["stored"] == 1
    assert len(harness.documents.rows) == 1
    assert len(harness.snapshots.rows) == 1
    snapshot = harness.snapshots.rows[0]
    assert harness.store.exists(snapshot.sha256)
    assert snapshot.content_type == "text/plain"
    document = harness.documents.rows[url]
    assert document.etag == '"v1"'
    # Plan is on the row now, not in request_headers (migration 0003).
    assert document.source_id == "fixture"
    assert document.extraction_family == "placement_report"
    assert document.entity_id == 7
    assert document.plan_state == "linked"
    assert frontier_rows(harness.frontier)[0].source_id == "fixture"
    assert frontier_rows(harness.frontier)[0].state == "done"
    # The ledger was persisted and is greppable JSON.
    recorded = json.loads((harness.ledger.runs_dir / f"{summary.run_id}.json").read_text())
    assert recorded["stage"] == "crawl"
    assert recorded["counts"]["stored"] == 1
    assert json.loads(summary.as_json())["run_id"] == summary.run_id


def test_crawl_304_bumps_last_seen_without_snapshot(tmp_path, registry) -> None:
    url = "https://college.ac.in/placement.txt"
    source = FakeSource(id="fixture")
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={source.id: FakeAdapter([_ref(url, family="placement_report", entity_id=1)])},
        fetcher=FakeFetcher({url: lambda request: response(url, status=304)}),
    )
    old = datetime(2025, 1, 1, tzinfo=UTC)
    harness.documents.upsert(
        DocumentUpsert(url=url, domain="college.ac.in", etag='"v1"', last_seen=old)
    )
    digest = harness.store.put(b"old body")
    document = harness.documents.rows[url]
    harness.snapshots.create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=str(harness.store.path_for(digest)),
            content_type="text/plain",
        )
    )

    summary = harness.orchestrator.crawl(limit=5)

    assert summary.counts["not_modified"] == 1
    assert summary.counts.get("stored", 0) == 0
    assert len(harness.snapshots.rows) == 1
    assert harness.fetcher.requests[0].etag == '"v1"'
    assert harness.documents.rows[url].last_seen > old
    record = frontier_rows(harness.frontier)[0]
    assert record.state == "done"
    assert record.attempts == 1
    # A 304 costs no crawl budget.
    assert harness.frontier.pages_today("college.ac.in") == 0


POST_URL = "https://nabh.co/wp-admin/admin-ajax.php?action=get_hospitals&page=2"
POST_BODY = {"action": "get_hospitals", "page": 2}
POST_HEADERS = {"Accept": "application/json", "Referer": "https://nabh.co/"}


def _post_ref(url: str = POST_URL, *, page: int = 2, source_id: str = "fixture") -> DocumentRef:
    return DocumentRef(
        url=url,
        source_id=source_id,
        metadata={
            "request_method": "POST",
            "request_body": {"action": "get_hospitals", "page": page},
            "request_headers": dict(POST_HEADERS),
        },
    )


def test_crawl_replays_persisted_post_method_body_and_headers(tmp_path, registry) -> None:
    source = FakeSource(id="fixture")
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={source.id: FakeAdapter([_post_ref()])},
        fetcher=FakeFetcher(
            {
                POST_URL: lambda request: response(
                    request.url,
                    body='{"rows":[{"id":2}]}',
                    headers={"content-type": "application/json"},
                )
            }
        ),
    )

    summary = harness.orchestrator.crawl(limit=5)

    assert summary.counts["stored"] == 1
    assert summary.metadata["request_methods"] == {"POST": 1}
    request = harness.fetcher.requests[0]
    assert request.method == "POST"
    assert request.body == POST_BODY
    assert request.headers["Accept"] == "application/json"
    document = harness.documents.rows[POST_URL]
    assert document.http_method == "POST"
    assert document.http_body == POST_BODY
    assert document.request_headers["Accept"] == "application/json"
    row = frontier_rows(harness.frontier)[0]
    assert row.http_method == "POST"
    assert row.http_body == POST_BODY


def test_crawl_304_preserves_the_post_shape_and_validators(tmp_path, registry) -> None:
    clock = FakeClock()
    frontier = make_frontier(clock=clock)
    frontier.repo.enqueue(
        FrontierUpsert.from_document_ref(
            _post_ref(),
            domain="nabh.co",
            next_allowed_at=clock.now(),
            state="queued",
        )
    )
    documents = MemoryDocuments()
    documents.upsert(
        DocumentUpsert(
            url=POST_URL,
            domain="nabh.co",
            etag='"v1"',
            http_method="POST",
            http_body=POST_BODY,
            request_headers=dict(POST_HEADERS),
        )
    )
    snapshots = MemorySnapshots()
    fetcher = FakeFetcher({POST_URL: lambda request: response(request.url, status=304)})
    crawler = Crawler(
        frontier=frontier,
        fetcher=fetcher,
        store=None,
        documents=documents,
        snapshots=snapshots,
        adapter_factory=lambda source: None,
    )
    summary = RunSummary(run_id="t", stage="crawl", started_at=datetime.now(UTC))

    crawler.drain(summary=summary)

    assert summary.counts["not_modified"] == 1
    request = fetcher.requests[0]
    assert request.method == "POST"
    assert request.body == POST_BODY
    assert request.headers["If-None-Match"] == '"v1"'
    document = documents.rows[POST_URL]
    assert document.http_method == "POST"
    assert document.http_body == POST_BODY
    assert document.request_headers["Accept"] == "application/json"
    assert document.request_headers["If-None-Match"] == '"v1"'
    assert snapshots.rows == []


def test_crawl_retry_never_downgrades_post_to_get(tmp_path, registry) -> None:
    clock = FakeClock()
    frontier = make_frontier(clock=clock)
    frontier.repo.enqueue(
        FrontierUpsert.from_document_ref(
            _post_ref(),
            domain="nabh.co",
            next_allowed_at=clock.now(),
            state="queued",
        )
    )
    statuses = iter([500, 200])
    fetcher = FakeFetcher(
        {
            POST_URL: lambda request: response(
                request.url,
                status=next(statuses),
                body='{"rows":[{"id":2}]}',
                headers={"content-type": "application/json"},
            )
        }
    )
    crawler = Crawler(
        frontier=frontier,
        fetcher=fetcher,
        store=SnapshotStore(tmp_path / "snapshots"),
        documents=MemoryDocuments(),
        snapshots=MemorySnapshots(),
        adapter_factory=lambda source: None,
    )
    summary = RunSummary(run_id="t", stage="crawl", started_at=datetime.now(UTC))

    crawler.drain(summary=summary)
    assert summary.counts["failed"] == 1
    clock.advance(120)
    crawler.drain(summary=summary)

    assert len(fetcher.requests) == 2
    assert [request.method for request in fetcher.requests] == ["POST", "POST"]
    assert [request.body for request in fetcher.requests] == [POST_BODY, POST_BODY]
    assert summary.counts["stored"] == 1


def test_paginated_post_pages_return_distinct_rows(tmp_path, registry) -> None:
    """The failure mode is silent page-1 reuse, so compare row ids, not counts."""
    page_one = "https://nabh.co/wp-admin/admin-ajax.php?action=get_hospitals&page=1"
    page_two = POST_URL
    source = FakeSource(id="fixture")

    def page(request):
        page_number = request.body["page"]
        return response(
            request.url,
            body=json.dumps({"rows": [{"id": page_number * 10}]}),
            headers={"content-type": "application/json"},
        )

    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={
            source.id: FakeAdapter(
                [
                    _post_ref(page_one, page=1),
                    _post_ref(page_two, page=2),
                ]
            )
        },
        fetcher=FakeFetcher({page_one: page, page_two: page}),
    )

    summary = harness.orchestrator.crawl(limit=5)

    assert summary.counts["stored"] == 2
    ids = {
        json.loads(harness.store.get(snapshot.sha256))["rows"][0]["id"]
        for snapshot in harness.snapshots.rows
    }
    assert ids == {10, 20}
    assert summary.counts.get("pagination_repeat", 0) == 0


def test_identical_consecutive_pages_raise_a_pagination_repeat_warning(tmp_path, registry) -> None:
    page_one = "https://nabh.co/wp-admin/admin-ajax.php?action=get_hospitals&page=1"
    page_two = POST_URL
    source = FakeSource(id="fixture")
    body = json.dumps({"rows": [{"id": 10}]})

    def page(request):
        return response(request.url, body=body, headers={"content-type": "application/json"})

    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={
            source.id: FakeAdapter([_post_ref(page_one, page=1), _post_ref(page_two, page=2)])
        },
        fetcher=FakeFetcher({page_one: page, page_two: page}),
    )

    summary = harness.orchestrator.crawl(limit=5)

    assert summary.counts["pagination_repeat"] == 1
    assert summary.counts["deduped"] == 1
    warnings = [row for row in summary.outcomes if row.get("reason") == "pagination_repeat"]
    assert len(warnings) == 1
    assert warnings[0]["previous_url"] == page_one
    assert warnings[0]["url"] == page_two
    assert summary.metadata["warnings"][0]["reason"] == "pagination_repeat"
    # A warning, not a failed run.
    assert summary.failed is False


def test_recrawl_does_not_downgrade_a_post_lead_to_get(tmp_path, registry) -> None:
    source = FakeSource(id="fixture")
    adapter = FakeAdapter([_post_ref()])
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={source.id: adapter},
        fetcher=FakeFetcher(
            {
                POST_URL: response(
                    POST_URL,
                    body='{"rows":[{"id":2}]}',
                    headers={"content-type": "application/json"},
                )
            }
        ),
    )
    harness.orchestrator.crawl(limit=5)

    # A later discovery pass that knows only the URL must not erase the shape.
    adapter.refs = [DocumentRef(url=POST_URL, source_id="fixture")]
    harness.orchestrator.crawl(limit=5)

    row = frontier_rows(harness.frontier)[0]
    assert row.http_method == "POST"
    assert row.http_body == POST_BODY
    assert row.request_headers["Accept"] == "application/json"


def test_crawl_dedupes_identical_content_across_urls(tmp_path, registry) -> None:
    first = "https://college.ac.in/a.txt"
    second = "https://college.ac.in/b.txt"
    source = FakeSource(id="fixture")
    body = "identical placement text"
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={
            source.id: FakeAdapter(
                [
                    _ref(first, family="placement_report", entity_id=1),
                    _ref(second, family="placement_report", entity_id=1),
                ]
            )
        },
        fetcher=FakeFetcher(
            {
                first: response(first, body=body, headers={"content-type": "text/plain"}),
                second: response(second, body=body, headers={"content-type": "text/plain"}),
            }
        ),
        frontend_limit=100,
    )

    summary = harness.orchestrator.crawl(limit=5)

    assert summary.counts["fetched"] == 2
    assert summary.counts["stored"] == 1
    assert summary.counts["deduped"] == 1
    assert len(harness.snapshots.rows) == 1
    assert summary.outcomes[-1]["reason"] == "duplicate_sha256"


def test_crawl_source_error_is_isolated_and_surfaced(tmp_path, registry) -> None:
    good = FakeSource(id="good")
    bad = FakeSource(id="bad")
    url = "https://good.ac.in/page.txt"
    harness = Harness(
        tmp_path,
        registry,
        sources=[good, bad],
        adapters={
            good.id: FakeAdapter([_ref(url, family="placement_report", entity_id=1)]),
            bad.id: FakeAdapter(error=SourceUnavailable("unreachable", detail="boom")),
        },
        fetcher=FakeFetcher(
            {url: response(url, body="text", headers={"content-type": "text/plain"})}
        ),
    )

    summary = harness.orchestrator.crawl(limit=5)

    assert summary.failed is False
    assert summary.counts["stored"] == 1
    statuses = {row["source_id"]: row["status"] for row in summary.sources}
    assert statuses == {"good": "ok", "bad": "error"}
    assert any(failure["source_id"] == "bad" for failure in summary.failures)


def test_crawl_dry_run_plans_urls_and_writes_nothing(tmp_path, registry) -> None:
    url = "https://college.ac.in/placement.txt"
    source = FakeSource(id="fixture")
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={source.id: FakeAdapter([_ref(url, family="placement_report", entity_id=1)])},
        fetcher=FakeFetcher({}),
    )

    summary = harness.orchestrator.crawl(limit=5, dry_run=True)

    assert summary.planned_urls == [url]
    assert summary.counts.get("enqueued", 0) == 0
    assert frontier_rows(harness.frontier) == []
    assert harness.documents.rows == {}
    assert harness.snapshots.rows == []
    assert harness.fetcher.requests == []
    assert not harness.ledger.runs_dir.exists()
    assert list(harness.store.objects.glob("*")) == []


def test_crawl_limit_caps_documents(tmp_path, registry) -> None:
    urls = [f"https://college.ac.in/{index}.txt" for index in range(3)]
    source = FakeSource(id="fixture")
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={
            source.id: FakeAdapter(
                [_ref(url, family="placement_report", entity_id=1) for url in urls]
            )
        },
        fetcher=FakeFetcher(
            {
                url: response(url, body=f"body {url}", headers={"content-type": "text/plain"})
                for url in urls
            }
        ),
        frontend_limit=100,
    )

    summary = harness.orchestrator.crawl(limit=2)

    assert summary.counts["enqueued"] == 2
    assert summary.counts["stored"] == 2


def _seed_snapshot(harness: Harness, document, text: str) -> None:
    digest = harness.store.put(text.encode("utf-8"))
    harness.snapshots.create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=str(harness.store.path_for(digest)),
            content_type="text/plain",
        )
    )


def test_crawl_plans_family_and_entity_without_ref_metadata(tmp_path, registry) -> None:
    """The regression this workstream exists for: no metadata, real claims."""
    url = "https://fixture.ac.in/placement-report.txt"
    source = FakeSource(id="fixture", params={"extraction_family": "placement_report"})
    text = (PLAN_FIXTURES / "e2e_placement_report.txt").read_text(encoding="utf-8")
    router = build_router(registry.models, mock=True, fixtures_dir=PLAN_LLM_FIXTURES)
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={source.id: FakeAdapter([_ref(url)])},
        fetcher=FakeFetcher(
            {url: response(url, body=text, headers={"content-type": "text/plain"})}
        ),
        router=router,
    )
    harness.orchestrator.planner = DocumentPlanner(
        candidates_provider=lambda: [
            EntityCandidate(
                entity_id=42,
                canonical_name="Fixture Institute of Medical Sciences",
                state="Delhi",
            )
        ],
        er_config=ERConfig.from_mapping(
            {"blocking": {"backend": "deterministic"}, "similarity": {"use_embeddings": False}}
        ),
    )

    crawl = harness.orchestrator.crawl(limit=5)
    assert crawl.counts["stored"] == 1
    document = harness.documents.rows[url]
    assert document.source_id == "fixture"
    assert document.extraction_family == "placement_report"
    assert document.entity_id == 42
    assert document.plan_state == "linked"

    extracted = harness.orchestrator.extract(limit=5)
    assert extracted.counts["extracted"] == 1
    assert extracted.counts["claims"] > 0
    assert {row.entity_id for row in harness.claims.rows} == {42}


def test_domain_entity_hint_links_a_discovered_pdf_to_its_institution(tmp_path, registry) -> None:
    from fmd.orchestrate.crawl import Crawler

    crawler = Crawler(
        frontier=make_frontier(),
        fetcher=FakeFetcher({}),
        store=None,
        documents=MemoryDocuments(),
        snapshots=MemorySnapshots(),
        adapter_factory=lambda source: None,
    )
    refs = crawler._with_entity_hints(
        [
            DocumentRef(url="https://www.alpha.example.ac.in/files/aqar.pdf"),
            DocumentRef(url="https://other.example.ac.in/files/aqar.pdf"),
        ],
        {"https://alpha.example.ac.in/": 7},
    )

    assert refs[0].metadata["entity_id"] == 7
    assert "entity_id" not in refs[1].metadata


def test_crawl_carries_an_anchor_family_hint_into_the_frontier(tmp_path, registry) -> None:
    """A generic PDF filename is classified by the link text that found it."""
    url = "https://college.ac.in/files/2024-25.pdf"
    source = FakeSource(id="fixture")
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={
            source.id: FakeAdapter(
                [
                    DocumentRef(
                        url=url,
                        source_id=source.id,
                        metadata={"anchor_text": "Annual Report 2024-25"},
                    )
                ]
            )
        },
        fetcher=FakeFetcher(
            {url: response(url, body=b"%PDF-1.4", headers={"content-type": "application/pdf"})}
        ),
    )

    summary = harness.orchestrator.crawl(limit=5)

    assert summary.counts.get("role.evidence") == 1
    document = harness.documents.rows[url]
    assert document.role == "evidence"
    assert document.extraction_family == "accreditation"


def test_extract_resumes_and_is_idempotent(tmp_path, registry) -> None:
    text = (EXTRACT_FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    harness = Harness(tmp_path, registry, router=router)
    first_doc = harness.documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/one.txt",
            domain="college.ac.in",
            kind="placement_report",
            extraction_family="placement_report",
            entity_id=1,
        )
    )
    second_doc = harness.documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/two.txt",
            domain="college.ac.in",
            kind="placement_report",
            extraction_family="placement_report",
            entity_id=1,
        )
    )
    _seed_snapshot(harness, first_doc, text)
    _seed_snapshot(harness, second_doc, text)

    first = harness.orchestrator.extract(limit=1)
    assert first.counts["extracted"] == 1
    assert len(harness.extractions.rows) == 1
    assert first.counts["claims"] > 0

    second_run = harness.orchestrator.extract(limit=10)
    assert second_run.counts["extracted"] == 1
    assert second_run.counts.get("skipped", 0) == 0
    assert len(harness.extractions.rows) == 2

    third = harness.orchestrator.extract(limit=10)
    assert third.counts.get("extracted", 0) == 0
    assert third.failed is False
    assert len(harness.extractions.rows) == 2


def test_extract_leaves_unplanned_documents_visible_with_a_reason(tmp_path, registry) -> None:
    text = (EXTRACT_FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    harness = Harness(tmp_path, registry, router=router)
    document = harness.documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/orphan.txt",
            domain="college.ac.in",
            extraction_family="placement_report",
            plan_state="needs_review",
        )
    )
    _seed_snapshot(harness, document, text)

    summary = harness.orchestrator.extract(limit=5)

    # A document with no entity is not selected (never extracted against a
    # guessed institution) and is surfaced as an unplanned bucket instead.
    assert summary.counts.get("extracted", 0) == 0
    assert summary.counts["unplanned"] == 1
    assert summary.metadata["unplanned"] == {"needs_review": 1}
    assert summary.outcomes == []
    assert harness.extractions.rows == []
    assert summary.failed is False


def test_extract_abstains_with_reason_for_an_unsupported_family(tmp_path, registry) -> None:
    text = (EXTRACT_FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    harness = Harness(tmp_path, registry, router=router)
    document = harness.documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/odd.txt",
            domain="college.ac.in",
            extraction_family="not_a_family",
            entity_id=1,
        )
    )
    _seed_snapshot(harness, document, text)

    summary = harness.orchestrator.extract(limit=5)

    assert summary.counts["abstained"] == 1
    assert summary.outcomes[0]["reason"] == "unsupported_family:not_a_family"
    assert harness.extractions.rows == []
    assert summary.failed is True



def test_acquisition_oom_fails_the_run_and_marks_the_document(
    tmp_path, registry, monkeypatch
) -> None:
    """A starved GPU is a visible failure, not ``ok:true`` with zero claims."""
    from fmd.acquire.errors import AcquisitionFailed

    text = (EXTRACT_FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    harness = Harness(tmp_path, registry, router=router)
    document = harness.documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/fee-structure.pdf",
            domain="college.ac.in",
            extraction_family="fee_structure",
            entity_id=1,
        )
    )
    _seed_snapshot(harness, document, text)

    def boom(self: Any, targets: Any) -> Any:
        raise AcquisitionFailed(
            "acquisition_failed:docling_cuda_oom",
            "CUDA out of memory. Tried to allocate 50.00 MiB",
        )

    monkeypatch.setattr("fmd.extract.pipeline.ExtractPipeline.run_batch", boom)

    summary = harness.orchestrator.extract(limit=5)
    payload = summary.as_dict()

    assert payload["ok"] is False
    assert summary.failed is True
    assert summary.counts["acquisition_failed"] == 1
    assert summary.failures[0]["reason"] == "acquisition_failed:docling_cuda_oom"
    assert summary.outcomes[0]["outcome"] == "acquisition_failed"
    assert harness.extractions.rows == []
    updated = harness.documents.get(document.id)
    assert updated is not None
    assert updated.plan_reason == "acquisition_failed:docling_cuda_oom"
    assert updated.plan_state == "needs_review"

def _programme(institution_id: int = 1) -> ProgrammeRecord:
    return ProgrammeRecord(
        id=institution_id,
        institution_id=institution_id,
        name="BSc MLT",
        credential_level="degree",
        recognition_status="unknown",
    )


def test_score_is_idempotent_and_never_persists_profile(tmp_path, registry) -> None:
    scores = MemoryScores()

    def persist(_database, result) -> None:
        scores.create(ScoreCreate(**result.to_score_create()))

    harness = Harness(tmp_path, registry)
    harness.scores = scores
    harness.programmes = MemoryProgrammes([_programme(1)])
    harness.orchestrator.scores = scores
    harness.orchestrator.programmes = harness.programmes

    first = harness.orchestrator.score(profile={"target_field": "mlt"}, persist=persist)
    assert first.counts["written"] == 1
    second = harness.orchestrator.score(profile={"target_field": "mlt"}, persist=persist)
    assert second.counts["written"] == 0
    assert second.counts["skipped"] == 1
    assert len(scores.rows) == 1
    rendered = json.dumps([row.model_dump(mode="json") for row in scores.rows])
    assert "target_field" not in rendered


def test_worker_one_tick_then_shutdown(tmp_path, registry) -> None:
    url = "https://college.ac.in/page.txt"
    source = FakeSource(id="fixture")
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={source.id: FakeAdapter([_ref(url, family="placement_report", entity_id=1)])},
        fetcher=FakeFetcher(
            {
                url: response(
                    url,
                    body=(EXTRACT_FIXTURES / "placement_report.txt").read_text(encoding="utf-8"),
                    headers={"content-type": "text/plain"},
                )
            }
        ),
        router=router,
    )

    summaries = harness.orchestrator.worker(
        limit=5, idle_seconds=0.0, max_ticks=1, sleep=lambda _seconds: None
    )

    assert [summary.stage for summary in summaries] == ["crawl", "ingest", "extract", "score"]
    ingest = summaries[1]
    assert ingest.counts.get("skipped") == 1
    assert ingest.metadata["reason"].startswith("ingest requires")
    heartbeat = json.loads(harness.ledger.heartbeat_path.read_text())
    assert heartbeat["state"] == "stopped"
    assert heartbeat["tick"] == 1
    assert heartbeat["process_id"] > 0
    assert len(list(harness.ledger.runs_dir.glob("*.json"))) == 4
