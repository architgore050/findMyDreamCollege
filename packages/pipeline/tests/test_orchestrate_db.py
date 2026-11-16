from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest
from test_orchestrate_support import FakeFetcher, FakeSource, response

from fmd.cli import main
from fmd.config import load_registry
from fmd.db import Database, run_migrations
from fmd.db.repositories import (
    ClaimsRepo,
    DocumentsRepo,
    EntitiesRepo,
    EntityCreate,
    PipelineRunsRepo,
    SnapshotsRepo,
    SourcesRepo,
)
from fmd.discover.base import DocumentRef
from fmd.extract.pipeline import build_router
from fmd.frontier.queue import Frontier
from fmd.orchestrate import JsonFileLedger, Orchestrator
from fmd.orchestrate.crawl import Crawler
from fmd.orchestrate.ledger import RunSummary
from fmd.seed import seed_database
from fmd.store import SnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
EXTRACT_FIXTURES = Path(__file__).parent / "fixtures" / "extract"
LLM_FIXTURES = EXTRACT_FIXTURES / "llm"
PLAN_FIXTURES = Path(__file__).parent / "fixtures" / "plan"
PLAN_LLM_FIXTURES = PLAN_FIXTURES / "llm"


@pytest.fixture
def fresh_database(pg_url: str):
    """A migrated, empty schema so seed -> crawl -> extract -> score starts clean."""
    schema = "test_orc_" + uuid.uuid4().hex[:10]
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    database = Database(pg_url, search_path=schema, min_size=1, max_size=3)
    run_migrations(database)
    try:
        yield database
    finally:
        database.close()
        with psycopg.connect(pg_url, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


class FixtureAdapter:
    """Deterministic crawl source: one placement-report document."""

    def __init__(self, entity_id: int) -> None:
        self.entity_id = entity_id

    def fetch(self):
        return [
            DocumentRef(
                url="https://fixture.ac.in/placement-report.txt",
                source_id="fixture_source",
                metadata={"family": "placement_report", "entity_id": self.entity_id},
            )
        ]


class UnplannedAdapter:
    """A real-world ref: it knows the URL and nothing about the target."""

    def __init__(self, url: str, source_id: str = "fixture_source") -> None:
        self.url = url
        self.source_id = source_id

    def fetch(self):
        return [DocumentRef(url=self.url, source_id=self.source_id)]


def test_seed_crawl_extract_score_on_empty_database(fresh_database, tmp_path) -> None:
    # pg_url skips cleanly when DATABASE_URL is unset or unreachable.
    registry = load_registry(CONFIG_DIR, env={})
    seed_database(fresh_database, registry)
    assert len(SourcesRepo(fresh_database).list()) == len(registry.sources.sources)

    entity = EntitiesRepo(fresh_database).create(
        EntityCreate(kind="institution", canonical_name="Fixture Institute", country="IN")
    )
    text = (EXTRACT_FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    url = "https://fixture.ac.in/placement-report.txt"
    source = FakeSource(id="fixture_source", adapter="institution_site", kind="institution")
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    orchestrator = Orchestrator(
        registry=registry,
        database=fresh_database,
        store=SnapshotStore(tmp_path / "snapshots"),
        frontier=Frontier.from_database(fresh_database),
        sources=[source],
        adapter_factory=lambda _source: FixtureAdapter(entity.id),
        fetcher=FakeFetcher(
            {url: response(url, body=text, headers={"content-type": "text/plain"})}
        ),
        router=router,
        ledger=JsonFileLedger(tmp_path / "ledger"),
        run_id_factory=lambda: f"it-{uuid.uuid4().hex[:8]}",
    )

    crawl = orchestrator.crawl(limit=10)
    assert crawl.counts["stored"] == 1
    assert crawl.failed is False

    extracted = orchestrator.extract(limit=10)
    assert extracted.counts["extracted"] == 1
    assert extracted.counts["claims"] > 0
    assert extracted.failed is False

    page = ClaimsRepo(fresh_database).evidence_page(entity.id)
    assert page.total > 0
    assert page.items[0].url == url
    assert len(page.items[0].sha256) == 64
    assert page.items[0].fetched_at is not None

    scored = orchestrator.score()
    assert scored.counts["programmes"] == 0  # no programmes seeded: nothing to do
    assert scored.failed is False

    # Every stage wrote a run record.
    assert len(list((tmp_path / "ledger" / "runs").glob("*.json"))) == 3


def test_crawl_extract_end_to_end_without_ref_metadata(fresh_database, tmp_path) -> None:
    """Real documents have no family/entity metadata; the planner must supply both."""
    registry = load_registry(CONFIG_DIR, env={})
    seed_database(fresh_database, registry)

    entity = EntitiesRepo(fresh_database).create(
        EntityCreate(
            kind="institution",
            canonical_name="Fixture Institute of Medical Sciences",
            country="IN",
        )
    )
    text = (PLAN_FIXTURES / "e2e_placement_report.txt").read_text(encoding="utf-8")
    url = "https://fixture.ac.in/e2e-placement-report.txt"
    source = FakeSource(
        id="fixture_source",
        adapter="institution_site",
        kind="institution",
        params={"extraction_family": "placement_report"},
    )
    router = build_router(registry.models, mock=True, fixtures_dir=PLAN_LLM_FIXTURES)
    orchestrator = Orchestrator(
        registry=registry,
        database=fresh_database,
        store=SnapshotStore(tmp_path / "snapshots"),
        frontier=Frontier.from_database(fresh_database),
        sources=[source],
        adapter_factory=lambda _source: UnplannedAdapter(url),
        fetcher=FakeFetcher(
            {url: response(url, body=text, headers={"content-type": "text/plain"})}
        ),
        router=router,
        run_id_factory=lambda: f"e2e-{uuid.uuid4().hex[:8]}",
    )
    # No ledger injection: the default is the SQL run ledger.

    crawl = orchestrator.crawl(limit=5)
    assert crawl.counts["stored"] == 1

    document = DocumentsRepo(fresh_database).get_by_url(url)
    assert document is not None
    assert document.source_id == "fixture_source"
    assert document.extraction_family == "placement_report"
    assert document.entity_id == entity.id
    assert document.plan_state == "linked"

    extracted = orchestrator.extract(limit=5)
    assert extracted.counts["extracted"] == 1
    assert extracted.counts["claims"] >= 1

    page = ClaimsRepo(fresh_database).evidence_page(entity.id)
    assert page.total >= 1

    runs = PipelineRunsRepo(fresh_database).list(limit=10)
    assert {row.stage for row in runs} == {"crawl", "extract"}
    assert all(row.status == "succeeded" for row in runs)
    crawl_run = PipelineRunsRepo(fresh_database).get(crawl.run_id)
    assert crawl_run is not None
    assert crawl_run.counts["stored"] == 1
    assert crawl_run.finished_at is not None
    assert PipelineRunsRepo(fresh_database).list(stage="extract")[0].counts["claims"] >= 1


def test_post_replay_survives_a_frontier_roundtrip(fresh_database, tmp_path) -> None:
    """The persisted POST shape is replayable after a DB roundtrip, not just in RAM."""
    url = "https://nabh.co/wp-admin/admin-ajax.php?action=get_hospitals&page=2"
    body = {"action": "get_hospitals", "page": 2}
    headers = {"Accept": "application/json"}
    ref = DocumentRef(
        url=url,
        source_id="fixture_source",
        metadata={
            "request_method": "POST",
            "request_body": body,
            "request_headers": headers,
        },
    )
    frontier = Frontier.from_database(fresh_database)
    frontier.enqueue_ref(ref)

    # A later discovery ref carrying only a URL must not erase the POST shape.
    frontier.enqueue_ref(DocumentRef(url=url, source_id="fixture_source"))
    stored = frontier.repo.get(url)
    assert stored is not None
    assert stored.http_method == "POST"
    assert stored.http_body == body

    documents = DocumentsRepo(fresh_database)
    fetcher = FakeFetcher(
        {
            url: lambda request: response(
                request.url,
                body='{"rows":[{"id":2}]}',
                headers={"content-type": "application/json"},
            )
        }
    )
    crawler = Crawler(
        frontier=frontier,
        fetcher=fetcher,
        store=SnapshotStore(tmp_path / "snapshots"),
        documents=documents,
        snapshots=SnapshotsRepo(fresh_database),
        adapter_factory=lambda _source: None,
    )
    summary = RunSummary(run_id="db-post", stage="crawl", started_at=datetime.now(UTC))

    crawler.drain(summary=summary)

    assert summary.counts["stored"] == 1
    request = fetcher.requests[0]
    assert request.method == "POST"
    assert request.body == body
    document = documents.get_by_url(url)
    assert document is not None
    assert document.http_method == "POST"
    assert document.http_body == body
    assert document.request_headers == headers


def test_run_ledger_records_a_failed_run_as_failed(fresh_database, tmp_path) -> None:
    registry = load_registry(CONFIG_DIR, env={})
    source = FakeSource(id="broken")
    orchestrator = Orchestrator(
        registry=registry,
        database=fresh_database,
        store=SnapshotStore(tmp_path / "snapshots"),
        frontier=Frontier.from_database(fresh_database),
        sources=[source],
        adapter_factory=lambda _source: UnplannedAdapter("https://fixture.ac.in/x.txt"),
        fetcher=FakeFetcher({}),
        router=build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES),
        run_id_factory=lambda: f"failed-{uuid.uuid4().hex[:8]}",
    )

    summary = orchestrator.crawl(limit=5)

    row = PipelineRunsRepo(fresh_database).get(summary.run_id)
    assert row is not None
    assert row.status == "failed"
    assert row.errors


def _dsn_with_schema(database: Database) -> str:
    option = quote(f"-c search_path={database.search_path}")
    separator = "&" if "?" in database.url else "?"
    return f"{database.url}{separator}options={option}"


def test_cli_extract_and_score_exit_zero_when_nothing_to_do(
    fresh_database, capsys, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("FMD_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    base = ["--database-url", _dsn_with_schema(fresh_database), "--config-dir", str(CONFIG_DIR)]

    assert main([*base, "extract"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["stage"] == "extract"
    assert payload["ok"] is True

    assert main([*base, "score"]) == 0
    captured = capsys.readouterr().out
    assert "fmd score: wrote 0" in captured
    payload = json.loads(captured.strip().splitlines()[-1])
    assert payload["stage"] == "score"
    assert payload["ok"] is True
