"""Migration 0004 and inventory ingest against Postgres (skipped without DATABASE_URL).

No network, no model calls. The schema is created in a throwaway Postgres schema
so the test never mutates another run's data.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

import psycopg
import pytest

from fmd.db import (
    Database,
    DocumentsRepo,
    DocumentUpsert,
    SnapshotCreate,
    SnapshotsRepo,
    run_migrations,
)
from fmd.db.repositories import (
    AliasesRepo,
    EntitiesRepo,
    InstitutionDomainsRepo,
)
from fmd.orchestrate.ingest import InventoryStage
from fmd.orchestrate.ledger import RunSummary, utcnow
from fmd.seed import seed_database
from fmd.store import SnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO_ROOT / "db" / "migrations"
UGC_PAYLOAD = (
    Path(__file__).parent / "fixtures" / "ingest" / "ugc_university_details.json"
).read_text(encoding="utf-8")


@pytest.fixture
def schema_database(pg_url: str):
    schema = "test_role_" + uuid.uuid4().hex[:10]
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    database = Database(pg_url, search_path=schema, min_size=1, max_size=3)
    try:
        yield database
    finally:
        database.close()
        with psycopg.connect(pg_url, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def db(schema_database):
    run_migrations(schema_database)
    return schema_database


def test_0004_adds_roles_and_the_institution_domain_queue(schema_database, tmp_path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in (
        "0001_init.sql",
        "0002_er_audit_and_replay.sql",
        "0003_document_plans_and_pipeline_runs.sql",
    ):
        shutil.copy(MIGRATIONS / name, legacy / name)

    assert run_migrations(schema_database, legacy) == [
        "0001_init",
        "0002_er_audit_and_replay",
        "0003_document_plans_and_pipeline_runs",
    ]
    assert run_migrations(schema_database) == [
        "0004_document_roles_and_inventory_targets",
        "0005_claim_verification",
        "0006_programme_materialisation",
        "0007_dedupe_basis",
    ]
    assert run_migrations(schema_database) == []

    with schema_database.connection() as conn:
        columns = {
            f"{row['table_name']}.{row['column_name']}"
            for row in conn.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name IN ('documents', 'entities', 'institution_domains')"
            ).fetchall()
        }
        indexes = {
            row["indexname"]
            for row in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
            ).fetchall()
        }
        # Existing rows become evidence: the migration is safe over a live corpus.
        conn.execute(
            "INSERT INTO documents (url, domain) VALUES "
            "('https://legacy.ac.in/a.pdf', 'legacy.ac.in')"
        )
        role = conn.execute(
            "SELECT role FROM documents WHERE url = 'https://legacy.ac.in/a.pdf'"
        ).fetchone()["role"]

    assert "documents.role" in columns
    assert "entities.website_url" in columns
    assert "entities.institution_type" in columns
    assert "documents_role_idx" in indexes
    assert role == "evidence"

    with schema_database.connection() as conn:
        conn.execute(
            "UPDATE documents SET plan_state = 'not_applicable' "
            "WHERE url = 'https://legacy.ac.in/a.pdf'"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "UPDATE documents SET role = 'listing' WHERE url = 'https://legacy.ac.in/a.pdf'"
            )


def test_ingest_is_idempotent_and_list_extractable_is_evidence_only(db, registry, tmp_path) -> None:
    seed_database(db, registry)  # alias provenance points at a real sources row
    documents = DocumentsRepo(db)
    snapshots = SnapshotsRepo(db)
    store = SnapshotStore(tmp_path / "snapshots")

    payload = UGC_PAYLOAD.encode("utf-8")
    digest = store.put(payload)
    assert hashlib.sha256(payload).hexdigest() == digest
    inventory_url = "https://www.ugc.gov.in/universitydetails/Getuniversity_details?unitypeID=1"
    inventory = documents.upsert(
        DocumentUpsert(
            url=inventory_url,
            domain="ugc.gov.in",
            source_id="ugc_university_list",
            role="inventory",
        )
    )
    snapshots.create(
        SnapshotCreate(
            document_id=inventory.id,
            sha256=digest,
            bytes_path=str(store.path_for(digest)),
        )
    )
    navigation = documents.upsert(
        DocumentUpsert(
            url="https://alpha-med.example.ac.in/",
            domain="alpha-med.example.ac.in",
            role="navigation",
            plan_state="not_applicable",
        )
    )
    snapshots.create(
        SnapshotCreate(document_id=navigation.id, sha256="c" * 64, bytes_path="objects/c")
    )

    stage = InventoryStage(
        documents=documents,
        snapshots=snapshots,
        store=store,
        entities=EntitiesRepo(db),
        aliases=AliasesRepo(db),
        domains=InstitutionDomainsRepo(db),
        sources=[registry.sources.by_id("ugc_university_list")],
        registry=registry,
    )
    first = RunSummary(run_id="ingest-1", stage="ingest", started_at=utcnow())
    selected = stage.run(summary=first)

    assert first.counts["inventory_documents"] == 1
    assert first.counts["entities_created"] == 3
    assert first.counts["domains_discovered"] == 2
    assert len(selected) == 2  # config max_domains_per_run=25, only 2 exist

    entities = EntitiesRepo(db)
    alpha = entities.find_normalised("institution", "alpha medical sciences")
    assert alpha is not None
    assert alpha.state == "Delhi"
    assert alpha.institution_type == "Central"
    assert alpha.website_url == "http://www.alpha-med.example.ac.in"
    assert AliasesRepo(db).list_for_entity(alpha.id)  # acronym alias

    domains = InstitutionDomainsRepo(db)
    assert domains.counts_by_state() == {"queued": 2}
    assert {row.domain for row in domains.list_by_state("queued")} == {
        "www.alpha-med.example.ac.in",
        "beta-health.example.ac.in",
    }

    # A re-ingest matches by normalised name and never duplicates rows/aliases.
    alias_count = len(AliasesRepo(db).list_for_entity(alpha.id))
    second = RunSummary(run_id="ingest-2", stage="ingest", started_at=utcnow())
    stage.run(summary=second)
    assert second.counts.get("entities_created", 0) == 0
    assert second.counts["entities_matched"] == 3
    assert entities.find_normalised("institution", "alpha medical sciences").id == alpha.id
    assert len(AliasesRepo(db).list_for_entity(alpha.id)) == alias_count
    assert domains.counts_by_state() == {"queued": 2}

    # Extraction never sees inventory/navigation; only evidence is extractable.
    assert snapshots.list_extractable(None, limit=50) == []
    assert documents.count_unplanned() == {"not_applicable": 2}
    assert documents.counts_by_role() == {"inventory": 1, "navigation": 1}
