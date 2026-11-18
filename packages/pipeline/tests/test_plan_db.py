"""Planner persistence and migration-0003 tests (Postgres only).

Skipped cleanly when ``DATABASE_URL`` is unset or unreachable (``pg_url``).
No network, no model calls.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import psycopg
import pytest

from fmd.db import (
    AliasCreate,
    AliasesRepo,
    Database,
    DocumentsRepo,
    DocumentUpsert,
    EntitiesRepo,
    EntityCreate,
    SnapshotCreate,
    SnapshotsRepo,
    applied_migrations,
    discover_migrations,
    run_migrations,
)
from fmd.er import ERConfig
from fmd.plan import DocumentPlanner, persist_plan

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO_ROOT / "db" / "migrations"


def _config() -> ERConfig:
    return ERConfig.from_mapping(
        {"blocking": {"backend": "deterministic"}, "similarity": {"use_embeddings": False}}
    )


@pytest.fixture
def schema_database(pg_url: str):
    """An empty schema; the caller decides which migrations to apply."""
    schema = "test_plan_" + uuid.uuid4().hex[:10]
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


def _planner(database) -> DocumentPlanner:
    return DocumentPlanner(
        entities=EntitiesRepo(database),
        aliases=AliasesRepo(database),
        er_config=_config(),
    )


def _entity(database, name: str, state: str = "Delhi") -> int:
    return (
        EntitiesRepo(database)
        .create(EntityCreate(kind="institution", canonical_name=name, country="IN", state=state))
        .id
    )


def _document(database, url: str, **fields) -> int:
    return (
        DocumentsRepo(database).upsert(DocumentUpsert(url=url, domain="example.ac.in", **fields)).id
    )


# ------------------------------------------------------------------ migration 0003


def test_0003_applies_over_0001_and_0002_and_is_idempotent(schema_database, tmp_path) -> None:
    legacy = tmp_path / "legacy_migrations"
    legacy.mkdir()
    for name in ("0001_init.sql", "0002_er_audit_and_replay.sql"):
        shutil.copy(MIGRATIONS / name, legacy / name)

    assert run_migrations(schema_database, legacy) == [
        "0001_init",
        "0002_er_audit_and_replay",
    ]
    # The applied set is exactly the files on disk, checksum for checksum.
    assert applied_migrations(schema_database) == {
        migration.version: migration.checksum for migration in discover_migrations(legacy)
    }

    assert run_migrations(schema_database) == [
        "0003_document_plans_and_pipeline_runs",
        "0004_document_roles_and_inventory_targets",
        "0005_claim_verification",
        "0006_programme_materialisation",
        "0007_dedupe_basis",
    ]
    assert run_migrations(schema_database) == []
    assert applied_migrations(schema_database) == {
        migration.version: migration.checksum for migration in discover_migrations()
    }

    with schema_database.connection() as conn:
        columns = {
            f"{row['table_name']}.{row['column_name']}"
            for row in conn.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name IN ('documents', 'crawl_frontier', 'pipeline_runs')"
            ).fetchall()
        }
        indexes = {
            row["indexname"]
            for row in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
            ).fetchall()
        }
        run_columns = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'pipeline_runs'"
            ).fetchall()
        }
    assert {
        "documents.source_id",
        "documents.extraction_family",
        "documents.entity_id",
        "documents.plan_state",
        "crawl_frontier.source_id",
    } <= columns
    assert {
        "documents_source_idx",
        "documents_family_idx",
        "documents_entity_idx",
        "crawl_frontier_source_idx",
    } <= indexes
    assert {
        "run_id",
        "stage",
        "started_at",
        "finished_at",
        "counts_json",
        "errors_json",
        "status",
    } == run_columns


# ------------------------------------------------------------------ planner persistence


def test_single_confident_candidate_is_persisted_on_the_document(db) -> None:
    name = f"Fixture Institute of Health Sciences {uuid.uuid4().hex[:6]}"
    entity_id = _entity(db, name)
    document_id = _document(db, f"https://example.ac.in/{uuid.uuid4().hex}.pdf")
    text = f"{name}\nPlacement Report 2024-25\nStudents placed: 96\n"

    document = DocumentsRepo(db).get(document_id)
    plan = _planner(db).plan(document, family_hint="placement_report", text_sample=text)
    stored = persist_plan(DocumentsRepo(db), document, plan)

    assert stored is not None
    assert stored.entity_id == entity_id
    assert stored.extraction_family == "placement_report"
    assert stored.plan_state == "linked"
    assert stored.plan_confidence is not None and stored.plan_confidence > 0
    assert DocumentsRepo(db).list_needs_review() == []


def test_ambiguous_entity_leaves_null_and_marks_needs_review(db) -> None:
    name = f"Twin Institute of Health Sciences {uuid.uuid4().hex[:6]}"
    first = _entity(db, name)
    second = _entity(db, name)
    document_id = _document(db, f"https://example.ac.in/{uuid.uuid4().hex}.pdf")
    text = f"{name}\nPlacement Report 2024-25\n"

    document = DocumentsRepo(db).get(document_id)
    plan = _planner(db).plan(document, family_hint="placement_report", text_sample=text)
    stored = persist_plan(DocumentsRepo(db), document, plan)

    assert plan.abstain_reason == "entity_ambiguous"
    assert stored.entity_id is None
    assert stored.plan_state == "needs_review"
    assert stored.plan_reason == "entity_ambiguous"
    queued = DocumentsRepo(db).list_needs_review()
    assert [row.id for row in queued] == [document_id]
    assert {first, second} == {
        item.entity_id for item in plan.entity_candidates if item.entity_id is not None
    }


def test_manual_alias_resolves_even_when_another_name_scores_higher(db) -> None:
    suffix = uuid.uuid4().hex[:6]
    exact_name = f"Exact Institute {suffix}"
    manual_target = _entity(db, f"Renamed Health University {suffix}")
    _entity(db, exact_name)
    AliasesRepo(db).create(
        AliasCreate(entity_id=manual_target, alias=exact_name, alias_type="manual")
    )
    document_id = _document(db, f"https://example.ac.in/{suffix}.pdf")
    text = f"{exact_name}\nPlacement Report 2024-25\n"

    document = DocumentsRepo(db).get(document_id)
    plan = _planner(db).plan(document, family_hint="placement_report", text_sample=text)

    assert plan.entity_id == manual_target
    assert plan.state == "linked"


def test_extract_queue_only_returns_planned_documents(db) -> None:
    name = f"Queue Institute {uuid.uuid4().hex[:6]}"
    entity_id = _entity(db, name)
    documents = DocumentsRepo(db)
    snapshots = SnapshotsRepo(db)

    linked = documents.upsert(
        DocumentUpsert(
            url=f"https://example.ac.in/linked-{uuid.uuid4().hex}.pdf",
            domain="example.ac.in",
            extraction_family="placement_report",
            entity_id=entity_id,
            plan_state="linked",
        )
    )
    review = documents.upsert(
        DocumentUpsert(
            url=f"https://example.ac.in/review-{uuid.uuid4().hex}.pdf",
            domain="example.ac.in",
            extraction_family="placement_report",
            plan_state="needs_review",
        )
    )
    for index, document in enumerate((linked, review)):
        snapshots.create(
            SnapshotCreate(
                document_id=document.id,
                sha256=("a" if index == 0 else "b") * 64,
                bytes_path=f"objects/{index}",
            )
        )

    pending = snapshots.list_extractable("1.0.0", limit=10)

    assert [row.document_id for row in pending] == [linked.id]
