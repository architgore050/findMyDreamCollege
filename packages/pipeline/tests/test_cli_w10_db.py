"""W10 CLI commands against a real (throw-away) schema: ``fmd acquire`` and
``fmd extract --batched/--from-cache/--no-acquire`` exit codes and summaries.

Skipped loudly without DATABASE_URL (same contract as test_cli.py).
"""

from __future__ import annotations

import json
import uuid

import psycopg
import pytest
from test_cli import CONFIG_DIR, _dsn_with_schema  # noqa: F401

from fmd.cli import main
from fmd.db import Database, run_migrations


@pytest.fixture
def cli_empty_database(pg_url):
    """A migrated, row-free schema (same contract as test_cli.empty_database).

    Defined here rather than imported: pytest registers an imported fixture
    under its source name, so the alias would never resolve — and ruff
    (rightly) flags parameter shadowing of an imported name.
    """
    schema = "test_cli_w10_" + uuid.uuid4().hex[:10]
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    database = Database(pg_url, search_path=schema, min_size=1, max_size=2)
    run_migrations(database)
    try:
        yield database
    finally:
        database.close()
        with psycopg.connect(pg_url, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def snapshot_root(tmp_path, monkeypatch):
    """A throw-away snapshot root (FMD_SNAPSHOT_DIR) for the run under test."""
    root = tmp_path / "snapshots"
    monkeypatch.setenv("FMD_SNAPSHOT_DIR", str(root))
    return root


def _snapshot_with_document(database, snapshot_root) -> str:
    """One planned evidence snapshot with stored bytes; returns its sha256."""
    from test_w10_support import PDF_BYTES

    from fmd.db.repositories import (
        DocumentsRepo,
        DocumentUpsert,
        EntitiesRepo,
        EntityCreate,
        SnapshotCreate,
        SnapshotsRepo,
    )
    from fmd.store import SnapshotStore

    store = SnapshotStore(snapshot_root)
    digest = store.put(PDF_BYTES)
    entity = EntitiesRepo(database).create(
        EntityCreate(kind="institution", canonical_name="CLI Test Institute", country="IN")
    )
    document = DocumentsRepo(database).upsert(
        DocumentUpsert(
            url="https://college.ac.in/report.pdf",
            domain="college.ac.in",
            kind="report",
            source_id="fixture",
            extraction_family="placement_report",
            entity_id=entity.id,
            plan_state="linked",
            role="evidence",
        )
    )
    SnapshotsRepo(database).create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=f"objects/{digest[:2]}/{digest}",
            content_type="application/pdf",
        )
    )
    return digest


def test_cli_acquire_dry_run_on_empty_schema(
    cli_empty_database, capsys, snapshot_root
) -> None:
    argv = [
        "--database-url",
        _dsn_with_schema(cli_empty_database),
        "--config-dir",
        str(CONFIG_DIR),
        "acquire",
        "--dry-run",
        "--limit",
        "5",
    ]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage"] == "acquire"
    assert summary["dry_run"] is True
    assert summary["ok"] is True
    assert summary["counts"]["selected"] == 0


def test_cli_extract_dry_run_on_empty_schema(cli_empty_database, capsys, snapshot_root) -> None:
    argv = [
        "--database-url",
        _dsn_with_schema(cli_empty_database),
        "--config-dir",
        str(CONFIG_DIR),
        "extract",
        "--dry-run",
        "--limit",
        "5",
    ]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage"] == "extract"
    assert summary["dry_run"] is True
    assert summary["ok"] is True


def test_cli_extract_batched_dry_run_prints_cycle_plan(
    cli_empty_database, capsys, monkeypatch, snapshot_root
) -> None:
    """The batched dry run plans cycles with an unreachable model URL and
    writes nothing (exit 0)."""
    monkeypatch.setenv("FMD_LOCAL_LLM_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("FMD_SERVER_CONTROL", "off")  # no server control in CI
    argv = [
        "--database-url",
        _dsn_with_schema(cli_empty_database),
        "--config-dir",
        str(CONFIG_DIR),
        "extract",
        "--batched",
        "--dry-run",
        "--limit",
        "5",
        "--batch-pages",
        "100",
    ]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage"] == "extract_batched"
    assert summary["dry_run"] is True
    assert summary["ok"] is True
    assert summary["metadata"]["batch_pages"] == 100
    assert summary["metadata"]["server_control"] == "degraded"


def test_cli_extract_from_cache_with_no_cache_is_non_zero(
    cli_empty_database, capsys, snapshot_root
) -> None:
    """--from-cache with a queued snapshot but nothing cached: nothing was
    extractable ⇒ exit 1, recorded as missing_parse_cache — never a silent
    clean pass."""
    _snapshot_with_document(cli_empty_database, snapshot_root)
    argv = [
        "--database-url",
        _dsn_with_schema(cli_empty_database),
        "--config-dir",
        str(CONFIG_DIR),
        "extract",
        "--from-cache",
        "--limit",
        "5",
    ]
    assert main(argv) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["metadata"]["mode"] == "from_cache"
