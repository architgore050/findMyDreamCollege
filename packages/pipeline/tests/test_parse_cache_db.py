"""W10 DB tests for the parse-cache lookup table (migration 0011).

Skipped loudly when DATABASE_URL is unset or Postgres is unreachable — the
same contract as the other repository tests (``conftest.pg_url``).
"""

from __future__ import annotations

import pytest

from fmd.acquire.parse_cache import ParseCache, parse_key, parse_policy_digest
from fmd.acquire.tiers import AcquirePolicy
from fmd.config import load_registry
from fmd.db.repositories import DocumentParsePut, DocumentParsesRepo
from fmd.store import SnapshotStore

pytestmark = pytest.mark.usefixtures("pg_database")

REPO_CONFIG = __import__("pathlib").Path(__file__).resolve().parents[3] / "config"


def _put(repo: DocumentParsesRepo, key: str, *, pages: int = 3) -> DocumentParsePut:
    put = DocumentParsePut(
        key_digest=key,
        snapshot_sha256="a" * 64,
        tier="T1",
        policy_digest=parse_policy_digest(load_registry(REPO_CONFIG, env={}).acquire),
        schema_version=1,
        object_sha256="b" * 64,
        pages=pages,
        parser="docling",
        parser_version="2.31.0",
    )
    repo.put(put)
    return put


def test_document_parses_roundtrip(pg_database) -> None:
    repo = DocumentParsesRepo(pg_database)
    key = parse_key("a" * 64, "T1", "p" * 64)
    put = _put(repo, key, pages=7)

    row = repo.get(key)
    assert row is not None
    assert row.snapshot_sha256 == "a" * 64
    assert row.tier == "T1"
    assert row.policy_digest == put.policy_digest
    assert row.schema_version == 1
    assert row.object_sha256 == "b" * 64
    assert row.pages == 7
    assert row.parser == "docling"
    assert row.parser_version == "2.31.0"
    assert row.created_at is not None
    assert repo.get("c" * 64) is None
    assert repo.count() >= 1


def test_document_parses_put_is_idempotent_on_key(pg_database) -> None:
    # The schema is shared across the file's tests (session fixture), so the
    # idempotency claim is relative: two puts on one key add exactly one row.
    repo = DocumentParsesRepo(pg_database)
    before = repo.count()
    key = parse_key("d" * 64, "T1", "e" * 64)
    _put(repo, key, pages=3)
    _put(repo, key, pages=5)

    row = repo.get(key)
    assert row is not None
    assert row.pages == 5
    assert repo.count() == before + 1


def test_migration_0011_is_applied_and_forward_only(pg_database) -> None:
    from fmd.db.migrations import applied_migrations

    applied = applied_migrations(pg_database)
    assert "0011_document_parses" in applied


def test_parse_cache_over_the_real_store(pg_database, tmp_path) -> None:
    """The full cache stack against Postgres: store → row → reconstruct."""
    registry = load_registry(REPO_CONFIG, env={})
    policy = AcquirePolicy.load(acquire=registry.acquire)
    store = SnapshotStore(tmp_path / "snapshots")
    repo = DocumentParsesRepo(pg_database)
    cache = ParseCache(repo, store)

    from test_w10_support import PDF_BYTES, CountingT0, CountingT1, report_doc

    from fmd.acquire.parse_cache import acquire_document

    digest = store.put(PDF_BYTES)
    t1 = CountingT1()
    doc, miss = acquire_document(
        PDF_BYTES,
        content_type="application/pdf",
        policy=policy,
        cache=cache,
        snapshot_sha256=digest,
        t0=CountingT0(),
        t1=t1,
    )
    assert doc.tier == "T1"
    assert miss.info["stored"] is True
    assert t1.calls == 1

    doc2, hit = acquire_document(
        PDF_BYTES,
        content_type="application/pdf",
        policy=policy,
        cache=cache,
        snapshot_sha256=digest,
        t0=CountingT0(),
        t1=t1,
    )
    assert t1.calls == 1
    assert hit.as_dict()["mode"] == "cache"
    assert doc2.text == report_doc().text
