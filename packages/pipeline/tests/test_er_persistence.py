"""ER persistence: audit rows, reversible merges, manual authority, claims hash.

Runs only against a live Postgres (``DATABASE_URL``); the fixture skips clearly
otherwise. No network, no model calls.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

import pytest

from fmd.db import (
    AliasCreate,
    AliasesRepo,
    ClaimCreate,
    ClaimsRepo,
    DocumentsRepo,
    DocumentUpsert,
    EntitiesRepo,
    EntityCreate,
    ExtractionCreate,
    ExtractionsRepo,
    SnapshotCreate,
    SnapshotsRepo,
)
from fmd.db.repositories import ERAuditCreate
from fmd.er.config import ERConfig
from fmd.er.service import resolve_with_audit
from fmd.er.types import Alias, AuditDecision, EntityCandidate, Mention

pytestmark = pytest.mark.usefixtures("pg_database")


def _config() -> ERConfig:
    return ERConfig.from_mapping(
        {"blocking": {"backend": "deterministic"}, "similarity": {"use_embeddings": False}}
    )


def _suffix() -> str:
    return uuid.uuid4().hex[:10]


def _entity(database, name: str, state: str = "Delhi") -> int:
    return (
        EntitiesRepo(database)
        .create(EntityCreate(kind="institution", canonical_name=name, country="IN", state=state))
        .id
    )


def test_schema_is_exactly_the_migration_files(pg_database) -> None:
    from fmd.db import applied_migrations, discover_migrations

    applied = applied_migrations(pg_database)
    on_disk = {migration.version: migration.checksum for migration in discover_migrations()}
    # Applied set == files on disk, checksums included. A migration *count* would
    # make every new forward-only file a test failure (and has already caused a
    # column to be skipped).
    assert applied == on_disk

    with pg_database.connection() as conn:
        columns = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'er_audit'"
            ).fetchall()
        }
    assert {
        "subject_entity_id",
        "candidate_entity_id",
        "band",
        "score",
        "features_json",
        "thresholds_version",
        "model",
        "model_verdict",
        "model_reason",
        "decided_by",
        "claims_hash",
        "created_at",
    } <= columns

    with pg_database.connection() as conn:
        alias_columns = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'entity_aliases'"
            ).fetchall()
        }
        entity_columns = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'entities'"
            ).fetchall()
        }
    assert {
        "span_start",
        "span_end",
        "quote",
        "source_document_id",
        "source_extraction_id",
        "confirmed_by",
        "confirmed_at",
    } <= alias_columns
    assert {
        "normalised_name",
        "blocking_key",
        "merged_into_entity_id",
        "merged_at",
    } <= entity_columns


def test_merge_then_unmerge_is_reversible_and_appends_audit_rows(pg_database) -> None:
    suffix = _suffix()
    entities = EntitiesRepo(pg_database)
    survivor = _entity(pg_database, f"Survivor Institute {suffix}")
    duplicate = _entity(pg_database, f"Duplicate Institute {suffix}")

    merged, audit = entities.merge(
        duplicate,
        survivor,
        score=0.93,
        features={"token_jaccard": 0.9, "char_ratio": 0.96},
        thresholds_version="er-v1",
        decided_by="auto",
    )
    assert merged.merged_into_entity_id == survivor
    assert merged.merged_at is not None
    assert audit.band == "accept"
    assert audit.features["token_jaccard"] == 0.9
    assert audit.decided_by == "auto"

    pair = entities.audit_for_pair(duplicate, survivor)
    assert [row.id for row in pair] == [audit.id]
    assert entities.audit_recent(limit=50, since=datetime.now(UTC).replace(year=2000))

    restored, undo = entities.unmerge(duplicate)
    assert restored.merged_into_entity_id is None
    assert restored.merged_at is None
    assert restored.status == "active"
    assert undo.band == "reject"
    assert undo.features["operation"] == "unmerge"

    history = entities.audit_for_pair(duplicate, survivor)
    assert [row.band for row in history] == ["reject", "accept"]

    # Unmerging an already-unmerged entity is an error, never a silent no-op.
    with pytest.raises(ValueError):
        entities.unmerge(duplicate)


def test_audit_row_records_claims_hash_and_threshold_version(pg_database) -> None:
    suffix = _suffix()
    entities = EntitiesRepo(pg_database)
    subject = _entity(pg_database, f"Hashed Institute {suffix}")
    target = _entity(pg_database, f"Hashed University {suffix}")

    url = f"https://example.ac.in/{suffix}"
    document = DocumentsRepo(pg_database).upsert(
        DocumentUpsert(url=url, domain="example.ac.in", kind="page")
    )
    data = b"evidence"
    digest = hashlib.sha256(data).hexdigest()
    snapshot = SnapshotsRepo(pg_database).create(
        SnapshotCreate(document_id=document.id, sha256=digest, bytes_path=f"o/{digest}")
    )
    text = "Affiliated to Hashed University."
    extraction = ExtractionsRepo(pg_database).create(
        ExtractionCreate(snapshot_id=snapshot.id, tool="test", text=text)
    )
    ClaimsRepo(pg_database).create(
        ClaimCreate(
            extraction_id=extraction.id,
            entity_id=subject,
            predicate="affiliation",
            value={"name": "Hashed University"},
            quote=text,
            char_start=0,
            char_end=len(text),
        )
    )
    claims_hash = ClaimsRepo(pg_database).claims_hash(subject)
    assert len(claims_hash) == 64

    _, audit = entities.merge(
        subject,
        target,
        score=0.88,
        thresholds_version="er-v1",
        claims_hash=claims_hash,
    )
    reloaded = entities.audit_for_pair(subject, target)[0]
    assert reloaded.claims_hash == claims_hash
    assert reloaded.thresholds_version == "er-v1"


def test_manual_alias_blocks_auto_merge_and_is_persisted(pg_database) -> None:
    suffix = _suffix()
    entities = EntitiesRepo(pg_database)
    aliases = AliasesRepo(pg_database)
    entity_id = _entity(pg_database, f"Manual Lock University {suffix}")
    aliases.create(
        AliasCreate(
            entity_id=entity_id,
            alias=f"Manual Lock Vishwavidyalaya {suffix}",
            alias_type="manual",
            confirmed_by="reviewer@example.org",
            confirmed_at=datetime.now(UTC),
        )
    )
    stored_aliases = aliases.list_for_entity(entity_id)
    candidate = EntityCandidate(
        entity_id=entity_id,
        canonical_name=f"Manual Lock University {suffix}",
        state="Delhi",
        aliases=[
            Alias(alias=row.alias, alias_type=row.alias_type, confidence=row.confidence or 1.0)
            for row in stored_aliases
        ],
    )

    recorded: list[AuditDecision] = []

    resume_repo = EntitiesRepo(pg_database)

    def sink(decision: AuditDecision) -> None:
        recorded.append(decision)
        resume_repo.record_audit(ERAuditCreate(**decision.model_dump()))

    subject_id = _entity(pg_database, f"Provisional {suffix}")
    mention = Mention(text=f"Manual Lock University {suffix}", state="Delhi", entity_id=subject_id)

    batch = resolve_with_audit(
        [mention],
        [candidate],
        config=_config(),
        audit_sink=sink,
        claims_hash_by_entity={subject_id: "c" * 64},
    )
    resolution = batch.resolutions[0]
    # A human-confirmed canonical name is never auto-merged on an exact match.
    assert resolution.chosen is None
    assert resolution.band == "adjudicate"
    assert "manual_canonical_lock" in resolution.audit.notes

    assert len(recorded) == 1
    row = entities.audit_for_pair(subject_id, entity_id)[0]
    assert row.band == "adjudicate"
    assert row.decided_by == "auto"
    assert row.claims_hash == "c" * 64
    assert row.features["token_jaccard"] == 1.0


def test_manual_alias_outranks_a_higher_scoring_candidate(pg_database) -> None:
    suffix = _suffix()
    subject_id = _entity(pg_database, f"Provisional Two {suffix}")
    exact_id = _entity(pg_database, f"Manual Alias University {suffix}")
    manual_id = _entity(pg_database, f"Something Else College {suffix}")

    exact = EntityCandidate(
        entity_id=exact_id,
        canonical_name=f"Manual Alias University {suffix}",
        state="Delhi",
    )
    manual_target = EntityCandidate(
        entity_id=manual_id,
        canonical_name=f"Something Else College {suffix}",
        state="Delhi",
        aliases=[Alias(alias=f"Manual Alias University {suffix}", alias_type="manual")],
    )

    resolutions = resolve_with_audit(
        [Mention(text=f"Manual Alias University {suffix}", state="Delhi", entity_id=subject_id)],
        [exact, manual_target],
        config=_config(),
    ).resolutions
    resolution = resolutions[0]
    assert resolution.chosen is not None
    assert resolution.chosen.entity_id == manual_id
    assert resolution.audit.decided_by == "manual"
    assert "manual_alias_authoritative" in resolution.audit.notes


def test_record_audit_rejects_self_pairs_and_bad_scores(pg_database) -> None:
    entity = _entity(pg_database, f"Audit Guard {_suffix()}")
    repo = EntitiesRepo(pg_database)
    with pytest.raises(ValueError):
        repo.record_audit(
            ERAuditCreate(
                subject_entity_id=entity,
                candidate_entity_id=entity,
                band="accept",
                score=1.0,
                thresholds_version="er-v1",
                decided_by="auto",
            )
        )
    other = _entity(pg_database, f"Audit Guard Other {_suffix()}")
    with pytest.raises(ValueError):
        repo.record_audit(
            ERAuditCreate(
                subject_entity_id=entity,
                candidate_entity_id=other,
                band="accept",
                score=1.5,
                thresholds_version="er-v1",
                decided_by="auto",
            )
        )
