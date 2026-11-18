"""Programme repository + materialisation query against Postgres.

Skipped cleanly without DATABASE_URL (see ``pg_database`` in conftest).
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from fmd.db import (
    ClaimCreate,
    ClaimsRepo,
    DocumentsRepo,
    DocumentUpsert,
    EntitiesRepo,
    EntityCreate,
    ExtractionCreate,
    ExtractionsRepo,
    ProgrammeCreate,
    ProgrammesRepo,
    SnapshotCreate,
    SnapshotsRepo,
)
from fmd.db.repositories import ClaimVerification
from fmd.orchestrate.ingest import ProgrammeStage
from fmd.orchestrate.ledger import RunSummary, utcnow

pytestmark = pytest.mark.usefixtures("pg_database")


def _suffix() -> str:
    return uuid.uuid4().hex[:8]


def _evidence(database, suffix: str, *, role: str = "evidence") -> tuple[int, int]:
    document = DocumentsRepo(database).upsert(
        DocumentUpsert(
            url=f"https://example.ac.in/{suffix}.pdf",
            domain="example.ac.in",
            kind="curriculum",
            role=role,
        )
    )
    text = f"B.Sc. MLT is a four-year programme. {suffix}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    snapshot = SnapshotsRepo(database).create(
        SnapshotCreate(document_id=document.id, sha256=digest, bytes_path=f"o/{digest}")
    )
    extraction = ExtractionsRepo(database).create(
        ExtractionCreate(snapshot_id=snapshot.id, tool="test", text=text)
    )
    return extraction.id, len(text)


def _claim(
    database,
    entity_id: int,
    extraction_id: int,
    text_len: int,
    predicate: str,
    value,
    *,
    unit: str | None = None,
    status: str = "supported",
):
    claim = ClaimsRepo(database).create(
        ClaimCreate(
            extraction_id=extraction_id,
            entity_id=entity_id,
            predicate=predicate,
            value=value,
            unit=unit,
            quote="B.Sc. MLT",
            char_start=0,
            char_end=min(9, text_len),
            confidence=0.9,
        )
    )
    if status != "unverified":
        ClaimsRepo(database).record_verification(
            claim.id,
            ClaimVerification(status=status, verified_by="test"),
        )
    return claim


def test_upsert_is_idempotent_and_replaces_materialised_fields(pg_database) -> None:
    institution = EntitiesRepo(pg_database).create(
        EntityCreate(kind="institution", canonical_name=f"One {_suffix()}", country="IN")
    )
    programmes = ProgrammesRepo(pg_database)
    first, created = programmes.upsert(
        ProgrammeCreate(
            institution_id=institution.id,
            name="B.Sc. Medical Laboratory Technology",
            credential_level="degree",
            normalised_name="bsc mlt",
            seats=40,
            field_sources={"curriculum.programme_name": "1", "seats.intake": "2"},
        )
    )
    assert created is True

    second, created_again = programmes.upsert(
        ProgrammeCreate(
            institution_id=institution.id,
            name="B.Sc. Medical Laboratory Technology",
            credential_level="degree",
            normalised_name="bsc mlt",
            field_sources={"curriculum.programme_name": "1"},
            needs_review=True,
            review_reasons=["credential_level_defaulted"],
        )
    )
    assert created_again is False
    assert second.id == first.id
    assert second.seats is None  # vanished claim must not linger
    assert second.needs_review is True
    assert second.field_sources == {"curriculum.programme_name": "1"}
    assert len(programmes.list_for_institution(institution.id)) == 1


def test_materialisation_claims_only_evidence_and_supported(pg_database) -> None:
    entities = EntitiesRepo(pg_database)
    suffix = _suffix()
    institution = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Two {suffix}", country="IN")
    )
    evidence_extraction, evidence_len = _evidence(pg_database, f"ev-{suffix}")
    nav_extraction, nav_len = _evidence(pg_database, f"nav-{suffix}", role="navigation")

    kept = _claim(
        pg_database,
        institution.id,
        evidence_extraction,
        evidence_len,
        "curriculum.programme_name",
        "B.Sc. MLT",
    )
    _claim(
        pg_database,
        institution.id,
        evidence_extraction,
        evidence_len,
        "seats.intake",
        10,
        unit="count",
        status="rejected",
    )
    _claim(
        pg_database,
        institution.id,
        nav_extraction,
        nav_len,
        "seats.programme_name",
        "B.Sc. MLT",
    )

    rows = ProgrammesRepo(pg_database).materialisation_claims()
    ids = {row["id"] for row in rows}
    assert kept.id in ids
    assert all(row["document_role"] == "evidence" for row in rows)
    assert all(row["verification_status"] == "supported" for row in rows)


def test_programme_stage_end_to_end_against_postgres(pg_database) -> None:
    entities = EntitiesRepo(pg_database)
    suffix = _suffix()
    institution = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Three {suffix}", country="IN")
    )
    extraction_id, text_len = _evidence(pg_database, f"stage-{suffix}")
    _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "curriculum.programme_name",
        "Bachelor of Medical Laboratory Sciences(B.MLS)",
    )
    _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "curriculum.duration_months",
        48,
        unit="months",
    )
    _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.intake",
        50,
        unit="count",
    )

    stage = ProgrammeStage.from_database(pg_database, registry=None)
    summary = RunSummary(run_id="test", stage="ingest", started_at=utcnow())
    written = stage.run(summary=summary)

    mine = [row for row in written if row.institution_id == institution.id]
    assert len(mine) == 1
    row = mine[0]
    assert row.normalised_name == "bsc mls"
    assert row.duration_months == 48
    assert row.seats == 50
    assert set(row.field_sources) == {
        "curriculum.programme_name",
        "curriculum.duration_months",
        "seats.intake",
    }

    again = RunSummary(run_id="test2", stage="ingest", started_at=utcnow())
    stage.run(summary=again)
    assert again.counts.get("programmes_created", 0) == 0
    assert again.counts["programmes_updated"] >= 1
