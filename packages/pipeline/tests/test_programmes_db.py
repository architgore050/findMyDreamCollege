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
from fmd.ingest.materialise import materialise_from_database
from fmd.orchestrate.ingest import ProgrammeStage
from fmd.orchestrate.ledger import RunSummary, utcnow

pytestmark = pytest.mark.usefixtures("pg_database")


def _suffix() -> str:
    return uuid.uuid4().hex[:8]


def _evidence(
    database, suffix: str, *, role: str = "evidence", page_range: str | None = None
) -> tuple[int, int]:
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
        ExtractionCreate(snapshot_id=snapshot.id, tool="test", text=text, page_range=page_range)
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
    quote: str = "B.Sc. MLT",
    char_end: int | None = None,
):
    claim = ClaimsRepo(database).create(
        ClaimCreate(
            extraction_id=extraction_id,
            entity_id=entity_id,
            predicate=predicate,
            value=value,
            unit=unit,
            quote=quote,
            char_start=0,
            char_end=min(char_end if char_end is not None else 9, text_len),
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


# ------------------------------------------------------------------- W8: on-demand materialisation


def test_materialise_from_existing_claims_separates_row_from_aggregate(pg_database) -> None:
    """One document with seats.intake=20 (row) and seats.total=195 (printed Total).

    Only 20 lands on the programme. The aggregate is refused with a reason and
    never written into a per-programme row; the claim itself stays in ``claims``.
    """
    entities = EntitiesRepo(pg_database)
    suffix = _suffix()
    institution = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Four {suffix}", country="IN")
    )
    extraction_id, text_len = _evidence(pg_database, f"mat-{suffix}")
    name = _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.programme_name",
        "B.Sc. MLT",
        quote="Programme: B.Sc. MLT",
    )
    intake = _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.intake",
        20,
        unit="count",
        quote="B.Sc. MLT | 20",
    )
    total = _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.total",
        195,
        unit="count",
        quote="Total | 195",
    )

    result = materialise_from_database(pg_database, registry=None)

    mine = [entry for entry in result.entries if entry.institution_id == institution.id]
    assert len(mine) == 1
    assert mine[0].action == "create"
    row = ProgrammesRepo(pg_database).get_by_key(institution.id, "bsc mlt")
    assert row is not None
    assert row.seats == 20
    assert row.field_sources == {
        "seats.programme_name": str(name.id),
        "seats.intake": str(intake.id),
    }
    assert "seats.total" not in row.field_sources
    refusals = [item for item in result.aggregates_refused if item.value == 195]
    assert len(refusals) == 1
    assert refusals[0].claim_id == str(total.id)
    assert refusals[0].reason == "seats_total_aggregate_not_programme_intake"


def test_materialise_rerun_is_idempotent_and_dry_run_writes_nothing(pg_database) -> None:
    entities = EntitiesRepo(pg_database)
    suffix = _suffix()
    institution = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Five {suffix}", country="IN")
    )
    extraction_id, text_len = _evidence(pg_database, f"rerun-{suffix}")
    _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.programme_name",
        "B.Sc. MLT",
    )
    _claim(pg_database, institution.id, extraction_id, text_len, "seats.intake", 20, unit="count")

    first = materialise_from_database(pg_database, registry=None)
    assert any(entry.action == "create" for entry in first.entries)

    repo = ProgrammesRepo(pg_database)
    before = repo.get_by_key(institution.id, "bsc mlt")

    rerun = materialise_from_database(pg_database, registry=None)
    mine = [entry for entry in rerun.entries if entry.institution_id == institution.id]
    assert mine == [entry for entry in mine if entry.action == "unchanged"]
    assert rerun.created == 0 and rerun.updated == 0
    after = repo.get_by_key(institution.id, "bsc mlt")
    assert after.model_dump() == before.model_dump()
    assert len(repo.list_for_institution(institution.id)) == 1  # no duplicate

    plan = materialise_from_database(pg_database, registry=None, dry_run=True, force=True)
    assert plan.written == rerun.unchanged + 0  # planned writes only
    untouched = repo.get_by_key(institution.id, "bsc mlt")
    assert untouched.model_dump() == before.model_dump()  # dry run wrote nothing


def test_changed_claim_updates_field_sources_on_rerun(pg_database) -> None:
    entities = EntitiesRepo(pg_database)
    suffix = _suffix()
    institution = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Six {suffix}", country="IN")
    )
    extraction_id, text_len = _evidence(pg_database, f"fix-{suffix}")
    old = _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.programme_name",
        "B.Sc. MLT",
    )
    old_intake = _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.intake",
        20,
        unit="count",
        quote="Intake 20",
    )
    materialise_from_database(pg_database, registry=None)

    correction = _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.intake",
        45,
        unit="count",
        quote="Intake 45",
    )
    ClaimsRepo(pg_database).supersede(old_intake.id, correction.id)
    result = materialise_from_database(pg_database, registry=None)

    mine = [entry for entry in result.entries if entry.institution_id == institution.id]
    assert len(mine) == 1 and mine[0].action == "update"
    changes = {change.field: change for change in mine[0].changed_fields}
    assert "seats" in changes
    assert changes["seats"].old_claim_id == str(old_intake.id)
    assert changes["seats"].new_claim_id == str(correction.id)
    row = ProgrammesRepo(pg_database).get_by_key(institution.id, "bsc mlt")
    assert row.seats == 45
    assert row.field_sources["seats.intake"] == str(correction.id)
    assert row.field_sources["seats.programme_name"] == str(old.id)  # unchanged claim kept


def test_inspect_evidence_cites_claims_quote_and_page(pg_database) -> None:
    entities = EntitiesRepo(pg_database)
    suffix = _suffix()
    institution = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Seven {suffix}", country="IN")
    )
    extraction_id, text_len = _evidence(pg_database, f"insp-{suffix}", page_range="3-4")
    _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.programme_name",
        "B.Sc. MLT",
        quote="Programme: B.Sc. MLT",
        char_end=12,
    )
    _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.intake",
        20,
        unit="count",
        quote="Intake 20",
        char_end=8,
    )
    materialise_from_database(pg_database, registry=None)

    repo = ProgrammesRepo(pg_database)
    row = repo.get_by_key(institution.id, "bsc mlt")
    evidence = repo.evidence_for_programme(row.id)
    by_predicate = {item.predicate: item for item in evidence}
    assert set(by_predicate) == {"seats.programme_name", "seats.intake"}
    intake = by_predicate["seats.intake"]
    assert intake.claim_id > 0
    assert intake.quote == "Intake 20"
    assert intake.page_range == "3-4"
    assert (intake.url or "").endswith(f"insp-{suffix}.pdf")
    assert intake.sha256 and len(intake.sha256) == 64
    assert intake.verification_status == "supported"


def test_get_by_key_and_missing_rows(pg_database) -> None:
    entities = EntitiesRepo(pg_database)
    suffix = _suffix()
    institution = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Eight {suffix}", country="IN")
    )
    extraction_id, text_len = _evidence(pg_database, f"key-{suffix}")
    _claim(
        pg_database,
        institution.id,
        extraction_id,
        text_len,
        "seats.programme_name",
        "B.Sc. MLT",
    )
    materialise_from_database(pg_database, registry=None)

    repo = ProgrammesRepo(pg_database)
    assert repo.get_by_key(institution.id, "bsc mlt") is not None
    assert repo.get_by_key(institution.id, "bsc mls") is None
    assert repo.get_by_key(institution.id + 1, "bsc mlt") is None
