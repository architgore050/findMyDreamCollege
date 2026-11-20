"""``fmd programmes materialise`` — the reusable, on-demand materialisation.

The crawl-path stage was the only writer before W8, so claims verified after a
crawl never became programme rows. These tests pin the extracted function in
``fmd/ingest/materialise.py``: same contract as the stage (offline, fake repos),
plus the plan/idempotency/aggregate-guard behaviour the CLI exposes.
"""

from __future__ import annotations

import json
from pathlib import Path

from fmd.db.repositories import ProgrammeRecord
from fmd.ingest.materialise import (
    materialise_programmes,
)
from fmd.ingest.programmes import (
    AGGREGATE_SEATS_REFUSED,
    ATTRIBUTE_FIELDS,
    ProgrammeMapping,
    field_for_predicate_map,
    programme_drafts,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "programmes"
REGISTRY_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def claim(
    claim_id: int,
    entity_id: int,
    predicate: str,
    value,
    *,
    unit: str | None = None,
    status: str = "supported",
    method: str = "extracted",
    confidence: float = 0.9,
    document_role: str = "evidence",
    superseded_by: int | None = None,
) -> dict:
    return {
        "id": claim_id,
        "entity_id": entity_id,
        "predicate": predicate,
        "value": value,
        "unit": unit,
        "confidence": confidence,
        "method": method,
        "verification_status": status,
        "document_role": document_role,
        "superseded_by": superseded_by,
    }


class FakeEntities:
    """Duck-typed EntitiesRepo (get / find / find_normalised)."""

    def __init__(self, rows: dict[int, dict]) -> None:
        self.rows = rows

    def get(self, entity_id: int):
        row = self.rows.get(entity_id)
        from types import SimpleNamespace

        return SimpleNamespace(id=entity_id, **row) if row else None

    def find(self, kind: str, canonical_name: str):
        from types import SimpleNamespace

        for entity_id, row in self.rows.items():
            if row["kind"] == kind and row["canonical_name"] == canonical_name:
                return SimpleNamespace(id=entity_id, **row)
        return None

    def find_normalised(self, kind: str, normalised_name: str):
        from types import SimpleNamespace

        for entity_id, row in self.rows.items():
            if row["kind"] == kind and row.get("normalised_name") == normalised_name:
                return SimpleNamespace(id=entity_id, **row)
        return None


class FakeProgrammes:
    """Duck-typed ProgrammesRepo with a key lookup and an upsert spy."""

    def __init__(self, claims: list[dict]) -> None:
        self.claims = claims
        self.rows: dict[tuple[int, str], ProgrammeRecord] = {}
        self.upsert_calls = 0

    def materialisation_claims(self) -> list[dict]:
        return [dict(item) for item in self.claims]

    def get_by_key(self, institution_id: int, normalised_name: str) -> ProgrammeRecord | None:
        return self.rows.get((institution_id, normalised_name))

    def upsert(self, programme) -> tuple[ProgrammeRecord, bool]:
        self.upsert_calls += 1
        key = (programme.institution_id, programme.normalised_name)
        created = key not in self.rows
        record = ProgrammeRecord(id=len(self.rows) + 1, created_at=None, **programme.model_dump())
        self.rows[key] = record
        return record, created


INSTITUTIONS = {
    1: {
        "kind": "institution",
        "canonical_name": "Stanley College",
        "normalised_name": "stanley college",
    },
    2: {
        "kind": "institution",
        "canonical_name": "Adesh University",
        "normalised_name": "adesh university",
    },
    3: {
        "kind": "institution",
        "canonical_name": "Affiliating University",
        "normalised_name": "affiliating university",
    },
}


def _programmes(claims: list[dict]) -> FakeProgrammes:
    return FakeProgrammes(claims)


def _entities() -> FakeEntities:
    return FakeEntities(INSTITUTIONS)


# ----------------------------------------------------------------- creation


def test_materialise_creates_rows_with_field_sources() -> None:
    repo = _programmes(
        [
            claim(1, 1, "seats.programme_name", "B.Sc. MLT"),
            claim(2, 1, "seats.intake", 20, unit="count"),
            claim(3, 1, "curriculum.duration_months", 48, unit="months"),
            claim(4, 1, "curriculum.internship_months", 6, unit="months"),
        ]
    )
    result = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    assert result.claims == 4
    assert result.created == 1 and result.updated == 0 and result.unchanged == 0
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.action == "create" and entry.written
    assert entry.field_sources == {
        "seats.programme_name": "1",
        "seats.intake": "2",
        "curriculum.duration_months": "3",
        "curriculum.internship_months": "4",
    }
    stored = repo.rows[(1, "bsc mlt")]
    assert stored.seats == 20 and stored.duration_months == 48
    assert stored.internship_months == 6.0


def test_missing_fields_stay_null_and_nothing_is_invented() -> None:
    repo = _programmes([claim(1, 1, "seats.programme_name", "B.Sc. MLT")])
    materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    stored = repo.rows[(1, "bsc mlt")]
    assert stored.seats is None
    assert stored.duration_months is None
    assert stored.internship_months is None
    assert stored.medium is None
    assert stored.recognition_status == "unknown"
    assert stored.field_sources == {"seats.programme_name": "1"}


# --------------------------------------------------------------- idempotency


def test_rerun_is_idempotent_and_skips_unchanged_rows() -> None:
    repo = _programmes(
        [
            claim(1, 1, "seats.programme_name", "B.Sc. MLT"),
            claim(2, 1, "seats.intake", 20, unit="count"),
        ]
    )
    first = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))
    assert first.created == 1

    again = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))
    assert again.created == 0
    assert again.updated == 0
    assert again.unchanged == 1
    assert again.written == 0
    assert repo.upsert_calls == 1  # the unchanged row was not rewritten
    assert len(repo.rows) == 1  # no duplicates


def test_force_rewrites_unchanged_rows() -> None:
    repo = _programmes([claim(1, 1, "seats.programme_name", "B.Sc. MLT")])
    materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    forced = materialise_programmes(
        repo, _entities(), ProgrammeMapping.from_params(None), force=True
    )
    assert forced.unchanged == 0
    assert forced.written == 1 and forced.updated == 1
    assert repo.upsert_calls == 2
    assert len(repo.rows) == 1


def test_changed_claim_updates_field_sources_and_reports_the_field() -> None:
    repo = _programmes(
        [
            claim(1, 1, "seats.programme_name", "B.Sc. MLT"),
            claim(2, 1, "seats.intake", 20, unit="count"),
        ]
    )
    materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    # A correction lands: the old claim is superseded, a new one replaces it.
    repo.claims = [
        claim(1, 1, "seats.programme_name", "B.Sc. MLT"),
        claim(2, 1, "seats.intake", 20, unit="count", superseded_by=9),
        claim(9, 1, "seats.intake", 60, unit="count"),
    ]
    again = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    assert again.updated == 1 and again.unchanged == 0
    entry = again.entries[0]
    changes = {change.field: change for change in entry.changed_fields}
    assert set(changes) == {"seats"}
    assert changes["seats"].old_claim_id == "2"
    assert changes["seats"].new_claim_id == "9"
    stored = repo.rows[(1, "bsc mlt")]
    assert stored.seats == 60
    assert stored.field_sources["seats.intake"] == "9"


def test_dry_run_writes_nothing_and_reports_the_plan() -> None:
    repo = _programmes(
        [
            claim(1, 1, "seats.programme_name", "B.Sc. MLT"),
            claim(2, 1, "seats.intake", 20, unit="count"),
        ]
    )
    result = materialise_programmes(
        repo, _entities(), ProgrammeMapping.from_params(None), dry_run=True
    )

    assert repo.upsert_calls == 0
    assert result.dry_run is True
    assert result.written == 1 and result.created == 1  # the plan, not the write
    assert result.entries[0].written is False
    assert result.entries[0].record_id is None
    assert not repo.rows  # nothing landed


# --------------------------------------------- the aggregate-vs-row guard (W8)


def _fixture_claims() -> list[dict]:
    payload = json.loads((FIXTURES / "seat_matrix_row_plus_total.json").read_text())
    return payload["claims"]


def test_row_intake_lands_and_aggregate_total_is_refused_with_a_reason() -> None:
    """Acceptance 5: one document, seats.intake=20 (row) and seats.total=195.

    Only 20 may land on the programme; 195 is an institution-level aggregate
    (a printed Total row). It is discarded with an explicit reason — the claim
    itself stays in ``claims`` (append-only evidence is never lost).
    """
    repo = _programmes(_fixture_claims())
    result = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    entry = result.entries[0]
    assert entry.normalised_name == "bsc mlt"
    assert entry.seats == 20
    assert entry.field_sources == {"seats.programme_name": "1", "seats.intake": "2"}
    stored = repo.rows[(1, "bsc mlt")]
    assert stored.seats == 20
    assert stored.field_sources.get("seats.total") is None

    assert len(result.aggregates_refused) == 1
    refusal = result.aggregates_refused[0]
    assert refusal.claim_id == "3"
    assert refusal.value == 195
    assert refusal.reason == AGGREGATE_SEATS_REFUSED
    assert refusal.institution_id == 1
    # The refusal is also visible in the plan's rejection list.
    reasons = [item["reason"] for item in result.rejected]
    assert AGGREGATE_SEATS_REFUSED in reasons


def test_document_with_only_an_aggregate_leaves_seats_null() -> None:
    repo = _programmes(
        [
            claim(1, 1, "seats.programme_name", "B.Sc. MLT"),
            claim(2, 1, "seats.total", 195, unit="count"),
        ]
    )
    result = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    stored = repo.rows[(1, "bsc mlt")]
    assert stored.seats is None  # UNKNOWN is a value; a wrong number is worse
    assert len(result.aggregates_refused) == 1


def test_pure_layer_reports_the_aggregate_refusal() -> None:
    drafts, rejected = programme_drafts(_fixture_claims(), ProgrammeMapping.from_params(None))
    assert drafts[0].seats == 20
    assert any(item.get("reason") == AGGREGATE_SEATS_REFUSED for item in rejected)


# ------------------------------------------------------- ER name grouping (W8)


def test_alias_confirmed_names_collapse_into_one_programme() -> None:
    """`B.Sc. MLT` and `Bachelor of Medical Laboratory Technology (BMLT)` are
    registry-confirmed aliases: one row, no duplicate, no review flag."""
    repo = _programmes(
        [
            claim(1, 1, "seats.programme_name", "B.Sc. MLT"),
            claim(
                2,
                1,
                "curriculum.programme_name",
                "Bachelor of Medical Laboratory Technology (BMLT)",
            ),
            claim(3, 1, "seats.intake", 20, unit="count"),
        ]
    )
    result = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    assert len(result.entries) == 1
    assert result.entries[0].normalised_name == "bsc mlt"
    assert result.entries[0].needs_review is False
    assert len(repo.rows) == 1


def test_unconfirmed_similar_names_stay_separate_and_flagged() -> None:
    """MLT vs MLS: similar, not confirmed — two rows, both needs_review."""
    repo = _programmes(
        [
            claim(1, 1, "seats.programme_name", "B.Sc. MLT"),
            claim(2, 1, "curriculum.programme_name", "Bachelor of Medical Laboratory Sciences"),
        ]
    )
    result = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    assert {entry.normalised_name for entry in result.entries} == {"bsc mlt", "bsc mls"}
    assert all(entry.needs_review for entry in result.entries)
    assert all(
        any("possible_duplicate_programme_name" in reason for reason in entry.review_reasons)
        for entry in result.entries
    )
    assert len(repo.rows) == 2


# ------------------------------------------------- evidence completeness (W8)


def test_evidence_completeness_counts_fields_and_registry_components(registry) -> None:
    """The list view's completeness: 6 materialisation fields, registry components."""
    from fmd.ingest.materialise import evidence_completeness

    mapping = ProgrammeMapping.from_params(None)
    institution_claims = [
        claim(1, 2, "curriculum.programme_name", "BMLS"),
        claim(2, 2, "curriculum.internship_months", 6, unit="months"),
        claim(3, 2, "accreditation.naac_grade", "A+"),
        claim(4, 2, "seats.total", 195, unit="count"),
        claim(5, 2, "fee.tuition", 95000, unit="INR"),  # no component; not a materialised field
    ]
    completeness = evidence_completeness(institution_claims, mapping, registry.predicates)

    assert completeness.fields_total == len(ATTRIBUTE_FIELDS) == 6
    assert completeness.fields_backed == 4  # name, credential_level, internship, aggregate
    assert set(completeness.backed_fields) == {
        "name",
        "credential_level",
        "internship_months",
        "seats_total_aggregate",
    }
    # credential_level rides on the name claim, never on a claim of its own.
    assert completeness.backed_fields[0] == "name"
    assert "academic_quality" in completeness.components_backed
    assert "clinical_training" in completeness.components_backed
    # seats.total feeds admission_realism too — the registry scores the
    # aggregate as an institution-level signal, even though materialisation
    # refuses to write it into a row.
    assert "admission_realism" in completeness.components_backed
    assert completeness.components_total == 7  # the committed weights.yaml sub-scores


def test_field_for_predicate_map_covers_every_consumed_predicate() -> None:
    mapping = ProgrammeMapping.from_params(None)
    bridge = field_for_predicate_map(mapping)
    assert bridge["seats.intake"] == "seats"
    assert bridge["seats.sanctioned"] == "seats"
    assert bridge["seats.total"] == "seats_total_aggregate"
    assert bridge["seats.duration_months"] == "duration_months"
    assert bridge["curriculum.duration_months"] == "duration_months"
    assert bridge["curriculum.internship_months"] == "internship_months"
    for predicate in mapping.name_predicates:
        assert bridge[predicate] == "name"


# ------------------------------------------- unresolvable entities still refused


def test_unresolvable_institution_is_refused_by_the_reusable_function() -> None:
    repo = _programmes([claim(1, 99, "seats.programme_name", "B.Sc. MLT")])
    result = materialise_programmes(repo, _entities(), ProgrammeMapping.from_params(None))

    assert result.entries == ()
    assert any(item["reason"] == "entity_unresolvable" for item in result.rejected)
    assert not repo.rows
