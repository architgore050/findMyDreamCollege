"""ProgrammeStage: verified claims -> rows, idempotently, with real provenance."""

from __future__ import annotations

from types import SimpleNamespace

from fmd.ingest.programmes import ProgrammeMapping
from fmd.orchestrate.ingest import ProgrammeStage
from fmd.orchestrate.ledger import RunSummary, utcnow


class FakeEntities:
    def __init__(self, entities: dict[int, dict]) -> None:
        self.rows = entities

    def get(self, entity_id: int):
        row = self.rows.get(entity_id)
        return SimpleNamespace(**row) if row else None

    def find(self, kind: str, canonical_name: str):
        for entity_id, row in self.rows.items():
            if row["kind"] == kind and row["canonical_name"] == canonical_name:
                return SimpleNamespace(id=entity_id, **row)
        return None

    def find_normalised(self, kind: str, normalised_name: str):
        for entity_id, row in self.rows.items():
            if row["kind"] == kind and row.get("normalised_name") == normalised_name:
                return SimpleNamespace(id=entity_id, **row)
        return None


class FakeProgrammes:
    def __init__(self, claims: list[dict]) -> None:
        self.claims = claims
        self.rows: dict[tuple[int, str], object] = {}

    def materialisation_claims(self) -> list[dict]:
        return list(self.claims)

    def upsert(self, programme):
        key = (programme.institution_id, programme.normalised_name)
        created = key not in self.rows
        self.rows[key] = programme
        return programme, created

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.rows),
            "needs_review": sum(1 for row in self.rows.values() if row.needs_review),
        }


def _claim(claim_id: int, entity_id: int, predicate: str, value, unit=None) -> dict:
    return {
        "id": claim_id,
        "entity_id": entity_id,
        "predicate": predicate,
        "value": value,
        "unit": unit,
        "confidence": 0.9,
        "method": "extracted",
        "verification_status": "supported",
        "document_role": "evidence",
        "superseded_by": None,
    }


def _summary() -> RunSummary:
    return RunSummary(run_id="r1", stage="ingest", started_at=utcnow())


def _stage(claims: list[dict], entities: dict[int, dict] | None = None) -> ProgrammeStage:
    return ProgrammeStage(
        programmes=FakeProgrammes(claims),
        entities=FakeEntities(
            entities
            or {
                1: {"kind": "institution", "canonical_name": "One University"},
                2: {"kind": "institution", "canonical_name": "Affiliating University"},
            }
        ),
        mapping=ProgrammeMapping.from_params({}),
    )


def test_stage_writes_programme_and_reports_counts() -> None:
    stage = _stage(
        [
            _claim(1, 1, "curriculum.programme_name", "B.Sc. MLT"),
            _claim(2, 1, "seats.total", 40, "count"),
        ]
    )
    summary = _summary()
    written = stage.run(summary=summary)

    assert len(written) == 1
    assert summary.counts["programmes"] == 1
    assert summary.counts["programmes_created"] == 1
    assert summary.counts["programme_claims"] == 2
    assert written[0].field_sources == {"curriculum.programme_name": "1", "seats.total": "2"}


def test_reingest_is_idempotent_and_does_not_duplicate() -> None:
    stage = _stage([_claim(1, 1, "curriculum.programme_name", "B.Sc. MLT")])
    stage.run(summary=_summary())

    again = _summary()
    stage.run(summary=again)

    assert len(stage.programmes.rows) == 1
    assert again.counts.get("programmes_created", 0) == 0
    assert again.counts["programmes_updated"] == 1


def test_affiliation_resolves_to_entity_and_records_claim_id() -> None:
    stage = _stage(
        [
            _claim(1, 1, "curriculum.programme_name", "B.Sc. MLT"),
            _claim(2, 1, "curriculum.affiliated_university", "Affiliating University"),
        ]
    )
    written = stage.run(summary=_summary())

    assert written[0].affiliated_university_id == 2
    assert written[0].field_sources["curriculum.affiliated_university"] == "2"
    assert written[0].needs_review is False


def test_unresolved_affiliation_stays_null_and_is_flagged() -> None:
    stage = _stage(
        [
            _claim(1, 1, "curriculum.programme_name", "B.Sc. MLT"),
            _claim(2, 1, "curriculum.affiliated_university", "Unknown University"),
        ]
    )
    written = stage.run(summary=_summary())

    assert written[0].affiliated_university_id is None
    assert written[0].needs_review is True
    assert "affiliated_university_unresolved" in written[0].review_reasons


def test_no_name_claim_creates_no_programme() -> None:
    stage = _stage([_claim(1, 1, "seats.total", 40, "count")])
    summary = _summary()
    written = stage.run(summary=summary)

    assert written == []
    assert summary.counts["programmes_rejected"] == 1
    assert summary.metadata["programme_rejections"][0]["reason"] == "no_programme_name_claim"
