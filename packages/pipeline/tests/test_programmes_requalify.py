"""W9 — scoping wired through materialisation and requalification.

Two write paths share one scoping contract:

* ``materialise_programmes(..., scoper=...)`` scopes every draft it writes, so
  a MBBS claim can never produce an in-scope row and a row-numbered duplicate
  claim merges into the existing row instead of creating one;
* ``requalify_programmes(...)`` re-scopes the rows already in the database
  (folding the duplicates materialised before the fix), is idempotent, and
  plans everything without writing under ``dry_run``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fmd.config import load_registry
from fmd.db.repositories import ProgrammeRecord
from fmd.ingest.materialise import materialise_programmes
from fmd.ingest.programmes import ProgrammeMapping
from fmd.programmes.classifier import decision_cache_key
from fmd.programmes.service import build_scoper, requalify_programmes
from fmd.programmes.taxonomy import SCOPE_UNCERTAIN_REASON

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture(scope="module")
def registry():
    return load_registry(CONFIG_PATH, env={})


def claim(claim_id, entity_id, predicate, value, *, unit=None, confidence=0.9, quote=None):
    return {
        "id": claim_id,
        "entity_id": entity_id,
        "predicate": predicate,
        "value": value,
        "unit": unit,
        "confidence": confidence,
        "method": "extracted",
        "verification_status": "supported",
        "document_role": "evidence",
        "superseded_by": None,
        "quote": quote,
    }


INSTITUTIONS = {
    1: {
        "kind": "institution",
        "canonical_name": "TN Dr. M.G.R. Medical University",
        "normalised_name": "tn dr mgr medical university",
        "country": "IN",
    },
    2: {
        "kind": "institution",
        "canonical_name": "Adesh University",
        "normalised_name": "adesh university",
        "country": "IN",
    },
    3: {
        "kind": "institution",
        "canonical_name": "Rani Lakshmi Bai Central Agricultural University",
        "normalised_name": "rani lakshmi bai central agricultural university",
        "country": "IN",
    },
    4: {
        "kind": "institution",
        "canonical_name": "Southern Cross University",
        "normalised_name": "southern cross university",
        "country": "AU",
    },
}


class FakeEntities:
    def get(self, entity_id: int):
        row = INSTITUTIONS.get(entity_id)
        return SimpleNamespace(id=entity_id, **row) if row else None


class FakeProgrammes:
    """Duck-typed ProgrammesRepo: upsert + update_scope + merge_into."""

    def __init__(self, claims: list[dict], rows: list[SimpleNamespace] | None = None) -> None:
        self.claims = claims
        self.rows: dict[int, object] = {}
        self._next_id = 0
        for row in rows or []:
            self._next_id = max(self._next_id, int(row.id))
            record = ProgrammeRecord(
                **{
                    key: value
                    for key, value in vars(row).items()
                    if key not in {"id", "created_at"}
                },
                id=int(row.id),
                created_at=None,
            )
            self.rows[int(row.id)] = record
        self.upsert_calls: list[object] = []
        self.scope_updates: list[dict] = []
        self.merges: list[tuple[int, int]] = []

    def materialisation_claims(self) -> list[dict]:
        return [dict(item) for item in self.claims]

    def get_by_key(self, institution_id: int, normalised_name: str):
        for row in self.rows.values():
            if row.institution_id == institution_id and row.normalised_name == normalised_name:
                return row
        return None

    def list_all(self) -> list[object]:
        return [row for _id, row in sorted(self.rows.items())]

    def upsert(self, programme) -> tuple[object, bool]:
        self.upsert_calls.append(programme)
        for row in self.rows.values():
            if row.institution_id == programme.institution_id and (
                row.normalised_name == programme.normalised_name
            ):
                for key, value in programme.model_dump().items():
                    setattr(row, key, value)
                return row, False
        self._next_id += 1
        record = ProgrammeRecord(id=self._next_id, created_at=None, **programme.model_dump())
        self.rows[record.id] = record
        return record, True

    def update_scope(self, programme_id: int, **kwargs):
        row = self.rows[programme_id]
        self.scope_updates.append({"programme_id": programme_id, **kwargs})
        for key, value in kwargs.items():
            setattr(row, key, value)
        return row

    def merge_into(self, source_id: int, target_id: int) -> dict[str, int]:
        self.merges.append((source_id, target_id))
        self.rows.pop(source_id, None)
        return {"cost_items_moved": 0, "rows_deleted": 1}


class FakeStore:
    def __init__(self) -> None:
        self.rows: dict[str, SimpleNamespace] = {}

    def get(self, cache_key: str):
        return self.rows.get(cache_key)

    def put(self, payload: dict):
        record = SimpleNamespace(**payload)
        self.rows[payload["cache_key"]] = record
        return record


# ------------------------------------------------- materialise with a scoper


def test_materialise_merges_row_numbered_duplicate_into_one_row(registry) -> None:
    # The measured defect: two claims naming the same programme, one of them
    # carrying the row number "3. ", produced two rows. With the W9
    # normalisation they share one grouping key and one row is written.
    claims = [
        claim(47, 1, "seats.programme_name", "B.Sc. Medical Laboratory Technology"),
        claim(48, 1, "seats.programme_name", "3. B.Sc. Medical Laboratory Technology"),
        claim(49, 1, "seats.intake", 20, unit="seats"),
    ]
    repo = FakeProgrammes(claims)
    scoper = build_scoper(registry)
    result = materialise_programmes(
        repo, FakeEntities(), ProgrammeMapping.from_params(None), scoper=scoper
    )
    rows = repo.list_all()
    assert len(rows) == 1
    assert rows[0].normalised_name == "bsc mlt"
    assert rows[0].in_scope is True
    assert rows[0].scope_tier == "rule"
    assert rows[0].scope_reason.startswith("rule_include:")
    assert result.created == 1


def test_materialise_marks_mbbs_and_agriculture_out_of_scope(registry) -> None:
    claims = [
        claim(1, 2, "seats.programme_name", "MBBS - Bachelor of Medicine and Bachelor of Surgery"),
        claim(2, 3, "seats.programme_name", "B.Sc (Hons) Agriculture"),
    ]
    repo = FakeProgrammes(claims)
    scoper = build_scoper(registry)
    materialise_programmes(repo, FakeEntities(), ProgrammeMapping.from_params(None), scoper=scoper)
    rows = {row.normalised_name: row for row in repo.list_all()}
    mbbs = rows["mbbs bachelor medicine bachelor surgery"]
    assert mbbs.in_scope is False
    assert mbbs.scope_tier == "rule"
    assert "mbbs" in mbbs.scope_reason
    agriculture = rows["b sc hons agriculture"]
    assert agriculture.in_scope is False
    assert agriculture.scope_tier == "rule"


def test_materialise_without_scoper_keeps_pre_w9_defaults(registry) -> None:
    claims = [claim(1, 1, "seats.programme_name", "B.Sc. MLT")]
    repo = FakeProgrammes(claims)
    materialise_programmes(repo, FakeEntities(), ProgrammeMapping.from_params(None))
    [row] = repo.list_all()
    assert row.in_scope is True
    assert row.scope_tier is None


def test_materialise_undecided_name_becomes_uncertain_needs_review(registry) -> None:
    # An AU one-off name no rule decides, no evidence, no router: conservative
    # uncertain, queued for review, never admitted.
    claims = [claim(1, 4, "seats.programme_name", "Bachelor of Cura Hyperbarica")]
    repo = FakeProgrammes(claims)
    scoper = build_scoper(registry)  # rules only, no router
    materialise_programmes(repo, FakeEntities(), ProgrammeMapping.from_params(None), scoper=scoper)
    [row] = repo.list_all()
    assert row.in_scope is False
    assert row.scope_tier == "semantic"
    assert row.needs_review is True
    assert SCOPE_UNCERTAIN_REASON in row.review_reasons
    assert row.scope_reason == "semantic_no_evidence_uncertain"


def test_materialise_decided_at_survives_idempotent_reruns(registry) -> None:
    claims = [claim(1, 2, "seats.programme_name", "MBBS")]
    repo = FakeProgrammes(claims)
    scoper = build_scoper(registry)
    mapping = ProgrammeMapping.from_params(None)
    materialise_programmes(repo, FakeEntities(), mapping, scoper=scoper)
    first = repo.get_by_key(2, "mbbs")
    decided_at = first.scope_decided_at
    assert decided_at is not None
    result = materialise_programmes(repo, FakeEntities(), mapping, scoper=scoper)
    assert result.unchanged == 1 and result.written == 0
    assert repo.get_by_key(2, "mbbs").scope_decided_at == decided_at


# ------------------------------------------------------------------ requalify


def _row(
    row_id: int,
    institution_id: int,
    name: str,
    normalised_name: str,
    *,
    in_scope: bool = True,
    scope_tier: str | None = None,
    scope_reason: str | None = None,
    scope_evidence: list[str] | None = None,
    needs_review: bool = False,
    review_reasons: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=row_id,
        institution_id=institution_id,
        name=name,
        normalised_name=normalised_name,
        credential_level="degree",
        in_scope=in_scope,
        scope_tier=scope_tier,
        scope_reason=scope_reason,
        scope_evidence=scope_evidence or [],
        scope_model=None,
        scope_prompt_version=None,
        scope_cached=False,
        scope_decided_at=None,
        needs_review=needs_review,
        review_reasons=review_reasons or [],
    )


def test_requalify_folds_the_row_numbering_duplicate(registry) -> None:
    rows = [
        _row(1, 1, "B.Sc. Medical Laboratory Technology", "bsc mlt"),
        _row(
            5, 1, "3. B.Sc. Medical Laboratory Technology", "3 b sc medical laboratory technology"
        ),
    ]
    repo = FakeProgrammes([], rows)
    result = requalify_programmes(
        repo, FakeEntities(), ProgrammeMapping.from_params(None), None, dry_run=True
    )
    assert result.merged == 1
    assert repo.merges == []  # dry run: planned, not written
    assert [entry.action for entry in result.entries].count("merge") == 1

    result = requalify_programmes(repo, FakeEntities(), ProgrammeMapping.from_params(None), None)
    assert result.merged == 1
    assert repo.merges == [(5, 1)]
    assert [row.id for row in repo.list_all()] == [1]
    # No claims were harmed: the fake never touched any claim list.
    assert repo.claims == []


def test_requalify_marks_mbbs_and_agriculture_out_of_scope(registry) -> None:
    rows = [
        _row(
            8,
            2,
            "MBBS - Bachelor of Medicine and Bachelor of Surgery",
            "mbbs bachelor medicine bachelor surgery",
        ),
        _row(11, 3, "B.Sc (Hons) Agriculture", "b sc hons agriculture"),
        _row(1, 1, "B.Sc. Medical Laboratory Technology", "bsc mlt"),
    ]
    repo = FakeProgrammes([], rows)
    scoper = build_scoper(registry)
    result = requalify_programmes(repo, FakeEntities(), ProgrammeMapping.from_params(None), scoper)
    # MBBS and Agriculture flip OUT; the in-scope MLT row gains its rule decision.
    assert result.rescoped == 3
    by_id = {row.id: row for row in repo.list_all()}
    assert by_id[8].in_scope is False and by_id[8].scope_tier == "rule"
    assert by_id[11].in_scope is False
    assert by_id[1].in_scope is True and by_id[1].scope_tier == "rule"


def test_requalify_is_idempotent(registry) -> None:
    rows = [
        _row(
            8,
            2,
            "MBBS - Bachelor of Medicine and Bachelor of Surgery",
            "mbbs bachelor medicine bachelor surgery",
        ),
        _row(1, 1, "B.Sc. Medical Laboratory Technology", "bsc mlt"),
    ]
    repo = FakeProgrammes([], rows)
    scoper = build_scoper(registry)
    mapping = ProgrammeMapping.from_params(None)
    requalify_programmes(repo, FakeEntities(), mapping, scoper)
    first_updates = len(repo.scope_updates)
    assert first_updates == 2
    result = requalify_programmes(repo, FakeEntities(), mapping, scoper)
    assert result.rescoped == 0 and result.unchanged == 2
    assert len(repo.scope_updates) == first_updates  # nothing rewritten


def test_requalify_applies_manual_decisions_over_the_classifier(registry) -> None:
    rows = [
        _row(
            3,
            1,
            "3. B.Sc. Medical Laboratory Technology",
            "3 b sc medical laboratory technology",
            in_scope=False,
            scope_tier="semantic",
            scope_reason="semantic_uncertain",
            needs_review=True,
            review_reasons=[SCOPE_UNCERTAIN_REASON],
        ),
    ]
    repo = FakeProgrammes([], rows)
    store = FakeStore()
    config = registry.programmes
    key = decision_cache_key(
        "b sc medical laboratory technology", "institution:IN", config.definition.version
    )
    store.rows[key] = SimpleNamespace(
        cache_key=key,
        decision="in_scope",
        tier="manual",
        confidence=None,
        rationale="human verified",
        evidence=["syllabus checked"],
        model=None,
        prompt_version=None,
        cached=False,
        needs_review=False,
        flags=[],
        decided_by="human",
    )
    scoper = build_scoper(registry, store=store)
    result = requalify_programmes(repo, FakeEntities(), ProgrammeMapping.from_params(None), scoper)
    assert result.rescoped == 1
    [row] = repo.list_all()
    assert row.in_scope is True
    assert row.scope_tier == "manual"
    assert row.needs_review is False
    assert SCOPE_UNCERTAIN_REASON not in row.review_reasons


def test_requalify_uncertain_rows_stay_out_and_queued(registry) -> None:
    rows = [_row(21, 4, "Bachelor of Laboratory Practice", "bachelor laboratory practice")]
    repo = FakeProgrammes([], rows)
    scoper = build_scoper(registry)  # no router: semantic unavailable
    result = requalify_programmes(repo, FakeEntities(), ProgrammeMapping.from_params(None), scoper)
    assert result.uncertain == 1
    [row] = repo.list_all()
    assert row.in_scope is False
    assert row.needs_review is True
    assert SCOPE_UNCERTAIN_REASON in row.review_reasons
    assert row.scope_reason == "semantic_unavailable_uncertain"


def test_requalify_empty_table_is_a_noop(registry) -> None:
    repo = FakeProgrammes([], [])
    result = requalify_programmes(repo, FakeEntities(), ProgrammeMapping.from_params(None), None)
    assert result.rows == 0 and result.rescoped == 0 and result.merged == 0
