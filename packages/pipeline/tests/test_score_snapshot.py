from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fmd.db.repositories import ScoreCreate
from fmd.score import score_programme
from fmd.score.evidence import ClaimIndex, claims_hash
from fmd.score.flags import detect

FIXTURE = Path(__file__).parent / "fixtures" / "score" / "sample_claims.json"
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
GOLDEN_CLAIMS_HASH = "9aa5c26517dd9c4f0e8b9a68d7d07ba3c309ba623c8d0d6903787da0c01cf9d0"


@pytest.fixture()
def data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def score(data: dict, claims: list | None = None):
    return score_programme(
        101,
        data["programme"],
        data["profile"],
        data["claims"] if claims is None else claims,
        now=NOW,
    )


def test_score_is_reproducible_from_claims_hash_and_versions(data: dict) -> None:
    first = score(data)
    second = score(copy.deepcopy(data))

    assert first.claims_hash == GOLDEN_CLAIMS_HASH == second.claims_hash
    assert first.scorer_version == "0.1.0"
    assert first.weights_version == "2026-11-16.1"
    assert first.as_dict() == second.as_dict()
    assert first.composite.point is not None
    assert first.evidence_tier in {"well_evidenced", "partial", "thin", "insufficient"}

    # every published number is inside its own interval
    for sub_score in first.sub_scores.values():
        assert sub_score.lo <= sub_score.point <= sub_score.hi
    assert first.composite.lo <= first.composite.point <= first.composite.hi


def test_changed_evidence_changes_the_hash_but_not_the_versions(data: dict) -> None:
    baseline = score(data)
    mutated = copy.deepcopy(data)
    mutated["claims"][0]["value"] = "unrecognised"
    changed = score(mutated)
    assert changed.claims_hash != baseline.claims_hash
    assert changed.scorer_version == baseline.scorer_version
    assert changed.weights_version == baseline.weights_version


def test_score_create_contract(data: dict) -> None:
    payload = score(data).to_score_create()
    row = ScoreCreate(**payload)
    assert row.entity_id == 101
    assert row.composite is not None
    assert row.composite_lo <= row.composite <= row.composite_hi
    assert set(row.sub_scores) == {
        "outcomes",
        "clinical_training",
        "academic_quality",
        "affordability_roi",
        "financing_feasibility",
        "student_life_safety",
        "admission_realism",
    }
    for sub_score in row.sub_scores.values():
        assert {"point", "lo", "hi", "n_claims", "n_sources", "evidence_tier"} <= set(sub_score)


def test_claims_hash_is_order_independent(data: dict) -> None:
    forward = claims_hash(ClaimIndex.build(data["claims"]))
    reversed_claims = claims_hash(ClaimIndex.build(list(reversed(data["claims"]))))
    assert forward == reversed_claims == GOLDEN_CLAIMS_HASH


def test_scoring_attaches_flags_and_insights(data: dict) -> None:
    result = score(data)
    assert result.insights is not None
    assert result.insights.mandatory
    assert any(item.kind == "recognition_status" for item in result.insights.mandatory)
    assert len(result.insights.selected) <= 5
    # the fixture is clean: no gaming flags on denominator-bearing outcomes
    flagged = {flag.flag_id for flag in result.gaming_flags}
    assert "internship_counted_as_placement" not in flagged


def test_scoring_does_not_turn_missing_evidence_into_a_value(data: dict) -> None:
    empty = score_programme(101, data["programme"], data["profile"], [], now=NOW)
    assert empty.composite.point is None
    assert empty.composite.lo == 0.0
    assert empty.composite.hi == 100.0
    assert empty.evidence_tier == "insufficient"
    for sub_score in empty.sub_scores.values():
        assert sub_score.point is None
        assert (sub_score.lo, sub_score.hi) == (0.0, 100.0)


def test_gaming_flags_are_detected_before_scoring(registry, data: dict) -> None:
    claims = [
        {
            "predicate": "placement_rate",
            "value": 0.9,
            "unit": "rate",
            "qualifier": {"outcome_type": "internship"},
            "claim_id": 99,
            "source_id": 9,
            "quote": "interns placed",
        }
    ]
    flags = detect(claims, registry.weights.gaming_flags)
    assert "internship_counted_as_placement" in {flag.flag_id for flag in flags}
