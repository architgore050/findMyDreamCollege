from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmd.score.config import ScoringConfig
from fmd.score.evidence import ClaimIndex
from fmd.score.sub_scores import (
    SUB_SCORE_IDS,
    ScoreContext,
    apply_transform,
    compute_sub_scores,
    evidence_tier,
)

FIXTURE = Path(__file__).parent / "fixtures" / "score" / "sample_claims.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scoring(registry) -> ScoringConfig:
    return ScoringConfig.from_config(registry.weights)


@pytest.fixture(scope="module")
def context(scoring, registry, fixture) -> ScoreContext:
    return ScoreContext(
        config=scoring,
        cost_rules=registry.cost_rules,
        profile=fixture["profile"],
    )


def test_all_seven_sub_scores_are_known_for_complete_evidence(
    fixture: dict, context: ScoreContext
) -> None:
    index = ClaimIndex.build(fixture["claims"])
    sub_scores = compute_sub_scores(index, context, profile=fixture["profile"])
    assert list(sub_scores) == list(SUB_SCORE_IDS)
    assert set(sub_scores) == set(context.config.weights)

    for name, score in sub_scores.items():
        assert score.point is not None, name
        assert score.lo <= score.point <= score.hi, name
        assert score.n_claims >= context.config.min_claims_for_score
        assert score.evidence_tier in {
            "well_evidenced",
            "partial",
            "thin",
            "insufficient",
        }
        payload = score.as_dict()
        assert {"point", "lo", "hi", "n_claims", "n_sources", "evidence_tier"} <= set(payload)
        assert payload["measured"] is not None
        assert payload["prior_adjusted"] is not None


def test_unknown_is_zero_to_hundred_never_zero_or_fifty(
    scoring: ScoringConfig,
) -> None:
    context = ScoreContext(config=scoring, profile={})
    sub_scores = compute_sub_scores(ClaimIndex.build([]), context, profile={})
    for score in sub_scores.values():
        assert score.point is None
        assert score.lo == 0.0
        assert score.hi == 100.0
        assert score.evidence_tier == "insufficient"
        assert score.unknown_reason


def test_below_min_claims_stays_unknown(scoring: ScoringConfig) -> None:
    claim = {
        "predicate": "internship_to_offer_conversion_12m",
        "value": 0.7,
        "unit": "rate",
        "qualifier": {"denominator": 40},
        "claim_id": 1,
        "source_id": 1,
    }
    context = ScoreContext(config=scoring, profile={})
    scores = compute_sub_scores(ClaimIndex.build([claim]), context, profile={})
    assert scores["outcomes"].point is None
    assert "min_claims_for_score" in (scores["outcomes"].unknown_reason or "")


def test_shrinkage_keeps_measured_and_prior_adjusted_separate(
    scoring: ScoringConfig,
) -> None:
    claims = [
        {
            "predicate": "internship_to_offer_conversion_12m",
            "value": 0.72,
            "unit": "rate",
            "qualifier": {"denominator": 100},
            "claim_id": 1,
            "source_id": 1,
        },
        {
            "predicate": "registration_conversion_12m",
            "value": 0.65,
            "unit": "rate",
            "qualifier": {"denominator": 100},
            "claim_id": 2,
            "source_id": 2,
        },
    ]
    context = ScoreContext(config=scoring, peer_prior={"outcomes": 20.0}, profile={})
    score = compute_sub_scores(ClaimIndex.build(claims), context, profile={})["outcomes"]
    assert score.measured == pytest.approx(68.5)
    assert score.prior_adjusted is not None
    assert score.point == score.prior_adjusted
    assert 20.0 < score.prior_adjusted < score.measured
    assert score.widened is True
    # shrinkage weight n/(n+k) with k = min_claims_for_confidence
    expected = (2 / 7) * 68.5 + (5 / 7) * 20.0
    assert score.prior_adjusted == pytest.approx(expected, abs=0.01)


def test_no_shrinkage_when_evidence_is_sufficient(scoring: ScoringConfig) -> None:
    claims = [
        {
            "predicate": predicate,
            "value": value,
            "claim_id": index,
            "source_id": index,
        }
        for index, (predicate, value) in enumerate(
            [
                ("city_safety_index", 70),
                ("hostel_available", True),
                ("gender_safety_measures", True),
                ("support_services_count", 4),
                ("review_aspect_safety", 65),
            ],
            start=1,
        )
    ]
    context = ScoreContext(config=scoring, peer_prior={"student_life_safety": 20.0}, profile={})
    score = compute_sub_scores(ClaimIndex.build(claims), context, profile={})["student_life_safety"]
    assert score.n_claims == 5
    assert score.widened is False
    assert score.prior_adjusted == score.measured


def test_absent_prior_is_explicit_not_silent(scoring: ScoringConfig) -> None:
    claims = [
        {
            "predicate": "city_safety_index",
            "value": 70,
            "claim_id": 1,
            "source_id": 1,
        },
        {
            "predicate": "hostel_available",
            "value": True,
            "claim_id": 2,
            "source_id": 2,
        },
    ]
    context = ScoreContext(config=scoring, profile={})
    score = compute_sub_scores(ClaimIndex.build(claims), context, profile={})[
        "student_life_safety"
    ]
    assert score.shrinkage_inactive is True
    assert score.prior_adjusted == score.measured
    assert score.as_dict()["shrinkage_inactive"] is True


def test_configured_naac_transform_is_monotone(scoring: ScoringConfig) -> None:
    spec = scoring.transforms["naac_grade"]
    assert apply_transform("A++", spec) == 100.0
    assert apply_transform("A+", spec) == 90.0
    assert apply_transform("a +", spec) == 90.0
    assert apply_transform("B++", spec) == 68.0
    assert apply_transform("D", spec) == 10.0
    assert apply_transform("unknown-grade", spec) is None

    recognition = scoring.transforms["recognition_status"]
    assert apply_transform("recognised", recognition) == 100.0
    assert apply_transform("unrecognised", recognition) == 0.0
    assert apply_transform("unknown", recognition) is None

    hours = scoring.transforms["internship_hours"]
    assert apply_transform(900, hours) == pytest.approx(50.0)
    assert apply_transform(1800, hours) == 100.0


def test_evidence_tier_thresholds_follow_config(scoring: ScoringConfig) -> None:
    tiers = scoring.evidence_tiers
    assert evidence_tier(10, 4, 0.1, tiers) == "well_evidenced"
    assert evidence_tier(4, 2, 0.5, tiers) == "partial"
    assert evidence_tier(1, 1, 1.0, tiers) == "thin"
    assert evidence_tier(0, 0, 0.0, tiers) == "insufficient"
    # modelled share above the well-evidenced ceiling demotes the tier
    assert evidence_tier(10, 4, 0.9, tiers) == "thin"


def test_flag_penalty_widens_the_component_interval(scoring: ScoringConfig) -> None:
    claims = [
        {
            "predicate": "p25_salary_to_col_ratio",
            "value": 2.4,
            "unit": "ratio",
            "claim_id": 9,
            "source_id": 1,
        },
        {
            "predicate": "registration_conversion_12m",
            "value": 0.65,
            "unit": "rate",
            "qualifier": {"denominator": 100},
            "claim_id": 10,
            "source_id": 2,
        },
    ]
    clean = ScoreContext(config=scoring, profile={})
    penalised = ScoreContext(config=scoring, flag_penalties={"9": 0.4}, profile={})
    index = ClaimIndex.build(claims)
    clean_score = compute_sub_scores(index, clean, profile={})["outcomes"]
    penalised_score = compute_sub_scores(index, penalised, profile={})["outcomes"]
    assert clean_score.point is not None
    assert penalised_score.point is not None
    assert penalised_score.lo < clean_score.lo
    assert penalised_score.hi > clean_score.hi
