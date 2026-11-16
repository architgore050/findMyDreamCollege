from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fmd.score.evidence import ClaimView
from fmd.score.flags import confidence_penalties, detect

NOW = datetime(2026, 11, 16, tzinfo=UTC)


@pytest.fixture(scope="module")
def weights(registry) -> dict[str, float]:
    return dict(registry.weights.confidence_penalties)


def claim(predicate: str, value: object, **overrides) -> ClaimView:
    payload = {
        "predicate": predicate,
        "value": value,
        "unit": overrides.pop("unit", None),
        "claim_id": overrides.pop("claim_id", 1),
        "qualifier": overrides.pop("qualifier", {}),
        "method": overrides.pop("method", "extracted"),
        "confidence": overrides.pop("confidence", 1.0),
        "observed_at": overrides.pop("observed_at", NOW),
        "source_id": overrides.pop("source_id", 1),
        "quote": overrides.pop("quote", None),
    }
    payload.update(overrides)
    return ClaimView(**payload)


def test_every_configured_flag_has_a_weight(weights: dict[str, float]) -> None:
    from fmd.score.flags import DETECTORS

    assert set(DETECTORS) == set(weights)


def test_internship_counted_as_placement(weights: dict[str, float]) -> None:
    flagged = claim(
        "placement_rate",
        0.9,
        unit="rate",
        qualifier={
            "denominator": 100,
            "outcome_type": "internship",
            "credential_level": "degree",
            "cohort_year": 2025,
        },
    )
    found = detect([flagged], weights)
    assert [f.flag_id for f in found] == ["internship_counted_as_placement"]
    assert found[0].weight == weights["internship_counted_as_placement"]
    assert found[0].claim_id == 1

    clean = claim(
        "placement_rate",
        0.9,
        unit="rate",
        qualifier={
            "denominator": 100,
            "outcome_type": "employment",
            "credential_level": "degree",
            "cohort_year": 2025,
        },
        quote="90 of 100 graduates joined employers",
    )
    assert [f.flag_id for f in detect([clean], weights)] == []


def test_missing_denominator(weights: dict[str, float]) -> None:
    found = detect([claim("placement_rate", 0.9, unit="rate")], weights)
    assert "missing_denominator" in {f.flag_id for f in found}
    with_denominator = claim("placement_rate", 0.9, unit="rate", qualifier={"denominator": 80})
    assert "missing_denominator" not in {f.flag_id for f in detect([with_denominator], weights)}


def test_no_distribution_and_headline_package(weights: dict[str, float]) -> None:
    salary = claim("median_salary", 400000, unit="INR", qualifier={"cohort_year": 2025})
    flags = {f.flag_id for f in detect([salary], weights)}
    assert "no_distribution" in flags

    spread = claim(
        "median_salary",
        400000,
        unit="INR",
        qualifier={"cohort_year": 2025, "q25": 300000, "q75": 500000},
    )
    assert "no_distribution" not in {f.flag_id for f in detect([spread], weights)}

    headline = claim("highest_package", 1800000, unit="INR")
    assert "headline_is_highest_package" in {f.flag_id for f in detect([headline], weights)}


def test_round_numbers_and_no_cohort_year(weights: dict[str, float]) -> None:
    flags = {f.flag_id for f in detect([claim("median_salary", 500000, unit="INR")], weights)}
    assert "round_numbers" in flags
    assert "no_cohort_year" in flags
    precise = claim("median_salary", 437650, unit="INR", qualifier={"cohort_year": 2025, "q25": 1})
    remaining = {f.flag_id for f in detect([precise], weights)}
    assert "round_numbers" not in remaining
    assert "no_cohort_year" not in remaining


def test_aggregated_across_degrees(weights: dict[str, float]) -> None:
    bundled = claim("placement_rate", 0.8, unit="rate", qualifier={"denominator": 50})
    assert "aggregated_across_degrees" in {f.flag_id for f in detect([bundled], weights)}
    separated = claim(
        "placement_rate",
        0.8,
        unit="rate",
        qualifier={"denominator": 50, "credential_level": "degree"},
    )
    assert "aggregated_across_degrees" not in {f.flag_id for f in detect([separated], weights)}


def test_recruiters_without_counts(weights: dict[str, float]) -> None:
    found = detect([claim("recruiters", ["Acme Labs", "Beta Diagnostics"])], weights)
    assert "recruiters_without_counts" in {f.flag_id for f in found}
    with_counts = claim("recruiters", ["Acme Labs"], qualifier={"count": 12})
    assert "recruiters_without_counts" not in {f.flag_id for f in detect([with_counts], weights)}


def test_offers_not_joinings(weights: dict[str, float]) -> None:
    offers = claim(
        "employment_rate",
        0.8,
        unit="rate",
        qualifier={"denominator": 100},
        quote="80 offer letters issued",
    )
    assert "offers_not_joinings" in {f.flag_id for f in detect([offers], weights)}
    joinings = claim(
        "employment_rate",
        0.8,
        unit="rate",
        qualifier={"denominator": 100, "outcome_type": "joining"},
        quote="80 candidates joined",
    )
    assert "offers_not_joinings" not in {f.flag_id for f in detect([joinings], weights)}


def test_yoy_jump_needs_peer_sigma(weights: dict[str, float]) -> None:
    previous = claim(
        "median_salary", 300000, unit="INR", claim_id=1, observed_at=NOW - timedelta(days=400)
    )
    current = claim(
        "median_salary",
        600000,
        unit="INR",
        claim_id=2,
        observed_at=NOW,
        qualifier={"peer_sigma": 50000},
    )
    flags = detect([previous, current], weights)
    assert "yoy_jump_gt_2sigma" in {f.flag_id for f in flags}

    no_sigma = claim("median_salary", 600000, unit="INR", claim_id=3, observed_at=NOW)
    assert "yoy_jump_gt_2sigma" not in {
        f.flag_id
        for f in detect([claim("median_salary", 300000, unit="INR", claim_id=4), no_sigma], weights)
    }


def test_confidence_penalties_are_multiplicative(weights: dict[str, float]) -> None:
    flags = detect(
        [
            claim("placement_rate", 0.9, unit="rate", claim_id=7),
            claim("highest_package", 2000000, unit="INR", claim_id=7),
        ],
        weights,
    )
    penalties = confidence_penalties(flags)
    ids = {flag.flag_id for flag in flags if flag.claim_id == 7}
    assert {"missing_denominator", "headline_is_highest_package"} <= ids
    expected = 1.0
    for flag in flags:
        if flag.claim_id == 7:
            expected *= 1.0 - weights[flag.flag_id]
    assert penalties["7"] == pytest.approx(1.0 - expected)


def test_confidence_penalties_are_not_shares(registry) -> None:
    from fmd.score.config import ScoringConfig

    penalties = ScoringConfig.from_config(registry.weights).confidence_penalty_weights
    # They are multipliers, so the sum is irrelevant and may exceed 1.
    assert all(0.0 <= value < 1.0 for value in penalties.values())
    assert sum(penalties.values()) != pytest.approx(1.0)


def test_confidence_penalty_composition_has_a_floor(weights: dict[str, float]) -> None:
    from fmd.score.flags import MIN_CONFIDENCE, GamingFlag

    flags = [
        GamingFlag(
            flag_id=f"flag_{index}",
            weight=0.5,
            claim_id=1,
            predicate="placement_rate",
            evidence="test",
            detail={},
        )
        for index in range(20)
    ]
    penalties = confidence_penalties(flags)
    assert penalties["1"] == pytest.approx(1.0 - MIN_CONFIDENCE)
    assert all(0.0 <= weight < 1.0 for weight in weights.values())


def test_penalty_weights_outside_unit_interval_are_rejected(registry) -> None:
    from fmd.score.config import ScoringConfig

    raw = registry.weights.model_dump()
    raw["confidence_penalties"]["missing_denominator"] = 1.0
    config = ScoringConfig.from_config(raw)
    assert any("0 <= w < 1" in problem for problem in config.validate())


def test_unknown_flag_ids_are_not_silently_applied() -> None:
    found = detect([claim("placement_rate", 0.9, unit="rate")], {"unknown_flag": 0.5})
    assert found == ()
