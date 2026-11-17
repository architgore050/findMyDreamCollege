from __future__ import annotations

import pytest

from fmd.score.aggregate import compare, composite, dominates, pareto_matrix, validate_weights
from fmd.score.sub_scores import SubScore


def _score(point: float | None, lo: float, hi: float, **kwargs) -> SubScore:
    return SubScore(
        point=point,
        lo=lo,
        hi=hi,
        n_claims=kwargs.pop("n_claims", 5),
        n_sources=kwargs.pop("n_sources", 2),
        evidence_tier=kwargs.pop("evidence_tier", "partial"),
        **kwargs,
    )


def test_weights_must_sum_to_one_hundred() -> None:
    with pytest.raises(ValueError, match="expected 100"):
        validate_weights({"a": 60.0, "b": 30.0})
    with pytest.raises(ValueError, match="non-negative"):
        validate_weights({"a": 120.0, "b": -20.0})


def test_composite_with_all_known_sub_scores() -> None:
    sub_scores = {
        "a": _score(80.0, 70.0, 90.0),
        "b": _score(60.0, 50.0, 70.0),
    }
    result = composite(sub_scores, {"a": 50.0, "b": 50.0})
    assert result.point == pytest.approx(70.0)
    # comonotonic: plain interval arithmetic
    assert result.lo_comonotonic == pytest.approx(60.0)
    assert result.hi_comonotonic == pytest.approx(80.0)
    # independent: quadrature, so strictly narrower than comonotonic
    assert result.lo_independent > result.lo_comonotonic
    assert result.hi_independent < result.hi_comonotonic
    assert result.bound == "independent"
    assert result.unknown_sub_scores == ()


def test_unknown_propagates_as_a_wide_partial_composite() -> None:
    sub_scores = {
        "a": _score(80.0, 70.0, 90.0),
        "b": _score(None, 0.0, 100.0, evidence_tier="insufficient", n_claims=0, n_sources=0),
    }
    result = composite(sub_scores, {"a": 50.0, "b": 50.0})
    # The point is the weighted mean of what is known, but the interval still
    # carries the unknown component's full [0,100] mass.
    assert result.point == pytest.approx(80.0)
    assert result.lo_comonotonic == pytest.approx(35.0)
    assert result.hi_comonotonic == pytest.approx(95.0)
    assert result.lo <= result.point <= result.hi
    assert result.bound == "partial_wide"
    assert result.partial is True
    assert result.components_used == ("a",)
    assert result.components_unknown == ("b",)
    assert result.unknown_sub_scores == ("b",)
    assert result.known_weight == 50.0


def test_zero_components_is_unknown() -> None:
    sub_scores = {
        "a": _score(None, 0.0, 100.0, evidence_tier="insufficient", n_claims=0, n_sources=0),
        "b": _score(None, 0.0, 100.0, evidence_tier="insufficient", n_claims=0, n_sources=0),
    }
    result = composite(sub_scores, {"a": 50.0, "b": 50.0})
    assert result.point is None
    assert (result.lo, result.hi) == (0.0, 100.0)
    assert result.bound == "unknown"
    assert result.components_used == ()
    assert result.partial is True
    assert result.evidence_tier == "insufficient"


def test_composite_tier_is_capped_when_a_sub_score_is_missing() -> None:
    tiers = {
        "well_evidenced": {"min_sources": 3, "min_claims": 8, "max_modelled_share": 0.25},
        "partial": {"min_sources": 2, "min_claims": 3, "max_modelled_share": 0.6},
        "thin": {"min_sources": 1, "min_claims": 1, "max_modelled_share": 1.0},
        "insufficient": {"min_sources": 0, "min_claims": 0, "max_modelled_share": 1.0},
    }
    sub_scores = {
        "a": _score(80.0, 70.0, 90.0, evidence_tier="well_evidenced"),
        "b": _score(None, 0.0, 100.0, evidence_tier="partial", n_claims=0, n_sources=0),
    }
    result = composite(sub_scores, {"a": 50.0, "b": 50.0}, tiers=tiers)
    assert result.evidence_tier == "partial"


def test_pareto_dominance_including_not_comparable() -> None:
    a = {"x": _score(80.0, 70.0, 90.0), "y": _score(60.0, 50.0, 70.0)}
    b = {"x": _score(70.0, 60.0, 80.0), "y": _score(50.0, 40.0, 60.0)}
    assert compare(a, b) == "better"
    assert compare(b, a) == "worse"
    assert dominates(a, b)

    # equal on every dimension: no strict improvement anywhere
    tie = {"x": _score(70.0, 60.0, 80.0), "y": _score(50.0, 40.0, 60.0)}
    assert compare(tie, b) == "not_comparable"

    # better on one, worse on another
    mixed = {"x": _score(90.0, 80.0, 100.0), "y": _score(40.0, 30.0, 50.0)}
    assert compare(mixed, a) == "not_comparable"
    assert compare(a, mixed) == "not_comparable"

    # an UNKNOWN dimension cannot establish dominance
    unknown = {"x": _score(80.0, 70.0, 90.0), "y": _score(None, 0.0, 100.0)}
    assert compare(a, unknown) == "not_comparable"
    assert compare(unknown, b) == "not_comparable"


def test_pareto_matrix_marks_undominated_pairs() -> None:
    rows = {
        "a": {"x": _score(80.0, 70.0, 90.0), "y": _score(60.0, 50.0, 70.0)},
        "b": {"x": _score(70.0, 60.0, 80.0), "y": _score(50.0, 40.0, 60.0)},
        "c": {"x": _score(90.0, 80.0, 100.0), "y": _score(40.0, 30.0, 50.0)},
    }
    matrix = pareto_matrix(rows)
    assert matrix["a"]["undominated"] is True
    assert matrix["b"]["undominated"] is False
    assert matrix["b"]["worse_than"] == ["a"]
    assert matrix["c"]["better_than"] == []
    assert matrix["c"]["not_comparable"] == ["a", "b"]
    assert matrix["c"]["undominated"] is True
