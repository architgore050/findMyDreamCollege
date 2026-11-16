from __future__ import annotations

from fmd.eval.drift import (
    compare_reports,
    ks_two_sample,
    population_stability_index,
    top_n_jaccard,
    unexplained_changes,
)


def test_identical_distributions_are_stable() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert population_stability_index(values, values) == 0.0
    statistic, p_value = ks_two_sample(values, values)
    assert statistic == 0.0
    assert p_value == 1.0


def test_shifted_distribution_is_detected() -> None:
    baseline = [float(value) for value in range(0, 200, 5)]
    current = [float(value + 100) for value in range(0, 200, 5)]
    psi = population_stability_index(baseline, current)
    statistic, p_value = ks_two_sample(baseline, current)

    assert psi > 0.25
    assert statistic > 0.3
    assert p_value < 0.05


def test_top_n_jaccard_measures_churn() -> None:
    assert top_n_jaccard(["a", "b", "c"], ["a", "b", "c"], n=3) == 1.0
    assert top_n_jaccard(["a", "b", "c"], ["x", "y", "z"], n=3) == 0.0
    assert top_n_jaccard(["a", "b", "c", "d"], ["a", "b", "x", "d"], n=4) == 0.6


def _report(claims_hash: str, f1: float, values: list[float]) -> dict:
    return {
        "versions": {"scorer_version": "0.2.0", "weights_version": "2026-11-16.2"},
        "extraction": {"f1": f1},
        "canaries": {"canary-1": {"claims_hash": claims_hash, "extraction": {"f1": f1}}},
        "sub_scores": {
            f"entity-{index}": {"outcomes": value} for index, value in enumerate(values)
        },
        "ranking": ["a", "b", "c"],
    }


def test_canary_metric_move_with_a_claims_diff_is_explained() -> None:
    baseline = _report("hash-a", 1.0, [10.0, 20.0, 30.0])
    current = _report("hash-b", 0.5, [10.0, 20.0, 30.0])
    drift = compare_reports(baseline, current)

    assert drift["unexplained"] is False
    assert drift["canaries"]["explained"][0]["explained_by"] == "claims_diff"
    assert drift["canaries"]["unexplained"] == []


def test_canary_metric_move_without_a_claims_diff_is_flagged() -> None:
    baseline = _report("same", 1.0, [10.0, 20.0, 30.0])
    current = _report("same", 0.5, [10.0, 20.0, 30.0])
    drift = compare_reports(baseline, current)

    assert drift["unexplained"] is True
    assert drift["canaries"]["unexplained"][0]["canary"] == "canary-1"
    assert unexplained_changes(drift)


def test_deliberately_shifted_sub_scores_are_flagged_without_a_version_change() -> None:
    baseline = _report("same", 1.0, [10.0, 20.0, 30.0, 40.0, 50.0])
    current = _report("same", 1.0, [60.0, 70.0, 80.0, 90.0, 100.0])
    drift = compare_reports(baseline, current)

    assert drift["unexplained"] is True
    assert drift["sub_scores"]["outcomes"]["psi"] > 0.25
    assert drift["sub_scores"]["outcomes"]["ks_p_value"] < 0.05
    assert drift["sub_scores"]["outcomes"]["explained_by"] == "claims_diff_or_unexplained"
