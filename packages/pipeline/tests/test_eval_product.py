"""Product-stream tests: proxy ranking, self-drift, cv-stability (offline).

The committed canary fixture + committed truth tables give the constructed
case with top-k overlap >= 1 and UNKNOWN handling; drift and cv-stability run
on committed synthetic snapshots.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fmd.eval.product import (
    ShortlistError,
    load_shortlist,
    run_cv_stability,
    run_naac_proxy,
    run_nirf_proxy,
    run_self_drift,
)
from fmd.eval.truth import load_truth_file

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eval_product"
WEIGHTS = {
    "outcomes": 30.0,
    "affordability_roi": 20.0,
    "clinical_training": 15.0,
    "academic_quality": 10.0,
    "student_life_safety": 10.0,
    "admission_realism": 10.0,
    "financing_feasibility": 5.0,
}


@pytest.fixture
def shortlist():
    return load_shortlist(FIXTURES / "shortlist_fixture.json")


def test_nirf_proxy_constructed_case_top3_overlap_at_least_one(shortlist) -> None:
    truth = load_truth_file(REPO_ROOT / "data" / "truth" / "nirf" / "nirf_2025_medical.yaml")
    case = run_nirf_proxy(shortlist, truth)
    assert case["top_k_overlap"]["3"] >= 1
    assert case["top_k_overlap"]["5"] == 3
    # the fixture's #1 is deliberately unmatched: UNKNOWN, not a miss
    assert case["best_available"] == {
        "hit": None,
        "product_top_institution": "city care paramedical college hyderabad",
        "scored": False,
    }
    assert case["rank_correlation"]["n_pairs"] == 3
    assert case["rank_correlation"]["spearman_rho"] is not None
    assert case["unknown"]["fraction"] == 0.4
    assert "no paramedical" in case["proxy_label"]


def test_naac_proxy_on_empty_truth_is_all_unknown(shortlist) -> None:
    truth = load_truth_file(REPO_ROOT / "data" / "truth" / "naac" / "naac_grades.yaml")
    case = run_naac_proxy(shortlist, truth)
    assert case["unknown"]["fraction"] == 1.0
    assert case["top_k_overlap"]["3"] == 0
    assert case["rank_correlation"]["spearman_rho"] is None
    assert case["best_available"]["hit"] is None
    assert "not programme-level outcome measures" in case["proxy_label"]


def test_self_drift_identical_snapshots_produce_zero_changes() -> None:
    snapshot = load_shortlist(FIXTURES / "snapshot_previous.json")
    report = run_self_drift(snapshot, snapshot)
    assert report["n_changes"] == 0
    assert report["unexplained_count"] == 0
    assert report["evidence_driven_fraction"] is None
    assert report["top10_churn"] == 0.0


def test_self_drift_labels_claims_diff_vs_unexplained() -> None:
    current = load_shortlist(FIXTURES / "snapshot_current.json")
    previous = load_shortlist(FIXTURES / "snapshot_previous.json")
    report = run_self_drift(current, previous)
    assert report["n_changes"] == 2
    by_institution = {change["institution"]: change for change in report["changes"]}
    gamma = by_institution["gamma school of paramedical studies"]
    assert gamma["explained_by"] == "claims_diff" and gamma["magnitude"] == 1
    beta = by_institution["beta college of health sciences"]
    assert beta["explained_by"] is None  # rank moved, claims unchanged: unexplained
    assert report["unexplained_count"] == 1
    assert report["evidence_driven_fraction"] == 0.5
    assert report["rank_changes"]["max_abs"] == 1


def test_self_drift_refuses_cross_profile_diff() -> None:
    current = load_shortlist(FIXTURES / "snapshot_current.json")
    previous = load_shortlist(FIXTURES / "snapshot_previous.json")
    object.__setattr__(current, "profile_id", "other-profile")
    with pytest.raises(ShortlistError, match="compares one profile"):
        run_self_drift(current, previous)


def test_cv_stability_numbers_stay_in_unit_interval() -> None:
    candidates = load_shortlist(FIXTURES / "cv_candidates.json")
    case = run_cv_stability(candidates, WEIGHTS)
    for holdout in case["leave_n_out"]:
        assert holdout["kendall_tau_b"] is not None
        assert 0.0 <= holdout["kendall_tau_b"] <= 1.0
    noise = case["weight_noise"]
    for value in (
        noise["kendall_tau_b_mean"],
        noise["kendall_tau_b_worst"],
        noise["top_k_survival_mean"],
        noise["top_k_survival_worst"],
        noise["heldout_in_topk_rate"],
    ):
        assert value is not None and 0.0 <= value <= 1.0
    # additive scoring: holding rows out cannot reorder survivors (honest note)
    assert all(holdout["kendall_tau_b"] == 1.0 for holdout in case["leave_n_out"])
    # the fixture's tail pair flips under some weight variants: worst < mean
    assert noise["kendall_tau_b_worst"] < noise["kendall_tau_b_mean"]
    assert noise["top_k_survival_mean"] == 1.0  # top-5 is robust in this set
    assert case["measures"].startswith("internal consistency only")


def test_cv_stability_needs_three_scored_rows() -> None:
    tiny = load_shortlist(FIXTURES / "shortlist_fixture.json")
    with pytest.raises(ShortlistError, match=">= 3 rows with sub_scores"):
        run_cv_stability(tiny, WEIGHTS)


def test_shortlist_loader_rejects_rank_gaps(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"schema_version": 1, "profile_id": "p", "rows": ['
        '{"rank": 1, "institution_name": "A Institute"},'
        '{"rank": 3, "institution_name": "B Institute"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ShortlistError, match="1..N series"):
        load_shortlist(bad)
