"""Expert-panel kit tests, including verification against Krippendorff (2011)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fmd.eval.panel import (
    PanelError,
    icc_two_way_single,
    krippendorff_alpha,
    load_panel_ratings,
    load_product_ranking,
    run_expert_panel,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eval_panel"


# --------------------------------------------------------------- paper verification


def test_alpha_reproduces_krippendorff_2011_binary_example() -> None:
    # Paper §A: 2 observers, 10 units, 4 mismatching -> alpha = 0.095
    # (n_0=14, n_1=6, n=20; reconstructed as 5 [0,0], 4 [0,1], 1 [1,1])
    units = [[0.0, 0.0]] * 5 + [[0.0, 1.0]] * 4 + [[1.0, 1.0]]
    assert krippendorff_alpha(units, metric="nominal") == 0.095238


def test_alpha_reproduces_krippendorff_2011_four_observer_example() -> None:
    # Paper §C: 4 observers x 12 units with missing values -> nominal alpha = 0.743
    a = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    b = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
    c = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    d = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
    units = [[a[i], b[i], c[i], d[i]] for i in range(12)]
    assert krippendorff_alpha(units, metric="nominal") == 0.743421


def test_alpha_properties() -> None:
    perfect = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert krippendorff_alpha(perfect, metric="nominal") == 1.0
    assert krippendorff_alpha(perfect, metric="ordinal") == 1.0
    # total disagreement among two raters on binary values: D_o=1.0, D_e=4/7
    disagree = [[1.0, 2.0], [2.0, 1.0], [1.0, 2.0], [2.0, 1.0]]
    assert krippendorff_alpha(disagree, metric="nominal") == pytest.approx(-0.75, abs=1e-6)
    # fewer than two pairable units is unmeasurable, never 1.0
    assert krippendorff_alpha([[1.0, None], [2.0, None]], metric="nominal") is None
    assert krippendorff_alpha([[1.0], [2.0]], metric="nominal") is None


def test_icc_two_way_single_on_known_matrices() -> None:
    perfect = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert icc_two_way_single(perfect) == 1.0
    # noisy matrix: ICC in (0, 1); a constant-rating matrix is unmeasurable
    noisy = [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 4.0]]
    value = icc_two_way_single(noisy)
    assert value is not None and -1.0 <= value <= 1.0
    assert icc_two_way_single([[5.0, 5.0], [5.0, 5.0]]) is None


# --------------------------------------------------------------- fixture stream


def test_panel_stream_on_committed_fixture() -> None:
    ratings = load_panel_ratings(FIXTURES)
    ranking = load_product_ranking(FIXTURES)
    case = run_expert_panel(ratings, ranking)
    reliability = case["inter_rater_reliability"]
    # values the canary report prints; all in [-1, 1] and interpretable
    assert reliability["krippendorff_alpha_nominal"] == pytest.approx(0.222222, abs=1e-5)
    assert reliability["krippendorff_alpha_ordinal"] == pytest.approx(0.604743, abs=1e-5)
    assert -1.0 <= reliability["icc_two_way_single"] <= 1.0
    assert reliability["pairwise_kendall_tau_b_mean"] == pytest.approx(0.222222, abs=1e-5)
    effectiveness = case["effectiveness"]
    assert effectiveness["mean"] == pytest.approx(2 / 3, abs=1e-5)
    # n=1 profile: an honest report refuses to print a bootstrap interval
    assert effectiveness["ci95_low"] is None and effectiveness["ci95_high"] is None
    assert "n=1" in effectiveness["bootstrap"]["note"]
    profile_case = case["cases"][0]
    assert profile_case["top3_overlap"] == 2
    assert profile_case["kendall_tau_b"] == pytest.approx(1 / 3, abs=1e-5)
    # audit protocol provenance survived the fixture round-trip
    assert all(rater["conflict_disclosure"] for rater in case["raters"])


def test_panel_fixture_without_conflict_disclosure_is_rejected(tmp_path: Path) -> None:
    text = (FIXTURES / "panel_ratings.yaml").read_text(encoding="utf-8")
    text = text.replace('conflict_disclosure: "none"', "conflict_disclosure: \"\"")
    (tmp_path / "panel_ratings.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(PanelError, match="conflict_disclosure"):
        load_panel_ratings(tmp_path)


def test_product_ranking_must_cover_every_profile(tmp_path: Path) -> None:
    ratings = load_panel_ratings(FIXTURES)
    (tmp_path / "panel_ratings.yaml").write_text(
        (FIXTURES / "panel_ratings.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "product_ranking.json").write_text(
        '{"schema_version": 1, "profiles": []}', encoding="utf-8"
    )
    ranking = load_product_ranking(tmp_path)
    with pytest.raises(PanelError, match="no entry for this profile"):
        run_expert_panel(ratings, ranking)
