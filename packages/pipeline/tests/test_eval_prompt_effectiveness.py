"""Prompt-effectiveness stream tests (offline; committed canary fixture + truth).

The product's top-3 (the prompt-driven shortlist order) is compared against
the nirf-proxy / naac-proxy data-baseline top-3s on the SAME shortlist.
Hand-computed expectations for the committed canary (verified against
data/truth by_key()/ranked_keys() before commit):
product top-3 = [aims, pgimer, city], NIRF baseline top-3 = [aims, pgimer]
(NIRF ranks 1-2, restricted to the shortlist), NAAC truth is empty so its
baseline top-3 is empty: overlap_nirf_top3 = 2, overlap_naac_top3 = 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from fmd.cli import main
from fmd.eval.product import (
    Shortlist,
    ShortlistRow,
    load_shortlist,
    load_truth_for_stream,
    prompt_effectiveness_aggregate,
    run_prompt_effectiveness,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eval_product"

KEY_AIMS = "all india institute of medical sciences delhi"
KEY_PGIMER = "post graduate institute of medical education and research"
KEY_CITY = "city care paramedical college hyderabad"  # in neither truth table


def _canary_case() -> dict:
    shortlist = load_shortlist(FIXTURES / "prompt_effectiveness_fixture.json")
    nirf_truth = load_truth_for_stream("nirf-proxy", REPO_ROOT / "data" / "truth", root=REPO_ROOT)
    naac_truth = load_truth_for_stream("naac-proxy", REPO_ROOT / "data" / "truth", root=REPO_ROOT)
    return run_prompt_effectiveness(shortlist, nirf_truth, naac_truth)


def test_run_prompt_effectiveness_expected_overlaps() -> None:
    case = _canary_case()
    assert case["profile_id"] == "fixture-mlt-prompt-effectiveness-v1"
    assert case["shortlist_rows"] == 3
    assert case["product_top3"] == [KEY_AIMS, KEY_PGIMER, KEY_CITY]
    # NIRF baseline: truth top-2 restricted to this shortlist (AIMS rank 1, PGIMER rank 2)
    assert case["nirf_proxy_top3"] == [KEY_AIMS, KEY_PGIMER]
    assert case["naac_proxy_top3"] == []  # NAAC table is empty in the committed data
    assert case["overlap_nirf_top3"] == 2  # {AIMS, PGIMER} shared
    assert case["overlap_naac_top3"] == 0
    assert case["product_top_in_nirf"] is True
    assert case["product_top_in_naac"] is False
    assert case["product_top_unknown"] is False
    aggregate = prompt_effectiveness_aggregate([case])
    assert aggregate["n_profiles"] == 1
    assert aggregate["mean_overlap_nirf"] == 2
    assert aggregate["mean_overlap_naac"] == 0
    assert aggregate["n_profiles_overlap_ge2"] == 1
    assert aggregate["prompting_working"] is True
    assert aggregate["product_top_unknown_frac"] == 0.0
    assert aggregate["n_profiles_top_in_nirf"] == 1
    assert aggregate["n_profiles_top_in_naac"] == 0


def test_product_top_unknown_when_not_in_truth() -> None:
    shortlist = Shortlist(
        profile_id="synthetic-unknown-top",
        rows=[
            ShortlistRow(
                rank=1,
                institution_name="Nowhere Institute of Allied Health Sciences",
                programme="B.Sc. Medical Laboratory Technology",
                state="Telangana",
            )
        ],
    )
    nirf_truth = load_truth_for_stream("nirf-proxy", REPO_ROOT / "data" / "truth", root=REPO_ROOT)
    naac_truth = load_truth_for_stream("naac-proxy", REPO_ROOT / "data" / "truth", root=REPO_ROOT)
    case = run_prompt_effectiveness(shortlist, nirf_truth, naac_truth)
    assert case["product_top3"] == ["nowhere institute of allied health sciences"]
    assert case["nirf_proxy_top3"] == [] and case["naac_proxy_top3"] == []
    assert case["overlap_nirf_top3"] == 0
    assert case["overlap_naac_top3"] == 0
    # UNKNOWN is a value (invariant 5): no truth entry anywhere, never a coerced match
    assert case["product_top_in_nirf"] is False
    assert case["product_top_in_naac"] is False
    assert case["product_top_unknown"] is True
    aggregate = prompt_effectiveness_aggregate([case])
    assert aggregate["n_profiles_overlap_ge2"] == 0
    assert aggregate["prompting_working"] is False
    assert aggregate["product_top_unknown_frac"] == 1.0


def test_run_is_deterministic() -> None:
    first = prompt_effectiveness_aggregate([_canary_case()])
    second = prompt_effectiveness_aggregate([_canary_case()])
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_cli_runs_stream(tmp_path: Path, capsys) -> None:
    report_path = tmp_path / "eval-prompt-effectiveness.json"
    exit_code = main([
        "eval", "--eval-kind", "product", "--stream", "prompt-effectiveness",
        "--report-out", str(report_path),
    ])
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["eval_kind"] == "product" and report["stream"] == "prompt-effectiveness"
    assert report["aggregate"]["n_profiles"] == 1
    assert report["aggregate"]["prompting_working"] is True
    assert report["cases"][0]["overlap_nirf_top3"] == 2
    assert "fmd eval" in capsys.readouterr().out
    # --dry-run renders the summary without writing a report
    assert main([
        "eval", "--eval-kind", "product", "--stream", "prompt-effectiveness", "--dry-run",
    ]) == 0
    assert "prompting_working" in capsys.readouterr().out
