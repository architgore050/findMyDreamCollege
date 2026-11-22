"""Truth-table loader + metric tests (docs/v1/eval/03-truth-ingestion.md contract).

Offline: exercises the committed NIRF/NAAC truth files, the sha256
re-verification, the ingestion-contract failure modes, and the pure ranking
metrics with hand-computable cases.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fmd.eval.truth import (
    NaacTruthFile,
    NirfTruthFile,
    TruthError,
    best_available_hit,
    kendall_tau_b,
    load_truth_file,
    normalise_institution_key,
    rank_correlation_on_overlap,
    spearman_rho,
    top_k_overlap,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
NIRF_MEDICAL = REPO_ROOT / "data" / "truth" / "nirf" / "nirf_2025_medical.yaml"
NIRF_OVERALL = REPO_ROOT / "data" / "truth" / "nirf" / "nirf_2025_overall.yaml"
NAAC = REPO_ROOT / "data" / "truth" / "naac" / "naac_grades.yaml"


def test_committed_nirf_truth_files_load_and_verify() -> None:
    medical = load_truth_file(NIRF_MEDICAL)
    overall = load_truth_file(NIRF_OVERALL)
    assert isinstance(medical, NirfTruthFile) and isinstance(overall, NirfTruthFile)
    assert len(medical.entries) == 50 and len(overall.entries) == 100
    assert medical.ranked_keys()[0] == "all india institute of medical sciences delhi"
    # provenance re-verified against the committed raw page copies
    assert medical.source_sha256 == medical.header.source_sha256
    assert medical.as_of == "2025" and medical.header.scope == "institution"
    # overall edition contains NIRF's tied ranks (two institutions share rank 27)
    ranks = [entry.rank for entry in overall.entries]
    assert len(ranks) != len(set(ranks))


def test_committed_naac_table_is_explicitly_uningested() -> None:
    naac = load_truth_file(NAAC)
    assert isinstance(naac, NaacTruthFile)
    assert naac.entries == []
    assert naac.header.truth_version == 0
    assert naac.header.acquisition_status == "not_ingested"
    assert naac.as_of is None


def test_sha256_mismatch_is_a_load_error(tmp_path: Path) -> None:
    # copy preserves the repo-relative layout so raw_file resolves under root
    copy_dir = tmp_path / "data" / "truth"
    shutil.copytree(REPO_ROOT / "data" / "truth", copy_dir)
    raw = copy_dir / "nirf" / "raw" / "2025_MedicalRanking.html"
    raw.write_bytes(raw.read_bytes() + b"tampered")
    with pytest.raises(TruthError, match="source_sha256 mismatch"):
        load_truth_file(copy_dir / "nirf" / "nirf_2025_medical.yaml", root=tmp_path)


def test_missing_raw_page_copy_is_a_load_error(tmp_path: Path) -> None:
    copy_dir = tmp_path / "data" / "truth"
    shutil.copytree(REPO_ROOT / "data" / "truth", copy_dir)
    (copy_dir / "nirf" / "raw" / "2025_MedicalRanking.html").unlink()
    with pytest.raises(TruthError, match="raw page copy missing"):
        load_truth_file(copy_dir / "nirf" / "nirf_2025_medical.yaml", root=tmp_path)


def test_wrong_join_key_is_a_load_error(tmp_path: Path) -> None:
    copy_dir = tmp_path / "data" / "truth"
    shutil.copytree(REPO_ROOT / "data" / "truth", copy_dir)
    truth_path = copy_dir / "nirf" / "nirf_2025_medical.yaml"
    text = truth_path.read_text(encoding="utf-8")
    truth_path.write_text(text.replace(
        "institution_key: all india institute of medical sciences delhi",
        "institution_key: aiims delhi",
    ), encoding="utf-8")
    with pytest.raises(TruthError, match="does not match normalise_institution_key"):
        load_truth_file(truth_path, root=tmp_path)


def test_version_zero_table_with_rows_contradicts_itself(tmp_path: Path) -> None:
    path = tmp_path / "naac_bad.yaml"
    path.write_text(
        "schema_version: 1\n"
        "truth_kind: naac_grades\n"
        "truth_version: 0\n"
        "scope: institution\n"
        "source_url: https://naac.gov.in/\n"
        "entries:\n"
        "  - institution_key: some college\n"
        "    grade: A\n",
        encoding="utf-8",
    )
    with pytest.raises(TruthError, match="entries must be empty"):
        load_truth_file(path, root=tmp_path)


def test_normalise_institution_key_rule() -> None:
    assert normalise_institution_key("M.S. Ramaiah Medical College") == (
        "m s ramaiah medical college"
    )
    assert normalise_institution_key("St. John's & Evangelist Institute") == (
        "st john s and evangelist institute"
    )
    assert normalise_institution_key("  ALL   INDIA Institute  ") == "all india institute"


def test_top_k_overlap_counts_shared_prefixes() -> None:
    # product top-3 {a,b,c} vs reference top-3 {c,b,x}: b and c are shared
    assert top_k_overlap(["a", "b", "c", "d"], ["c", "b", "x", "a"], 3) == 2
    assert top_k_overlap(["a", "b", "c"], ["a", "b", "c"], 3) == 3
    assert top_k_overlap([], ["a"], 3) == 0


def test_spearman_and_kendall_on_known_orders() -> None:
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    # one adjacent transposition of 4 items: 5 concordant, 1 discordant pair
    assert kendall_tau_b([1, 2, 3, 4], [2, 1, 3, 4]) == round(4 / 6, 6)
    assert spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    # fewer than two comparable pairs is unmeasurable, never 0
    assert spearman_rho([1], [1]) is None
    assert kendall_tau_b([1], [1]) is None


def test_spearman_handles_ties_with_average_ranks() -> None:
    # x has a tie: ranks [1.5, 1.5, 3]; Pearson on ranks against [1, 2, 3]
    assert spearman_rho([10, 10, 20], [1, 2, 3]) == 0.866025


def test_rank_correlation_on_overlap_excludes_unknown() -> None:
    correlation = rank_correlation_on_overlap(
        ["a", "b", "unknown-one"],
        ["a", "b"],
        product_positions={"a": 1, "b": 2, "unknown-one": 3},
        reference_positions={"a": 1, "b": 2},
    )
    assert correlation["n_pairs"] == 2 and correlation["kendall_tau_b"] == 1.0
    too_few = rank_correlation_on_overlap(
        ["a"], ["a"], product_positions={"a": 1}, reference_positions={"a": 1}
    )
    assert too_few == {"spearman_rho": None, "kendall_tau_b": None, "n_pairs": 1}


def test_best_available_hit_uses_reference_top3() -> None:
    assert best_available_hit(["a", "x"], ["a", "b", "c"]) is True
    assert best_available_hit(["d", "a"], ["a", "b", "c"]) is False
    assert best_available_hit([], ["a", "b", "c"]) is False
