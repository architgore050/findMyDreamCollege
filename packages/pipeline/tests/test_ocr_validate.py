"""``scripts/ocr_validate.py`` — offline tests for the metrics and the gates.

The measurement itself runs on the GPU host against real scans (runbook §3.6);
these tests pin the metric behaviour, the ground-truth gate and the batch cap
using committed PDF fixtures, with engine runs faked.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ocr_validate.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("fmd_ocr_validate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("fmd_ocr_validate", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ov() -> Any:
    return _load_module()


@pytest.fixture
def sample_pdf() -> bytes:
    return (Path(__file__).parent / "fixtures" / "extract" / "placement_report.pdf").read_bytes()


# --------------------------------------------------------------------------- metrics


def test_cer_exact_match_is_zero_and_disjoint_is_one(ov: Any) -> None:
    assert ov.cer("कुल शुल्क 50000", "कुल शुल्क 50000") == 0.0
    assert ov.cer("abcdefghij", "0123456789") == 1.0
    assert ov.cer("", "") == 0.0
    assert ov.cer("", "x") == 1.0


def test_cer_counts_edit_distance(ov: Any) -> None:
    # one substitution in ten characters
    assert ov.cer("abcdefghij", "abcdefgXij") == pytest.approx(0.1)


def test_chrf_exact_is_one_and_disjoint_is_zero(ov: Any) -> None:
    assert ov.chrf_plus_plus("कुल शुल्क ₹५०,००० प्रति वर्ष", "कुल शुल्क ₹५०,००० प्रति वर्ष") == 1.0
    # no shared characters at all (not even whitespace)
    assert ov.chrf_plus_plus("कुलशुल्क", "totalfee") == 0.0


def test_chrf_orders_hypotheses_monotonically(ov: Any) -> None:
    ref = "Eligible students 120 placed 96 median salary 420000"
    perfect = ov.chrf_plus_plus(ref, ref)
    typo = ov.chrf_plus_plus(ref, "Eligible students 120 placed 96 median salary 42000")
    garbage = ov.chrf_plus_plus(ref, "zzz qqq www")
    assert perfect > typo > garbage >= 0.0


def test_chrf_mojibake_scores_low_correct_hindi_high(ov: Any) -> None:
    """The metric must separate a mojibake decoder from a correct Hindi read."""
    ref = "कुल शुल्क ₹५०,००० प्रति वर्ष और पात्रता की जानकारी"
    correct = "कुल शुल्क ₹50,000 प्रति वर्ष और पात्रता की जानकारी"
    mojibake = "à¤•à¥ à¤² à¤¶à¥à¤²à¥ à¤° à¥ªà¥¦ à¥¦à¥¦à¥¦ à¤ªà¥à¤°à¤¤à¤¿ à¤µà¤°à¥à¤·"
    assert ov.chrf_plus_plus(ref, correct) > 0.75
    assert ov.chrf_plus_plus(ref, mojibake) < 0.2


def test_baseline_constants_are_committed(ov: Any) -> None:
    """The measured baseline (2026-11-19) ships with the script, not from memory."""
    assert ov.BASELINE["cer"] == 0.109
    assert ov.BASELINE["pages_per_min"] == 25.8
    assert ov.BASELINE["seconds_per_page"] == 2.33
    assert ov.BASELINE["device"] == "cpu"


def test_ground_truth_seed_is_doc_61(ov: Any) -> None:
    assert ov.GROUND_TRUTH_DOCS[0]["id"] == 61
    assert list(ov.GROUND_TRUTH_DOCS[0]["pages"]) == [3, 5, 7]


# --------------------------------------------------------------------------- gates


def test_fewer_than_three_documents_exits_nonzero(
    ov: Any, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """< 3 documents with ground truth ⇒ non-zero, the report still prints."""
    docs = [
        ov.GroundTruthDoc(doc_id=1, name="a", pages=(1,), data=b""),
        ov.GroundTruthDoc(doc_id=2, name="b", pages=(1,), data=b""),
    ]
    monkeypatch.setattr(ov, "load_ground_truth_docs", lambda **kwargs: docs)
    rc = ov.main(["--ground-truth", "--json"])
    assert rc == 1


def test_over_ten_documents_is_refused_without_i_know(ov: Any, monkeypatch) -> None:
    docs = [
        ov.GroundTruthDoc(doc_id=i, name=f"doc{i}", pages=(1,), data=b"x")
        for i in range(11)
    ]
    monkeypatch.setattr(ov, "load_ground_truth_docs", lambda **kwargs: docs)
    with pytest.raises(SystemExit) as excinfo:
        ov.main(["--ground-truth", "--json"])
    assert "--i-know" in str(excinfo.value)


def test_full_measurement_against_a_real_text_layer(
    ov: Any, monkeypatch: pytest.MonkeyPatch, sample_pdf: bytes
) -> None:
    """A cheating engine that returns the real text layer scores CER 0 / chrF 1."""
    doc = ov.GroundTruthDoc(
        doc_id=None,
        name="placement_report.pdf",
        pages=(1,),
        data=sample_pdf,
    )
    monkeypatch.setattr(ov, "load_ground_truth_docs", lambda **kwargs: [doc])

    class CheatingEngine:
        name = "rapidocr"

        def run(self, data: bytes, pages: Any) -> list:
            results = []
            for page_no in pages:
                text = ov._page_text_layer(data, page_no)
                results.append(ov.PageResult(page_no=page_no, text=text, seconds=0.05))
            return results

    class EmptyEngine:
        name = "qwen3vl"

        def run(self, data: bytes, pages: Any) -> list:
            return [
                ov.PageResult(page_no=p, text="", seconds=0.01) for p in pages
            ]

    monkeypatch.setattr(
        ov,
        "build_engines",
        lambda requested: ({"rapidocr": CheatingEngine(), "qwen3vl": EmptyEngine()}, {}),
    )

    args = argparse_namespace()
    report = ov.run_measurement(args)
    payload = json.loads(json.dumps(report))
    stats = payload["aggregate"]
    assert stats["rapidocr"]["cer_mean"] == 0.0
    assert stats["rapidocr"]["chrf_pp_mean"] == 1.0
    assert stats["rapidocr"]["pages_per_min_mean"] > 0
    assert stats["qwen3vl"]["cer_mean"] == 1.0
    assert stats["qwen3vl"]["chrf_pp_mean"] == 0.0
    # The committed decision rule inputs are reported for the measured gap.
    assert payload["decision_rule"]["chrf_pp_gap_rapidocr_minus_qwen3vl"] == 1.0
    assert payload["baseline"]["pages_per_min"] == 25.8
    assert payload["documents_with_ground_truth"] == 1


def test_explicit_doc_runs_without_the_ground_truth_flag(
    ov: Any, monkeypatch: pytest.MonkeyPatch, sample_pdf: bytes, capsys
) -> None:
    """``--engine … --doc … --pages …`` is a run, not an argparse help dump."""
    doc = ov.GroundTruthDoc(
        doc_id=61, name="doc 61", pages=(1,), data=sample_pdf
    )
    monkeypatch.setattr(ov, "load_ground_truth_docs", lambda **kwargs: [doc])

    class Engine:
        name = "rapidocr"

        def run(self, data: bytes, pages: Any) -> list:
            return [
                ov.PageResult(
                    page_no=p, text=ov._page_text_layer(data, p), seconds=0.01
                )
                for p in pages
            ]

    monkeypatch.setattr(ov, "build_engines", lambda requested: ({"rapidocr": Engine()}, {}))
    rc = ov.main(["--engine", "rapidocr", "--doc", "61", "--pages", "1", "--json"])
    out = capsys.readouterr()
    assert rc == 0
    assert '"documents_with_ground_truth": 1' in out.out
    assert "usage:" not in out.out
    assert "single-document run" in out.err


def test_no_mode_still_prints_help(ov: Any, capsys) -> None:
    rc = ov.main([])
    out = capsys.readouterr()
    assert rc == 2
    assert "usage:" in out.out


def argparse_namespace() -> Any:
    import argparse

    return argparse.Namespace(
        ground_truth=True,
        engine="all",
        limit=None,
        doc=None,
        pages=None,
        documents=None,
        bytes_root=None,
        models_root=None,
        i_know=True,
        json=True,
    )
