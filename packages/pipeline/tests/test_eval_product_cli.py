"""CLI wiring tests for the three eval kinds (offline; mock router only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmd.cli import main

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS = REPO_ROOT / "reports"


def test_eval_still_requires_golden_for_pipeline_kind(capsys) -> None:
    assert main(["eval"]) == 2
    assert "--golden is required" in capsys.readouterr().err


def test_eval_unknown_stream_is_a_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["eval", "--eval-kind", "product", "--stream", "does-not-exist"])
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "--eval-kind", "product", "--stream", "nirf-proxy", "--dry-run"],
        ["eval", "--eval-kind", "product", "--stream", "naac-proxy", "--dry-run"],
        ["eval", "--eval-kind", "product", "--stream", "self-drift", "--dry-run"],
        ["eval", "--eval-kind", "product", "--stream", "cv-stability", "--dry-run"],
        ["eval", "--eval-kind", "product", "--stream", "expert-panel", "--dry-run"],
        ["eval", "--eval-kind", "verdict", "--stream", "verdict-fast", "--dry-run"],
        [
            "eval", "--eval-kind", "verdict", "--stream", "verdict-fast", "--dry-run",
            "--fixture-path",
            str(REPO_ROOT / "packages/pipeline/tests/fixtures/eval_verdict/profiles"),
        ],
    ],
)
def test_all_streams_parse_and_run_offline(argv: list[str], capsys) -> None:
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert "fmd eval" in captured.out


def test_nirf_proxy_writes_report(tmp_path: Path, capsys) -> None:
    report_path = tmp_path / "eval-nirf-proxy.json"
    exit_code = main([
        "eval", "--eval-kind", "product", "--stream", "nirf-proxy",
        "--report-out", str(report_path),
    ])
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["eval_kind"] == "product" and report["stream"] == "nirf-proxy"
    assert report["truth"]["truth_kind"] == "nirf_ranking"
    case = report["cases"][0]
    assert case["top_k_overlap"]["3"] >= 1
    assert "proxy_label" in report and report["proxy_label"]
    assert case["unknown"]["fraction"] > 0


def test_verdict_fast_writes_report_with_audit_trail(tmp_path: Path, capsys) -> None:
    report_path = tmp_path / "eval-verdict-fast.json"
    exit_code = main([
        "eval", "--eval-kind", "verdict", "--stream", "verdict-fast",
        "--report-out", str(report_path),
    ])
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["aggregate"]["n_profiles"] == 5
    assert report["aggregate"]["audit_failures"] == 0
    for case in report["cases"]:
        for pointer in case["audit"]["evidence"]:
            assert pointer["url"] and len(pointer["sha256"]) == 64 and pointer["quote"]


def test_verdict_kind_rejects_unbuilt_slow_stream(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["eval", "--eval-kind", "verdict", "--stream", "verdict-deep"])
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_pipeline_golden_gate_still_passes(capsys) -> None:
    exit_code = main(["eval", "--golden", str(REPO_ROOT / "data" / "golden")])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "provenance coverage: 1.0" in captured.out
    assert "gate provenance>=1.0: PASS" in captured.out
