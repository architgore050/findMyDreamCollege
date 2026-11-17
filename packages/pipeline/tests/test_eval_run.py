from __future__ import annotations

import json
from pathlib import Path

from fmd.cli import main
from fmd.eval.golden import DEFAULT_GOLDEN_DIR
from fmd.eval.run import apply_gate, run_evaluation


def test_full_offline_eval_run_is_deterministic() -> None:
    first = run_evaluation(DEFAULT_GOLDEN_DIR)
    second = run_evaluation(DEFAULT_GOLDEN_DIR)

    assert first.report == second.report
    report = first.report
    assert report["ok"] is True
    assert report["golden"]["entries"] == 5
    assert report["golden"]["verified_rows"] == 16
    assert report["golden"]["unverified_rows"] == 1
    assert report["golden"]["row_context_rows"] == 1
    assert report["golden"]["row_context_verified_rows"] == 1
    assert report["golden"]["measured_entries"] == 1
    assert report["golden"]["unmeasured_verified_rows"] == 8
    assert report["extraction"]["evaluated"] == 8
    assert report["extraction"]["f1"] == 1.0
    assert report["extraction"]["unit_normalisation_accuracy"] == 1.0
    assert report["extraction"]["abstention_correctness"] == 1.0
    assert report["extraction"]["unverified_excluded"] == 1
    assert report["extraction"]["row_context_rows"] == 0
    assert report["extraction"]["row_attribution_checks"] == 0
    assert report["extraction"]["row_attribution_accuracy"] is None
    assert report["extraction"]["wrong_attributions"] == 0
    assert report["rejection"]["expected_rejects"] == 0
    assert report["provenance"]["coverage"] == 1.0
    assert report["provenance"]["total"] == 4
    assert report["provenance"]["resolved"] == 4
    assert report["interval_coverage"]["measurable"] is False
    assert report["golden_traceability"]["citation_only"] == 0


def test_provenance_gate_controls_exit_code(tmp_path: Path, capsys) -> None:
    report_path = tmp_path / "eval.json"
    argv = [
        "eval",
        "--golden",
        str(DEFAULT_GOLDEN_DIR),
        "--report-out",
        str(report_path),
    ]
    assert main(argv) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["gates"]["provenance_coverage"]["passed"] is True

    assert main([*argv, "--fail-under", "1.01"]) == 1
    captured = capsys.readouterr()
    assert "below --fail-under" in captured.err
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False


def test_provenance_gate_failure_exits_nonzero(monkeypatch, tmp_path: Path) -> None:
    import fmd.eval as eval_package
    from fmd.eval.run import EvalOutcome

    report = {
        "extraction": {},
        "provenance": {"coverage": 0.5, "resolved": 1, "total": 2},
        "golden": {"entries": 1, "verified_rows": 2, "unverified_rows": 0},
        "interval_coverage": {"n": 0, "coverage": None},
        "gates": {},
    }
    monkeypatch.setattr(
        eval_package, "run_evaluation", lambda *args, **kwargs: EvalOutcome(report=report)
    )
    assert main(["eval", "--golden", str(tmp_path)]) == 1


def test_baseline_report_produces_drift_block(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"extraction": {"f1": 0.5}, "canaries": {}, "versions": {}}),
        encoding="utf-8",
    )
    outcome = run_evaluation(DEFAULT_GOLDEN_DIR, baseline_path=baseline_path)
    assert outcome.report["drift"] is not None
    assert outcome.report["drift"]["metric_changes"]


def test_gate_reports_failure_for_low_coverage() -> None:
    report = {"provenance": {"coverage": 0.5}}
    failures = apply_gate(report, 1.0)
    assert failures
    assert report["gates"]["provenance_coverage"]["passed"] is False


def test_eval_without_golden_is_still_a_stub(capsys) -> None:
    assert main(["eval"]) == 2
    captured = capsys.readouterr()
    assert "not implemented (WS-F," in captured.err
