"""Evaluation harness — golden set, extraction metrics, provenance gate, drift.

Public entrypoints: :func:`fmd.eval.run.run_evaluation` and the ``fmd eval``
command. The harness is offline and deterministic; it never calls a model or the
network, and it never asserts a value without a committed fixture or a cited,
dated source (see :mod:`fmd.eval.golden`).
"""

from __future__ import annotations

from fmd.eval.drift import compare_reports, ks_two_sample, population_stability_index
from fmd.eval.golden import GoldenEntry, GoldenError, GoldenSet, GoldenValue, load_golden
from fmd.eval.metrics import (
    ExtractionMetrics,
    IntervalReport,
    ProvenanceReport,
    RejectionMetrics,
    provenance_coverage,
)
from fmd.eval.run import (
    DEFAULT_FAIL_UNDER,
    EvalError,
    EvalOutcome,
    apply_gate,
    render_summary,
    run_evaluation,
    write_report,
)

__all__ = [
    "DEFAULT_FAIL_UNDER",
    "EvalError",
    "EvalOutcome",
    "ExtractionMetrics",
    "GoldenEntry",
    "GoldenError",
    "GoldenSet",
    "GoldenValue",
    "IntervalReport",
    "ProvenanceReport",
    "RejectionMetrics",
    "apply_gate",
    "compare_reports",
    "ks_two_sample",
    "load_golden",
    "population_stability_index",
    "provenance_coverage",
    "render_summary",
    "run_evaluation",
    "write_report",
]
