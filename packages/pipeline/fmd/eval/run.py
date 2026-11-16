"""Offline evaluation runner (brief §19).

The runner replays the extraction *validation* path against committed fixtures:
recorded guided JSON -> family schema -> :func:`fmd.extract.validate.validate_document`
-> claims. No database, router, network or clock is involved. That means the
report is byte-for-byte reproducible for a fixed checkout.

It produces one JSON report with:

* golden-set verified/unverified counts (unverified rows never enter accuracy);
* extraction span P/R/F1, unit normalisation, abstention metrics;
* rejection correctness;
* provenance coverage (hard gate);
* interval coverage when interval predictions exist;
* per-entry canaries with a stable ``claims_hash``;
* drift vs ``--baseline`` when supplied.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fmd.eval.drift import compare_reports
from fmd.eval.golden import REPO_ROOT, GoldenEntry, GoldenSet, GoldenValue, load_golden
from fmd.eval.metrics import (
    GoldenRow,
    Prediction,
    PublishedNumber,
    interval_coverage,
    provenance_coverage,
    rejection_correctness,
    score_field,
    summarise_fields,
)
from fmd.extract.schemas import schema_for
from fmd.extract.validate import Bands, validate_document
from fmd.paths import PathResolutionError, resolve_config_dir
from fmd.score.evidence import claims_hash

SCHEMA_VERSION = 1
DEFAULT_FAIL_UNDER = 1.0


class EvalError(RuntimeError):
    """The evaluation cannot run with the given inputs."""


@dataclass
class EvalOutcome:
    report: dict[str, Any]
    report_path: Path | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _versions() -> dict[str, Any]:
    try:
        config_dir = resolve_config_dir()
        data = yaml.safe_load((config_dir / "weights.yaml").read_text(encoding="utf-8")) or {}
    except (PathResolutionError, OSError, yaml.YAMLError):
        return {"scorer_version": None, "weights_version": None}
    return {
        "scorer_version": data.get("scorer_version"),
        "weights_version": data.get("weights_version"),
    }


def _golden_row(name: str, row: GoldenValue) -> GoldenRow:
    return GoldenRow(
        field=name,
        value=row.value,
        unit=row.unit,
        quote=row.quote,
        status=row.status,
        expectation=row.expectation,
        printed_value=row.printed_value,
        printed_unit=row.printed_unit,
        reason=row.reason,
    )


def _entry_rows(entry: GoldenEntry) -> list[GoldenRow]:
    return [_golden_row(name, row) for name, row in entry.expected.items()]


def _load_recorded(entry: GoldenEntry, root: Path) -> tuple[str, dict[str, Any]]:
    if not entry.fixture or not entry.llm_fixture:
        raise EvalError(f"{entry.id}: extraction needs both fixture and llm_fixture")
    text_path = root / entry.fixture
    recorded_path = root / entry.llm_fixture
    if not text_path.is_file():
        raise EvalError(f"{entry.id}: text fixture not found: {entry.fixture}")
    if not recorded_path.is_file():
        raise EvalError(f"{entry.id}: recorded LLM fixture not found: {entry.llm_fixture}")
    payload = json.loads(recorded_path.read_text(encoding="utf-8"))
    content = payload.get("content") if isinstance(payload, Mapping) else None
    if not isinstance(content, Mapping):
        raise EvalError(f"{entry.id}: recorded fixture has no 'content' object")
    return text_path.read_text(encoding="utf-8"), dict(content)


def _evaluate_entry(
    entry: GoldenEntry, root: Path, *, bands: Bands, entity_id: int
) -> tuple[list[Any], str, dict[str, Any], list[Any]]:
    """Return ``(claims, text, per-entry extraction metrics, rejections)``."""

    if not entry.family:
        raise EvalError(f"{entry.id}: extraction entry has no family")
    text, content = _load_recorded(entry, root)
    model = schema_for(entry.family)
    try:
        document = model.model_validate(content)
    except Exception as exc:
        raise EvalError(f"{entry.id}: recorded content does not fit {entry.family}: {exc}") from exc
    validation = validate_document(
        document,
        text,
        extraction_id=0,
        entity_id=entity_id,
        bands=bands,
    )
    predictions = [
        Prediction(
            field=claim.predicate,
            value=claim.value,
            unit=claim.unit,
            quote=claim.quote,
            char_start=claim.char_start,
            char_end=claim.char_end,
            method=claim.method,
        )
        for claim in validation.claims
    ]
    rows = _entry_rows(entry)
    unverified = sum(1 for row in rows if row.status != "verified")
    results = [
        score_field(row, next((p for p in predictions if p.field == row.field), None), text)
        for row in rows
        if row.status == "verified" and row.expectation in {"value", "abstain"}
    ]
    entry_metrics = summarise_fields(results, unverified_excluded=unverified).as_dict()
    entry_metrics["claims"] = len(validation.claims)
    entry_metrics["rejections"] = len(validation.rejections)
    return validation.claims, text, entry_metrics, list(validation.rejections)


def _golden_traceability(golden: GoldenSet) -> dict[str, Any]:
    traces = golden.traces
    return {
        "verified_rows": golden.verified_rows,
        "unverified_rows": golden.unverified_rows,
        "fixture_verified": sum(1 for trace in traces if trace.traceability == "fixture"),
        "citation_only": sum(1 for trace in traces if trace.traceability == "citation"),
        "excluded": [trace.detail for trace in traces if trace.traceability == "unverified"],
    }


def run_evaluation(
    golden_path: str | Path,
    *,
    root: str | Path | None = None,
    baseline_path: str | Path | None = None,
) -> EvalOutcome:
    base = Path(root) if root is not None else REPO_ROOT
    golden = load_golden(golden_path, root=base)
    bands = Bands.load()

    all_results: list[Any] = []
    unverified_excluded = 0
    all_rows: list[GoldenRow] = []
    canaries: dict[str, Any] = {}
    published: list[PublishedNumber] = []
    rejection_rows: list[GoldenRow] = []
    rejection_items: list[dict[str, Any]] = []
    measured_entries = 0

    for index, entry in enumerate(golden.entries, start=1):
        rows = _entry_rows(entry)
        all_rows.extend(rows)
        rejection_rows.extend(rows)
        unverified_excluded += sum(1 for row in rows if row.status != "verified")
        if not entry.family:
            canaries[entry.id] = {
                "measured": False,
                "reason": "not an extraction fixture",
            }
            continue
        claims, text, entry_metrics, rejections = _evaluate_entry(
            entry, base, bands=bands, entity_id=index
        )
        measured_entries += 1
        predictions = {claim.predicate: claim for claim in claims}
        entry_results = [
            score_field(row, predictions.get(row.field), text)
            for row in rows
            if row.status == "verified" and row.expectation in {"value", "abstain"}
        ]
        all_results.extend(entry_results)
        rejection_items.extend(rejection.as_dict() for rejection in rejections)
        entry_metrics["claims_hash"] = claims_hash(claims)
        canaries[entry.id] = {
            "measured": True,
            "claims": len(claims),
            "claims_hash": entry_metrics.pop("claims_hash"),
            "extraction": entry_metrics,
        }
        published.extend(PublishedNumber.from_claim(claim, text=text) for claim in claims)

    extraction = summarise_fields(all_results, unverified_excluded=unverified_excluded)
    provenance = provenance_coverage(published, require_snapshot=False)
    rejection = rejection_correctness(rejection_rows, rejection_items)
    intervals = interval_coverage([])  # no interval predictions on committed fixtures

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "versions": _versions(),
        "golden": {
            "source": _display_path(Path(golden_path), base),
            **golden.as_dict(),
            "measured_entries": measured_entries,
            "unmeasured_verified_rows": _unmeasured_rows(golden, canaries),
        },
        "extraction": extraction.as_dict(),
        "rejection": rejection.as_dict(),
        "provenance": provenance.as_dict(),
        "interval_coverage": intervals.as_dict(),
        "golden_traceability": _golden_traceability(golden),
        "canaries": canaries,
        "drift": None,
        "gates": {
            "provenance_coverage": {
                "threshold": DEFAULT_FAIL_UNDER,
                "value": provenance.coverage,
                "passed": not provenance.gate_failed(DEFAULT_FAIL_UNDER),
            }
        },
        "ok": True,
    }

    if baseline_path is not None:
        baseline = _load_report(Path(baseline_path))
        report["drift"] = compare_reports(baseline, report)

    return EvalOutcome(report=report)


def _unmeasured_rows(golden: GoldenSet, canaries: Mapping[str, Any]) -> int:
    total = 0
    for entry in golden.entries:
        canary = canaries.get(entry.id) or {}
        if canary.get("measured"):
            continue
        total += len(entry.verified_rows)
    return total


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"baseline report cannot be read: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvalError(f"baseline report must be a JSON object: {path}")
    return payload


def apply_gate(report: Mapping[str, Any], fail_under: float = DEFAULT_FAIL_UNDER) -> list[str]:
    """Return gate failure messages; empty means the run may pass."""

    provenance = report.get("provenance") or {}
    coverage = float(provenance.get("coverage", 0.0))
    gates = report.setdefault("gates", {})
    gates["provenance_coverage"] = {
        "threshold": fail_under,
        "value": coverage,
        "passed": coverage >= fail_under,
    }
    failures: list[str] = []
    if coverage < fail_under:
        failures.append(
            f"provenance coverage {coverage:.4f} is below --fail-under {fail_under:.4f}"
        )
    return failures


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return target


def render_summary(report: Mapping[str, Any], *, report_path: Path | None = None) -> str:
    extraction = report.get("extraction") or {}
    provenance = report.get("provenance") or {}
    golden = report.get("golden") or {}
    lines = [
        "fmd eval: golden entries {entries} "
        "(verified rows {verified}, unverified {unverified})".format(
            entries=golden.get("entries", 0),
            verified=golden.get("verified_rows", 0),
            unverified=golden.get("unverified_rows", 0),
        ),
        "  extraction: evaluated {evaluated} precision {precision} recall {recall} f1 {f1} "
        "value_acc {value_accuracy} unit_acc {unit} abstention {abstention}".format(
            evaluated=extraction.get("evaluated", 0),
            precision=extraction.get("precision"),
            recall=extraction.get("recall"),
            f1=extraction.get("f1"),
            value_accuracy=extraction.get("value_accuracy"),
            unit=extraction.get("unit_normalisation_accuracy"),
            abstention=extraction.get("abstention_correctness"),
        ),
        "  provenance coverage: {coverage} "
        "({resolved}/{total}, modelled excluded {modelled})".format(
            coverage=provenance.get("coverage"),
            resolved=provenance.get("resolved", 0),
            total=provenance.get("total", 0),
            modelled=provenance.get("modelled_excluded", 0),
        ),
        "  interval coverage: n={n} coverage={coverage}".format(
            n=(report.get("interval_coverage") or {}).get("n", 0),
            coverage=(report.get("interval_coverage") or {}).get("coverage"),
        ),
    ]
    gate = (report.get("gates") or {}).get("provenance_coverage") or {}
    lines.append(
        "  gate provenance>={threshold}: {status}".format(
            threshold=gate.get("threshold"),
            status="PASS" if gate.get("passed") else "FAIL",
        )
    )
    drift = report.get("drift")
    if isinstance(drift, Mapping):
        lines.append(
            "  drift: unexplained={unexplained} ranking_churn={churn}".format(
                unexplained=drift.get("unexplained"),
                churn=(drift.get("ranking") or {}).get("churn"),
            )
        )
    if report_path is not None:
        lines.append(f"  report: {report_path}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_FAIL_UNDER",
    "EvalError",
    "EvalOutcome",
    "apply_gate",
    "render_summary",
    "run_evaluation",
    "write_report",
]
