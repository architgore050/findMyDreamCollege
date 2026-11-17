"""Deterministic evaluation metrics (brief §19).

Every metric is a pure function over plain records. Nothing here reads the
network, a clock, or a database. The definitions are deliberately narrow so a
number like "precision 1.0" cannot be read as more than it proves:

* **exact field match** requires the value, the normalised unit, and a quote
  that resolves at the reported span *and* equals the golden quote;
* **provenance coverage** is resolved published numbers / all published
  numbers, and is a hard gate (default 1.0);
* **abstention correctness** only counts abstentions; a run that abstains on
  everything gets a low recall, not a high score;
* **row attribution** is checked separately when a golden row names a context:
  the same value in the wrong row is a false positive (wrong attribution), never
  a miss, and is counted in ``wrong_attributions``. ``row_attribution_accuracy``
  is ``None`` until at least one row-context claim has a prediction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from fmd.eval.rowctx import (
    canonical_equal,
    inside,
    occurrence_spans,
    row_block,
)

NUMERIC_TOLERANCE = 1e-9
PUBLISHED_METHODS = frozenset({"extracted", "derived"})
INTERVAL_NOMINAL = 0.80


# --------------------------------------------------------------------------- units


@lru_cache(maxsize=1)
def _configured_units() -> tuple[frozenset[str], frozenset[str]]:
    from fmd.extract.validate import Bands

    bands = Bands.load()
    return bands.units, bands.currencies


def normalise_unit(unit: str | None) -> str | None:
    """Same normalisation the extraction gate applies, kept in one place."""

    if unit is None:
        return None
    stripped = str(unit).strip()
    if not stripped:
        return None
    _, currencies = _configured_units()
    if stripped.upper() in currencies:
        return stripped.upper()
    return stripped.lower()


# --------------------------------------------------------------------------- values


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(
            float(expected), float(actual), rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            values_equal(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Prediction:
    """One claim-shaped prediction (extracted or otherwise)."""

    field: str
    value: Any
    unit: str | None = None
    quote: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    method: str = "extracted"
    context: str | None = None
    disambiguation: str | None = None
    row_label: str | None = None
    column_header: str | None = None
    value_cell: str | None = None
    table_page: int | None = None


@dataclass(frozen=True)
class GoldenRow:
    """The metric-facing projection of a golden ``GoldenValue``."""

    field: str
    value: Any = None
    unit: str | None = None
    quote: str | None = None
    status: str = "verified"
    expectation: str = "value"
    printed_value: Any = None
    printed_unit: str | None = None
    reason: str | None = None
    context: str | None = None
    disambiguation: str | None = None
    row_label: str | None = None
    column_header: str | None = None
    value_cell: str | None = None
    table_page: int | None = None

    @property
    def scored(self) -> bool:
        return self.status == "verified" and self.expectation in {"value", "abstain"}

    @property
    def has_row_context(self) -> bool:
        return bool(self.context)


@dataclass(frozen=True)
class FieldResult:
    field: str
    expected: Any = None
    predicted: Any = None
    expected_unit: str | None = None
    predicted_unit: str | None = None
    expectation: str = "value"
    status: str = "verified"
    has_prediction: bool = False
    span_ok: bool = False
    value_ok: bool = False
    unit_ok: bool = False
    exact: bool = False
    abstained: bool = False
    abstention_correct: bool = False
    has_row_context: bool = False
    attribution_ok: bool = True
    wrong_attribution: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "expected": self.expected,
            "predicted": self.predicted,
            "expected_unit": self.expected_unit,
            "predicted_unit": self.predicted_unit,
            "expectation": self.expectation,
            "status": self.status,
            "has_prediction": self.has_prediction,
            "span_ok": self.span_ok,
            "value_ok": self.value_ok,
            "unit_ok": self.unit_ok,
            "exact": self.exact,
            "abstained": self.abstained,
            "abstention_correct": self.abstention_correct,
            "has_row_context": self.has_row_context,
            "attribution_ok": self.attribution_ok,
            "wrong_attribution": self.wrong_attribution,
            "detail": self.detail,
        }


@dataclass
class ExtractionMetrics:
    evaluated: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    value_correct: int = 0
    unit_checks: int = 0
    unit_correct: int = 0
    abstentions: int = 0
    correct_abstentions: int = 0
    unverified_excluded: int = 0
    wrong_attributions: int = 0
    row_context_rows: int = 0
    row_attribution_checks: int = 0
    row_attribution_correct: int = 0
    per_field: list[FieldResult] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return None if denominator == 0 else self.true_positives / denominator

    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return None if denominator == 0 else self.true_positives / denominator

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def value_accuracy(self) -> float | None:
        predicted = self.true_positives + self.false_positives
        return None if predicted == 0 else self.value_correct / predicted

    @property
    def unit_normalisation_accuracy(self) -> float | None:
        return None if self.unit_checks == 0 else self.unit_correct / self.unit_checks

    @property
    def abstention_rate(self) -> float | None:
        return None if self.evaluated == 0 else self.abstentions / self.evaluated

    @property
    def abstention_correctness(self) -> float | None:
        return None if self.abstentions == 0 else self.correct_abstentions / self.abstentions

    @property
    def row_attribution_accuracy(self) -> float | None:
        return (
            None
            if self.row_attribution_checks == 0
            else self.row_attribution_correct / self.row_attribution_checks
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "value_accuracy": _round(self.value_accuracy),
            "unit_normalisation_accuracy": _round(self.unit_normalisation_accuracy),
            "unit_checks": self.unit_checks,
            "abstention_rate": _round(self.abstention_rate),
            "abstention_correctness": _round(self.abstention_correctness),
            "abstentions": self.abstentions,
            "correct_abstentions": self.correct_abstentions,
            "unverified_excluded": self.unverified_excluded,
            "row_context_rows": self.row_context_rows,
            "row_attribution_checks": self.row_attribution_checks,
            "row_attribution_correct": self.row_attribution_correct,
            "row_attribution_accuracy": _round(self.row_attribution_accuracy),
            "wrong_attributions": self.wrong_attributions,
            "per_field": [result.as_dict() for result in self.per_field],
        }


@dataclass
class RejectionMetrics:
    expected_rejects: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_field: list[dict[str, Any]] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return None if denominator == 0 else self.true_positives / denominator

    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return None if denominator == 0 else self.true_positives / denominator

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_rejects": self.expected_rejects,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "per_field": self.per_field,
        }


@dataclass
class IntervalReport:
    n: int = 0
    inside: int = 0
    nominal: float = INTERVAL_NOMINAL

    @property
    def coverage(self) -> float | None:
        return None if self.n == 0 else self.inside / self.n

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "inside": self.inside,
            "coverage": _round(self.coverage),
            "nominal": self.nominal,
            "measurable": self.n > 0,
        }


# --------------------------------------------------------------------------- scoring


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _resolve_quote(text: str | None, quote: str | None, start: int | None, end: int | None) -> bool:
    if text is None or not quote:
        return False
    if start is None or end is None or start < 0 or end < start or end > len(text):
        return False
    return text[start:end] == quote


def score_field(
    row: GoldenRow,
    prediction: Prediction | None,
    text: str | None,
) -> FieldResult:
    """Score one golden field. Missing predictions are abstentions, not zeros."""

    has_prediction = prediction is not None
    span_ok = bool(
        prediction
        and _resolve_quote(text, prediction.quote, prediction.char_start, prediction.char_end)
        and (row.quote is None or prediction.quote == row.quote)
    )
    value_ok = bool(prediction and values_equal(row.value, prediction.value))
    expected_unit = normalise_unit(row.unit)
    predicted_unit = normalise_unit(prediction.unit) if prediction else None
    unit_ok = expected_unit == predicted_unit
    has_row_context = row.has_row_context
    attribution_ok = _attribution_ok(row, prediction, text)
    exact = bool(has_prediction and span_ok and value_ok and unit_ok and attribution_ok)
    wrong_attribution = bool(
        has_prediction
        and has_row_context
        and not attribution_ok
        and value_ok
    )
    abstained = not has_prediction
    abstention_correct = abstained and row.expectation == "abstain"
    detail = ""
    if has_prediction and not exact:
        missing = [
            name
            for name, ok in (
                ("span", span_ok),
                ("value", value_ok),
                ("unit", unit_ok),
                ("row", attribution_ok),
            )
            if not ok
        ]
        detail = "mismatch: " + ", ".join(missing)
    elif abstained and not abstention_correct:
        detail = "missing prediction for a verified value"
    if wrong_attribution:
        detail = "wrong row attribution: the value is real but the span is in another row"
    return FieldResult(
        field=row.field,
        expected=row.value,
        predicted=None if prediction is None else prediction.value,
        expected_unit=row.unit,
        predicted_unit=None if prediction is None else prediction.unit,
        expectation=row.expectation,
        status=row.status,
        has_prediction=has_prediction,
        span_ok=span_ok,
        value_ok=value_ok,
        unit_ok=unit_ok,
        exact=exact,
        abstained=abstained,
        abstention_correct=abstention_correct,
        has_row_context=has_row_context,
        attribution_ok=attribution_ok,
        wrong_attribution=wrong_attribution,
        detail=detail,
    )


def _attribution_ok(row: GoldenRow, prediction: Prediction | None, text: str | None) -> bool:
    """Did the prediction land in the golden row's block?

    Rows without context are backward compatible: attribution is not claimed,
    so it cannot be wrong. A row-context claim must have a unique context in the
    extraction text and the prediction span must fall inside that block. Claim
    metadata (context, row label) is checked when the prediction supplies it;
    its absence is not fabricated into a pass.
    """

    if not row.has_row_context:
        return True
    if prediction is None or text is None:
        return False
    if prediction.char_start is None or prediction.char_end is None:
        return False
    if prediction.char_start < 0 or prediction.char_end < prediction.char_start:
        return False
    context_spans = occurrence_spans(text, row.context)
    if len(context_spans) != 1:
        return False
    block = row_block(text, context_spans[0])
    if not inside([(prediction.char_start, prediction.char_end)], block):
        return False
    if prediction.context and not canonical_equal(row.context, prediction.context):
        return False
    if row.row_label and prediction.row_label:
        return canonical_equal(row.row_label, prediction.row_label)
    return True


def summarise_fields(
    results: Sequence[FieldResult], *, unverified_excluded: int = 0
) -> ExtractionMetrics:
    metrics = ExtractionMetrics(unverified_excluded=unverified_excluded)
    for result in results:
        metrics.per_field.append(result)
        if result.status != "verified":
            continue
        if result.expectation not in {"value", "abstain"}:
            continue
        metrics.evaluated += 1
        if result.has_row_context:
            metrics.row_context_rows += 1
            if result.has_prediction:
                metrics.row_attribution_checks += 1
                if result.attribution_ok:
                    metrics.row_attribution_correct += 1
        if result.wrong_attribution:
            metrics.wrong_attributions += 1
        if result.abstained:
            metrics.abstentions += 1
            if result.abstention_correct:
                metrics.correct_abstentions += 1
        if result.exact:
            metrics.true_positives += 1
            metrics.value_correct += 1
        elif result.has_prediction:
            metrics.false_positives += 1
            if result.value_ok:
                metrics.value_correct += 1
        elif result.expectation == "value":
            metrics.false_negatives += 1
        if result.expected_unit is not None and result.has_prediction:
            metrics.unit_checks += 1
            if result.unit_ok:
                metrics.unit_correct += 1
    return metrics


def score_extraction(
    rows: Sequence[GoldenRow],
    predictions: Mapping[str, Prediction] | Sequence[Prediction],
    text: str | None,
    *,
    unverified_excluded: int = 0,
) -> ExtractionMetrics:
    by_field = (
        dict(predictions)
        if isinstance(predictions, Mapping)
        else {prediction.field: prediction for prediction in predictions}
    )
    results = [
        score_field(row, by_field.get(row.field), text)
        for row in rows
        if row.status == "verified" and row.expectation in {"value", "abstain"}
    ]
    return summarise_fields(results, unverified_excluded=unverified_excluded)


def rejection_correctness(
    rows: Sequence[GoldenRow],
    rejections: Iterable[Mapping[str, Any] | str],
) -> RejectionMetrics:
    """Compare expected rejections against the pipeline's rejection fields."""

    rejected = {
        str(item.get("field")) if isinstance(item, Mapping) else str(item) for item in rejections
    }
    metrics = RejectionMetrics()
    for row in rows:
        if row.status != "verified":
            continue
        expected_reject = row.expectation == "reject"
        was_rejected = row.field in rejected
        if expected_reject:
            metrics.expected_rejects += 1
        if expected_reject and was_rejected:
            metrics.true_positives += 1
        elif not expected_reject and was_rejected:
            metrics.false_positives += 1
        elif expected_reject and not was_rejected:
            metrics.false_negatives += 1
        if expected_reject or was_rejected:
            metrics.per_field.append(
                {
                    "field": row.field,
                    "expected_reject": expected_reject,
                    "rejected": was_rejected,
                    "reason": row.reason,
                }
            )
    return metrics


# --------------------------------------------------------------------------- provenance


@dataclass(frozen=True)
class PublishedNumber:
    predicate: str
    value: Any
    quote: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    unit: str | None = None
    method: str = "extracted"
    text: str | None = None
    url: str | None = None
    fetched_at: Any = None
    snapshot_sha256: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    bboxes: tuple[tuple[float, float, float, float], ...] = ()
    page_no: int | None = None
    page_box: tuple[float, float, float, float] | None = None
    bbox_granularity: str = "unknown"
    document_tier: str | None = None

    @classmethod
    def from_claim(cls, claim: Any, *, text: str | None = None) -> PublishedNumber:
        get = (
            claim.get
            if isinstance(claim, Mapping)
            else lambda key, default=None: getattr(claim, key, default)
        )
        qualifier = get("qualifier") or {}
        if not isinstance(qualifier, Mapping):
            qualifier = {}
        raw_boxes = qualifier.get("bboxes") or ()
        boxes: list[tuple[float, float, float, float]] = []
        for raw in raw_boxes:
            box = raw.get("bbox") if isinstance(raw, Mapping) else None
            if box is None and isinstance(raw, list | tuple) and len(raw) == 4:
                box = raw
            if box is not None and len(box) == 4:
                boxes.append(tuple(float(value) for value in box))  # type: ignore[arg-type]
        fallback_box = qualifier.get("bbox")
        bbox: tuple[float, float, float, float] | None = None
        if fallback_box is not None and len(fallback_box) == 4:
            bbox = tuple(float(value) for value in fallback_box)  # type: ignore[arg-type]
        elif boxes:
            bbox = boxes[0]
        raw_page_box = qualifier.get("page_box")
        page_box: tuple[float, float, float, float] | None = None
        if raw_page_box is not None and len(raw_page_box) == 4:
            page_box = tuple(float(value) for value in raw_page_box)  # type: ignore[arg-type]
        return cls(
            predicate=str(get("predicate", "")),
            value=get("value"),
            quote=get("quote"),
            char_start=get("char_start"),
            char_end=get("char_end"),
            unit=get("unit"),
            method=str(get("method", "extracted")),
            text=text,
            url=get("url"),
            fetched_at=get("fetched_at"),
            snapshot_sha256=get("sha256", None) or get("snapshot_sha256", None),
            bbox=bbox,
            bboxes=tuple(boxes),
            page_no=qualifier.get("page_no"),
            page_box=page_box,
            bbox_granularity=str(qualifier.get("bbox_granularity") or "unknown"),
            document_tier=qualifier.get("document_tier"),
        )


# --------------------------------------------------------------------------- bbox / offsets


BBox = tuple[float, float, float, float]
BBOX_GATE_TIERS: tuple[str, ...] = ("T0", "T1")
BBOX_EPSILON = 1.0


def _box_inside(inner: Sequence[float], outer: Sequence[float], *, epsilon: float) -> bool:
    x0, y0, x1, y1 = (float(value) for value in inner)
    ox0, oy0, ox1, oy1 = (float(value) for value in outer)
    if x1 < x0 or y1 < y0:
        return False
    return (
        x0 >= ox0 - epsilon and y0 >= oy0 - epsilon and x1 <= ox1 + epsilon and y1 <= oy1 + epsilon
    )


@dataclass
class BBoxReport:
    """Published numbers with a non-null bbox inside their page box."""

    total: int = 0
    covered: int = 0
    gated_total: int = 0
    gated_covered: int = 0
    by_granularity: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    gate_tiers: tuple[str, ...] = BBOX_GATE_TIERS

    @property
    def coverage(self) -> float | None:
        return None if self.total == 0 else self.covered / self.total

    @property
    def gated_coverage(self) -> float | None:
        return None if self.gated_total == 0 else self.gated_covered / self.gated_total

    @property
    def gate_applicable(self) -> bool:
        return self.gated_total > 0

    def gate_failed(self, threshold: float = 1.0) -> bool:
        """Gate only documents acquired through the configured tiers.

        OCR pages report coverage but are not gated: a block box cannot be finer
        than a block, so demanding 1.0 there would be a false promise.
        """
        if not self.gate_applicable:
            return False
        return (self.gated_coverage or 0.0) < threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": _round(self.coverage),
            "gated_coverage": _round(self.gated_coverage),
            "total": self.total,
            "covered": self.covered,
            "gated_total": self.gated_total,
            "gated_covered": self.gated_covered,
            "by_granularity": dict(sorted(self.by_granularity.items())),
            "gate_tiers": list(self.gate_tiers),
            "gate_applicable": self.gate_applicable,
            "gate_passed": not self.gate_failed(),
            "failures": self.failures,
        }


@dataclass
class GranularityOffsets:
    total: int = 0
    resolved: int = 0

    @property
    def resolvability(self) -> float | None:
        return None if self.total == 0 else self.resolved / self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "resolved": self.resolved,
            "resolvability": _round(self.resolvability),
        }


@dataclass
class OffsetReport:
    """Quote-to-character-offset resolvability, split by bbox granularity."""

    by_granularity: dict[str, GranularityOffsets] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(item.total for item in self.by_granularity.values())

    @property
    def resolved(self) -> int:
        return sum(item.resolved for item in self.by_granularity.values())

    @property
    def resolvability(self) -> float | None:
        return None if self.total == 0 else self.resolved / self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolvability": _round(self.resolvability),
            "total": self.total,
            "resolved": self.resolved,
            "by_granularity": {
                key: value.as_dict() for key, value in sorted(self.by_granularity.items())
            },
        }


def bbox_coverage(
    numbers: Iterable[PublishedNumber],
    *,
    epsilon: float = BBOX_EPSILON,
    gate_tiers: Iterable[str] = BBOX_GATE_TIERS,
) -> BBoxReport:
    """Published numbers whose bboxes all lie inside their page box."""
    report = BBoxReport(gate_tiers=tuple(gate_tiers))
    for number in numbers:
        if number.method not in PUBLISHED_METHODS or not is_number(number.value):
            continue
        report.total += 1
        gated = number.document_tier in report.gate_tiers
        if gated:
            report.gated_total += 1
        granularity = number.bbox_granularity or "unknown"
        report.by_granularity[granularity] = report.by_granularity.get(granularity, 0) + 1
        boxes = number.bboxes or ((number.bbox,) if number.bbox is not None else ())
        reason: str | None = None
        if not boxes:
            reason = "missing_bbox"
        elif number.page_box is None:
            reason = "missing_page_box"
        elif any(not _box_inside(box, number.page_box, epsilon=epsilon) for box in boxes):
            reason = "bbox_outside_page"
        if reason is None:
            report.covered += 1
            if gated:
                report.gated_covered += 1
        else:
            report.failures.append(
                {
                    "predicate": number.predicate,
                    "value": number.value,
                    "granularity": granularity,
                    "document_tier": number.document_tier,
                    "reason": reason,
                }
            )
    return report


def offset_resolvability(numbers: Iterable[PublishedNumber]) -> OffsetReport:
    """Do reported offsets still select the quoted text, per granularity?"""
    report = OffsetReport()
    for number in numbers:
        if number.method not in PUBLISHED_METHODS or not is_number(number.value):
            continue
        key = number.bbox_granularity or "unknown"
        bucket = report.by_granularity.setdefault(key, GranularityOffsets())
        bucket.total += 1
        if _resolve_quote(number.text, number.quote, number.char_start, number.char_end):
            bucket.resolved += 1
    return report


@dataclass
class ProvenanceReport:
    total: int = 0
    resolved: int = 0
    modelled_excluded: int = 0
    non_numeric_excluded: int = 0
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    require_snapshot: bool = False
    bbox: BBoxReport = field(default_factory=BBoxReport)
    offsets: OffsetReport = field(default_factory=OffsetReport)

    @property
    def coverage(self) -> float:
        return 1.0 if self.total == 0 else self.resolved / self.total

    @property
    def vacuous(self) -> bool:
        return self.total == 0

    def gate_failed(self, threshold: float = 1.0) -> bool:
        return self.coverage < threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": _round(self.coverage),
            "total": self.total,
            "resolved": self.resolved,
            "modelled_excluded": self.modelled_excluded,
            "non_numeric_excluded": self.non_numeric_excluded,
            "vacuous": self.vacuous,
            "require_snapshot": self.require_snapshot,
            "unresolved": self.unresolved,
            "bbox_coverage": self.bbox.as_dict(),
            "offset_resolvability": self.offsets.as_dict(),
        }


def provenance_coverage(
    numbers: Iterable[PublishedNumber],
    *,
    require_snapshot: bool = False,
) -> ProvenanceReport:
    """Published numbers whose claim resolves to a retrievable quote / all published.

    ``method="modelled"`` numbers are excluded and counted: they are labelled as
    modelled, so they are not pretending to be extracted evidence. A run with no
    published numbers reports 1.0 and ``vacuous=true`` so an empty pipeline
    cannot masquerade as a healthy one.
    """

    report = ProvenanceReport(require_snapshot=require_snapshot)
    materialised = list(numbers)
    for number in materialised:
        if number.method == "modelled":
            report.modelled_excluded += 1
            continue
        if number.method not in PUBLISHED_METHODS:
            report.non_numeric_excluded += 1
            continue
        if not is_number(number.value):
            report.non_numeric_excluded += 1
            continue
        report.total += 1
        reason = _unresolved_reason(number, require_snapshot=require_snapshot)
        if reason is None:
            report.resolved += 1
        else:
            report.unresolved.append(
                {
                    "predicate": number.predicate,
                    "value": number.value,
                    "quote": number.quote,
                    "reason": reason,
                }
            )
    report.bbox = bbox_coverage(materialised)
    report.offsets = offset_resolvability(materialised)
    return report


def _unresolved_reason(number: PublishedNumber, *, require_snapshot: bool) -> str | None:
    if not number.quote:
        return "missing_quote"
    if not _resolve_quote(number.text, number.quote, number.char_start, number.char_end):
        return "span_does_not_resolve"
    if require_snapshot:
        if not number.url:
            return "missing_source_url"
        if number.fetched_at is None:
            return "missing_fetched_at"
        if not number.snapshot_sha256:
            return "missing_snapshot_sha256"
    return None


# --------------------------------------------------------------------------- intervals


def interval_coverage(
    observations: Iterable[tuple[float, float, float]],
    *,
    nominal: float = INTERVAL_NOMINAL,
) -> IntervalReport:
    """Do intervals contain the golden value? ``observations = (truth, lo, hi)``."""

    report = IntervalReport(nominal=nominal)
    for truth, lo, hi in observations:
        report.n += 1
        if float(lo) <= float(truth) <= float(hi):
            report.inside += 1
    return report


__all__ = [
    "BBOX_GATE_TIERS",
    "INTERVAL_NOMINAL",
    "BBoxReport",
    "GranularityOffsets",
    "OffsetReport",
    "bbox_coverage",
    "offset_resolvability",
    "ExtractionMetrics",
    "FieldResult",
    "GoldenRow",
    "IntervalReport",
    "Prediction",
    "ProvenanceReport",
    "PublishedNumber",
    "RejectionMetrics",
    "interval_coverage",
    "is_number",
    "normalise_unit",
    "provenance_coverage",
    "rejection_correctness",
    "score_extraction",
    "score_field",
    "summarise_fields",
    "values_equal",
]
