"""Validation gates for extracted claims (brief §9).

A claim is **rejected, never silently fixed**, when:

* the quote is absent at ``[char_start, char_end)`` and no fuzzy window reaches
  0.95 similarity (a 0.95+ window is accepted and the span is rewritten, logged);
* a required unit is missing or outside the configured unit vocabulary;
* a rate has no denominator in its qualifier;
* a value falls outside the configured plausibility band for its predicate.

Rejections are structured so the review console can render them verbatim.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from fmd.db.repositories import ClaimCreate
from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec, JsonScalar

logger = logging.getLogger(__name__)

BANDS_YAML = Path(__file__).with_name("bands.yaml")
FUZZY_THRESHOLD = 0.95
REWRITE_CONFIDENCE_PENALTY = 0.95

RejectionReason = Literal[
    "missing_quote",
    "span_mismatch",
    "missing_unit",
    "unparseable_unit",
    "missing_denominator",
    "unparseable_value",
    "out_of_band",
]

_NUMERIC_FORMAT = re.compile(
    r"^(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<rest>[^\d]*)$",
)
_CURRENCY_PREFIX = re.compile(
    r"^\s*(?:rs\.?|inr|usd|gbp|eur|aud|cad|aed|sgd|jpy|chf|₹|\$|£|€)\s*",
    re.I,
)
_UNPARSEABLE_MAGNITUDES = re.compile(
    r"\b(lakh|lakhs|lac|lacs|crore|crores|million|billion|k|m)\b", re.I
)


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Band:
    minimum: float | None = None
    maximum: float | None = None

    def contains(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        return not (self.maximum is not None and value > self.maximum)


@dataclass(frozen=True)
class Bands:
    units: frozenset[str]
    currencies: frozenset[str]
    ranges: dict[str, Band]

    @classmethod
    def load(cls, path: str | Path | None = None) -> Bands:
        source = Path(path) if path is not None else BANDS_YAML
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{source}: bands file must be a mapping")
        ranges: dict[str, Band] = {}
        for predicate, raw in (data.get("bands") or {}).items():
            if not isinstance(raw, dict):
                raise ValueError(f"{source}: band for {predicate!r} must be a mapping")
            ranges[str(predicate)] = Band(
                minimum=raw.get("min"),
                maximum=raw.get("max"),
            )
        return cls(
            units=frozenset(str(unit) for unit in data.get("units") or []),
            currencies=frozenset(str(unit) for unit in data.get("currencies") or []),
            ranges=ranges,
        )

    def band(self, predicate: str) -> Band | None:
        return self.ranges.get(predicate)


@dataclass(frozen=True)
class Rejection:
    field: str
    predicate: str | None
    reason: RejectionReason
    detail: str
    quote: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "predicate": self.predicate,
            "reason": self.reason,
            "detail": self.detail,
            "quote": self.quote,
        }


@dataclass(frozen=True)
class SpanRewrite:
    field: str
    predicate: str
    original_quote: str
    corrected_quote: str
    before_start: int | None
    before_end: int | None
    after_start: int
    after_end: int
    ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "predicate": self.predicate,
            "original_quote": self.original_quote,
            "corrected_quote": self.corrected_quote,
            "before_start": self.before_start,
            "before_end": self.before_end,
            "after_start": self.after_start,
            "after_end": self.after_end,
            "ratio": self.ratio,
        }


@dataclass
class ValidationResult:
    claims: list[ClaimCreate] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    rewrites: list[SpanRewrite] = field(default_factory=list)

    def rejection_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
        return counts


# --------------------------------------------------------------------------- parsing


def parse_number(value: JsonScalar | None) -> float | None:
    """Parse a printed number; ``None`` when it needs a human (lakh/crore, text)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = _CURRENCY_PREFIX.sub("", str(value).strip())
    text = text.replace(",", "").replace("_", "").strip()
    text = text.rstrip("%").strip()
    text = re.sub(r"percent$", "", text, flags=re.I).strip()
    match = _NUMERIC_FORMAT.match(text)
    if match is None:
        return None
    rest = (match.group("rest") or "").strip().lower()
    if _UNPARSEABLE_MAGNITUDES.search(rest):
        return None
    return float(match.group("number"))


def _as_claim_value(number: float, original: JsonScalar) -> int | float:
    if isinstance(original, bool):
        return original
    if float(number).is_integer():
        return int(number)
    return number


# --------------------------------------------------------------------------- gate


class _SpanError(Exception):
    pass


def _resolve_span(
    text: str,
    quote: str,
    char_start: int | None,
    char_end: int | None,
    *,
    threshold: float = FUZZY_THRESHOLD,
) -> tuple[int, int, str, float, bool]:
    """Return ``(start, end, matched_text, ratio, rewritten)``.

    A rewrite means the claim must quote ``matched_text`` (the source window),
    not the model's near-miss string, so the span always resolves.
    """
    if (
        char_start is not None
        and char_end is not None
        and 0 <= char_start <= char_end
        and text[char_start:char_end] == quote
    ):
        return char_start, char_end, quote, 1.0, False

    exact = text.find(quote)
    if exact != -1:
        return exact, exact + len(quote), quote, 1.0, True

    best = _best_fuzzy_window(text, quote)
    if best is not None and best[0] >= threshold:
        ratio, start, window = best
        return start, start + len(window), window, ratio, True

    raise _SpanError(
        f"quote does not occur at the reported span, exactly, or with >= {threshold:.2f} similarity"
    )


def _best_fuzzy_window(text: str, quote: str) -> tuple[float, int, str] | None:
    """Return ``(ratio, start, window)`` for the best near-match window.

    A truncated prefix window is skipped: it would otherwise score ~0.97 by
    matching only the beginning of the quote.
    """
    if not quote or len(quote) > len(text):
        return None
    matcher = difflib.SequenceMatcher(None, quote, text, autojunk=False)
    match = matcher.find_longest_match(0, len(quote), 0, len(text))
    if match.size == 0:
        return None
    anchor = max(0, match.b - match.a)
    best: tuple[float, int, str] | None = None
    for delta in range(-2, 3):
        start = anchor + delta
        if start < 0:
            continue
        for width_delta in range(-1, 2):
            width = len(quote) + width_delta
            if width <= 0 or start + width > len(text):
                continue
            window = text[start : start + width]
            if width < len(quote) and quote.startswith(window):
                # A truncated prefix scores deceptively high; never accept it.
                continue
            ratio = difflib.SequenceMatcher(None, quote, window, autojunk=False).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, start, window)
    return best


def _normalise_unit(unit: str, currencies: frozenset[str]) -> str:
    stripped = unit.strip()
    if stripped.upper() in currencies:
        return stripped.upper()
    return stripped.lower()


def validate_document(
    document: FamilyDocument,
    text: str,
    *,
    extraction_id: int,
    entity_id: int,
    bands: Bands | None = None,
    observed_at: datetime | None = None,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    default_confidence: float = 0.5,
) -> ValidationResult:
    """Turn guided-JSON fields into validated ``ClaimCreate`` rows."""
    bands = bands or Bands.load()
    result = ValidationResult()
    context = document.context_values()

    for name, spec, extracted in document.fields():
        if extracted is None or extracted.value is None:
            # Null is a value: "the document did not say". Emit nothing, reject nothing.
            continue
        verdict = _validate_field(
            name=name,
            spec=spec,
            extracted=extracted,
            document=document,
            text=text,
            bands=bands,
            context=context,
            extraction_id=extraction_id,
            entity_id=entity_id,
            observed_at=observed_at,
            fuzzy_threshold=fuzzy_threshold,
            default_confidence=default_confidence,
        )
        if isinstance(verdict, Rejection):
            result.rejections.append(verdict)
            logger.warning("rejected %s (%s): %s", name, verdict.reason, verdict.detail)
            continue
        claim, rewrite = verdict
        result.claims.append(claim)
        if rewrite is not None:
            result.rewrites.append(rewrite)
            logger.info(
                "rewrote span for %s from %s..%s to %s..%s (ratio %.3f)",
                name,
                rewrite.before_start,
                rewrite.before_end,
                rewrite.after_start,
                rewrite.after_end,
                rewrite.ratio,
            )
    return result


def _validate_field(  # noqa: PLR0911, PLR0913 - gates are deliberately explicit
    *,
    name: str,
    spec: FieldSpec,
    extracted: ExtractedField,
    document: FamilyDocument,
    text: str,
    bands: Bands,
    context: dict[str, JsonScalar],
    extraction_id: int,
    entity_id: int,
    observed_at: datetime | None,
    fuzzy_threshold: float,
    default_confidence: float,
) -> Rejection | tuple[ClaimCreate, SpanRewrite | None]:
    quote = (extracted.quote or "").strip()
    if not quote:
        return Rejection(
            field=name,
            predicate=spec.predicate,
            reason="missing_quote",
            detail="value has no verbatim quote; an unquotable value is not evidence",
        )

    try:
        start, end, matched_quote, ratio, rewritten = _resolve_span(
            text, quote, extracted.char_start, extracted.char_end, threshold=fuzzy_threshold
        )
    except _SpanError as exc:
        return Rejection(
            field=name,
            predicate=spec.predicate,
            reason="span_mismatch",
            detail=str(exc),
            quote=quote,
        )

    rewrite: SpanRewrite | None = None
    if rewritten:
        rewrite = SpanRewrite(
            field=name,
            predicate=spec.predicate,
            original_quote=quote,
            corrected_quote=matched_quote,
            before_start=extracted.char_start,
            before_end=extracted.char_end,
            after_start=start,
            after_end=end,
            ratio=ratio,
        )
        quote = matched_quote

    unit = _normalise_unit(extracted.unit, bands.currencies) if extracted.unit else None
    if spec.unit_required and not unit:
        return Rejection(
            field=name,
            predicate=spec.predicate,
            reason="missing_unit",
            detail="a quantity without a unit cannot be compared or aggregated",
            quote=quote,
        )
    if unit is not None:
        allowed = bands.units | bands.currencies
        if unit not in allowed:
            return Rejection(
                field=name,
                predicate=spec.predicate,
                reason="unparseable_unit",
                detail=f"unit {unit!r} is not in the configured unit vocabulary",
                quote=quote,
            )
        if spec.unit_kind == "money" and unit not in bands.currencies:
            return Rejection(
                field=name,
                predicate=spec.predicate,
                reason="unparseable_unit",
                detail=f"money value must carry a currency, got {unit!r}",
                quote=quote,
            )

    qualifier: dict[str, Any] = dict(context.items())
    value: Any = extracted.value
    number: float | None = None

    numeric_expected = (
        spec.unit_kind
        in {
            "money",
            "count",
            "hours",
            "months",
            "years",
            "percent",
        }
        or bands.band(spec.predicate) is not None
    )

    if spec.is_rate:
        number = parse_number(value)
        if number is None:
            return Rejection(
                field=name,
                predicate=spec.predicate,
                reason="unparseable_value",
                detail=f"rate value {value!r} is not a parseable number",
                quote=quote,
            )
        if unit in {"percent", "percentage", "%"}:
            if number > 1:
                number /= 100
            unit = "ratio"
        elif unit in {None, "ratio", "fraction"}:
            unit = "ratio"
        value = _as_claim_value(round(number, 6), 1.0)
    elif numeric_expected:
        number = parse_number(value)
        if number is None:
            return Rejection(
                field=name,
                predicate=spec.predicate,
                reason="unparseable_value",
                detail=f"value {value!r} is not a parseable number",
                quote=quote,
            )
        value = _as_claim_value(number, extracted.value)

    if spec.is_rate:
        denominator = _denominator(document, spec)
        if denominator is None:
            return Rejection(
                field=name,
                predicate=spec.predicate,
                reason="missing_denominator",
                detail=(
                    "rate has no denominator in its qualifier; the denominator must be "
                    f"extracted from the same document (expected {spec.denominator_field!r})"
                ),
                quote=quote,
            )
        qualifier["denominator"] = _as_claim_value(denominator, 1.0)
        qualifier["numerator"] = value

    band = bands.band(spec.predicate)
    if band is not None:
        if number is None:
            number = parse_number(value)
        if number is None:
            return Rejection(
                field=name,
                predicate=spec.predicate,
                reason="unparseable_value",
                detail=f"value {value!r} cannot be checked against its plausibility band",
                quote=quote,
            )
        if not band.contains(number):
            return Rejection(
                field=name,
                predicate=spec.predicate,
                reason="out_of_band",
                detail=(
                    f"value {number} is outside the plausibility band "
                    f"[{band.minimum}, {band.maximum}] for {spec.predicate}"
                ),
                quote=quote,
            )

    if spec.unit_kind == "money":
        qualifier["period"] = spec.period
        qualifier["as_of"] = observed_at.date().isoformat() if observed_at is not None else None

    confidence = extracted.confidence
    if confidence is None:
        confidence = default_confidence
    if rewrite is not None:
        confidence *= REWRITE_CONFIDENCE_PENALTY
    confidence = max(0.0, min(1.0, confidence))

    claim = ClaimCreate(
        extraction_id=extraction_id,
        entity_id=entity_id,
        predicate=spec.predicate,
        value=value,
        unit=unit,
        qualifier=qualifier or None,
        quote=quote,
        char_start=start,
        char_end=end,
        confidence=round(confidence, 4),
        observed_at=observed_at,
        method="extracted",
    )
    return claim, rewrite


def _denominator(document: FamilyDocument, spec: FieldSpec) -> float | None:
    field_name = spec.denominator_field
    if not field_name:
        return None
    sibling = getattr(document, field_name, None)
    if sibling is None or sibling.value is None:
        return None
    number = parse_number(sibling.value)
    if number is None or number <= 0:
        return None
    return number
