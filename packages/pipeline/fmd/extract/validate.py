"""Validation gates for extracted claims (brief §9).

The model emits a **quote**, never a number and never an offset. This module
turns that quote into a claim deterministically:

1. **Resolve the span.** Exact occurrence first; if absent, canonicalise both
   sides (NFKC, unified quotes/dashes, Devanagari digits to ASCII, ``₹``/``Rs.``/
   ``Rs``/``INR`` to ``INR``, Indian digit grouping removed, whitespace
   collapsed) and retry. Only a **unique** occurrence is accepted: several
   matches are ``ambiguous_span`` and resolve to UNKNOWN, never a first-match
   guess. Zero matches get exactly one repair pass (strip edge punctuation /
   take the longest line) and then abstain.
2. **Derive the value.** The resolved span is parsed by a type-specific parser
   (count, money+period, percentage, integer, date, grade, boolean, text). No
   model-supplied number, unit or conversion is ever trusted.
3. **Apply semantic guards** that are independent of the model: a salary must
   carry a printed period, a rate's numerator cannot exceed its denominator, a
   placement rate cannot exceed 100%, placements cannot exceed eligible
   students, seat quotas cannot exceed the sanctioned total.

A claim is rejected, never silently fixed. Rejections and per-field outcomes
(``resolved``, ``ambiguous``, ``unresolved``, ``rejected_guard:<name>``) are
structured so the review console can render them verbatim.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from fmd.db.repositories import ClaimCreate
from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec, JsonScalar

logger = logging.getLogger(__name__)

BANDS_YAML = Path(__file__).with_name("bands.yaml")
REWRITE_CONFIDENCE_PENALTY = 0.95

RejectionReason = Literal[
    "missing_quote",
    "ambiguous_span",
    "unresolved_span",
    "unparseable_value",
    "unparseable_unit",
    "missing_denominator",
    "missing_period",
    "out_of_band",
    "guard_violation",
]

OutcomeKind = Literal["resolved", "ambiguous", "unresolved", "rejected", "abstained"]


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
    guard: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "predicate": self.predicate,
            "reason": self.reason,
            "detail": self.detail,
            "quote": self.quote,
            "guard": self.guard,
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


@dataclass(frozen=True)
class FieldOutcome:
    """One per-field result for the review console (never a score)."""

    field: str
    predicate: str
    outcome: str
    detail: str = ""
    quote: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    repaired: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "predicate": self.predicate,
            "outcome": self.outcome,
            "detail": self.detail,
            "quote": self.quote,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "repaired": self.repaired,
        }


@dataclass
class ValidationResult:
    claims: list[ClaimCreate] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    rewrites: list[SpanRewrite] = field(default_factory=list)
    outcomes: list[FieldOutcome] = field(default_factory=list)

    def rejection_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
        return counts

    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            key = outcome.outcome
            counts[key] = counts.get(key, 0) + 1
        return counts


# --------------------------------------------------------------------------- canonical text

_QUOTE_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
    }
)
_CURRENCY_TOKEN = re.compile(
    r"\u20b9|\brs\.?|\binr\b|\binr(?=\d)|\$|\u00a3|\u20ac|\b(?:usd|gbp|eur)\b",
    re.I,
)
_CURRENCY_NAMES = {"\u20b9": "INR", "$": "USD", "\u00a3": "GBP", "\u20ac": "EUR"}
_CURRENCY_CODE = re.compile(r"\b(INR|USD|GBP|EUR|AUD|CAD|AED|SGD|JPY|CHF)\b")
_GROUPING_COMMA = re.compile(r"(?<=\d),(?=\d)")
_WHITESPACE = re.compile(r"\s+")
_MAGNITUDE = re.compile(
    r"\b(lakh|lakhs|lac|lacs|crore|crores|million|millions|billion|billions|thousand)\b", re.I
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PERCENT_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*(\d+(?:\.\d+)?)", re.I)
_PERCENT_MARK = re.compile(r"%|\bper\s*cent\b|\bpercent(?:age)?\b", re.I)
_GRADE = re.compile(r"\b(O|A\+\+|A\+|A|B\+\+|B\+|B|C\+\+|C)\b")

_PERIOD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("per_month", re.compile(r"per\s*month|monthly|p\.?\s*m\.?|/\s*month\b", re.I)),
    ("per_semester", re.compile(r"per\s*semester|semester|/\s*sem\b", re.I)),
    (
        "per_year",
        re.compile(
            r"per\s*annum|per\s*year|annually|annual|p\.?\s*a\.?|/\s*year\b|\ba\s+year\b",
            re.I,
        ),
    ),
    ("one_time", re.compile(r"one[-\s]?time|registration\s+fee|application\s+fee", re.I)),
)

_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("%Y-%m-%d", re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")),
    ("%d/%m/%Y", re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")),
    ("%d-%m-%Y", re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4})\b")),
    ("%d.%m.%Y", re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b")),
    ("%d %B %Y", re.compile(r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b")),
    ("%d %b %Y", re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b")),
)


def _map_replace(
    text: str,
    mapping: list[tuple[int, int]],
    pattern: re.Pattern[str],
    replacement: str,
) -> tuple[str, list[tuple[int, int]]]:
    out: list[str] = []
    out_map: list[tuple[int, int]] = []
    cursor = 0
    for match in pattern.finditer(text):
        for index in range(cursor, match.start()):
            out.append(text[index])
            out_map.append(mapping[index])
        for char in replacement:
            out.append(char)
            out_map.append((match.start(), match.end()))
        cursor = match.end()
    for index in range(cursor, len(text)):
        out.append(text[index])
        out_map.append(mapping[index])
    return "".join(out), out_map


def _transform_chars(
    text: str,
    mapping: list[tuple[int, int]],
    transform: Any,
) -> tuple[str, list[tuple[int, int]]]:
    out: list[str] = []
    out_map: list[tuple[int, int]] = []
    for index, char in enumerate(text):
        for produced in transform(char):
            out.append(produced)
            out_map.append(mapping[index])
    return "".join(out), out_map


def _char_variants(char: str) -> str:
    if char in {"\u2018", "\u2019", "\u201a", "\u201b", "\u201c", "\u201d", "\u201e"}:
        return char.translate(_QUOTE_TABLE)
    if char in {"\u2013", "\u2014", "\u2212", "\u00a0", "\u202f"}:
        return char.translate(_QUOTE_TABLE)
    return unicodedata.normalize("NFKC", char)


def _expand_digits(variants: str) -> str:
    out: list[str] = []
    for char in variants:
        if char.isascii():
            out.append(char)
            continue
        try:
            digit = unicodedata.digit(char)
        except (TypeError, ValueError):
            out.append(char)
            continue
        out.append(str(digit))
    return "".join(out)


def _currency_code(token: str) -> str:
    if token in _CURRENCY_NAMES:
        return _CURRENCY_NAMES[token]
    lowered = token.lower().rstrip(".")
    if lowered in {"rs", "inr"}:
        return "INR"
    return token.upper()


def _map_currency(text: str, mapping: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]]:
    out: list[str] = []
    out_map: list[tuple[int, int]] = []
    cursor = 0
    for match in _CURRENCY_TOKEN.finditer(text):
        for index in range(cursor, match.start()):
            out.append(text[index])
            out_map.append(mapping[index])
        code = _currency_code(match.group(0))
        if match.end() < len(text) and text[match.end()].isdigit():
            # ``₹85`` and ``INR850`` both canonicalise to ``INR 850``.
            code = code + " "
        for char in code:
            out.append(char)
            out_map.append((match.start(), match.end()))
        cursor = match.end()
    for index in range(cursor, len(text)):
        out.append(text[index])
        out_map.append(mapping[index])
    return "".join(out), out_map


def canonicalise(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Return ``(canonical_text, map)`` where ``map[i]`` is the source span.

    The map keeps every canonical character anchored to the original document,
    so a match on canonical text can be reported as exact document offsets. It
    is deliberately lossless for offsets: expanded tokens (``₹`` -> ``INR``) map
    every produced character back to the whole source token.
    """
    identity = [(index, index + 1) for index in range(len(text))]
    value, mapping = _map_currency(text, identity)
    value, mapping = _transform_chars(value, mapping, _char_variants)
    value, mapping = _transform_chars(value, mapping, _expand_digits)
    value, mapping = _map_replace(value, mapping, _GROUPING_COMMA, "")
    value, mapping = _map_replace(value, mapping, _WHITESPACE, " ")
    start = len(value) - len(value.lstrip(" "))
    end = len(value.rstrip(" "))
    if end <= start:
        return "", []
    return value[start:end], mapping[start:end]


# --------------------------------------------------------------------------- span resolution


@dataclass(frozen=True)
class SpanMatch:
    start: int
    end: int
    quote: str
    mode: Literal["exact", "normalised", "repaired"]
    ratio: float

    @property
    def penalty(self) -> float:
        return 1.0 if self.mode == "exact" else REWRITE_CONFIDENCE_PENALTY


@dataclass(frozen=True)
class SpanResolution:
    status: Literal["resolved", "ambiguous", "unresolved"]
    match: SpanMatch | None = None
    detail: str = ""

    @property
    def repaired(self) -> bool:
        return self.match is not None and self.match.mode == "repaired"


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    import difflib

    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def _unique(haystack: str, needle: str) -> tuple[int, int] | Literal["ambiguous"] | None:
    first = haystack.find(needle)
    if first == -1:
        return None
    if haystack.find(needle, first + 1) != -1:
        return "ambiguous"
    return first, first + len(needle)


def _match_exact(text: str, quote: str) -> SpanResolution:
    found = _unique(text, quote)
    if found == "ambiguous":
        return SpanResolution("ambiguous", detail="quote occurs more than once in the text")
    if found is None:
        return SpanResolution("unresolved", detail="quote does not occur in the text")
    start, end = found
    return SpanResolution("resolved", SpanMatch(start, end, text[start:end], "exact", 1.0))


def _match_canonical(text: str, quote: str) -> SpanResolution:
    canonical_text, mapping = canonicalise(text)
    canonical_quote, _ = canonicalise(quote)
    if not canonical_quote or not canonical_text:
        return SpanResolution("unresolved", detail="quote is empty after canonicalisation")
    found = _unique(canonical_text, canonical_quote)
    if found == "ambiguous":
        return SpanResolution(
            "ambiguous", detail="quote matches more than one normalised occurrence"
        )
    if found is None:
        return SpanResolution("unresolved", detail="quote does not occur after normalisation")
    start, end = found
    original_start = mapping[start][0]
    original_end = mapping[end - 1][1]
    matched = text[original_start:original_end]
    return SpanResolution(
        "resolved",
        SpanMatch(original_start, original_end, matched, "normalised", _ratio(quote, matched)),
    )


def _repair_candidates(quote: str) -> list[str]:
    """Exactly one deterministic repair pass; no fuzzy window search."""
    candidates: list[str] = []
    stripped = quote.strip(" \t\r\n.,;:()[]{}\"'")
    if stripped and stripped != quote:
        candidates.append(stripped)
    lines = [line.strip() for line in quote.splitlines() if line.strip()]
    if len(lines) > 1:
        candidates.append(max(lines, key=len))
    elif lines and lines[0] != quote:
        candidates.append(lines[0])
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_quote(text: str, quote: str, *, repair: bool = True) -> SpanResolution:
    """Resolve a quote to a unique document span, deterministically."""
    raw = (quote or "").strip()
    if not raw:
        return SpanResolution("unresolved", detail="empty quote")
    resolution = _match_exact(text, raw)
    if resolution.status != "unresolved":
        return resolution
    resolution = _match_canonical(text, raw)
    if resolution.status != "unresolved":
        return resolution
    if not repair:
        return resolution
    for candidate in _repair_candidates(raw):
        for helper in (_match_exact, _match_canonical):
            repaired = helper(text, candidate)
            if repaired.status == "ambiguous":
                return repaired
            if repaired.status == "resolved" and repaired.match is not None:
                return SpanResolution(
                    "resolved",
                    SpanMatch(
                        repaired.match.start,
                        repaired.match.end,
                        repaired.match.quote,
                        "repaired",
                        repaired.match.ratio,
                    ),
                )
    return resolution


# --------------------------------------------------------------------------- parsing


def parse_number(value: JsonScalar | None) -> float | None:
    """Parse a printed number; ``None`` when it needs a human (lakh/crore, text)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    clean, _ = canonicalise(str(value).strip())
    clean = re.sub(r"^(?:INR|USD|GBP|EUR|AUD|CAD|AED|SGD|JPY|CHF)\s*", "", clean).strip()
    clean = clean.replace("_", "").strip()
    clean = re.sub(r"\s*%$", "", clean).strip()
    clean = re.sub(r"\s*percent$", "", clean, flags=re.I).strip()
    match = re.match(r"^(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<rest>.*)$", clean)
    if match is None:
        return None
    if _MAGNITUDE.search(match.group("rest")):
        return None
    try:
        return float(match.group("number").replace(",", ""))
    except ValueError:
        return None


def _as_claim_value(number: float) -> int | float:
    return int(number) if float(number).is_integer() else number


def _first_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    if match is None:
        return None
    return float(match.group(0))


def _period_from_span(text: str) -> str | None:
    for period, pattern in _PERIOD_PATTERNS:
        if pattern.search(text):
            return period
    return None


def _parse_date(text: str) -> str | None:
    for fmt, pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            return datetime.strptime(match.group(1), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_boolean(text: str) -> bool | None:
    lowered = text.lower()
    negatives = (
        "not included",
        "not counted",
        "not considered",
        "excluded",
        "excludes",
        "does not",
        "false",
        r"\bno\b",
        r"\bnot\b",
        r"\bnever\b",
    )
    positives = (
        "included",
        "counted",
        "considered",
        "yes",
        "true",
        "required",
        "mandatory",
        "requires",
    )
    for marker in negatives:
        if re.search(marker, lowered):
            return False
    for marker in positives:
        if re.search(marker, lowered):
            return True
    return None


@dataclass(frozen=True)
class DerivedValue:
    value: Any
    unit: str | None = None
    period: str | None = None
    reason: RejectionReason | None = None
    detail: str = ""


def _derive(
    spec: FieldSpec,
    span_text: str,
    *,
    default_currency: str | None = None,
) -> DerivedValue:
    """Derive ``value``/``unit`` from the resolved span with a typed parser."""
    kind = spec.unit_kind
    clean, _ = canonicalise(span_text)
    if kind == "money":
        currency_match = _CURRENCY_CODE.search(clean)
        currency = currency_match.group(1) if currency_match else default_currency
        if currency is None:
            return DerivedValue(
                None,
                reason="unparseable_unit",
                detail="money span prints no currency; a currency cannot be inferred",
            )
        number = _first_number(clean)
        if number is None:
            return DerivedValue(
                None,
                reason="unparseable_value",
                detail=f"money span {span_text!r} has no parseable number",
            )
        if _MAGNITUDE.search(clean):
            return DerivedValue(
                None,
                reason="unparseable_value",
                detail="magnitude words (lakh/crore) are not auto-scaled; quote the printed figure",
            )
        period = _period_from_span(clean)
        if period is None and not spec.period_required:
            period = spec.period
        return DerivedValue(_as_claim_value(number), unit=currency, period=period)
    if kind in {"count", "hours", "months", "years"}:
        number = _first_number(clean)
        if number is None:
            return DerivedValue(
                None,
                reason="unparseable_value",
                detail=f"span {span_text!r} has no parseable integer",
            )
        if _MAGNITUDE.search(clean):
            return DerivedValue(
                None,
                reason="unparseable_value",
                detail="magnitude words are not auto-scaled; quote the printed figure",
            )
        if not float(number).is_integer():
            return DerivedValue(
                None,
                reason="unparseable_value",
                detail=f"span {span_text!r} is not a whole count",
            )
        return DerivedValue(int(number), unit=str(kind))
    if kind == "percent":
        ratio = _PERCENT_RATIO.search(clean)
        if ratio is not None:
            numerator = float(ratio.group(1))
            denominator = float(ratio.group(2))
            if denominator <= 0:
                return DerivedValue(
                    None, reason="unparseable_value", detail="rate has a zero denominator"
                )
            return DerivedValue(_as_claim_value(numerator / denominator), unit="ratio")
        if not _PERCENT_MARK.search(clean):
            return DerivedValue(
                None,
                reason="unparseable_value",
                detail="rate span prints no percent sign or x/y ratio",
            )
        number = _first_number(clean)
        if number is None:
            return DerivedValue(
                None, reason="unparseable_value", detail="rate span has no parseable number"
            )
        return DerivedValue(_as_claim_value(number / 100), unit="ratio")
    if kind == "date":
        parsed = _parse_date(span_text)
        if parsed is None:
            return DerivedValue(
                None,
                reason="unparseable_value",
                detail=f"span {span_text!r} is not a recognisable date",
            )
        return DerivedValue(parsed)
    if kind == "boolean":
        parsed_bool = _parse_boolean(span_text)
        if parsed_bool is None:
            return DerivedValue(
                None,
                reason="unparseable_value",
                detail=f"span {span_text!r} does not state yes/no",
            )
        return DerivedValue(parsed_bool)
    if kind == "grade":
        match = _GRADE.search(span_text)
        if match is None:
            return DerivedValue(
                None, reason="unparseable_value", detail=f"span {span_text!r} has no grade"
            )
        return DerivedValue(match.group(1), unit="grade")
    # text and untyped fields: the quote is the value, optionally narrowed.
    value = span_text.strip()
    if spec.value_pattern:
        match = re.search(spec.value_pattern, value, re.I | re.S)
        if match is not None:
            value = (match.group(1) if match.groups() else match.group(0)).strip()
    value = value.rstrip(".")
    if not value:
        return DerivedValue(
            None, reason="unparseable_value", detail="text span is empty after narrowing"
        )
    return DerivedValue(value)


# --------------------------------------------------------------------------- validation


@dataclass
class _ResolvedField:
    name: str
    spec: FieldSpec
    extracted: ExtractedField
    resolution: SpanResolution
    confidence: float
    derived: DerivedValue | None = None
    rejection: Rejection | None = None

    @property
    def quote(self) -> str | None:
        return self.resolution.match.quote if self.resolution.match else None


def validate_document(
    document: FamilyDocument,
    text: str,
    *,
    extraction_id: int,
    entity_id: int,
    bands: Bands | None = None,
    observed_at: datetime | None = None,
    fuzzy_threshold: float | None = None,
    default_confidence: float = 0.5,
    repair: bool = True,
) -> ValidationResult:
    """Turn quote-first guided JSON into validated ``ClaimCreate`` rows.

    ``fuzzy_threshold`` is accepted for API stability but no longer used: the new
    contract never searches fuzzy windows, because a near-match is not a span.
    """
    del fuzzy_threshold
    bands = bands or Bands.load()
    result = ValidationResult()

    resolved: list[_ResolvedField] = []
    for name, spec, extracted in document.fields():
        if extracted is None or not (extracted.quote or "").strip():
            # Null is a value: "the document did not say". Emit nothing, reject nothing.
            result.outcomes.append(
                FieldOutcome(
                    field=name,
                    predicate=spec.predicate,
                    outcome="abstained",
                    detail=(extracted.reason_if_null if extracted is not None else "") or "",
                )
            )
            continue
        resolution = resolve_quote(text, str(extracted.quote), repair=repair)
        confidence = (
            extracted.confidence if extracted.confidence is not None else default_confidence
        )
        confidence = max(0.0, min(1.0, confidence))
        entry = _ResolvedField(
            name=name,
            spec=spec,
            extracted=extracted,
            resolution=resolution,
            confidence=confidence,
        )
        if resolution.status == "ambiguous":
            entry.rejection = Rejection(
                field=name,
                predicate=spec.predicate,
                reason="ambiguous_span",
                detail=resolution.detail,
                quote=extracted.quote,
            )
        elif resolution.status == "unresolved":
            entry.rejection = Rejection(
                field=name,
                predicate=spec.predicate,
                reason="unresolved_span",
                detail=(resolution.detail or "quote does not resolve")
                + "; abstaining after one repair pass",
                quote=extracted.quote,
            )
        resolved.append(entry)
        if (
            entry.rejection is None
            and resolution.match is not None
            and resolution.match.mode != "exact"
        ):
            result.rewrites.append(
                SpanRewrite(
                    field=name,
                    predicate=spec.predicate,
                    original_quote=str(extracted.quote),
                    corrected_quote=resolution.match.quote,
                    before_start=None,
                    before_end=None,
                    after_start=resolution.match.start,
                    after_end=resolution.match.end,
                    ratio=resolution.match.ratio,
                )
            )

    default_currency = _context_currency(resolved)
    for entry in resolved:
        if entry.rejection is not None or entry.resolution.match is None:
            continue
        entry.derived = _derive(
            entry.spec,
            entry.resolution.match.quote,
            default_currency=default_currency,
        )
        if entry.derived.reason is not None:
            entry.rejection = Rejection(
                field=entry.name,
                predicate=entry.spec.predicate,
                reason=entry.derived.reason,
                detail=entry.derived.detail,
                quote=entry.quote,
            )

    guards = _cross_guards(resolved)

    context = _context_values(resolved)
    for entry in resolved:
        if entry.rejection is not None:
            result.rejections.append(entry.rejection)
            result.outcomes.append(_outcome(entry))
            logger.warning(
                "rejected %s (%s): %s", entry.name, entry.rejection.reason, entry.rejection.detail
            )
            continue
        guard = guards.get(entry.name)
        if guard is not None:
            guard_name, detail = guard
            entry.rejection = Rejection(
                field=entry.name,
                predicate=entry.spec.predicate,
                reason="guard_violation",
                detail=detail,
                quote=entry.quote,
                guard=guard_name,
            )
            result.rejections.append(entry.rejection)
            result.outcomes.append(_outcome(entry))
            logger.warning("guard %s rejected %s: %s", guard_name, entry.name, detail)
            continue
        claim = _build_claim(
            entry,
            context=context,
            bands=bands,
            extraction_id=extraction_id,
            entity_id=entity_id,
            observed_at=observed_at,
            resolved=resolved,
        )
        if isinstance(claim, Rejection):
            entry.rejection = claim
            result.rejections.append(claim)
            result.outcomes.append(_outcome(entry))
            continue
        result.claims.append(claim)
        result.outcomes.append(_outcome(entry))
    return result


def _outcome(entry: _ResolvedField) -> FieldOutcome:
    match = entry.resolution.match
    if entry.rejection is None:
        outcome = "resolved"
        detail = match.mode if match is not None else ""
    elif entry.rejection.guard is not None:
        outcome = f"rejected_guard:{entry.rejection.guard}"
        detail = entry.rejection.detail
    elif entry.rejection.reason == "ambiguous_span":
        outcome = "ambiguous"
        detail = entry.rejection.detail
    elif entry.rejection.reason == "unresolved_span":
        outcome = "unresolved"
        detail = entry.rejection.detail
    else:
        outcome = "rejected"
        detail = f"{entry.rejection.reason}: {entry.rejection.detail}"
    return FieldOutcome(
        field=entry.name,
        predicate=entry.spec.predicate,
        outcome=outcome,
        detail=detail,
        quote=match.quote if match is not None else entry.extracted.quote,
        char_start=match.start if match is not None else None,
        char_end=match.end if match is not None else None,
        repaired=entry.resolution.repaired,
    )


def _context_currency(resolved: list[_ResolvedField]) -> str | None:
    for entry in resolved:
        if entry.spec.predicate != "fee.currency" or entry.resolution.match is None:
            continue
        match = _CURRENCY_CODE.search(entry.resolution.match.quote.upper())
        if match is not None:
            return match.group(1)
    return None


def _context_values(resolved: list[_ResolvedField]) -> dict[str, JsonScalar]:
    values: dict[str, JsonScalar] = {}
    for entry in resolved:
        if entry.spec.context and entry.derived is not None and entry.derived.value is not None:
            values[entry.name] = entry.derived.value
    return values


def _cross_guards(resolved: list[_ResolvedField]) -> dict[str, tuple[str, str]]:
    """Semantic guards that hold regardless of what the model said."""
    guards: dict[str, tuple[str, str]] = {}
    by_predicate = {
        entry.spec.predicate: entry
        for entry in resolved
        if entry.rejection is None and entry.derived is not None
    }

    def values(predicate: str) -> tuple[str, Any] | None:
        entry = by_predicate.get(predicate)
        if entry is None or entry.derived is None or entry.derived.value is None:
            return None
        return entry.name, entry.derived.value

    placed = values("placement.placed_count")
    eligible = values("placement.eligible_count")
    rate = values("placement.rate")

    if placed is not None and eligible is not None:
        placed_value = float(placed[1])
        eligible_value = float(eligible[1])
        if placed_value > eligible_value:
            guards[placed[0]] = (
                "placed_exceeds_eligible",
                f"placed {placed_value:g} exceeds eligible {eligible_value:g}",
            )
    if rate is not None:
        rate_value = float(rate[1])
        if rate_value > 1.0:
            guards[rate[0]] = (
                "placement_rate_above_100",
                f"placement rate {rate_value:g} exceeds 100%",
            )
        elif eligible is not None:
            eligible_value = float(eligible[1])
            implied = rate_value * eligible_value
            implied = min(implied, eligible_value)
            if placed is not None:
                placed_value = float(placed[1])
                tolerance = max(1.0, 0.05 * eligible_value)
                if abs(implied - placed_value) > tolerance:
                    guards[rate[0]] = (
                        "rate_inconsistent_with_counts",
                        f"rate {rate_value:g} x eligible {eligible_value:g} = "
                        f"{implied:.0f}, not placed {placed_value:g}",
                    )

    total = values("seats.total")
    government = values("seats.government_quota")
    management = values("seats.management_quota")
    if total is not None:
        total_value = float(total[1])
        for quota in (government, management):
            if quota is not None and float(quota[1]) > total_value:
                guards[quota[0]] = (
                    "quota_exceeds_sanctioned_intake",
                    f"quota {float(quota[1]):g} exceeds sanctioned intake {total_value:g}",
                )
        if government is not None and management is not None:
            combined = float(government[1]) + float(management[1])
            if combined > total_value:
                for quota in (government, management):
                    guards.setdefault(
                        quota[0],
                        (
                            "quotas_exceed_sanctioned_intake",
                            f"quota total {combined:g} exceeds sanctioned intake {total_value:g}",
                        ),
                    )
    return guards


def _build_claim(  # noqa: PLR0913 - explicit gates are clearer than a bag of state
    entry: _ResolvedField,
    *,
    context: dict[str, JsonScalar],
    bands: Bands,
    extraction_id: int,
    entity_id: int,
    observed_at: datetime | None,
    resolved: list[_ResolvedField],
) -> Rejection | ClaimCreate:
    match = entry.resolution.match
    if match is None or entry.derived is None:
        return Rejection(
            field=entry.name,
            predicate=entry.spec.predicate,
            reason="unresolved_span",
            detail="no resolved span to derive a value from",
            quote=entry.extracted.quote,
        )
    spec = entry.spec
    value = entry.derived.value
    unit = entry.derived.unit

    if spec.unit_required and unit is None and spec.unit_kind not in {"date", "text", "boolean"}:
        return Rejection(
            field=entry.name,
            predicate=spec.predicate,
            reason="missing_period" if spec.period_required else "unparseable_unit",
            detail="quantity has no unit derivable from its span",
            quote=entry.quote,
        )
    if spec.period_required and entry.derived.period is None:
        return Rejection(
            field=entry.name,
            predicate=spec.predicate,
            reason="missing_period",
            detail="salary span states no period (per month / per year); unusable without one",
            quote=entry.quote,
        )
    if unit is not None and spec.unit_kind == "money" and unit not in bands.currencies:
        return Rejection(
            field=entry.name,
            predicate=spec.predicate,
            reason="unparseable_unit",
            detail=f"currency {unit!r} is not in the configured currency vocabulary",
            quote=entry.quote,
        )

    qualifier: dict[str, Any] = dict(context)
    if spec.is_rate:
        denominator = _denominator_lookup(entry, resolved)
        if denominator is None:
            return Rejection(
                field=entry.name,
                predicate=spec.predicate,
                reason="missing_denominator",
                detail=(
                    "rate has no denominator in its qualifier; the denominator must be "
                    f"extracted from the same document (expected {spec.denominator_field!r})"
                ),
                quote=entry.quote,
            )
        qualifier["denominator"] = _as_claim_value(denominator)
        qualifier["numerator"] = value

    band = bands.band(spec.predicate)
    if (
        band is not None
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and not band.contains(float(value))
    ):
        return Rejection(
            field=entry.name,
            predicate=spec.predicate,
            reason="out_of_band",
            detail=(
                f"value {value} is outside the plausibility band "
                f"[{band.minimum}, {band.maximum}] for {spec.predicate}"
            ),
            quote=entry.quote,
        )

    if spec.unit_kind == "money":
        qualifier["period"] = entry.derived.period
        qualifier["as_of"] = observed_at.date().isoformat() if observed_at is not None else None

    confidence = entry.confidence * match.penalty
    confidence = max(0.0, min(1.0, confidence))
    return ClaimCreate(
        extraction_id=extraction_id,
        entity_id=entity_id,
        predicate=spec.predicate,
        value=value,
        unit=unit,
        qualifier=qualifier or None,
        quote=match.quote,
        char_start=match.start,
        char_end=match.end,
        confidence=round(confidence, 4),
        observed_at=observed_at,
        method="extracted",
    )


def _denominator_lookup(entry: _ResolvedField, resolved: list[_ResolvedField]) -> float | None:
    """Find the sibling field that supplies a rate's denominator."""
    field_name = entry.spec.denominator_field
    if not field_name:
        return None
    for sibling in resolved:
        if sibling.name != field_name or sibling.derived is None:
            continue
        number = sibling.derived.value
        if isinstance(number, bool) or not isinstance(number, int | float):
            return None
        return float(number) if number > 0 else None
    return None
