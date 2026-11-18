"""Validation gates for extracted claims (brief §9).

The model emits a **quote**, never a number and never an offset. This module
turns that quote into a claim deterministically:

1. **Resolve the span.** Exact occurrence first; if absent, canonicalise both
   sides (NFKC, unified quotes/dashes, Devanagari digits to ASCII, ``₹``/``Rs.``/
   ``Rs``/``INR`` to ``INR``, Indian digit grouping removed, whitespace
   collapsed) and retry. Only a **unique** occurrence is accepted. Several
   matches are disambiguated deterministically by the field's optional
   ``context`` (the row label / column header / institute name):

   * ``unique`` — the quote occurs exactly once;
   * ``context_row`` — the quote is ambiguous but exactly one of its
     occurrences lies inside the table block whose anchor cell is the context
     (a block is the run of rows under one institute/heading cell, built from
     the acquisition table cells when available);
   * ``context_window`` — no table matched, but exactly one occurrence of the
     quote lies on the same line as, or within ``CONTEXT_WINDOW_CHARS`` of,
     exactly one occurrence of the context;
   * ``none`` — context was absent or did not disambiguate; the field abstains.

   A context that names a real table block but does not contain the quote is a
   contradiction and is rejected, never retried against a looser window. Zero
   matches get exactly one repair pass (strip edge punctuation / take the
   longest line) and then abstain.
2. **Derive the value.** The resolved span is parsed by a type-specific parser
   (count, money+period, percentage, integer, date, grade, boolean, text). No
   model-supplied number, unit or conversion is ever trusted.
3. **Apply semantic guards** that are independent of the model: a salary must
   carry a printed period, a rate's numerator cannot exceed its denominator, a
   placement rate cannot exceed 100%, placements cannot exceed eligible
   students, seat quotas cannot exceed the sanctioned total, and admitted
   students cannot exceed the sanctioned intake. Every money field must end up
   with a currency and a period **on the claim itself**: the period may come
   from the field's own span or from the document-level period sibling named by
   ``FieldSpec.period_source`` (``fee.period``, ``placement.salary_period``) —
   never from a default. A money claim that still has no period is rejected.

A claim is rejected, never silently fixed. Rejections and per-field outcomes
(``resolved``, ``ambiguous``, ``unresolved``, ``rejected_guard:<name>``) are
structured so the review console can render them verbatim.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
    "uncertain_column_header",
    "unparseable_value",
    "unparseable_unit",
    "missing_denominator",
    "missing_period",
    "out_of_band",
    "guard_violation",
]

OutcomeKind = Literal["resolved", "ambiguous", "unresolved", "rejected", "abstained"]
Disambiguation = Literal["unique", "context_row", "context_window", "none"]


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
    disambiguation: Disambiguation = "none"

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
            "disambiguation": self.disambiguation,
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
    # Devanagari words are printed period words, not a guess: वर्ष = year,
    # मासिक/माह = monthly, सेमेस्टर = semester (Hindi fee/salary pages).
    (
        "per_month",
        re.compile(r"per\s*month|monthly|p\.?\s*m\.?|/\s*month\b|मासिक|माह", re.I),
    ),
    (
        "per_semester",
        re.compile(r"per\s*semester|semester|/\s*sem\b|सेमेस्टर", re.I),
    ),
    (
        "per_year",
        re.compile(
            r"per\s*annum|per\s*year|annually|annual|p\.?\s*a\.?|/\s*year\b|\ba\s+year\b"
            r"|first[-\s]?year|वर्ष",
            re.I,
        ),
    ),
    ("one_time", re.compile(r"one[-\s]?time|registration\s+fee|application\s+fee", re.I)),
)

#: ``fee.period`` narrows to this vocabulary; each word maps to a claim period.
_DOCUMENT_PERIOD_WORDS = {
    "year": "per_year",
    "semester": "per_semester",
    "month": "per_month",
}

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
class RowIdentity:
    """Which table row a resolved value belongs to (provenance, not a guess)."""

    table_index: int
    table_page: int | None
    row_index: int
    row_label: str | None
    column_header: str | None
    value_cell: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "table_index": self.table_index,
            "table_page": self.table_page,
            "row_index": self.row_index,
            "row_label": self.row_label,
            "column_header": self.column_header,
            "value_cell": self.value_cell,
        }


@dataclass(frozen=True)
class SpanResolution:
    status: Literal["resolved", "ambiguous", "unresolved"]
    match: SpanMatch | None = None
    detail: str = ""
    disambiguation: Disambiguation = "none"
    context_quote: str | None = None
    context_span: tuple[int, int] | None = None
    row: RowIdentity | None = None
    row_span: tuple[int, int] | None = None
    column_uncertain: bool = False

    @property
    def repaired(self) -> bool:
        return self.match is not None and self.match.mode == "repaired"


#: Same-line or +/- this many characters: the fallback block for a quote whose
#: context is not a table cell. Deliberately small: a row is local, a page is not.
CONTEXT_WINDOW_CHARS = 200

_NUMERIC_CELL = re.compile(r"^\d[\d,]*(?:\.\d+)?\s*%?$")


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


# --------------------------------------------------------------------------- context
# A table row is only meaningful with its row header. The model supplies
# ``context``; this block turns it into a location, never a guess.
#
# Tables arrive from acquisition as duck-typed objects (or mappings) with
# ``rows``, ``page``, ``cells`` and ``column_header_uncertain`` so extraction
# stays independent of the acquisition package's concrete classes.


def _attr(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _cell_norm(value: Any) -> str:
    text, _ = canonicalise(str(value or ""))
    return text.strip()


def _rows_of(table: Any) -> list[list[str]]:
    rows = _attr(table, "rows", None) or []
    return [[str(cell) for cell in row] for row in rows]


def _is_numeric_cell(value: str | None) -> bool:
    return bool(value and _NUMERIC_CELL.match(_cell_norm(value)))


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    """Every occurrence, exact first, then canonical (mapped to source offsets)."""
    if not needle:
        return []
    exact: list[tuple[int, int]] = []
    start = text.find(needle)
    while start != -1:
        exact.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    if exact:
        return exact
    canonical_text, mapping = canonicalise(text)
    canonical_needle, _ = canonicalise(needle)
    if not canonical_needle or not canonical_text:
        return []
    hits: list[tuple[int, int]] = []
    start = canonical_text.find(canonical_needle)
    while start != -1 and start + len(canonical_needle) <= len(mapping):
        span = (mapping[start][0], mapping[start + len(canonical_needle) - 1][1])
        if span not in hits:
            hits.append(span)
        start = canonical_text.find(canonical_needle, start + 1)
    return hits


def _quote_occurrences(text: str, quote: str) -> list[tuple[int, int, str]]:
    """Quote hits as ``(start, end, mode)``: exact hits, else normalised hits."""
    hits: list[tuple[int, int, str]] = []
    start = text.find(quote)
    while start != -1:
        hits.append((start, start + len(quote), "exact"))
        start = text.find(quote, start + 1)
    if hits:
        return hits
    for start, end in _occurrences(text, quote):
        hits.append((start, end, "normalised"))
    return hits


@dataclass(frozen=True)
class _TableBlock:
    table_index: int
    page: int | None
    context_text: str
    start: int
    end: int
    rows: list[list[str]]
    first_row: int
    last_row: int
    anchor_col: int
    column_header_uncertain: bool


def _next_anchor_text(tables: list[Any], table_index: int, last_row: int, col: int) -> str | None:
    """The cell that closes this block: next row's anchor, else the next table."""
    rows = _rows_of(tables[table_index])
    if last_row < len(rows) and col < len(rows[last_row]) and _cell_norm(rows[last_row][col]):
        return rows[last_row][col]
    for index in range(table_index + 1, len(tables)):
        for row in _rows_of(tables[index]):
            for cell in row:
                if _cell_norm(cell):
                    return cell
    return None


def _table_block(text: str, context: str, tables: list[Any]) -> _TableBlock | None:
    """Locate the table block whose anchor cell is exactly this context.

    A block starts at the row that prints the header cell (institute, group,
    scheme) and runs until the same column is filled again. Exactly one such
    anchor may match: a context that names two blocks does not disambiguate.
    """
    context_norm = _cell_norm(context)
    if not context_norm:
        return None
    anchors: list[tuple[int, int, int]] = []
    for table_index, table in enumerate(tables):
        for row_index, row in enumerate(_rows_of(table)):
            for col_index, cell in enumerate(row):
                if _cell_norm(cell) == context_norm:
                    anchors.append((table_index, row_index, col_index))
    if len(anchors) != 1:
        return None
    table_index, row_index, col_index = anchors[0]
    rows = _rows_of(tables[table_index])
    last_row = len(rows)
    for candidate in range(row_index + 1, len(rows)):
        row = rows[candidate]
        if col_index < len(row) and _cell_norm(row[col_index]):
            last_row = candidate
            break
    context_text = rows[row_index][col_index]
    context_hits = _occurrences(text, context_text)
    if len(context_hits) != 1:
        return None
    start = context_hits[0][0]
    end = len(text)
    anchor = _next_anchor_text(tables, table_index, last_row, col_index)
    if anchor:
        boundary = text.find(anchor, start)
        if boundary == -1:
            boundary = text.find(_cell_norm(anchor), start)
        if boundary > start:
            end = boundary
    if end <= start:
        return None
    return _TableBlock(
        table_index=table_index,
        page=_attr(tables[table_index], "page", None),
        context_text=context_text,
        start=start,
        end=end,
        rows=rows,
        first_row=row_index,
        last_row=last_row,
        anchor_col=col_index,
        column_header_uncertain=bool(_attr(tables[table_index], "column_header_uncertain", False)),
    )


def _column_header_for(
    tables: list[Any], table_index: int, row_index: int, value_cell: str | None
) -> tuple[str | None, bool]:
    """Header text and uncertainty for the row's numeric cell.

    ``column_header_uncertain`` is acquisition's merged-cell safety flag. When
    per-cell links exist they decide cell by cell; only when a table carries no
    links does the table-wide flag apply.
    """
    table = tables[table_index]
    links = _attr(table, "cells", None) or ()
    if not links:
        return None, bool(_attr(table, "column_header_uncertain", False))
    wanted = _cell_norm(value_cell)
    row_links = [
        link
        for link in links
        if int(_attr(link, "row", -1)) == row_index
        and _is_numeric_cell(str(_attr(link, "text", "") or ""))
        and (not wanted or _cell_norm(_attr(link, "text", "")) == wanted)
    ]
    if not row_links:
        return None, False
    link = row_links[-1]
    return (
        _attr(link, "header_text", None),
        bool(_attr(link, "ambiguous", False)),
    )


def _identity_for_row(
    text: str,
    tables: list[Any],
    table_index: int,
    row_index: int,
    row: list[str],
    quote: str,
    match_span: tuple[int, int],
    search_from: int,
) -> tuple[RowIdentity, tuple[int, int] | None, bool]:
    """Build the row identity (label, value cell, header) for one known row."""
    labels = [cell.strip() for cell in row if cell.strip() and not _is_numeric_cell(cell)]
    row_label = labels[-1] if labels else None
    tokens = _cell_norm(quote).split()
    last_token = tokens[-1] if tokens else ""
    numerics = [cell.strip() for cell in row if _is_numeric_cell(cell)]
    value_cell = next(
        (cell for cell in reversed(numerics) if _cell_norm(cell) == last_token),
        numerics[-1] if numerics else None,
    )
    header, uncertain = _column_header_for(tables, table_index, row_index, value_cell)
    row_span: tuple[int, int] | None = None
    if row_label:
        position = text.rfind(row_label, search_from, match_span[1])
        if position != -1:
            row_span = (position, match_span[1])
    identity = RowIdentity(
        table_index=table_index,
        table_page=_attr(tables[table_index], "page", None),
        row_index=row_index,
        row_label=row_label,
        column_header=header,
        value_cell=value_cell,
    )
    return identity, row_span, uncertain


def _row_identity(
    text: str, block: _TableBlock, quote: str, tables: list[Any], match_span: tuple[int, int]
) -> tuple[RowIdentity | None, tuple[int, int] | None, bool]:
    """Which row of the block the matched span belongs to, plus its provenance."""
    quote_norm = _cell_norm(quote)
    candidates: list[int] = []
    for row_index in range(block.first_row, block.last_row):
        joined = " ".join(_cell_norm(cell) for cell in block.rows[row_index] if _cell_norm(cell))
        if quote_norm and quote_norm in joined:
            candidates.append(row_index)
    if len(candidates) != 1:
        return None, None, False
    row_index = candidates[0]
    identity, row_span, uncertain = _identity_for_row(
        text,
        tables,
        block.table_index,
        row_index,
        block.rows[row_index],
        quote,
        match_span,
        block.start,
    )
    return identity, row_span, uncertain


def locate_row(
    text: str, quote: str, tables: list[Any], match_span: tuple[int, int]
) -> tuple[RowIdentity, tuple[int, int] | None, bool] | None:
    """Attach row provenance to an already-unique span, if exactly one row holds it.

    Used for numeric fields so the value is parsed from the row's numeric cell
    (never the serial number) and so an uncertain column header still rejects
    the claim. More than one matching row ⇒ no attribution (the span was unique
    by itself; guessing a row would add nothing).
    """
    quote_norm = _cell_norm(quote)
    if not quote_norm:
        return None
    candidates: list[tuple[int, int]] = []
    for table_index, table in enumerate(tables):
        for row_index, row in enumerate(_rows_of(table)):
            joined = " ".join(_cell_norm(cell) for cell in row if _cell_norm(cell))
            if quote_norm in joined:
                candidates.append((table_index, row_index))
    if len(candidates) != 1:
        return None
    table_index, row_index = candidates[0]
    rows = _rows_of(tables[table_index])
    return _identity_for_row(
        text,
        tables,
        table_index,
        row_index,
        rows[row_index],
        quote,
        match_span,
        0,
    )


def _resolve_in_block(
    text: str, quote: str, block: _TableBlock, tables: list[Any], detail: str
) -> SpanResolution:
    hits = _quote_occurrences(text, quote)
    inside = [hit for hit in hits if block.start <= hit[0] and hit[1] <= block.end]
    if len(inside) != 1:
        if not inside:
            reason = (
                f"context {block.context_text!r} names a table block that does not "
                "contain the quote"
            )
        else:
            reason = (
                f"{len(inside)} occurrences of the quote fall in the block headed by "
                f"{block.context_text!r}"
            )
        return SpanResolution("ambiguous", detail=f"{detail}; {reason}")
    start, end, mode = inside[0]
    matched = text[start:end]
    identity, row_span, uncertain = _row_identity(text, block, quote, tables, (start, end))
    return SpanResolution(
        "resolved",
        SpanMatch(
            start,
            end,
            matched,
            mode,  # type: ignore[arg-type]
            1.0 if mode == "exact" else _ratio(quote, matched),
        ),
        disambiguation="context_row",
        context_quote=block.context_text,
        context_span=(block.start, block.start + len(block.context_text)),
        row=identity,
        row_span=row_span,
        column_uncertain=uncertain,
    )


def _same_line(text: str, start: int, end: int, position: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return line_start <= position < line_end


def _resolve_in_window(
    text: str,
    quote: str,
    hits: list[tuple[int, int, str]],
    context: str,
    detail: str,
) -> SpanResolution:
    context_hits = _occurrences(text, context)
    if not context_hits:
        return SpanResolution(
            "ambiguous", detail=f"{detail}; context {context!r} is not in the text"
        )
    candidates: list[tuple[tuple[int, int, str], tuple[int, int]]] = []
    for hit in hits:
        near = [
            span
            for span in context_hits
            if _same_line(text, hit[0], hit[1], span[0])
            or (
                hit[0] - CONTEXT_WINDOW_CHARS <= span[0]
                and span[1] <= hit[1] + CONTEXT_WINDOW_CHARS
            )
        ]
        if len(near) == 1:
            candidates.append((hit, near[0]))
    if len(candidates) != 1:
        found = len(candidates)
        reason = (
            "no occurrence of the quote is next to exactly one occurrence of the context"
            if found == 0
            else f"{found} occurrences of the quote are next to the context"
        )
        return SpanResolution("ambiguous", detail=f"{detail}; {reason}")
    (start, end, mode), context_span = candidates[0]
    matched = text[start:end]
    return SpanResolution(
        "resolved",
        SpanMatch(
            start,
            end,
            matched,
            mode,  # type: ignore[arg-type]
            1.0 if mode == "exact" else _ratio(quote, matched),
        ),
        disambiguation="context_window",
        context_quote=context,
        context_span=context_span,
    )


def _resolve_ambiguous(
    text: str,
    quote: str,
    detail: str,
    *,
    context: str | None,
    tables: list[Any] | None,
) -> SpanResolution:
    context_text = (context or "").strip()
    if not context_text:
        return SpanResolution("ambiguous", detail=f"{detail}; no row context was supplied")
    hits = _quote_occurrences(text, quote)
    if not hits:
        return SpanResolution("ambiguous", detail=detail)
    if tables:
        block = _table_block(text, context_text, tables)
        if block is not None:
            return _resolve_in_block(text, quote, block, tables, detail)
    return _resolve_in_window(text, quote, hits, context_text, detail)


def _tag(resolution: SpanResolution, rule: Disambiguation) -> SpanResolution:
    return SpanResolution(
        resolution.status,
        resolution.match,
        resolution.detail,
        disambiguation=rule,
        context_quote=resolution.context_quote,
        context_span=resolution.context_span,
        row=resolution.row,
        row_span=resolution.row_span,
        column_uncertain=resolution.column_uncertain,
    )


def resolve_quote(
    text: str,
    quote: str,
    *,
    context: str | None = None,
    tables: list[Any] | None = None,
    repair: bool = True,
) -> SpanResolution:
    """Resolve a quote to a unique document span, deterministically.

    Precedence: an exact unique quote; a canonical unique quote; the optional
    ``context`` (table block first, bounded window second). Nothing here tries
    a second candidate after the first fails: a row that cannot be located is
    ``ambiguous_span``.
    """
    raw = (quote or "").strip()
    if not raw:
        return SpanResolution("unresolved", detail="empty quote")
    resolution = _match_exact(text, raw)
    if resolution.status == "resolved":
        return _tag(resolution, "unique")
    if resolution.status == "unresolved":
        canonical = _match_canonical(text, raw)
        if canonical.status == "resolved":
            return _tag(canonical, "unique")
        resolution = canonical
    if resolution.status == "ambiguous":
        return _resolve_ambiguous(text, raw, resolution.detail, context=context, tables=tables)
    if not repair:
        return resolution
    for candidate in _repair_candidates(raw):
        for helper in (_match_exact, _match_canonical):
            repaired = helper(text, candidate)
            if repaired.status == "ambiguous":
                return _resolve_ambiguous(
                    text, raw, repaired.detail, context=context, tables=tables
                )
            if repaired.status == "resolved" and repaired.match is not None:
                return _tag(
                    SpanResolution(
                        "resolved",
                        SpanMatch(
                            repaired.match.start,
                            repaired.match.end,
                            repaired.match.quote,
                            "repaired",
                            repaired.match.ratio,
                        ),
                    ),
                    "unique",
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
    #: Predicate of the sibling claim that supplied the period, when the span
    #: itself printed none. Review-console provenance, never a model guess.
    period_from: str | None = None


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


_NUMERIC_UNIT_KINDS = frozenset({"count", "percent", "hours", "months", "years"})
#: Kinds whose span is attributed to a table row (value cell + column header).
#: Money is included for the header-safety gate, but its value is still parsed
#: from the whole quote: a money cell is often printed without its currency,
#: which lives in the column header.
_ROW_SCOPED_UNIT_KINDS = _NUMERIC_UNIT_KINDS | {"money"}


def _value_span_text(entry: _ResolvedField) -> str:
    """The span the value is parsed from.

    A row-scoped quote legitimately contains the row label ("3. B.Sc. ... 20"),
    and the first number in that string is the serial number, not the intake.
    When the resolver identified the row's numeric cell, the value is parsed
    from **that cell**, never from the first digit in the quote.
    """
    match = entry.resolution.match
    if match is None:
        return ""
    row = entry.resolution.row
    if row is not None and row.value_cell and entry.spec.unit_kind in _NUMERIC_UNIT_KINDS:
        return row.value_cell
    return match.quote


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
    tables: list[Any] | None = None,
) -> ValidationResult:
    """Turn quote-first guided JSON into validated ``ClaimCreate`` rows.

    ``tables`` are the acquisition table objects for this document (``rows``,
    ``page``, ``cells``, ``column_header_uncertain``). They are used only to
    turn an ambiguous quote plus its row ``context`` into one location, and to
    refuse a value whose column header is uncertain.

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
        resolution = resolve_quote(
            text,
            str(extracted.quote),
            context=extracted.context,
            tables=tables,
            repair=repair,
        )
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
        elif (
            tables
            and resolution.row is None
            and spec.unit_kind in _ROW_SCOPED_UNIT_KINDS
            and resolution.match is not None
        ):
            # A unique quote may still print a serial number before the value,
            # and its column may still be ambiguous. Attribute the row, if one
            # row holds this exact span.
            located = locate_row(
                text,
                str(extracted.quote),
                tables,
                (resolution.match.start, resolution.match.end),
            )
            if located is not None:
                identity, row_span, uncertain = located
                resolution = replace(
                    resolution,
                    row=identity,
                    row_span=row_span,
                    column_uncertain=uncertain,
                )
                entry.resolution = resolution
        if (
            entry.rejection is None
            and resolution.column_uncertain
            and spec.unit_kind not in {"text", "date", "boolean", "grade"}
        ):
            # A resolving quote that points at the wrong column is worse than
            # null: the number would be attributed to a field it does not belong to.
            entry.rejection = Rejection(
                field=name,
                predicate=spec.predicate,
                reason="uncertain_column_header",
                detail=(
                    "the numeric cell's column header association is uncertain "
                    f"(acquisition flag); row context {resolution.context_quote!r} "
                    "is not a safe attribution"
                ),
                quote=resolution.match.quote if resolution.match else extracted.quote,
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
            _value_span_text(entry),
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

    _apply_period_sources(resolved)

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
        disambiguation=entry.resolution.disambiguation,
    )


def _context_currency(resolved: list[_ResolvedField]) -> str | None:
    for entry in resolved:
        if entry.spec.predicate != "fee.currency" or entry.resolution.match is None:
            continue
        match = _CURRENCY_CODE.search(entry.resolution.match.quote.upper())
        if match is not None:
            return match.group(1)
    return None


def _sibling_period_value(text: str) -> str | None:
    """Canonical claim period for a period sibling's quoted text.

    Accepts both the printed phrase (``per annum``) and the narrowed
    ``fee.period`` vocabulary (``year`` | ``semester`` | ``month``). Anything
    else returns ``None``: an unmappable phrase is not a period.
    """
    period = _period_from_span(text)
    if period is not None:
        return period
    return _DOCUMENT_PERIOD_WORDS.get(text.strip().casefold())


def _apply_period_sources(resolved: list[_ResolvedField]) -> None:
    """Fill a money field's period from the sibling claim the schema names.

    A fee table prints the period once (a header or a note), not on every
    cell. When a money span carries no period of its own, the sibling named by
    ``FieldSpec.period_source`` (``fee.period``, ``placement.salary_period``)
    supplies it from its own quoted claim, and the money claim records that
    provenance. A money field whose period is still missing is left ``None``
    and is rejected downstream by ``period_required`` — never defaulted.
    """
    by_name = {entry.name: entry for entry in resolved}
    for entry in resolved:
        if entry.rejection is not None or entry.derived is None:
            continue
        if entry.derived.period is not None or entry.spec.period_source is None:
            continue
        sibling = by_name.get(entry.spec.period_source)
        if sibling is None or sibling.rejection is not None:
            continue
        if sibling.derived is None or sibling.derived.value is None:
            continue
        period = _sibling_period_value(str(sibling.derived.value))
        if period is None:
            continue
        entry.derived = replace(entry.derived, period=period, period_from=sibling.spec.predicate)


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
    admitted = values("seats.admitted")
    if total is not None and admitted is not None:
        total_value = float(total[1])
        admitted_value = float(admitted[1])
        if admitted_value > total_value:
            guards[admitted[0]] = (
                "admitted_exceeds_sanctioned_intake",
                f"admitted {admitted_value:g} exceeds sanctioned intake {total_value:g}",
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
        source = (
            f" and its {spec.period_source!r} sibling states none" if spec.period_source else ""
        )
        return Rejection(
            field=entry.name,
            predicate=spec.predicate,
            reason="missing_period",
            detail=(
                "money span states no period (per month / per year / per semester)"
                f"{source}; unusable without one"
            ),
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
    qualifier["disambiguation"] = entry.resolution.disambiguation
    if entry.resolution.context_quote:
        qualifier["context_quote"] = entry.resolution.context_quote
    if entry.resolution.context_span is not None:
        qualifier["context_char_start"], qualifier["context_char_end"] = (
            entry.resolution.context_span
        )
    if entry.resolution.row is not None:
        qualifier["row"] = entry.resolution.row.as_dict()
    if entry.resolution.row_span is not None:
        qualifier["row_char_start"], qualifier["row_char_end"] = entry.resolution.row_span
    value_text = _value_span_text(entry)
    if value_text and value_text != match.quote:
        qualifier["value_quote"] = value_text
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
        if entry.derived.period_from is not None:
            qualifier["period_source"] = entry.derived.period_from
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
