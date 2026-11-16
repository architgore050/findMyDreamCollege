"""Claim views shared by scoring and hard filters.

Scoring and filtering both read append-only ``claims`` rows. They must never
mutate them and must never turn absence into a value: a missing claim is
``None`` here and ``UNKNOWN`` upstream.

Everything in this module is pure. Inputs may be pydantic ``ClaimRecord`` rows,
plain mappings (API JSON) or any object exposing the same attribute names.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Predicate vocabulary used by filters and sub-scores.
#
# These are data-model names (extraction schema), not country/weight/exam
# registry entries. Callers that need a different vocabulary can build a
# ClaimIndex over renamed predicates; nothing here branches on a country.
# ---------------------------------------------------------------------------

CREDENTIAL_LEVEL = "credential_level"
RECOGNITION_STATUS = "recognition_status"
ELIGIBILITY_MIN_AGGREGATE_PCT = "eligibility_min_aggregate_pct"
ELIGIBILITY_MIN_PCB_PCT = "eligibility_min_pcb_pct"
ELIGIBILITY_MAX_AGE = "eligibility_max_age"
ELIGIBILITY_REQUIRES_SUBJECTS = "eligibility_requires_subjects"
LANGUAGE_TEST_REQUIRED = "language_test_required"
LANGUAGE_MIN_SCORE = "language_min_score"
MEDIUM_OF_INSTRUCTION = "medium_of_instruction"
EXAM_REQUIRED = "exam_required"
EXAM_REGISTRATION_OPENS = "exam_registration_opens"
EXAM_REGISTRATION_CLOSES = "exam_registration_closes"
EXAM_REQUIRES_TRAVEL = "exam_requires_travel"
INTAKE_CYCLES = "intake_cycles"
TCOA = "tcoa"
TCOA_LO = "tcoa_lo"
TCOA_HI = "tcoa_hi"
NET_PRICE = "net_price"
DSR25 = "dsr25"
AFFORDABILITY_GAP = "affordability_gap"
SALARY_P25 = "p25_first_year_salary"
SALARY_COL_RATIO = "p25_salary_to_col_ratio"
INTERNSHIP_TO_OFFER = "internship_to_offer_conversion_12m"
REGISTRATION_CONVERSION = "registration_conversion_12m"

# A "rate" is a share of a stated cohort. A bare ratio (e.g. DSR25) or an
# index on a 0..100 scale is not a rate and must not be flagged for lacking a
# cohort denominator.
RATE_UNITS = frozenset({"rate", "fraction"})
RATE_PREDICATE_SUFFIXES = ("_rate", "_conversion_12m", "_share")
MONEY_UNITS = frozenset({"inr", "usd", "gbp", "eur", "aud", "cad", "currency"})

_PERCENT_UNITS = frozenset({"%", "percent", "pct", "percentage"})


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def is_rate_claim(claim: ClaimView) -> bool:
    """True when the claim asserts a rate/ratio and therefore needs a denominator."""

    unit = (claim.unit or "").strip().lower()
    if unit in RATE_UNITS:
        return True
    return claim.predicate.endswith(RATE_PREDICATE_SUFFIXES)


def is_money_claim(claim: ClaimView) -> bool:
    unit = (claim.unit or "").strip().lower()
    return unit in MONEY_UNITS or claim.predicate.startswith(("p25_", "salary_", "tcoa"))


def is_percent_claim(claim: ClaimView) -> bool:
    unit = (claim.unit or "").strip().lower()
    return unit in _PERCENT_UNITS or claim.predicate.endswith(("_pct", "_percent"))


def denominator(claim: ClaimView) -> float | None:
    """Return the denominator a rate claim is required to declare, if present."""

    for key in ("denominator", "n", "sample_size", "cohort_n"):
        raw = claim.qualifier.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def numeric(value: Any) -> float | None:
    """Coerce an extracted value to a finite float, or ``None``.

    Booleans are deliberately not numbers here; ``True`` must not become 1.0.
    """

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        try:
            number = float(cleaned)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def canonical_value(value: Any) -> Any:
    """JSON-safe, order-stable projection of a claim value for hashing."""

    if isinstance(value, Mapping):
        return {
            str(k): canonical_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_value(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ClaimView:
    """Read-only projection of a claim row."""

    predicate: str
    value: Any
    claim_id: int | str | None = None
    unit: str | None = None
    qualifier: Mapping[str, Any] = field(default_factory=dict)
    method: str = "extracted"
    confidence: float = 1.0
    observed_at: datetime | None = None
    source_id: str | int | None = None
    quote: str | None = None
    entity_id: int | None = None

    @property
    def effective_confidence(self) -> float:
        return min(1.0, max(0.0, float(self.confidence)))

    @property
    def number(self) -> float | None:
        return numeric(self.value)

    def canonical(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "predicate": self.predicate,
            "value": canonical_value(self.value),
            "unit": self.unit,
            "qualifier": canonical_value(dict(self.qualifier)),
            "method": self.method,
            "confidence": round(float(self.confidence), 6),
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "entity_id": self.entity_id,
        }


def _as_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def claim_view(raw: Any) -> ClaimView:
    """Normalise a claim row/dict into a :class:`ClaimView`."""

    if isinstance(raw, ClaimView):
        return raw
    qualifier = _get(raw, "qualifier", None)
    if qualifier is None:
        qualifier = _get(raw, "qualifier_json", None)
    value = _get(raw, "value", None)
    if value is None:
        value = _get(raw, "value_json", None)
    source_id = _get(raw, "source_id", None)
    if source_id is None:
        source_id = _get(raw, "snapshot_id", None)
    if source_id is None:
        source_id = _get(raw, "extraction_id", None)
    return ClaimView(
        predicate=str(_get(raw, "predicate", "")),
        value=value,
        claim_id=_get(raw, "claim_id", None)
        if _get(raw, "claim_id", None) is not None
        else _get(raw, "id", None),
        unit=_get(raw, "unit", None),
        qualifier=dict(qualifier or {}),
        method=str(_get(raw, "method", "extracted") or "extracted"),
        confidence=float(_get(raw, "confidence", 1.0) or 1.0),
        observed_at=_as_datetime(_get(raw, "observed_at", None)),
        source_id=source_id,
        quote=_get(raw, "quote", None),
        entity_id=_get(raw, "entity_id", None),
    )


def _sort_key(claim: ClaimView) -> tuple[str, str, str]:
    observed = claim.observed_at.isoformat() if claim.observed_at else ""
    return (claim.predicate, observed, str(claim.claim_id))


class ClaimIndex:
    """Deterministic, read-only index over claims."""

    __slots__ = ("_claims",)

    def __init__(self, claims: Iterable[ClaimView] = ()) -> None:
        self._claims = tuple(sorted(claims, key=_sort_key))

    @classmethod
    def build(cls, raw_claims: Iterable[Any] | None) -> ClaimIndex:
        if raw_claims is None:
            return cls(())
        return cls(claim_view(raw) for raw in raw_claims)

    def __iter__(self) -> Iterator[ClaimView]:
        return iter(self._claims)

    def __len__(self) -> int:
        return len(self._claims)

    def all(self, predicate: str | None = None) -> tuple[ClaimView, ...]:
        if predicate is None:
            return self._claims
        return tuple(claim for claim in self._claims if claim.predicate == predicate)

    def latest(self, predicate: str, *, exam_id: str | None = None) -> ClaimView | None:
        matches = self.all(predicate)
        if exam_id is not None:
            matching = [c for c in matches if str(c.qualifier.get("exam_id", "")) == str(exam_id)]
            if not matching:
                return None
            matches = tuple(matching)
        return matches[-1] if matches else None

    def first(self, predicate: str, *, exam_id: str | None = None) -> ClaimView | None:
        matches = self.all(predicate)
        if exam_id is not None:
            matches = tuple(
                c for c in matches if str(c.qualifier.get("exam_id", "")) == str(exam_id)
            )
        return matches[0] if matches else None

    def number(self, predicate: str, *, exam_id: str | None = None) -> float | None:
        claim = self.latest(predicate, exam_id=exam_id)
        return claim.number if claim is not None else None

    def has(self, predicate: str) -> bool:
        return bool(self.all(predicate))

    def n_sources(self, claims: Sequence[ClaimView] | None = None) -> int:
        relevant = self._claims if claims is None else tuple(claims)
        return len({str(c.source_id) for c in relevant if c.source_id is not None})

    def modelled_share(self, claims: Sequence[ClaimView] | None = None) -> float:
        relevant = self._claims if claims is None else tuple(claims)
        if not relevant:
            return 0.0
        modelled = sum(1 for c in relevant if c.method == "modelled")
        return modelled / len(relevant)


def claims_hash(claims: Iterable[Any]) -> str:
    """Stable hash of the evidence behind a score.

    The hash covers identity-stable claim fields only (never ``computed_at``),
    so the same evidence always produces the same hash and a changed hash
    always means changed evidence.
    """

    views = [claim_view(raw).canonical() for raw in claims]
    payload = json.dumps(views, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
