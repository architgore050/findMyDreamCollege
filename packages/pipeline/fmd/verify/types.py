"""Typed records shared by the verifier, its runner and its metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Verdict = Literal["supported", "contradicted", "absent", "wrong_field"]
FinalStatus = Literal["unverified", "supported", "rejected"]
VERDICTS: tuple[str, ...] = ("supported", "contradicted", "absent", "wrong_field")


@dataclass(frozen=True)
class ClaimView:
    """The verifier-visible projection of a claim.

    ``value`` is deliberately absent from the dataclass used to build prompts:
    the second pass must not anchor on the number the extractor produced.
    ``value`` is carried only for metrics/tests, never sent to the model.

    Row context (verify@v2): when the extractor resolved the quote against a
    table row, the claim's ``qualifier`` carries the row identifier the
    extractor used (``context_quote``) plus the resolved row identity
    (``row.row_label`` / ``row.column_header``) and, when available, the source
    spans of the context anchor and of the row. The context travels to the
    model so it can judge the *association* (does this value belong to this
    row/column?), and the spans widen the source window to cover that row.
    ``expected`` is the value type the predicate registry declares for the
    predicate (``count``, ``money+period``, ...) and ``description`` is the
    registry's one-sentence definition of the predicate, scope included
    (verify@v3); the verifier fills both in from ``config/predicates.yaml`` —
    the claim never supplies either, and an unknown predicate stays ``None``.
    """

    key: str
    predicate: str
    quote: str
    char_start: int
    char_end: int
    claim_id: int | None = None
    value: Any = None
    context: str | None = None
    row_label: str | None = None
    column_header: str | None = None
    context_span: tuple[int, int] | None = None
    row_span: tuple[int, int] | None = None
    expected: str | None = None
    description: str | None = None

    @classmethod
    def from_claim(cls, claim: Any, *, key: str | None = None) -> ClaimView:
        get = claim.get if isinstance(claim, dict) else lambda name, default=None: getattr(
            claim, name, default
        )
        claim_id = get("id")
        start = int(get("char_start") or 0)
        end = int(get("char_end") or start)
        predicate = str(get("predicate", ""))
        qualifier = get("qualifier") or {}
        if not isinstance(qualifier, Mapping):  # pragma: no cover - malformed claim
            qualifier = {}
        row = qualifier.get("row") or {}
        if not isinstance(row, Mapping):  # pragma: no cover - malformed claim
            row = {}
        context_span = _span_of(qualifier, "context_char_start", "context_char_end")
        row_span = _span_of(qualifier, "row_char_start", "row_char_end")
        return cls(
            key=key or (str(claim_id) if claim_id is not None else f"{predicate}:{start}:{end}"),
            predicate=predicate,
            quote=str(get("quote") or ""),
            char_start=start,
            char_end=end,
            claim_id=int(claim_id) if claim_id is not None else None,
            value=get("value"),
            context=_text_or_none(qualifier.get("context_quote")),
            row_label=_text_or_none(row.get("row_label")),
            column_header=_text_or_none(row.get("column_header")),
            context_span=context_span,
            row_span=row_span,
        )


def _text_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _span_of(source: Mapping[str, Any], start_key: str, end_key: str) -> tuple[int, int] | None:
    start, end = source.get(start_key), source.get(end_key)
    if start is None or end is None:
        return None
    try:
        return int(start), int(end)
    except (TypeError, ValueError):  # pragma: no cover - malformed qualifier
        return None


@dataclass(frozen=True)
class VerificationDecision:
    key: str
    claim_id: int | None
    status: FinalStatus
    verdict: Verdict | None = None
    reason: str | None = None
    repair_attempted: bool = False
    corrected_quote: str | None = None
    prompt_version: str = ""
    verified_by: str | None = None
    expected_type: str | None = None
    row_context: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "claim_id": self.claim_id,
            "status": self.status,
            "verdict": self.verdict,
            "reason": self.reason,
            "repair_attempted": self.repair_attempted,
            "corrected_quote": self.corrected_quote,
            "prompt_version": self.prompt_version,
            "verified_by": self.verified_by,
            "expected_type": self.expected_type,
            "row_context": self.row_context,
        }


@dataclass
class VerificationRun:
    decisions: list[VerificationDecision] = field(default_factory=list)
    calls: int = 0
    repairs: int = 0
    estimated_cost_usd: float = 0.0
    budget_exhausted: bool = False
    error: str | None = None

    def counts(self) -> dict[str, int]:
        counts = {"attempted": len(self.decisions), "supported": 0, "rejected": 0, "unverified": 0}
        for decision in self.decisions:
            counts[decision.status] += 1
        counts["calls"] = self.calls
        counts["repairs"] = self.repairs
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "calls": self.calls,
            "repairs": self.repairs,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "budget_exhausted": self.budget_exhausted,
            "error": self.error,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }
