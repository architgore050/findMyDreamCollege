"""Typed records shared by the verifier, its runner and its metrics."""

from __future__ import annotations

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
    """

    key: str
    predicate: str
    quote: str
    char_start: int
    char_end: int
    claim_id: int | None = None
    value: Any = None

    @classmethod
    def from_claim(cls, claim: Any, *, key: str | None = None) -> ClaimView:
        get = claim.get if isinstance(claim, dict) else lambda name, default=None: getattr(
            claim, name, default
        )
        claim_id = get("id")
        start = int(get("char_start") or 0)
        end = int(get("char_end") or start)
        predicate = str(get("predicate", ""))
        return cls(
            key=key or (str(claim_id) if claim_id is not None else f"{predicate}:{start}:{end}"),
            predicate=predicate,
            quote=str(get("quote") or ""),
            char_start=start,
            char_end=end,
            claim_id=int(claim_id) if claim_id is not None else None,
            value=get("value"),
        )


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
