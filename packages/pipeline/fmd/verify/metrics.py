"""Verification metrics (brief §19).

``correct_rejected`` is the metric that matters most: a second pass that catches
fabrications but also rejects correct claims is worse than no second pass. It
must stay at 0 on the committed fabrication fixture.

Definitions:

* ``verified_rate`` — claims that received a verdict / claims attempted. A run
  that stops on its budget cap lowers this number instead of inflating
  ``supported``.
* ``fabrication_caught`` — of the claims labelled plausible-but-absent or
  wrong-column, how many were rejected (count and rate).
* ``correct_rejected`` — of the claims whose value is genuinely supported, how
  many were rejected. Must be 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationMetrics:
    total: int = 0
    supported: int = 0
    rejected: int = 0
    unverified: int = 0
    fabricated_total: int = 0
    fabrication_caught: int = 0
    false_accepts: int = 0
    correct_total: int = 0
    correct_rejected: int = 0
    per_claim: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verified_rate(self) -> float | None:
        if self.total == 0:
            return None
        return (self.supported + self.rejected) / self.total

    @property
    def fabrication_caught_rate(self) -> float | None:
        if self.fabricated_total == 0:
            return None
        return self.fabrication_caught / self.fabricated_total

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "supported": self.supported,
            "rejected": self.rejected,
            "unverified": self.unverified,
            "verified_rate": _round(self.verified_rate),
            "fabrication_caught": self.fabrication_caught,
            "fabricated_total": self.fabricated_total,
            "fabrication_caught_rate": _round(self.fabrication_caught_rate),
            "false_accepts": self.false_accepts,
            "correct_total": self.correct_total,
            "correct_rejected": self.correct_rejected,
            "per_claim": self.per_claim,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def verification_metrics(
    decisions: list[Any],
    expected: dict[str, str],
) -> VerificationMetrics:
    """Score decisions against ``expected`` labels per claim key.

    Labels are ``correct``, ``fabricated`` (plausible-but-absent field) or
    ``wrong_field`` (a real fact in the wrong column). Both non-correct labels
    count as fabrications for the catch rate.
    """
    metrics = VerificationMetrics()
    for decision in decisions:
        key = getattr(decision, "key", None) or decision.get("key")
        status = getattr(decision, "status", None) or decision.get("status")
        label = expected.get(str(key), "correct")
        metrics.total += 1
        if status == "supported":
            metrics.supported += 1
        elif status == "rejected":
            metrics.rejected += 1
        else:
            metrics.unverified += 1
        if label == "correct":
            metrics.correct_total += 1
            if status == "rejected":
                metrics.correct_rejected += 1
        else:
            metrics.fabricated_total += 1
            if status == "rejected":
                metrics.fabrication_caught += 1
            elif status == "supported":
                metrics.false_accepts += 1
        metrics.per_claim.append(
            {
                "key": key,
                "expected": label,
                "status": status,
                "verdict": getattr(decision, "verdict", None)
                or (decision.get("verdict") if isinstance(decision, dict) else None),
                "reason": getattr(decision, "reason", None)
                or (decision.get("reason") if isinstance(decision, dict) else None),
            }
        )
    return metrics


__all__ = ["VerificationMetrics", "verification_metrics"]
