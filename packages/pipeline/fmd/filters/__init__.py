"""Hard-filter registry (workstream F owns implementations, brief §12).

Filters are binary cutoffs, never weighted penalties. ``config/weights.yaml``
lists the filters that must exist; registry loading fails if a configured filter
has no implementation name.

Each implementation returns a :class:`FilterDecision` — ``pass | fail |
unknown`` plus the deciding claim id and reason. ``unknown`` is never silently
treated as ``pass``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fmd.filters.impl import (
    FilterContext,
    FilterDecision,
    FilterResult,
    budget_ceiling,
    credential_level,
    default_context,
    eligibility,
    exam_registration_open,
    exam_travel,
    intake_year_available,
    language,
    recognition_status,
)

__all__ = [
    "FilterContext",
    "FilterDecision",
    "FilterResult",
    "FilterSummary",
    "IMPLEMENTATIONS",
    "budget_ceiling",
    "credential_level",
    "default_context",
    "eligibility",
    "evaluate",
    "evaluate_all",
    "exam_registration_open",
    "exam_travel",
    "intake_year_available",
    "language",
    "recognition_status",
]

# filter id in config/weights.yaml -> callable name in this module
IMPLEMENTATIONS: dict[str, str] = {
    "credential_level": "credential_level",
    "recognition_status": "recognition_status",
    "eligibility": "eligibility",
    "language": "language",
    "exam_registration_open": "exam_registration_open",
    "exam_travel": "exam_travel",
    "intake_year_available": "intake_year_available",
    "budget_ceiling": "budget_ceiling",
}


def resolve(filter_id: str) -> Any:
    implementation = IMPLEMENTATIONS.get(filter_id)
    if implementation is None:
        raise KeyError(f"filter {filter_id!r} is not registered in IMPLEMENTATIONS")
    return globals()[implementation]


@dataclass(frozen=True)
class FilterSummary:
    decisions: Mapping[str, FilterDecision]
    order: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        counts = {"pass": 0, "fail": 0, "unknown": 0}
        for decision in self.decisions.values():
            counts[decision.result] += 1
        return counts

    @property
    def excluded(self) -> tuple[str, ...]:
        return tuple(key for key, value in self.decisions.items() if value.result == "fail")

    @property
    def unverified(self) -> tuple[str, ...]:
        return tuple(key for key, value in self.decisions.items() if value.result == "unknown")

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "counts": self.counts,
            "excluded": list(self.excluded),
            "unverified": list(self.unverified),
            "decisions": {key: value.as_dict() for key, value in self.decisions.items()},
        }


def evaluate(
    filter_id: str,
    programme: Any,
    profile: Any,
    *,
    context: FilterContext | None = None,
) -> FilterDecision:
    return resolve(filter_id)(programme, profile, context=context)


def evaluate_all(
    programme: Any,
    profile: Any,
    *,
    context: FilterContext | None = None,
    filter_ids: Sequence[str] | None = None,
) -> FilterSummary:
    """Evaluate filters in the configured order (or a caller-supplied order)."""

    if context is None:
        context = default_context()
    order = tuple(filter_ids) if filter_ids is not None else tuple(IMPLEMENTATIONS)
    decisions = {
        filter_id: evaluate(filter_id, programme, profile, context=context) for filter_id in order
    }
    return FilterSummary(decisions=decisions, order=order)
