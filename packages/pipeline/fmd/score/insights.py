"""Insight selection: greedy max-coverage with mandatory inclusions (brief §11).

Insights are chosen to cover *predicates/aspects*, not to please. Hard-filter
failures, gaming flags and recognition status are always included, separately
from the greedy budget, and the coverage report is returned so the selection is
auditable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Insight:
    id: str
    predicate: str
    text: str
    coverage: frozenset[str] = frozenset()
    claim_ids: tuple[int | str, ...] = ()
    kind: str = "evidence"
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "predicate": self.predicate,
            "text": self.text,
            "coverage": sorted(self.coverage),
            "claim_ids": list(self.claim_ids),
            "kind": self.kind,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class InsightSelection:
    selected: tuple[Insight, ...]
    mandatory: tuple[Insight, ...]
    coverage: float
    covered_units: tuple[str, ...]
    uncovered_units: tuple[str, ...]
    stopped_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": [i.as_dict() for i in self.selected],
            "mandatory": [i.as_dict() for i in self.mandatory],
            "coverage": round(self.coverage, 4),
            "covered_units": list(self.covered_units),
            "uncovered_units": list(self.uncovered_units),
            "stopped_reason": self.stopped_reason,
        }


def from_filter_failures(decisions: Iterable[Any]) -> tuple[Insight, ...]:
    insights: list[Insight] = []
    for decision in decisions:
        status = getattr(decision, "result", None) or (
            decision.get("result") if isinstance(decision, Mapping) else None
        )
        if status == "pass":
            continue
        filter_id = getattr(decision, "filter_id", None) or (
            decision.get("filter_id") if isinstance(decision, Mapping) else None
        )
        reason = getattr(decision, "reason", None) or (
            decision.get("reason") if isinstance(decision, Mapping) else None
        )
        claim_id = getattr(decision, "claim_id", None) or (
            decision.get("claim_id") if isinstance(decision, Mapping) else None
        )
        if filter_id is None:
            continue
        insights.append(
            Insight(
                id=f"filter:{filter_id}",
                predicate=filter_id,
                text=f"{filter_id}: {status} — {reason or 'no deciding claim'}",
                coverage=frozenset({f"filter:{filter_id}"}),
                claim_ids=(claim_id,) if claim_id is not None else (),
                kind="filter_failure" if status == "fail" else "filter_unknown",
                detail={"status": status},
            )
        )
    return tuple(insights)


def from_gaming_flags(flags: Iterable[Any]) -> tuple[Insight, ...]:
    insights: list[Insight] = []
    for flag in flags:
        flag_id = getattr(flag, "flag_id", None) or (
            flag.get("flag_id") if isinstance(flag, Mapping) else None
        )
        if flag_id is None:
            continue
        claim_id = getattr(flag, "claim_id", None) or (
            flag.get("claim_id") if isinstance(flag, Mapping) else None
        )
        evidence = getattr(flag, "evidence", None) or (
            flag.get("evidence") if isinstance(flag, Mapping) else None
        )
        predicate = getattr(flag, "predicate", None) or (
            flag.get("predicate") if isinstance(flag, Mapping) else None
        )
        insights.append(
            Insight(
                id=f"flag:{flag_id}:{claim_id}",
                predicate=str(predicate or flag_id),
                text=f"{flag_id}: {evidence or ''}".strip(),
                coverage=frozenset({f"flag:{flag_id}"}),
                claim_ids=(claim_id,) if claim_id is not None else (),
                kind="gaming_flag",
            )
        )
    return tuple(insights)


def recognition_insight(claim: Any | None, status: str | None = None) -> Insight:
    claim_id = getattr(claim, "claim_id", None) if claim is not None else None
    value = getattr(claim, "value", None) if claim is not None else None
    resolved = status or value or "unknown"
    return Insight(
        id="recognition_status",
        predicate="recognition_status",
        text=f"Recognition status: {resolved}",
        coverage=frozenset({"recognition_status"}),
        claim_ids=(claim_id,) if claim_id is not None else (),
        kind="recognition_status",
        detail={"status": resolved},
    )


def select(
    insights: Sequence[Insight],
    *,
    mandatory: Sequence[Insight] = (),
    max_insights: int | None = None,
    min_marginal_gain: float = 0.0,
    require_predicate_diversity: bool = True,
) -> InsightSelection:
    """Greedy max-coverage over information units.

    The greedy budget applies to optional insights; mandatory insights are
    always included even when they exceed ``max_insights``.
    """

    if max_insights is not None and max_insights < 0:
        raise ValueError("max_insights must be >= 0")

    seen_ids = {item.id for item in mandatory}
    mandatory_tuple = tuple(item for item in mandatory)

    universe: set[str] = set()
    for item in (*mandatory_tuple, *insights):
        universe.update(item.coverage)
    if not universe:
        return InsightSelection(
            selected=(),
            mandatory=mandatory_tuple,
            coverage=1.0,
            covered_units=(),
            uncovered_units=(),
            stopped_reason="no_information_units",
        )

    covered: set[str] = set()
    for item in mandatory_tuple:
        covered.update(item.coverage)

    candidates = [item for item in insights if item.id not in seen_ids]
    selected: list[Insight] = []
    covered_predicates = {item.predicate for item in mandatory_tuple}
    budget = max_insights if max_insights is not None else len(candidates)
    stopped_reason = "max_insights_reached" if max_insights is not None else "no_candidates_left"

    while len(selected) < budget and candidates:
        best: Insight | None = None
        best_gain = -1.0
        for candidate in candidates:
            new_units = candidate.coverage - covered
            gain = len(new_units) / len(universe)
            is_diverse = candidate.predicate not in covered_predicates
            if not is_diverse and require_predicate_diversity:
                continue
            if gain > best_gain + 1e-12 or (
                abs(gain - best_gain) <= 1e-12 and (best is None or candidate.id < best.id)
            ):
                best, best_gain = candidate, gain

        if best is None:
            stopped_reason = "predicate_diversity_exhausted"
            break
        if best_gain < min_marginal_gain or best_gain <= 0.0:
            stopped_reason = "min_marginal_gain_reached"
            break

        selected.append(best)
        covered.update(best.coverage)
        covered_predicates.add(best.predicate)
        candidates = [item for item in candidates if item.id != best.id]
        if len(selected) >= budget:
            stopped_reason = "max_insights_reached"

    if not candidates and len(selected) < budget:
        stopped_reason = "no_candidates_left"

    return InsightSelection(
        selected=tuple(selected),
        mandatory=mandatory_tuple,
        coverage=len(covered) / len(universe),
        covered_units=tuple(sorted(covered)),
        uncovered_units=tuple(sorted(universe - covered)),
        stopped_reason=stopped_reason,
    )
