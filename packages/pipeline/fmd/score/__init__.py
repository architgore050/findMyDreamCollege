"""Scoring entrypoint: claims -> sub-scores -> composite -> insights (brief §11).

Numbers are computed here; the model router never emits a score. A score is
reproducible from ``(claims_hash, scorer_version, weights_version)``: identical
evidence and versions produce an identical payload apart from ``computed_at``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fmd.score.aggregate import Composite, compare, composite, dominates, pareto_matrix
from fmd.score.config import InsightRules, ScoringConfig
from fmd.score.evidence import ClaimIndex, claims_hash
from fmd.score.flags import GamingFlag, confidence_penalties, detect
from fmd.score.insights import (
    Insight,
    InsightSelection,
    from_filter_failures,
    from_gaming_flags,
    recognition_insight,
    select,
)
from fmd.score.quantiles import (
    QuantileEstimate,
    QuantileRules,
    estimate,
    from_mean_median,
    from_mean_only,
    from_median_iqr,
    from_median_range,
)
from fmd.score.sub_scores import (
    ComponentScore,
    ScoreContext,
    SubScore,
    compute_sub_scores,
)

__all__ = [
    "ComponentScore",
    "Composite",
    "GamingFlag",
    "Insight",
    "InsightRules",
    "InsightSelection",
    "QuantileEstimate",
    "QuantileRules",
    "ScoreContext",
    "ScoreResult",
    "ScoringConfig",
    "SubScore",
    "compare",
    "composite",
    "dominance_matrix",
    "dominates",
    "estimate",
    "from_mean_median",
    "from_mean_only",
    "from_median_iqr",
    "from_median_range",
    "score_programme",
]


@dataclass(frozen=True)
class ScoreResult:
    entity_id: int | None
    scorer_version: str
    weights_version: str
    claims_hash: str
    sub_scores: Mapping[str, SubScore]
    composite: Composite
    evidence_tier: str
    computed_at: datetime
    insights: InsightSelection | None = None
    gaming_flags: tuple[GamingFlag, ...] = ()
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    unknown_predicates: tuple[str, ...] = ()
    alias_predicates: tuple[str, ...] = ()
    predicate_registry_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "scorer_version": self.scorer_version,
            "weights_version": self.weights_version,
            "claims_hash": self.claims_hash,
            "sub_scores": {key: value.as_dict() for key, value in self.sub_scores.items()},
            "composite": self.composite.as_dict(),
            "evidence_tier": self.evidence_tier,
            "computed_at": self.computed_at.isoformat(),
            "insights": None if self.insights is None else self.insights.as_dict(),
            "gaming_flags": [flag.as_dict() for flag in self.gaming_flags],
            "assumptions": list(self.assumptions),
            "unknown_predicates": list(self.unknown_predicates),
            "alias_predicates": list(self.alias_predicates),
            "predicate_registry_version": self.predicate_registry_version,
        }

    def to_score_create(self) -> dict[str, Any]:
        """Payload compatible with :class:`fmd.db.repositories.ScoreCreate`."""

        return {
            "entity_id": self.entity_id,
            "scorer_version": self.scorer_version,
            "weights_version": self.weights_version,
            "claims_hash": self.claims_hash,
            "sub_scores": {key: value.as_dict() for key, value in self.sub_scores.items()},
            "evidence_tier": self.evidence_tier,
            "composite": self.composite.point,
            "composite_lo": self.composite.lo,
            "composite_hi": self.composite.hi,
            "computed_at": self.computed_at,
        }


def score_programme(
    entity_id: int | None,
    programme: Any,
    profile: Any,
    claims: Sequence[Any],
    *,
    config: ScoringConfig | None = None,
    cost: Any = None,
    cost_rules: Any = None,
    peer_prior: Mapping[str, float] | None = None,
    filter_decisions: Sequence[Any] = (),
    insight_candidates: Sequence[Insight] = (),
    now: datetime | None = None,
) -> ScoreResult:
    """Score one programme from its claims.

    ``cost`` is an optional :class:`fmd.cost.CostSnapshot`; when supplied it
    feeds ``affordability_roi`` without recomputation.
    """

    if config is None:
        from fmd.config import get_registry

        registry = get_registry()
        scoring = ScoringConfig.from_config(registry.weights, registry.predicates)
        if cost_rules is None:
            cost_rules = registry.cost_rules
    else:
        scoring = config
    index = ClaimIndex.build(claims)

    problems = scoring.validate()
    if problems:
        raise ValueError("invalid scoring config: " + "; ".join(problems))

    predicate_registry = scoring.predicates
    assert predicate_registry is not None  # validate() guarantees it
    unknown_predicates = predicate_registry.unknown_predicates(index)
    alias_predicates = predicate_registry.alias_predicates(index)

    flags = detect(index, scoring.confidence_penalty_weights)
    penalties = confidence_penalties(flags)

    context = ScoreContext(
        config=scoring,
        peer_prior=dict(peer_prior or {}),
        flag_penalties=penalties,
        profile=profile,
        cost_rules=cost_rules,
    )
    sub_scores = compute_sub_scores(index, context, profile=profile, cost=cost)
    aggregate = composite(sub_scores, scoring)

    mandatory: list[Insight] = []
    mandatory.extend(from_filter_failures(filter_decisions))
    mandatory.extend(from_gaming_flags(flags))
    recognition_claim = index.latest("accreditation.recognition_status") or index.latest(
        "recognition_status"
    )
    status = (
        getattr(programme, "recognition_status", None)
        if not isinstance(programme, Mapping)
        else programme.get("recognition_status")
    )
    mandatory.append(recognition_insight(recognition_claim, status))

    selection = select(
        tuple(insight_candidates),
        mandatory=tuple(mandatory),
        max_insights=scoring.insights.max_insights,
        min_marginal_gain=scoring.insights.min_marginal_gain,
        require_predicate_diversity=scoring.insights.require_predicate_diversity,
    )

    tier = aggregate.evidence_tier
    # Sanity: a programme with zero claims cannot be better than insufficient.
    if not index:
        tier = "insufficient"

    return ScoreResult(
        entity_id=entity_id,
        scorer_version=scoring.scorer_version,
        weights_version=scoring.weights_version,
        claims_hash=claims_hash(index),
        sub_scores=sub_scores,
        composite=aggregate,
        evidence_tier=tier,
        computed_at=now or datetime.now(UTC),
        insights=selection,
        gaming_flags=flags,
        assumptions=(
            "sub-score component weights are equal within a sub-score; "
            "sub-score weights come from config/weights.yaml",
            "predicate -> component mapping comes from config/predicates.yaml; "
            "unknown predicates are reported, never guessed",
            "a subset composite is partial: its interval carries the full [0,100] "
            "mass of every unknown component",
        ),
        unknown_predicates=unknown_predicates,
        alias_predicates=alias_predicates,
        predicate_registry_version=predicate_registry.version,
    )


def dominance_matrix(
    results: Mapping[str, ScoreResult],
) -> dict[str, dict[str, Any]]:
    """Pareto matrix over sub-score points, with explicit non-comparability."""

    rows = {
        key: {name: score.as_dict() for name, score in result.sub_scores.items()}
        for key, result in results.items()
    }
    return pareto_matrix(rows)
