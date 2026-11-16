"""Typed, immutable view of the scoring registry.

Scoring and cost code never read YAML directly: they receive this object (or a
plain mapping with the same shape), so a test can supply the committed config
or a synthetic one and prove behaviour is registry-driven.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fmd.score.quantiles import QuantileRules


def _plain(value: Any) -> Any:
    """Best-effort conversion of pydantic config models to plain data."""

    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    return value


@dataclass(frozen=True)
class InsightRules:
    method: str = "greedy_max_coverage"
    max_insights: int = 5
    require_predicate_diversity: bool = True
    always_include: tuple[str, ...] = ("filter_failures", "gaming_flags", "recognition_status")
    min_marginal_gain: float = 0.0


@dataclass(frozen=True)
class ScoringConfig:
    """Everything the scorer needs, and nothing about an applicant."""

    weights: Mapping[str, float]
    weights_version: str
    scorer_version: str
    min_claims_for_score: int
    min_claims_for_confidence: int
    transforms: Mapping[str, Any]
    evidence_tiers: Mapping[str, Mapping[str, Any]]
    gaming_flag_weights: Mapping[str, float]
    salary: QuantileRules
    insights: InsightRules
    peer_group_keys: tuple[str, ...] = ()
    peer_group_min_size: int = 0
    peer_group_fallback: str = "country_prior"
    filter_policy: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, weights: Any) -> ScoringConfig:
        """Build from ``WeightsConfig`` (typed) or its YAML mapping."""

        raw = _plain(weights)
        if not isinstance(raw, Mapping):
            raise TypeError("weights config must be a mapping or pydantic model")

        salary_raw = raw.get("salary") or {}
        insight_raw = raw.get("insights") or {}
        peer = raw.get("peer_groups") or {}
        tiers = raw.get("evidence_tiers") or {}

        return cls(
            weights={str(k): float(v) for k, v in (raw.get("weights") or {}).items()},
            weights_version=str(raw.get("weights_version", "unversioned")),
            scorer_version=str(raw.get("scorer_version", "0.0.0")),
            min_claims_for_score=int(raw.get("min_claims_for_score", 1)),
            min_claims_for_confidence=int(raw.get("min_claims_for_confidence", 1)),
            transforms=dict(raw.get("transforms") or {}),
            evidence_tiers={
                str(name): {
                    "min_sources": int(spec.get("min_sources", 0)),
                    "min_claims": int(spec.get("min_claims", 0)),
                    "max_modelled_share": float(spec.get("max_modelled_share", 1.0)),
                }
                for name, spec in tiers.items()
            },
            gaming_flag_weights={
                str(k): float(v) for k, v in (raw.get("gaming_flags") or {}).items()
            },
            salary=QuantileRules.from_config(salary_raw),
            insights=InsightRules(
                method=str(insight_raw.get("method", "greedy_max_coverage")),
                max_insights=int(insight_raw.get("max_insights", 5)),
                require_predicate_diversity=bool(
                    insight_raw.get("require_predicate_diversity", True)
                ),
                always_include=tuple(
                    str(x)
                    for x in (
                        insight_raw.get("always_include")
                        or ("filter_failures", "gaming_flags", "recognition_status")
                    )
                ),
                min_marginal_gain=float(insight_raw.get("min_marginal_gain", 0.0)),
            ),
            peer_group_keys=tuple(str(k) for k in (peer.get("keys") or ())),
            peer_group_min_size=int(peer.get("min_group_size", 0)),
            peer_group_fallback=str(peer.get("fallback", "country_prior")),
            filter_policy=dict(raw.get("filter_policy") or {}),
            extras={
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "weights",
                    "weights_version",
                    "scorer_version",
                    "min_claims_for_score",
                    "min_claims_for_confidence",
                    "transforms",
                    "evidence_tiers",
                    "gaming_flags",
                    "salary",
                    "insights",
                    "peer_groups",
                    "filter_policy",
                }
            },
        )

    def validate(self) -> list[str]:
        """Return problems instead of silently renormalising anything."""

        problems: list[str] = []
        total = sum(self.weights.values())
        if abs(total - 100.0) > 1e-6:
            problems.append(f"sub-score weights sum to {total:g}, expected 100")
        if any(weight < 0 for weight in self.weights.values()):
            problems.append("sub-score weights must be non-negative")
        if self.salary.min_n_for_empirical_quantile < 2:
            problems.append("salary.min_n_for_empirical_quantile must be >= 2")
        if not 0.5 < self.salary.confidence < 1.0:
            problems.append("salary confidence must be between 0.5 and 1")
        return problems
