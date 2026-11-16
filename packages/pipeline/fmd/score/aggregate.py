"""Weighted composite, interval propagation and Pareto dominance (brief §11).

The composite is only shown as a point when every configured sub-score is
known. A single ``UNKNOWN`` sub-score widens the composite interval to include
its ``[0, 100]`` range and withholds the point, because the only alternatives
would be 0 or 50 — both forbidden.

Two interval bounds are computed and both are reported:

* ``independent``  — sub-score errors are treated as independent; the weighted
  standard error is combined in quadrature across 80% intervals.
* ``comonotonic``  — worst case, every sub-score lands at the same end of its
  interval; this is plain interval arithmetic and never narrower.

Dominance is strict Pareto over *known* sub-scores. A pair with any dimension
that is unknown on either side is ``not_comparable`` rather than guessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

from fmd.score.config import ScoringConfig
from fmd.score.evidence import ClaimView
from fmd.score.sub_scores import TIER_ORDER, SubScore, evidence_tier

Dominance = Literal["better", "worse", "not_comparable"]
Z80 = NormalDist().inv_cdf(0.9)


@dataclass(frozen=True)
class Composite:
    point: float | None
    lo: float
    hi: float
    lo_independent: float
    hi_independent: float
    lo_comonotonic: float
    hi_comonotonic: float
    bound: str
    evidence_tier: str
    known_weight: float
    total_weight: float
    unknown_sub_scores: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": None if self.point is None else round(self.point, 2),
            "lo": round(self.lo, 2),
            "hi": round(self.hi, 2),
            "bound": self.bound,
            "lo_independent": round(self.lo_independent, 2),
            "hi_independent": round(self.hi_independent, 2),
            "lo_comonotonic": round(self.lo_comonotonic, 2),
            "hi_comonotonic": round(self.hi_comonotonic, 2),
            "evidence_tier": self.evidence_tier,
            "known_weight": round(self.known_weight, 4),
            "total_weight": round(self.total_weight, 4),
            "unknown_sub_scores": list(self.unknown_sub_scores),
        }


def _as_sub_score(value: Any) -> SubScore:
    if isinstance(value, SubScore):
        return value
    return SubScore(
        point=value.get("point"),
        lo=float(value.get("lo", 0.0)),
        hi=float(value.get("hi", 100.0)),
        n_claims=int(value.get("n_claims", 0)),
        n_sources=int(value.get("n_sources", 0)),
        evidence_tier=str(value.get("evidence_tier", "insufficient")),
        measured=value.get("measured"),
        prior_adjusted=value.get("prior_adjusted"),
    )


def validate_weights(weights: Mapping[str, float]) -> None:
    total = sum(float(v) for v in weights.values())
    if abs(total - 100.0) > 1e-6:
        raise ValueError(
            f"sub-score weights sum to {total:g}, expected 100; "
            "fix config/weights.yaml instead of renormalising silently"
        )
    if any(float(v) < 0 for v in weights.values()):
        raise ValueError("sub-score weights must be non-negative")


def composite(
    sub_scores: Mapping[str, Any],
    weights: Mapping[str, float] | ScoringConfig,
    *,
    tiers: Mapping[str, Mapping[str, Any]] | None = None,
) -> Composite:
    """Weighted composite with independent and comonotonic interval bounds."""

    if isinstance(weights, ScoringConfig):
        config = weights
        weight_map = config.weights
        tier_map = config.evidence_tiers
    else:
        weight_map = weights
        tier_map = tiers or {}

    validate_weights(weight_map)

    known_weight = 0.0
    total_weight = 0.0
    unknown: list[str] = []
    point_sum = 0.0
    lo_comonotonic_sum = 0.0
    hi_comonotonic_sum = 0.0
    variance = 0.0

    for sub_score_id, weight_raw in weight_map.items():
        weight = float(weight_raw)
        total_weight += weight
        score = _as_sub_score(sub_scores[sub_score_id]) if sub_score_id in sub_scores else None
        if score is None or score.point is None:
            unknown.append(sub_score_id)
            # UNKNOWN contributes its full [0, 100] range, never a value.
            lo_comonotonic_sum += weight * 0.0
            hi_comonotonic_sum += weight * 100.0
            variance += (weight * (100.0 / (2.0 * Z80))) ** 2
            continue
        known_weight += weight
        point_sum += weight * score.point
        lo_comonotonic_sum += weight * score.lo
        hi_comonotonic_sum += weight * score.hi
        half_width = (score.hi - score.lo) / 2.0
        sigma = half_width / Z80
        variance += (weight * sigma) ** 2

    if total_weight <= 0:
        raise ValueError("no weights configured")

    lo_comonotonic = lo_comonotonic_sum / total_weight
    hi_comonotonic = hi_comonotonic_sum / total_weight
    sigma_composite = math.sqrt(variance) / total_weight
    point = point_sum / total_weight if not unknown else None

    if point is None:
        lo_independent = max(0.0, _weighted_lower(sub_scores, weight_map, total_weight))
        hi_independent = min(100.0, _weighted_upper(sub_scores, weight_map, total_weight))
    else:
        lo_independent = max(0.0, point - Z80 * sigma_composite)
        hi_independent = min(100.0, point + Z80 * sigma_composite)

    known_scores = [_as_sub_score(sub_scores[key]) for key in weight_map if key in sub_scores]
    tier = _composite_tier(known_scores, unknown, tier_map)

    return Composite(
        point=point,
        lo=lo_independent if point is not None else lo_comonotonic,
        hi=hi_independent if point is not None else hi_comonotonic,
        lo_independent=lo_independent,
        hi_independent=hi_independent,
        lo_comonotonic=lo_comonotonic,
        hi_comonotonic=hi_comonotonic,
        bound="independent" if point is not None else "comonotonic",
        evidence_tier=tier,
        known_weight=known_weight,
        total_weight=total_weight,
        unknown_sub_scores=tuple(unknown),
    )


def _weighted_lower(
    sub_scores: Mapping[str, Any], weights: Mapping[str, float], total: float
) -> float:
    return (
        sum(
            float(weights[k]) * (_as_sub_score(sub_scores[k]).lo if k in sub_scores else 0.0)
            for k in weights
        )
        / total
    )


def _weighted_upper(
    sub_scores: Mapping[str, Any], weights: Mapping[str, float], total: float
) -> float:
    return (
        sum(
            float(weights[k]) * (_as_sub_score(sub_scores[k]).hi if k in sub_scores else 100.0)
            for k in weights
        )
        / total
    )


def _composite_tier(
    known_scores: Sequence[SubScore],
    unknown: Sequence[str],
    tiers: Mapping[str, Mapping[str, Any]],
) -> str:
    if not known_scores:
        return "insufficient"

    def rank(tier: str) -> int:
        return TIER_ORDER.index(tier) if tier in TIER_ORDER else len(TIER_ORDER)

    worst = max(known_scores, key=lambda score: rank(score.evidence_tier)).evidence_tier
    if unknown and rank(worst) < rank("thin"):
        # A missing sub-score cannot leave the record "well evidenced".
        worst = "thin"
    if tiers and known_scores:
        total_claims = sum(score.n_claims for score in known_scores)
        total_sources = sum(score.n_sources for score in known_scores)
        share = sum(score.modelled_share * score.n_claims for score in known_scores) / max(
            total_claims, 1
        )
        measured = evidence_tier(total_claims, total_sources, share, tiers)
        if rank(measured) > rank(worst):
            worst = measured
    return worst


def _points(sub_scores: Mapping[str, Any], keys: Sequence[str]) -> dict[str, float | None]:
    points: dict[str, float | None] = {}
    for key in keys:
        raw = sub_scores.get(key)
        if raw is None:
            points[key] = None
            continue
        points[key] = _as_sub_score(raw).point
    return points


def compare(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> Dominance:
    """Pareto comparison over known sub-scores only.

    ``a`` is ``better`` only if it is >= ``b`` on every *known* dimension and
    strictly greater on at least one. If ``a`` is worse on any known dimension
    it is ``worse``. Everything else — including ties and dimensions where one
    side is unknown — is ``not_comparable``.
    """

    dimension_keys = tuple(keys) if keys is not None else tuple(a.keys() | b.keys())
    a_points = _points(a, dimension_keys)
    b_points = _points(b, dimension_keys)

    better_somewhere = False
    worse_somewhere = False
    unknown_somewhere = False
    for key in dimension_keys:
        av, bv = a_points[key], b_points[key]
        if av is None or bv is None:
            unknown_somewhere = True
            continue
        if av < bv:
            worse_somewhere = True
        elif av > bv:
            better_somewhere = True
    if worse_somewhere and not better_somewhere:
        return "worse"
    if better_somewhere and not worse_somewhere and not unknown_somewhere:
        return "better"
    return "not_comparable"


def dominates(a: Mapping[str, Any], b: Mapping[str, Any], **kwargs: Any) -> bool:
    return compare(a, b, **kwargs) == "better"


def pareto_matrix(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Undominated set plus explicit comparability for every pair."""

    ids = sorted(rows)
    matrix: dict[str, dict[str, Any]] = {}
    for left in ids:
        better_than: list[str] = []
        worse_than: list[str] = []
        not_comparable: list[str] = []
        for right in ids:
            if left == right:
                continue
            verdict = compare(rows[left], rows[right], keys=keys)
            if verdict == "better":
                better_than.append(right)
            elif verdict == "worse":
                worse_than.append(right)
            else:
                not_comparable.append(right)
        matrix[left] = {
            "undominated": not worse_than,
            "better_than": better_than,
            "worse_than": worse_than,
            "not_comparable": not_comparable,
        }
    return matrix


def claims_for_hash(claims: Sequence[ClaimView]) -> tuple[ClaimView, ...]:
    return tuple(claims)
