"""Seven sub-scores from extracted claims (brief §11).

Each sub-score returns ``{point, lo, hi, n_claims, n_sources, evidence_tier}``
plus the diagnostics the UI needs. Nothing here calls a model, and nothing here
turns missing evidence into a value: a sub-score with too few usable claims is
``UNKNOWN`` with interval ``[0, 100]`` and ``point=None``.

Monotone maps (NAAC grade, recognition status, internship hours, seats) come from
``config/weights.yaml:transforms``. Where the registry has no map for a needed
quantity the component records ``method="modelled_*"`` and the gap is listed in
the final report; no tuned numeric weight is hardcoded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from fmd.score.config import ScoringConfig
from fmd.score.evidence import (
    INTERNSHIP_TO_OFFER,
    REGISTRATION_CONVERSION,
    SALARY_COL_RATIO,
    ClaimIndex,
    ClaimView,
    denominator,
    is_percent_claim,
    is_rate_claim,
    numeric,
)

TIER_ORDER = ("well_evidenced", "partial", "thin", "insufficient")
SUB_SCORE_IDS = (
    "outcomes",
    "clinical_training",
    "academic_quality",
    "affordability_roi",
    "financing_feasibility",
    "student_life_safety",
    "admission_realism",
)
ZERO_TO_HUNDRED = (0.0, 100.0)


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class ComponentScore:
    name: str
    label: str
    value: Any
    score: float
    lo: float
    hi: float
    method: str
    claim_id: int | str | None = None
    source_id: str | int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "value": self.value,
            "score": round(self.score, 2),
            "lo": round(self.lo, 2),
            "hi": round(self.hi, 2),
            "method": self.method,
            "claim_id": self.claim_id,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class SubScore:
    point: float | None
    lo: float
    hi: float
    n_claims: int
    n_sources: int
    evidence_tier: str
    measured: float | None = None
    prior_adjusted: float | None = None
    components: tuple[ComponentScore, ...] = ()
    unknown_reason: str | None = None
    modelled_share: float = 0.0
    widened: bool = False
    shrinkage_inactive: bool = False

    @property
    def known(self) -> bool:
        return self.point is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": None if self.point is None else round(self.point, 2),
            "lo": round(self.lo, 2),
            "hi": round(self.hi, 2),
            "n_claims": self.n_claims,
            "n_sources": self.n_sources,
            "evidence_tier": self.evidence_tier,
            "measured": None if self.measured is None else round(self.measured, 2),
            "prior_adjusted": (
                None if self.prior_adjusted is None else round(self.prior_adjusted, 2)
            ),
            "components": [c.as_dict() for c in self.components],
            "unknown_reason": self.unknown_reason,
            "modelled_share": round(self.modelled_share, 4),
            "widened": self.widened,
            "shrinkage_inactive": self.shrinkage_inactive,
        }


@dataclass(frozen=True)
class ScoreContext:
    config: ScoringConfig
    peer_prior: Mapping[str, float] = field(default_factory=dict)
    flag_penalties: Mapping[str, float] = field(default_factory=dict)
    profile: Any = None
    cost_rules: Any = None

    def penalty_for(self, claim: ClaimView) -> float:
        if claim.claim_id is None:
            return 0.0
        try:
            return max(0.0, float(self.flag_penalties.get(str(claim.claim_id), 0.0)))
        except (TypeError, ValueError):
            return 0.0


def unknown_sub_score(reason: str, *, n_claims: int = 0, n_sources: int = 0) -> SubScore:
    return SubScore(
        point=None,
        lo=ZERO_TO_HUNDRED[0],
        hi=ZERO_TO_HUNDRED[1],
        n_claims=n_claims,
        n_sources=n_sources,
        evidence_tier="insufficient",
        unknown_reason=reason,
        shrinkage_inactive=True,
    )


# --------------------------------------------------------------------------- transforms


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def apply_transform(raw: Any, spec: Mapping[str, Any] | None) -> float | None:
    """Apply a configured monotone transform to a raw value."""

    if spec is None:
        return None
    kind = str(spec.get("type", "")).strip()
    if kind == "ordinal":
        mapping = spec.get("map") or {}
        value = str(raw)
        if value in mapping:
            mapped = mapping[value]
        else:
            normalised = value.strip().upper().replace(" ", "")
            lookup = {str(k).strip().upper().replace(" ", ""): v for k, v in mapping.items()}
            mapped = lookup.get(normalised)
        if mapped is None:
            return None
        return clamp(float(mapped))
    number = numeric(raw)
    if number is None:
        return None
    raw = number
    if kind == "piecewise":
        points = sorted((float(x), float(y)) for x, y in spec.get("points") or [])
        if not points:
            return None
        if raw <= points[0][0]:
            return clamp(points[0][1])
        if raw >= points[-1][0]:
            return clamp(points[-1][1])
        for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
            if x0 <= raw <= x1:
                if x1 == x0:
                    return clamp(y1)
                share = (raw - x0) / (x1 - x0)
                return clamp(y0 + share * (y1 - y0))
        return None
    if kind == "log":
        scale = float(spec.get("scale", 0) or 0)
        if scale <= 0:
            return None
        return clamp(100.0 * math.log1p(max(raw, 0.0)) / math.log1p(scale))
    if kind == "linear":
        low = float(spec.get("min", 0.0))
        high = float(spec.get("max", 100.0))
        if high <= low:
            return None
        return clamp(100.0 * (raw - low) / (high - low))
    return None


def evidence_tier(
    n_claims: int,
    n_sources: int,
    modelled_share: float,
    tiers: Mapping[str, Mapping[str, Any]],
) -> str:
    """Classify evidence using ``weights.yaml:evidence_tiers`` thresholds."""

    for name in TIER_ORDER:
        spec = tiers.get(name)
        if spec is None:
            continue
        if (
            n_sources >= int(spec.get("min_sources", 0))
            and n_claims >= int(spec.get("min_claims", 0))
            and modelled_share <= float(spec.get("max_modelled_share", 1.0))
        ):
            return name
    return "insufficient"


# --------------------------------------------------------------------------- components


@dataclass(frozen=True)
class ComponentSpec:
    """How one predicate becomes a 0..100 component."""

    name: str
    predicate: str
    label: str
    kind: str
    transform: str | None = None
    scale: float = 1.0
    requires_denominator: bool = False
    factor: float = 1.0


def _profile_get(profile: Any, path: str, default: Any = None) -> Any:
    node = profile
    for part in path.split("."):
        if node is None:
            return default
        node = node.get(part) if isinstance(node, Mapping) else getattr(node, part, None)
    return default if node is None else node


def _percent_value(claim: ClaimView, factor: float = 1.0) -> float | None:
    value = claim.number
    if value is None:
        return None
    unit = (claim.unit or "").strip().lower()
    if unit in {"rate", "ratio", "fraction"} or (value <= 1.0 and is_percent_claim(claim)):
        value *= 100.0
    return clamp(value * factor)


def _presence_value(claim: ClaimView) -> float | None:
    value = claim.value
    if isinstance(value, bool):
        return 100.0 if value else 0.0
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"yes", "true", "available", "present", "published"}:
            return 100.0
        if normalised in {"no", "false", "none", "absent", "not available"}:
            return 0.0
        return None
    number = numeric(value)
    if number is None:
        return None
    return 100.0 if number > 0 else 0.0


def _saturating_value(raw: float, scale: float) -> float:
    """Monotone 0..100 curve for unbounded counts: ``100 * (1 - exp(-x/scale))``."""

    if raw <= 0:
        return 0.0
    if scale <= 0:
        return None  # type: ignore[return-value]
    return clamp(100.0 * (1.0 - math.exp(-raw / scale)))


def _decay_value(raw: float, scale: float) -> float | None:
    if raw < 0 or scale <= 0:
        return None
    return clamp(100.0 * math.exp(-raw / scale))


def _margin_value(margin: float, scale: float) -> float | None:
    if scale <= 0:
        return None
    return clamp(100.0 * NormalDist().cdf(margin / scale))


def _component_score(
    spec: ComponentSpec,
    claim: ClaimView,
    context: ScoreContext,
    *,
    profile: Any,
) -> ComponentScore | None:
    if spec.requires_denominator and is_rate_claim(claim) and denominator(claim) is None:
        return None

    raw = claim.number
    if spec.kind == "percent":
        score = _percent_value(claim, spec.factor)
        method = "unit_percent" if is_percent_claim(claim) else "unit_rate"
    elif spec.kind == "presence":
        score = _presence_value(claim)
        method = "presence"
    elif spec.kind == "transform":
        score = apply_transform(claim.value, context.config.transforms.get(spec.transform or ""))
        method = f"config_transform:{spec.transform}"
    elif spec.kind == "saturating":
        score = None if raw is None else _saturating_value(raw, spec.scale)
        method = f"modelled_saturating:scale={spec.scale:g}"
    elif spec.kind == "decay":
        score = None if raw is None else _decay_value(raw, spec.scale)
        method = f"modelled_decay:scale={spec.scale:g}"
    elif spec.kind == "cutoff_margin" or spec.kind == "eligibility_margin":
        if raw is None:
            return None
        marks = numeric(_profile_get(profile, "marks.class12_aggregate_pct"))
        if marks is None:
            return None
        scale = numeric(claim.qualifier.get("sd")) or spec.scale
        score = _margin_value(marks - raw, scale)
        method = f"modelled_margin:sd={scale:g}"
    else:  # pragma: no cover - spec table is closed
        return None

    if score is None:
        return None

    confidence = claim.effective_confidence * (1.0 - min(1.0, context.penalty_for(claim)))
    confidence = min(1.0, max(0.0, confidence))
    lo = clamp(score - (1.0 - confidence) * score)
    hi = clamp(score + (1.0 - confidence) * (100.0 - score))
    return ComponentScore(
        name=spec.name,
        label=spec.label,
        value=claim.value,
        score=score,
        lo=lo,
        hi=hi,
        method=method,
        claim_id=claim.claim_id,
        source_id=claim.source_id,
    )


def _collect(
    specs: Sequence[ComponentSpec],
    index: ClaimIndex,
    context: ScoreContext,
    *,
    profile: Any,
    extra_claims: Sequence[ClaimView] = (),
) -> tuple[list[ComponentScore], list[ClaimView]]:
    components: list[ComponentScore] = []
    used: list[ClaimView] = []
    for spec in specs:
        found = index.latest(spec.predicate)
        if found is None:
            continue
        component = _component_score(spec, found, context, profile=profile)
        if component is None:
            continue
        components.append(component)
        used.append(found)
    for claim in extra_claims:
        used.append(claim)
    return components, used


def _finalise(
    components: Sequence[ComponentScore],
    used: Sequence[ClaimView],
    context: ScoreContext,
    sub_score_id: str,
    *,
    extra_claims: Sequence[ClaimView] = (),
) -> SubScore:
    config = context.config
    claims_pair = list(used) + list(extra_claims)
    n_claims = len({str(c.claim_id) if c.claim_id is not None else id(c) for c in claims_pair})
    n_sources = len({str(c.source_id) for c in claims_pair if c.source_id is not None})
    modelled = sum(1 for c in claims_pair if c.method == "modelled")
    modelled_share = (modelled / n_claims) if n_claims else 0.0
    tier = evidence_tier(n_claims, n_sources, modelled_share, config.evidence_tiers)

    if not components:
        return unknown_sub_score(
            f"no usable evidence for {sub_score_id}", n_claims=n_claims, n_sources=n_sources
        )
    if n_claims < config.min_claims_for_score:
        return SubScore(
            point=None,
            lo=0.0,
            hi=100.0,
            n_claims=n_claims,
            n_sources=n_sources,
            evidence_tier="insufficient",
            measured=None,
            components=tuple(components),
            unknown_reason=(
                f"{n_claims} usable claims < min_claims_for_score={config.min_claims_for_score}"
            ),
            modelled_share=modelled_share,
            shrinkage_inactive=True,
        )

    weights = [1.0 for _ in components]
    total = sum(weights)
    measured = sum(c.score * w for c, w in zip(components, weights, strict=True)) / total
    lo = sum(c.lo * w for c, w in zip(components, weights, strict=True)) / total
    hi = sum(c.hi * w for c, w in zip(components, weights, strict=True)) / total

    prior = context.peer_prior.get(sub_score_id)
    threshold = config.min_claims_for_confidence
    widened = False
    shrinkage_inactive = prior is None
    if prior is not None and n_claims < threshold:
        shrink = n_claims / (n_claims + threshold) if (n_claims + threshold) else 0.0
        adjusted = shrink * measured + (1.0 - shrink) * float(prior)
        widen = math.sqrt(threshold / max(n_claims, 1))
        lo = clamp(adjusted - (adjusted - lo) * widen)
        hi = clamp(adjusted + (hi - adjusted) * widen)
        widened = True
    else:
        # No suppliable prior: never invent one. Report measured unchanged.
        adjusted = measured

    return SubScore(
        point=adjusted,
        lo=lo,
        hi=hi,
        n_claims=n_claims,
        n_sources=n_sources,
        evidence_tier=tier,
        measured=measured,
        prior_adjusted=adjusted,
        components=tuple(components),
        modelled_share=modelled_share,
        widened=widened,
        shrinkage_inactive=shrinkage_inactive,
    )


# --------------------------------------------------------------------------- specs


OUTCOME_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        "internship_to_offer",
        INTERNSHIP_TO_OFFER,
        "Internship-to-offer conversion (12m)",
        "percent",
        requires_denominator=True,
    ),
    ComponentSpec(
        "registration_conversion",
        REGISTRATION_CONVERSION,
        "Registration conversion (12m)",
        "percent",
        requires_denominator=True,
    ),
    ComponentSpec(
        "salary_growth_3y",
        "salary_growth_3y",
        "Salary growth over 3 years",
        "percent",
    ),
    ComponentSpec(
        "salary_vs_col",
        SALARY_COL_RATIO,
        "P25 salary relative to city cost of living",
        "saturating",
        scale=1.0,
    ),
)

CLINICAL_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        "internship_hours",
        "internship_hours",
        "Internship hours",
        "transform",
        transform="internship_hours",
    ),
    ComponentSpec("nabl_lab", "nabl_accredited_lab", "NABL-accredited training lab", "presence"),
    ComponentSpec("nabh_hospital", "nabh_hospital", "NABH-accredited hospital access", "presence"),
    ComponentSpec(
        "departments_rotated",
        "departments_rotated",
        "Departments rotated",
        "saturating",
        scale=6.0,
    ),
    ComponentSpec(
        "equipment_verified",
        "lab_equipment_verified",
        "Lab equipment verified",
        "presence",
    ),
)

ACADEMIC_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec("naac_grade", "naac_grade", "NAAC grade", "transform", transform="naac_grade"),
    ComponentSpec(
        "recognition",
        "recognition_status",
        "Recognition status",
        "transform",
        transform="recognition_status",
    ),
    ComponentSpec("autonomous", "autonomous_status", "Autonomous status", "presence"),
    ComponentSpec("faculty_count", "faculty_count", "Faculty count", "saturating", scale=40.0),
    ComponentSpec(
        "nirf_proxy",
        "nirf_rank_institution",
        "NIRF institution-level proxy",
        "decay",
        scale=100.0,
    ),
)

FINANCING_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec("loan_terms", "loan_terms_published", "Published loan terms", "presence"),
    ComponentSpec("subsidy", "subsidy_eligible", "Subsidy eligibility", "presence"),
    ComponentSpec(
        "collateral_free", "collateral_free_available", "Collateral-free option", "presence"
    ),
    ComponentSpec(
        "student_work_hours",
        "student_work_legal_hours",
        "Legal student-work entitlement",
        "saturating",
        scale=20.0,
    ),
)

STUDENT_LIFE_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec("city_safety", "city_safety_index", "City safety index", "percent"),
    ComponentSpec("hostel", "hostel_available", "Hostel availability", "presence"),
    ComponentSpec("gender_safety", "gender_safety_measures", "Gender-safety measures", "presence"),
    ComponentSpec(
        "support_services", "support_services_count", "Support services", "saturating", scale=3.0
    ),
    ComponentSpec("review_safety", "review_aspect_safety", "Review: safety aspect", "percent"),
)

ADMISSION_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        "cutoff_margin",
        "cutoff_pct",
        "Cutoff vs applicant marks",
        "cutoff_margin",
        scale=10.0,
    ),
    ComponentSpec(
        "eligibility_margin",
        "eligibility_min_aggregate_pct",
        "Eligibility margin",
        "eligibility_margin",
        scale=8.0,
    ),
    ComponentSpec("seats", "seats", "Seats available", "transform", transform="seats"),
    ComponentSpec(
        "applicant_seat_ratio",
        "applicants_per_seat",
        "Applicants per seat",
        "decay",
        scale=10.0,
    ),
)


# --------------------------------------------------------------------------- sub-scores


def outcomes(
    index: ClaimIndex, context: ScoreContext, *, profile: Any = None, cost: Any = None
) -> SubScore:
    components, used = _collect(OUTCOME_SPECS, index, context, profile=profile)
    return _finalise(components, used, context, "outcomes")


def clinical_training(
    index: ClaimIndex, context: ScoreContext, *, profile: Any = None, cost: Any = None
) -> SubScore:
    components, used = _collect(CLINICAL_SPECS, index, context, profile=profile)
    return _finalise(components, used, context, "clinical_training")


def academic_quality(
    index: ClaimIndex, context: ScoreContext, *, profile: Any = None, cost: Any = None
) -> SubScore:
    components, used = _collect(ACADEMIC_SPECS, index, context, profile=profile)
    return _finalise(components, used, context, "academic_quality")


def _band_score(value: float, bands: Mapping[str, Any]) -> float | None:
    """Monotone 0..100 score anchored on configured upper band bounds.

    Score is 100 at 0 and falls linearly to 0 at the last configured bound;
    the intermediate bands are equally spaced. No numeric cut-off is invented.
    """

    bounds = sorted(float(spec["max"]) for spec in bands.values() if spec.get("max") is not None)
    if not bounds:
        return None
    if bounds[0] <= 0:
        return 0.0 if value >= bounds[0] else 100.0
    knots: list[tuple[float, float]] = [(0.0, 100.0)]
    count = len(bounds)
    for position, bound in enumerate(bounds, start=1):
        knots.append((bound, 100.0 * (count - position) / count))
    if value <= 0:
        return 100.0
    if value >= bounds[-1]:
        return 0.0
    for (x0, y0), (x1, y1) in zip(knots, knots[1:], strict=False):
        if x0 <= value <= x1:
            share = (value - x0) / (x1 - x0) if x1 != x0 else 1.0
            return clamp(y0 + share * (y1 - y0))
    return 0.0


def _cost_bands(context: ScoreContext, cost: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """DSR25 bands and affordability threshold from a snapshot or the registry."""

    bands = getattr(cost, "dsr_bands", None) or getattr(cost, "bands", None)
    flag = getattr(cost, "affordability", None)
    if bands:
        return bands, flag or {}
    if context.cost_rules is not None:
        from fmd.cost import CostRules

        rules = CostRules.from_config(context.cost_rules)
        return rules.dsr_bands, {"flag_above": rules.affordability_flag_above}
    return {}, flag or {}


def affordability_roi(
    index: ClaimIndex,
    context: ScoreContext,
    *,
    profile: Any = None,
    cost: Any = None,
) -> SubScore:
    """DSR25 and affordability gap, scored against configured bands."""

    components: list[ComponentScore] = []
    used: list[ClaimView] = []

    dsr_claim = index.latest("dsr25")
    gap_claim = index.latest("affordability_gap")
    dsr_bands, gap_spec = _cost_bands(context, cost)

    dsr_value = None
    if cost is not None and getattr(cost, "dsr25", None) is not None:
        dsr_value = getattr(cost.dsr25, "value", None)
    if dsr_value is None and dsr_claim is not None:
        dsr_value = dsr_claim.number
    if dsr_value is not None and dsr_bands:
        band_score = _band_score(float(dsr_value), dsr_bands)
        if band_score is None:
            raise ValueError("DSR25 bands are not configured; cannot score affordability")
        score = band_score
        confidence = dsr_claim.effective_confidence if dsr_claim is not None else 0.75
        components.append(
            ComponentScore(
                name="dsr25",
                label="DSR25 (first-year EMI / P25 salary)",
                value=dsr_value,
                score=score,
                lo=clamp(score - (1 - confidence) * score),
                hi=clamp(score + (1 - confidence) * (100 - score)),
                method="cost_band:dsr25",
                claim_id=dsr_claim.claim_id if dsr_claim is not None else None,
                source_id=dsr_claim.source_id if dsr_claim is not None else None,
            )
        )
        if dsr_claim is not None:
            used.append(dsr_claim)

    gap_value = None
    if cost is not None and getattr(cost, "affordability_gap", None) is not None:
        gap_value = getattr(cost.affordability_gap, "value", None)
    if gap_value is None and gap_claim is not None:
        gap_value = gap_claim.number
    flag_above = float((gap_spec or {}).get("flag_above", 1.0) or 1.0)
    if gap_value is not None:
        gap_score = _band_score(float(gap_value), {"flag": {"max": flag_above}})
        if gap_score is None:
            raise ValueError("affordability_gap.flag_above is not configured")
        score = gap_score
        confidence = gap_claim.effective_confidence if gap_claim is not None else 0.75
        components.append(
            ComponentScore(
                name="affordability_gap",
                label="Family out-of-pocket / annual income",
                value=gap_value,
                score=score,
                lo=clamp(score - (1 - confidence) * score),
                hi=clamp(score + (1 - confidence) * (100 - score)),
                method=f"config_band:affordability_gap:flag_above={flag_above:g}",
                claim_id=gap_claim.claim_id if gap_claim is not None else None,
                source_id=gap_claim.source_id if gap_claim is not None else None,
            )
        )
        if gap_claim is not None:
            used.append(gap_claim)

    return _finalise(components, used, context, "affordability_roi")


def financing_feasibility(
    index: ClaimIndex, context: ScoreContext, *, profile: Any = None, cost: Any = None
) -> SubScore:
    components, used = _collect(FINANCING_SPECS, index, context, profile=profile)
    return _finalise(components, used, context, "financing_feasibility")


def student_life_safety(
    index: ClaimIndex, context: ScoreContext, *, profile: Any = None, cost: Any = None
) -> SubScore:
    components, used = _collect(STUDENT_LIFE_SPECS, index, context, profile=profile)
    return _finalise(components, used, context, "student_life_safety")


def admission_realism(
    index: ClaimIndex, context: ScoreContext, *, profile: Any = None, cost: Any = None
) -> SubScore:
    components, used = _collect(ADMISSION_SPECS, index, context, profile=profile)
    return _finalise(components, used, context, "admission_realism")


IMPLEMENTATIONS: Mapping[str, Any] = {
    "outcomes": outcomes,
    "clinical_training": clinical_training,
    "academic_quality": academic_quality,
    "affordability_roi": affordability_roi,
    "financing_feasibility": financing_feasibility,
    "student_life_safety": student_life_safety,
    "admission_realism": admission_realism,
}


def compute_sub_scores(
    index: ClaimIndex,
    context: ScoreContext,
    *,
    profile: Any = None,
    cost: Any = None,
) -> dict[str, SubScore]:
    """Run every configured sub-score implementation."""

    results: dict[str, SubScore] = {}
    for sub_score_id, implementation in IMPLEMENTATIONS.items():
        results[sub_score_id] = implementation(index, context, profile=profile, cost=cost)
    return results
