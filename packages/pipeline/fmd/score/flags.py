"""Gaming / red-flag detectors (brief §11 "gaming resistance").

Flags attach to the *claim*, never to the university, and reduce that claim's
confidence. They are displayed verbatim. Penalties come from
``config/weights.yaml:confidence_penalties``; a detector with no configured
weight is still reported via :data:`MISSING_DETECTORS` rather than silently
ignored.

Each configured value is a per-flag multiplier penalty (``0 <= w < 1``), never a
share. Penalties on one claim compose multiplicatively::

    remaining = max(MIN_CONFIDENCE, prod(1 - w_i))
    penalty   = 1 - remaining

Every detector is pure: claims in, flags out.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from fmd.score.evidence import (
    ClaimIndex,
    ClaimView,
    denominator,
    is_rate_claim,
    numeric,
)

# Composition floor: repeated flags may never drive claim confidence below this.
MIN_CONFIDENCE = 0.05

# Detectors implemented in code; validated against the config keys.
DETECTORS: tuple[str, ...] = (
    "missing_denominator",
    "no_distribution",
    "headline_is_highest_package",
    "round_numbers",
    "yoy_jump_gt_2sigma",
    "aggregated_across_degrees",
    "recruiters_without_counts",
    "offers_not_joinings",
    "internship_counted_as_placement",
    "no_cohort_year",
)

OUTCOME_PREDICATES = (
    "placement_rate",
    "employment_rate",
    "internship_to_offer_conversion_12m",
    "registration_conversion_12m",
    "median_salary",
    "p25_first_year_salary",
    "mean_salary",
    "highest_package",
    "salary_growth_3y",
)

INTERNSHIP_MARKERS = re.compile(r"\bintern(ship)?s?\b", re.IGNORECASE)
OFFER_MARKERS = re.compile(r"\boffers?\b", re.IGNORECASE)
JOINING_MARKERS = re.compile(r"\bjoin(ing|ed|ings)?\b", re.IGNORECASE)


@dataclass(frozen=True)
class GamingFlag:
    flag_id: str
    weight: float
    claim_id: int | str | None
    predicate: str
    evidence: str
    detail: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "flag_id": self.flag_id,
            "weight": self.weight,
            "claim_id": self.claim_id,
            "predicate": self.predicate,
            "evidence": self.evidence,
            "detail": dict(self.detail),
        }


def _is_outcome_claim(claim: ClaimView) -> bool:
    return claim.predicate in OUTCOME_PREDICATES or claim.predicate.endswith(
        ("_package", "_salary", "_rate", "_conversion_12m")
    )


def _flag(
    flag_id: str,
    weight: float,
    claim: ClaimView,
    evidence: str,
    **detail: Any,
) -> GamingFlag:
    return GamingFlag(
        flag_id=flag_id,
        weight=weight,
        claim_id=claim.claim_id,
        predicate=claim.predicate,
        evidence=evidence,
        detail=detail,
    )


def missing_denominator(claim: ClaimView, weight: float) -> GamingFlag | None:
    if not is_rate_claim(claim) or denominator(claim) is not None:
        return None
    return _flag(
        "missing_denominator",
        weight,
        claim,
        "rate published without a cohort denominator",
        unit=claim.unit,
    )


def no_distribution(claim: ClaimView, weight: float) -> GamingFlag | None:
    if not claim.predicate.endswith(("salary", "package", "ctc")):
        return None
    qualifier = claim.qualifier
    has_spread = any(
        qualifier.get(key) is not None for key in ("q25", "q1", "q75", "q3", "iqr", "range")
    )
    if has_spread or claim.predicate.startswith("p25_"):
        return None
    return _flag(
        "no_distribution",
        weight,
        claim,
        "salary published as a single figure with no distribution",
    )


def headline_is_highest_package(claim: ClaimView, weight: float) -> GamingFlag | None:
    qualifier = claim.qualifier
    if claim.predicate not in {"highest_package", "max_salary", "highest_ctc"} and not (
        qualifier.get("headline") and claim.predicate.endswith(("package", "salary", "ctc"))
    ):
        return None
    return _flag(
        "headline_is_highest_package",
        weight,
        claim,
        "headline figure is the highest package, not a central tendency",
    )


def round_numbers(claim: ClaimView, weight: float) -> GamingFlag | None:
    value = claim.number
    if value is None or value < 10_000:
        return None
    if value % 10_000 != 0:
        return None
    return _flag(
        "round_numbers",
        weight,
        claim,
        f"suspiciously round figure {value:g}",
        value=value,
    )


def yoy_jump_gt_2sigma(
    claim: ClaimView,
    weight: float,
    *,
    peer_sigma: float | None = None,
    previous: ClaimView | None = None,
) -> GamingFlag | None:
    sigma = numeric(claim.qualifier.get("peer_sigma"))
    if sigma is None:
        sigma = peer_sigma
    if sigma is None or sigma <= 0 or previous is None:
        return None
    current, prior = claim.number, previous.number
    if current is None or prior is None:
        return None
    if abs(current - prior) <= 2.0 * sigma:
        return None
    return _flag(
        "yoy_jump_gt_2sigma",
        weight,
        claim,
        "year-on-year jump exceeds 2 peer standard deviations",
        current=current,
        previous=prior,
        peer_sigma=sigma,
    )


def aggregated_across_degrees(claim: ClaimView, weight: float) -> GamingFlag | None:
    qualifier = claim.qualifier
    if not _is_outcome_claim(claim):
        return None
    if qualifier.get("credential_level") is None or qualifier.get("aggregated_across_degrees"):
        return _flag(
            "aggregated_across_degrees",
            weight,
            claim,
            "programme-level outcome not separated by credential level",
            credential_level=qualifier.get("credential_level"),
        )
    return None


def recruiters_without_counts(claim: ClaimView, weight: float) -> GamingFlag | None:
    if claim.predicate not in {"recruiters", "recruiter_list", "companies_visited"}:
        return None
    if claim.qualifier.get("count") is not None or claim.qualifier.get("n") is not None:
        return None
    if isinstance(claim.value, (list, tuple)) and not claim.value:
        return None
    return _flag(
        "recruiters_without_counts",
        weight,
        claim,
        "recruiter list published without counts",
    )


def offers_not_joinings(claim: ClaimView, weight: float) -> GamingFlag | None:
    if claim.predicate not in {"placement_rate", "employment_rate", "offers", "offer_count"}:
        return None
    outcome = str(claim.qualifier.get("outcome_type", "")).lower()
    if outcome == "joining":
        return None
    evidence = claim.quote or str(claim.value)
    if outcome != "offer" and not OFFER_MARKERS.search(evidence):
        return None
    if JOINING_MARKERS.search(evidence):
        return None
    return _flag(
        "offers_not_joinings",
        weight,
        claim,
        "offer letters presented as joinings",
        outcome_type=outcome or "offer",
    )


def internship_counted_as_placement(claim: ClaimView, weight: float) -> GamingFlag | None:
    outcome = str(claim.qualifier.get("outcome_type", "")).lower()
    evidence = claim.quote or str(claim.value)
    if claim.predicate.startswith("internship_to_offer"):
        # internship-to-offer conversion legitimately mentions interns
        return None
    if outcome == "internship" or (
        _is_outcome_claim(claim) and INTERNSHIP_MARKERS.search(evidence)
    ):
        return _flag(
            "internship_counted_as_placement",
            weight,
            claim,
            "internship counted as employment/placement (NCAHP: internship is part of the degree)",
            outcome_type=outcome or "internship",
        )
    return None


def no_cohort_year(claim: ClaimView, weight: float) -> GamingFlag | None:
    if not _is_outcome_claim(claim):
        return None
    qualifier = claim.qualifier
    if qualifier.get("cohort_year") is not None or qualifier.get("year") is not None:
        return None
    return _flag(
        "no_cohort_year",
        weight,
        claim,
        "outcome published without a cohort year",
    )


SIMPLE_DETECTORS: dict[str, Any] = {
    "missing_denominator": missing_denominator,
    "no_distribution": no_distribution,
    "headline_is_highest_package": headline_is_highest_package,
    "round_numbers": round_numbers,
    "aggregated_across_degrees": aggregated_across_degrees,
    "recruiters_without_counts": recruiters_without_counts,
    "offers_not_joinings": offers_not_joinings,
    "internship_counted_as_placement": internship_counted_as_placement,
    "no_cohort_year": no_cohort_year,
}

MISSING_DETECTORS: tuple[str, ...] = tuple(
    flag_id
    for flag_id in DETECTORS
    if flag_id not in SIMPLE_DETECTORS and flag_id != "yoy_jump_gt_2sigma"
)


def detect(
    claims: Iterable[Any],
    flag_weights: Mapping[str, float],
    *,
    peer_sigma: float | None = None,
) -> tuple[GamingFlag, ...]:
    """Run every configured detector over every claim."""

    index = ClaimIndex.build(claims)
    flags: list[GamingFlag] = []

    for claim in index:
        for flag_id, detector in SIMPLE_DETECTORS.items():
            weight = flag_weights.get(flag_id)
            if weight is None:
                continue
            found = detector(claim, float(weight))
            if found is not None:
                flags.append(found)
        if "yoy_jump_gt_2sigma" in flag_weights:
            previous_claims = [
                other
                for other in index.all(claim.predicate)
                if other is not claim
                and other.observed_at is not None
                and claim.observed_at is not None
                and other.observed_at < claim.observed_at
            ]
            previous = previous_claims[-1] if previous_claims else None
            found = yoy_jump_gt_2sigma(
                claim,
                float(flag_weights["yoy_jump_gt_2sigma"]),
                peer_sigma=peer_sigma,
                previous=previous,
            )
            if found is not None:
                flags.append(found)

    return tuple(sorted(flags, key=lambda f: (f.flag_id, str(f.claim_id), f.predicate)))


def confidence_penalties(
    flags: Iterable[GamingFlag], *, floor: float = MIN_CONFIDENCE
) -> dict[str, float]:
    """Per-claim multiplicative penalty: ``1 - max(floor, prod(1 - weight))``.

    A floor keeps a pile-up of low-weight flags from reducing confidence to
    zero (which would silently erase a claim instead of displaying it).
    """

    by_claim: dict[str, float] = {}
    for flag in flags:
        if flag.claim_id is None:
            continue
        key = str(flag.claim_id)
        remaining = by_claim.get(key, 1.0) * (1.0 - min(1.0, max(0.0, flag.weight)))
        by_claim[key] = remaining
    return {
        key: min(1.0, max(0.0, 1.0 - max(floor, remaining)))
        for key, remaining in by_claim.items()
    }
