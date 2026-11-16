"""Prioritise which discovered institution domains get crawled (pure).

An inventory of thousands of universities cannot all be crawled in one run. The
budget is bounded by registry parameters and spent on the domains most likely to
publish outcomes evidence. Every weight is data:

    params:
      targeting:
        max_domains_per_run: 25
        base: 0
        type_weights: {central: 40, deemed: 30, state: 20}
        known_website_bonus: 15
        naac_bonus: 25

``priority_for`` is deterministic and explainable: it returns a score plus the
reasons that produced it, so a run ledger can say *why* a domain was selected.
Nothing here reads the network or the database.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetWeights:
    base: int = 0
    type_weights: Mapping[str, int] = field(default_factory=dict)
    known_website_bonus: int = 0
    naac_bonus: int = 0
    #: Hard cap on domains handed to the institution-site adapter in one run.
    max_domains_per_run: int = 25

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> TargetWeights:
        raw = params.get("targeting") or {}
        if not isinstance(raw, Mapping):
            raw = {}
        type_weights = raw.get("type_weights") or {}
        weights: dict[str, int] = {}
        if isinstance(type_weights, Mapping):
            for key, value in type_weights.items():
                try:
                    weights[str(key).strip().casefold()] = int(value)
                except (TypeError, ValueError):
                    continue
        return cls(
            base=_int(raw.get("base"), 0),
            type_weights=weights,
            known_website_bonus=_int(raw.get("known_website_bonus"), 0),
            naac_bonus=_int(raw.get("naac_bonus"), 0),
            max_domains_per_run=max(0, _int(raw.get("max_domains_per_run"), 25)),
        )

    @classmethod
    def from_source(cls, source: object) -> TargetWeights:
        from fmd.discover.base import params_from_source

        return cls.from_params(params_from_source(source))


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TargetScore:
    priority: int
    reasons: tuple[str, ...] = ()


def priority_for(
    *,
    institution_type: str | None = None,
    has_website: bool = False,
    naac_available: bool = False,
    weights: TargetWeights | None = None,
) -> TargetScore:
    """Score one domain from facts ingest actually observed."""
    weights = weights or TargetWeights()
    score = weights.base
    reasons: list[str] = []
    if institution_type:
        bonus = weights.type_weights.get(institution_type.strip().casefold(), 0)
        if bonus:
            score += bonus
            reasons.append(f"type:{institution_type.strip().casefold()}+{bonus}")
    if has_website and weights.known_website_bonus:
        score += weights.known_website_bonus
        reasons.append(f"known_website+{weights.known_website_bonus}")
    if naac_available and weights.naac_bonus:
        score += weights.naac_bonus
        reasons.append(f"naac_available+{weights.naac_bonus}")
    return TargetScore(priority=score, reasons=tuple(reasons))


def select_targets(
    candidates: Iterable[Any],
    *,
    weights: TargetWeights | None = None,
    naac_domains: Iterable[str] = (),
    limit: int | None = None,
) -> list[Any]:
    """Return the bounded, deterministic subset to crawl.

    ``candidates`` is anything with ``domain``/``seed_url`` plus optional
    ``institution_type``/``priority`` attributes (an ``InstitutionDomainRecord``
    or a plain object in tests). Order: priority desc, then domain asc so two
    runs over the same data take the same decision.
    """
    weights = weights or TargetWeights()
    naac = {value.casefold() for value in naac_domains if value}
    cap = weights.max_domains_per_run if limit is None else max(0, limit)
    seen: set[str] = set()
    scored: list[tuple[int, str, Any]] = []
    for candidate in candidates:
        domain = str(getattr(candidate, "domain", "") or "").strip().lower()
        if not domain or domain in seen:
            continue
        seen.add(domain)
        existing = _int(getattr(candidate, "priority", 0), 0)
        computed = priority_for(
            institution_type=getattr(candidate, "institution_type", None),
            has_website=bool(getattr(candidate, "seed_url", None)),
            naac_available=domain in naac,
            weights=weights,
        )
        scored.append((max(existing, computed.priority), domain, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _priority, _domain, candidate in scored[:cap]]


__all__ = ["TargetScore", "TargetWeights", "priority_for", "select_targets"]
