"""Expert-panel evaluation kit (docs/v1/eval/02-findings-viable.md P4; design §2.2).

A small panel of domain experts produces ranked preferences for a handful of
profiles; the harness compares the panel consensus to the product's ranking
and reports inter-rater reliability separately. This stream is *slow* (human
time) and fixture-driven offline: committed fixtures are synthetic and clearly
marked; real panels attach provenance per rater (who, expertise, conflict
disclosure, when) and the harness refuses files without it.

Harness-computed metrics (invariant 2 — raters emit preferences, never scores):

* inter-rater reliability: Krippendorff's alpha (nominal and ordinal) over
  rank values on the pairable (profile, institution) units, and ICC(2,1)
  (two-way random, single measures) on the complete-case matrix;
* agreement: per-profile top-3 overlap and Kendall tau-b between the panel's
  mean-rank consensus and the product ranking;
* effectiveness interval: mean top-3 overlap with a percentile-bootstrap
  confidence interval (deterministic: fixed seed, no wall-clock randomness).

The panel measures agreement with experts — it cannot prove outcome
correctness (selection/panel-composition bias; design §2.2 "Gaps").
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from fmd.eval.truth import kendall_tau_b

SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20261122
BOOTSTRAP_RESAMPLES = 1000


class PanelError(ValueError):
    """A panel fixture violates the panel-evaluation contract."""


class Rater(BaseModel):
    """One panel member with the provenance the audit protocol requires."""

    model_config = ConfigDict(extra="forbid")

    rater_id: str
    expertise: str
    conflict_disclosure: str
    rated_at: str

    @model_validator(mode="after")
    def _rater_rules(self) -> Rater:
        if not self.expertise.strip():
            raise ValueError(f"{self.rater_id}: expertise must be recorded")
        if not self.conflict_disclosure.strip():
            raise ValueError(
                f"{self.rater_id}: conflict_disclosure must be recorded "
                "(may be 'none', not omitted)"
            )
        return self


class PanelProfileRatings(BaseModel):
    """One profile rated by the panel: each rater's ranked subset."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    ratings: dict[str, list[str]]  # rater_id -> institutions in preference order

    @model_validator(mode="after")
    def _ratings_rules(self) -> PanelProfileRatings:
        if not self.ratings:
            raise ValueError(f"{self.profile_id}: no rater ratings")
        for rater_id, ranking in self.ratings.items():
            if len(set(ranking)) != len(ranking):
                raise ValueError(f"{self.profile_id}/{rater_id}: duplicate institution in ranking")
        return self


class PanelRatings(BaseModel):
    """The panel fixture: raters + per-profile rankings + honesty flags."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    panel_id: str
    synthetic: bool = False
    notes: str | None = None
    raters: list[Rater]
    profiles: list[PanelProfileRatings]

    @model_validator(mode="after")
    def _panel_rules(self) -> PanelRatings:
        if not self.synthetic:
            raise ValueError(
                f"{self.panel_id}: panel fixtures must declare synthetic: true "
                "(real panels carry rater provenance and sign-off instead)"
            )
        known = {rater.rater_id for rater in self.raters}
        for profile in self.profiles:
            unknown = set(profile.ratings) - known
            if unknown:
                raise ValueError(
                    f"{profile.profile_id}: ratings reference unknown raters {sorted(unknown)}"
                )
        return self


class ProductRanking(BaseModel):
    """The product's ranking per profile, frozen at eval time."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    notes: str | None = None
    profiles: list[ProductRankingProfile]


class ProductRankingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    ranking: list[str]


def load_panel_ratings(fixture_dir: str | Path) -> PanelRatings:
    directory = Path(fixture_dir)
    path = directory / "panel_ratings.yaml"
    if not path.is_file():
        raise PanelError(f"panel ratings not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PanelError(f"cannot read {path}: {exc}") from exc
    try:
        return PanelRatings.model_validate(payload)
    except Exception as exc:
        raise PanelError(f"{path}: panel ratings invalid: {exc}") from exc


def load_product_ranking(fixture_dir: str | Path) -> ProductRanking:
    directory = Path(fixture_dir)
    path = directory / "product_ranking.json"
    if not path.is_file():
        raise PanelError(f"product ranking not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PanelError(f"cannot read {path}: {exc}") from exc
    try:
        return ProductRanking.model_validate(payload)
    except Exception as exc:
        raise PanelError(f"{path}: product ranking invalid: {exc}") from exc


# --------------------------------------------------------------------------- reliability


def krippendorff_alpha(
    units: list[list[float | None]], metric: Literal["nominal", "ordinal"] = "ordinal"
) -> float | None:
    """Krippendorff's alpha over incomplete unit × rater values.

    Implements Krippendorff (2011), "Computing Krippendorff's
    Alpha-Reliability", §C–D (any number of observers, missing data):

    * coincidence matrix ``o_ck = Σ_u n_{u,c}·n_{u,k}/(m_u−1)`` for c≠k and
      ``o_cc = Σ_u n_{u,c}·(n_{u,c}−1)/(m_u−1)`` (ordered pairs of distinct
      ratings; each pairable unit contributes m_u(m_u−1)/(m_u−1) = m_u);
    * margins n_c count only pairable units' values; n = Σ margins;
    * ``D_o = Σ_{c≠k} o_ck·δ_ck / n`` and
      ``D_e = Σ_{c≠k} n_c·n_k·δ_ck / (n(n−1))``; α = 1 − D_o/D_e;
    * nominal δ ∈ {0,1}; ordinal δ = (Σ_{g=c..k} n_g − (n_c+n_k)/2)² over the
      value ranks (standardisation is optional and does not affect α).

    Verified against the paper's printed examples: the binary two-observer
    case (α = 0.095) and the 4-observer/12-unit nominal case (α = 0.743) are
    reproduced exactly by the committed tests.

    Returns None when fewer than two pairable units exist — never 1.0 for an
    unmeasurable case.
    """

    pairable_rows = [[v for v in unit if v is not None] for unit in units]
    pairable_rows = [row for row in pairable_rows if len(row) >= 2]
    if len(pairable_rows) < 2:
        return None
    values = sorted({v for row in pairable_rows for v in row})
    if len(values) < 2:
        return None
    # margins over pairable units only (the paper's n_c / n)
    counts = {v: sum(1 for row in pairable_rows for v2 in row if v2 == v) for v in values}
    n = sum(counts.values())
    if n < 2:
        return None

    def distance(a: float, b: float) -> float:
        if a == b:
            return 0.0
        if metric == "nominal":
            return 1.0
        lo, hi = (a, b) if a <= b else (b, a)
        cum = sum(
            counts[v]
            for v in values[values.index(lo) : values.index(hi) + 1]
        )
        cum -= (counts[lo] + counts[hi]) / 2
        return cum**2

    # coincidence matrix
    off_diagonal = 0.0
    for a in values:
        for b in values:
            if a == b:
                continue
            total = 0.0
            for row in pairable_rows:
                m = len(row)
                n_a = sum(1 for v in row if v == a)
                n_b = sum(1 for v in row if v == b)
                total += n_a * n_b / (m - 1)
            off_diagonal += total * distance(a, b)
    d_o = off_diagonal / n

    expected = 0.0
    for a in values:
        for b in values:
            if a == b:
                continue
            expected += counts[a] * counts[b] * distance(a, b)
    d_e = expected / (n * (n - 1))
    if d_e <= 0:
        return None
    return round(1.0 - d_o / d_e, 6)


def icc_two_way_single(matrix: list[list[float]]) -> float | None:
    """ICC(2,1): two-way random effects, single measures (complete cases)."""

    n = len(matrix)
    k = len(matrix[0]) if n else 0
    if n < 2 or k < 2 or any(len(row) != k for row in matrix):
        return None
    grand = sum(sum(row) for row in matrix) / (n * k)
    row_means = [sum(row) / k for row in matrix]
    col_means = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]
    ssr = k * sum((m - grand) ** 2 for m in row_means)
    ssc = n * sum((m - grand) ** 2 for m in col_means)
    sse = sum(
        (matrix[i][j] - row_means[i] - col_means[j] + grand) ** 2
        for i in range(n)
        for j in range(k)
    )
    bms = ssr / (n - 1)
    jms = ssc / (k - 1)
    ems = sse / ((n - 1) * (k - 1))
    denom = bms + (k - 1) * ems + k * (jms - ems) / n
    if denom <= 0:
        return None
    return round((bms - ems) / denom, 6)


# --------------------------------------------------------------------------- panel run


def _mean_rank_consensus(rankings: list[list[str]]) -> list[str]:
    """Panel consensus: institutions by mean rank (ties broken by name)."""

    positions: dict[str, list[int]] = {}
    for ranking in rankings:
        for position, name in enumerate(ranking, start=1):
            positions.setdefault(name, []).append(position)
    means = {
        name: sum(pos) / len(pos) for name, pos in positions.items() if len(pos) >= 2
    }
    return [name for name, _ in sorted(means.items(), key=lambda kv: (kv[1], kv[0]))]


def _alpha_units(panel: PanelRatings) -> list[list[float | None]]:
    """Unit = (profile, institution); value = rater's rank; None = not ranked."""

    names: list[str] = []
    seen: set[str] = set()
    for profile in panel.profiles:
        for ranking in profile.ratings.values():
            for name in ranking:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    rater_ids = [rater.rater_id for rater in panel.raters]
    units: list[list[float | None]] = []
    for profile in panel.profiles:
        for name in names:
            if not any(name in ranking for ranking in profile.ratings.values()):
                continue  # institution never rated for this profile: not a unit
            row: list[float | None] = []
            for rater_id in rater_ids:
                ranking = profile.ratings.get(rater_id) or []
                row.append(float(ranking.index(name) + 1) if name in ranking else None)
            units.append(row)
    return units


def run_expert_panel(panel: PanelRatings, ranking: ProductRanking, *, k: int = 3) -> dict[str, Any]:
    product_by_profile = {profile.profile_id: profile.ranking for profile in ranking.profiles}
    cases: list[dict[str, Any]] = []
    overlaps: list[float] = []
    taus: list[float] = []
    for profile in panel.profiles:
        product_order = product_by_profile.get(profile.profile_id)
        if product_order is None:
            raise PanelError(
                f"{profile.profile_id}: product ranking fixture has no entry for this profile"
            )
        consensus = _mean_rank_consensus(list(profile.ratings.values()))
        top_k = min(k, len(consensus), len(product_order))
        overlap = len(set(consensus[:top_k]) & set(product_order[:top_k])) if top_k else 0
        overlaps.append(overlap / top_k if top_k else 0.0)
        shared = [name for name in product_order if name in set(consensus)]
        if len(shared) >= 2:
            product_pos = {name: pos for pos, name in enumerate(product_order, start=1)}
            consensus_pos = {name: pos for pos, name in enumerate(consensus, start=1)}
            tau = kendall_tau_b(
                [product_pos[name] for name in shared],
                [consensus_pos[name] for name in shared],
            )
        else:
            tau = None
        if tau is not None:
            taus.append(tau)
        cases.append(
            {
                "profile_id": profile.profile_id,
                "panel_consensus_top3": consensus[:3],
                "product_top3": product_order[:3],
                "top3_overlap": overlap,
                "top3_overlap_fraction": overlaps[-1],
                "kendall_tau_b": tau,
                "n_raters": len(profile.ratings),
            }
        )

    units = _alpha_units(panel)
    alpha_nominal = krippendorff_alpha(units, metric="nominal")
    alpha_ordinal = krippendorff_alpha(units, metric="ordinal")
    complete = [
        row
        for row in _alpha_units(panel)
        if all(v is not None for v in row)
    ]
    icc = icc_two_way_single(complete) if complete else None

    low, high, bootstrap_note = _bootstrap_interval(overlaps)
    return {
        "panel_id": panel.panel_id,
        "raters": [rater.model_dump() for rater in panel.raters],
        "cases": cases,
        "inter_rater_reliability": {
            "krippendorff_alpha_nominal": alpha_nominal,
            "krippendorff_alpha_ordinal": alpha_ordinal,
            "icc_two_way_single": icc,
            "pairwise_kendall_tau_b_mean": _pairwise_rater_tau(panel),
            "n_complete_case_units": len(complete),
            "n_units": len(units),
            "note": (
                "alpha on rank values is deliberately harsh: it measures absolute "
                "rank-position agreement, not order agreement; the pairwise tau "
                "below is the order-agreement view"
            ),
        },
        "effectiveness": {
            "metric": "mean top-3 overlap fraction (panel consensus vs product)",
            "mean": round(sum(overlaps) / len(overlaps), 6) if overlaps else None,
            "ci95_low": low,
            "ci95_high": high,
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "method": "percentile bootstrap over profiles (deterministic)",
                "note": bootstrap_note,
            },
            "kendall_tau_b_mean": round(sum(taus) / len(taus), 6) if taus else None,
        },
        "audit_note": (
            "panel measures agreement with the consulted experts; it cannot prove "
            "outcome correctness (panel-composition and selection bias, design §2.2)"
        ),
    }


def _pairwise_rater_tau(panel: PanelRatings) -> float | None:
    """Mean Kendall tau-b between every rater pair on shared institutions."""

    rater_ids = [rater.rater_id for rater in panel.raters]
    taus: list[float] = []
    for i, left in enumerate(rater_ids):
        for right in rater_ids[i + 1 :]:
            for profile in panel.profiles:
                left_rank = profile.ratings.get(left) or []
                right_rank = profile.ratings.get(right) or []
                shared = [name for name in left_rank if name in set(right_rank)]
                if len(shared) < 2:
                    continue
                left_pos = {name: pos for pos, name in enumerate(left_rank, start=1)}
                right_pos = {name: pos for pos, name in enumerate(right_rank, start=1)}
                tau = kendall_tau_b(
                    [left_pos[name] for name in shared],
                    [right_pos[name] for name in shared],
                )
                if tau is not None:
                    taus.append(tau)
    return round(sum(taus) / len(taus), 6) if taus else None


def _bootstrap_interval(
    values: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED, level: float = 0.95,
) -> tuple[float | None, float | None, str]:
    """Deterministic percentile bootstrap over profile-level overlaps.

    Returns ``(low, high, note)``; the interval is None with an honest note
    when fewer than two profiles exist (a bootstrap over one value is a point,
    and calling it a confidence interval would be a lie).
    """

    if len(values) < 2:
        return None, None, "ci95 requires >= 2 profiles; n=1 gives a point, not an interval"
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - level) / 2
    low = means[max(0, int(alpha * resamples))]
    high = means[min(resamples - 1, int((1 - alpha) * resamples) - 1)]
    return round(low, 6), round(high, 6), "percentile bounds over resample means"


def render_panel_summary(case: dict[str, Any]) -> str:
    reliability = case.get("inter_rater_reliability") or {}
    effectiveness = case.get("effectiveness") or {}
    lines = [
        f"fmd eval (expert-panel): {case.get('panel_id')} "
        f"({len(case.get('raters') or [])} raters, {len(case.get('cases') or [])} profile(s))",
        "  alpha_nominal {an} alpha_ordinal {ao} icc(2,1) {icc}".format(
            an=reliability.get("krippendorff_alpha_nominal"),
            ao=reliability.get("krippendorff_alpha_ordinal"),
            icc=reliability.get("icc_two_way_single"),
        ),
        "  effectiveness: mean {mean} ci95 [{lo}, {hi}]".format(
            mean=effectiveness.get("mean"),
            lo=effectiveness.get("ci95_low"),
            hi=effectiveness.get("ci95_high"),
        ),
    ]
    for profile_case in case.get("cases") or []:
        lines.append(
            "  {profile}: top3_overlap {ov} tau {tau}".format(
                profile=profile_case.get("profile_id"),
                ov=profile_case.get("top3_overlap"),
                tau=profile_case.get("kendall_tau_b"),
            )
        )
    return "\n".join(lines)


__all__ = [
    "BOOTSTRAP_SEED",
    "PanelError",
    "PanelRatings",
    "ProductRanking",
    "Rater",
    "icc_two_way_single",
    "krippendorff_alpha",
    "load_panel_ratings",
    "load_product_ranking",
    "render_panel_summary",
    "run_expert_panel",
]
