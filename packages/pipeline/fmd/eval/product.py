"""Product streams for ``fmd eval --eval-kind product`` (docs/v1/eval/01-eval-designs.md).

Four offline, deterministic streams:

* ``nirf-proxy`` / ``naac-proxy`` — rank a shortlist against the committed
  truth tables. NIRF is a weak institutional proxy (no paramedical category);
  NAAC CGPA is a medium-maturity institutional proxy. **Neither is a
  programme-level outcome measure**, and every report carries that label.
* ``self-drift`` — diff two versioned shortlist snapshots: rank-change
  magnitude, churn, and how many changes a claims diff or a version bump
  explains (mirrors ``fmd/eval/drift.py`` semantics at shortlist level).
* ``cv-stability`` — leave-N-out and ±ε weight-noise stability of a ranking
  over the product's own candidate set. This measures internal consistency,
  not external correctness; the report says so.

No user-specific logic lives here (invariant 1): profiles enter as fixture
paths at runtime. Every number in a report is computed by this harness
(invariant 2); UNKNOWN is a value, never 0 (invariant 5).
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fmd.eval.drift import top_n_jaccard
from fmd.eval.run import write_report
from fmd.eval.truth import (
    NaacTruthFile,
    NirfTruthFile,
    TruthError,
    best_available_hit,
    kendall_tau_b,
    load_truth_file,
    normalise_institution_key,
    rank_correlation_on_overlap,
    top_k_overlap,
)

SCHEMA_VERSION = 1
DEFAULT_K_VALUES = (3, 5)

PROXY_LABEL_NIRF = (
    "institution-level proxy: NIRF publishes no paramedical/MLT category, so a "
    "rank says nothing about a specific B.Sc. MLT programme or its outcomes"
)
PROXY_LABEL_NAAC = (
    "institution-level proxy, medium maturity: NAAC grades are institutional, "
    "on a 5-year cycle, and are not programme-level outcome measures"
)


class ShortlistError(ValueError):
    """A shortlist / snapshot fixture is malformed."""


class ShortlistRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    institution_name: str
    institution_key: str | None = None
    programme: str = ""
    state: str | None = None
    claims_hash: str | None = None
    evidence_urls: list[str] = Field(default_factory=list)
    sub_scores: dict[str, float] = Field(default_factory=dict)
    score: float | None = None

    @model_validator(mode="after")
    def _row_rules(self) -> ShortlistRow:
        if self.rank < 1:
            raise ValueError(f"{self.institution_name}: rank must be >= 1")
        if not self.institution_name.strip():
            raise ValueError("shortlist row with empty institution_name")
        if self.institution_key is None:
            self.institution_key = normalise_institution_key(self.institution_name)
        if self.sub_scores:
            unknown = {v for v in self.sub_scores.values() if isinstance(v, str)}
            if unknown:
                raise ValueError(
                    f"{self.institution_name}: sub_scores must be numeric; UNKNOWN is "
                    "expressed by omitting the component, never as a string value"
                )
        return self


class Shortlist(BaseModel):
    """A product shortlist fixture (synthetic in tests, runtime input in evals)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profile_id: str
    synthetic: bool = False
    notes: str | None = None
    captured_at: str | None = None
    scorer_version: str | None = None
    weights_version: str | None = None
    rows: list[ShortlistRow]

    @model_validator(mode="after")
    def _list_rules(self) -> Shortlist:
        if not self.rows:
            raise ValueError(f"{self.profile_id}: shortlist has no rows")
        ranks = [row.rank for row in self.rows]
        if ranks != sorted(ranks) or ranks[0] != 1 or len(set(ranks)) != len(ranks):
            raise ValueError(f"{self.profile_id}: ranks must be a 1..N series without gaps")
        return self

    def ordered_keys(self) -> list[str]:
        return [row.institution_key for row in self.rows if row.institution_key]


def load_shortlist(path: str | Path) -> Shortlist:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShortlistError(f"cannot read shortlist {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ShortlistError(f"{path}: shortlist must be a JSON object")
    try:
        return Shortlist.model_validate(payload)
    except Exception as exc:
        raise ShortlistError(f"{path}: shortlist invalid: {exc}") from exc


# --------------------------------------------------------------------------- proxy streams


def run_proxy_stream(
    shortlist: Shortlist,
    truth: NirfTruthFile | NaacTruthFile,
    *,
    proxy_label: str,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Score one shortlist against one truth table (contract §1.2).

    The reference ranking is the truth table's order restricted to the
    institutions the shortlist names. A shortlist institution with no truth
    entry is UNKNOWN: it is excluded from correlations, cannot match the
    reference top-k, and is counted in ``unknown`` coverage — never coerced
    to 0 and never imputed.
    """

    ranked_keys = truth.ranked_keys()
    by_key = truth.by_key()
    reference_order = [key for key in ranked_keys if key in set(shortlist.ordered_keys())]
    reference_positions = {key: position for position, key in enumerate(reference_order, start=1)}
    product_positions = {row.institution_key: row.rank for row in shortlist.rows}

    covered_order = [
        row.institution_key
        for row in shortlist.rows
        if row.institution_key in by_key
    ]
    unknown_keys = [key for key in shortlist.ordered_keys() if key not in by_key]

    overlaps: dict[str, int] = {}
    for k in k_values:
        overlaps[str(k)] = top_k_overlap(shortlist.ordered_keys(), reference_order, k)

    product_top = shortlist.ordered_keys()[0] if shortlist.rows else None
    top_is_unknown = product_top is None or product_top not in by_key

    correlation = rank_correlation_on_overlap(
        covered_order,
        reference_order,
        product_positions=product_positions,
        reference_positions=reference_positions,
    )

    return {
        "profile_id": shortlist.profile_id,
        "shortlist_rows": len(shortlist.rows),
        "truth_entries": len(truth.entries),
        "top_k_overlap": overlaps,
        "top_k_overlap_fraction": {
            k: round(value / min(int(k), len(reference_order)), 6) if reference_order else None
            for k, value in overlaps.items()
        },
        "best_available": {
            # UNKNOWN product #1 is a None, not a miss: it cannot be scored.
            "hit": (
                None if top_is_unknown
                else best_available_hit(shortlist.ordered_keys(), reference_order)
            ),
            "product_top_institution": product_top,
            "scored": not top_is_unknown,
        },
        "rank_correlation": correlation,
        "unknown": {
            "count": len(unknown_keys),
            "fraction": (
                round(len(unknown_keys) / len(shortlist.rows), 6) if shortlist.rows else None
            ),
            "institutions": unknown_keys[:10],
        },
        "reference_order": reference_order,
        "proxy_label": proxy_label,
    }


def run_nirf_proxy(
    shortlist: Shortlist, truth: NirfTruthFile, *, k_values: tuple[int, ...] = DEFAULT_K_VALUES
) -> dict[str, Any]:
    return run_proxy_stream(shortlist, truth, proxy_label=PROXY_LABEL_NIRF, k_values=k_values)


def run_naac_proxy(
    shortlist: Shortlist, truth: NaacTruthFile, *, k_values: tuple[int, ...] = DEFAULT_K_VALUES
) -> dict[str, Any]:
    return run_proxy_stream(shortlist, truth, proxy_label=PROXY_LABEL_NAAC, k_values=k_values)


# --------------------------------------------------------------------------- self-drift


def run_self_drift(current: Shortlist, previous: Shortlist) -> dict[str, Any]:
    """Diff two snapshots of the same profile's shortlist (design §2.4).

    A rank change is ``explained_by: claims_diff`` when the row's claims hash
    changed, ``version_change`` when scorer/weights versions changed without a
    claims diff, and otherwise unexplained. Added/dropped rows follow the same
    rule on claims hashes. Identical snapshots produce zero changes — that is
    the offline canary.
    """

    if current.profile_id != previous.profile_id:
        raise ShortlistError(
            f"self-drift compares one profile; got "
            f"{current.profile_id!r} vs {previous.profile_id!r}"
        )
    version_changed = (
        current.scorer_version != previous.scorer_version
        or current.weights_version != previous.weights_version
    )
    prev_rows = {row.institution_key: row for row in previous.rows}
    curr_rows = {row.institution_key: row for row in current.rows}

    changes: list[dict[str, Any]] = []
    rank_changes: list[int] = []
    for key, curr in curr_rows.items():
        prev = prev_rows.get(key)
        if prev is None:
            explained = (
                "claims_diff"
                if curr.claims_hash and curr.claims_hash not in _hashes(previous.rows)
                else None
            )
            changes.append(
                {"institution": key, "change": "added", "explained_by": explained,
                 "claims_hash": curr.claims_hash}
            )
            continue
        delta = prev.rank - curr.rank
        if delta != 0:
            rank_changes.append(abs(delta))
            if curr.claims_hash != prev.claims_hash:
                explained_by = "claims_diff"
            elif version_changed:
                explained_by = "version_change"
            else:
                explained_by = None
            changes.append(
                {"institution": key, "change": "rank_move", "from": prev.rank, "to": curr.rank,
                 "magnitude": abs(delta), "explained_by": explained_by,
                 "claims_hash_prev": prev.claims_hash, "claims_hash_curr": curr.claims_hash}
            )
    for key, prev in prev_rows.items():
        if key not in curr_rows:
            new_claims = prev.claims_hash and prev.claims_hash not in _hashes(current.rows)
            explained = "claims_diff" if new_claims else None
            changes.append(
                {"institution": key, "change": "dropped", "explained_by": explained,
                 "claims_hash": prev.claims_hash}
            )

    explained_count = sum(1 for change in changes if change["explained_by"] is not None)
    return {
        "profile_id": current.profile_id,
        "scorer_version": {
            "previous": previous.scorer_version, "current": current.scorer_version
        },
        "weights_version": {
            "previous": previous.weights_version, "current": current.weights_version
        },
        "captured_at": {"previous": previous.captured_at, "current": current.captured_at},
        "changes": changes,
        "n_changes": len(changes),
        "rank_changes": {
            "count": len(rank_changes),
            "max_abs": max(rank_changes) if rank_changes else 0,
            "mean_abs": round(sum(rank_changes) / len(rank_changes), 6) if rank_changes else 0.0,
        },
        "top10_churn": round(1.0 - top_n_jaccard(
            [row.institution_key for row in previous.rows],
            [row.institution_key for row in current.rows],
            n=10,
        ), 6),
        "evidence_driven_fraction": (
            round(explained_count / len(changes), 6) if changes else None
        ),
        "unexplained_count": len(changes) - explained_count,
    }


def _hashes(rows: list[ShortlistRow]) -> set[str]:
    return {row.claims_hash for row in rows if row.claims_hash}


# --------------------------------------------------------------------------- cv-stability


def _weighted_score(sub_scores: dict[str, float], weights: dict[str, float]) -> float:
    """Harness-computed composite (invariant 2): sum(w_i * s_i) / sum(w_i)."""

    used = {name: weight for name, weight in weights.items() if name in sub_scores}
    total_weight = sum(used.values())
    if total_weight <= 0:
        return 0.0
    return sum(sub_scores[name] * weight for name, weight in used.items()) / total_weight


def _rank_order(candidates: list[ShortlistRow], weights: dict[str, float]) -> list[str]:
    """Deterministic ranking: score desc, then institution_key asc for ties."""

    scored = [(_weighted_score(row.sub_scores, weights), row.institution_key) for row in candidates]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [key for _score, key in scored]


def _perturb(weights: dict[str, float], epsilon: float) -> list[dict[str, float]]:
    """All ±ε sign combinations over the configured weights (deterministic)."""

    names = sorted(weights)
    variants: list[dict[str, float]] = []
    for signs in itertools.product((1.0, -1.0), repeat=len(names)):
        variant = {}
        for name, sign in zip(names, signs, strict=True):
            delta = epsilon * sign
            new_weight = weights[name] * (1.0 + delta)
            variant[name] = max(new_weight, 0.0)  # a weight never turns negative
        variants.append(variant)
    return variants


def run_cv_stability(
    shortlist: Shortlist,
    weights: dict[str, float],
    *,
    k: int = 5,
    epsilon: float = 0.1,
    holdout_n: int = 3,
) -> dict[str, Any]:
    """Leave-N-out and ±ε weight-noise stability (design §2.1).

    With additive per-programme scoring, holding candidates out cannot change
    the survivors' order — the leave-N-out Kendall-tau is reported anyway (it
    is the design's metric) but flagged as structurally 1.0. The informative
    numbers are the ±ε weight-noise tau and the top-k survival rates. Missing
    sub-score components are the honest UNKNOWN propagation: a candidate with
    a missing component scores from the components it has (the scorer's own
    ``min_components`` rule), which the noise can reorder.
    """

    rows = [row for row in shortlist.rows if row.sub_scores]
    if len(rows) < 3:
        raise ShortlistError(
            f"{shortlist.profile_id}: cv-stability needs >= 3 rows with sub_scores"
        )
    base_order = _rank_order(rows, weights)

    # Leave-N-out: drop the top-N, compare survivor order to the base order
    # restricted to survivors (structurally identical under additive scoring).
    holdout_results = []
    for n in range(1, min(holdout_n, len(rows) - 2) + 1):
        survivors = base_order[n:]
        base_restricted = [key for key in base_order if key in set(survivors)]
        tau = kendall_tau_b(
            [base_restricted.index(key) for key in survivors],
            [survivors.index(key) for key in survivors],
        )
        holdout_results.append({"n": n, "kendall_tau_b": tau, "n_survivors": len(survivors)})

    variants = _perturb(weights, epsilon)
    taus: list[float] = []
    survival: list[float] = []
    heldout_surface: list[float] = []
    base_top_k = set(base_order[:k])
    for variant in variants:
        variant_order = _rank_order(rows, variant)
        tau = kendall_tau_b(
            [base_order.index(key) for key in variant_order],
            list(range(1, len(variant_order) + 1)),
        )
        if tau is not None:
            taus.append(tau)
        variant_top_k = set(variant_order[: min(k, len(variant_order))])
        if base_top_k:
            survival.append(len(base_top_k & variant_top_k) / len(base_top_k))
        for n in range(1, min(holdout_n, len(rows) - 2) + 1):
            held_out = base_order[n - 1]  # the row that leave-n removes
            if held_out in variant_top_k:
                heldout_surface.append(1.0)
            else:
                heldout_surface.append(0.0)

    return {
        "profile_id": shortlist.profile_id,
        "n_candidates": len(rows),
        "weights": dict(sorted(weights.items())),
        "epsilon": epsilon,
        "k": k,
        "leave_n_out": holdout_results,
        "leave_n_out_note": (
            "structurally 1.0: scores are additive per programme, so holding "
            "candidates out cannot reorder the survivors; the weight-noise "
            "metrics below carry the information"
        ),
        "weight_noise": {
            "n_variants": len(variants),
            "kendall_tau_b_mean": round(sum(taus) / len(taus), 6) if taus else None,
            "kendall_tau_b_worst": round(min(taus), 6) if taus else None,
            "top_k_survival_mean": round(sum(survival) / len(survival), 6) if survival else None,
            "top_k_survival_worst": round(min(survival), 6) if survival else None,
            "heldout_in_topk_rate": (
                round(sum(heldout_surface) / len(heldout_surface), 6) if heldout_surface else None
            ),
        },
        "measures": "internal consistency only; not external correctness (design §2.1)",
    }


# --------------------------------------------------------------------------- reports


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def default_report_path(reports_dir: str | Path, stream: str, timestamp: str | None = None) -> Path:
    stamp = timestamp or utc_timestamp()
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"eval-{stream}-{stamp}.json"


def build_product_report(
    stream: str,
    cases: list[dict[str, Any]],
    *,
    truth_header: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the report skeleton; every metric inside ``cases`` is harness-computed."""

    aggregate: dict[str, Any] = {"n_cases": len(cases)}
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_kind": "product",
        "stream": stream,
        "generated_at": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(
        ).replace("+00:00", "Z"),
        "truth": truth_header or {},
        "cases": cases,
        "aggregate": aggregate,
        "ok": True,
        **(extra or {}),
    }


def truth_summary(truth: NirfTruthFile | NaacTruthFile) -> dict[str, Any]:
    header = truth.header
    return {
        "truth_kind": header.truth_kind,
        "truth_version": header.truth_version,
        "scope": header.scope,
        "as_of": truth.as_of,
        "source_url": header.source_url,
        "source_sha256": truth.source_sha256,
        "fetched_at": truth.fetched_at,
        "entries": len(truth.entries),
    }


def load_truth_for_stream(
    stream: str, truth_dir: str | Path, *, root: Path | None = None
) -> NirfTruthFile | NaacTruthFile:
    """Resolve the committed truth file for a proxy stream (contract §1)."""

    base = Path(truth_dir)
    if stream == "nirf-proxy":
        path = base / "nirf" / "nirf_2025_medical.yaml"
    elif stream == "naac-proxy":
        path = base / "naac" / "naac_grades.yaml"
    else:
        raise ShortlistError(f"stream {stream!r} does not use a truth table")
    try:
        return load_truth_file(path, root=root)
    except TruthError as exc:
        raise ShortlistError(f"truth table for {stream} failed to load: {exc}") from exc


def render_product_summary(report: dict[str, Any]) -> str:
    """One-screen summary for the CLI (mirrors run.render_summary style)."""

    stream = report.get("stream")
    lines = [f"fmd eval ({stream}): {report.get('aggregate', {}).get('n_cases', 0)} case(s)"]
    truth = report.get("truth") or {}
    if truth:
        lines.append(
            "  truth: {kind} v{version} as_of {as_of} entries {entries} sha256_ok".format(
                kind=truth.get("truth_kind"),
                version=truth.get("truth_version"),
                as_of=truth.get("as_of"),
                entries=truth.get("entries"),
            )
        )
    for case in report.get("cases") or []:
        if stream in {"nirf-proxy", "naac-proxy"}:
            lines.append(
                "  {profile}: top3 {t3} top5 {t5} best_available {ba} "
                "spearman {sr} kendall {kt} unknown {uf}".format(
                    profile=case.get("profile_id"),
                    t3=case.get("top_k_overlap", {}).get("3"),
                    t5=case.get("top_k_overlap", {}).get("5"),
                    ba=case.get("best_available", {}).get("hit"),
                    sr=case.get("rank_correlation", {}).get("spearman_rho"),
                    kt=case.get("rank_correlation", {}).get("kendall_tau_b"),
                    uf=case.get("unknown", {}).get("fraction"),
                )
            )
        elif stream == "self-drift":
            lines.append(
                "  {profile}: changes {n} rank_max {mx} churn {churn} "
                "evidence_driven {ev} unexplained {un}".format(
                    profile=case.get("profile_id"),
                    n=case.get("n_changes"),
                    mx=case.get("rank_changes", {}).get("max_abs"),
                    churn=case.get("top10_churn"),
                    ev=case.get("evidence_driven_fraction"),
                    un=case.get("unexplained_count"),
                )
            )
        elif stream == "cv-stability":
            noise = case.get("weight_noise") or {}
            lines.append(
                "  {profile}: tau_mean {tm} tau_worst {tw} topk_survival {sv} "
                "heldout_in_topk {ho}".format(
                    profile=case.get("profile_id"),
                    tm=noise.get("kendall_tau_b_mean"),
                    tw=noise.get("kendall_tau_b_worst"),
                    sv=noise.get("top_k_survival_mean"),
                    ho=noise.get("heldout_in_topk_rate"),
                )
            )
    lines.append(f"  proxy_label: {report.get('proxy_label')}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_K_VALUES",
    "PROXY_LABEL_NAAC",
    "PROXY_LABEL_NIRF",
    "Shortlist",
    "ShortlistError",
    "ShortlistRow",
    "build_product_report",
    "default_report_path",
    "load_shortlist",
    "load_truth_for_stream",
    "render_product_summary",
    "run_cv_stability",
    "run_naac_proxy",
    "run_nirf_proxy",
    "run_proxy_stream",
    "run_self_drift",
    "truth_summary",
    "utc_timestamp",
    "write_report",
]
