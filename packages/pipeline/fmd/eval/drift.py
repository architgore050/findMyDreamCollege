"""Release drift diagnostics (brief §19).

Reports are JSON dictionaries so successive runs can be diffed. Nothing here
uses randomness or a clock.

Definitions
-----------
PSI
    Population Stability Index over equal-width bins:
    ``sum((actual_share - baseline_share) * ln(actual_share / baseline_share))``.
    Guards: < 0.10 stable, < 0.25 moderate, >= 0.25 significant.
KS
    Two-sample Kolmogorov-Smirnov statistic with the asymptotic
    Kolmogorov p-value (no SciPy dependency).
Jaccard churn
    ``1 - |baseline_top_n ∩ current_top_n| / |union|`` over the top-N ids.
Canary
    A fixed golden entry re-scored on every run. If its claims hash is unchanged
    but its metrics moved, the movement cannot be explained by a claims diff and
    is flagged. A changed claims hash is reported as ``explained_by: claims_diff``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25
KS_ALPHA = 0.05
JACCARD_CHURN_ALERT = 0.10


# --------------------------------------------------------------------------- distribution


def _shares(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    counts = [0.0] * (len(edges) - 1)
    for value in values:
        for index in range(len(edges) - 1):
            low, high = edges[index], edges[index + 1]
            last = index == len(edges) - 2
            if (low <= value < high) or (last and value == high):
                counts[index] += 1.0
                break
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [count / total for count in counts]


def _edges(baseline: Sequence[float], current: Sequence[float], bins: int) -> list[float]:
    combined = list(baseline) + list(current)
    low, high = min(combined), max(combined)
    if high == low:
        return [low, high + 1.0]
    step = (high - low) / bins
    return [low + step * index for index in range(bins)] + [high]


def population_stability_index(
    baseline: Iterable[float],
    current: Iterable[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI between two samples. Returns 0.0 for two empty/identical samples."""

    left = [float(value) for value in baseline]
    right = [float(value) for value in current]
    if not left and not right:
        return 0.0
    if not left or not right:
        return float("inf")
    edges = _edges(left, right, bins)
    expected = _shares(left, edges)
    actual = _shares(right, edges)
    total = 0.0
    for expected_share, actual_share in zip(expected, actual, strict=True):
        e = max(expected_share, epsilon)
        a = max(actual_share, epsilon)
        total += (a - e) * math.log(a / e)
    return round(total, 6)


def ks_two_sample(baseline: Iterable[float], current: Iterable[float]) -> tuple[float, float]:
    """Return ``(statistic, asymptotic p-value)`` for two empirical CDFs."""

    left = sorted(float(value) for value in baseline)
    right = sorted(float(value) for value in current)
    if not left or not right:
        return 0.0, 1.0
    n, m = len(left), len(right)
    i = j = 0
    statistic = 0.0
    while i < n and j < m:
        value = min(left[i], right[j])
        while i < n and left[i] <= value:
            i += 1
        while j < m and right[j] <= value:
            j += 1
        statistic = max(statistic, abs(i / n - j / m))
    effective = math.sqrt(n * m / (n + m))
    lam = (effective + 0.12 + 0.11 / effective) * statistic
    if lam <= 0:
        return 0.0, 1.0
    series = 0.0
    for k in range(1, 101):
        term = 2 * (-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam)
        series += term
        if abs(term) < 1e-12:
            break
    p_value = max(0.0, min(1.0, series))
    return round(statistic, 6), round(p_value, 6)


def top_n_jaccard(baseline: Sequence[str], current: Sequence[str], *, n: int = 20) -> float:
    """Jaccard similarity of the two top-N id sets (1.0 = identical)."""

    left = set(baseline[:n])
    right = set(current[:n])
    if not left and not right:
        return 1.0
    return round(len(left & right) / len(left | right), 6)


def churn(baseline: Sequence[str], current: Sequence[str], *, n: int = 20) -> float:
    return round(1.0 - top_n_jaccard(baseline, current, n=n), 6)


# --------------------------------------------------------------------------- reports


def _metric_vector(report: Mapping[str, Any]) -> dict[str, float]:
    """Flatten the numeric eval metrics worth drift-checking."""

    vector: dict[str, float] = {}
    for section in ("extraction", "rejection", "provenance", "interval_coverage"):
        block = report.get(section)
        if not isinstance(block, Mapping):
            continue
        for key, value in block.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                vector[f"{section}.{key}"] = float(value)
    return vector


def _canaries(report: Mapping[str, Any]) -> Mapping[str, Any]:
    block = report.get("canaries")
    return block if isinstance(block, Mapping) else {}


def compare_reports(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    psi_threshold: float = PSI_STABLE,
    ks_alpha: float = KS_ALPHA,
    churn_threshold: float = JACCARD_CHURN_ALERT,
) -> dict[str, Any]:
    """Diff two eval reports and label every movement as explained or not."""

    flags: list[dict[str, Any]] = []
    explained: list[dict[str, Any]] = []
    versions = {
        key: {
            "baseline": (baseline.get("versions") or {}).get(key),
            "current": (current.get("versions") or {}).get(key),
        }
        for key in ("scorer_version", "weights_version")
    }
    version_changed = any(pair["baseline"] != pair["current"] for pair in versions.values())

    left_canaries = _canaries(baseline)
    right_canaries = _canaries(current)
    for entry_id in sorted(set(left_canaries) & set(right_canaries)):
        left = left_canaries[entry_id] or {}
        right = right_canaries[entry_id] or {}
        claims_changed = left.get("claims_hash") != right.get("claims_hash")
        left_metrics = _metric_vector({"extraction": left.get("extraction") or {}})
        right_metrics = _metric_vector({"extraction": right.get("extraction") or {}})
        changed = {
            key: {"baseline": left_metrics.get(key), "current": right_metrics.get(key)}
            for key in sorted(set(left_metrics) | set(right_metrics))
            if left_metrics.get(key) != right_metrics.get(key)
        }
        if not changed:
            continue
        record = {"canary": entry_id, "changes": changed}
        if claims_changed:
            record["explained_by"] = "claims_diff"
            explained.append(record)
        elif version_changed:
            record["explained_by"] = "version_change"
            explained.append(record)
        else:
            record["explained_by"] = None
            flags.append(record)

    psi_block: dict[str, Any] = {}
    left_scores = baseline.get("sub_scores") or {}
    right_scores = current.get("sub_scores") or {}
    if isinstance(left_scores, Mapping) and isinstance(right_scores, Mapping):
        for sub_score in sorted(
            {
                key
                for row in (*left_scores.values(), *right_scores.values())
                if isinstance(row, Mapping)
                for key in row
            }
        ):
            left_values = [
                float(row[sub_score])
                for row in left_scores.values()
                if isinstance(row, Mapping) and isinstance(row.get(sub_score), int | float)
            ]
            right_values = [
                float(row[sub_score])
                for row in right_scores.values()
                if isinstance(row, Mapping) and isinstance(row.get(sub_score), int | float)
            ]
            if not left_values or not right_values:
                continue
            psi = population_stability_index(left_values, right_values)
            statistic, p_value = ks_two_sample(left_values, right_values)
            entry = {"psi": psi, "ks": statistic, "ks_p_value": p_value}
            if psi >= psi_threshold or p_value < ks_alpha:
                entry["explained_by"] = (
                    "version_change" if version_changed else "claims_diff_or_unexplained"
                )
            psi_block[sub_score] = entry

    left_ranking = [str(item) for item in baseline.get("ranking") or []]
    right_ranking = [str(item) for item in current.get("ranking") or []]
    jaccard = top_n_jaccard(left_ranking, right_ranking)
    churn_value = round(1.0 - jaccard, 6)
    ranking = {
        "top_n": 20,
        "jaccard": jaccard,
        "churn": churn_value,
        "alert": churn_value > churn_threshold,
    }

    metric_changes = {
        key: {"baseline": left, "current": right}
        for key, left, right in _metric_diffs(baseline, current)
    }
    return {
        "versions": versions,
        "version_changed": version_changed,
        "metric_changes": metric_changes,
        "canaries": {"unexplained": flags, "explained": explained},
        "sub_scores": psi_block,
        "ranking": ranking,
        "unexplained": bool(flags)
        or any(
            block.get("explained_by") == "claims_diff_or_unexplained"
            for block in psi_block.values()
            if isinstance(block, Mapping)
        ),
    }


def _metric_diffs(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> list[tuple[str, float | None, float | None]]:
    left = _metric_vector(baseline)
    right = _metric_vector(current)
    return [
        (key, left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]


def unexplained_changes(report: Mapping[str, Any]) -> list[str]:
    """Human-readable list of movements no claims diff or version bump explains."""

    reasons: list[str] = []
    canaries = report.get("canaries") or {}
    for item in canaries.get("unexplained") or []:
        reasons.append(f"canary {item.get('canary')}: metrics moved with unchanged claims")
    for name, block in (report.get("sub_scores") or {}).items():
        if isinstance(block, Mapping) and block.get("explained_by") == "claims_diff_or_unexplained":
            reasons.append(f"sub-score {name}: distribution moved without a version change")
    if report.get("unexplained"):
        return reasons or ["drift report marked unexplained"]
    return reasons


__all__ = [
    "JACCARD_CHURN_ALERT",
    "KS_ALPHA",
    "PSI_SIGNIFICANT",
    "PSI_STABLE",
    "churn",
    "compare_reports",
    "ks_two_sample",
    "population_stability_index",
    "top_n_jaccard",
    "unexplained_changes",
]
