"""Wave v1-w7: scoring coverage diagnostics.

The operator's question — "why is this programme 1/7?" — answered from the
scored artefacts themselves, with no weight changes and no writes. For every
in-scope programme the report lists:

* which of the seven sub-scores carry a point estimate, and which are
  ``unknown`` (with the stable ``unknown_reason`` the scorer recorded);
* the components inside each sub-score, each with its ``method``, value and
  the **claim id** that fed it — the click-through path from a number to the
  evidence that produced it;
* per-programme populated/total counts, an aggregate histogram (how many
  programmes are 1/7, 2/7, …), and the per-sub-score missing ranking — where
  acquisition effort would move coverage most.

Pure and read-only: rows in, report out. The claim ids are the same ids the
provenance API serves, so nothing here re-derives evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

#: The seven configured sub-scores (registry-driven in the scorer; mirrored
#: here only to keep the report's "n/7" honest if one is added).
SUB_SCORE_IDS = (
    "outcomes",
    "clinical_training",
    "academic_quality",
    "affordability_roi",
    "financing_feasibility",
    "student_life_safety",
    "admission_realism",
)


def programme_coverage(
    programme_id: Any,
    entity_id: Any,
    name: str,
    sub_scores: Mapping[str, Any],
) -> dict[str, Any]:
    """One programme's coverage row from its ``ScoreResult.sub_scores``."""
    components_out: dict[str, Any] = {}
    feeding_claims: list[int | str] = []
    populated = 0
    for sub_id in SUB_SCORE_IDS:
        sub = sub_scores.get(sub_id)
        if sub is None:
            components_out[sub_id] = {"status": "absent", "reason": "no_sub_score"}
            continue
        component_rows = []
        for component in getattr(sub, "components", ()) or ():
            info = component.as_dict()
            component_rows.append(info)
            if info.get("claim_id") is not None:
                feeding_claims.append(info["claim_id"])
        unknown_reason = getattr(sub, "unknown_reason", None)
        if getattr(sub, "point", None) is not None:
            populated += 1
            components_out[sub_id] = {
                "status": "populated",
                "point": round(float(sub.point), 2),
                "n_claims": int(getattr(sub, "n_claims", 0) or 0),
                "n_sources": int(getattr(sub, "n_sources", 0) or 0),
                "components": component_rows,
            }
        else:
            components_out[sub_id] = {
                "status": "unknown",
                "reason": unknown_reason or "no_component_scored",
                "n_claims": int(getattr(sub, "n_claims", 0) or 0),
                "components": component_rows,
            }
    return {
        "programme_id": programme_id,
        "entity_id": entity_id,
        "name": name,
        "populated": populated,
        "total": len(SUB_SCORE_IDS),
        "feeding_claim_ids": sorted({str(claim) for claim in feeding_claims}),
        "sub_scores": components_out,
    }


def build_coverage_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-programme rows into the operator's coverage report."""
    histogram: Counter[int] = Counter()
    missing_by_sub: Counter[str] = Counter()
    programmes: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        programmes.append(entry)
        histogram[int(entry["populated"])] += 1
        for sub_id, info in entry["sub_scores"].items():
            if info["status"] != "populated":
                missing_by_sub[sub_id] += 1
    return {
        "programmes": len(programmes),
        "populated_histogram": {
            f"{count}/{len(SUB_SCORE_IDS)}": histogram[count] for count in sorted(histogram)
        },
        "missing_by_sub_score": dict(missing_by_sub.most_common()),
        "rows": programmes[:500],
    }


def render_coverage(report: Mapping[str, Any]) -> str:
    """Human lines: histogram, missing ranking, then per-programme one-liners."""
    lines = [f"programmes={report['programmes']}"]
    for label, count in report["populated_histogram"].items():
        lines.append(f"  {label}: {count} programme(s)")
    missing = report["missing_by_sub_score"]
    if missing:
        lines.append("missing most: " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    for row in report["rows"][:20]:
        lines.append(
            f"  programme={row['programme_id']} {row['populated']}/{row['total']}"
            f" claims={len(row['feeding_claim_ids'])}"
        )
    return "\n".join(lines)
