"""Wave v1-w5: why documents are ``missing_family``, and what would fix it.

Read-only diagnosis of the classification backlog. The deterministic chain is
replayed per evidence document — the exact order the planner applies
(registry-declared > phase-B text > phase-A URL tokens > filename shape) with
the configured floors — and every outcome is bucketed:

* ``declared`` / ``family_set`` — nothing to fix; counted per family source.
* ``would_classify_with_text`` — the rules pass on the fetched sample: the
  document is only waiting for a phase-B planning pass.
* ``would_classify_from_url`` — URL token evidence alone clears the phase-A
  floor on replay (stale plan row).
* ``below_floor`` — evidence exists but does not reach the floor: the best
  score and the floor are both reported so the operator can judge the gap.
* ``ambiguous`` — two families tie: refused by design, never a guess.
* ``no_signal`` — the rules are silent on every channel.

``below_floor`` / ``ambiguous`` / ``no_signal`` rows are the semantic-tier
candidates (LOCAL-QUEUE LQ-7): the model may propose, but admission still
requires verbatim quoted evidence (``fmd.plan.semantic``) — the report only
names the queue, it never admits anything.

No function here writes: reports are derived from rows the caller supplies.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from fmd.plan.roles import family_hints_from_structure
from fmd.plan.rules import DEFAULT_RULES, PlanRules, detect_family, score_families

#: Buckets that make a document a candidate for the (queued, local) semantic
#: classification pass. The deterministic rules had their chance and abstained.
SEMANTIC_BUCKETS = ("below_floor", "ambiguous", "no_signal")


def replay_family(
    *,
    url: str,
    sample: str | None,
    rules: PlanRules = DEFAULT_RULES,
    phase_b_min_score: float,
    phase_b_margin: float,
    phase_a_min_score: float,
    phase_a_margin: float,
    recurring_min: int | None = None,
    table_min_rows: int | None = None,
) -> tuple[str | None, float, str | None, dict[str, Any]]:
    """Replay the planner's phase-B family chain for one document.

    Mirrors ``DocumentPlanner._family`` for a non-declared document: text
    first (the URL was phase A's evidence and must not outrank text), then
    URL tokens as the weaker fallback, then a single filename hint. Returns
    ``(family, best_score, reason, signals)`` with ``best_score`` meaning the
    raw top score even when detection abstains — the acceptance floor and the
    acquisition floor are different questions.
    """
    signals: dict[str, Any] = {}
    hints = family_hints_from_structure(url=url, content_type=None)
    signals["structure_hints"] = list(hints)
    best_score = 0.0
    reason: str | None = None
    if sample:
        family, score, detected, reason = detect_family(
            url="",
            anchor_text=None,
            text_sample=sample,
            rules=rules,
            min_score=phase_b_min_score,
            margin=phase_b_margin,
            recurring_min=recurring_min,
            table_min_rows=table_min_rows,
        )
        signals.update(detected)
        best_score = max(best_score, score)
        if family is not None:
            return family, score, None, signals
        url_family, url_score, url_signals, url_reason = detect_family(
            url=url,
            anchor_text=None,
            rules=rules,
            min_score=phase_a_min_score,
            margin=phase_a_margin,
        )
        signals.update({f"phase_a_{k}": v for k, v in url_signals.items()})
        best_score = max(best_score, url_score)
        if url_family is not None:
            signals["fallback"] = "phase_a_url"
            return url_family, url_score, None, signals
        if len(hints) == 1 and (url_reason or "family_undetected") == "family_undetected":
            signals["fallback"] = "structure_hint"
            return hints[0], float(rules.weights.structure), None, signals
        reason = reason or url_reason
    else:
        # No sample yet: URL evidence only, against the phase-A floor.
        family, score, detected, reason = detect_family(
            url=url,
            anchor_text=None,
            rules=rules,
            min_score=phase_a_min_score,
            margin=phase_a_margin,
        )
        signals.update(detected)
        best_score = score
        if family is not None:
            return family, score, None, signals
        if len(hints) == 1:
            signals["structure_hint"] = hints[0]
            best = score_families(url=url, rules=rules)
            best_score = max(best_score, best[0].score if best else 0.0)
        elif len(hints) > 1:
            signals["structure_hints_ambiguous"] = 1
    ranked = score_families(url=url, text_sample=sample, rules=rules)
    if ranked:
        best_score = max(best_score, ranked[0].score)
    return None, round(best_score, 6), reason, signals


def build_report(
    rows: Iterable[Mapping[str, Any]],
    rules: PlanRules = DEFAULT_RULES,
    *,
    phase_b_min_score: float,
    phase_b_margin: float,
    phase_a_min_score: float,
    phase_a_margin: float,
    recurring_min: int | None = None,
    table_min_rows: int | None = None,
    sample_for: Callable[[Mapping[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Bucket every evidence-role document; family-set rows are counted only.

    ``rows`` are mappings with ``url``, ``role``, ``extraction_family``,
    ``plan_state`` and (optional) ``declared_family``; ``sample_for`` fetches
    the bounded text sample when the caller wants the with-text replay
    (``None`` ⇒ URL-only replay). Deterministic; writes nothing.
    """
    buckets: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    candidates: list[dict[str, Any]] = []
    declared = 0
    family_set_by_source: Counter[str] = Counter()
    total = 0
    for row in rows:
        if str(row.get("role") or "evidence") != "evidence":
            continue  # inventory/navigation were never candidates
        total += 1
        url = str(row.get("url") or "")
        family = row.get("extraction_family") or row.get("declared_family")
        if family:
            source = str(row.get("family_source") or "registry")
            if source == "registry":
                declared += 1
            family_set_by_source[source] += 1
            by_family[str(family)]["family_set"] += 1
            continue
        sample = sample_for(row) if sample_for is not None else None
        replayed, best_score, reason, signals = replay_family(
            url=url,
            sample=sample,
            rules=rules,
            phase_b_min_score=phase_b_min_score,
            phase_b_margin=phase_b_margin,
            phase_a_min_score=phase_a_min_score,
            phase_a_margin=phase_a_margin,
            recurring_min=recurring_min,
            table_min_rows=table_min_rows,
        )
        if replayed is not None:
            bucket = "would_classify_with_text" if sample else "would_classify_from_url"
        elif (reason or "") == "ambiguous_family":
            bucket = "ambiguous"
        elif best_score > 0.0:
            bucket = "below_floor"
        else:
            bucket = "no_signal"
        buckets[bucket] += 1
        entry = {
            "url": url,
            "bucket": bucket,
            "best_score": best_score,
            "reason": reason,
            "plan_state": row.get("plan_state"),
        }
        if bucket in SEMANTIC_BUCKETS:
            entry["signals"] = {key: value for key, value in signals.items() if value}
            candidates.append(entry)
        if replayed is not None:
            by_family[replayed][bucket] += 1
    return {
        "evidence_documents": total,
        "declared": declared,
        "family_set_by_source": dict(family_set_by_source),
        "missing_family": dict(buckets),
        "semantic_candidates": len(candidates),
        "candidate_rows": candidates[:500],
        "families": {family: dict(counter) for family, counter in sorted(by_family.items())},
    }


def render_report(report: Mapping[str, Any]) -> str:
    """Human lines: buckets first, then the semantic-queue head."""
    lines = [
        "evidence_documents={evidence_documents} declared={declared}".format(**report),
        "family_set_by_source=" + str(report["family_set_by_source"]),
        "missing_family=" + str(report["missing_family"]),
        f"semantic_candidates={report['semantic_candidates']}",
    ]
    for entry in report["candidate_rows"][:10]:
        lines.append(
            "  candidate url={url} bucket={bucket} best_score={best_score}".format(**entry)
        )
    return "\n".join(lines)
