"""Insight selection: greedy max-coverage with mandatory inclusions (brief §11).

Insights are chosen to cover *predicates/aspects*, not to please. Hard-filter
failures, gaming flags and recognition status are always included, separately
from the greedy budget, and the coverage report is returned so the selection is
auditable.

Semantic diversity (optional, embeddings-gated): when a caller supplies an
embedder, a candidate whose text is semantically near-identical (cosine ≥
``semantic_diversity_min_similarity``) to an already-included insight's text is
suppressed as a ``semantic_duplicate`` — two predicates phrasing the same fact
must not both reach the shortlist. When embeddings are absent the gate is
inactive and the documented lexical behaviour (predicate-string diversity
only) applies; the ``embedding_mode`` field on the selection always records
which path ran (``embeddings`` vs ``embeddings_unavailable`` + reason), so the
degradation is visible in ``insight_sets.coverage_json``, never silent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Insight:
    id: str
    predicate: str
    text: str
    coverage: frozenset[str] = frozenset()
    claim_ids: tuple[int | str, ...] = ()
    kind: str = "evidence"
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "predicate": self.predicate,
            "text": self.text,
            "coverage": sorted(self.coverage),
            "claim_ids": list(self.claim_ids),
            "kind": self.kind,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class InsightSelection:
    selected: tuple[Insight, ...]
    mandatory: tuple[Insight, ...]
    coverage: float
    covered_units: tuple[str, ...]
    uncovered_units: tuple[str, ...]
    stopped_reason: str
    # Surfaced embedding mode (W1 embed-provider): "embeddings" when a supplied
    # embedder actually gated the selection, "embeddings_unavailable" (+ note)
    # when the semantic gate ran lexically because no backend was resolvable.
    embedding_mode: str = "embeddings_unavailable"
    embedding_note: str | None = None
    semantic_duplicates: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": [i.as_dict() for i in self.selected],
            "mandatory": [i.as_dict() for i in self.mandatory],
            "coverage": round(self.coverage, 4),
            "covered_units": list(self.covered_units),
            "uncovered_units": list(self.uncovered_units),
            "stopped_reason": self.stopped_reason,
            "embedding_mode": self.embedding_mode,
            "embedding_note": self.embedding_note,
            "semantic_duplicates": list(self.semantic_duplicates),
        }


class DiversityEmbedder(Protocol):
    """Duck-typed pairwise embedder (``fmd.er.similarity.EmbeddingScorer`` fits).

    ``similarity`` may return ``None`` (backend failed for that pair): the gate
    then fails open for that pair and the recorded note carries the reason.
    """

    available: bool

    def similarity(self, left: str, right: str) -> float | None: ...


def from_filter_failures(decisions: Iterable[Any]) -> tuple[Insight, ...]:
    insights: list[Insight] = []
    for decision in decisions:
        status = getattr(decision, "result", None) or (
            decision.get("result") if isinstance(decision, Mapping) else None
        )
        if status == "pass":
            continue
        filter_id = getattr(decision, "filter_id", None) or (
            decision.get("filter_id") if isinstance(decision, Mapping) else None
        )
        reason = getattr(decision, "reason", None) or (
            decision.get("reason") if isinstance(decision, Mapping) else None
        )
        claim_id = getattr(decision, "claim_id", None) or (
            decision.get("claim_id") if isinstance(decision, Mapping) else None
        )
        if filter_id is None:
            continue
        insights.append(
            Insight(
                id=f"filter:{filter_id}",
                predicate=filter_id,
                text=f"{filter_id}: {status} — {reason or 'no deciding claim'}",
                coverage=frozenset({f"filter:{filter_id}"}),
                claim_ids=(claim_id,) if claim_id is not None else (),
                kind="filter_failure" if status == "fail" else "filter_unknown",
                detail={"status": status},
            )
        )
    return tuple(insights)


def from_gaming_flags(flags: Iterable[Any]) -> tuple[Insight, ...]:
    insights: list[Insight] = []
    for flag in flags:
        flag_id = getattr(flag, "flag_id", None) or (
            flag.get("flag_id") if isinstance(flag, Mapping) else None
        )
        if flag_id is None:
            continue
        claim_id = getattr(flag, "claim_id", None) or (
            flag.get("claim_id") if isinstance(flag, Mapping) else None
        )
        evidence = getattr(flag, "evidence", None) or (
            flag.get("evidence") if isinstance(flag, Mapping) else None
        )
        predicate = getattr(flag, "predicate", None) or (
            flag.get("predicate") if isinstance(flag, Mapping) else None
        )
        insights.append(
            Insight(
                id=f"flag:{flag_id}:{claim_id}",
                predicate=str(predicate or flag_id),
                text=f"{flag_id}: {evidence or ''}".strip(),
                coverage=frozenset({f"flag:{flag_id}"}),
                claim_ids=(claim_id,) if claim_id is not None else (),
                kind="gaming_flag",
            )
        )
    return tuple(insights)


def recognition_insight(claim: Any | None, status: str | None = None) -> Insight:
    claim_id = getattr(claim, "claim_id", None) if claim is not None else None
    value = getattr(claim, "value", None) if claim is not None else None
    resolved = status or value or "unknown"
    return Insight(
        id="recognition_status",
        predicate="recognition_status",
        text=f"Recognition status: {resolved}",
        coverage=frozenset({"recognition_status"}),
        claim_ids=(claim_id,) if claim_id is not None else (),
        kind="recognition_status",
        detail={"status": resolved},
    )


def select(
    insights: Sequence[Insight],
    *,
    mandatory: Sequence[Insight] = (),
    max_insights: int | None = None,
    min_marginal_gain: float = 0.0,
    require_predicate_diversity: bool = True,
    embedder: DiversityEmbedder | None = None,
    semantic_diversity_min_similarity: float = 0.95,
) -> InsightSelection:
    """Greedy max-coverage over information units.

    The greedy budget applies to optional insights; mandatory insights are
    always included even when they exceed ``max_insights``.

    ``embedder`` (optional) enables the semantic-duplicate gate described in the
    module docstring. An embedder that is absent, unavailable or erroring keeps
    the lexical behaviour and records ``embedding_mode="embeddings_unavailable"``
    with the reason — the fallback is visible in the persisted selection.
    """

    if max_insights is not None and max_insights < 0:
        raise ValueError("max_insights must be >= 0")

    semantic_gate = embedder is not None and bool(getattr(embedder, "available", False))
    if embedder is None:
        embedding_mode, embedding_note = (
            "embeddings_unavailable",
            (
                "no embedding backend supplied; semantic diversity gate inactive "
                "(lexical predicate diversity only)"
            ),
        )
    elif not semantic_gate:
        reason = getattr(embedder, "reason", None)
        embedding_mode, embedding_note = (
            "embeddings_unavailable",
            (f"embedder unavailable: {reason}" if reason else "embedder unavailable"),
        )
    else:
        embedding_mode, embedding_note = "embeddings", None

    seen_ids = {item.id for item in mandatory}
    mandatory_tuple = tuple(item for item in mandatory)

    universe: set[str] = set()
    for item in (*mandatory_tuple, *insights):
        universe.update(item.coverage)
    if not universe:
        return InsightSelection(
            selected=(),
            mandatory=mandatory_tuple,
            coverage=1.0,
            covered_units=(),
            uncovered_units=(),
            stopped_reason="no_information_units",
            embedding_mode=embedding_mode,
            embedding_note=embedding_note,
        )

    covered: set[str] = set()
    for item in mandatory_tuple:
        covered.update(item.coverage)

    candidates = [item for item in insights if item.id not in seen_ids]
    selected: list[Insight] = []
    covered_predicates = {item.predicate for item in mandatory_tuple}
    covered_texts = [item.text for item in mandatory_tuple]
    semantic_duplicates: list[str] = []
    budget = max_insights if max_insights is not None else len(candidates)
    stopped_reason = "max_insights_reached" if max_insights is not None else "no_candidates_left"

    while len(selected) < budget and candidates:
        if semantic_gate:
            surviving: list[Insight] = []
            for candidate in candidates:
                if _semantic_duplicate(
                    candidate,
                    covered_texts,
                    embedder,
                    semantic_diversity_min_similarity,
                ):
                    semantic_duplicates.append(candidate.id)
                else:
                    surviving.append(candidate)
            candidates = surviving
            if not candidates:
                stopped_reason = "semantic_duplicates_suppressed"
                break
        best: Insight | None = None
        best_gain = -1.0
        for candidate in candidates:
            new_units = candidate.coverage - covered
            gain = len(new_units) / len(universe)
            is_diverse = candidate.predicate not in covered_predicates
            if not is_diverse and require_predicate_diversity:
                continue
            if gain > best_gain + 1e-12 or (
                abs(gain - best_gain) <= 1e-12 and (best is None or candidate.id < best.id)
            ):
                best, best_gain = candidate, gain

        if best is None:
            stopped_reason = "predicate_diversity_exhausted"
            break
        if best_gain < min_marginal_gain or best_gain <= 0.0:
            stopped_reason = "min_marginal_gain_reached"
            break

        selected.append(best)
        covered.update(best.coverage)
        covered_predicates.add(best.predicate)
        covered_texts.append(best.text)
        candidates = [item for item in candidates if item.id != best.id]
        if len(selected) >= budget:
            stopped_reason = "max_insights_reached"

    if (
        not candidates
        and len(selected) < budget
        and stopped_reason != "semantic_duplicates_suppressed"
    ):
        stopped_reason = "no_candidates_left"

    return InsightSelection(
        selected=tuple(selected),
        mandatory=mandatory_tuple,
        coverage=len(covered) / len(universe),
        covered_units=tuple(sorted(covered)),
        uncovered_units=tuple(sorted(universe - covered)),
        stopped_reason=stopped_reason,
        embedding_mode=embedding_mode,
        embedding_note=embedding_note,
        semantic_duplicates=tuple(semantic_duplicates),
    )


def _semantic_duplicate(
    candidate: Insight,
    covered_texts: Sequence[str],
    embedder: DiversityEmbedder,
    threshold: float,
) -> bool:
    """True when ``candidate.text`` is near-identical to an included text.

    A pair the embedder cannot score (returns ``None``) fails open: the
    candidate stays eligible and the embedder's own note records why.
    """
    for text in covered_texts:
        score = embedder.similarity(candidate.text, text)
        if score is not None and score >= threshold:
            return True
    return False
