"""Score orchestration: reuse the existing ``fmd score`` implementation.

The scoring maths lives in :mod:`fmd.score`. This module is the loop that was
previously inline in the CLI: select programmes, load their claims, evaluate the
hard filters, score, and persist only when the evidence changed. It is factored
out so ``fmd score`` and the worker run the *same* code, and so an offline test
can inject fake repositories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fmd.db.repositories import (
    ClaimsRepo,
    ProgrammeFilter,
    ProgrammesRepo,
    ScoresRepo,
)
from fmd.filters import evaluate_all
from fmd.score import score_programme
from fmd.score.config import ScoringConfig
from fmd.score.persist import persist_score


class ScoreError(RuntimeError):
    pass


@dataclass
class ScoreOutcome:
    written: int = 0
    skipped: int = 0
    unknown: int = 0
    programmes: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "skipped": self.skipped,
            "unknown": self.unknown,
            "programmes": self.programmes,
            "failures": self.failures,
        }


def score_programmes(
    *,
    registry: Any,
    profile: dict[str, Any] | None = None,
    programmes: ProgrammesRepo | Any,
    claims: ClaimsRepo | Any,
    scores: ScoresRepo | Any,
    limit: int = 500,
    dry_run: bool = False,
    force: bool = False,
    persist: Any = None,
) -> ScoreOutcome:
    config = ScoringConfig.from_config(registry.weights)
    problems = config.validate()
    if problems:
        raise ScoreError("invalid scoring config: " + "; ".join(problems))

    persist_fn = persist or persist_score
    database = getattr(scores, "db", None)
    if not dry_run and persist is None and database is None:
        raise ScoreError("score persistence requires a database-backed ScoresRepo")

    profile = profile or {}
    outcome = ScoreOutcome()
    rows = programmes.list(ProgrammeFilter(limit=max(1, limit), offset=0))
    for programme in rows:
        try:
            entity_claims = claims.list_for_entity(programme.institution_id)
            view = programme.model_dump(mode="json")
            view["claims"] = entity_claims
            summary = evaluate_all(view, profile)
            result = score_programme(
                programme.id,
                view,
                profile,
                entity_claims,
                config=config,
                cost_rules=registry.cost_rules,
                filter_decisions=list(summary.decisions.values()),
            )
        except Exception as exc:  # one bad programme must not abort the run
            outcome.failures.append(
                {
                    "programme_id": getattr(programme, "id", None),
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            continue
        outcome.programmes += 1
        if result.composite.point is None:
            outcome.unknown += 1
        if dry_run:
            point = result.composite.point
            rendered = "UNKNOWN" if point is None else f"{point:.2f}"
            outcome.lines.append(
                f"[dry-run] programme={programme.id} composite={rendered} "
                f"tier={result.evidence_tier} claims={len(entity_claims)}"
            )
            continue
        latest = scores.latest_for_entity(programme.institution_id)
        if (
            not force
            and latest is not None
            and latest.claims_hash == result.claims_hash
            and latest.scorer_version == result.scorer_version
            and latest.weights_version == result.weights_version
        ):
            outcome.skipped += 1
            continue
        persist_fn(database, result)
        outcome.written += 1

    return outcome
