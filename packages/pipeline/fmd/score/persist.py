"""Persist a :class:`fmd.score.ScoreResult` plus its insight set.

``scores`` are append-only; a rerun with identical
``(claims_hash, scorer_version, weights_version)`` should be skipped by the
caller (see ``fmd.cli score``). Insight sets are written alongside the score row
so the selection and its coverage report stay auditable.
"""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from fmd.db.connection import Database
from fmd.db.repositories import ScoreCreate, ScoreRecord, ScoresRepo
from fmd.score import ScoreResult


def persist_score(
    database: Database,
    result: ScoreResult,
    *,
    profile_hash: str | None = None,
) -> ScoreRecord:
    """Insert the score and its insight set in separate statements.

    ``profile_hash`` is accepted only so callers can record provenance in
    process-local logs; the profile itself is never written to any table.
    """

    record = ScoresRepo(database).create(ScoreCreate(**result.to_score_create()))
    selection = result.insights
    if selection is None:
        return record

    payload: dict[str, Any] = selection.as_dict()
    insight_ids = [
        item["id"] for item in [*payload["mandatory"], *payload["selected"]]
    ]
    with database.connection() as conn:
        conn.execute(
            """
            INSERT INTO insight_sets
                (entity_id, scores_id, insight_ids_json, coverage_json)
            VALUES (%s, %s, %s, %s)
            """,
            (
                record.entity_id,
                record.id,
                Jsonb(insight_ids),
                Jsonb(payload),
            ),
        )
    return record
