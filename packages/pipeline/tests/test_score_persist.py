from __future__ import annotations

import json
import uuid
from pathlib import Path

from fmd.db.repositories import EntitiesRepo, EntityCreate, ScoresRepo
from fmd.score import score_programme
from fmd.score.persist import persist_score

FIXTURE = Path(__file__).parent / "fixtures" / "score" / "sample_claims.json"


def test_persist_score_writes_insight_set(pg_database) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    entity = EntitiesRepo(pg_database).create(
        EntityCreate(
            kind="institution",
            canonical_name=f"Persist Institute {uuid.uuid4().hex[:10]}",
            country="IN",
            state="MH",
        )
    )
    result = score_programme(
        entity.id,
        data["programme"],
        data["profile"],
        data["claims"],
    )
    record = persist_score(pg_database, result)
    assert record.id > 0
    assert record.entity_id == entity.id

    latest = ScoresRepo(pg_database).latest_for_entity(entity.id)
    assert latest is not None
    assert latest.claims_hash == result.claims_hash
    assert latest.weights_version == result.weights_version
    assert latest.scorer_version == result.scorer_version

    with pg_database.connection() as conn:
        row = conn.execute(
            "SELECT insight_ids_json, coverage_json FROM insight_sets WHERE scores_id = %s",
            (record.id,),
        ).fetchone()
    assert row is not None
    assert row["insight_ids_json"]
    assert "coverage" in row["coverage_json"]
