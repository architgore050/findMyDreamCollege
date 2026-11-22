"""Wave v1-w7: scoring coverage diagnostics.

Offline for the pure report (fake SubScore/ComponentScore objects shaped
exactly like the scorer's); DB-gated end-to-end through the CLI handler with
real repos in an isolated schema. The report must never write and never
change any weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmd.score.coverage import (
    SUB_SCORE_IDS,
    build_coverage_report,
    programme_coverage,
    render_coverage,
)
from fmd.score.sub_scores import ComponentScore, SubScore, unknown_sub_score


def _component(name: str, claim_id: int | None, score: float = 70.0) -> ComponentScore:
    return ComponentScore(
        name=name,
        label=name.replace("_", " "),
        value=100.0,
        score=score,
        lo=60.0,
        hi=80.0,
        method="measured",
        claim_id=claim_id,
        source_id=42 if claim_id is not None else None,
    )


def _sub(point: float | None, *, reason: str | None = None, components=()) -> SubScore:
    return SubScore(
        point=point,
        lo=0.0 if point is None else point - 5,
        hi=100.0 if point is None else point + 5,
        n_claims=len(components),
        n_sources=1 if components else 0,
        evidence_tier="measured" if point is not None else "none",
        components=components,
        unknown_reason=reason,
    )


def test_programme_coverage_counts_populated_and_lists_feeding_claims() -> None:
    subs = {
        "outcomes": _sub(
            80.0, components=(_component("placement_rate", 11), _component("median_salary", 12))
        ),
        "affordability_roi": _sub(None, reason="no_fee_claim"),
        "admission_realism": _sub(55.0, components=(_component("cutoff_percentile", 13),)),
        "clinical_training": unknown_sub_score("no_claims"),
    }
    row = programme_coverage(7, 3, "Demo Institute", subs)
    assert row["populated"] == 2 and row["total"] == len(SUB_SCORE_IDS) == 7
    assert row["feeding_claim_ids"] == ["11", "12", "13"]
    assert row["sub_scores"]["outcomes"]["status"] == "populated"
    assert row["sub_scores"]["affordability_roi"] == {
        "status": "unknown",
        "reason": "no_fee_claim",
        "n_claims": 0,
        "components": [],
    }
    # an unknown_sub_score without components and without a reason
    assert row["sub_scores"]["clinical_training"]["reason"] == "no_claims"


def test_coverage_report_aggregates_histogram_and_missing_ranking() -> None:
    subs_a = {
        "outcomes": _sub(80.0, components=(_component("placement_rate", 11),)),
        "admission_realism": _sub(55.0),
    }
    subs_b = {sub: unknown_sub_score("no_claims") for sub in SUB_SCORE_IDS}
    rows = [
        programme_coverage(1, 1, "A", subs_a),
        programme_coverage(2, 2, "B", subs_b),
    ]
    report = build_coverage_report(rows)
    assert report["programmes"] == 2
    assert report["populated_histogram"]["2/7"] == 1
    assert report["populated_histogram"]["0/7"] == 1
    # the 0/7 programme is missing all seven; A is missing five
    assert report["missing_by_sub_score"]["outcomes"] == 1
    assert report["missing_by_sub_score"]["clinical_training"] == 2
    text = render_coverage(report)
    assert text.startswith("programmes=2")
    assert "missing most:" in text


def test_coverage_report_is_read_only_by_construction() -> None:
    """No persistence seam exists: the module imports no repo and takes no db."""
    import inspect

    from fmd.score import coverage

    source = inspect.getsource(coverage)
    for forbidden in ("persist", "INSERT", "UPDATE", "set_plan", "commit"):
        assert forbidden not in source, forbidden


# ------------------------------------------------------------------ DB-gated


def test_coverage_report_cli_end_to_end(pg_database, tmp_path: Path) -> None:
    """Real repos through the CLI handler: a seeded in-scope programme with no
    claims reports 0/7 with the scorer's unknown reasons — read-only."""
    import uuid

    from fmd.db.repositories import (
        EntitiesRepo,
        EntityCreate,
        ProgrammeCreate,
        ProgrammesRepo,
    )
    from fmd.score.coverage import SUB_SCORE_IDS

    entities = EntitiesRepo(pg_database)
    entity = entities.create(
        EntityCreate(
            kind="institution",
            canonical_name=f"Coverage Institute {uuid.uuid4().hex[:8]}",
            country="IN",
            state="MH",
        )
    )
    programmes = ProgrammesRepo(pg_database)
    programme = programmes.create(
        ProgrammeCreate(
            institution_id=entity.id,
            name="MBBS",
            credential_level="degree",
        )
    )

    from fmd.cli import _cmd_coverage_report
    from tests.conftest import CONFIG_DIR

    out = tmp_path / "coverage.json"
    args = argparse.Namespace(
        command="coverage-report",
        database_url=None,  # unused: the fixture connection is injected
        config_dir=CONFIG_DIR,
        profile=None,
        limit=50,
        json_out=out,
    )
    # the fixture's schema-scoped connection, so the seeded rows are visible
    assert _cmd_coverage_report(args, database=pg_database) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["programmes"] >= 1
    row = next(r for r in report["rows"] if r["programme_id"] == programme.id)
    assert row["populated"] == 0 and row["total"] == len(SUB_SCORE_IDS)
    assert set(row["sub_scores"]) == set(SUB_SCORE_IDS)
    # every unknown sub-score carries the scorer's stable reason, never blank
    assert all(info["reason"] for info in row["sub_scores"].values())
