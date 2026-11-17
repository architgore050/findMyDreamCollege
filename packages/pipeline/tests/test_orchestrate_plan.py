"""Pipeline order: defer -> acquire sample -> re-plan -> extract (offline).

This is the regression WS-PLAN2 exists for. A document whose URL is ambiguous
must not be thrown away before its text is read, and the text must be bought at
most once per planning run.
"""

from __future__ import annotations

from test_orchestrate import PLAN_FIXTURES, PLAN_LLM_FIXTURES, Harness
from test_orchestrate_support import MemoryDocuments

from fmd.db.repositories import DocumentUpsert, SnapshotCreate
from fmd.er import EntityCandidate, ERConfig
from fmd.extract.pipeline import build_router
from fmd.plan import DocumentPlanner

ENTITY_NAME = "Fixture Institute of Medical Sciences"


def _planner(registry) -> DocumentPlanner:
    return DocumentPlanner(
        candidates_provider=lambda: [
            EntityCandidate(entity_id=42, canonical_name=ENTITY_NAME, state="Delhi")
        ],
        er_config=ERConfig.from_mapping(
            {"blocking": {"backend": "deterministic"}, "similarity": {"use_embeddings": False}}
        ),
        plan_config=registry.plan,
    )


def _deferred_document(
    harness: Harness,
    url: str,
    *,
    plan_state: str = "planned",
    plan_reason: str | None = "defer_for_text",
) -> object:
    return harness.documents.upsert(
        DocumentUpsert(
            url=url,
            domain="college.ac.in",
            plan_state=plan_state,
            plan_reason=plan_reason,
        )
    )


def _seed_snapshot(harness: Harness, document, body: bytes, content_type: str) -> None:
    digest = harness.store.put(body)
    harness.snapshots.create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=str(harness.store.path_for(digest)),
            content_type=content_type,
        )
    )


def test_deferred_document_is_planned_at_phase_b_then_extracted(tmp_path, registry) -> None:
    text = (PLAN_FIXTURES / "e2e_placement_report.txt").read_text(encoding="utf-8")
    router = build_router(registry.models, mock=True, fixtures_dir=PLAN_LLM_FIXTURES)
    harness = Harness(tmp_path, registry, router=router)
    harness.orchestrator.planner = _planner(registry)
    url = "https://college.ac.in/files/placement-2025.txt"
    document = _deferred_document(harness, url)
    _seed_snapshot(harness, document, text.encode(), "text/plain")

    summary = harness.orchestrator.extract(limit=5)

    stored = harness.documents.get(document.id)
    assert stored is not None
    assert stored.extraction_family == "placement_report"
    assert stored.entity_id == 42
    assert stored.plan_state == "linked"
    assert summary.counts["plan.deferred"] == 1
    assert summary.counts["plan.phase_b.resolved_family"] == 1
    assert summary.counts["plan.phase_b.resolved_entity"] == 1
    assert summary.counts["plan.phase_b.pages_wasted"] == 0
    assert summary.counts["extracted"] == 1
    assert {row.entity_id for row in harness.claims.rows} == {42}


def test_second_extract_does_not_replan_or_reacquire(tmp_path, registry) -> None:
    text = (PLAN_FIXTURES / "e2e_placement_report.txt").read_text(encoding="utf-8")
    router = build_router(registry.models, mock=True, fixtures_dir=PLAN_LLM_FIXTURES)
    harness = Harness(tmp_path, registry, router=router)
    harness.orchestrator.planner = _planner(registry)
    document = _deferred_document(
        harness, "https://college.ac.in/files/placement-2026.txt"
    )
    _seed_snapshot(harness, document, text.encode(), "text/plain")

    first = harness.orchestrator.extract(limit=5)
    second = harness.orchestrator.extract(limit=5)

    assert first.counts["plan.deferred"] == 1
    assert second.counts.get("plan.deferred", 0) == 0
    assert second.counts.get("plan.phase_b.pages_acquired", 0) == 0
    # The second pass sees the same snapshot as already extracted at this tool
    # version and skips it; no document was acquired to discover that.
    assert second.counts.get("extracted", 0) == 0


def test_noise_document_is_never_acquired(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    planner = _planner(registry)
    harness.orchestrator.planner = planner
    url = "https://college.ac.in/files/office-order-17119.pdf"

    # Phase A on the URL alone is below the acquisition floor: the document is
    # abstained, never deferred, so no bytes are ever read for it.
    from types import SimpleNamespace

    phase_a = planner.plan(SimpleNamespace(id=1, url=url, extraction_family=None), phase="a")
    assert phase_a.state == "abstained"
    assert phase_a.abstain_reason == "family_undetected"

    document = _deferred_document(
        harness,
        url,
        plan_state=phase_a.persisted_state,
        plan_reason=phase_a.persisted_reason,
    )
    _seed_snapshot(harness, document, b"Office order\nBy order\n", "text/plain")

    summary = harness.orchestrator.extract(limit=5)

    assert summary.counts.get("plan.deferred", 0) == 0
    assert summary.counts.get("plan.phase_b.pages_acquired", 0) == 0
    assert harness.extractions.rows == []


def test_memory_documents_support_the_new_state_without_a_db_migration() -> None:
    documents = MemoryDocuments()
    row = documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/deferred.txt",
            domain="college.ac.in",
            plan_state="planned",
            plan_reason="defer_for_text",
        )
    )
    assert row.plan_reason == "defer_for_text"
