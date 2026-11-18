"""The standalone planning pass (`fmd plan`) over existing documents — offline.

Regression scope: after snapshot recovery / backfills, harvest runs reported
``{"unplanned": 84, "claims": 0}`` because those documents were never planned
and extraction skips unplanned documents. These tests prove the pass plans them
with the same two-phase planner crawl uses, never plans from nothing, and that
`fmd extract` consumes the result without any change to extract.py.

No network, no model calls: the LLM router replays recorded fixtures.
"""

from __future__ import annotations

from typing import Any

from test_orchestrate import PLAN_FIXTURES, PLAN_LLM_FIXTURES, Harness
from test_orchestrate_support import MemoryDocuments

from fmd.db.repositories import DocumentUpsert, SnapshotCreate
from fmd.er import EntityCandidate, ERConfig
from fmd.extract.pipeline import build_router
from fmd.orchestrate.ledger import RunSummary, utcnow
from fmd.orchestrate.plan import PlanningPass
from fmd.plan import DocumentPlanner

ENTITY_NAME = "Fixture Institute of Medical Sciences"
VERSION = "1"

PLANNER_CODES = {
    "family_undetected",
    "ambiguous_family",
    "entity_undetected",
    "entity_unresolved",
    "entity_ambiguous",
}


class PassDocuments(MemoryDocuments):
    """MemoryDocuments with the planning-pass candidate queue.

    Mirrors ``DocumentsRepo.list_plan_candidates`` (the real SQL is proven in
    ``test_plan_db.py``): the default queue is never-planned plus Phase A
    deferrals; ``force`` widens to decided rows whose reason is planner-shaped.
    With ``strict_filter`` off, the force queue returns *every* decided row so
    the pass's own version/prose guards are exercised too.
    """

    def __init__(self) -> None:
        super().__init__()
        self.set_plan_calls = 0
        self.strict_filter = True

    def set_plan(self, document_id: int, **fields: Any) -> Any:
        self.set_plan_calls += 1
        return super().set_plan(document_id, **fields)

    def list_plan_candidates(
        self, *, limit: int = 500, force: bool = False, planner_version: str | None = None
    ) -> list[Any]:
        rows = sorted(self.rows.values(), key=lambda row: row.id)
        if not force:
            queue = [row for row in rows if row.plan_state is None or row.plan_state == "planned"]
            return queue[:limit]
        queue: list[Any] = []
        for row in rows:
            if row.plan_state not in ("linked", "needs_review", "abstained", "not_applicable"):
                continue
            reason = row.plan_reason
            if self.strict_filter:
                shaped = (
                    reason is None
                    or reason in PLANNER_CODES
                    or reason.startswith("unsupported_family:")
                    or reason.startswith("plan_v")
                    or "|plan_v" in reason
                )
                if not shaped:
                    continue  # human-review prose: never replanned
                if planner_version and (
                    reason == f"plan_v{planner_version}"
                    or (reason or "").endswith(f"|plan_v{planner_version}")
                ):
                    continue  # already decided at this planner version
            queue.append(row)
        return queue[:limit]


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


def _summary() -> RunSummary:
    return RunSummary(run_id="plan-pass-test", stage="crawl", started_at=utcnow())


def _pass(harness: Harness, registry, *, documents: Any | None = None, **kwargs: Any):
    if documents is None:
        # Default to the candidate-capable fake sharing the harness rows, so
        # every test exercises the same queue contract the real repo implements.
        documents = PassDocuments()
        documents.rows = harness.documents.rows
    return PlanningPass(
        planner=_planner(registry),
        documents=documents,
        snapshots=harness.snapshots,
        store=harness.store,
        text_sample_chars=registry.plan.phase_b.text_sample_chars,
        version=VERSION,
        **kwargs,
    )


def _document(
    harness: Harness,
    url: str,
    *,
    role: str | None = None,
    **fields: Any,
) -> Any:
    return harness.documents.upsert(
        DocumentUpsert(url=url, domain="college.ac.in", role=role, **fields)
    )


def _seed_snapshot(harness: Harness, document: Any, body: bytes, content_type: str) -> None:
    digest = harness.store.put(body)
    harness.snapshots.create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=str(harness.store.path_for(digest)),
            content_type=content_type,
        )
    )


_PLACEMENT_TEXT = (PLAN_FIXTURES / "e2e_placement_report.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------- unplanned -> planned


def test_unplanned_document_is_planned_and_becomes_extractable(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    document = _document(harness, "https://college.ac.in/files/placement-2025.pdf")
    _seed_snapshot(harness, document, _PLACEMENT_TEXT.encode(), "text/plain")

    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.extraction_family == "placement_report"
    assert stored.entity_id == 42
    assert stored.plan_state == "linked"
    assert stored.plan_reason == "plan_v1"
    assert summary.counts["plan.candidates"] == 1
    assert summary.counts["planned"] == 1
    assert summary.counts["linked"] == 1
    assert summary.counts.get("still_abstained", 0) == 0
    assert summary.metadata["planner_version"] == VERSION


def test_phase_b_reads_the_stored_snapshot_not_the_url(tmp_path, registry) -> None:
    """A recovered PDF with an opaque URL is classified from its own text."""
    harness = Harness(tmp_path, registry)
    url = "https://college.ac.in/files/FMD-2025-final.pdf"  # no family signal
    document = _document(harness, url)
    _seed_snapshot(harness, document, _PLACEMENT_TEXT.encode(), "text/plain")

    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.extraction_family == "placement_report"
    assert stored.entity_id == 42
    assert stored.plan_state == "linked"


def test_defer_for_text_is_resolved_by_phase_b(tmp_path, registry) -> None:
    """A crawl-era Phase A deferral with a stored snapshot reaches Phase B."""
    harness = Harness(tmp_path, registry)
    document = _document(
        harness,
        "https://college.ac.in/files/report-2026.pdf",
        plan_state="planned",
        plan_reason="defer_for_text",
    )
    _seed_snapshot(harness, document, _PLACEMENT_TEXT.encode(), "text/plain")

    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.plan_state == "linked"
    assert stored.extraction_family == "placement_report"
    assert stored.entity_id == 42
    assert stored.plan_reason == "plan_v1"
    assert summary.counts["planned"] == 1
    assert summary.counts["linked"] == 1


def test_reference_hints_survive_an_awaiting_text_document(tmp_path, registry) -> None:
    """A no-text document with stored reference hints is re-affirmed at Phase A."""
    harness = Harness(tmp_path, registry)
    document = _document(
        harness,
        "https://college.ac.in/files/report.pdf",
        extraction_family="placement_report",
        entity_id=42,
    )

    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.plan_state == "linked"
    assert stored.extraction_family == "placement_report"
    assert stored.entity_id == 42
    assert summary.counts["linked"] == 1


# ------------------------------------------------------------------ never from nothing


def test_evidence_without_snapshot_awaits_text(tmp_path, registry) -> None:
    """A confident URL is never enough: no snapshot means awaiting_text."""
    harness = Harness(tmp_path, registry)
    document = _document(harness, "https://college.ac.in/files/placement-2025.pdf")

    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.plan_state == "planned"
    assert stored.plan_reason == "defer_for_text"
    assert stored.extraction_family is None
    assert stored.entity_id is None
    assert summary.counts["awaiting_text"] == 1
    assert summary.counts.get("linked", 0) == 0
    # And extraction still cannot select it: there is nothing to read.
    assert harness.snapshots.list_extractable("1.0.0", limit=10) == []


def test_snapshot_without_a_text_layer_awaits_text(tmp_path, registry) -> None:
    """Scanned bytes have no T0 text: awaiting text, never an OCR call here."""
    from types import SimpleNamespace

    harness = Harness(tmp_path, registry)
    document = _document(harness, "https://college.ac.in/files/scan.pdf")
    _seed_snapshot(harness, document, b"%PDF-1.4 scanned image only", "application/pdf")

    summary = _summary()
    _pass(harness, registry, acquire=lambda *a, **k: SimpleNamespace(text="")).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.plan_state == "planned"
    assert stored.plan_reason == "defer_for_text"
    assert summary.counts["awaiting_text"] == 1


def test_acquisition_failure_leaves_the_row_untouched(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    document = _document(harness, "https://college.ac.in/files/placement.pdf")
    _seed_snapshot(harness, document, _PLACEMENT_TEXT.encode(), "text/plain")

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("pdftext exploded")

    summary = _summary()
    _pass(harness, registry, acquire=_boom).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.plan_state is None  # retried on the next run
    assert summary.counts["acquisition_failed"] == 1
    assert summary.counts.get("planned", 0) == 0


def test_roles_that_were_never_candidates_stay_not_applicable(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    inventory = _document(harness, "https://gov.in/ugc-list.json", role="inventory")
    navigation = _document(harness, "https://college.ac.in/", role="navigation")
    _seed_snapshot(harness, inventory, b'{"colleges": []}', "application/json")

    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    assert harness.documents.get(inventory.id).plan_state == "not_applicable"
    assert harness.documents.get(navigation.id).plan_state == "not_applicable"
    assert summary.counts["skipped_not_applicable"] == 2


def test_abstained_documents_carry_a_stable_reason(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    document = _document(harness, "https://college.ac.in/files/office-order-17119.pdf")
    _seed_snapshot(harness, document, b"Office order\nBy order\n", "text/plain")
    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.plan_state == "abstained"
    assert stored.plan_reason == "family_undetected|plan_v1"
    assert summary.counts["still_abstained"] == 1
    assert summary.metadata["reasons"] == {"family_undetected": 1}
    assert summary.counts["plan.reason.family_undetected"] == 1


def test_unsupported_registry_family_is_reported_not_guessed(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    document = _document(
        harness,
        "https://college.ac.in/files/report.pdf",
        extraction_family="bogus_family",
        entity_id=42,
    )
    _seed_snapshot(harness, document, _PLACEMENT_TEXT.encode(), "text/plain")

    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    stored = harness.documents.get(document.id)
    assert stored.plan_state == "abstained"
    assert stored.plan_reason == "unsupported_family:bogus_family|plan_v1"
    assert summary.metadata["reasons"] == {"unsupported_family:bogus_family": 1}


# --------------------------------------------------------------- idempotency & flags


def test_second_run_makes_no_changes(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    documents = PassDocuments()
    documents.rows = harness.documents.rows
    planned = _document(harness, "https://college.ac.in/files/placement-2025.pdf")
    _seed_snapshot(harness, planned, _PLACEMENT_TEXT.encode(), "text/plain")
    waiting = _document(harness, "https://college.ac.in/files/scan.pdf")

    stage = _pass(harness, registry, documents=documents)
    first = _summary()
    stage.run(summary=first)
    writes_after_first = documents.set_plan_calls

    second = _summary()
    stage.run(summary=second)

    assert first.counts["planned"] == 2
    # The waiting document is re-decided (it is still deferred) but nothing is
    # written: the decision equals the stored row.
    assert second.counts["plan.candidates"] == 1
    assert second.counts["awaiting_text"] == 1
    assert second.counts["unchanged"] == 1
    assert documents.set_plan_calls == writes_after_first
    assert harness.documents.get(waiting.id).plan_reason == "defer_for_text"


def test_already_planned_documents_are_not_candidates(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    linked = _document(
        harness,
        "https://college.ac.in/files/placement-2025.pdf",
        extraction_family="placement_report",
        entity_id=42,
        plan_state="linked",
        plan_reason="plan_v1",
    )
    abstained = _document(
        harness,
        "https://college.ac.in/files/office-order.pdf",
        plan_state="abstained",
        plan_reason="family_undetected|plan_v1",
    )

    summary = _summary()
    _pass(harness, registry).run(summary=summary)

    assert summary.counts["plan.candidates"] == 0
    assert summary.counts.get("planned", 0) == 0
    assert harness.documents.get(linked.id).plan_state == "linked"
    assert harness.documents.get(abstained.id).plan_state == "abstained"


def test_dry_run_computes_counters_but_writes_nothing(tmp_path, registry) -> None:
    documents = PassDocuments()
    harness = Harness(tmp_path, registry)
    documents.rows = harness.documents.rows
    document = _document(harness, "https://college.ac.in/files/placement-2025.pdf")
    _seed_snapshot(harness, document, _PLACEMENT_TEXT.encode(), "text/plain")

    summary = _summary()
    summary.dry_run = True
    _pass(harness, registry, documents=documents).run(summary=summary, dry_run=True)

    assert summary.counts["linked"] == 1  # the decision was computed
    assert documents.set_plan_calls == 0
    stored = harness.documents.get(document.id)
    assert stored.plan_state is None  # nothing was written
    assert stored.extraction_family is None


def test_force_replans_stale_versions_only(tmp_path, registry) -> None:
    documents = PassDocuments()
    documents.strict_filter = False  # the pass's own guards must do the filtering
    harness = Harness(tmp_path, registry)
    documents.rows = harness.documents.rows
    stale_stamp = _document(
        harness,
        "https://college.ac.in/files/a.pdf",
        extraction_family="placement_report",
        entity_id=42,
        plan_state="linked",
        plan_reason="family_undetected|plan_v0",
    )
    crawl_era = _document(
        harness,
        "https://college.ac.in/files/b.pdf",
        extraction_family="placement_report",
        entity_id=42,
        plan_state="linked",
        plan_reason=None,
    )
    current = _document(
        harness,
        "https://college.ac.in/files/c.pdf",
        extraction_family="placement_report",
        entity_id=42,
        plan_state="linked",
        plan_reason="plan_v1",
    )
    human = _document(
        harness,
        "https://college.ac.in/files/d.pdf",
        extraction_family="placement_report",
        entity_id=7,
        plan_state="linked",
        plan_reason="resolved by human review (link_entity)",
    )

    summary = _summary()
    _pass(harness, registry, documents=documents).run(summary=summary, force=True)

    assert summary.counts["plan.candidates"] == 4  # every decided row surfaced
    assert summary.counts["plan.skipped_current"] == 2  # current stamp + human prose
    assert summary.counts["linked"] == 2  # stale stamp + crawl era, replanned
    assert harness.documents.get(stale_stamp.id).plan_reason == "plan_v1"
    assert harness.documents.get(crawl_era.id).plan_reason == "plan_v1"
    # The current-version stamp and the human decision are untouched.
    assert harness.documents.get(current.id).plan_reason == "plan_v1"
    assert harness.documents.get(human.id).plan_reason == "resolved by human review (link_entity)"
    assert harness.documents.get(human.id).entity_id == 7


def test_force_respects_the_strict_candidate_queue(tmp_path, registry) -> None:
    """The repository queue itself excludes current stamps and human prose."""
    documents = PassDocuments()  # strict_filter is the default
    harness = Harness(tmp_path, registry)
    documents.rows = harness.documents.rows
    current = _document(
        harness,
        "https://college.ac.in/files/c.pdf",
        extraction_family="placement_report",
        entity_id=42,
        plan_state="linked",
        plan_reason="plan_v1",
    )
    human = _document(
        harness,
        "https://college.ac.in/files/d.pdf",
        extraction_family="placement_report",
        entity_id=7,
        plan_state="linked",
        plan_reason="resolved by human review (link_entity)",
    )

    summary = _summary()
    _pass(harness, registry, documents=documents).run(summary=summary, force=True)

    assert summary.counts["plan.candidates"] == 0
    assert summary.counts.get("planned", 0) == 0
    assert harness.documents.get(current.id).plan_reason == "plan_v1"
    assert harness.documents.get(human.id).plan_reason == "resolved by human review (link_entity)"


def test_limit_bounds_the_run_and_resumes(tmp_path, registry) -> None:
    harness = Harness(tmp_path, registry)
    first = _document(harness, "https://college.ac.in/files/placement-2025.pdf")
    _seed_snapshot(harness, first, _PLACEMENT_TEXT.encode(), "text/plain")
    second = _document(harness, "https://college.ac.in/files/placement-2026.pdf")
    _seed_snapshot(harness, second, _PLACEMENT_TEXT.encode(), "text/plain")

    summary = _summary()
    _pass(harness, registry).run(summary=summary, limit=1)

    assert summary.counts["plan.candidates"] == 1
    assert summary.counts["linked"] == 1
    states = {row.id: harness.documents.get(row.id).plan_state for row in (first, second)}
    assert states[first.id] == "linked"
    assert states[second.id] is None  # still queued for the next run

    resumed = _summary()
    _pass(harness, registry).run(summary=resumed)
    assert resumed.counts["linked"] == 1
    assert harness.documents.get(second.id).plan_state == "linked"


# ------------------------------------------------------------- extract consumes it


def test_extract_processes_documents_the_pass_planned(tmp_path, registry) -> None:
    """Acceptance: `fmd plan` then `fmd extract` produces claims — no extract.py change."""
    text = _PLACEMENT_TEXT
    router = build_router(registry.models, mock=True, fixtures_dir=PLAN_LLM_FIXTURES)
    harness = Harness(tmp_path, registry, router=router)
    harness.orchestrator.planner = _planner(registry)
    documents = PassDocuments()
    documents.rows = harness.documents.rows
    document = _document(harness, "https://college.ac.in/files/placement-2025.pdf")
    _seed_snapshot(harness, document, text.encode(), "text/plain")

    # Before the pass, extraction has nothing to read: the document is unplanned.
    before = harness.orchestrator.extract(limit=5)
    assert before.counts.get("extracted", 0) == 0
    assert harness.claims.rows == []

    plan_summary = _summary()
    _pass(harness, registry, documents=documents).run(summary=plan_summary)
    assert harness.documents.get(document.id).plan_state == "linked"
    after = harness.orchestrator.extract(limit=5)
    assert after.counts["extracted"] == 1
    assert after.counts["claims"] > 0
    assert {row.entity_id for row in harness.claims.rows} == {42}


def test_pass_run_summary_is_ledger_shaped(tmp_path, registry) -> None:
    """The summary prints as the single JSON object the run ledger persists."""
    import json

    harness = Harness(tmp_path, registry)
    document = _document(harness, "https://college.ac.in/files/placement-2025.pdf")
    _seed_snapshot(harness, document, _PLACEMENT_TEXT.encode(), "text/plain")

    summary = _summary()
    _pass(harness, registry).run(summary=summary)
    summary.finish()
    payload = json.loads(summary.as_json())

    assert payload["stage"] == "crawl"  # pipeline_runs.stage is a closed set
    assert payload["metadata"]["stage_detail"] == "planning_pass"
    for counter in ("planned", "linked", "plan.candidates"):
        assert counter in payload["counts"]
    assert payload["metadata"]["reasons"] == {}
