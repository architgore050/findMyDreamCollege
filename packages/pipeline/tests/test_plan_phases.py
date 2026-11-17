"""Two-phase planner: Phase A defers, Phase B reads text (no network, no DB).

The fixture text is a cover/index/fee excerpt in the shape of a real university
PDF; the URLs are deliberately ordinary so Phase A cannot rely on a filename
that already contains the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from test_orchestrate_support import MemoryDocuments

from fmd.db.repositories import DocumentUpsert
from fmd.er import EntityCandidate, ERConfig
from fmd.orchestrate.ledger import RunSummary, utcnow
from fmd.orchestrate.plan import PlanStage
from fmd.plan import DocumentPlan, DocumentPlanner
from fmd.plan.planner import DEFER_CODE

FIXTURES = Path(__file__).parent / "fixtures" / "plan"
FEE_TEXT = (FIXTURES / "fee_structure_cover.txt").read_text(encoding="utf-8")
NOISE_TEXT = (FIXTURES / "noise_office_order.txt").read_text(encoding="utf-8")
ENTITY_NAME = "Demo University of Health Sciences"
PLACEMENT_TEXT = (FIXTURES / "placement_report.txt").read_text(encoding="utf-8")


def _config() -> ERConfig:
    return ERConfig.from_mapping(
        {"blocking": {"backend": "deterministic"}, "similarity": {"use_embeddings": False}}
    )


def _planner(candidates: list[EntityCandidate] | None = None) -> DocumentPlanner:
    return DocumentPlanner(
        candidates_provider=lambda: list(candidates or []),
        er_config=_config(),
    )


def _document(url: str) -> SimpleNamespace:
    return SimpleNamespace(id=1, url=url, source_id=None, extraction_family=None, entity_id=None)


@dataclass
class _Snapshot:
    id: int
    document_id: int
    sha256: str
    content_type: str = "application/pdf"


@dataclass
class _Store:
    data: dict[str, bytes]
    calls: int = 0

    def get(self, sha: str) -> bytes:
        self.calls += 1
        return self.data[sha]


@dataclass
class _Snapshots:
    rows: list[_Snapshot] = field(default_factory=list)

    def list_without_extraction(self, tool_version=None, limit=100):
        return list(self.rows)[:limit]


def _stage(
    planner: DocumentPlanner,
    documents: MemoryDocuments,
    snapshots: _Snapshots,
    store: _Store,
    pages: int = 3,
):
    acquired: list[str] = []

    def acquire(data, *, content_type=None, policy=None):
        acquired.append(content_type or "")
        return SimpleNamespace(text=data.decode("utf-8"), page_map=[object()] * pages)

    stage = PlanStage(
        planner=planner,
        documents=documents,
        snapshots=snapshots,
        store=store,
        acquire=acquire,
        text_sample_chars=4000,
    )
    return stage, acquired


def test_ambiguous_url_defers_then_text_plans_family_and_entity() -> None:
    url = "https://example.ac.in/uploads/fee-2025.pdf"
    document = _document(url)
    planner = _planner(
        [EntityCandidate(entity_id=7, canonical_name=ENTITY_NAME, state="Delhi")]
    )

    phase_a = planner.plan(document, phase="a")
    assert phase_a.state == "defer_for_text"
    assert phase_a.family is None
    assert phase_a.confidence > 0

    phase_b = planner.plan(document, text_sample=FEE_TEXT, phase="b")
    assert phase_b.state == "linked"
    assert phase_b.family == "fee_structure"
    assert phase_b.entity_id == 7
    assert phase_b.defer_reason is None


def test_noise_text_abstains_without_buying_any_pages() -> None:
    url = "https://example.ac.in/uploads/office-order-17119.pdf"
    planner = _planner([EntityCandidate(entity_id=1, canonical_name=ENTITY_NAME)])
    document = _document(url)

    phase_a = planner.plan(document, phase="a")
    assert phase_a.state == "abstained"
    assert phase_a.abstain_reason == "family_undetected"

    documents = MemoryDocuments()
    row = documents.upsert(
        DocumentUpsert(
            url=url,
            domain="example.ac.in",
            plan_state=phase_a.persisted_state,
            plan_reason=phase_a.persisted_reason,
        )
    )
    snapshots = _Snapshots([_Snapshot(id=1, document_id=row.id, sha256="a" * 64)])
    store = _Store({"a" * 64: NOISE_TEXT.encode()})
    stage, acquired = _stage(planner, documents, snapshots, store)

    summary = RunSummary(run_id="noise", stage="extract", started_at=utcnow())
    assert stage.run(summary=summary) == 0
    assert acquired == []
    assert summary.counts.get("plan.phase_b.pages_acquired", 0) == 0


def test_phase_b_text_beats_phase_a_url_at_equal_scope() -> None:
    # The URL says "placement", the document is a fee order. Text wins.
    plan = _planner().plan(
        _document("https://example.ac.in/uploads/placement-circular.pdf"),
        text_sample=FEE_TEXT,
        phase="b",
    )

    assert plan.family == "fee_structure"
    assert plan.family_source == "heuristic"


def test_phase_b_falls_back_to_phase_a_url_when_text_is_silent() -> None:
    plan = _planner().plan(
        _document("https://example.ac.in/uploads/placement-placements.pdf"),
        text_sample="Office order\nBy order\n",
        phase="b",
    )

    assert plan.family == "placement_report"
    assert plan.signals.get("fallback") == "phase_a_url"


def test_registry_family_is_never_downgraded_by_text() -> None:
    plan = _planner().plan(
        _document("https://example.ac.in/files/2024-25.pdf"),
        source={"params": {"extraction_family": "fee_structure"}},
        text_sample=PLACEMENT_TEXT,
        phase="b",
    )

    assert plan.family == "fee_structure"
    assert plan.family_source == "registry"
    assert plan.signals["declared"] == "fee_structure"


def test_phase_b_ambiguous_entity_is_review_not_a_guess() -> None:
    plan = _planner(
        [
            EntityCandidate(entity_id=1, canonical_name=ENTITY_NAME, state="Delhi"),
            EntityCandidate(entity_id=2, canonical_name=ENTITY_NAME, state="Delhi"),
        ]
    ).plan(
        _document("https://example.ac.in/uploads/fee-order.pdf"),
        text_sample=FEE_TEXT,
        phase="b",
    )

    assert plan.family == "fee_structure"
    assert plan.entity_id is None
    assert plan.state == "needs_review"
    assert plan.abstain_reason == "entity_ambiguous"


def test_plan_stage_is_idempotent_and_acquires_each_sha_once() -> None:
    url = "https://example.ac.in/uploads/fee-2025.pdf"
    planner = _planner([EntityCandidate(entity_id=7, canonical_name=ENTITY_NAME)])
    phase_a = planner.plan(_document(url), phase="a")
    assert phase_a.state == "defer_for_text"

    documents = MemoryDocuments()
    row = documents.upsert(
        DocumentUpsert(
            url=url,
            domain="example.ac.in",
            plan_state=phase_a.persisted_state,
            plan_reason=phase_a.persisted_reason,
        )
    )
    # Two snapshots of the same content must not buy the text twice.
    snapshots = _Snapshots(
        [
            _Snapshot(id=1, document_id=row.id, sha256="b" * 64),
            _Snapshot(id=2, document_id=row.id, sha256="b" * 64),
        ]
    )
    store = _Store({"b" * 64: FEE_TEXT.encode()})
    stage, acquired = _stage(planner, documents, snapshots, store, pages=11)

    first = RunSummary(run_id="one", stage="extract", started_at=utcnow())
    assert stage.run(summary=first) == 1
    stored = documents.get(row.id)
    assert stored is not None
    assert stored.extraction_family == "fee_structure"
    assert stored.entity_id == 7
    assert stored.plan_state == "linked"

    second = RunSummary(run_id="two", stage="extract", started_at=utcnow())
    assert stage.run(summary=second) == 0
    assert acquired == ["application/pdf"]


def test_run_summary_reports_resolution_and_wasted_acquisition_cost() -> None:
    planner = _planner([EntityCandidate(entity_id=7, canonical_name=ENTITY_NAME)])
    documents = MemoryDocuments()
    resolved = documents.upsert(
        DocumentUpsert(
            url="https://example.ac.in/uploads/fee-order-2025.pdf",
            domain="example.ac.in",
            plan_state="planned",
            plan_reason=DEFER_CODE,
        )
    )
    wasted = documents.upsert(
        DocumentUpsert(
            url="https://example.ac.in/uploads/notice-99.pdf",
            domain="example.ac.in",
            plan_state="planned",
            plan_reason=DEFER_CODE,
        )
    )
    snapshots = _Snapshots(
        [
            _Snapshot(id=1, document_id=resolved.id, sha256="c" * 64),
            _Snapshot(id=2, document_id=wasted.id, sha256="d" * 64),
        ]
    )
    store = _Store({"c" * 64: FEE_TEXT.encode(), "d" * 64: NOISE_TEXT.encode()})
    stage, _ = _stage(planner, documents, snapshots, store, pages=4)

    summary = RunSummary(run_id="counts", stage="extract", started_at=utcnow())
    stage.run(summary=summary)

    assert summary.counts["plan.deferred"] == 2
    assert summary.counts["plan.phase_b.resolved_family"] == 1
    assert summary.counts["plan.phase_b.pages_acquired"] == 8
    assert summary.counts["plan.phase_b.pages_wasted"] == 4
    assert summary.metadata["phase_b"]["abstain_reasons"] == {"family_undetected": 1}


def test_plan_payload_exposes_the_deferred_state() -> None:
    plan = DocumentPlan(state="defer_for_text", defer_reason="entity_undetected")
    assert plan.deferred is True
    assert plan.persisted_state == "planned"
    assert plan.persisted_reason == DEFER_CODE
