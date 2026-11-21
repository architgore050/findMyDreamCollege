"""Shared fakes and builders for the W10 parse-cache / batched-acquire tests.

Everything here is offline: no Docling import, no network, no GPU. The "live
conversion" is the real tier dispatcher (``run_tiers``) driven by a fake T0
probe document and a fake T1 that builds the report document exactly the way
``fmd.acquire.tiers._docling_doc`` would from Docling items — the same shape a
cached payload reconstructs.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from test_orchestrate_support import (
    MemoryClaims,
    MemoryDocuments,
    MemoryEntities,
    MemoryExtractions,
    MemorySnapshots,
)

from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
from fmd.acquire.outline import OutlinePolicy
from fmd.acquire.parse_cache import ParseCache
from fmd.acquire.tables import CellHeaderLink
from fmd.acquire.types import ExtractedDoc, PageSpan, Table
from fmd.db.repositories import (
    DocumentParsePut,
    DocumentParseRecord,
    DocumentUpsert,
    SnapshotCreate,
)
from fmd.store import SnapshotStore

FETCHED_AT = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)

#: Fake PDF bytes: real enough for the dispatcher's ``%PDF-`` magic probe.
PDF_BYTES = b"%PDF-1.4\n% fake report for the parse-cache tests\n"


# --------------------------------------------------------------------- fakes


class CountingT1:
    """A fake T1 tier: counts invocations (a "converter call" in this suite)."""

    def __init__(self, doc_factory: Any = None) -> None:
        self.calls = 0
        self.doc_factory = doc_factory or report_doc
        self.error: Exception | None = None
        self.fail_on_call: int | None = None

    def __call__(self, data: bytes) -> ExtractedDoc:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise self.error or RuntimeError("fake converter exploded")
        return self.doc_factory()


class CountingT0:
    """A fake T0 tier that plans T1: its text layer looks like a fee table."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, data: bytes) -> ExtractedDoc:
        self.calls += 1
        return t0_table_doc()


class RaisingRouter:
    """A router whose every call is a test failure — acquire must never call it."""

    def __init__(self, models: Any = None) -> None:
        self.models = models
        self.calls: list[Any] = []

    def call(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(args)
        raise AssertionError("acquire stage must make zero LLM calls")


class FakeParsesRepo:
    """In-memory ``document_parses`` repo with the production repo's surface."""

    def __init__(self) -> None:
        self.rows: dict[str, DocumentParseRecord] = {}
        self.fail_put = False

    def put(self, parse: DocumentParsePut) -> DocumentParseRecord:
        if self.fail_put:
            raise RuntimeError("simulated repo outage")
        record = DocumentParseRecord(**parse.model_dump(), created_at=FETCHED_AT)
        self.rows[record.key_digest] = record
        return record

    def get(self, key_digest: str) -> DocumentParseRecord | None:
        return self.rows.get(key_digest)

    def count(self) -> int:
        return len(self.rows)


class FakeServerController:
    """Records the control sequence; readiness is a flag the test flips."""

    def __init__(self, *, healthy: bool = True, fail_start: bool = False) -> None:
        self.events: list[str] = []
        self.healthy = healthy
        self.fail_start = fail_start
        self.stopped = True
        self.wait_calls = 0

    def stop(self) -> None:
        self.events.append("stop")
        self.stopped = True
        self.healthy = False

    def start(self) -> None:
        self.events.append("start")
        if self.fail_start:
            raise RuntimeError("simulated server start failure")
        self.healthy = True

    def wait_ready(self, timeout_s: float) -> None:
        self.events.append("wait_ready")
        self.wait_calls += 1
        if not self.healthy:
            raise RuntimeError("server not ready (fake)")


class ExplodingT1:
    """A T1 that always raises — used to prove failure isolation."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error or RuntimeError("docling exploded mid-document")

    def __call__(self, data: bytes) -> ExtractedDoc:
        self.calls += 1
        raise self.error


# ------------------------------------------------------------ report document


def report_blocks() -> list[TextBlock]:
    """Three pages of blocks: title/outline labels, one table, placements."""
    return [
        TextBlock(
            page_no=1,
            text="Admission and Fee Structure 2026",
            bbox=(40.0, 40.0, 400.0, 60.0),
            granularity="span",
            outline_label="title",
        ),
        TextBlock(
            page_no=1,
            text="Institute of Medical Laboratory Sciences",
            bbox=(40.0, 80.0, 420.0, 100.0),
            granularity="span",
            outline_label="section_header",
        ),
        TextBlock(
            page_no=1,
            text="The institute offers the B.Sc. Medical Laboratory Technology programme.",
            bbox=(40.0, 120.0, 500.0, 140.0),
            granularity="span",
        ),
        TextBlock(
            page_no=2,
            text="Fee particulars are given below.",
            bbox=(40.0, 40.0, 300.0, 60.0),
            granularity="span",
        ),
        TextBlock(page_no=2, text="Fee Head", bbox=(40.0, 80.0, 140.0, 100.0), granularity="span"),
        TextBlock(
            page_no=2, text="Amount (INR)", bbox=(160.0, 80.0, 300.0, 100.0), granularity="span"
        ),
        TextBlock(
            page_no=2, text="Tuition fee", bbox=(40.0, 120.0, 140.0, 140.0), granularity="span"
        ),
        TextBlock(page_no=2, text="50000", bbox=(160.0, 120.0, 300.0, 140.0), granularity="span"),
        TextBlock(
            page_no=2, text="Hostel fee", bbox=(40.0, 160.0, 140.0, 180.0), granularity="span"
        ),
        TextBlock(page_no=2, text="12000", bbox=(160.0, 160.0, 300.0, 180.0), granularity="span"),
        TextBlock(
            page_no=3,
            text="Placement Outcome",
            bbox=(40.0, 40.0, 260.0, 60.0),
            granularity="span",
            outline_label="section_header",
        ),
        TextBlock(
            page_no=3,
            text="Eligible students: 120",
            bbox=(40.0, 80.0, 300.0, 100.0),
            granularity="span",
        ),
        TextBlock(
            page_no=3,
            text="Students placed: 96",
            bbox=(40.0, 120.0, 300.0, 140.0),
            granularity="span",
        ),
        TextBlock(
            page_no=3,
            text="Placement rate: 80%",
            bbox=(40.0, 160.0, 300.0, 180.0),
            granularity="span",
        ),
        TextBlock(
            page_no=3,
            text="Median salary: INR 420000 per annum",
            bbox=(40.0, 200.0, 460.0, 220.0),
            granularity="span",
        ),
        TextBlock(
            page_no=3,
            text="Internships are included in the placement count.",
            bbox=(40.0, 240.0, 480.0, 260.0),
            granularity="span",
        ),
    ]


def report_pages() -> list[PageGeometry]:
    return [
        PageGeometry(
            page_no=page,
            width=595.0,
            height=842.0,
            bbox=(0.0, 0.0, 595.0, 842.0),
            char_start=0,
            char_end=0,
        )
        for page in (1, 2, 3)
    ]


def report_table() -> Table:
    cells = (
        CellHeaderLink(
            row=1, col=0, text="Tuition fee", header_text=None, header_row=None, ambiguous=False
        ),
        CellHeaderLink(
            row=1, col=1, text="50000", header_text="Amount (INR)", header_row=0, ambiguous=False
        ),
        CellHeaderLink(
            row=2, col=0, text="Hostel fee", header_text=None, header_row=None, ambiguous=False
        ),
        CellHeaderLink(
            row=2, col=1, text="12000", header_text="Amount (INR)", header_row=0, ambiguous=False
        ),
    )
    return Table(
        rows=[["Fee Head", "Amount (INR)"], ["Tuition fee", "50000"], ["Hostel fee", "12000"]],
        page=2,
        caption=None,
        cells=cells,
        column_header_uncertain=False,
        confidence=0.92,
    )


def report_doc(policy: OutlinePolicy | None = None) -> ExtractedDoc:
    """The T1 document exactly as ``_docling_doc`` would build it."""
    blocks = report_blocks()
    tables = [report_table()]
    canonical = build_canonical(blocks, report_pages(), outline_policy=policy or OutlinePolicy())
    return ExtractedDoc(
        text=canonical.text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=tables,
        tier="T1",
        tool="docling",
        tool_version="2.31.0",
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity="span",
        two_column=False,
        table_confidence=0.92,
    )


def t0_table_doc() -> ExtractedDoc:
    """A T0 result whose text layer looks like a fee table (plans T1)."""
    lines = [
        "Fee schedule for the programmes offered:",
        "Tuition 50000 12000",
        "Hostel 60000 15000",
        "Exam 40000 9000",
        "Library 20000 3000",
    ]
    text = "\n".join(lines)
    block = TextBlock(page_no=1, text=text, bbox=(10.0, 10.0, 500.0, 400.0), granularity="span")
    canonical = build_canonical(
        [block],
        [
            PageGeometry(
                page_no=1,
                width=595.0,
                height=842.0,
                bbox=(0.0, 0.0, 595.0, 842.0),
                char_start=0,
                char_end=0,
            )
        ],
    )
    return ExtractedDoc(
        text=canonical.text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=[],
        tier="T0",
        tool="pdftext",
        canonical=canonical,
        pages=canonical.pages,
        blocks=(block,),
        bbox_granularity="span",
    )


def stub_payload() -> dict[str, Any]:
    """Quotes drawn verbatim from the report document's canonical text."""
    return {
        "cohort_year": {"quote": "2026", "confidence": 0.9, "reason_if_null": None},
        "eligible_count": {
            "quote": "Eligible students: 120",
            "confidence": 0.95,
            "reason_if_null": None,
        },
        "placed_count": {
            "quote": "Students placed: 96",
            "confidence": 0.95,
            "reason_if_null": None,
        },
        "placement_rate": {
            "quote": "Placement rate: 80%",
            "confidence": 0.9,
            "reason_if_null": None,
        },
        "salary_period": {
            "quote": "Median salary: INR 420000 per annum",
            "confidence": 0.9,
            "reason_if_null": None,
        },
        "median_salary": {
            "quote": "Median salary: INR 420000 per annum",
            "confidence": 0.93,
            "reason_if_null": None,
        },
        "highest_package": {
            "quote": None,
            "confidence": None,
            "reason_if_null": "the report prints a median only",
        },
        "internship_counted_as_employment": {
            "quote": "Internships are included in the placement count.",
            "confidence": 0.88,
            "reason_if_null": None,
        },
        "outcome_basis": {
            "quote": "Internships are included in the placement count.",
            "confidence": 0.7,
            "reason_if_null": None,
        },
    }


class StubRouter:
    """Returns a fixed quote-first guided-JSON payload; no provider, no network."""

    def __init__(self, models: Any, payload: dict[str, Any]) -> None:
        self.models = models
        self.payload = payload
        self.calls: list[tuple[str, str]] = []
        self.fail_for: set[str] = set()

    def call(self, purpose, schema, system, user, input_ref):
        self.calls.append((purpose, input_ref))
        if input_ref in self.fail_for or "*" in self.fail_for:
            raise RuntimeError("simulated LLM failure")
        return schema.model_validate(self.payload)


# ------------------------------------------------------------ document harness


class SnapshotHarness:
    """Snapshots + documents + repos + store + parse cache for one test."""

    def __init__(self, tmp_path: Path, registry: Any, *, router: Any) -> None:
        self.registry = registry
        self.tmp_path = tmp_path
        self.store = SnapshotStore(tmp_path / "snapshots")
        self.documents = MemoryDocuments()
        self.extractions = MemoryExtractions()
        self.claims = MemoryClaims()
        self.entities = MemoryEntities()
        self.snapshots = MemorySnapshots(self.extractions, self.documents)
        self.parse_repo = FakeParsesRepo()
        self.parse_cache = ParseCache(self.parse_repo, self.store)
        self.router = router
        self._document_ids = itertools.count(1)
        self._snapshot_ids = itertools.count(1)

    def add_pdf_document(
        self,
        url: str,
        *,
        family: str = "placement_report",
        entity_id: int = 42,
        content: bytes = PDF_BYTES,
        content_type: str = "application/pdf",
    ) -> tuple[Any, Any]:
        """Store the bytes, create the document row and its snapshot."""
        digest = self.store.put(content)
        document_id = next(self._document_ids)
        document = self.documents.upsert(
            DocumentUpsert(
                url=url,
                domain="college.ac.in",
                kind="report",
                source_id="fixture",
                extraction_family=family,
                entity_id=entity_id,
                plan_state="linked",
                role="evidence",
            )
        )
        snapshot = self.snapshots.create(
            SnapshotCreate(
                document_id=document_id,
                sha256=digest,
                bytes_path=f"objects/{digest[:2]}/{digest}",
                content_type=content_type,
                fetched_at=FETCHED_AT,
            )
        )
        return document, snapshot
