"""W10 batched orchestrator: page-based batch sizing, injected server control,
failure isolation, resume and degraded mode — all offline.

The "model server" is a fake controller recording its call sequence; the "LLM"
is a stub router; conversions run through injected fake T0/T1 callables whose
page counts are set per document. No GPU, no network, no Docling import.
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path
from typing import Any

import pytest
from test_orchestrate_support import (
    MemoryClaims,
    MemoryDocuments,
    MemoryEntities,
    MemoryExtractions,
    MemoryProgrammes,
    MemoryScores,
    MemorySnapshots,
    make_frontier,
)
from test_w10_support import (
    CountingT0,
    FakeParsesRepo,
    SnapshotHarness,
    StubRouter,
    stub_payload,
)

from fmd.acquire.canonical import PageGeometry, build_canonical
from fmd.acquire.parse_cache import ParseCache
from fmd.acquire.tiers import AcquirePolicy
from fmd.config import load_registry
from fmd.orchestrate import Orchestrator
from fmd.orchestrate.batched import (
    BatchedRunner,
    ScriptServerController,
    ServerControlError,
    resolve_script,
    take_batch,
)
from fmd.orchestrate.ledger import RunSummary, utcnow
from fmd.store import SnapshotStore

REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"


def _registry():
    return load_registry(REPO_CONFIG, env={})


def _pdf(marker: str) -> bytes:
    return b"%PDF-1.4\n% " + marker.encode("ascii") + b"\n"


def _doc_with_pages(pages: int):
    """The report document with ``pages`` pages of page geometry."""
    from test_w10_support import report_blocks, report_doc

    if pages <= 3:
        return report_doc()
    extra = tuple(
        PageGeometry(
            page_no=p,
            width=595.0,
            height=842.0,
            bbox=(0.0, 0.0, 595.0, 842.0),
            char_start=0,
            char_end=0,
        )
        for p in range(4, pages + 1)
    )
    canonical = build_canonical(list(report_blocks()), tuple(report_doc().pages) + extra)
    doc = report_doc()
    return _replace_doc(doc, canonical, pages)


def _replace_doc(doc, canonical, pages: int):
    from fmd.acquire.types import ExtractedDoc, PageSpan

    return ExtractedDoc(
        text=canonical.text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=doc.tables,
        tier=doc.tier,
        tool=doc.tool,
        tool_version=doc.tool_version,
        canonical=canonical,
        pages=canonical.pages,
        blocks=doc.blocks,
        bbox_granularity=doc.bbox_granularity,
        two_column=doc.two_column,
        table_confidence=doc.table_confidence,
    )


class PageCountT1:
    """Fake T1 whose page count comes from the document's marker bytes."""

    def __init__(self, pages_by_marker: dict[str, int], *, events: list[str] | None = None) -> None:
        self.pages_by_marker = pages_by_marker
        self.events = events if events is not None else []
        self.calls: list[str] = []

    def __call__(self, data: bytes) -> Any:
        for marker in self.pages_by_marker:
            if marker.encode("ascii") in data:
                self.calls.append(marker)
                self.events.append(f"convert:{marker}")
                return _doc_with_pages(self.pages_by_marker[marker])
        raise AssertionError(f"no page count configured for {data[:40]!r}")


class RecordingRouter(StubRouter):
    """StubRouter that records call order into the shared event list."""

    def __init__(self, models, payload, *, events: list[str] | None = None) -> None:
        super().__init__(models, payload)
        self.events = events if events is not None else []

    def call(self, purpose, schema, system, user, input_ref):
        self.events.append("llm_call")
        return super().call(purpose, schema, system, user, input_ref)


class FakeController:
    """Records the control sequence; extraction readiness is a flag."""

    def __init__(self, *, events: list[str], fail_start: bool = False) -> None:
        self.events = events
        self.fail_start = fail_start
        self.stopped = True
        self.starts = 0

    def stop(self) -> None:
        self.events.append("stop")
        self.stopped = True

    def start(self) -> None:
        self.events.append("start")
        self.starts += 1
        if self.fail_start:
            raise ServerControlError("simulated server start failure")
        self.stopped = False

    def wait_ready(self, timeout_s: float) -> None:
        self.events.append("wait_ready")
        if self.fail_start:
            raise ServerControlError("server not ready (fake)")
        self.stopped = False


class BatchedHarness:
    """Full orchestrator + runner with recorded events and per-doc pages."""

    def __init__(self, tmp_path: Path, registry, *, batch_pages: int = 100) -> None:
        self.registry = registry
        self.store = SnapshotStore(tmp_path / "snapshots")
        self.documents = MemoryDocuments()
        self.extractions = MemoryExtractions()
        self.claims = MemoryClaims()
        self.entities = MemoryEntities()
        self.snapshots = MemorySnapshots(self.extractions, self.documents)
        self.parse_repo = FakeParsesRepo()
        self.parse_cache = ParseCache(self.parse_repo, self.store)
        self.events: list[str] = []
        self.router = RecordingRouter(registry.models, stub_payload(), events=self.events)
        self.t1_pages: dict[str, int] = {}
        self.t1 = PageCountT1(self.t1_pages, events=self.events)
        self.orchestrator = Orchestrator(
            registry=registry,
            database=None,
            store=self.store,
            frontier=make_frontier(),
            documents=self.documents,
            snapshots=self.snapshots,
            extractions=self.extractions,
            claims=self.claims,
            programmes=MemoryProgrammes(),
            scores=MemoryScores(),
            entities=self.entities,
            router=self.router,
            parse_cache=self.parse_cache,
            t0=CountingT0(),
            t1=self.t1,
        )
        self.controller = FakeController(events=self.events)
        self.lines: list[str] = []
        self.runner = BatchedRunner(
            orchestrator=self.orchestrator,
            controller=self.controller,
            batch_pages=batch_pages,
            device="cuda",
            t0=CountingT0(),
            t1=self.t1,
            line_sink=self.lines.append,
        )

    def add_document(self, url: str, marker: str, pages: int, *, store_bytes: bytes | None = None):
        self.t1_pages[marker] = pages
        content = store_bytes if store_bytes is not None else _pdf(marker)
        harness = SnapshotHarness.__new__(SnapshotHarness)
        harness.registry = self.registry
        harness.tmp_path = Path("/tmp/unused")
        harness.store = self.store
        harness.documents = self.documents
        harness.extractions = self.extractions
        harness.claims = self.claims
        harness.entities = self.entities
        harness.snapshots = self.snapshots
        harness.parse_repo = self.parse_repo
        harness.parse_cache = self.parse_cache
        harness.router = self.router
        from test_w10_support import FETCHED_AT

        from fmd.db.repositories import DocumentUpsert, SnapshotCreate

        digest = self.store.put(content)
        document = self.documents.upsert(
            DocumentUpsert(
                url=url,
                domain="college.ac.in",
                kind="report",
                source_id="fixture",
                extraction_family="placement_report",
                entity_id=42,
                plan_state="linked",
                role="evidence",
            )
        )
        snapshot = self.snapshots.create(
            SnapshotCreate(
                document_id=document.id,
                sha256=digest,
                bytes_path=f"objects/{digest[:2]}/{digest}",
                content_type="application/pdf",
                fetched_at=FETCHED_AT,
            )
        )
        return document, snapshot


def test_controller_call_order_stop_acquire_start_wait_extract_then_stop(tmp_path) -> None:
    """The injected controller drives exactly: stop → acquire → start →
    wait_ready → extract, and the server is stopped at the end (try/finally)."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/r.pdf", "r", 3)

    summary = harness.runner.run(limit=5)

    assert summary.failed is False
    assert harness.events == [
        "stop",
        "convert:r",
        "start",
        "wait_ready",
        "llm_call",  # bulk_extract
        "llm_call",  # the verification pass
        "stop",  # the try/finally stop that ends every cycle
        "stop",  # and the run-final stop
    ]
    assert harness.controller.stopped is True
    assert summary.counts["cycles"] == 1
    assert summary.counts["extracted"] == 1
    assert summary.counts["converted"] == 1
    # line_pages counts known pages too, even when no conversion ran
    assert "pages=3" in harness.lines[0]
    assert summary.counts["llm_calls"] >= 1
    assert harness.lines and harness.lines[0].startswith("[batch] cycle=1")
    assert "device=cuda" in harness.lines[0]
    assert "pages=3" in harness.lines[0]


def test_140_page_document_converts_in_full_and_extracts_in_same_cycle(tmp_path) -> None:
    """batch_pages=100: a 140-page document is taken (its size is unknowable
    before conversion), converted in full, extracted in the same cycle, and
    the cycle ends — the 40-page document behind it is never started there."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/big.pdf", "big", 140)
    harness.add_document("https://college.ac.in/next.pdf", "next", 40)

    summary = harness.runner.run(limit=5)

    assert summary.counts["cycles"] == 2
    # Cycle 1: the 140-page document alone, converted in full, then extracted.
    assert "convert:big" in harness.events
    first_extract = harness.events.index("wait_ready")
    assert harness.events.index("convert:big") < first_extract
    assert harness.events.index("llm_call") > first_extract
    assert "convert:next" not in harness.events[: harness.events.index("stop", 1)]
    # Cycle 2 exists for the 40-page document.
    assert harness.events.count("convert:next") == 1
    assert summary.counts["extracted"] == 2
    assert summary.counts["pages"] == 180
    # The 140-page document was never truncated: its parse has 140 pages.
    rows = harness.parse_repo.rows.values()
    assert sorted(row.pages for row in rows) == [40, 140]


def test_known_page_count_that_would_cross_is_never_started(tmp_path) -> None:
    """A cached 40-page document after 80 known converted pages is deferred
    whole to the next cycle — sizing never revisits a completed document."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/a.pdf", "a", 80)
    harness.add_document("https://college.ac.in/b.pdf", "b", 40)
    harness.add_document("https://college.ac.in/c.pdf", "c", 10)

    # Pre-acquire all three documents so every page count is known.
    from fmd.orchestrate.acquire import AcquisitionStage

    stage = AcquisitionStage(
        snapshots=harness.snapshots,
        documents=harness.documents,
        store=harness.store,
        parse_cache=harness.parse_cache,
        policy=AcquirePolicy.load(acquire=registry.acquire),
        t0=CountingT0(),
        t1=harness.t1,
    )
    stage.run(summary=RunSummary(run_id="s", stage="acquire", started_at=utcnow()), limit=10)
    assert harness.parse_repo.count() == 3

    harness.events.clear()
    summary = harness.runner.run(limit=5)

    # a(80) fits; b(40) would cross (120 > 100) → deferred; c(10) fits.
    # Cycle composition is asserted, not just page totals: the deferred document
    # is never started (documents=1 in the first cycle), and it does not get
    # skipped in favour of later work — b and c share the second cycle.
    cycles = []
    for line in harness.lines:
        documents = int(line.split("documents=")[1].split()[0])
        pages = int(line.split("pages=")[1].split()[0])
        cycles.append((documents, pages))
    assert cycles == [(1, 80), (2, 50)]
    assert summary.counts["cycles"] == 2
    assert summary.counts["cache_hits"] == 3  # one parse consumed per document


def test_434_page_document_runs_alone(tmp_path) -> None:
    """A document larger than batch_pages runs alone: taken first, converted
    in full, extracted, and no other document joins the cycle."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/annual.pdf", "annual", 434)
    harness.add_document("https://college.ac.in/small.pdf", "small", 3)

    summary = harness.runner.run(limit=5)

    assert summary.counts["cycles"] == 2
    first_cycle_end = harness.events.index("stop", 1)  # stop after cycle 1
    assert "convert:annual" in harness.events[:first_cycle_end]
    assert "convert:small" not in harness.events[:first_cycle_end]
    assert "llm_call" in harness.events[:first_cycle_end]
    assert [row.pages for row in harness.parse_repo.rows.values()].count(434) == 1
    assert summary.counts["extracted"] == 2


def test_conversion_failure_records_and_batch_continues_next_run_retries(tmp_path) -> None:
    """One document failing to convert is recorded; the batch continues; the
    next run retries it from scratch — no partial parse is ever read."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/broken.pdf", "broken", 5)
    harness.add_document("https://college.ac.in/fine.pdf", "fine", 3)

    class FailFirst(PageCountT1):
        def __init__(self, inner: PageCountT1) -> None:
            self.inner = inner
            self.failed = False
            self.events = inner.events
            self.calls: list[str] = []

        def __call__(self, data: bytes) -> Any:
            if not self.failed:
                self.failed = True
                self.calls.append("boom")
                raise RuntimeError("docling exploded mid-document")
            return self.inner(data)

    harness.runner.t1 = FailFirst(harness.t1)
    summary = harness.runner.run(limit=5)

    assert summary.counts["cycles"] == 1
    assert summary.counts["abstained"] == 1
    assert summary.counts["converted"] == 1
    assert summary.counts["extracted"] == 1  # the fine document still extracted
    assert harness.parse_repo.count() == 1  # nothing cached for the failure
    assert harness.controller.stopped is True
    outcomes = [row for row in summary.outcomes if row.get("outcome") in {"abstained", "extracted"}]
    assert {row["outcome"] for row in outcomes} == {"abstained", "extracted"}

    # Second run: the failed document is retried from scratch and succeeds.
    harness.runner.t1 = harness.t1
    harness.events.clear()
    summary2 = harness.runner.run(limit=5)
    assert summary2.counts["converted"] == 1
    assert summary2.counts["extracted"] == 1
    assert harness.parse_repo.count() == 2


def test_server_start_failure_aborts_cycle_and_leaves_server_stopped(tmp_path) -> None:
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/r.pdf", "r", 3)
    harness.controller.fail_start = True

    summary = harness.runner.run(limit=5)

    assert summary.failed is True
    assert any("server" in str(f.get("reason", "")).lower() for f in summary.failures)
    assert harness.controller.stopped is True  # left stopped, never left running
    assert summary.counts.get("extracted", 0) == 0


def test_llm_failure_is_isolated_and_server_still_stopped(tmp_path) -> None:
    """A document whose extraction (LLM) fails is recorded; the parse stays
    cached; the server is stopped at the end; the next run retries it."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/r.pdf", "r", 3)
    harness.router.fail_for = {"*"}

    summary = harness.runner.run(limit=5)

    assert summary.counts.get("extracted", 0) == 0
    assert summary.failed is True  # nothing extractable this run
    assert harness.parse_repo.count() == 1  # the parse survived
    assert harness.controller.stopped is True

    harness.router.fail_for = set()
    harness.events.clear()
    summary2 = harness.runner.run(limit=5)
    assert summary2.counts["cache_hits"] == 1  # zero conversion on the retry
    assert summary2.counts["extracted"] == 1


def test_sigint_stops_the_run_and_leaves_the_server_stopped(tmp_path) -> None:
    """The SIGINT handler sets the stop flag; the run ends between documents
    and the try/finally still stops the server."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/a.pdf", "a", 60)
    harness.add_document("https://college.ac.in/b.pdf", "b", 60)

    stop = threading.Event()

    # Trip the flag when the first document's conversion completes (Ctrl-C
    # arriving mid-cycle): the in-flight document finishes, the second one
    # never starts, the server is never started, and the finally stops it.
    class TripOnA(PageCountT1):
        def __init__(self, inner: PageCountT1) -> None:
            self.inner = inner
            self.events = inner.events
            self.calls: list[str] = []

        def __call__(self, data: bytes) -> Any:
            doc = self.inner(data)
            if self.inner.calls and self.inner.calls[-1] == "a":
                stop.set()
            return doc

    harness.runner.t1 = TripOnA(harness.t1)
    summary = harness.runner.run(limit=5, stop_event=stop)

    assert summary.counts["cycles"] == 1  # the second cycle never started
    assert summary.counts["converted"] == 1  # the in-flight document finished
    assert summary.counts.get("extracted", 0) == 0  # extraction never ran
    assert "convert:b" not in harness.events  # the second document never started
    assert "start" not in harness.events  # no server was started for nothing
    assert harness.controller.stopped is True  # the finally still stopped it
    # Persisted work survives the interrupt: the parse of "a" is in the cache.
    assert harness.parse_repo.count() == 1


def test_sigint_handler_is_installed_and_restored() -> None:
    """The runner's SIGINT path: the handler sets the event, then restore()
    puts the previous handler back."""
    from fmd.orchestrate.batched import _install_sigint

    stop = threading.Event()
    restore = _install_sigint(stop)
    try:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop.is_set()
    finally:
        restore()
    assert signal.getsignal(signal.SIGINT) is not stop  # previous handler back


def test_batch_dry_run_plan_writes_nothing_with_unreachable_model_url(
    tmp_path, monkeypatch
) -> None:
    """``--batched --dry-run`` prints the cycle plan, contacts no model URL
    (the router would raise), converts nothing and writes nothing."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.orchestrator.router = _UnreachableRouter()
    monkeypatch.setenv("FMD_LOCAL_LLM_URL", "http://127.0.0.1:9/v1")
    harness.add_document("https://college.ac.in/a.pdf", "a", 140)
    harness.add_document("https://college.ac.in/b.pdf", "b", 40)

    summary = harness.runner.run(limit=5, dry_run=True)

    assert summary.failed is False
    assert harness.controller.events == []  # no server control in a dry run
    assert harness.t1.calls == []  # no conversion
    assert harness.parse_repo.count() == 0  # nothing written
    assert harness.router.calls == []  # no model contact
    cycles = summary.metadata["cycles"]
    assert len(cycles) >= 1
    assert cycles[0]["expected_conversions"] == 2
    assert summary.counts["planned"] == 2


class _UnreachableRouter:
    """A router wired at an unreachable URL: any call is a test failure."""

    def __init__(self) -> None:
        self.models = None
        self.calls: list[Any] = []

    def call(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(args)
        raise AssertionError("dry run must not contact the model URL")


def test_resume_skips_completed_documents(tmp_path) -> None:
    """Re-running skips documents that already have a cached parse AND a
    completed extraction; interrupting a cycle loses only in-flight work."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/a.pdf", "a", 3)
    harness.add_document("https://college.ac.in/b.pdf", "b", 3)

    first = harness.runner.run(limit=5)
    assert first.counts["extracted"] == 2
    total_converts = len(harness.t1.calls)
    assert total_converts == 2

    harness.events.clear()
    harness.lines.clear()
    second = harness.runner.run(limit=5)

    assert second.counts.get("extracted", 0) == 0  # both already extracted
    assert second.counts.get("converted", 0) == 0
    assert len(harness.t1.calls) == total_converts  # zero new conversions
    assert second.counts["cycles"] == 0  # nothing left in the queue


def test_degraded_mode_falls_back_to_cpu_inline(tmp_path, monkeypatch) -> None:
    """``FMD_SERVER_CONTROL=off``: CPU acquisition + inline extraction, said
    out loud in the summary — the run still happens."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/a.pdf", "a", 3)
    monkeypatch.setenv("FMD_SERVER_CONTROL", "off")

    summary = harness.runner.run(limit=5)

    assert summary.metadata["server_control"] == "degraded"
    assert summary.counts["extracted"] == 1
    assert summary.failed is False
    # No server control was attempted (the conversion ran inline on CPU).
    assert not {"stop", "start", "wait_ready"} & set(harness.controller.events)


def test_degraded_mode_dry_run_plans_without_server(tmp_path, monkeypatch) -> None:
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/a.pdf", "a", 3)
    monkeypatch.setenv("FMD_SERVER_CONTROL", "off")

    summary = harness.runner.run(limit=5, dry_run=True)

    assert summary.metadata["server_control"] == "degraded"
    assert summary.counts["planned"] == 1
    assert harness.controller.events == []


def test_keep_server_leaves_it_running(tmp_path) -> None:
    """``--keep-server`` is the only exit path that leaves the server up."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/a.pdf", "a", 3)
    harness.runner.keep_server = True

    summary = harness.runner.run(limit=5)

    assert summary.counts["extracted"] == 1
    assert harness.controller.stopped is False  # left running by request
    assert summary.metadata["keep_server"] is True


def test_take_batch_sizing_rule() -> None:
    """Unit check of the sizing rule: cutoff at document boundaries only; the
    crossing document always finishes; known-size documents that would cross
    are never started; a larger-than-limit document runs alone."""

    def item(snapshot_id: int, known: int | None) -> Any:
        snapshot = type("S", (), {"id": snapshot_id, "sha256": f"{snapshot_id:064d}"})()
        return type(
            "W",
            (),
            {
                "snapshot": snapshot,
                "known_pages": known,
                "url": None,
                "cache_hit": known is not None,
            },
        )()

    # Known 140-page doc crosses and finishes alone; the 40-page doc defers.
    batch = take_batch([item(1, 140), item(2, 40)], batch_pages=100, pages_for=lambda sha: None)
    assert [i.snapshot.id for i in batch] == [1]

    # An UNCACHED document's size is unknowable before conversion: the dry plan
    # takes it (contributing nothing known) and keeps packing; the live cycle
    # re-evaluates after each conversion and stops the batch there.
    batch = take_batch([item(1, None), item(2, 40)], batch_pages=100, pages_for=lambda sha: None)
    assert [i.snapshot.id for i in batch] == [1, 2]

    # Known sizes pack until the limit; a document that would cross defers and
    # ENDS the batch — later documents never jump the queue (priority order).
    batch = take_batch(
        [item(1, 80), item(2, 40), item(3, 10)], batch_pages=100, pages_for=lambda sha: None
    )
    assert [i.snapshot.id for i in batch] == [1]

    # A 434-page document runs alone.
    batch = take_batch([item(1, 434), item(2, 3)], batch_pages=100, pages_for=lambda sha: None)
    assert [i.snapshot.id for i in batch] == [1]


def test_script_server_controller_health_poll_and_errors(tmp_path) -> None:
    """The default adapter shells out to the script and polls /health with a
    timeout; a missing script and a never-ready server are named errors."""
    script = tmp_path / "serve_fake.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n  stop) echo stopped ;;\n  start) echo started ;;\n'
        "  *) exit 3 ;;\nesac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    controller = ScriptServerController(script, base_url="http://127.0.0.1:9/v1")
    assert controller.health_url == "http://127.0.0.1:9/health"
    controller.stop()
    controller.start()
    with pytest.raises(Exception) as excinfo:
        controller.wait_ready(0.1)
    assert "not ready" in str(excinfo.value)
    assert "127.0.0.1:9/health" in str(excinfo.value)

    missing = ScriptServerController(tmp_path / "absent.sh")
    with pytest.raises(Exception) as excinfo2:
        missing.stop()
    assert "does not exist" in str(excinfo2.value)


def test_resolve_script_finds_repo_relative_path() -> None:
    resolved = resolve_script("scripts/serve_local.sh")
    assert resolved.is_file()
    assert resolved.name == "serve_local.sh"


class _StubT2Cache:
    """Runner-facing T2 tier cache stub: sizing + lookup, no store."""

    def __init__(self, rows: dict[str, tuple[int, Any]]) -> None:
        self.rows = rows

    def has_parse(self, sha, policy) -> bool:
        return sha in self.rows

    def pages_for(self, sha, policy):
        row = self.rows.get(sha)
        return row[0] if row else None

    def lookup(self, sha, policy):
        from fmd.acquire.parse_cache import CacheHit, CacheMiss

        row = self.rows.get(sha)
        if row is None:
            return None, CacheMiss(None, "no_cache_entry", "stub")
        return row[1], CacheHit(
            key_digest="stub" * 8,
            object_sha256="o" * 64,
            pages=row[0],
            parser="rapidocr",
            parser_version="3.10.0",
        )


def test_sizing_reads_t2_cache_rows(tmp_path) -> None:
    """Wave-3 sizing note: a T2-cached scan is known-pages to the batch sizer —
    it goes straight to the extract phase of its first cycle and no converter
    is ever invoked for it."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    _document, scan = harness.add_document(
        "https://college.ac.in/scan.pdf", "scan-doc", 3
    )
    del harness.t1_pages["scan-doc"]  # a live conversion would raise: must not run
    from dataclasses import replace as _replace

    from test_w10_support import report_doc

    t2doc = _replace(
        report_doc(), tier="T2", tool="rapidocr", tool_version="3.10.0"
    )
    harness.orchestrator._t2_parse_cache_instance = _StubT2Cache(
        {scan.sha256: (3, t2doc)}
    )

    summary = harness.runner.run(limit=5)

    assert summary.failed is False
    assert summary.counts["extracted"] == 1
    assert summary.counts.get("converted", 0) == 0
    assert "convert:scan-doc" not in harness.events, (
        "a T2-cached scan must not be converted in the acquire phase"
    )
    # line_pages counts known pages too, even when no conversion ran
    assert "pages=3" in harness.lines[0]
