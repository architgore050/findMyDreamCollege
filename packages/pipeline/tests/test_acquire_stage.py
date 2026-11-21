"""W10 acquisition stage (``fmd acquire``): zero LLM calls, cache writes,
page budgets, dry-run plans and failure isolation.

Offline: the router is a raiser (any call fails the test), conversions run
through injected fake T0/T1 callables, the repo is in memory.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_w10_support import (
    CountingT0,
    CountingT1,
    RaisingRouter,
    SnapshotHarness,
)

from fmd.acquire.errors import AcquisitionFailed
from fmd.acquire.tiers import AcquirePolicy
from fmd.config import load_registry
from fmd.orchestrate.acquire import AcquisitionStage
from fmd.orchestrate.ledger import RunSummary, utcnow

REPO_CONFIG = __import__("pathlib").Path(__file__).resolve().parents[3] / "config"


def _registry():
    return load_registry(REPO_CONFIG, env={})


def _policy(registry):
    return AcquirePolicy.load(acquire=registry.acquire)


def _summary(stage: str = "acquire") -> RunSummary:
    return RunSummary(run_id="test", stage=stage, started_at=utcnow())


def _stage(harness: SnapshotHarness, registry, **kwargs) -> AcquisitionStage:
    defaults: dict[str, Any] = {
        "snapshots": harness.snapshots,
        "documents": harness.documents,
        "store": harness.store,
        "parse_cache": harness.parse_cache,
        "policy": _policy(registry),
        "t0": CountingT0(),
        "t1": CountingT1(),
    }
    defaults.update(kwargs)
    return AcquisitionStage(**defaults)


def _harness(tmp_path, registry, router) -> SnapshotHarness:
    return SnapshotHarness(tmp_path / "harness", registry, router=router)


def _pdf(marker: str) -> bytes:
    """Distinct PDF bytes per document: the cache is content-addressed, so
    identical bytes share one parse (by design) and would collide in tests."""
    return b"%PDF-1.4\n% " + marker.encode("ascii") + b"\n"


def test_acquire_makes_zero_llm_calls(tmp_path) -> None:
    """``fmd acquire`` never touches the router: a raiser proves it."""
    registry = _registry()
    router = RaisingRouter(registry.models)
    harness = _harness(tmp_path, registry, router)
    harness.add_pdf_document("https://college.ac.in/report.pdf")
    stage = _stage(harness, registry)

    summary = _summary()
    stage.run(summary=summary, limit=5)

    assert router.calls == [], "acquire must make zero LLM calls"
    assert summary.counts["converted"] == 1
    assert summary.counts["parsed"] == 1
    assert summary.counts["pages"] == 3
    assert summary.metadata["zero_llm_calls"] is True
    assert summary.failed is False


def test_acquire_writes_cache_extract_consumes_it(tmp_path) -> None:
    """acquire converts once; extract --from-cache then needs no conversion."""
    registry = _registry()
    t1 = CountingT1()
    harness = _harness(tmp_path, registry, RaisingRouter())
    harness.add_pdf_document("https://college.ac.in/report.pdf")
    stage = _stage(harness, registry, t1=t1)

    acquire_summary = _summary()
    stage.run(summary=acquire_summary, limit=5)
    assert t1.calls == 1
    assert harness.parse_repo.count() == 1

    # The extractor in cache-only mode reuses the parse: still one conversion.
    from fmd.orchestrate.extract import Extractor

    extractor = Extractor(
        snapshots=harness.snapshots,
        documents=harness.documents,
        extractions=harness.extractions,
        claims=harness.claims,
        entities=harness.entities,
        store=harness.store,
        router=StubPayloadRouter(registry),
        tool_version="1.0.0",
        parse_cache=harness.parse_cache,
        t0=CountingT0(),
        t1=t1,
        allow_convert=False,
    )
    summary = _summary("extract")
    extractor.run(summary=summary, limit=5)
    assert t1.calls == 1  # unchanged — zero Docling on the extract side
    assert summary.counts["cache_hits"] == 1
    assert summary.counts["extracted"] == 1
    assert summary.counts["llm_calls"] >= 1
    assert "missing_parse_cache" not in summary.counts


class StubPayloadRouter:
    """Minimal router for the extract side of the stage tests."""

    def __init__(self, registry) -> None:
        from test_w10_support import stub_payload

        self.models = registry.models
        self.payload = stub_payload()
        self.calls = 0

    def call(self, purpose, schema, system, user, input_ref):
        self.calls += 1
        return schema.model_validate(self.payload)


def _orchestrator(harness: SnapshotHarness, registry, router):
    """A production Orchestrator over the harness's memory repositories."""
    from test_orchestrate_support import (
        MemoryProgrammes,
        MemoryScores,
        make_frontier,
    )

    from fmd.orchestrate import Orchestrator

    return Orchestrator(
        registry=registry,
        database=None,
        store=harness.store,
        frontier=make_frontier(),
        documents=harness.documents,
        snapshots=harness.snapshots,
        extractions=harness.extractions,
        claims=harness.claims,
        programmes=MemoryProgrammes(),
        scores=MemoryScores(),
        entities=harness.entities,
        router=router,
        parse_cache=harness.parse_cache,
        t0=CountingT0(),
        t1=CountingT1(),
    )


def test_no_acquire_with_missing_parse_records_reason_and_continues(tmp_path) -> None:
    """``--no-acquire``: missing parses are recorded; exit non-zero only when
    nothing at all was extractable (the orchestrator's exit rule)."""
    registry = _registry()
    t1 = CountingT1()
    harness = _harness(tmp_path, registry, StubPayloadRouter(registry))
    # Two documents, neither acquired: both recorded as missing_parse_cache.
    harness.add_pdf_document("https://college.ac.in/a.pdf", content=_pdf("a"))
    harness.add_pdf_document("https://college.ac.in/b.pdf", content=_pdf("b"))
    orchestrator = _orchestrator(harness, registry, StubPayloadRouter(registry))

    summary = orchestrator.extract(no_acquire=True, limit=5)

    assert t1.calls == 0  # no conversion happened at all
    assert summary.counts["missing_parse_cache"] == 2
    assert summary.counts.get("extracted", 0) == 0
    assert summary.failed is True  # nothing was extractable
    assert summary.metadata["mode"] == "no_acquire"

    # Acquire one document by the stage; the other still has no parse.
    stage = _stage(harness, registry, t1=t1)
    stage.run(summary=_summary(), selected=[harness.snapshots.rows[0]])
    summary2 = orchestrator.extract(no_acquire=True, limit=5)
    assert summary2.counts["missing_parse_cache"] == 1
    assert summary2.counts["extracted"] == 1
    assert summary2.failed is False  # something was extractable


def test_page_budget_crosses_at_document_boundaries(tmp_path) -> None:
    """``--pages`` stops taking new documents once the converted total reaches
    the budget; the document that crossed it was already finished."""
    registry = _registry()
    t1 = CountingT1()
    harness = _harness(tmp_path, registry, RaisingRouter())
    harness.add_pdf_document("https://college.ac.in/a.pdf", content=_pdf("a"))
    harness.add_pdf_document("https://college.ac.in/b.pdf", content=_pdf("b"))
    stage = _stage(harness, registry, t1=t1)

    # Each fake report is 3 pages: a budget of 3 takes exactly one document.
    summary = _summary()
    stage.run(summary=summary, limit=5, pages_budget=3)
    assert summary.counts["converted"] == 1
    assert summary.counts["deferred"] == 1
    assert t1.calls == 1

    # The next run (no budget) converts the deferred document from scratch.
    summary2 = _summary()
    stage.run(summary=summary2, limit=5)
    assert summary2.counts["cache_hits"] == 1
    assert summary2.counts["converted"] == 1
    assert t1.calls == 2
    assert harness.parse_repo.count() == 2


def test_acquire_dry_run_prints_plan_and_writes_nothing(tmp_path) -> None:
    registry = _registry()
    t0 = CountingT0()
    t1 = CountingT1()
    harness = _harness(tmp_path, registry, RaisingRouter())
    harness.add_pdf_document("https://college.ac.in/a.pdf", content=_pdf("a"))
    harness.add_pdf_document("https://college.ac.in/b.pdf", content=_pdf("b"))
    stage = _stage(harness, registry, t0=t0, t1=t1)

    summary = _summary()
    stage.run(summary=summary, limit=5, dry_run=True)

    assert t0.calls == 0 and t1.calls == 0  # no conversion in a dry run
    assert harness.parse_repo.count() == 0  # nothing written
    assert summary.counts["to_convert"] == 2
    assert summary.counts.get("cache_hits", 0) == 0
    outcomes = [row["outcome"] for row in summary.outcomes]
    assert outcomes == ["would_convert", "would_convert"]
    assert all(row["pages"] is None for row in summary.outcomes)


def test_conversion_failure_records_reason_and_run_continues(tmp_path) -> None:
    """A mid-conversion failure caches nothing for that document; the run
    continues with the rest, and a retry starts that document from scratch."""
    registry = _registry()
    t0 = CountingT0()
    harness = _harness(tmp_path, registry, RaisingRouter())
    harness.add_pdf_document("https://college.ac.in/broken.pdf", content=_pdf("broken"))
    harness.add_pdf_document("https://college.ac.in/fine.pdf", content=_pdf("fine"))
    t1 = CountingT1()
    t1.error = RuntimeError("docling exploded mid-document")
    t1.fail_on_call = 1  # first conversion fails, later ones succeed
    stage = _stage(harness, registry, t0=t0, t1=t1)

    summary = _summary()
    stage.run(summary=summary, limit=5)

    assert t1.calls == 2  # the second document still converted
    assert summary.counts["abstained"] == 1
    assert summary.counts["converted"] == 1
    assert summary.failed is False  # the run continued and produced a parse
    outcomes = {row["url"]: row["outcome"] for row in summary.outcomes}
    assert outcomes["https://college.ac.in/broken.pdf"] == "abstained"
    assert outcomes["https://college.ac.in/fine.pdf"] == "converted"
    assert harness.parse_repo.count() == 1  # nothing cached for the failure

    # Next run: the failed document is retried from scratch (no partial parse
    # is ever read) and now succeeds.
    summary2 = _summary()
    stage.run(summary=summary2, limit=5)
    assert summary2.counts["converted"] == 1
    assert summary2.counts["cache_hits"] == 1
    assert harness.parse_repo.count() == 2


def test_acquisition_failed_parks_document_and_fails_summary(tmp_path) -> None:
    """``AcquisitionFailed`` (device/budget abort) marks the document
    needs_review with the stable reason and fails the run summary."""
    registry = _registry()
    harness = _harness(tmp_path, registry, RaisingRouter())
    harness.add_pdf_document("https://college.ac.in/oom.pdf")

    def oom(data: bytes):
        raise AcquisitionFailed(
            "acquisition_failed:docling_cuda_oom", "CUDA OOM under the 26B's KV"
        )

    stage = _stage(harness, registry, t0=CountingT0(), t1=oom)
    summary = _summary()
    stage.run(summary=summary, limit=5)

    assert summary.failed is True
    assert summary.counts["failed"] == 1
    assert summary.failures[0]["reason"] == "acquisition_failed:docling_cuda_oom"
    document = harness.documents.get_by_url("https://college.ac.in/oom.pdf")
    assert document.plan_state == "needs_review"
    assert document.plan_reason == "acquisition_failed:docling_cuda_oom"
    assert harness.parse_repo.count() == 0  # nothing cached
    assert summary.outcomes[0]["outcome"] == "acquisition_failed"


def test_plan_reports_cache_status_without_touching_anything(tmp_path) -> None:
    registry = _registry()
    t1 = CountingT1()
    harness = _harness(tmp_path, registry, RaisingRouter())
    harness.add_pdf_document("https://college.ac.in/a.pdf")
    stage = _stage(harness, registry, t1=t1)

    items = stage.plan(limit=5)
    assert [item.cache_hit for item in items] == [False]
    assert items[0].pages is None  # pages unknown before conversion

    stage.run(summary=_summary(), limit=5)
    items = stage.plan(limit=5)
    assert [item.cache_hit for item in items] == [True]
    assert items[0].pages == 3  # known from the cached parse row


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_override_pins_fmd_docling_device(tmp_path, device, monkeypatch) -> None:
    registry = _registry()
    harness = _harness(tmp_path, registry, RaisingRouter())
    harness.add_pdf_document("https://college.ac.in/a.pdf")
    monkeypatch.delenv("FMD_DOCLING_DEVICE", raising=False)
    stage = _stage(harness, registry, device=device)

    import os

    from fmd.orchestrate.acquire import DEVICE_ENV

    stage.run(summary=_summary(), limit=5)
    assert os.environ.get(DEVICE_ENV) is None  # restored after the run
    assert summary_device(stage) == device


def summary_device(stage: AcquisitionStage) -> str:
    summary = _summary()
    stage.run(summary=summary, limit=5)
    assert summary.metadata["device"] == stage.device
    return stage.device


def test_parse_cache_is_never_a_run_failure(tmp_path) -> None:
    """A broken parse repo degrades acquisition to live conversion; the parse
    is still produced, the reason is recorded, nothing crashes."""
    registry = _registry()
    harness = _harness(tmp_path, registry, RaisingRouter())
    harness.add_pdf_document("https://college.ac.in/a.pdf")
    harness.parse_repo.fail_put = True
    t1 = CountingT1()
    stage = _stage(harness, registry, t1=t1)

    summary = _summary()
    stage.run(summary=summary, limit=5)

    assert t1.calls == 1
    assert summary.counts["converted"] == 1
    assert summary.counts["parsed"] == 1
    assert summary.counts["cache_store_skipped"] == 1
    assert summary.failed is False
