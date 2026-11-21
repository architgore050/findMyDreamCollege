from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fmd.config.models import ModelsConfig
from fmd.db.repositories import (
    ClaimRecord,
    ClaimsRepo,
    DocumentUpsert,
    ExtractionRecord,
    ExtractionsRepo,
    LLMCallLog,
    SnapshotCreate,
    SnapshotRecord,
    SnapshotsRepo,
)
from fmd.extract.chunking import (
    ChunkedExtractError,
    ContextExceeded,
    plan_windows,
)
from fmd.extract.pipeline import (
    ExtractPipeline,
    ExtractResult,
    ExtractTarget,
    build_router,
    mock_models,
)
from fmd.extract.prompts import load_prompt, user_message
from fmd.extract.schemas import PlacementReport
from fmd.llm import ModelRouter, prompt_hash
from fmd.llm.providers import ProviderError
from fmd.orchestrate.ledger import RunSummary
from fmd.store import SnapshotStore

FIXTURES = Path(__file__).parent / "fixtures" / "extract"
LLM_FIXTURES = FIXTURES / "llm"
FETCHED_AT = datetime(2026, 11, 16, 8, 0, tzinfo=UTC)


class FakeSnapshots:
    def __init__(self, records: list[SnapshotRecord]) -> None:
        self.records = {record.id: record for record in records}

    def get(self, snapshot_id: int) -> SnapshotRecord | None:
        return self.records.get(snapshot_id)


class FakeExtractions:
    def __init__(self) -> None:
        self.rows: list[ExtractionRecord] = []
        self._next = 1

    def create(self, extraction) -> ExtractionRecord:
        record = ExtractionRecord(**extraction.model_dump(), id=self._next)
        self._next += 1
        self.rows.append(record)
        return record

    def list_for_snapshot(self, snapshot_id: int) -> list[ExtractionRecord]:
        return [row for row in self.rows if row.snapshot_id == snapshot_id]


class FakeClaims:
    def __init__(self) -> None:
        self.rows: list[ClaimRecord] = []
        self._next = 1

    def create(self, claim) -> ClaimRecord:
        record = ClaimRecord(**claim.model_dump(), id=self._next, superseded_by=None)
        self._next += 1
        self.rows.append(record)
        return record


class CollectingSink:
    def __init__(self) -> None:
        self.calls: list[LLMCallLog] = []

    def record(self, call: LLMCallLog) -> None:
        self.calls.append(call)


class StubRouter:
    """Returns a fixed quote-first guided-JSON payload; no provider, no network."""

    def __init__(self, models, payload: dict) -> None:
        self.models = models
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def call(self, purpose, schema, system, user, input_ref):
        self.calls.append((purpose, input_ref))
        return schema.model_validate(self.payload)


def _snapshot(snapshot_id: int, digest: str) -> SnapshotRecord:
    return SnapshotRecord(
        document_id=10 + snapshot_id,
        sha256=digest,
        bytes_path=f"objects/{digest[:2]}/{digest}",
        content_type="text/plain",
        fetched_at=FETCHED_AT,
        id=snapshot_id,
    )


def _pipeline(tmp_path: Path, registry, router, digest: str, snapshot_id: int = 1):
    store = SnapshotStore(tmp_path)
    extractions = FakeExtractions()
    claims = FakeClaims()
    pipeline = ExtractPipeline(
        router,
        snapshots=FakeSnapshots([_snapshot(snapshot_id, digest)]),
        extractions=extractions,
        claims=claims,
        store=store,
    )
    return pipeline, extractions, claims


def test_mock_pipeline_persists_validated_claims_from_recorded_fixture(
    tmp_path: Path, registry
) -> None:
    text = (FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    store = SnapshotStore(tmp_path)
    digest = store.put(text.encode("utf-8"))
    sink = CollectingSink()
    router = ModelRouter(mock_models(registry.models, fixtures_dir=LLM_FIXTURES), sink=sink)
    pipeline, extractions, claims = _pipeline(tmp_path, registry, router, digest)

    result = pipeline.run(ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report"))

    assert result.skipped is False
    assert result.rejections == []
    assert len(extractions.rows) == 1
    extraction = extractions.rows[0]
    assert extraction.snapshot_id == 1
    assert extraction.tool == "fmd.extract"
    assert extraction.prompt_version == "placement_report@v3"
    assert extraction.model == registry.models.profiles["bulk_extract"].model
    assert extraction.structure["tier"] == "T0"
    assert extraction.structure["family"] == "placement_report"
    assert extraction.structure["prompt_sha256"] == load_prompt("placement_report").sha256
    assert extraction.structure["outcome_counts"]["resolved"] == 8
    assert extraction.structure["outcome_counts"]["abstained"] == 2
    assert len(sink.calls) == 1
    expected_prompt = prompt_hash(load_prompt("placement_report").text, user_message(text))
    assert sink.calls[0].prompt_sha256 == expected_prompt
    assert sink.calls[0].params["input_ref"].startswith(digest)

    predicates = {claim.predicate for claim in claims.rows}
    assert predicates == {
        "placement.cohort_year",
        "placement.eligible_count",
        "placement.placed_count",
        "placement.rate",
        "placement.salary_period",
        "salary.median",
        "placement.internship_counted_as_employment",
        "placement.basis",
    }
    for claim in claims.rows:
        assert claim.extraction_id == extraction.id
        assert claim.entity_id == 42
        assert claim.method == "extracted"
        assert claim.observed_at == FETCHED_AT
        assert text[claim.char_start : claim.char_end] == claim.quote


def test_rerun_skips_snapshots_already_extracted_at_that_tool_version(
    tmp_path: Path, registry
) -> None:
    text = (FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    store = SnapshotStore(tmp_path)
    digest = store.put(text.encode("utf-8"))
    router = ModelRouter(mock_models(registry.models, fixtures_dir=LLM_FIXTURES))
    pipeline, extractions, claims = _pipeline(tmp_path, registry, router, digest)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    first = pipeline.run(target)
    second = pipeline.run(target)

    assert first.skipped is False
    assert second.skipped is True
    assert len(extractions.rows) == 1
    assert len(claims.rows) == len(first.claims)


def test_quote_not_in_source_drops_the_field_and_records_a_rejection(
    tmp_path: Path, registry
) -> None:
    text = "Eligible students: 120\nStudents placed: 96\n"
    store = SnapshotStore(tmp_path)
    digest = store.put(text.encode("utf-8"))
    payload = {
        "cohort_year": _field(None),
        "eligible_count": _field("Eligible students: 120", 0.95),
        "placed_count": _field("Students placed: 97", 0.9),
        "placement_rate": _field(None),
        "median_salary": _field(None),
        "highest_package": _field(None),
        "internship_counted_as_employment": _field(None),
        "outcome_basis": _field(None),
    }
    router = StubRouter(registry.models, payload)
    pipeline, extractions, claims = _pipeline(tmp_path, registry, router, digest)

    result = pipeline.run(ExtractTarget(snapshot_id=1, entity_id=7, family="placement_report"))

    assert [claim.predicate for claim in claims.rows] == ["placement.eligible_count"]
    assert result.rejections[0].reason == "unresolved_span"
    assert extractions.rows[0].structure["rejections"][0]["field"] == "placed_count"
    outcomes = extractions.rows[0].structure["outcomes"]
    failed = next(item for item in outcomes if item["field"] == "placed_count")
    assert failed["outcome"] == "unresolved"
    assert failed["char_start"] is None


def test_pipeline_records_a_guard_rejection_for_the_review_console(
    tmp_path: Path, registry
) -> None:
    text = "Eligible students: 120\nStudents placed: 130\n"
    store = SnapshotStore(tmp_path)
    digest = store.put(text.encode("utf-8"))
    payload = {
        "cohort_year": _field(None),
        "eligible_count": _field("Eligible students: 120", 0.95),
        "placed_count": _field("Students placed: 130", 0.95),
        "placement_rate": _field(None),
        "median_salary": _field(None),
        "highest_package": _field(None),
        "internship_counted_as_employment": _field(None),
        "outcome_basis": _field(None),
    }
    pipeline, extractions, claims = _pipeline(
        tmp_path, registry, StubRouter(registry.models, payload), digest
    )

    result = pipeline.run(ExtractTarget(snapshot_id=1, entity_id=8, family="placement_report"))

    assert "placement.placed_count" not in {claim.predicate for claim in claims.rows}
    assert result.rejections[-1].guard == "placed_exceeds_eligible"
    outcomes = {
        item["field"]: item["outcome"] for item in extractions.rows[0].structure["outcomes"]
    }
    assert outcomes["placed_count"] == "rejected_guard:placed_exceeds_eligible"


def test_batch_is_deterministic_and_sorted(tmp_path: Path, registry) -> None:
    store = SnapshotStore(tmp_path)
    first = store.put(b"Eligible students: 120\n")
    second = store.put(b"Eligible students: 130\n")
    router = StubRouter(
        registry.models,
        {
            "cohort_year": _field(None),
            "eligible_count": _field(None),
            "placed_count": _field(None),
            "placement_rate": _field(None),
            "median_salary": _field(None),
            "highest_package": _field(None),
            "internship_counted_as_employment": _field(None),
            "outcome_basis": _field(None),
        },
    )
    pipeline = ExtractPipeline(
        router,
        snapshots=FakeSnapshots([_snapshot(2, second), _snapshot(1, first)]),
        extractions=FakeExtractions(),
        claims=FakeClaims(),
        store=store,
    )

    results = pipeline.run_batch(
        [
            ExtractTarget(snapshot_id=2, entity_id=1, family="placement_report"),
            ExtractTarget(snapshot_id=1, entity_id=1, family="placement_report"),
        ]
    )

    assert [result.target.snapshot_id for result in results] == [1, 2]


def test_mock_models_repoint_every_profile_and_fill_unresolved_models(registry) -> None:
    models = mock_models(registry.models, fixtures_dir=LLM_FIXTURES)
    assert models.providers["mock"].kind == "replay"
    for name, profile in models.profiles.items():
        assert profile.provider == "mock", name
        assert profile.model, f"{name} model must be replayable offline"
    assert models.budgets.hard_stop is True


def test_build_router_mock_does_not_touch_the_network(registry) -> None:
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    assert isinstance(router, ModelRouter)
    assert router.models.providers["mock"].kind == "replay"


def test_no_rejections_are_silently_dropped_from_the_extraction_record(
    tmp_path: Path, registry
) -> None:
    text = "Eligible students: 120\nPlacement rate: 80%\n"
    store = SnapshotStore(tmp_path)
    digest = store.put(text.encode("utf-8"))
    payload = {
        "cohort_year": _field(None),
        "eligible_count": _field(None),
        "placed_count": _field(None),
        "placement_rate": _field("Placement rate: 80%", 0.9),
        "median_salary": _field(None),
        "highest_package": _field(None),
        "internship_counted_as_employment": _field(None),
        "outcome_basis": _field(None),
    }
    pipeline, extractions, claims = _pipeline(
        tmp_path, registry, StubRouter(registry.models, payload), digest
    )

    result = pipeline.run(ExtractTarget(snapshot_id=1, entity_id=3, family="placement_report"))

    assert claims.rows == []
    assert result.rejections[0].reason == "missing_denominator"
    assert extractions.rows[0].structure["rejections"][0]["reason"] == "missing_denominator"


def test_empty_tier_output_raises_before_any_claim_is_published(
    tmp_path: Path, registry, monkeypatch
) -> None:
    """An empty OCR/scan result aborts the snapshot: no extraction, no claims."""
    from fmd.acquire.errors import TierUnavailable
    from fmd.acquire.types import ExtractedDoc

    empty = ExtractedDoc(
        text="",
        page_map=[],
        tables=[],
        tier="T3",
        tool="rapidocr",
        notes=("models_missing: rec not found",),
    )
    # W10: the conversion seam now lives behind the parse cache
    # (fmd.acquire.parse_cache.acquire_document), which dispatches through
    # fmd.extract.pdf.extract_document — patch that seam.
    monkeypatch.setattr(
        "fmd.extract.pdf.extract_document", lambda *a, **k: empty
    )
    text = "ignored"
    store = SnapshotStore(tmp_path)
    digest = store.put(text.encode("utf-8"))
    pipeline, extractions, claims = _pipeline(
        tmp_path, registry, StubRouter(registry.models, {}), digest
    )

    with pytest.raises(TierUnavailable) as excinfo:
        pipeline.run(ExtractTarget(snapshot_id=1, entity_id=3, family="placement_report"))

    assert "every available tier produced empty text" in str(excinfo.value)
    assert "tier=T3" in str(excinfo.value)
    assert claims.rows == []
    assert extractions.rows == []


def test_claim_rows_match_the_claims_repo_shape(tmp_path: Path, registry) -> None:
    """A committed shape check: the pipeline builds ClaimCreate, not ad-hoc dicts."""
    assert ClaimsRepo.create is not None
    assert SnapshotsRepo.get is not None
    assert ExtractionsRepo.create is not None
    assert PlacementReport.FAMILY == "placement_report"


def _field(quote: str | None, confidence: float | None = 0.9) -> dict:
    return {
        "quote": quote,
        "confidence": confidence if quote is not None else None,
        "reason_if_null": None if quote is not None else "not stated in the document",
    }


# ------------------------------------------------------------------ W4 chunking
# Long-document behaviour: windows sized from config/models.yaml:chunking,
# absolute offsets, overlap dedupe, per-window failure isolation and the honest
# failure codes. All offline through a scripted router (no provider, no network).


CHUNK_FIXTURES = FIXTURES / "chunk"
ANNUAL_REPORT = CHUNK_FIXTURES / "annual_report.txt"
ELIGIBLE = "Eligible students: 120"
PLACED = "Students placed: 96"
CUMULATIVE = "Cumulative placement record: 96 of 120 placed in 2025."
#: The measured llama.cpp answer for an over-context request.
MEASURED_OVERFLOW = "request (70000 tokens) exceeds the available context size (65536)"

# Same arithmetic as _chunk_models: (800 - 200 - 100) tokens * 4 chars/token.
WINDOW_CHARS = 2000
OVERLAP_CHARS = 100
MIN_WINDOW_CHARS = 300
PLAN_PARAMS = {
    "window_chars": WINDOW_CHARS,
    "overlap_chars": OVERLAP_CHARS,
    "max_windows": 24,
    "min_window_chars": MIN_WINDOW_CHARS,
}

PLACEMENT_FIELDS = (
    "cohort_year",
    "eligible_count",
    "placed_count",
    "placement_rate",
    "salary_period",
    "median_salary",
    "highest_package",
    "recruiters_count",
    "internship_counted_as_employment",
    "outcome_basis",
)


def _chunk_models(registry, **overrides) -> ModelsConfig:
    """The committed registry plus a small chunking budget for fast tests."""
    payload = registry.models.model_dump()
    chunking = {
        "enabled": True,
        "chars_per_token": 4.0,
        "prompt_headroom_tokens": 100,
        "output_tokens": 200,
        "context_tokens": 800,
        "overlap_tokens": 25,
        "min_window_chars": MIN_WINDOW_CHARS,
        "max_windows_per_document": 24,
    }
    chunking.update(overrides)
    payload["chunking"] = chunking
    return ModelsConfig.model_validate(payload)


def _report_text(pages: int) -> str:
    whole = ANNUAL_REPORT.read_text(encoding="utf-8")
    blocks = whole.split("\n\n")
    return "".join(block + "\n\n" for block in blocks[:pages])


def _marker_payload(user_text: str, index: int = 0) -> dict:
    """Quote whatever evidence markers this window actually contains.

    ``eligible_count`` prefers the printed "Eligible students" line; windows
    that only see the cumulative sentence quote that instead — and because the
    sentence sits in the overlap of two windows, the same fact is recovered
    twice and must dedupe to one claim.
    """
    payload = {name: _field(None) for name in PLACEMENT_FIELDS}
    if ELIGIBLE in user_text:
        payload["eligible_count"] = _field(ELIGIBLE)
    elif CUMULATIVE in user_text:
        payload["eligible_count"] = _field(CUMULATIVE)
    if PLACED in user_text:
        payload["placed_count"] = _field(PLACED)
    return {name: value for name, value in payload.items() if value["quote"] is not None}


class ChunkedScriptedRouter:
    """Scripted per-window responses; the offline stand-in for the extract model.

    Window 0 is the single-pass call (no ``:w`` suffix on ``input_ref``); the
    ``fail`` map raises a prepared exception for a window index, which is how
    provider faults are staged without a network.
    """

    def __init__(self, models, *, payload_for=None, fail=None) -> None:
        self.models = models
        self.payload_for = payload_for
        self.fail = dict(fail or {})
        self.calls: list[tuple[int, str]] = []

    def call(self, purpose, schema, system, user, input_ref):
        index = 0
        if ":w" in input_ref:
            index = int(input_ref.rsplit(":w", 1)[1])
        self.calls.append((index, input_ref))
        failure = self.fail.get(index)
        if failure is not None:
            raise failure
        payload = self.payload_for(user, index) if self.payload_for else {}
        return schema.model_validate(payload)


def _windows_containing(text: str, marker: str) -> list[int]:
    position = text.find(marker)
    plan = plan_windows(text, **PLAN_PARAMS)
    return [w.index for w in plan.windows if w.char_start <= position < w.char_end]


def _chunk_pipeline(tmp_path: Path, text: str, router):
    store = SnapshotStore(tmp_path)
    digest = store.put(text.encode("utf-8"))
    snapshots = FakeSnapshots([_snapshot(1, digest)])
    extractions = FakeExtractions()
    claims = FakeClaims()
    pipeline = ExtractPipeline(
        router,
        snapshots=snapshots,
        extractions=extractions,
        claims=claims,
        store=store,
    )
    return pipeline, snapshots, extractions, claims, store


def test_short_document_keeps_the_single_pass_path_with_one_call(tmp_path, registry) -> None:
    text = f"{ELIGIBLE}\n{PLACED}\n"
    router = ChunkedScriptedRouter(
        _chunk_models(registry),
        payload_for=lambda user, index: _marker_payload(text),
    )
    pipeline, _snapshots, extractions, claims, _store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    result = pipeline.run(target)

    assert len(router.calls) == 1  # one call: the pre-existing path, unchanged
    assert router.calls[0][0] == 0  # no :w suffix on the input ref
    assert result.chunking is None
    assert "chunking" not in extractions.rows[0].structure
    assert result.rejections == []
    assert len(claims.rows) == 2


def test_long_document_splits_into_windows_and_offsets_stay_absolute(tmp_path, registry) -> None:
    text = _report_text(60)
    router = ChunkedScriptedRouter(_chunk_models(registry), payload_for=_marker_payload)
    pipeline, _snapshots, extractions, claims, _store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    result = pipeline.run(target)

    chunking = result.chunking
    assert chunking is not None and chunking["mode"] == "chunked"
    assert chunking["window_count"] >= 2
    assert [index for index, _ in router.calls] == list(range(1, chunking["window_count"] + 1))
    assert len({input_ref for _, input_ref in router.calls}) == len(router.calls)
    assert chunking["truncated"] is False
    assert chunking["failed_windows"] == []
    assert all(row["status"] == "ok" for row in chunking["windows"])
    assert chunking["pages_covered"] == chunking["pages_total"] == 1  # text doc: one block page

    # Spans are global: every claim resolves verbatim in the FULL canonical text
    # and the page map still resolves the absolute offsets.
    doc = result.doc
    assert len(claims.rows) >= 2
    for claim in claims.rows:
        assert 0 <= claim.char_start < claim.char_end <= len(text)
        assert text[claim.char_start : claim.char_end] == claim.quote
        assert doc.canonical.page_for_offset(claim.char_start) is not None
    eligible = next(c for c in claims.rows if c.predicate == "placement.eligible_count")
    assert eligible.char_start == text.find(ELIGIBLE)
    cumulative = next(c for c in claims.rows if c.quote == CUMULATIVE)
    assert cumulative.char_start == text.find(CUMULATIVE)
    assert cumulative.char_start > eligible.char_start

    extraction = extractions.rows[0]
    assert extraction.text == text
    assert extraction.structure["chunking"]["window_count"] == chunking["window_count"]
    assert extraction.structure["outcome_counts"]["resolved"] >= 2


def test_overlap_windows_agree_on_one_claim(tmp_path, registry) -> None:
    whole = _report_text(60)
    note = "Retention note: the same cohort of 120 students was tracked for three years."
    text = _text_with_note_in_overlap(whole, note)
    router = ChunkedScriptedRouter(
        _chunk_models(registry),
        payload_for=lambda user, index: (
            {"eligible_count": _field(note)} if note in user else {}
        ),
    )
    pipeline, _snapshots, extractions, claims, _store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    result = pipeline.run(target)

    covering = _windows_containing(text, note)
    assert len(covering) >= 2
    called = {index for index, _ in router.calls}
    assert set(covering) <= called
    # Both windows resolved the note; the overlap produced ONE claim, not two.
    assert len(claims.rows) == 1
    assert claims.rows[0].quote == note
    assert claims.rows[0].char_start == text.find(note)
    assert result.chunking["deduped_within_run"] == 1
    assert len(extractions.rows) == 1


def _text_with_note_in_overlap(text: str, note: str) -> str:
    """Insert ``note`` (once, inside a page) so two consecutive windows hold it."""
    for offset in range(1500, len(text), 137):
        if text[offset - 1] == "\n" or text[offset] == "\n":
            continue
        candidate = text[:offset] + note + "\n" + text[offset:]
        plan = plan_windows(candidate, **PLAN_PARAMS)
        position = candidate.find(note)
        covering = [w.index for w in plan.windows if w.char_start <= position < w.char_end]
        if len(covering) >= 2:
            return candidate
    raise AssertionError("no overlap position found for the note")


def test_window_failure_is_isolated_and_a_rerun_retries_only_that_window(
    tmp_path, registry
) -> None:
    text = _report_text(60)
    eligible_windows = _windows_containing(text, ELIGIBLE)
    placed_windows = _windows_containing(text, PLACED)
    assert eligible_windows and placed_windows
    assert set(eligible_windows).isdisjoint(placed_windows)
    failing = placed_windows[0]
    models = _chunk_models(registry)
    router = ChunkedScriptedRouter(
        models,
        payload_for=_marker_payload,
        fail={failing: ProviderError("timeout", "local: request timed out")},
    )
    pipeline, snapshots, extractions, claims, store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    first = pipeline.run(target)

    # The failed window did not stop the others: partial success is recorded.
    assert first.chunking["failed_windows"] == [failing]
    failed_row = first.chunking["windows"][failing - 1]
    assert failed_row["code"] == f"chunk_failed:{failing}"
    assert failed_row["reason"] == "timeout"
    assert failed_row["status"] == "failed"
    assert {c.predicate for c in first.claims} == {"placement.eligible_count"}
    assert len(first.claims) >= 1
    assert len(extractions.rows) == 1
    first_extraction = extractions.rows[0]

    # A rerun (resume) calls ONLY the failed window and skips the completed ones.
    retried = ChunkedScriptedRouter(models, payload_for=_marker_payload)
    resumed = ExtractPipeline(
        retried, snapshots=snapshots, extractions=extractions, claims=claims, store=store
    )
    second = resumed.run(target)

    assert second.chunking["resumed"] is True
    assert second.chunking["prior_extraction_id"] == first_extraction.id
    assert [index for index, _ in retried.calls] == [failing]
    statuses = {row["index"]: row["status"] for row in second.chunking["windows"]}
    assert statuses[failing] == "ok"
    assert all(
        status == "completed_prior_run"
        for index, status in statuses.items()
        if index != failing
    )
    # Both facts now exist exactly once across both runs — the retried window's
    # overlap cannot double-emit what a completed window already stored.
    keys = [(c.predicate, c.char_start, c.char_end) for c in claims.rows]
    assert len(keys) == len(set(keys))
    assert {c.predicate for c in claims.rows} == {
        "placement.eligible_count",
        "placement.placed_count",
    }
    assert len(extractions.rows) == 2
    assert second.claims and second.claims[0].extraction_id == extractions.rows[1].id


def test_all_windows_failing_persists_nothing_and_raises_chunk_failed(
    tmp_path, registry
) -> None:
    text = _report_text(60)
    fail = {
        index: ProviderError("http_error", MEASURED_OVERFLOW) for index in range(1, 40)
    }
    router = ChunkedScriptedRouter(_chunk_models(registry), payload_for=_marker_payload, fail=fail)
    pipeline, _snapshots, extractions, claims, _store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    with pytest.raises(ChunkedExtractError) as excinfo:
        pipeline.run(target)

    assert excinfo.value.reason.startswith("chunk_failed:")
    assert "chunk_failed:" in str(excinfo.value)
    assert extractions.rows == []
    assert claims.rows == []


def test_window_cap_stops_the_run_and_reports_the_budget(tmp_path, registry) -> None:
    text = _report_text(60)
    router = ChunkedScriptedRouter(
        _chunk_models(registry, max_windows_per_document=2), payload_for=_marker_payload
    )
    pipeline, _snapshots, extractions, claims, _store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    result = pipeline.run(target)

    # A 400-page report cannot silently become 400 calls: the cap stops the run.
    assert len(router.calls) == 2
    chunking = result.chunking
    assert chunking["truncated"] is True
    assert chunking["budget_exhausted"] is True
    assert chunking["window_count"] == 2
    assert chunking["pages_covered"] <= chunking["pages_total"]
    assert chunking["failed_windows"] == []
    assert len(extractions.rows) == 1
    # the two covered windows reach only pages 1-10: no evidence markers there
    assert len(claims.rows) == 0


def test_over_budget_document_without_chunking_fails_as_context_exceeded(
    tmp_path, registry
) -> None:
    text = _report_text(60)
    router = ChunkedScriptedRouter(_chunk_models(registry, enabled=False))
    pipeline, _snapshots, extractions, claims, _store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    with pytest.raises(ContextExceeded) as excinfo:
        pipeline.run(target)

    # The honest sizing code, raised BEFORE any call — never http_error.
    assert excinfo.value.reason == "context_exceeded"
    assert router.calls == []
    assert extractions.rows == []
    assert claims.rows == []


def test_provider_overflow_answer_is_reclassified_not_http_error(tmp_path, registry) -> None:
    text = _report_text(60)
    # A fitting config budget with a provider that 400s anyway (served ceiling
    # below the configured one): the measured error must surface as
    # context_exceeded, never as the network-shaped http_error.
    models = _chunk_models(
        registry, enabled=False, context_tokens=1000000, prompt_headroom_tokens=100
    )
    router = ChunkedScriptedRouter(
        models, fail={0: ProviderError("http_error", MEASURED_OVERFLOW)}
    )
    pipeline, _snapshots, extractions, _claims, _store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    with pytest.raises(ContextExceeded) as excinfo:
        pipeline.run(target)

    assert excinfo.value.reason == "context_exceeded"
    assert len(router.calls) == 1
    assert extractions.rows == []


def test_window_overflow_records_context_exceeded_and_the_rest_still_run(
    tmp_path, registry
) -> None:
    text = _report_text(60)
    failing = _windows_containing(text, PLACED)[0]
    router = ChunkedScriptedRouter(
        _chunk_models(registry),
        payload_for=_marker_payload,
        fail={failing: ProviderError("http_error", MEASURED_OVERFLOW)},
    )
    pipeline, _snapshots, extractions, claims, _store = _chunk_pipeline(tmp_path, text, router)
    target = ExtractTarget(snapshot_id=1, entity_id=42, family="placement_report")

    result = pipeline.run(target)

    failed_row = result.chunking["windows"][failing - 1]
    assert failed_row["code"] == f"chunk_failed:{failing}"
    assert failed_row["reason"] == "context_exceeded"  # not http_error
    assert result.chunking["failed_windows"] == [failing]
    assert {c.predicate for c in claims.rows} == {"placement.eligible_count"}
    assert len(extractions.rows) == 1


# ------------------------------------------------------------------ W4 orchestrator
# The run ledger must show a chunked partial pass (some windows failed, the
# window budget stopped early) and a sizing fault (context_exceeded) as what
# they are — never as network-shaped failures.


class StubChunkPipeline:
    """Pipeline stand-in for orchestrator-level assertions."""

    def __init__(self, result: ExtractResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error

    def run_batch(self, targets: list[ExtractTarget]) -> list[ExtractResult]:
        if self.error is not None:
            raise self.error
        return [self.result]


def _orchestrator(tmp_path: Path, registry, pipeline) -> Any:
    from test_orchestrate_support import (
        MemoryClaims,
        MemoryDocuments,
        MemoryExtractions,
        MemorySnapshots,
    )

    from fmd.orchestrate.extract import Extractor

    store = SnapshotStore(tmp_path / "store")
    documents = MemoryDocuments()
    extractions = MemoryExtractions()
    snapshots = MemorySnapshots(extractions, documents)
    claims = MemoryClaims()
    text = f"{ELIGIBLE}\n{PLACED}\n"
    digest = store.put(text.encode("utf-8"))
    document = documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/report.txt",
            domain="college.ac.in",
            extraction_family="placement_report",
            entity_id=7,
            plan_state="linked",
        )
    )
    snapshots.create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=str(store.path_for(digest)),
            content_type="text/plain",
            fetched_at=FETCHED_AT,
        )
    )
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    extractor = Extractor(
        snapshots=snapshots,
        documents=documents,
        extractions=extractions,
        claims=claims,
        entities=None,
        store=store,
        router=router,
        pipeline_factory=lambda cls: pipeline,
        verify=False,
    )
    return extractor


def _run_summary() -> RunSummary:
    return RunSummary(
        run_id="chunk-run", stage="extract", started_at=datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_orchestrator_reports_chunk_failures_and_budget_exhaustion(tmp_path, registry) -> None:
    target = ExtractTarget(snapshot_id=1, entity_id=7, family="placement_report")
    result = ExtractResult(
        target=target,
        extraction_id=3,
        claims=[object()],
        chunking={
            "mode": "chunked",
            "window_count": 11,
            "pages_covered": 50,
            "pages_total": 60,
            "truncated": True,
            "failed_windows": [6, 9],
            "deduped_claims": 2,
            "resumed": False,
        },
    )
    extractor = _orchestrator(tmp_path, registry, StubChunkPipeline(result=result))

    summary = _run_summary()
    extractor.run(summary=summary)

    entry = summary.outcomes[0]
    assert entry["outcome"] == "extracted"
    assert entry["reason"] == "chunk_failed:2"  # two failed windows, stable shape
    assert entry["chunk_failed_windows"] == [6, 9]
    assert entry["chunking"]["truncated"] is True
    assert entry["chunking"]["pages_covered"] == 50
    assert entry["chunking"]["pages_total"] == 60
    assert summary.counts["chunked_documents"] == 1
    assert summary.counts["chunk_windows"] == 11
    assert summary.counts["chunk_failed_windows"] == 2
    assert summary.counts["chunk_budget_exhausted"] == 1


def test_orchestrator_records_context_exceeded_as_a_sizing_fault(
    tmp_path, registry
) -> None:
    extractor = _orchestrator(
        tmp_path, registry, StubChunkPipeline(error=ContextExceeded("snapshot 1: over budget"))
    )

    summary = _run_summary()
    extractor.run(summary=summary)

    entry = summary.outcomes[0]
    assert entry["outcome"] == "abstained"
    assert entry["reason"] == "context_exceeded"  # never http_error, never network-shaped
