from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fmd.db.repositories import (
    ClaimRecord,
    ClaimsRepo,
    ExtractionRecord,
    ExtractionsRepo,
    LLMCallLog,
    SnapshotRecord,
    SnapshotsRepo,
)
from fmd.extract.pipeline import (
    ExtractPipeline,
    ExtractTarget,
    build_router,
    mock_models,
)
from fmd.extract.prompts import load_prompt, user_message
from fmd.extract.schemas import PlacementReport
from fmd.llm import ModelRouter, prompt_hash
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
