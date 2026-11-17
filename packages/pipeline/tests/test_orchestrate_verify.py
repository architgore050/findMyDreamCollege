"""Extraction-run verification wiring: counts, honest degradation, budget state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from test_orchestrate_support import (
    MemoryClaims,
    MemoryDocuments,
    MemoryExtractions,
    MemorySnapshots,
)

from fmd.db.repositories import DocumentUpsert, SnapshotCreate
from fmd.extract.pipeline import build_router
from fmd.orchestrate.extract import Extractor
from fmd.orchestrate.ledger import RunSummary
from fmd.store import SnapshotStore
from fmd.verify.types import VerificationDecision, VerificationRun

TESTS = Path(__file__).parent
EXTRACT_FIXTURES = TESTS / "fixtures" / "extract"
LLM_FIXTURES = EXTRACT_FIXTURES / "llm"


class FakeVerifier:
    def __init__(self, run: VerificationRun) -> None:
        self.run = run
        self.started = 0
        self.prompt = type("Prompt", (), {"ref": "verify@v1"})()
        self.model = "recorded-verifier"

    def start_run(self) -> None:
        self.started += 1

    def verify_claims(self, claims: list[Any], *, text: str, persist: bool = True) -> Any:
        decisions = []
        for index, claim in enumerate(claims):
            status = "rejected" if index == 0 else "supported"
            decisions.append(
                VerificationDecision(
                    key=f"claim-{claim.id}",
                    claim_id=claim.id,
                    status=status,
                    verdict="absent" if status == "rejected" else "supported",
                    reason="synthetic",
                    prompt_version="verify@v1",
                    verified_by=self.model,
                )
            )
        return VerificationRun(decisions=decisions, calls=1, repairs=0, budget_exhausted=True)


def _extractor(
    registry, tmp_path: Path, *, verifier: Any = None, verify: bool = True
) -> tuple[
    Extractor,
    MemoryClaims,
    MemoryDocuments,
    MemorySnapshots,
    MemoryExtractions,
    SnapshotStore,
]:
    store = SnapshotStore(tmp_path / "snapshots")
    documents = MemoryDocuments()
    extractions = MemoryExtractions()
    snapshots = MemorySnapshots(extractions, documents)
    claims = MemoryClaims()
    router = build_router(registry.models, mock=True, fixtures_dir=LLM_FIXTURES)
    extractor = Extractor(
        snapshots=snapshots,
        documents=documents,
        extractions=extractions,
        claims=claims,
        entities=None,
        store=store,
        router=router,
        verifier=verifier,
        verify=verify,
    )
    return extractor, claims, documents, snapshots, extractions, store


def _seed(
    documents: MemoryDocuments, snapshots: MemorySnapshots, store: SnapshotStore
) -> None:
    text = (EXTRACT_FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    digest = store.put(text.encode("utf-8"))
    document = documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/placement.txt",
            domain="college.ac.in",
            extraction_family="placement_report",
            entity_id=1,
            plan_state="linked",
        )
    )
    snapshots.create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=str(store.path_for(digest)),
            content_type="text/plain",
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )


def _summary() -> RunSummary:
    return RunSummary(
        run_id="verify-run", stage="extract", started_at=datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_extract_summary_reports_verified_and_unverified_counts(registry, tmp_path) -> None:
    verifier = FakeVerifier(VerificationRun())
    extractor, claims, documents, snapshots, _extractions, store = _extractor(
        registry, tmp_path, verifier=verifier
    )
    _seed(documents, snapshots, store)
    summary = _summary()

    extractor.run(summary=summary)

    assert verifier.started == 1
    assert summary.counts["extracted"] == 1
    assert summary.counts["verified"] == len(claims.rows) - 1
    assert summary.counts["verification_rejected"] == 1
    assert summary.counts["unverified"] == 0
    assert summary.metadata["verifier"]["model"] == "recorded-verifier"
    assert summary.metadata["verifier"]["prompt_version"] == "verify@v1"
    assert summary.metadata["verifier"]["budget_exhausted"] is True
    assert summary.outcomes[0]["verified"] == len(claims.rows) - 1


def test_extract_can_skip_verification_explicitly(registry, tmp_path) -> None:
    extractor, _claims, documents, snapshots, _extractions, store = _extractor(
        registry, tmp_path, verify=False
    )
    _seed(documents, snapshots, store)
    summary = _summary()

    extractor.run(summary=summary)

    assert summary.counts["extracted"] == 1
    assert "verifier" not in summary.metadata
    assert summary.counts.get("verified", 0) == 0


def test_missing_verdict_fixtures_leave_claims_unverified_not_supported(
    registry, tmp_path
) -> None:
    """Offline replay without recorded verdicts must degrade honestly."""
    extractor, claims, documents, snapshots, _extractions, store = _extractor(
        registry, tmp_path
    )
    _seed(documents, snapshots, store)
    summary = _summary()

    extractor.run(summary=summary)

    assert summary.counts["extracted"] == 1
    assert summary.counts["verified"] == 0
    assert summary.counts["unverified"] == len(claims.rows)
    assert summary.metadata["verifier"]["error"] == "fixture_missing"
    assert all(row.verification_status == "unverified" for row in claims.rows)
