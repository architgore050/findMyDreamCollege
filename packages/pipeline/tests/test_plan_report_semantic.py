"""Wave v1-w5: the classification report and the semantic admission contract.

Offline for the pure pieces (report buckets, admission contract, apply);
DB-gated for ``plan_evidence_rows`` and the persist path (real repo, isolated
schema from conftest). The report never writes; the admission contract never
guesses: a model-proposed family is admitted only with a verbatim quoted
evidence string from the document's own text sample.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fmd.plan.report import (
    SEMANTIC_BUCKETS,
    build_report,
    render_report,
    replay_family,
)
from fmd.plan.semantic import (
    REF_AMBIGUOUS,
    REF_EMPTY_QUOTE,
    REF_NO_SAMPLE,
    REF_QUOTE_NOT_VERBATIM,
    REF_UNKNOWN_FAMILY,
    SemanticAdmission,
    apply_admissions,
    validate_admissions,
)

FIXTURES = Path(__file__).parent / "fixtures" / "plan"
FEE_SAMPLE = (FIXTURES / "fee_structure.txt").read_text(encoding="utf-8")
FEE_URL = "https://college.ac.in/admissions/fee-structure-2025.pdf"
SILENT_URL = "https://college.ac.in/sites/default/files/report-final-draft.pdf"


def _thresholds():
    planner = __import__("fmd.plan.planner", fromlist=["DocumentPlanner"]).DocumentPlanner(
        candidates_provider=lambda: []
    )
    thresholds = planner.thresholds
    return (
        thresholds.phase_b_min_score,
        thresholds.phase_b_margin,
        thresholds.phase_a_min_score,
        thresholds.phase_a_margin,
    )


def _report(rows, sample_for=None):
    pb_min, pb_margin, pa_min, pa_margin = _thresholds()
    return build_report(
        rows,
        phase_b_min_score=pb_min,
        phase_b_margin=pb_margin,
        phase_a_min_score=pa_min,
        phase_a_margin=pa_margin,
        sample_for=sample_for,
    )


def _row(url: str, **overrides) -> dict:
    row = {
        "id": 1,
        "url": url,
        "role": "evidence",
        "plan_state": "abstained",
        "extraction_family": None,
        "source_id": "src-1",
        "declared_family": None,
    }
    row.update(overrides)
    return row


def test_report_buckets_declared_family_and_family_set() -> None:
    report = _report(
        [
            _row("https://a.ac.in/x.pdf", extraction_family="fee_structure"),
            _row("https://b.ac.in/y.pdf", declared_family="seat_matrix"),
            _row("https://c.ac.in/nav", role="navigation"),
        ]
    )
    assert report["evidence_documents"] == 2  # navigation never counted
    assert report["declared"] == 2  # one persisted, one source-declared
    assert report["family_set_by_source"] == {"registry": 2}
    assert report["missing_family"] == {}
    assert report["semantic_candidates"] == 0


def test_report_buckets_fee_url_as_classifying_or_below_floor() -> None:
    """The fee URL carries token evidence: with no sample it either clears the
    phase-A floor (would_classify_from_url) or sits below it (below_floor) —
    both are deterministic, never no_signal."""
    report = _report([_row(FEE_URL)])
    missing = report["missing_family"]
    assert missing.get("no_signal") is None
    assert missing.get("would_classify_from_url", 0) + missing.get("below_floor", 0) == 1


def test_report_replays_text_and_finds_the_family() -> None:
    report = _report(
        [_row(FEE_URL), _row(SILENT_URL)],
        sample_for=lambda row: FEE_SAMPLE if row["url"] == FEE_URL else "",
    )
    assert report["missing_family"].get("would_classify_with_text") == 1
    families = report["families"]["fee_structure"]
    assert families.get("would_classify_with_text") == 1


def test_report_silent_url_is_a_semantic_candidate() -> None:
    report = _report([_row(SILENT_URL)])
    assert report["missing_family"].get("no_signal") == 1
    assert report["semantic_candidates"] == 1
    entry = report["candidate_rows"][0]
    assert entry["bucket"] in SEMANTIC_BUCKETS
    assert render_report(report).startswith("evidence_documents=1")


def test_replay_family_returns_signals_even_when_abstaining() -> None:
    pb_min, pb_margin, pa_min, pa_margin = _thresholds()
    family, score, reason, signals = replay_family(
        url=FEE_URL,
        sample=None,
        phase_b_min_score=pb_min,
        phase_b_margin=pb_margin,
        phase_a_min_score=pa_min,
        phase_a_margin=pa_margin,
    )
    assert reason is None or reason == "family_undetected" or score >= 0
    assert "structure_hints" in signals
    assert score >= 0.0


def test_admission_contract_requires_verbatim_quote() -> None:
    validated, refused = validate_admissions(
        [
            SemanticAdmission(
                document_url=FEE_URL,
                family="fee_structure",
                quote="Tuition fee: INR 100000 per annum",
            ),
            SemanticAdmission(
                document_url=FEE_URL, family="fee_structure", quote="this text is nowhere"
            ),
            SemanticAdmission(document_url=FEE_URL, family="fee_structure", quote="   "),
            SemanticAdmission(document_url=FEE_URL, family="not_a_family", quote="Tuition"),
        ],
        {FEE_URL: FEE_SAMPLE},
    )
    assert len(validated) == 1
    assert validated[0].quote_sha256  # evidence digest is recorded
    reasons = {refusal.reason for refusal in refused}
    assert reasons == {REF_QUOTE_NOT_VERBATIM, REF_EMPTY_QUOTE, REF_UNKNOWN_FAMILY}


def test_admission_contract_refuses_ambiguity_and_missing_samples() -> None:
    other_url = "https://college.ac.in/no-sample.pdf"
    validated, refused = validate_admissions(
        [
            SemanticAdmission(
                document_url=FEE_URL,
                family="fee_structure",
                quote="Tuition fee: INR 100000 per annum",
            ),
            SemanticAdmission(document_url=FEE_URL, family="placement_report", quote="Tuition"),
            SemanticAdmission(document_url=other_url, family="fee_structure", quote="Tuition"),
        ],
        {FEE_URL: FEE_SAMPLE},
    )
    assert validated == []  # two families for one document: all refused
    reasons = {refusal.reason for refusal in refused}
    assert REF_AMBIGUOUS in reasons
    assert REF_NO_SAMPLE in reasons


def test_apply_admissions_persists_defer_for_text_with_evidence_signals() -> None:
    validated, _ = validate_admissions(
        [
            SemanticAdmission(
                document_url=FEE_URL,
                family="fee_structure",
                quote="Tuition fee: INR 100000 per annum",
                score=0.9,
            )
        ],
        {FEE_URL: FEE_SAMPLE},
    )
    assert len(validated) == 1

    written: list[dict] = []

    class FakeDocuments:
        def get_by_url(self, url: str):
            if url == FEE_URL:
                return SimpleNamespace(id=7, source_id="src-1", url=url)
            return None

        def set_plan(self, document_id, **fields):
            written.append({"document_id": document_id, **fields})
            return fields

    outcome = apply_admissions(
        validated,
        documents=FakeDocuments(),
        planner=object(),
        sample_for=lambda url: FEE_SAMPLE,
    )
    assert outcome["applied"] == 1 and outcome["missing_document"] == 0
    assert written[0]["extraction_family"] == "fee_structure"
    assert written[0]["plan_state"] == "planned"
    assert written[0]["entity_id"] is None  # admission is a family decision only


# ------------------------------------------------------------------ DB-gated


def test_plan_evidence_rows_reads_seeded_documents(pg_database) -> None:
    from fmd.db.repositories import DocumentsRepo, DocumentUpsert

    documents = DocumentsRepo(pg_database)
    upserted = documents.upsert(
        DocumentUpsert(url="https://db.ac.in/fee-structure.pdf", domain="db.ac.in")
    )
    rows = documents.plan_evidence_rows(limit=50)
    assert any(row["url"] == "https://db.ac.in/fee-structure.pdf" for row in rows)
    row = next(r for r in rows if r["url"] == upserted.url)
    assert row["role"] in (None, "evidence")  # migration default is evidence
    assert row["extraction_family"] is None


def test_apply_admissions_persists_through_the_real_repo(pg_database) -> None:
    from fmd.db.repositories import DocumentsRepo, DocumentUpsert
    from fmd.plan.semantic import ValidatedAdmission

    documents = DocumentsRepo(pg_database)
    record = documents.upsert(
        DocumentUpsert(url="https://db2.ac.in/fee-structure.pdf", domain="db2.ac.in")
    )
    admission = ValidatedAdmission(
        document_url=record.url,
        family="fee_structure",
        quote="Tuition fee: INR 100000 per annum",
        quote_sha256="a" * 64,
        score=0.9,
    )
    outcome = apply_admissions(
        [admission], documents=documents, planner=object(), sample_for=lambda url: FEE_SAMPLE
    )
    assert outcome["applied"] == 1
    updated = documents.get_by_url(record.url)
    assert updated.extraction_family == "fee_structure"
    assert updated.plan_state == "planned"  # persisted form of defer_for_text
