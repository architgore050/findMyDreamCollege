"""W5 family coverage: the representative set of the 83 abstained documents.

The measured problem (2026-11-19): of 135 crawled snapshots, 83 abstained with
``family_undetected`` — predominantly annual reports, prospectuses and NIRF
submissions that *do* carry accreditation, placement and fee evidence. This
module pins the intended classification of every document type in that set at
the planner level (Phase A deferral → Phase B decision → persistence-ready
state), the negative corpus that must never classify, and the precedence
invariants that the new signals must not disturb.

Offline and deterministic: fixture text only, a fixed candidate universe, the
deterministic blocking backend, embeddings disabled.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fmd.er import EntityCandidate, ERConfig
from fmd.plan import DocumentPlanner, detect_family
from fmd.plan.roles import family_hints_from_structure
from fmd.plan.rules import score_families

FIXTURES = Path(__file__).parent / "fixtures" / "plan"
ENTITY = "Demo Institute of Health Sciences"
ENTITY_UNIVERSITY = "Demo University of Health Sciences"
NEUTRAL_URL = "https://example.ac.in/uploads/doc.pdf"


def _config() -> ERConfig:
    return ERConfig.from_mapping(
        {"blocking": {"backend": "deterministic"}, "similarity": {"use_embeddings": False}}
    )


def _planner(entity: str = ENTITY) -> DocumentPlanner:
    return DocumentPlanner(
        candidates_provider=lambda: [EntityCandidate(entity_id=7, canonical_name=entity)],
        er_config=_config(),
    )


def _document(url: str = NEUTRAL_URL) -> SimpleNamespace:
    return SimpleNamespace(id=1, url=url, source_id=None, extraction_family=None, entity_id=None)


def _text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ------------------------------------------------- the representative positives
# Each row is a document shape measured in the 83-abstained corpus.


@pytest.mark.parametrize(
    ("fixture", "url", "expected"),
    [
        # Annual reports → accreditation, placement sections recorded as a hint.
        (
            "annual_report.txt",
            "https://example.ac.in/sites/Annual-Report-2023-24.pdf",
            "accreditation",
        ),
        # Prospectus chains resolve by their own tables.
        (
            "prospectus_fees.txt",
            "https://example.ac.in/admissions/Prospectus-2025-26.pdf",
            "fee_structure",
        ),
        (
            "prospectus_intake.txt",
            "https://example.ac.in/admissions/prospectus-2025.pdf",
            "seat_matrix",
        ),
        (
            "prospectus_general.txt",
            "https://example.ac.in/admissions/prospectus-2025.pdf",
            "curriculum",
        ),
        # NIRF / AQAR / SSR / IQAC submissions → accreditation.
        ("nirf_submission.txt", "https://example.ac.in/sites/NIRF-2024.pdf", "accreditation"),
        ("aqar_report.txt", "https://example.ac.in/iqac/AQAR-2023-24.pdf", "accreditation"),
        # Fee circulars / fee structures → fee_structure.
        (
            "fee_circular.txt",
            "https://example.ac.in/uploads/Fee-Circular-03-2025.pdf",
            "fee_structure",
        ),
        # Seat matrix / intake statements → seat_matrix.
        (
            "seat_intake_statement.txt",
            "https://example.ac.in/uploads/Seat-Matrix-2025-26.pdf",
            "seat_matrix",
        ),
    ],
)
def test_abstained_corpus_now_classifies(fixture: str, url: str, expected: str) -> None:
    """Acceptance 5: every representative shape classifies; the plan is linked.

    Every fixture in this table names its institution on the cover, because a
    plan is only extractable when the entity also resolves. Bare-detection
    coverage for fixtures without a cover line lives in ``test_plan.py``.
    """
    plan = _planner(ENTITY_UNIVERSITY if "university" in fixture else ENTITY).plan(
        _document(url), text_sample=_text(fixture), phase="b"
    )

    assert plan.family == expected
    assert plan.family_source == "heuristic"
    assert plan.state == "linked"
    assert plan.extractable is True
    assert plan.abstain_reason is None


def test_annual_report_carries_the_placement_hint_in_the_plan() -> None:
    plan = _planner().plan(
        _document("https://example.ac.in/sites/Annual-Report-2023-24.pdf"),
        text_sample=_text("annual_report.txt"),
        phase="b",
    )

    assert plan.family == "accreditation"
    assert plan.signals["secondary_hint"] == 1
    assert plan.signals["secondary_hint_family_score"] > 0


def test_annual_report_without_placement_sections_has_no_hint_in_the_plan() -> None:
    plan = _planner().plan(
        _document("https://example.ac.in/sites/Annual-Report-2022-23.pdf"),
        text_sample=_text("annual_report_no_placement.txt"),
        phase="b",
    )

    assert plan.family == "accreditation"
    assert "secondary_hint" not in plan.signals


def test_exam_registration_notification_is_extractable_not_a_schedule() -> None:
    plan = _planner(ENTITY_UNIVERSITY).plan(
        _document("https://example.ac.in/uploads/exam-registration-2025.pdf"),
        text_sample=_text("exam_registration_notification.txt"),
        phase="b",
    )

    assert plan.family == "exam_notification"
    assert plan.state == "linked"


# ------------------------------------------------------------- negative corpus


@pytest.mark.parametrize(
    ("fixture", "url"),
    [
        ("staff_circular.txt", "https://example.ac.in/uploads/office-order-17119.pdf"),
        ("tender_notice.txt", "https://example.ac.in/tenders/Tender-Notice-07-2025.pdf"),
        ("event_news_page.txt", "https://example.ac.in/events/annual-day-2025.pdf"),
        ("convocation_notice.txt", "https://example.ac.in/uploads/Convocation-Notice-2025.pdf"),
        ("exam_schedule.txt", "https://example.ac.in/exam/Exam-Schedule-2025.pdf"),
        ("fee_admin_order.txt", "https://example.ac.in/main/Guidelines-Fee-for-recognition.pdf"),
        ("fee_structure_cover.txt", "https://example.ac.in/main/Fee-Other-details.pdf"),
        ("noise_office_order.txt", "https://example.ac.in/uploads/office-order-99.pdf"),
    ],
)
def test_negative_corpus_stays_family_undetected(fixture: str, url: str) -> None:
    """Acceptance 2: guards are non-negotiable; every negative abstains."""
    plan = _planner().plan(_document(url), text_sample=_text(fixture), phase="b")

    assert plan.family is None
    assert plan.state == "abstained"
    assert plan.abstain_reason == "family_undetected"
    assert plan.extractable is False


# ------------------------------------------------------------------ precedence


def test_registry_family_wins_over_annual_report_text() -> None:
    """Acceptance 4: new signals can never override a declared family."""
    plan = _planner().plan(
        _document("https://example.ac.in/sites/Annual-Report-2023-24.pdf"),
        source={"params": {"extraction_family": "fee_structure"}},
        text_sample=_text("annual_report.txt"),
        phase="b",
    )

    assert plan.family == "fee_structure"
    assert plan.family_source == "registry"
    assert plan.signals["declared"] == "fee_structure"


def test_text_evidence_outranks_the_filename_hint() -> None:
    """Precedence: text > structure. A misnamed file is decided by its text."""
    # Filename says "fee structure"; the document is an annual report.
    url = "https://example.ac.in/uploads/Fee-Structure-2025.pdf"
    hints = family_hints_from_structure(url=url, content_type="application/pdf")
    assert hints == ("fee_structure",)

    fee = next(
        item
        for item in score_families(url=url, text_sample=_text("annual_report.txt"))
        if item.family == "fee_structure"
    )
    # The URL token fires, but every text-derived fee channel is silent (the
    # annual report's "research centre" line vetoes the fee body).
    assert fee.signals["url"] == 1
    assert fee.signals["text"] == 0
    assert fee.signals["title"] == 0

    plan = _planner().plan(_document(url), text_sample=_text("annual_report.txt"), phase="b")
    assert plan.family == "accreditation"
    assert plan.family_source == "heuristic"


def test_phase_b_text_outranks_url_tokens_at_equal_scope() -> None:
    # URL says "placement"; the document is a fee structure (existing invariant).
    plan = _planner().plan(
        _document("https://example.ac.in/uploads/placement-circular.pdf"),
        text_sample=_text("fee_circular.txt"),
        phase="b",
    )

    assert plan.family == "fee_structure"


# ------------------------------------------------------ structure-hint fallback


def test_silent_text_with_annual_report_filename_links_accreditation() -> None:
    """The weakest tier: cover text silent, filename names the family."""
    # The sample has the institution name (for ER) but no family vocabulary.
    url = "https://example.ac.in/sites/Annual-Report-2023-24.pdf"
    sample = f"{ENTITY}\nThe accounts of the Institute are enclosed herewith."

    plan = _planner().plan(_document(url), text_sample=sample, phase="b")

    assert plan.family == "accreditation"
    assert plan.family_source == "heuristic"
    assert plan.signals["fallback"] == "structure_hint"
    assert plan.signals["structure_family"] == "accreditation"
    assert plan.state == "linked"


def test_phase_a_defers_instead_of_abstaining_on_a_single_hint() -> None:
    """A single filename hint is worth one T0 read: defer, never abstain."""
    url = "https://example.ac.in/sites/Annual-Report-2023-24.pdf"

    phase_a = _planner().plan(_document(url), phase="a")

    assert phase_a.family is None  # the hint never declares a family
    assert phase_a.state == "defer_for_text"
    assert phase_a.signals["structure_hint"] == "accreditation"


def test_two_structure_hints_are_a_refusal_not_a_guess() -> None:
    """A filename naming two families never classifies from structure."""
    url = "https://example.ac.in/uploads/prospectus-fee-structure-2025.pdf"
    sample = f"{ENTITY}\nThe accounts of the Institute are enclosed herewith."

    plan = _planner().plan(_document(url), text_sample=sample, phase="b")

    assert plan.family is None
    assert plan.state == "abstained"
    assert plan.abstain_reason == "family_undetected"
    assert len(plan.signals["structure_hints"]) == 2
    assert plan.signals["structure_hints_ambiguous"] == 1


def test_administrative_filename_never_gains_a_hint() -> None:
    for url in (
        "https://example.ac.in/tenders/Tender-Notice-2025.pdf",
        "https://example.ac.in/uploads/Convocation-Notice-2025.pdf",
        "https://example.ac.in/exam/Exam-Schedule-April-2025.pdf",
        "https://example.ac.in/main/Guidelines-Fee-for-recognition-of-Institution.pdf",
    ):
        assert family_hints_from_structure(url=url, content_type="application/pdf") == ()

        phase_b = _planner().plan(
            _document(url), text_sample=_text("noise_office_order.txt"), phase="b"
        )
        assert phase_b.family is None
        assert phase_b.state == "abstained"


def test_structure_hints_do_not_fire_for_non_documents() -> None:
    """An inventory payload or a nav page is not evidence, whatever its name."""
    assert (
        family_hints_from_structure(
            url="https://example.ac.in/reports/Annual-Report-2024.xlsx",
            content_type="application/vnd.ms-excel",
        )
        == ()
    )
    assert (
        family_hints_from_structure(
            url="https://example.ac.in/news/annual-report-2024",
            content_type="text/html",
        )
        == ()
    )


# ------------------------------------------------------- ambiguity is a refusal


def test_planner_returns_needs_review_for_an_ambiguous_family() -> None:
    text = "tuition fee per annum as approved\nsanctioned intake number of seats as notified"

    plan = _planner().plan(_document(), text_sample=text, phase="b")

    assert plan.family is None
    assert plan.state == "needs_review"
    assert plan.abstain_reason == "ambiguous_family"


def test_ambiguous_family_is_stable_across_signal_channels() -> None:
    """Acceptance 3: the margin refusal holds whichever channels tie."""
    # URL-level tie (the original pinned case).
    family, _, _, reason = detect_family(url="https://example.ac.in/naac-accredit-placement-career")
    assert (family, reason) == (None, "ambiguous_family")

    # Text-level tie with richer channels.
    text = "tuition fee per annum as approved\nsanctioned intake number of seats as notified"
    family, _, _, reason = detect_family(url=NEUTRAL_URL, text_sample=text)
    assert (family, reason) == (None, "ambiguous_family")


# ------------------------------------------------------------- explainability


def test_every_decision_records_which_signals_fired() -> None:
    """Acceptance 3: signals name the channels and their counts, every time."""
    text = _text("annual_report.txt")
    family, confidence, signals, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "accreditation"
    for channel in ("url", "anchor", "text", "heading", "recurring", "title"):
        assert channel in signals
    assert signals["heading"] >= 1
    assert signals["title"] >= 1
    assert 0.0 < confidence <= 1.0
    assert sum(signals.values()) > 0


def test_runner_up_is_recorded_for_auditability() -> None:
    text = _text("nirf_submission.txt")
    _family, _, signals, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert signals["runner_up_family_score"] >= 0  # the audit trail keeps the runner-up
