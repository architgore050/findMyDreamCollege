"""Planner unit tests: family detection, entity linking, abstention codes.

Offline and deterministic: fixture text only, a fixed candidate universe, the
deterministic blocking backend and embeddings disabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fmd.db.repositories import DocumentUpsert
from fmd.er import Alias, EntityCandidate, ERConfig
from fmd.extract.schemas import FAMILIES
from fmd.plan import DocumentPlan, DocumentPlanner, detect_family
from fmd.plan.rules import DEFAULT_RULES, PlanRules, SignalWeights, score_families

FIXTURES = Path(__file__).parent / "fixtures" / "plan"
NEUTRAL_URL = "https://example.ac.in/notice-2025"
ENTITY_NAME = "Demo Institute of Health Sciences"


def _config() -> ERConfig:
    return ERConfig.from_mapping(
        {"blocking": {"backend": "deterministic"}, "similarity": {"use_embeddings": False}}
    )


def _document(url: str = NEUTRAL_URL) -> DocumentUpsert:
    return DocumentUpsert(url=url, domain="example.ac.in")


def _planner(candidates: list[EntityCandidate] | None = None) -> DocumentPlanner:
    return DocumentPlanner(
        candidates_provider=lambda: list(candidates or []),
        er_config=_config(),
    )


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("placement_report.txt", "placement_report"),
        ("fee_structure.txt", "fee_structure"),
        ("seat_matrix.txt", "seat_matrix"),
        ("accreditation.txt", "accreditation"),
        ("curriculum.txt", "curriculum"),
        ("exam_notification.txt", "exam_notification"),
    ],
)
def test_family_detection_on_fixtures(fixture: str, expected: str) -> None:
    text = (FIXTURES / fixture).read_text(encoding="utf-8")

    family, confidence, signals, reason = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == expected
    assert reason is None
    assert 0.0 < confidence <= 1.0
    assert sum(signals.values()) > 0


def test_detection_abstains_when_no_signal_fires() -> None:
    family, confidence, signals, reason = detect_family(url=NEUTRAL_URL, text_sample="hello")

    assert family is None
    assert confidence == 0.0
    assert signals == {}
    assert reason == "family_undetected"


def test_detection_abstains_on_a_tie() -> None:
    # Two configured families score identically; picking either would be a guess.
    family, _, _, reason = detect_family(url="https://example.ac.in/naac-accredit-placement-career")

    assert family is None
    assert reason == "ambiguous_family"


# --- W5 family coverage: signal channels (2026-11-19) -------------------------
# The 83-abstained measurement (135 snapshots) was dominated by annual reports,
# prospectuses and NIRF submissions whose text vocabulary never scored. The
# channels below are the new evidence: heading-shaped lines, the cover-title
# line, and tokens that recur across the sample.


def test_heading_and_title_channels_are_recorded() -> None:
    text = (FIXTURES / "annual_report.txt").read_text(encoding="utf-8")

    family, confidence, signals, reason = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "accreditation"
    assert reason is None
    assert signals["heading"] >= 1  # "Annual Report 2023-24" is heading-shaped
    assert signals["title"] >= 1  # the cover names the document type
    assert signals["title_decisive"] == 1  # cover identity decides the document
    assert confidence > 0


def test_recurring_channel_fires_on_repetition() -> None:
    text = "\n".join(
        ["Fee particulars of the B.Sc. MLT programme are:", "Tuition fee: Rs. 95000"]
        + [f"Special fee {n}: Rs. {n}000" for n in range(1, 5)]
    )

    _, _, signals, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert signals["recurring"] >= 1  # "fee"/"fees" occur far beyond recurring_min


def test_recurring_channel_does_not_count_single_mentions() -> None:
    # A single "fee" mention is not a fee structure: below recurring_min.
    text = "The prospectus explains that one special fee is charged at admission."

    _, _, signals, reason = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert signals.get("recurring", 0) == 0
    assert reason == "family_undetected"


def test_secondary_hint_marks_placement_sections_in_an_annual_report() -> None:
    text = (FIXTURES / "annual_report.txt").read_text(encoding="utf-8")

    family, _, signals, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "accreditation"
    assert signals["secondary_hint"] == 1
    assert signals["secondary_hint_family_score"] > 0


def test_annual_report_without_placement_sections_has_no_hint() -> None:
    text = (FIXTURES / "annual_report_no_placement.txt").read_text(encoding="utf-8")

    family, _, signals, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "accreditation"
    assert "secondary_hint" not in signals


def test_prospectus_with_fee_tables_resolves_fee_structure() -> None:
    text = (FIXTURES / "prospectus_fees.txt").read_text(encoding="utf-8")

    family, confidence, signals, reason = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "fee_structure"
    assert reason is None
    assert signals["prospectus_fee_table_rows"] >= 3
    assert signals["prospectus_intake_table_rows"] == 0
    assert confidence > 0


def test_prospectus_with_intake_tables_resolves_seat_matrix() -> None:
    text = (FIXTURES / "prospectus_intake.txt").read_text(encoding="utf-8")

    family, _, signals, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "seat_matrix"
    assert signals["prospectus_intake_table_rows"] >= 3
    assert signals["prospectus_fee_table_rows"] == 0


def test_prospectus_without_tables_resolves_curriculum() -> None:
    text = (FIXTURES / "prospectus_general.txt").read_text(encoding="utf-8")

    family, _, signals, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "curriculum"
    assert signals["prospectus_fee_table_rows"] == 0
    assert signals["prospectus_intake_table_rows"] == 0


def test_prospectus_pseudo_family_never_escapes_detection() -> None:
    text = (FIXTURES / "prospectus_fees.txt").read_text(encoding="utf-8")

    for _ in range(2):
        family, _, _, _ = detect_family(url=NEUTRAL_URL, text_sample=text)
        assert family in FAMILIES  # the shape chain must emit a real schema family


def test_nirf_submission_classifies_accreditation_despite_body_noise() -> None:
    # NIRF submissions carry fee and placement tables; the cover identity wins.
    text = (FIXTURES / "nirf_submission.txt").read_text(encoding="utf-8")

    family, _, signals, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "accreditation"
    assert signals["title_decisive"] == 1


# --- W5 precision guards: the negative corpus must stay abstained -------------


@pytest.mark.parametrize(
    "fixture",
    [
        "staff_circular.txt",
        "tender_notice.txt",
        "event_news_page.txt",
        "convocation_notice.txt",
        "exam_schedule.txt",
        "noise_office_order.txt",
        "fee_admin_order.txt",
    ],
)
def test_negative_documents_never_classify(fixture: str) -> None:
    text = (FIXTURES / fixture).read_text(encoding="utf-8")

    family, _, signals, reason = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family is None
    assert reason == "family_undetected"
    assert signals.get("title_decisive") is None


def test_exam_registration_notification_still_classifies() -> None:
    # Acceptance 2: examination schedules abstain *unless* they are
    # registration notifications — the registration shape keeps its family.
    text = (FIXTURES / "exam_registration_notification.txt").read_text(encoding="utf-8")

    family, _, _, _ = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family == "exam_notification"


def test_fee_veto_cancels_every_text_derived_channel() -> None:
    # The administrative-fee cover names "Fee Details" and "Annual fee" — all
    # text-derived fee channels must be cancelled, not just the body tokens.
    text = (FIXTURES / "fee_structure_cover.txt").read_text(encoding="utf-8")

    fee = next(
        item
        for item in score_families(url=NEUTRAL_URL, text_sample=text)
        if item.family == "fee_structure"
    )

    assert fee.score == 0.0
    assert fee.signals["text_vetoed"] >= 1
    assert fee.signals["text"] == 0
    assert fee.signals["heading"] == 0
    assert fee.signals["recurring"] == 0
    assert fee.signals["title"] == 0


def test_ambiguous_text_above_the_floor_returns_ambiguous_family() -> None:
    # Two families tie above the floor with no cover identity: never a guess.
    text = "tuition fee per annum as approved\nsanctioned intake number of seats as notified"

    family, _, signals, reason = detect_family(url=NEUTRAL_URL, text_sample=text)

    assert family is None
    assert reason == "ambiguous_family"
    assert signals["runner_up_family_score"] > 0


def test_detection_is_deterministic_across_the_whole_corpus() -> None:
    # Every decision, including the signal dict, is byte-stable across runs.
    fixtures = sorted(FIXTURES.glob("*.txt"))
    assert len(fixtures) >= 20

    def decide() -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for fixture in fixtures:
            text = fixture.read_text(encoding="utf-8")
            out[fixture.name] = detect_family(url=NEUTRAL_URL, text_sample=text)
        return out

    first = decide()
    second = decide()
    assert first == second


def test_rules_are_injectable_data() -> None:
    rules = PlanRules(
        families=(
            type(DEFAULT_RULES.families[0])(family="custom_family", url_tokens=("annual-day",)),
        ),
        weights=SignalWeights(url=0.9, min_score=0.5),
    )
    family, confidence, _, reason = detect_family(
        url="https://example.ac.in/annual-day", rules=rules
    )

    assert (family, reason) == ("custom_family", None)
    assert confidence == pytest.approx(0.9)


def test_registry_declared_family_wins_over_heuristics() -> None:
    text = (FIXTURES / "placement_report.txt").read_text(encoding="utf-8")
    planner = _planner()
    plan = planner.plan(
        _document(),
        source={"params": {"extraction_family": "fee_structure"}},
        text_sample=text,
    )

    assert plan.family == "fee_structure"
    assert plan.family_source == "registry"
    assert plan.confidence > 0
    assert plan.entity_id is None
    assert plan.abstain_reason == "entity_undetected"


def test_reference_hint_fills_family_when_registry_is_silent() -> None:
    text = (FIXTURES / "placement_report.txt").read_text(encoding="utf-8")

    plan = _planner().plan(_document(), family_hint="placement_report", text_sample=text)

    assert plan.family == "placement_report"
    assert plan.family_source == "reference"


def test_unknown_declared_family_abstains_instead_of_guessing() -> None:
    plan = _planner().plan(_document(), source={"params": {"extraction_family": "not_a_family"}})

    assert plan.family is None
    assert plan.state == "abstained"
    assert plan.abstain_reason == "unsupported_family:not_a_family"


# --- administrative-fee negative case (W1 2026-11-18) -------------------------
# Observed on RUHS (2026-11-17): "Guidelines … Fee for recognition of Institution
# as Research Centre for Ph.D. Course" — an administrative order whose URL token
# "fee" is an institution-paid charge. It must never classify as fee_structure.

ADMIN_FEE_URL = (
    "https://ruhsraj.org/main/Guidelines-Fee-for-recognition-of-"
    "Institution-as-Research-Centre-for-Ph.D-Course.pdf"
)


def test_admin_fee_order_is_vetoed_as_fee_structure() -> None:
    text = (FIXTURES / "fee_admin_order.txt").read_text(encoding="utf-8")

    fee = next(
        item
        for item in score_families(url=ADMIN_FEE_URL, text_sample=text)
        if item.family == "fee_structure"
    )
    assert fee.score == 0.0
    assert fee.signals["url_vetoed"] >= 1
    assert fee.signals["text_vetoed"] >= 1

    family, confidence, _signals, reason = detect_family(url=ADMIN_FEE_URL, text_sample=text)
    assert family is None
    assert confidence == 0.0
    assert reason == "family_undetected"


def test_admin_fee_order_planner_abstains_without_a_family() -> None:
    text = (FIXTURES / "fee_admin_order.txt").read_text(encoding="utf-8")

    plan = _planner().plan(_document(url=ADMIN_FEE_URL), text_sample=text)

    assert plan.family is None
    assert plan.family_source == "none"
    assert plan.state == "abstained"
    assert plan.abstain_reason == "family_undetected"


def test_fee_fixture_still_detects_fee_structure() -> None:
    text = (FIXTURES / "fee_structure.txt").read_text(encoding="utf-8")

    family, confidence, _signals, reason = detect_family(
        url="https://example.ac.in/files/fee/Fee-Structure-2025-26.pdf", text_sample=text
    )

    assert family == "fee_structure"
    assert reason is None
    assert confidence > 0


def test_reference_hint_carries_fee_structure_and_registry_outranks_it() -> None:
    text = (FIXTURES / "placement_report.txt").read_text(encoding="utf-8")

    hinted = _planner().plan(_document(), family_hint="fee_structure", text_sample=text)
    assert hinted.family == "fee_structure"
    assert hinted.family_source == "reference"

    registry = _planner().plan(
        _document(),
        source={"params": {"extraction_family": "placement_report"}},
        family_hint="fee_structure",
        text_sample=text,
    )
    assert registry.family == "placement_report"
    assert registry.family_source == "registry"


def test_single_confident_candidate_is_linked() -> None:
    text = f"{ENTITY_NAME}\nPlacement Report 2024-25\nStudents placed: 96\n"
    planner = _planner(
        [
            EntityCandidate(entity_id=7, canonical_name=ENTITY_NAME, state="Delhi"),
            EntityCandidate(entity_id=8, canonical_name="Another College", state="Kerala"),
        ]
    )

    plan = planner.plan(_document(), family_hint="placement_report", text_sample=text)

    assert plan.entity_id == 7
    assert plan.state == "linked"
    assert plan.extractable is True
    assert plan.abstain_reason is None
    assert plan.entity_candidates[0].entity_id == 7


def test_ambiguous_entity_is_never_guessed() -> None:
    text = f"{ENTITY_NAME}\nPlacement Report 2024-25\n"
    planner = _planner(
        [
            EntityCandidate(entity_id=1, canonical_name=ENTITY_NAME, state="Delhi"),
            EntityCandidate(entity_id=2, canonical_name=ENTITY_NAME, state="Delhi"),
        ]
    )

    plan = planner.plan(_document(), family_hint="placement_report", text_sample=text)

    assert plan.entity_id is None
    assert plan.state == "needs_review"
    assert plan.abstain_reason == "entity_ambiguous"
    assert plan.extractable is False
    assert {item.entity_id for item in plan.entity_candidates} == {1, 2}


def test_manual_alias_is_authoritative() -> None:
    text = f"{ENTITY_NAME}\nPlacement Report 2024-25\n"
    planner = _planner(
        [
            EntityCandidate(entity_id=1, canonical_name=ENTITY_NAME, state="Delhi"),
            EntityCandidate(
                entity_id=2,
                canonical_name="Renamed Health University",
                state="Delhi",
                aliases=[Alias(alias=ENTITY_NAME, alias_type="manual")],
            ),
        ]
    )

    plan = planner.plan(_document(), family_hint="placement_report", text_sample=text)

    assert plan.entity_id == 2
    assert plan.state == "linked"


def test_no_candidate_universe_is_an_honest_abstention() -> None:
    text = f"{ENTITY_NAME}\nPlacement Report 2024-25\n"
    plan = _planner().plan(_document(), family_hint="placement_report", text_sample=text)

    assert plan.entity_id is None
    assert plan.abstain_reason == "entity_unresolved"
    assert plan.state == "abstained"


def test_plan_is_serialisable_for_the_run_ledger() -> None:
    plan = DocumentPlan(family="placement_report", entity_id=3, state="linked", confidence=0.9)

    payload = plan.as_dict()

    assert payload["family"] == "placement_report"
    assert payload["state"] == "linked"
    assert payload["entity_id"] == 3
