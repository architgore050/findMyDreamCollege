"""Planner unit tests: family detection, entity linking, abstention codes.

Offline and deterministic: fixture text only, a fixed candidate universe, the
deterministic blocking backend and embeddings disabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fmd.db.repositories import DocumentUpsert
from fmd.er import Alias, EntityCandidate, ERConfig
from fmd.plan import DocumentPlan, DocumentPlanner, detect_family
from fmd.plan.rules import DEFAULT_RULES, PlanRules, SignalWeights

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
