from __future__ import annotations

import pytest

from fmd.score.insights import (
    Insight,
    from_filter_failures,
    from_gaming_flags,
    recognition_insight,
    select,
)


def insight(id: str, predicate: str, units: set[str]) -> Insight:
    return Insight(
        id=id,
        predicate=predicate,
        text=f"{predicate} insight",
        coverage=frozenset(units),
    )


def test_greedy_prefers_maximum_coverage() -> None:
    candidates = [
        insight("a", "outcomes", {"placement", "salary"}),
        insight("b", "clinical_training", {"nabl", "internship", "rotations"}),
        insight("c", "academic_quality", {"naac"}),
        insight("d", "student_life_safety", {"safety", "hostel"}),
    ]
    selection = select(candidates, max_insights=2)
    assert [item.id for item in selection.selected] == ["b", "a"]
    assert selection.coverage == pytest.approx(5 / 8)
    assert selection.stopped_reason == "max_insights_reached"


def test_predicate_diversity_is_enforced() -> None:
    candidates = [
        insight("a1", "outcomes", {"one", "two"}),
        insight("a2", "outcomes", {"three"}),
        insight("b1", "academic_quality", {"four"}),
    ]
    selection = select(candidates, max_insights=3)
    assert [item.id for item in selection.selected] == ["a1", "b1"]


def test_min_marginal_gain_stops_early() -> None:
    candidates = [
        insight("a", "outcomes", {"x", "y", "z"}),
        insight("b", "academic_quality", {"z"}),
        insight("c", "clinical_training", {"y"}),
    ]
    selection = select(candidates, max_insights=5, min_marginal_gain=0.3)
    assert [item.id for item in selection.selected] == ["a"]
    assert selection.stopped_reason == "min_marginal_gain_reached"


def test_mandatory_insights_are_always_included_even_past_max() -> None:
    decisions = [
        {
            "filter_id": "recognition_status",
            "result": "fail",
            "reason": "unrecognised",
            "claim_id": 9,
        },
        {"filter_id": "eligibility", "result": "unknown", "reason": "no marks", "claim_id": 10},
        {"filter_id": "language", "result": "pass", "reason": "ok"},
    ]
    mandatory = list(from_filter_failures(decisions))
    mandatory.append(recognition_insight({"claim_id": 11, "value": "recognised"}))
    assert len(mandatory) == 3
    assert {item.kind for item in mandatory} == {
        "filter_failure",
        "filter_unknown",
        "recognition_status",
    }

    selection = select(
        [insight("a", "outcomes", {"x"})],
        mandatory=mandatory,
        max_insights=0,
    )
    assert selection.selected == ()
    assert len(selection.mandatory) == 3
    assert "filter:recognition_status" in selection.covered_units


def test_gaming_flags_become_insights() -> None:
    flags = [
        {
            "flag_id": "internship_counted_as_placement",
            "weight": 0.4,
            "claim_id": 5,
            "predicate": "placement_rate",
            "evidence": "internship counted",
        }
    ]
    insights = from_gaming_flags(flags)
    assert insights[0].kind == "gaming_flag"
    assert insights[0].claim_ids == (5,)
    assert "internship counted" in insights[0].text


def test_empty_universe_returns_full_coverage() -> None:
    selection = select([], mandatory=[])
    assert selection.coverage == 1.0
    assert selection.stopped_reason == "no_information_units"
