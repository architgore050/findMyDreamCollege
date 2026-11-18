"""Planner-version stamping (fmd.plan.versioning) — offline unit tests.

The stamp is how ``fmd plan --force`` tells a plan written at the current
planner version from a stale one without a schema migration.
"""

from __future__ import annotations

from fmd.plan.versioning import (
    is_current_version,
    is_replannable_reason,
    planner_version,
    stamp_reason,
    stamped_version,
    unstamped_reason,
)


def test_planner_version_reads_plan_yaml_and_falls_back() -> None:
    class _Config:
        version = "v2"

    assert planner_version(_Config()) == "2"
    assert planner_version(None) == "0"

    class _Empty:
        version = "  "

    assert planner_version(_Empty()) == "0"


def test_stamp_round_trip_for_bare_and_coded_reasons() -> None:
    assert stamp_reason(None, "1") == "plan_v1"
    assert stamp_reason("entity_ambiguous", "1") == "entity_ambiguous|plan_v1"

    assert unstamped_reason("plan_v1") is None
    assert unstamped_reason("entity_ambiguous|plan_v1") == "entity_ambiguous"
    assert unstamped_reason("unsupported_family:bogus|plan_v1") == "unsupported_family:bogus"
    # A reason without a stamp is returned as-is; human prose is untouched.
    assert unstamped_reason("family_undetected") == "family_undetected"
    assert (
        unstamped_reason("resolved by human review (link_entity)")
        == "resolved by human review (link_entity)"
    )


def test_stamped_version_extracts_the_writer_version() -> None:
    assert stamped_version("plan_v1") == "1"
    assert stamped_version("family_undetected|plan_v7") == "7"
    assert stamped_version("family_undetected") is None
    assert stamped_version(None) is None
    # Version numbers are matched exactly: v1 is not v11.
    assert stamped_version("plan_v11") == "11"
    assert is_current_version("plan_v11", "1") is False
    assert is_current_version("plan_v11", "11") is True
    assert is_current_version("family_undetected", "1") is False


def test_replannable_reasons_exclude_human_review_prose() -> None:
    # Crawl-era rows: planner codes with no stamp.
    assert is_replannable_reason(None) is True
    assert is_replannable_reason("family_undetected") is True
    assert is_replannable_reason("entity_unresolved") is True
    assert is_replannable_reason("unsupported_family:bogus") is True
    # Any pass stamp, any version.
    assert is_replannable_reason("plan_v0") is True
    assert is_replannable_reason("entity_ambiguous|plan_v1") is True
    # Human decisions are never replanned.
    assert is_replannable_reason("resolved by human review (link_entity)") is False
    assert is_replannable_reason("resolved by human review (mark_not_applicable)") is False
    assert is_replannable_reason("two institutions share this domain") is False
