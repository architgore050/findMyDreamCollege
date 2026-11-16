"""Document role classification and honest abstention bucketing (no DB, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_orchestrate_support import MemoryDocuments, MemoryExtractions, MemorySnapshots

from fmd.db.repositories import DocumentUpsert, SnapshotCreate
from fmd.orchestrate.extract import Extractor
from fmd.orchestrate.ledger import RunSummary, utcnow
from fmd.plan import DocumentPlan, DocumentPlanner, RoleRules, classify_content, role_for
from fmd.plan.roles import RoleConfigError


class _Source:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.id = "fixture"
        self.adapter = "fixture"
        self.params = params or {}


class _Doc:
    def __init__(self, url: str) -> None:
        self.url = url
        self.id = 1
        self.source_id = "fixture"
        self.extraction_family = None
        self.entity_id = None


# ----------------------------------------------------------------- role rules


@pytest.mark.parametrize(
    "url,content_type,expected_role",
    [
        (
            "https://x.gov.in/universitydetails/Getuniversity_details",
            "application/json",
            "inventory",
        ),
        ("https://x.gov.in/list.xlsx", "application/vnd.ms-excel", "inventory"),
        ("https://college.ac.in/admissions", "text/html", "navigation"),
        ("https://college.ac.in/", None, "navigation"),
        ("https://college.ac.in/files/placement-2025.pdf", "application/pdf", "evidence"),
        ("https://college.ac.in/aqar.pdf", None, "evidence"),
        ("https://college.ac.in/report", "application/pdf", "evidence"),
    ],
)
def test_role_follows_content_class(url, content_type, expected_role) -> None:
    assert role_for(url=url, content_type=content_type) == expected_role


def test_registry_role_map_overrides_the_page_default() -> None:
    source = _Source({"document_role": {"page": "inventory"}})
    assert (
        role_for(url="https://x.ac.in/college-list", content_type="text/html", source=source)
        == "inventory"
    )


def test_a_declared_family_is_always_evidence() -> None:
    source = _Source({"document_role": {"page": "navigation"}})
    assert (
        role_for(
            url="https://x.ac.in/page",
            content_type="text/html",
            source=source,
            family="seat_matrix",
        )
        == "evidence"
    )


def test_invalid_role_map_fails_loudly() -> None:
    with pytest.raises(RoleConfigError):
        RoleRules.from_source(_Source({"document_role": {"page": "maybe"}}))
    with pytest.raises(RoleConfigError):
        role_for(url="https://x.ac.in/", source=_Source({"document_role": {"page": "maybe"}}))


def test_structured_url_pattern_marks_a_mislabelled_xhr() -> None:
    source = _Source({"structured_url_patterns": [r"(?i)getuniversity_details"]})
    assert (
        classify_content(
            url="https://ugc.gov.in/Getuniversity_details",
            content_type="text/html",
            rules=RoleRules.from_source(source),
        )
        == "structured"
    )


# --------------------------------------------------------------- planner role


def test_planner_marks_navigation_not_applicable_without_an_abstention() -> None:
    planner = DocumentPlanner()
    plan = planner.plan(_Doc("https://college.ac.in/"), role="navigation")

    assert isinstance(plan, DocumentPlan)
    assert plan.state == "not_applicable"
    assert plan.not_applicable is True
    assert plan.abstain_reason is None  # never family_undetected
    assert plan.family is None


def test_planner_marks_inventory_not_applicable() -> None:
    plan = DocumentPlanner().plan(_Doc("https://ugc.gov.in/list.json"), role="inventory")
    assert plan.state == "not_applicable"
    assert plan.abstain_reason is None


def test_planner_still_plans_evidence() -> None:
    text = (Path(__file__).parent / "fixtures" / "plan" / "placement_report.txt").read_text()
    plan = DocumentPlanner().plan(
        _Doc("https://college.ac.in/placement.pdf"),
        role="evidence",
        text_sample=text,
    )
    assert plan.state != "not_applicable"


# ------------------------------------------------------- abstention bucketing


class _FakeEntities:
    def list(self, *args, **kwargs):
        return []


def test_not_applicable_is_never_counted_as_an_extraction_failure() -> None:
    documents = MemoryDocuments()
    extractions = MemoryExtractions()
    snapshots = MemorySnapshots(extractions, documents)
    documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/",
            domain="college.ac.in",
            role="navigation",
            plan_state="not_applicable",
        )
    )
    documents.upsert(
        DocumentUpsert(
            url="https://college.ac.in/news",
            domain="college.ac.in",
            role="navigation",
            plan_state="not_applicable",
        )
    )
    evidence = documents.upsert(
        DocumentUpsert(url="https://college.ac.in/aqar.pdf", domain="college.ac.in")
    )
    snapshots.create(
        SnapshotCreate(document_id=evidence.id, sha256="a" * 64, bytes_path="objects/a")
    )

    summary = RunSummary(run_id="run-1", stage="extract", started_at=utcnow())
    extractor = Extractor(
        snapshots=snapshots,
        documents=documents,
        extractions=extractions,
        claims=None,
        entities=_FakeEntities(),
        store=None,
        router=None,
    )
    extractor._count_unplanned(summary)
    summary.failed = False

    assert summary.counts["not_applicable"] == 2
    assert summary.counts.get("unplanned", 0) == 1
    assert summary.metadata["unplanned"] == {"missing_family": 1}
    assert summary.counts.get("plan.reason.family_undetected", 0) == 0
    assert summary.failed is False
