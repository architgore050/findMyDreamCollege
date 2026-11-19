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
from fmd.plan.roles import RoleConfigError, family_hints_from_structure


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


# ------------------------------------------------------- structure family hints
# W5 family coverage: the file name's shape hints a family, structure only. A
# hint is URL-tier evidence — the planner uses it only when registry, text and
# URL token evidence are all silent, and administrative shapes veto every hint.


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Annual-Report-2023-24.pdf", ("accreditation",)),
        ("AnnualReport2023.pdf", ("accreditation",)),
        ("AQAR-2023-24.pdf", ("accreditation",)),
        ("SSR-Cycle-1.pdf", ("accreditation",)),
        ("NIRF-2024.pdf", ("accreditation",)),
        ("self-study-report.pdf", ("accreditation",)),
        ("Prospectus-2025-26.pdf", ("curriculum",)),
        ("Admission-Brochure-2025.pdf", ("curriculum",)),
        ("BSc-MLT-Syllabus.pdf", ("curriculum",)),
        ("Fee-Structure-2025-26.pdf", ("fee_structure",)),
        ("Fee-Circular-03-2025.pdf", ("fee_structure",)),
        ("Seat-Matrix-2025-26.pdf", ("seat_matrix",)),
        ("Sanctioned-Intake-2025.pdf", ("seat_matrix",)),
        ("Placement-Report-2024-25.pdf", ("placement_report",)),
    ],
)
def test_filename_shapes_hint_their_family(filename: str, expected: tuple[str, ...]) -> None:
    assert (
        family_hints_from_structure(
            url=f"https://college.ac.in/uploads/{filename}", content_type="application/pdf"
        )
        == expected
    )


@pytest.mark.parametrize(
    "filename",
    [
        "Tender-Notice-07-2025.pdf",
        "Convocation-Notification-2025.pdf",
        "Exam-Schedule-April-2025.pdf",
        "Examination-Schedule.pdf",
        "Date-Sheet-2025.pdf",
        "Admit-Card.pdf",
        "Office-Order-17119.pdf",
        "Press-Release-2025.pdf",
        "Annual-Day-Event.pdf",
        "Recruitment-Notice.pdf",
        # the known administrative-fee case: carries "fee", never a fee structure
        "Guidelines-Fee-for-recognition-of-Institution-as-Research-Centre-for-Ph.D-Course.pdf",
    ],
)
def test_administrative_filenames_never_hint(filename: str) -> None:
    assert (
        family_hints_from_structure(
            url=f"https://college.ac.in/main/{filename}", content_type="application/pdf"
        )
        == ()
    )


def test_hints_require_a_document_content_class() -> None:
    # A spreadsheet is an inventory payload by default: no family from structure.
    assert (
        family_hints_from_structure(
            url="https://college.ac.in/uploads/Annual-Report-2024.xlsx",
            content_type="application/vnd.ms-excel",
        )
        == ()
    )
    # An HTML page is navigation: no family from structure.
    assert (
        family_hints_from_structure(
            url="https://college.ac.in/news/annual-report-2024",
            content_type="text/html",
        )
        == ()
    )
    # A bare directory path has no filename to hint from.
    assert (
        family_hints_from_structure(url="https://college.ac.in/reports/", content_type=None)
        == ()
    )


def test_registry_can_override_the_hint_vocabulary() -> None:
    source = _Source(
        {"family_hints": {"seat_matrix": [r"intake"], "accreditation": []}}
    )
    assert (
        family_hints_from_structure(
            url="https://college.ac.in/uploads/Intake-Statement-2025.pdf",
            content_type="application/pdf",
            source=source,
        )
        == ("seat_matrix",)
    )
    assert (
        family_hints_from_structure(
            url="https://college.ac.in/uploads/Annual-Report-2025.pdf",
            content_type="application/pdf",
            source=source,
        )
        == ()
    )


def test_hints_are_deterministic_and_ordered() -> None:
    url = "https://college.ac.in/uploads/Annual-Report-2024.pdf"
    first = family_hints_from_structure(url=url, content_type="application/pdf")
    second = family_hints_from_structure(url=url, content_type="application/pdf")
    assert first == second == ("accreditation",)


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
