"""Placement / internship report family (brief §4, §11)."""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class PlacementReport(FamilyDocument):
    FAMILY = "placement_report"
    SPECS = {
        "cohort_year": FieldSpec("placement.cohort_year", context=True),
        "eligible_count": FieldSpec(
            "placement.eligible_count", unit_required=True, unit_kind="count", context=True
        ),
        "placed_count": FieldSpec("placement.placed_count", unit_required=True, unit_kind="count"),
        "placement_rate": FieldSpec(
            "placement.rate",
            unit_required=True,
            unit_kind="percent",
            is_rate=True,
            denominator_field="eligible_count",
        ),
        "median_salary": FieldSpec(
            "salary.median", unit_required=True, unit_kind="money", period="per_year"
        ),
        "highest_package": FieldSpec(
            "salary.highest_package", unit_required=True, unit_kind="money", period="per_year"
        ),
        "internship_counted_as_employment": FieldSpec("placement.internship_counted_as_employment"),
        "outcome_basis": FieldSpec("placement.basis", unit_kind="text"),
    }

    cohort_year: ExtractedField | None = None
    eligible_count: ExtractedField | None = None
    placed_count: ExtractedField | None = None
    placement_rate: ExtractedField | None = None
    median_salary: ExtractedField | None = None
    highest_package: ExtractedField | None = None
    internship_counted_as_employment: ExtractedField | None = None
    outcome_basis: ExtractedField | None = None
