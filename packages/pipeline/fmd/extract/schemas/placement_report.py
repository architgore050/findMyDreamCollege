"""Placement / internship report family (brief §4, §11).

``salary_period`` is the document-level period claim: a salary whose own span
does not print "per month"/"per annum" takes its period from this quoted
sibling, never from a default. A salary that ends up with no period anywhere
is rejected — the affordability half of the product cannot annualise a guess.
"""

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
        "salary_period": FieldSpec("placement.salary_period", unit_kind="text", context=True),
        "median_salary": FieldSpec(
            "salary.median",
            unit_required=True,
            unit_kind="money",
            period="per_year",
            period_required=True,
            period_source="salary_period",
        ),
        "highest_package": FieldSpec(
            "salary.highest_package",
            unit_required=True,
            unit_kind="money",
            period="per_year",
            period_required=True,
            period_source="salary_period",
        ),
        "recruiters_count": FieldSpec(
            "placement.recruiters_count", unit_required=True, unit_kind="count"
        ),
        "internship_counted_as_employment": FieldSpec(
            "placement.internship_counted_as_employment", unit_kind="boolean"
        ),
        "outcome_basis": FieldSpec(
            "placement.basis",
            unit_kind="text",
            value_pattern=r"(?:\bare\b|\bis\b|\bwere\b|\bwas\b)\s+(.+?)\.?$",
        ),
    }

    cohort_year: ExtractedField | None = None
    eligible_count: ExtractedField | None = None
    placed_count: ExtractedField | None = None
    placement_rate: ExtractedField | None = None
    salary_period: ExtractedField | None = None
    median_salary: ExtractedField | None = None
    highest_package: ExtractedField | None = None
    recruiters_count: ExtractedField | None = None
    internship_counted_as_employment: ExtractedField | None = None
    outcome_basis: ExtractedField | None = None
