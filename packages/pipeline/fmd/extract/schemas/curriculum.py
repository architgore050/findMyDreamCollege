"""Curriculum family: duration and internship are separate facts (brief §4)."""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class Curriculum(FamilyDocument):
    FAMILY = "curriculum"
    SPECS = {
        "programme_name": FieldSpec("curriculum.programme_name", unit_kind="text"),
        "duration_months": FieldSpec(
            "curriculum.duration_months", unit_required=True, unit_kind="months", context=True
        ),
        "semester_count": FieldSpec("curriculum.semesters", unit_required=True, unit_kind="count"),
        "internship_hours": FieldSpec(
            "curriculum.internship_hours", unit_required=True, unit_kind="hours"
        ),
        "internship_months": FieldSpec(
            "curriculum.internship_months", unit_required=True, unit_kind="months"
        ),
    }

    programme_name: ExtractedField | None = None
    duration_months: ExtractedField | None = None
    semester_count: ExtractedField | None = None
    internship_hours: ExtractedField | None = None
    internship_months: ExtractedField | None = None
