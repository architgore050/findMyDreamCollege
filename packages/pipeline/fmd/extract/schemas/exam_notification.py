"""Exam notification family: registration windows are time-critical claims."""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class ExamNotification(FamilyDocument):
    FAMILY = "exam_notification"
    SPECS = {
        "exam_name": FieldSpec("exam.name", unit_kind="text", context=True),
        "cycle_year": FieldSpec("exam.cycle_year", context=True),
        "registration_opens": FieldSpec("exam.registration_opens", unit_kind="date"),
        "registration_closes": FieldSpec("exam.registration_closes", unit_kind="date"),
        "exam_date": FieldSpec("exam.date", unit_kind="date"),
        "requires_travel": FieldSpec("exam.requires_travel"),
        "fee_amount": FieldSpec(
            "exam.fee", unit_required=True, unit_kind="money", period="one_time"
        ),
    }

    exam_name: ExtractedField | None = None
    cycle_year: ExtractedField | None = None
    registration_opens: ExtractedField | None = None
    registration_closes: ExtractedField | None = None
    exam_date: ExtractedField | None = None
    requires_travel: ExtractedField | None = None
    fee_amount: ExtractedField | None = None
