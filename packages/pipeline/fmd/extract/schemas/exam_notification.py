"""Exam notification family: registration windows are time-critical claims.

v3 fields (``scope``, ``state``, ``sitting_dates``) extend the v1/v2 field set;
every field stays nullable so the older prompt versions remain valid against
this schema (missing keys default to null, nothing is removed). Two dates are
deliberately distinct and must never be conflated:

* ``registration_opens`` / ``registration_closes`` — the window that gates the
  hard filter (brief §12 filter 5);
* ``sitting_dates`` / ``exam_date`` — informational sitting evidence, never a
  filter input.

A printed month ("March 2027") is quoted verbatim and parsed downstream with
``precision: month``; the extractor never fabricates a day.
"""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class ExamNotification(FamilyDocument):
    FAMILY = "exam_notification"
    SPECS = {
        "exam_name": FieldSpec("exam.name", unit_kind="text", context=True),
        "cycle_year": FieldSpec("exam.cycle_year", context=True),
        "scope": FieldSpec("exam.scope", unit_kind="text"),
        "state": FieldSpec("exam.state", unit_kind="text"),
        "registration_opens": FieldSpec("exam.registration_opens", unit_kind="date"),
        "registration_closes": FieldSpec("exam.registration_closes", unit_kind="date"),
        "exam_date": FieldSpec("exam.date", unit_kind="date"),
        "sitting_dates": FieldSpec("exam.sitting_dates", unit_kind="text"),
        "requires_travel": FieldSpec("exam.requires_travel", unit_kind="boolean"),
        "fee_amount": FieldSpec(
            "exam.fee", unit_required=True, unit_kind="money", period="one_time"
        ),
    }

    exam_name: ExtractedField | None = None
    cycle_year: ExtractedField | None = None
    scope: ExtractedField | None = None
    state: ExtractedField | None = None
    registration_opens: ExtractedField | None = None
    registration_closes: ExtractedField | None = None
    exam_date: ExtractedField | None = None
    sitting_dates: ExtractedField | None = None
    requires_travel: ExtractedField | None = None
    fee_amount: ExtractedField | None = None
