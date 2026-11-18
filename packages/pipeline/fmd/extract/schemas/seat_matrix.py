"""Counselling / intake seat matrix family.

``total_seats`` (``seats.total``) is the printed sanctioned intake — the
registry declares ``seats.sanctioned`` as its alias, so both vocabularies
resolve to one row instead of emitting the same number twice. ``admitted`` is
the *filled* count, a different fact, and is guarded against exceeding the
sanctioned intake.
"""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class SeatMatrix(FamilyDocument):
    FAMILY = "seat_matrix"
    SPECS = {
        "intake_year": FieldSpec("seats.intake_year", context=True),
        "programme_name": FieldSpec("seats.programme_name", unit_kind="text", context=True),
        "total_seats": FieldSpec("seats.total", unit_required=True, unit_kind="count"),
        "admitted": FieldSpec("seats.admitted", unit_required=True, unit_kind="count"),
        "duration_months": FieldSpec("seats.duration_months", unit_kind="months"),
        "government_quota": FieldSpec(
            "seats.government_quota", unit_required=True, unit_kind="count"
        ),
        "management_quota": FieldSpec(
            "seats.management_quota", unit_required=True, unit_kind="count"
        ),
        "quota_note": FieldSpec("seats.quota_note", unit_kind="text"),
    }

    intake_year: ExtractedField | None = None
    programme_name: ExtractedField | None = None
    total_seats: ExtractedField | None = None
    admitted: ExtractedField | None = None
    duration_months: ExtractedField | None = None
    government_quota: ExtractedField | None = None
    management_quota: ExtractedField | None = None
    quota_note: ExtractedField | None = None
