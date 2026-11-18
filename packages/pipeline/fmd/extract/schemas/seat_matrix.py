"""Counselling / intake seat matrix family.

``intake`` (``seats.intake``) is the sanctioned intake printed for ONE
programme row at ONE institution — the Stanley case: a per-row value, never an
aggregate. ``aggregate_seats`` (``seats.total``) exists only for a total the
document itself states (a printed "Total" row); a row value must never be
published under it, and an aggregate must never be attached to one programme.
The registry declares ``seats.sanctioned`` (and the bare ``seats`` spelling) as
aliases of ``seats.intake``, so the legacy vocabularies resolve to the row-scoped
predicate. ``admitted`` is the *filled* count, a different fact, and is guarded
against exceeding the sanctioned intake.
"""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class SeatMatrix(FamilyDocument):
    FAMILY = "seat_matrix"
    SPECS = {
        "intake_year": FieldSpec("seats.intake_year", context=True),
        "programme_name": FieldSpec("seats.programme_name", unit_kind="text", context=True),
        "intake": FieldSpec("seats.intake", unit_required=True, unit_kind="count"),
        "aggregate_seats": FieldSpec("seats.total", unit_required=True, unit_kind="count"),
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
    intake: ExtractedField | None = None
    aggregate_seats: ExtractedField | None = None
    admitted: ExtractedField | None = None
    duration_months: ExtractedField | None = None
    government_quota: ExtractedField | None = None
    management_quota: ExtractedField | None = None
    quota_note: ExtractedField | None = None
