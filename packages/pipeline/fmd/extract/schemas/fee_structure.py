"""Fee structure family. Money always carries a printed currency (brief §13)."""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class FeeStructure(FamilyDocument):
    FAMILY = "fee_structure"
    SPECS = {
        "academic_year": FieldSpec("fee.academic_year", context=True),
        "currency": FieldSpec("fee.currency", context=True),
        "tuition_amount": FieldSpec(
            "fee.tuition", unit_required=True, unit_kind="money", period="per_year"
        ),
        "hostel_amount": FieldSpec(
            "fee.hostel", unit_required=True, unit_kind="money", period="per_year"
        ),
        "mandatory_fees": FieldSpec(
            "fee.mandatory", unit_required=True, unit_kind="money", period="per_year"
        ),
        "total_first_year": FieldSpec(
            "fee.total_first_year", unit_required=True, unit_kind="money", period="per_year"
        ),
        "fee_period_note": FieldSpec("fee.period_note", unit_kind="text"),
    }

    academic_year: ExtractedField | None = None
    currency: ExtractedField | None = None
    tuition_amount: ExtractedField | None = None
    hostel_amount: ExtractedField | None = None
    mandatory_fees: ExtractedField | None = None
    total_first_year: ExtractedField | None = None
    fee_period_note: ExtractedField | None = None
