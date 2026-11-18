"""Fee structure family. Money always carries a printed currency (brief §13).

Every money field is period-required: a fee without a printed period (in its
own span, or in the document-level ``fee.period`` claim) is rejected, never
annualised by default. That is what keeps DSR25 arithmetic honest — a semester
fee read as a yearly one doubles the cost of attendance.
"""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec

_FEE_PERIOD_PATTERN = r"\b(year|semester|month)\b"


class FeeStructure(FamilyDocument):
    FAMILY = "fee_structure"
    SPECS = {
        "academic_year": FieldSpec("fee.academic_year", context=True),
        "currency": FieldSpec("fee.currency", context=True),
        "fee_period": FieldSpec(
            "fee.period",
            unit_kind="text",
            period_source=None,
            value_pattern=_FEE_PERIOD_PATTERN,
            context=True,
        ),
        "tuition_amount": FieldSpec(
            "fee.tuition",
            unit_required=True,
            unit_kind="money",
            period_required=True,
            period_source="fee_period",
        ),
        "hostel_amount": FieldSpec(
            "fee.hostel",
            unit_required=True,
            unit_kind="money",
            period_required=True,
            period_source="fee_period",
        ),
        "mess_amount": FieldSpec(
            "fee.mess",
            unit_required=True,
            unit_kind="money",
            period_required=True,
            period_source="fee_period",
        ),
        "other_amount": FieldSpec(
            "fee.other",
            unit_required=True,
            unit_kind="money",
            period_required=True,
            period_source="fee_period",
        ),
        "mandatory_fees": FieldSpec(
            "fee.mandatory",
            unit_required=True,
            unit_kind="money",
            period_required=True,
            period_source="fee_period",
        ),
        "total_first_year": FieldSpec(
            "fee.total_first_year",
            unit_required=True,
            unit_kind="money",
            period_required=True,
            period_source="fee_period",
        ),
        "fee_period_note": FieldSpec("fee.period_note", unit_kind="text"),
    }

    academic_year: ExtractedField | None = None
    currency: ExtractedField | None = None
    fee_period: ExtractedField | None = None
    tuition_amount: ExtractedField | None = None
    hostel_amount: ExtractedField | None = None
    mess_amount: ExtractedField | None = None
    other_amount: ExtractedField | None = None
    mandatory_fees: ExtractedField | None = None
    total_first_year: ExtractedField | None = None
    fee_period_note: ExtractedField | None = None
