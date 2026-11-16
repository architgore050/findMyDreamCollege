"""Field-level VLM adjudication response (T3 pass, brief §9)."""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class AdjudicationVerdict(FamilyDocument):
    """One verdict per sampled field. ``supported=None`` means "unreadable"."""

    FAMILY = "adjudication"
    SPECS = {
        "field_name": FieldSpec("adjudication.field_name", unit_kind="text"),
        "supported": FieldSpec("adjudication.supported"),
        "corrected_value": FieldSpec("adjudication.corrected_value"),
        "corrected_quote": FieldSpec("adjudication.corrected_quote", unit_kind="text"),
        "reason": FieldSpec("adjudication.reason", unit_kind="text"),
    }

    field_name: ExtractedField | None = None
    supported: ExtractedField | None = None
    corrected_value: ExtractedField | None = None
    corrected_quote: ExtractedField | None = None
    reason: ExtractedField | None = None
