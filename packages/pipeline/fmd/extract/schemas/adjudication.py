"""Field-level VLM adjudication response (T3 pass, brief §9).

Adjudication is not an extraction contract: it *checks* an already-resolved
claim, so its verdict may carry a value and a corrected quote. It still never
carries offsets — a corrected quote is re-resolved through
:func:`fmd.extract.validate.validate_document`, never spliced in.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from fmd.extract.schemas.base import FamilyDocument, FieldSpec, JsonScalar


class VerdictField(BaseModel):
    """One verdict sub-field. ``null`` means "the VLM could not tell"."""

    model_config = ConfigDict(extra="forbid")

    value: JsonScalar | None = None
    quote: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AdjudicationVerdict(FamilyDocument):
    """One verdict per sampled field. ``supported=None`` means "unreadable"."""

    FAMILY = "adjudication"
    SPECS = {
        "field_name": FieldSpec("adjudication.field_name", unit_kind="text"),
        "supported": FieldSpec("adjudication.supported", unit_kind="boolean"),
        "corrected_value": FieldSpec("adjudication.corrected_value"),
        "corrected_quote": FieldSpec("adjudication.corrected_quote", unit_kind="text"),
        "reason": FieldSpec("adjudication.reason", unit_kind="text"),
    }

    field_name: VerdictField | None = None
    supported: VerdictField | None = None
    corrected_value: VerdictField | None = None
    corrected_quote: VerdictField | None = None
    reason: VerdictField | None = None
