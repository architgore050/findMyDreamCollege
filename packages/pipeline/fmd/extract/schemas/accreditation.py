"""Accreditation / recognition family (hard-filter input, brief §12)."""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class Accreditation(FamilyDocument):
    FAMILY = "accreditation"
    SPECS = {
        "institution_name": FieldSpec("accreditation.institution_name", unit_kind="text"),
        "programme_name": FieldSpec("accreditation.programme_name", unit_kind="text"),
        "naac_grade": FieldSpec("accreditation.naac_grade", unit_kind="grade"),
        "recognition_status": FieldSpec("accreditation.recognition_status", unit_kind="text"),
        "recognition_body": FieldSpec("accreditation.body", unit_kind="text"),
        "valid_until": FieldSpec("accreditation.valid_until", unit_kind="date"),
    }

    institution_name: ExtractedField | None = None
    programme_name: ExtractedField | None = None
    naac_grade: ExtractedField | None = None
    recognition_status: ExtractedField | None = None
    recognition_body: ExtractedField | None = None
    valid_until: ExtractedField | None = None
