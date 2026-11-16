"""Extraction: tiered PDF text, guided JSON, validation, claims — WS-D (brief §9).

Public surface:

* :func:`fmd.extract.pdf.extract_document` — tiered text extraction (T0..T2).
* :func:`fmd.extract.schemas.schema_for` — guided-JSON family registry.
* :func:`fmd.extract.validate.validate_document` — rejection gates.
* :func:`fmd.extract.pipeline.ExtractPipeline` — snapshot -> claims.
* :func:`fmd.extract.pipeline.build_router` — router factory incl. ``--mock``.
"""

from __future__ import annotations

from fmd.extract.pdf import (
    ExtractedDoc,
    PageSpan,
    ScanProfile,
    Table,
    TierPolicy,
    TierUnavailable,
    VLMAdjudicator,
    extract_document,
    extract_pdf,
    probe,
    sample_for_adjudication,
    select_tier,
    t0_extract,
    t1_extract,
    t2_extract,
)
from fmd.extract.pipeline import (
    ExtractError,
    ExtractPipeline,
    ExtractResult,
    ExtractTarget,
    build_router,
    mock_models,
    run_extract,
)
from fmd.extract.prompts import Prompt, PromptNotFound, load_prompt, prompt_versions
from fmd.extract.schemas import FAMILIES, FamilyDocument, schema_for
from fmd.extract.validate import (
    Band,
    Bands,
    Rejection,
    SpanRewrite,
    ValidationResult,
    parse_number,
    validate_document,
)

__all__ = [
    "FAMILIES",
    "Band",
    "Bands",
    "ExtractError",
    "ExtractPipeline",
    "ExtractResult",
    "ExtractTarget",
    "ExtractedDoc",
    "FamilyDocument",
    "PageSpan",
    "Prompt",
    "PromptNotFound",
    "Rejection",
    "ScanProfile",
    "SpanRewrite",
    "Table",
    "TierPolicy",
    "TierUnavailable",
    "VLMAdjudicator",
    "ValidationResult",
    "build_router",
    "extract_document",
    "extract_pdf",
    "load_prompt",
    "mock_models",
    "parse_number",
    "probe",
    "prompt_versions",
    "run_extract",
    "sample_for_adjudication",
    "schema_for",
    "select_tier",
    "t0_extract",
    "t1_extract",
    "t2_extract",
    "validate_document",
]
