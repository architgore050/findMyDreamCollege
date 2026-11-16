"""Text acquisition and provenance-preserving extraction tiers (brief §9).

The acquisition layer is deliberately separate from LLM extraction:

* :mod:`fmd.acquire.canonical` builds **one canonical string per page** from
  geometry-bearing blocks and keeps a lossless character map back to
  ``(page_no, bbox, pdfium_char_idx)``. Quotes are resolved by
  :mod:`fmd.extract.validate` against this text; bboxes are derived from the
  resolved offsets, never guessed.
* :mod:`fmd.acquire.tiers` chooses and runs T0 (pdftext), T1 (Docling),
  T2 (PaddleOCR-VL), T2b (Qwen3-VL Devanagari fallback) and T3 (human queue).
* :mod:`fmd.acquire.guards` flags repetition loops and entropy collapse so an
  OCR hallucination can never reach claim extraction.
* :mod:`fmd.acquire.tables` records ``(row, col, header_text)`` for numeric
  table cells and marks ambiguous header associations.

Every threshold and backend name is read from ``config/acquire.yaml`` through
the registry; nothing here hardcodes a model, a limit or an institution.
"""

from fmd.acquire.canonical import (
    BBox,
    CanonicalText,
    CharAnchor,
    PageGeometry,
    SpanHit,
    TextBlock,
    build_canonical,
)
from fmd.acquire.errors import TierUnavailable
from fmd.acquire.guards import AnomalyReport, BlockAnomaly, detect_anomalies
from fmd.acquire.tables import CellHeaderLink, TableCellAssociation, associate_headers
from fmd.acquire.tiers import (
    AcquirePolicy,
    TierDecision,
    apply_anomaly_guard,
    indic_self_check,
    plan_tiers,
    probe,
    run_tiers,
    select_tier,
    t0_extract,
    t1_extract,
    t2_extract,
    t2b_extract,
)
from fmd.acquire.types import ExtractedDoc, PageSpan, ScanProfile, Table

__all__ = [
    "AcquirePolicy",
    "AnomalyReport",
    "BBox",
    "BlockAnomaly",
    "CanonicalText",
    "CellHeaderLink",
    "CharAnchor",
    "ExtractedDoc",
    "PageGeometry",
    "PageSpan",
    "ScanProfile",
    "SpanHit",
    "Table",
    "TableCellAssociation",
    "TextBlock",
    "TierDecision",
    "TierUnavailable",
    "apply_anomaly_guard",
    "associate_headers",
    "build_canonical",
    "detect_anomalies",
    "indic_self_check",
    "plan_tiers",
    "probe",
    "run_tiers",
    "select_tier",
    "t0_extract",
    "t1_extract",
    "t2_extract",
    "t2b_extract",
]
