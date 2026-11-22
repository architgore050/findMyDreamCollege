"""Document records shared by the acquisition tiers.

``ExtractedDoc`` is the single object the extraction pipeline consumes. It
carries the canonical text, the page geometry needed to validate bboxes, the
blocks the text was built from, the anomaly verdict and the table cell
associations. Everything except the anchors is JSON-serialisable through
:meth:`ExtractedDoc.to_structure`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fmd.acquire.canonical import (
    CanonicalText,
    PageGeometry,
    TextBlock,
)
from fmd.acquire.guards import GuardTrip

Granularity = Literal["glyph", "span", "block"]


@dataclass(frozen=True)
class PageSpan:
    page: int
    char_start: int
    char_end: int

    def as_dict(self) -> dict[str, int]:
        return {"page": self.page, "char_start": self.char_start, "char_end": self.char_end}


@dataclass(frozen=True)
class Table:
    rows: list[list[str]]
    page: int | None = None
    caption: str | None = None
    #: The numeric cell → column-header links (``CellHeaderLink``), as built by
    #: ``associate_headers``; ``None``/absent for OCR tables without headers.
    cells: tuple[Any, ...] = ()
    column_header_uncertain: bool = False
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "page": self.page,
            "caption": self.caption,
            "column_header_uncertain": self.column_header_uncertain,
            "confidence": self.confidence,
            "cells": [cell.as_dict() for cell in self.cells],
        }


@dataclass(frozen=True)
class ScanProfile:
    page_count: int
    char_count: int
    chars_per_page: float
    devanagari: bool
    tables_detected: bool
    two_column: bool = False
    table_confidence: float | None = None
    image_only: bool = False


@dataclass(frozen=True)
class ExtractedDoc:
    """Canonical text plus everything needed to audit where it came from."""

    text: str
    page_map: list[PageSpan]
    tables: list[Table]
    tier: str
    tool: str
    tool_version: str | None = None
    fallback_from: str | None = None
    notes: tuple[str, ...] = ()
    canonical: CanonicalText | None = None
    pages: tuple[PageGeometry, ...] = ()
    blocks: tuple[TextBlock, ...] = ()
    bbox_granularity: Granularity = "span"
    ocr_anomaly: bool = False
    anomaly_reasons: tuple[str, ...] = ()
    anomalous_pages: tuple[int, ...] = ()
    guard_trips: tuple[GuardTrip, ...] = ()
    needs_review: bool = False
    #: Stable code for a review routed to T3 for a non-model reason (e.g.
    #: ``artifacts_unavailable_offline``). Anomaly reasons stay separate.
    review_reason: str | None = None
    two_column: bool = False
    table_confidence: float | None = None
    #: OCR engine provenance (engine, engine_version, model_ids, device,
    #: per-page confidence/flags). ``None`` for born-digital tiers (T0/T1),
    #: a flat JSON-serialisable dict for OCR tiers — every scanned page
    #: records what read it, so a mixed-engine corpus stays reproducible.
    engine_meta: dict[str, Any] | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def guard_trips_by_name(self) -> dict[str, int]:
        """Guard refusals counted by name for the run summary."""
        counts: dict[str, int] = {}
        for trip in self.guard_trips:
            counts[trip.guard] = counts.get(trip.guard, 0) + 1
        return dict(sorted(counts.items()))

    def page_box(self, page_no: int) -> tuple[float, float, float, float] | None:
        for page in self.pages:
            if page.page_no == page_no:
                return page.bbox
        return None

    def bboxes_for_offsets(self, start: int, end: int) -> list[dict[str, Any]]:
        if self.canonical is None:
            return []
        return [hit.as_dict() for hit in self.canonical.bboxes(start, end)]

    def to_structure(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "tool": self.tool,
            "tool_version": self.tool_version,
            "fallback_from": self.fallback_from,
            "notes": list(self.notes),
            "page_map": [span.as_dict() for span in self.page_map],
            "tables": [table.as_dict() for table in self.tables],
            "bbox_granularity": self.bbox_granularity,
            "pages": [
                {
                    **page.as_dict(),
                    "ocr_anomaly": page.page_no in self.anomalous_pages,
                }
                for page in self.pages
            ],
            "ocr_anomaly": self.ocr_anomaly,
            "anomaly_reasons": list(self.anomaly_reasons),
            "anomalous_pages": list(self.anomalous_pages),
            "guard_trips": [trip.as_dict() for trip in self.guard_trips],
            "guard_trips_by_name": self.guard_trips_by_name(),
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
            "two_column": self.two_column,
            "table_confidence": self.table_confidence,
            "engine_meta": self.engine_meta,
            "table_cells_uncertain": any(table.column_header_uncertain for table in self.tables),
            "block_count": len(self.blocks),
        }

    def as_review_item(self) -> dict[str, Any]:
        return {
            "tier": "T3",
            "tool": "human_queue",
            "reasons": (
                [self.review_reason]
                if self.review_reason
                else (list(self.anomaly_reasons) or ["tier_escalation"])
            ),
            "guard_trips": [trip.as_dict() for trip in self.guard_trips],
            "guard_trips_by_name": self.guard_trips_by_name(),
            "anomalous_pages": list(self.anomalous_pages),
            "page_count": len(self.pages or self.page_map),
            "table_confidence": self.table_confidence,
            "engine_meta": self.engine_meta,
        }


def page_spans(pages: tuple[PageGeometry, ...]) -> list[PageSpan]:
    return [
        PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
        for page in pages
    ]


__all__ = [
    "ExtractedDoc",
    "Granularity",
    "PageSpan",
    "ScanProfile",
    "Table",
    "page_spans",
]
