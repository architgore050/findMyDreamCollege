"""One canonical string per page, with a lossless map back to the PDF.

The model never emits offsets: it emits a verbatim quote. Everything else is
derived here.

Build order
-----------
``TextBlock`` records (page number, bbox, text) come from T0 pdftext spans,
T1 Docling items or T2 OCR blocks. :func:`build_canonical` concatenates their
text in reading order with deterministic separators and stores **one anchor per
canonical character**:

    canonical_start, canonical_end -> (page_no, bbox, pdfium_char_idx_range)

Separators (newlines between blocks/pages) carry no anchor. A quote that
resolves to offsets spanning two blocks therefore yields **two** bboxes, never
one merged box that would be a lie about where the source is.

Granularity is honest about what the source could provide:

* ``glyph`` — pdftext per-character boxes (``keep_chars=True``);
* ``span``  — Docling item / pdftext span boxes, one box per block;
* ``block`` — OCR block boxes (PaddleOCR-VL / Qwen3-VL), never finer.

Nothing here reads a clock, a network or a model. The same blocks produce the
same string and the same anchors on every run.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

BBox = tuple[float, float, float, float]
Granularity = Literal["glyph", "span", "block"]

BLOCK_SEPARATOR = "\n"
PAGE_SEPARATOR = "\n\n"


def merge_boxes(boxes: Iterable[BBox]) -> BBox:
    """Smallest axis-aligned box covering ``boxes`` (deterministic)."""
    values = list(boxes)
    if not values:
        raise ValueError("cannot merge an empty list of boxes")
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def box_inside(inner: BBox, outer: BBox | None, *, epsilon: float = 0.0) -> bool:
    """Is ``inner`` a non-degenerate box inside ``outer`` (with tolerance)?"""
    x0, y0, x1, y1 = inner
    if x1 < x0 or y1 < y0:
        return False
    if outer is None:
        return False
    ox0, oy0, ox1, oy1 = outer
    return (
        x0 >= ox0 - epsilon and y0 >= oy0 - epsilon and x1 <= ox1 + epsilon and y1 <= oy1 + epsilon
    )


@dataclass(frozen=True)
class TextBlock:
    """A geometry-bearing unit of text from one tier.

    ``char_boxes`` is populated only when the tier can supply per-glyph boxes,
    and must have exactly one box per character of ``text``. ``char_indexes``
    is the pdfium textpage character index for each character (inclusive), also
    optional and same-length.
    """

    page_no: int
    text: str
    bbox: BBox | None = None
    granularity: Granularity = "span"
    char_boxes: tuple[BBox, ...] | None = None
    char_indexes: tuple[int, ...] | None = None
    block_id: int | None = None
    line: int | None = None

    def __post_init__(self) -> None:
        if self.char_boxes is not None and len(self.char_boxes) != len(self.text):
            raise ValueError("char_boxes must match text length one-to-one")
        if self.char_indexes is not None and len(self.char_indexes) != len(self.text):
            raise ValueError("char_indexes must match text length one-to-one")


@dataclass(frozen=True)
class CharAnchor:
    """Where one canonical character came from."""

    page_no: int
    block_id: int
    bbox: BBox | None
    pdfium_char_idx: int | None
    granularity: Granularity

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "block_id": self.block_id,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "pdfium_char_idx": self.pdfium_char_idx,
            "granularity": self.granularity,
        }


@dataclass(frozen=True)
class PageGeometry:
    page_no: int
    width: float
    height: float
    bbox: BBox
    char_start: int
    char_end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "width": self.width,
            "height": self.height,
            "bbox": list(self.bbox),
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True)
class SpanHit:
    """A resolved canonical span projected back onto the page."""

    page_no: int
    bbox: BBox
    canonical_start: int
    canonical_end: int
    pdfium_char_start: int | None
    pdfium_char_end: int | None
    granularity: Granularity
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "bbox": list(self.bbox),
            "canonical_start": self.canonical_start,
            "canonical_end": self.canonical_end,
            "pdfium_char_start": self.pdfium_char_start,
            "pdfium_char_end": self.pdfium_char_end,
            "bbox_granularity": self.granularity,
            "text": self.text,
        }


@dataclass(frozen=True)
class CanonicalText:
    """Canonical page text plus the per-character provenance map."""

    text: str
    pages: tuple[PageGeometry, ...]
    anchors: tuple[CharAnchor | None, ...]
    block_boxes: dict[tuple[int, int], BBox]
    granularity: Granularity = "span"

    @property
    def page_starts(self) -> list[int]:
        return [page.char_start for page in self.pages]

    def page_for_offset(self, offset: int) -> PageGeometry | None:
        starts = self.page_starts
        if not starts:
            return None
        index = bisect_right(starts, offset) - 1
        if index < 0:
            index = 0
        page = self.pages[index]
        if page.char_start <= offset <= page.char_end:
            return page
        return None

    def page_box(self, page_no: int) -> BBox | None:
        for page in self.pages:
            if page.page_no == page_no:
                return page.bbox
        return None

    def bboxes(self, start: int, end: int) -> list[SpanHit]:
        """Project a canonical range onto the blocks it covers.

        Returns one hit per contiguous block run. Multi-block quotes return
        multiple boxes; a separator-only range returns no boxes.
        """
        if start < 0 or end <= start or end > len(self.text):
            return []
        hits: list[SpanHit] = []
        run_key: tuple[int, int] | None = None
        run_start = start
        boxes: list[BBox] = []
        indexes: list[int] = []
        run_granularity: Granularity = "span"

        def flush(run_end: int) -> None:
            nonlocal boxes, indexes, run_key
            if run_key is None or not boxes:
                boxes = []
                indexes = []
                return
            page_no, _ = run_key
            hits.append(
                SpanHit(
                    page_no=page_no,
                    bbox=merge_boxes(boxes),
                    canonical_start=run_start,
                    canonical_end=run_end,
                    pdfium_char_start=min(indexes) if indexes else None,
                    pdfium_char_end=(max(indexes) + 1) if indexes else None,
                    granularity=run_granularity,
                    text=self.text[run_start:run_end],
                )
            )
            boxes = []
            indexes = []

        for offset in range(start, end):
            anchor = self.anchors[offset]
            if anchor is None or anchor.bbox is None:
                flush(offset)
                run_key = None
                continue
            key = (anchor.page_no, anchor.block_id)
            if key != run_key:
                flush(offset)
                run_key = key
                run_start = offset
                run_granularity = anchor.granularity
            boxes.append(anchor.bbox)
            if anchor.pdfium_char_idx is not None:
                indexes.append(anchor.pdfium_char_idx)
        flush(end)
        return hits

    def resolve(self, quote: str) -> list[SpanHit]:
        """Resolve a verbatim quote; ambiguous quotes resolve to no boxes."""
        if not quote:
            return []
        first = self.text.find(quote)
        if first < 0:
            return []
        if self.text.find(quote, first + 1) != -1:
            return []
        return self.bboxes(first, first + len(quote))


def build_canonical(
    blocks: Sequence[TextBlock],
    pages: Sequence[PageGeometry] | None = None,
    *,
    block_separator: str = BLOCK_SEPARATOR,
    page_separator: str = PAGE_SEPARATOR,
) -> CanonicalText:
    """Concatenate blocks into canonical text with a per-character anchor map."""
    ordered = sorted(blocks, key=lambda block: (block.page_no, block.block_id or 0))
    by_page: dict[int, list[TextBlock]] = {}
    for block in ordered:
        if not block.text:
            continue
        by_page.setdefault(block.page_no, []).append(block)

    page_numbers = sorted({page.page_no for page in (pages or ())} | set(by_page))
    given = {page.page_no: page for page in (pages or ())}

    parts: list[str] = []
    anchors: list[CharAnchor | None] = []
    block_boxes: dict[tuple[int, int], BBox] = {}
    out_pages: list[PageGeometry] = []
    cursor = 0
    next_block_id = 0

    for page_index, page_no in enumerate(page_numbers):
        if page_index:
            for char in page_separator:
                parts.append(char)
                anchors.append(None)
                cursor += 1
        page_start = cursor
        page_blocks = by_page.get(page_no, [])
        for block_index, block in enumerate(page_blocks):
            if block_index:
                for char in block_separator:
                    parts.append(char)
                    anchors.append(None)
                    cursor += 1
            block_id = block.block_id
            if block_id is None:
                block_id = next_block_id
                next_block_id += 1
            else:
                next_block_id = max(next_block_id, block_id + 1)
            if block.bbox is not None:
                block_boxes[(page_no, block_id)] = block.bbox
            per_char = block.char_boxes is not None and len(block.char_boxes) == len(block.text)
            for index, char in enumerate(block.text):
                parts.append(char)
                char_bbox = block.char_boxes[index] if per_char else block.bbox
                anchors.append(
                    CharAnchor(
                        page_no=page_no,
                        block_id=block_id,
                        bbox=char_bbox,
                        pdfium_char_idx=(block.char_indexes[index] if block.char_indexes else None),
                        granularity=(
                            block.granularity if per_char or block.bbox is not None else "block"
                        ),
                    )
                )
                cursor += 1
        page_end = cursor
        source = given.get(page_no)
        out_pages.append(
            PageGeometry(
                page_no=page_no,
                width=source.width if source else 0.0,
                height=source.height if source else 0.0,
                bbox=source.bbox if source else (0.0, 0.0, 0.0, 0.0),
                char_start=page_start,
                char_end=page_end,
            )
        )

    granularity = _document_granularity(anchors)
    return CanonicalText(
        text="".join(parts),
        pages=tuple(out_pages),
        anchors=tuple(anchors),
        block_boxes=block_boxes,
        granularity=granularity,
    )


def _document_granularity(anchors: Iterable[CharAnchor | None]) -> Granularity:
    seen: set[str] = set()
    for anchor in anchors:
        if anchor is not None:
            seen.add(anchor.granularity)
    if not seen:
        return "block"
    if seen == {"glyph"}:
        return "glyph"
    if "block" in seen:
        return "block"
    return "span"


__all__ = [
    "BLOCK_SEPARATOR",
    "PAGE_SEPARATOR",
    "BBox",
    "CanonicalText",
    "CharAnchor",
    "Granularity",
    "PageGeometry",
    "SpanHit",
    "TextBlock",
    "box_inside",
    "build_canonical",
    "merge_boxes",
]
