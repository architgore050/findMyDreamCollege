"""Tier orchestration: cheapest tier that can actually read the document.

Order (all thresholds in ``config/acquire.yaml``):

1. T0 pdftext. Text layer present and ``text_min_chars_per_page`` met, no
   tables/two-column layout, Latin script → T0 is enough.
2. Tables / two-column layout → T1 Docling (layout + table structure).
3. Image-only pages, a T1 table-confidence below the configured floor, or
   Devanagari text → T2 PaddleOCR-VL.
4. T2's Indic self-check fails → T2b Qwen3-VL (vLLM).
5. Repetition-loop / entropy anomaly → T3 human queue, never claims.

Heavy dependencies are imported inside the tier functions. A base install
imports this module and runs T0; T1/T2/T2b raise :class:`TierUnavailable` with
the exact fix. CI runs T0 plus injected fakes only.
"""

from __future__ import annotations

import base64
import importlib
import importlib.metadata
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
from fmd.acquire.errors import TierUnavailable
from fmd.acquire.guards import detect_anomalies
from fmd.acquire.tables import associate_headers, table_confidence
from fmd.acquire.types import (
    ExtractedDoc,
    PageSpan,
    ScanProfile,
    Table,
)

TIERS_YAML = Path(__file__).resolve().parents[1] / "extract" / "tiers.yaml"
TIER_ORDER = ("T0", "T1", "T2", "T2b", "T3")
_DEVANAGARI = range(0x0900, 0x0980)


# --------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class AcquirePolicy:
    """Acquisition thresholds (registry) plus router profile names (tiers.yaml)."""

    acquire: Any
    profiles: dict[str, str]
    adjudication_sample_rate: float

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        acquire: Any | None = None,
    ) -> AcquirePolicy:
        source = Path(path) if path is not None else TIERS_YAML
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{source}: tier policy must be a mapping")
        profiles = dict(data.get("profiles") or {})
        if acquire is None:
            from fmd.config import get_registry

            acquire = get_registry().acquire
        return cls(
            acquire=acquire,
            profiles={str(key): str(value) for key, value in profiles.items()},
            adjudication_sample_rate=float(acquire.tiers.adjudication_sample_rate),
        )

    @property
    def min_chars_per_page(self) -> int:
        return int(self.acquire.tiers.text_min_chars_per_page)

    @property
    def table_min_rows(self) -> int:
        return int(self.acquire.tiers.table_min_rows)

    @property
    def table_min_numeric_columns(self) -> int:
        return int(self.acquire.tiers.table_min_numeric_columns)

    @property
    def devanagari_forces_t2(self) -> bool:
        return bool(self.acquire.tiers.devanagari_forces_t2)

    @property
    def t1_table_confidence_min(self) -> float:
        return float(self.acquire.tiers.t1_table_confidence_min)

    @property
    def indic_self_check_min_ratio(self) -> float:
        return float(self.acquire.tiers.t2_indic_self_check_min_ratio)

    @property
    def anomaly(self) -> Any:
        return self.acquire.anomaly

    @property
    def bbox(self) -> Any:
        return self.acquire.bbox

    def backend(self, tier: str) -> Any:
        try:
            return self.acquire.backends[tier]
        except KeyError as exc:  # validated at registry load; defensive here
            raise TierUnavailable(f"acquire.yaml has no backend configured for {tier}") from exc

    def profile_for(self, tier: str) -> str:
        return self.profiles.get(tier, "bulk_extract")


# Backwards-compatible name used by the extraction package.
TierPolicy = AcquirePolicy


@dataclass(frozen=True)
class TierDecision:
    tier: str
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"tier": self.tier, "reasons": list(self.reasons)}


# --------------------------------------------------------------------------- probing


def probe(doc: ExtractedDoc, policy: AcquirePolicy | None = None) -> ScanProfile:
    policy = policy or AcquirePolicy.load()
    pages = max(len(doc.pages or doc.page_map), 1)
    chars = len(doc.text)
    return ScanProfile(
        page_count=pages,
        char_count=chars,
        chars_per_page=chars / pages,
        devanagari=any(ord(char) in _DEVANAGARI for char in doc.text),
        tables_detected=bool(doc.tables)
        or _looks_like_tables(
            doc.text,
            min_rows=policy.table_min_rows,
            min_numeric_columns=policy.table_min_numeric_columns,
        ),
        two_column=doc.two_column,
        table_confidence=doc.table_confidence,
        image_only=chars == 0,
    )


def plan_tiers(profile: ScanProfile, policy: AcquirePolicy | None = None) -> TierDecision:
    """Cheapest tier that can plausibly answer, with stable reason codes."""
    policy = policy or AcquirePolicy.load()
    reasons: list[str] = []
    if policy.devanagari_forces_t2 and profile.devanagari:
        reasons.append("devanagari")
        return TierDecision("T2", tuple(reasons))
    if profile.image_only:
        reasons.append("image_only")
        return TierDecision("T2", tuple(reasons))
    if profile.tables_detected:
        reasons.append("tables_detected")
    if profile.two_column:
        reasons.append("two_column")
    if (
        profile.table_confidence is not None
        and profile.table_confidence < policy.t1_table_confidence_min
    ):
        reasons.append("low_table_confidence")
        return TierDecision("T2", tuple(reasons))
    if profile.chars_per_page < policy.min_chars_per_page:
        reasons.append("sparse_text_layer")
        return TierDecision("T1", tuple(reasons))
    if reasons:
        return TierDecision("T1", tuple(reasons))
    return TierDecision("T0", ("clean_text_layer",))


def select_tier(profile: ScanProfile, policy: AcquirePolicy | None = None) -> str:
    return plan_tiers(profile, policy).tier


def _looks_like_tables(text: str, *, min_rows: int, min_numeric_columns: int) -> bool:
    rows = 0
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < min_numeric_columns + 1:
            continue
        numeric = sum(1 for token in tokens if _is_numeric_token(token))
        if numeric >= min_numeric_columns:
            rows += 1
            if rows >= min_rows:
                return True
    return False


def _is_numeric_token(token: str) -> bool:
    cleaned = token.strip(",.;:()[]|%₹$£€")
    return bool(cleaned) and cleaned.replace(".", "", 1).replace(",", "").isdigit()


def _two_column_page(page: dict[str, Any]) -> bool:
    blocks = page.get("blocks") or []
    if len(blocks) < 4:
        return False
    width = float(page.get("width") or 0)
    if width <= 0:
        return False
    midpoint = width / 2
    left = sum(1 for block in blocks if float(block["bbox"][2]) <= midpoint)
    right = sum(1 for block in blocks if float(block["bbox"][0]) >= midpoint)
    return left >= 2 and right >= 2


# --------------------------------------------------------------------------- T0


def t0_extract(data: bytes, *, policy: AcquirePolicy | None = None) -> ExtractedDoc:
    """Born-digital text via pdftext (lazy import; per-glyph provenance)."""
    extraction = _import_pdftext()
    path = _write_temp(data)
    try:
        try:
            pages = extraction.dictionary_output(
                str(path), sort=True, keep_chars=True, flatten_pdf=False, disable_links=True
            )
        except Exception as exc:  # pdftext/pdfium error types vary by version
            raise TierUnavailable(f"pdftext could not read the PDF: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    return _pdftext_doc(pages)


def _import_pdftext() -> Any:
    try:
        return importlib.import_module("pdftext.extraction")
    except ImportError as exc:
        raise TierUnavailable(
            "T0 (pdftext) is not installed. Install the extraction extra with "
            "`uv sync --all-extras`, or the dispatcher will report the missing tier."
        ) from exc


def _pdftext_doc(pages: list[dict[str, Any]]) -> ExtractedDoc:
    blocks: list[TextBlock] = []
    geoms: list[PageGeometry] = []
    two_column = False
    for page in pages:
        page_no = int(page.get("page", 0)) + 1
        if _two_column_page(page):
            two_column = True
        width = float(page.get("width") or 0)
        height = float(page.get("height") or 0)
        bbox = tuple(float(value) for value in page.get("bbox") or (0, 0, width, height))
        geoms.append(
            PageGeometry(
                page_no=page_no,
                width=width,
                height=height,
                bbox=bbox,  # type: ignore[arg-type]
                char_start=0,
                char_end=0,
            )
        )
        for block in page.get("blocks") or []:
            for line in block.get("lines") or []:
                line_parts: list[str] = []
                line_boxes: list[tuple[float, float, float, float]] = []
                line_indexes: list[int] = []
                glyph = True
                for span in line.get("spans") or []:
                    text = str(span.get("text") or "")
                    if not text or not text.strip():
                        continue
                    line_parts.append(text)
                    chars = span.get("chars") or []
                    if len(chars) == len(text):
                        line_boxes.extend(
                            tuple(float(value) for value in char["bbox"]) for char in chars
                        )
                        line_indexes.extend(int(char["char_idx"]) for char in chars)
                    else:
                        glyph = False
                line_text = "".join(line_parts)
                if not line_text:
                    continue
                line_bbox = line.get("bbox")
                bbox = tuple(float(value) for value in line_bbox) if line_bbox is not None else None
                blocks.append(
                    TextBlock(
                        page_no=page_no,
                        text=line_text,
                        bbox=bbox,  # type: ignore[arg-type]
                        granularity="glyph" if glyph else "span",
                        char_boxes=tuple(line_boxes) if glyph else None,
                        char_indexes=tuple(line_indexes) if glyph else None,
                    )
                )

    canonical = build_canonical(blocks, geoms)
    return ExtractedDoc(
        text=canonical.text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=[],
        tier="T0",
        tool="pdftext",
        tool_version=_version("pdftext"),
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity=canonical.granularity,
        two_column=two_column,
    )


# --------------------------------------------------------------------------- T1


def t1_extract(
    data: bytes,
    *,
    converter: Any | None = None,
    policy: AcquirePolicy | None = None,
) -> ExtractedDoc:
    """Docling layout + tables. Lazy import: the base image does not ship it."""
    policy = policy or AcquirePolicy.load()
    if converter is None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise TierUnavailable(
                "T1 (Docling) is not installed. Install the extraction extra with "
                "`uv sync --all-extras`, or the dispatcher will fall back to T0 text."
            ) from exc
        converter = DocumentConverter()

    path = _write_temp(data)
    try:
        try:
            result = converter.convert(str(path))
        except Exception as exc:  # docling raises model/runtime specific errors
            raise TierUnavailable(f"Docling conversion failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    document = getattr(result, "document", result)
    return _docling_doc(document, policy)


def _docling_doc(document: Any, policy: AcquirePolicy) -> ExtractedDoc:
    pages_meta = getattr(document, "pages", None) or {}
    heights: dict[int, float] = {}
    widths: dict[int, float] = {}
    for page_no, page in pages_meta.items():
        size = getattr(page, "size", None)
        heights[int(getattr(page, "page_no", page_no))] = float(getattr(size, "height", 0.0) or 0.0)
        widths[int(getattr(page, "page_no", page_no))] = float(getattr(size, "width", 0.0) or 0.0)

    blocks: list[TextBlock] = []
    tables: list[Table] = []
    for item, _level in document.iterate_items():
        page_no, bbox = _docling_prov(item, heights)
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip() and bbox is not None:
            blocks.append(TextBlock(page_no=page_no, text=text, bbox=bbox, granularity="span"))
        data = _field(item, "data")
        cells = _field(data, "table_cells")
        if cells:
            association = associate_headers(cells)
            conf = table_confidence(cells)
            rows = _docling_rows(data, cells)
            tables.append(
                Table(
                    rows=rows,
                    page=page_no,
                    cells=association.cells,
                    column_header_uncertain=association.column_header_uncertain,
                    confidence=conf,
                )
            )
            for cell in cells:
                cell_text = str(_field(cell, "text", "") or "")
                cell_bbox = _cell_bbox(cell, page_no, heights)
                if cell_text.strip() and cell_bbox is not None:
                    blocks.append(
                        TextBlock(
                            page_no=page_no,
                            text=cell_text,
                            bbox=cell_bbox,
                            granularity="span",
                        )
                    )

    geoms = [
        PageGeometry(
            page_no=page_no,
            width=widths.get(page_no, 0.0),
            height=heights.get(page_no, 0.0),
            bbox=(0.0, 0.0, widths.get(page_no, 0.0), heights.get(page_no, 0.0)),
            char_start=0,
            char_end=0,
        )
        for page_no in sorted(heights or {1: 0.0})
    ]
    if not geoms and blocks:
        pages = sorted({block.page_no for block in blocks})
        geoms = [
            PageGeometry(
                page_no=p,
                width=0.0,
                height=0.0,
                bbox=(0.0, 0.0, 0.0, 0.0),
                char_start=0,
                char_end=0,
            )
            for p in pages
        ]
    canonical = build_canonical(blocks, geoms)
    confidences = [table.confidence for table in tables if table.confidence is not None]
    return ExtractedDoc(
        text=canonical.text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=tables,
        tier="T1",
        tool="docling",
        tool_version=_version("docling"),
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity=canonical.granularity,
        table_confidence=min(confidences) if confidences else None,
    )


def _docling_prov(
    item: Any, heights: dict[int, float]
) -> tuple[int, tuple[float, float, float, float] | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return 1, None
    first = prov[0]
    page_no = int(getattr(first, "page_no", 1) or 1)
    bbox = getattr(first, "bbox", None)
    return page_no, _to_top_left(bbox, heights.get(page_no))


def _field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _cell_bbox(
    cell: Any, page_no: int, heights: dict[int, float]
) -> tuple[float, float, float, float] | None:
    return _to_top_left(_field(cell, "bbox"), heights.get(page_no))


def _to_top_left(bbox: Any, page_height: float | None) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    try:
        if hasattr(bbox, "to_top_left_origin"):
            converted = bbox.to_top_left_origin(page_height or 0.0)
            return (
                float(converted.l),
                float(converted.t),
                float(converted.r),
                float(converted.b),
            )
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError, AttributeError):
        return None
    if len(values) != 4:
        return None
    return values  # type: ignore[return-value]


def _docling_rows(data: Any, cells: list[Any]) -> list[list[str]]:
    num_rows = int(_field(data, "num_rows", 0) or 0)
    num_cols = int(_field(data, "num_cols", 0) or 0)
    if num_rows <= 0 or num_cols <= 0:
        return []
    rows = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells:
        row = int(_field(cell, "start_row_offset_idx", 0) or 0)
        col = int(_field(cell, "start_col_offset_idx", 0) or 0)
        if 0 <= row < num_rows and 0 <= col < num_cols:
            rows[row][col] = str(_field(cell, "text", "") or "")
    return rows


# --------------------------------------------------------------------------- T2 / T2b


class OCRBackend(Protocol):
    def extract(self, data: bytes) -> dict[str, Any]: ...


def t2_extract(
    data: bytes,
    *,
    backend: OCRBackend | None = None,
    endpoint: str | None = None,
    policy: AcquirePolicy | None = None,
    timeout: float = 120.0,
) -> ExtractedDoc:
    """PaddleOCR-VL (block-level boxes) behind a backend or HTTP endpoint."""
    policy = policy or AcquirePolicy.load()
    spec = policy.backend("T2")
    return _ocr_tier(
        "T2",
        data,
        spec=spec,
        backend=backend,
        endpoint=endpoint,
        timeout=timeout,
        policy=policy,
    )


def t2b_extract(
    data: bytes,
    *,
    backend: OCRBackend | None = None,
    endpoint: str | None = None,
    policy: AcquirePolicy | None = None,
    timeout: float = 120.0,
) -> ExtractedDoc:
    """Qwen3-VL Devanagari fallback (vLLM). Interface-only without an endpoint."""
    policy = policy or AcquirePolicy.load()
    spec = policy.backend("T2b")
    return _ocr_tier(
        "T2b",
        data,
        spec=spec,
        backend=backend,
        endpoint=endpoint,
        timeout=timeout,
        policy=policy,
    )


def _ocr_tier(
    tier: str,
    data: bytes,
    *,
    spec: Any,
    backend: OCRBackend | None,
    endpoint: str | None,
    timeout: float,
    policy: AcquirePolicy,
) -> ExtractedDoc:
    if backend is not None:
        payload = backend.extract(data)
        return _ocr_doc(payload, tier=tier, spec=spec, policy=policy)
    base = (endpoint or os.environ.get(spec.endpoint_env or "") or "").strip()
    if not base:
        raise TierUnavailable(
            f"{tier} ({spec.model}) needs a vLLM endpoint: set {spec.endpoint_env} or pass "
            "a backend. This tier is interface-only until an endpoint is configured."
        )
    try:
        response = httpx.post(
            base.rstrip("/") + "/extract",
            json={"pdf_base64": base64.b64encode(data).decode("ascii")},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise TierUnavailable(f"{tier} endpoint {base!r} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise TierUnavailable(f"{tier} endpoint returned a non-object response")
    return _ocr_doc(payload, tier=tier, spec=spec, policy=policy)


def _ocr_doc(
    payload: dict[str, Any],
    *,
    tier: str,
    spec: Any,
    policy: AcquirePolicy,
) -> ExtractedDoc:
    raw_blocks = payload.get("blocks") or []
    blocks: list[TextBlock] = []
    if raw_blocks:
        for order, raw in enumerate(raw_blocks):
            if not isinstance(raw, dict):
                continue
            text = raw.get("block_content") or raw.get("text") or ""
            if not str(text).strip():
                continue
            page_no = int(raw.get("page") or raw.get("page_no") or 1)
            bbox = raw.get("block_bbox") or raw.get("bbox")
            blocks.append(
                TextBlock(
                    page_no=page_no,
                    text=str(text),
                    bbox=(tuple(float(value) for value in bbox) if bbox is not None else None),  # type: ignore[arg-type]
                    granularity="block",
                    block_id=int(raw.get("block_order", order)),
                )
            )
        blocks.sort(key=lambda block: (block.page_no, block.block_id or 0))
        geoms = _ocr_pages(payload, blocks)
        canonical = build_canonical(blocks, geoms)
        text = canonical.text
    else:
        text = str(payload.get("text") or "")
        if not text.strip():
            raise TierUnavailable(f"{tier} backend returned no text")
        page_bbox = payload.get("page_bbox")
        canonical = build_canonical(
            [
                TextBlock(
                    page_no=1,
                    text=text,
                    bbox=(
                        tuple(float(value) for value in page_bbox)
                        if page_bbox is not None
                        else None
                    ),  # type: ignore[arg-type]
                    granularity="block",
                )
            ],
            [
                PageGeometry(
                    page_no=1,
                    width=float(payload.get("width") or 0.0),
                    height=float(payload.get("height") or 0.0),
                    bbox=(
                        tuple(float(value) for value in page_bbox)
                        if page_bbox is not None
                        else (0.0, 0.0, 0.0, 0.0)
                    ),  # type: ignore[arg-type]
                    char_start=0,
                    char_end=0,
                )
            ],
        )

    tables = _ocr_tables(payload.get("tables") or [])
    confidences = [table.confidence for table in tables if table.confidence is not None]
    table_conf = payload.get("table_confidence")
    if table_conf is None and confidences:
        table_conf = min(confidences)
    return ExtractedDoc(
        text=text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=tables,
        tier=tier,
        tool=str(payload.get("tool") or spec.tool),
        tool_version=payload.get("tool_version"),
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity=str(spec.granularity),
        table_confidence=float(table_conf) if table_conf is not None else None,
    )


def _ocr_pages(payload: dict[str, Any], blocks: list[TextBlock]) -> list[PageGeometry]:
    raw_pages = payload.get("pages") or []
    geoms: list[PageGeometry] = []
    for raw in raw_pages:
        if not isinstance(raw, dict):
            continue
        page_no = int(raw.get("page") or raw.get("page_no") or 1)
        width = float(raw.get("width") or 0.0)
        height = float(raw.get("height") or 0.0)
        bbox = raw.get("bbox") or (0.0, 0.0, width, height)
        geoms.append(
            PageGeometry(
                page_no=page_no,
                width=width,
                height=height,
                bbox=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
                char_start=0,
                char_end=0,
            )
        )
    if geoms:
        return geoms
    pages = sorted({block.page_no for block in blocks})
    for page_no in pages:
        boxes = [block.bbox for block in blocks if block.page_no == page_no and block.bbox]
        if boxes:
            bbox = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)
        geoms.append(
            PageGeometry(
                page_no=page_no,
                width=bbox[2],
                height=bbox[3],
                bbox=bbox,
                char_start=0,
                char_end=0,
            )
        )
    return geoms


def _ocr_tables(raw_tables: list[Any]) -> list[Table]:
    tables: list[Table] = []
    for raw in raw_tables:
        rows: list[list[str]] | None = None
        page: int | None = None
        if isinstance(raw, dict):
            candidate = raw.get("rows")
            if isinstance(candidate, list):
                rows = [[str(cell) for cell in row] for row in candidate if isinstance(row, list)]
            page = raw.get("page")
        elif isinstance(raw, list) and raw and all(isinstance(row, list) for row in raw):
            rows = [[str(cell) for cell in row] for row in raw]
        if rows:
            tables.append(Table(rows=rows, page=page))
    return tables


def indic_self_check(text: str) -> float:
    """Share of script letters that are Devanagari on Devanagari-bearing lines.

    A decoder that emits Latin mojibake for a Hindi block scores near 0; a
    correct (possibly English-mixed) Hindi block scores high. Deterministic and
    model-free.
    """
    relevant = [
        line for line in text.splitlines() if any(ord(char) in _DEVANAGARI for char in line)
    ]
    if not relevant:
        return 1.0
    devanagari = 0
    latin = 0
    for line in relevant:
        for char in line:
            if ord(char) in _DEVANAGARI:
                devanagari += 1
            elif char.isascii() and char.isalpha():
                latin += 1
    total = devanagari + latin
    return 1.0 if total == 0 else devanagari / total


# --------------------------------------------------------------------------- orchestration


def apply_anomaly_guard(doc: ExtractedDoc, policy: AcquirePolicy) -> ExtractedDoc:
    """Mark anomalous blocks/pages and route the document to T3."""
    if not doc.ocr_anomaly:
        thresholds = policy.anomaly
        report = detect_anomalies(
            doc.blocks,
            block_length_multiplier=thresholds.block_length_multiplier,
            entropy_ratio_floor=thresholds.entropy_ratio_floor,
            min_block_chars=thresholds.min_block_chars,
            min_page_chars=thresholds.min_page_chars,
            repeated_block_min_occurrences=thresholds.repeated_block_min_occurrences,
        )
        if report.ocr_anomaly:
            return replace(
                doc,
                ocr_anomaly=True,
                anomaly_reasons=report.reasons,
                anomalous_pages=report.pages,
                needs_review=True,
                tier="T3",
            )
    return doc


def run_tiers(
    data: bytes,
    *,
    policy: AcquirePolicy | None = None,
    t0: Any = None,
    t1: Any = None,
    t2: Any = None,
    t2b: Any = None,
) -> ExtractedDoc:
    """Probe with T0, escalate only for a documented reason, fall back loudly."""
    policy = policy or AcquirePolicy.load()
    base = apply_anomaly_guard((t0 or t0_extract)(data), policy)
    if base.ocr_anomaly:
        return base
    decision = plan_tiers(probe(base, policy), policy)
    if decision.tier == "T0":
        return base

    if decision.tier == "T1":
        try:
            result = (t1 or t1_extract)(data)
        except TierUnavailable as exc:
            return _fallback(base, "T1", exc)
        result = apply_anomaly_guard(result, policy)
        if result.ocr_anomaly or result.needs_review:
            return result
        escalated = plan_tiers(probe(result, policy), policy)
        if escalated.tier == "T2" and set(escalated.reasons) & {
            "devanagari",
            "low_table_confidence",
        }:
            decision = TierDecision("T2", decision.reasons + escalated.reasons)
        else:
            return result

    try:
        result = (t2 or t2_extract)(data)
    except TierUnavailable as exc:
        return _fallback(base, "T2", exc)
    result = apply_anomaly_guard(result, policy)
    if result.ocr_anomaly or result.needs_review:
        return result

    if any(ord(char) in _DEVANAGARI for char in result.text):
        ratio = indic_self_check(result.text)
        if ratio < policy.indic_self_check_min_ratio:
            try:
                fallback = (t2b or t2b_extract)(data)
            except TierUnavailable as exc:
                return replace(
                    result,
                    tier="T3",
                    needs_review=True,
                    notes=result.notes
                    + (
                        f"T2 Indic self-check {ratio:.3f} < "
                        f"{policy.indic_self_check_min_ratio:g}; T2b unavailable: {exc}",
                    ),
                )
            fallback = apply_anomaly_guard(fallback, policy)
            if fallback.ocr_anomaly or fallback.needs_review:
                return fallback
            return replace(
                fallback,
                notes=fallback.notes + (f"T2b after Indic self-check {ratio:.3f}",),
            )
    return result


def _fallback(base: ExtractedDoc, tier: str, exc: Exception) -> ExtractedDoc:
    if base.is_empty:
        raise TierUnavailable(
            f"{tier} was selected but is unavailable ({exc}); T0 also produced no "
            "text, so this document cannot be extracted on this host."
        ) from exc
    return replace(
        base,
        tier="T0",
        fallback_from=tier,
        notes=base.notes + (f"{tier} unavailable: {exc}",),
    )


# --------------------------------------------------------------------------- helpers


def _write_temp(data: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(data)
        handle.flush()
        return Path(handle.name)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = [
    "AcquirePolicy",
    "OCRBackend",
    "TierDecision",
    "TierPolicy",
    "apply_anomaly_guard",
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
