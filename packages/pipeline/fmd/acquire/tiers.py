"""Tier orchestration: cheapest tier that can actually read the document.

Order (all thresholds in ``config/acquire.yaml``):

1. T0 pdftext. Text layer present and ``text_min_chars_per_page`` met, no
   tables/two-column layout, Latin script → T0 is enough.
2. Tables / two-column layout → T1 Docling (layout + table structure).
3. Image-only pages, a T1 table-confidence below the configured floor, or
   Devanagari text → T2 OCR. ``backends.T2.backend`` selects the engine:
   ``rapidocr`` (in-process PP-OCR via the RapidOCR runner, default on this
   host), ``qwen3vl`` (vLLM endpoint) or ``paddleocr`` (alternate engine).
4. T2's Indic self-check fails → the configured Devanagari fallback
   (``backends.T2.devanagari_fallback``: the Qwen3-VL endpoint, or the T2b
   tier) — never a silent mojibake transcript.
5. Repetition-loop / entropy / length anomaly **in OCR output** → T3 human
   queue, never claims. T0/T1 output is never refused on length: a born-digital
   table is evidence, and the page-median rule is a log diagnostic there.

Heavy dependencies are imported inside the tier functions. A base install
imports this module and runs T0; T1/T2/T2b raise :class:`TierUnavailable` with
the exact fix. CI runs T0 plus injected fakes only.

Budget aborts (page/time) and device failures raise :class:`AcquisitionFailed`
with stable ``acquisition_failed:`` reasons and are **never** converted into a
cheaper-tier fallback: an aborted OCR publishes no partial claims (D18).
"""

from __future__ import annotations

import base64
import importlib
import importlib.metadata
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
from fmd.acquire.errors import (
    ACQUISITION_FAILED_PREFIX,
    AcquisitionFailed,
    ArtifactsUnavailable,
    TierUnavailable,
)
from fmd.acquire.guards import (
    OCR_TIERS,
    born_digital_length_warnings,
    detect_anomalies,
)
from fmd.acquire.rapidocr_backend import (
    RapidOCRBackend,
    RapidOCRSettings,
)
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
#: Stable review reason (never prose): offline mode has no Docling artifacts.
ARTIFACTS_UNAVAILABLE_OFFLINE = "artifacts_unavailable_offline"
#: Stable review reason: OCR could not run and the source is a scan, so the
#: whitespace "text layer" is not evidence (never send it to the LLM).
OCR_UNAVAILABLE_SPARSE_SCAN = "ocr_unavailable_sparse_scan"
_TRUTHY = {"1", "true", "yes", "on"}
_DEVICES = {"cpu", "auto", "cuda"}

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- offline


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def offline_enabled() -> bool:
    """Is the process banned from outbound network access?

    ``FMD_OFFLINE`` is the project switch; honouring ``HF_HUB_OFFLINE`` as well
    means a Hugging Face-only offline flag (set by a container or a shell) still
    produces a fast, named failure instead of a stalled artifact fetch.
    """
    return _enabled(os.environ.get("FMD_OFFLINE")) or _enabled(os.environ.get("HF_HUB_OFFLINE"))


def _enforce_offline_env() -> None:
    """Pin Hub clients offline before a heavy import gets a chance to call out."""
    if not offline_enabled():
        return
    os.environ["FMD_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def docling_device(policy: AcquirePolicy) -> str:
    """Resolve the Docling accelerator from env then registry, default ``cpu``.

    ``config/acquire.yaml:backends.T1.device`` is the source of truth; the
    ``FMD_DOCLING_DEVICE`` env var is a per-run override (useful on a host whose
    GPU is free). Anything outside ``cpu|auto|cuda|cuda:N`` is a config error,
    not a silent ``auto``.
    """
    spec = policy.backend("T1")
    value = str(
        os.environ.get("FMD_DOCLING_DEVICE")
        or getattr(spec, "device", None)
        or "cpu"
    ).strip().lower()
    base = value.split(":", 1)[0]
    if base not in _DEVICES:
        raise AcquisitionFailed(
            f"{ACQUISITION_FAILED_PREFIX}invalid_docling_device",
            f"device {value!r} is not one of cpu|auto|cuda|cuda:N",
        )
    return value


def _preflight_cuda(device: str, policy: AcquirePolicy) -> None:
    """Fail loudly before allocating a model on an already-busy GPU.

    A local LLM server at 64k leaves only a sliver of a 16 GB card. Docling
    would then OOM *after* downloading/loading weights, which used to be caught
    as a generic tier fallback and reported as a clean run with zero claims.
    This check turns that into ``acquisition_failed:insufficient_gpu_memory``
    before the first allocation.
    """
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:  # Docling itself is unusable without torch
        raise AcquisitionFailed(
            f"{ACQUISITION_FAILED_PREFIX}torch_unavailable",
            "device=cuda requires torch (installed with Docling)",
        ) from exc
    if not torch.cuda.is_available():
        raise AcquisitionFailed(
            f"{ACQUISITION_FAILED_PREFIX}cuda_unavailable",
            "device=cuda requested but torch.cuda.is_available() is false",
        )
    raw_min = getattr(policy.backend("T1"), "min_free_vram_gib", 1.0)
    try:
        minimum = float(raw_min)
    except (TypeError, ValueError):
        minimum = 1.0
    free_bytes, _total = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    if free_gib < minimum:
        raise AcquisitionFailed(
            f"{ACQUISITION_FAILED_PREFIX}insufficient_gpu_memory",
            f"only {free_gib:.2f} GiB free, need >= {minimum:.2f} GiB; "
            "stop the local LLM server or set FMD_DOCLING_DEVICE=cpu",
        )


def _is_oom(exc: BaseException) -> bool:
    """True for torch/CUDA allocation failures (type or message based)."""
    if type(exc).__name__ in {"OutOfMemoryError", "CudaError"}:
        return True
    text = str(exc).lower()
    return "out of memory" in text or "outofmemory" in text


def docling_artifacts_path(policy: AcquirePolicy) -> Path | None:
    """The configured Docling artifact directory, or ``None``.

    Registry first (``config/acquire.yaml:backends.T1.artifacts_path``, which
    itself interpolates ``${FMD_DOCLING_ARTIFACTS:-}``), then the environment.
    """
    spec = policy.backend("T1")
    configured = getattr(spec, "artifacts_path", None) or os.environ.get(
        "FMD_DOCLING_ARTIFACTS"
    )
    value = str(configured or "").strip()
    return Path(value).expanduser() if value else None


def _require_docling_artifacts(path: Path | None) -> Path:
    """Fail *before* Docling constructs a model loader that would fetch."""
    if path is None:
        raise ArtifactsUnavailable(
            f"{ARTIFACTS_UNAVAILABLE_OFFLINE}: T1 (Docling) needs a local artifact cache "
            "and none is configured. Set FMD_DOCLING_ARTIFACTS (or "
            "acquire.yaml:backends.T1.artifacts_path) to a directory populated once, "
            "online, with `docling-tools models download`."
        )
    if not path.is_dir() or next(path.iterdir(), None) is None:
        raise ArtifactsUnavailable(
            f"{ARTIFACTS_UNAVAILABLE_OFFLINE}: T1 (Docling) artifact cache {str(path)!r} "
            "is missing or empty. Run `docling-tools models download` once with the "
            "network enabled (FMD_OFFLINE=0, HF_HUB_OFFLINE=0), then point "
            "FMD_DOCLING_ARTIFACTS at that directory."
        )
    return path


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
    def indic_self_check_min_chars(self) -> int:
        return int(getattr(self.acquire.tiers, "t2_indic_self_check_min_chars", 5))

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
    # Whitespace-only text (a scan's page breaks) is not a text layer: a PDF
    # with 14 pages and 26 newlines is image-only, not "sparse text".
    stripped = doc.text.strip()
    chars = len(stripped)
    return ScanProfile(
        page_count=pages,
        char_count=chars,
        chars_per_page=chars / pages,
        devanagari=any(ord(char) in _DEVANAGARI for char in stripped),
        tables_detected=bool(doc.tables)
        or _looks_like_tables(
            stripped,
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
    """Docling layout + tables. Lazy import: the base image does not ship it.

    Offline mode (``FMD_OFFLINE=1``) never lets Docling reach Hugging Face: the
    cache path must already exist, otherwise this raises
    :class:`ArtifactsUnavailable` immediately. An injected ``converter`` is used
    as-is (tests and the native text-only probe need no artifacts).
    """
    policy = policy or AcquirePolicy.load()
    _enforce_offline_env()
    device: str | None = None
    if converter is None:
        artifacts = docling_artifacts_path(policy)
        if offline_enabled():
            _require_docling_artifacts(artifacts)
        device = docling_device(policy)
        if device != "cpu":
            # PyTorch's caching allocator fragments under a co-resident LLM;
            # expandable segments let it grow/shrink instead of failing late.
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _preflight_cuda(device, policy)
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise TierUnavailable(
                "T1 (Docling) is not installed. Install the extraction extra with "
                "`uv sync --all-extras`, or the dispatcher will fall back to T0 text."
            ) from exc
        converter = _docling_converter(DocumentConverter, artifacts, device=device)

    path = _write_temp(data)
    try:
        try:
            result = converter.convert(str(path))
        except Exception as exc:  # docling raises model/runtime specific errors
            if _is_oom(exc):
                raise AcquisitionFailed(
                    f"{ACQUISITION_FAILED_PREFIX}docling_cuda_oom",
                    f"Docling ran out of memory on device {device or 'injected'}: {exc}",
                ) from exc
            raise TierUnavailable(f"Docling conversion failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    document = getattr(result, "document", result)
    return _docling_doc(document, policy)


def _docling_converter(
    factory: Any, artifacts: Path | None, *, device: str | None = None
) -> Any:
    """Build a converter pinned to the configured local artifacts and device."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    options = PdfPipelineOptions()
    # HOTFIX (see prompts/cloud/FMD-CLOUD-014-acquisition-bounds.md): Docling's
    # built-in OCR runs unbounded on CPU for image-only pages and can stall a run
    # for hours. Default off; T2/T3 own OCR. Set FMD_DOCLING_OCR=1 to re-enable.
    if os.environ.get("FMD_DOCLING_OCR", "0") not in {"1", "true", "yes"}:
        options.do_ocr = False
    if artifacts is not None:
        options.artifacts_path = str(artifacts)
    if device is not None:
        from docling.datamodel.accelerator_options import AcceleratorOptions

        options.accelerator_options = AcceleratorOptions(device=device)
    return factory(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


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


#: T2 engine selection matrix (``backends.T2.backend``). ``rapidocr`` runs the
#: PP-OCR ONNX graphs in-process (the runner, not a model); ``qwen3vl`` and
#: ``paddleocr`` speak the shared ``/extract`` HTTP contract through
#: ``endpoint_env``. Qwen3-VL is never the default and never removed.
T2_BACKENDS = ("rapidocr", "qwen3vl", "paddleocr")


def t2_backend(policy: AcquirePolicy) -> str:
    """The configured T2 engine, validated against the selection matrix."""
    spec = policy.backend("T2")
    backend = str(getattr(spec, "backend", "") or "qwen3vl").strip().lower()
    if backend not in T2_BACKENDS:
        raise AcquisitionFailed(
            f"{ACQUISITION_FAILED_PREFIX}unknown_t2_backend",
            f"backends.T2.backend {backend!r} is not one of {', '.join(T2_BACKENDS)}",
        )
    return backend


def _rapidocr_extract(data: bytes, *, spec: Any, policy: AcquirePolicy) -> ExtractedDoc:
    """In-process RapidOCR (PP-OCR) T2 — no HTTP service, no endpoint env."""
    settings = RapidOCRSettings.from_spec(spec, env=dict(os.environ))
    backend = RapidOCRBackend(settings)
    payload = backend.extract(data)
    return _ocr_doc(payload, tier="T2", spec=spec, policy=policy)


def t2_extract(
    data: bytes,
    *,
    backend: OCRBackend | None = None,
    endpoint: str | None = None,
    policy: AcquirePolicy | None = None,
    timeout: float = 120.0,
) -> ExtractedDoc:
    """T2 OCR through the configured backend (rapidocr | qwen3vl | paddleocr)."""
    policy = policy or AcquirePolicy.load()
    spec = policy.backend("T2")
    if backend is not None:
        return _ocr_tier(
            "T2",
            data,
            spec=spec,
            backend=backend,
            endpoint=endpoint,
            timeout=timeout,
            policy=policy,
        )
    if t2_backend(policy) == "rapidocr":
        return _rapidocr_extract(data, spec=spec, policy=policy)
    return _ocr_tier(
        "T2",
        data,
        spec=spec,
        backend=None,
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
    """T2b: the alternate OCR engine (PaddleOCR-VL in this registry).

    Retained and selectable, never deleted. When ``T2.backend`` is not
    ``qwen3vl`` it is also the legacy self-check fallback target; the
    configured ``T2.devanagari_fallback`` picks Qwen3-VL instead.
    """
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
        engine_meta=_engine_meta_of(payload, spec),
    )


def _engine_meta_of(payload: dict[str, Any], spec: Any) -> dict[str, Any] | None:
    """Engine provenance from the payload, synthesised for HTTP backends.

    The in-process RapidOCR backend ships a full ``engine_meta`` block; the
    ``/extract`` HTTP services report ``engine`` / ``tool_version`` / ``model``
    keys, which are folded into the same shape so every OCR path records
    engine + version + model ids (never silently mixed).
    """
    explicit = payload.get("engine_meta")
    if isinstance(explicit, dict):
        return dict(explicit)
    engine = payload.get("engine") or payload.get("tool")
    if engine is None:
        return None
    model_ids = payload.get("model_ids")
    if not isinstance(model_ids, dict):
        model = payload.get("model")
        model_ids = {"model": model} if model else {}
    return {
        "engine": str(engine),
        "engine_version": payload.get("engine_version") or payload.get("tool_version"),
        "model_ids": model_ids,
        "device": payload.get("device"),
        "device_reason": payload.get("device_reason"),
        "granularity": str(spec.granularity),
    }


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
    """Guard OCR/VLM output; born-digital T0/T1 output is never refused.

    Only T2/T2b blocks are model-generated, so only they are checked for
    decoder loops, length explosions and truncation. A T0/T1 document that
    trips the length diagnostic logs a warning and continues to extraction:
    a long born-digital fee table is evidence, not an anomaly.
    """
    if doc.ocr_anomaly:
        return doc
    thresholds = policy.anomaly
    if doc.tier not in OCR_TIERS:
        for message in born_digital_length_warnings(
            doc.blocks,
            block_length_multiplier=thresholds.block_length_multiplier,
            min_block_chars=thresholds.min_block_chars,
        ):
            logger.warning("born-digital length diagnostic (not a refusal): %s", message)
        return doc
    report = detect_anomalies(
        doc.blocks,
        tier=doc.tier,
        entropy_ratio_floor=thresholds.entropy_ratio_floor,
        min_block_chars=thresholds.min_block_chars,
        min_page_chars=thresholds.min_page_chars,
        repeated_block_min_occurrences=thresholds.repeated_block_min_occurrences,
        block_chars_cap=thresholds.ocr_block_chars_cap,
        page_chars_cap=thresholds.ocr_page_chars_cap,
        repeated_substring_ratio_floor=thresholds.ocr_repeated_substring_ratio,
        truncated_page_min_chars=thresholds.ocr_truncated_page_min_chars,
        truncated_page_min_pages=thresholds.ocr_truncated_page_min_pages,
    )
    if report.ocr_anomaly:
        return replace(
            doc,
            ocr_anomaly=True,
            anomaly_reasons=report.reasons,
            anomalous_pages=report.pages,
            guard_trips=report.trips,
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

    t1_note: str | None = None
    if decision.tier == "T1":
        try:
            result = (t1 or t1_extract)(data)
        except AcquisitionFailed:
            # A resource failure is not a "use the flat text instead" signal.
            # Re-raise so orchestration marks the document and the run failed.
            raise
        except ArtifactsUnavailable as exc:
            # Offline with no local layout/table artifacts. A scan has a next
            # tier that needs no Docling artifacts (T2/RapidOCR), so try it
            # before parking the document; a born-digital table/layout question
            # is not answered by T0's flat text, so that still goes to T3.
            if not _scan_needs_ocr(base, policy):
                return replace(
                    base,
                    tier="T3",
                    fallback_from="T1",
                    needs_review=True,
                    review_reason=ARTIFACTS_UNAVAILABLE_OFFLINE,
                    notes=base.notes + (f"T1 unavailable offline: {exc}",),
                )
            decision = TierDecision("T2", decision.reasons + ("t1_artifacts_unavailable",))
            t1_note = f"T1 unavailable offline: {exc}"
        except TierUnavailable as exc:
            if not _scan_needs_ocr(base, policy):
                return _fallback(base, "T1", exc, policy)
            decision = TierDecision("T2", decision.reasons + ("t1_unavailable",))
            t1_note = f"T1 unavailable: {exc}"
        if decision.tier != "T2":
            result = apply_anomaly_guard(result, policy)
            if result.ocr_anomaly or result.needs_review:
                return result
            escalated = plan_tiers(probe(result, policy), policy)
            escalated_reasons = set(escalated.reasons)
            if escalated.tier == "T2" and escalated_reasons & {
                "devanagari",
                "low_table_confidence",
            }:
                decision = TierDecision("T2", decision.reasons + escalated.reasons)
            elif result.is_empty:
                # Docling read no usable text from a scan (or only a page-break
                # whitespace layer): the document is image-only, so OCR is the
                # only tier that can answer. Never hand an empty T1 result to the
                # claim-extraction LLM.
                decision = TierDecision(
                    "T2", decision.reasons + tuple(escalated.reasons or ("empty_t1_output",))
                )
            else:
                return result

    try:
        result = (t2 or t2_extract)(data)
    except AcquisitionFailed:
        # A budget abort (D18) or a device failure is not a "use the flat text
        # instead" signal: no partial claims, the document is parked with the
        # stable reason and the run continues with the next document.
        raise
    except TierUnavailable as exc:
        return _fallback(base, "T2", exc, policy)
    if t1_note:
        result = replace(result, notes=result.notes + (t1_note,))
    result = apply_anomaly_guard(result, policy)
    if result.ocr_anomaly or result.needs_review:
        return result

    if any(ord(char) in _DEVANAGARI for char in result.text):
        devanagari_chars = sum(1 for char in result.text if ord(char) in _DEVANAGARI)
        if devanagari_chars < policy.indic_self_check_min_chars:
            # A stray glyph is not a Hindi page; the fallback engine is not a
            # routine for one OCR misread on an English scan.
            return result
        ratio = indic_self_check(result.text)
        if ratio < policy.indic_self_check_min_ratio:
            fallback_doc, fallback_note = _devanagari_fallback(
                data,
                base=base,
                t2b=t2b,
                policy=policy,
                ratio=ratio,
            )
            if fallback_doc is None:
                return replace(
                    result,
                    tier="T3",
                    needs_review=True,
                    notes=result.notes + (fallback_note or "",),
                )
            fallback_doc = apply_anomaly_guard(fallback_doc, policy)
            if fallback_doc.ocr_anomaly or fallback_doc.needs_review:
                return fallback_doc
            return replace(
                fallback_doc,
                notes=fallback_doc.notes + (fallback_note,),
            )
    return result


def _devanagari_fallback(
    data: bytes,
    *,
    base: ExtractedDoc,
    t2b: Any,
    policy: AcquirePolicy,
    ratio: float,
) -> tuple[ExtractedDoc | None, str | None]:
    """The configured Devanagari fallback after a failed Indic self-check.

    ``backends.T2.devanagari_fallback`` selects the engine: ``qwen3vl`` (the
    vLLM endpoint behind ``backends.T2.endpoint_env`` — Qwen3-VL stays fully
    wired as the Devanagari fallback, never the default, never silent) or
    ``t2b`` (the alternate engine tier, the legacy behaviour). An injected
    ``t2b`` callable keeps its legacy precedence so tests and callers that
    provide one get exactly what they passed.
    """
    spec = policy.backend("T2")
    configured = str(getattr(spec, "devanagari_fallback", "") or "").strip().lower()
    note_prefix = (
        f"T2 Indic self-check {ratio:.3f} < {policy.indic_self_check_min_ratio:g}"
    )
    if t2b is None and configured == "qwen3vl" and t2_backend(policy) != "qwen3vl":
        try:
            fallback = _ocr_tier(
                "T2",
                data,
                spec=spec,
                backend=None,
                endpoint=None,
                timeout=120.0,
                policy=policy,
            )
        except AcquisitionFailed:
            raise
        except TierUnavailable as exc:
            return None, (
                f"{note_prefix}; qwen3vl fallback unavailable: {exc}"
            )
        return fallback, f"{note_prefix}; qwen3vl fallback engine recorded"
    try:
        fallback = (t2b or t2b_extract)(data)
    except TierUnavailable as exc:
        return None, f"{note_prefix}; T2b unavailable: {exc}"
    return fallback, f"{note_prefix}; T2b fallback run"


def _scan_needs_ocr(base: ExtractedDoc, policy: AcquirePolicy) -> bool:
    """True when T0 has no text layer worth trusting (a scan, not a table doc).

    A table-like/sparse born-digital layer is still evidence: only an image-only
    page, or sparse *unstructured* text (page numbers), is treated as a scan.
    """
    profile = probe(base, policy)
    if profile.image_only:
        return True
    return (
        profile.chars_per_page < policy.min_chars_per_page
        and not profile.tables_detected
    )


def _fallback(
    base: ExtractedDoc, tier: str, exc: Exception, policy: AcquirePolicy
) -> ExtractedDoc:
    if base.is_empty:
        raise TierUnavailable(
            f"{tier} was selected but is unavailable ({exc}); T0 also produced no "
            "text, so this document cannot be extracted on this host."
        ) from exc
    if tier in OCR_TIERS and _scan_needs_ocr(base, policy):
        # A near-empty layer on a scan is page furniture, not evidence. Falling
        # back to it would hand a 26-character "document" to the text LLM.
        return replace(
            base,
            tier="T3",
            fallback_from=tier,
            needs_review=True,
            review_reason=OCR_UNAVAILABLE_SPARSE_SCAN,
            notes=base.notes
            + (f"{tier} unavailable: {exc}", "sparse scan: T0 text is not evidence"),
        )
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
    "ARTIFACTS_UNAVAILABLE_OFFLINE",
    "T2_BACKENDS",
    "AcquirePolicy",
    "OCRBackend",
    "TierDecision",
    "TierPolicy",
    "apply_anomaly_guard",
    "docling_artifacts_path",
    "docling_device",
    "indic_self_check",
    "offline_enabled",
    "plan_tiers",
    "probe",
    "run_tiers",
    "select_tier",
    "t0_extract",
    "t1_extract",
    "t2_backend",
    "t2_extract",
    "t2b_extract",
]
