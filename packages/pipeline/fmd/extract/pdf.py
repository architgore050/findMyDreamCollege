"""Tiered document text extraction (brief §9).

T0 ``pypdfium2`` / ``pdftext`` — born-digital text, the cheap default.
T1 Docling — layout + tables on clean scans.
T2 MinerU / dots.ocr on vLLM — scanned tables and Devanagari headers.
T3 VLM (olmOCR-class) — deterministic 1% field adjudication and failures.

Heavy tiers are imported lazily: a base install can still import this module and
run T0/HTML extraction. When a tier's dependency or endpoint is missing the tier
raises :class:`TierUnavailable`; dispatch falls back only when T0 already
produced usable text, otherwise the failure is explicit.
"""

from __future__ import annotations

import base64
import hashlib
import html
import importlib
import importlib.metadata
import logging
import os
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

logger = logging.getLogger(__name__)

TIERS_YAML = Path(__file__).with_name("tiers.yaml")
TIER_ORDER = ("T0", "T1", "T2", "T3")


class TierUnavailable(RuntimeError):
    """A tier cannot run on this install/host. Never a silent empty result."""


# --------------------------------------------------------------------------- results


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

    def as_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "page": self.page, "caption": self.caption}


@dataclass(frozen=True)
class ExtractedDoc:
    text: str
    page_map: list[PageSpan]
    tables: list[Table]
    tier: str
    tool: str
    tool_version: str | None = None
    fallback_from: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def to_structure(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "tool": self.tool,
            "tool_version": self.tool_version,
            "fallback_from": self.fallback_from,
            "notes": list(self.notes),
            "page_map": [span.as_dict() for span in self.page_map],
            "tables": [table.as_dict() for table in self.tables],
        }


@dataclass(frozen=True)
class ScanProfile:
    page_count: int
    char_count: int
    chars_per_page: float
    devanagari: bool
    tables_detected: bool


# --------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class TierPolicy:
    min_chars_per_page: int = 80
    table_min_rows: int = 3
    table_min_numeric_columns: int = 2
    devanagari_forces_t2: bool = True
    profiles: dict[str, str] = field(
        default_factory=lambda: {
            "T0": "bulk_extract",
            "T1": "bulk_extract",
            "T2": "hard_pdf",
            "T3": "hard_pdf",
        }
    )
    adjudication_sample_rate: float = 0.01

    @classmethod
    def load(cls, path: str | Path | None = None) -> TierPolicy:
        source = Path(path) if path is not None else TIERS_YAML
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{source}: tier policy must be a mapping")
        known = {
            "min_chars_per_page",
            "table_min_rows",
            "table_min_numeric_columns",
            "devanagari_forces_t2",
            "profiles",
            "adjudication_sample_rate",
        }
        return cls(**{key: value for key, value in data.items() if key in known})

    def profile_for(self, tier: str) -> str:
        return self.profiles.get(tier, "bulk_extract")


# --------------------------------------------------------------------------- probing

_DEVANAGARI = range(0x0900, 0x0980)


def probe(doc: ExtractedDoc, policy: TierPolicy | None = None) -> ScanProfile:
    policy = policy or TierPolicy.load()
    text = doc.text
    pages = max(len(doc.page_map), 1)
    chars = len(text)
    tables = _looks_like_tables(
        text,
        min_rows=policy.table_min_rows,
        min_numeric_columns=policy.table_min_numeric_columns,
    )
    return ScanProfile(
        page_count=pages,
        char_count=chars,
        chars_per_page=chars / pages,
        devanagari=any(ord(char) in _DEVANAGARI for char in text),
        tables_detected=tables,
    )


def select_tier(profile: ScanProfile, policy: TierPolicy | None = None) -> str:
    """Cheapest tier that can plausibly read this document."""
    policy = policy or TierPolicy.load()
    if profile.char_count == 0:
        if policy.devanagari_forces_t2 and profile.devanagari:
            return "T2"
        return "T1"
    if profile.chars_per_page < policy.min_chars_per_page:
        if policy.devanagari_forces_t2 and profile.devanagari:
            return "T2"
        return "T1"
    if profile.tables_detected:
        return "T1"
    return "T0"


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


# --------------------------------------------------------------------------- tiers


def t0_extract(data: bytes) -> ExtractedDoc:
    """Born-digital text via pypdfium2 (lazy import; no pdftext dependency)."""
    pdfium = _import("pypdfium2")
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:  # pypdfium2 raises its own error types
        raise TierUnavailable(f"pypdfium2 could not open the PDF: {exc}") from exc

    chunks: list[str] = []
    page_map: list[PageSpan] = []
    offset = 0
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            page_text = textpage.get_text_bounded() or ""
            if index:
                offset += 2  # the "\n\n" separator
            start = offset
            chunks.append(page_text)
            offset = start + len(page_text)
            page_map.append(PageSpan(page=index + 1, char_start=start, char_end=offset))
    finally:
        close = getattr(pdf, "close", None)
        if callable(close):
            close()

    return ExtractedDoc(
        text="\n\n".join(chunks),
        page_map=page_map,
        tables=[],
        tier="T0",
        tool="pypdfium2",
        tool_version=_version("pypdfium2"),
    )


def t1_extract(data: bytes, *, fallback: ExtractedDoc | None = None) -> ExtractedDoc:
    """Docling layout + tables. Lazy import: the base image does not ship it."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise TierUnavailable(
            "T1 (Docling) is not installed. Install the extraction extra with "
            "`uv sync --extra extract`, or the dispatcher will fall back to T0 text."
        ) from exc

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(data)
            handle.flush()
            result = DocumentConverter().convert(handle.name)
        document = getattr(result, "document", result)
        text = document.export_to_markdown()
    except Exception as exc:  # docling raises model/runtime specific errors
        raise TierUnavailable(f"Docling conversion failed: {exc}") from exc

    tables = _docling_tables(document)
    return ExtractedDoc(
        text=text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(text))],
        tables=tables,
        tier="T1",
        tool="docling",
        tool_version=_version("docling"),
    )


def _docling_tables(document: Any) -> list[Table]:
    try:
        exported = document.export_to_dict()
    except Exception:  # pragma: no cover - depends on docling version
        return []
    raw = exported.get("tables") if isinstance(exported, dict) else None
    rows: list[list[str]] = []
    if isinstance(raw, list):
        for item in raw:
            data = item.get("data") if isinstance(item, dict) else None
            if isinstance(data, list):
                rows.append([str(cell) for cell in data])
    return [Table(rows=rows)] if rows else []


class T2Backend(Protocol):
    def extract(self, data: bytes) -> dict[str, Any]: ...


def t2_extract(
    data: bytes,
    *,
    backend: T2Backend | None = None,
    endpoint: str | None = None,
    timeout: float = 120.0,
) -> ExtractedDoc:
    """MinerU / dots.ocr on vLLM.

    There is no universal wire format yet, so this is an interface: pass a
    ``backend`` or point ``FMD_MINERU_URL`` at a MinerU-compatible HTTP server
    that answers ``POST <endpoint>/extract`` with
    ``{"text": str, "tables": [[...]], "tool": str, "tool_version": str}``.
    Without either, the tier fails loudly instead of returning empty text.
    """
    if backend is not None:
        payload = backend.extract(data)
        return _t2_doc(payload)
    base = (endpoint or os.environ.get("FMD_MINERU_URL") or "").strip()
    if not base:
        raise TierUnavailable(
            "T2 (MinerU/dots.ocr) needs a vLLM endpoint: set FMD_MINERU_URL or pass "
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
        raise TierUnavailable(f"T2 endpoint {base!r} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise TierUnavailable("T2 endpoint returned a non-object response")
    return _t2_doc(payload)


def _t2_doc(payload: dict[str, Any]) -> ExtractedDoc:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise TierUnavailable("T2 backend returned no text")
    raw_tables = payload.get("tables") or []
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
    return ExtractedDoc(
        text=text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(text))],
        tables=tables,
        tier="T2",
        tool=str(payload.get("tool") or "mineru"),
        tool_version=payload.get("tool_version"),
    )


# --------------------------------------------------------------------------- dispatch


def extract_pdf(
    data: bytes,
    *,
    policy: TierPolicy | None = None,
    t0: Any = None,
    t1: Any = None,
    t2: Any = None,
) -> ExtractedDoc:
    """Probe with T0, then run the selected tier with explicit fallbacks."""
    policy = policy or TierPolicy.load()
    base = (t0 or t0_extract)(data)
    profile = probe(base, policy)
    tier = select_tier(profile, policy)
    logger.debug(
        "pdf tier dispatch tier=%s pages=%d chars_per_page=%.1f tables=%s devanagari=%s",
        tier,
        profile.page_count,
        profile.chars_per_page,
        profile.tables_detected,
        profile.devanagari,
    )
    if tier == "T0":
        return base

    runner = (t1 or t1_extract) if tier == "T1" else (t2 or t2_extract)
    try:
        result: ExtractedDoc = runner(data)
    except TierUnavailable as exc:
        if base.is_empty:
            raise TierUnavailable(
                f"{tier} was selected but is unavailable ({exc}); T0 also produced no "
                "text, so this document cannot be extracted on this host."
            ) from exc
        logger.warning("tier %s unavailable (%s); falling back to T0 text", tier, exc)
        return ExtractedDoc(
            text=base.text,
            page_map=base.page_map,
            tables=base.tables,
            tier="T0",
            tool=base.tool,
            tool_version=base.tool_version,
            fallback_from=tier,
            notes=(f"{tier} unavailable: {exc}",),
        )
    if result.is_empty:
        raise TierUnavailable(f"{tier} produced no text for this document")
    return result


def extract_document(
    data: bytes,
    *,
    content_type: str | None = None,
    policy: TierPolicy | None = None,
    t0: Any = None,
    t1: Any = None,
    t2: Any = None,
) -> ExtractedDoc:
    """Entry point used by the pipeline: PDFs are tiered, everything else decoded."""
    if _is_pdf(data, content_type):
        return extract_pdf(data, policy=policy, t0=t0, t1=t1, t2=t2)
    if content_type and "html" in content_type.lower():
        return _html_doc(data)
    return _text_doc(data)


def _is_pdf(data: bytes, content_type: str | None) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return data.lstrip()[:5] == b"%PDF-"


def _text_doc(data: bytes) -> ExtractedDoc:
    text = data.decode("utf-8", errors="replace")
    return ExtractedDoc(
        text=text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(text))],
        tables=[],
        tier="T0",
        tool="text-decoder",
        tool_version=None,
    )


class _TextOnlyHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _html_doc(data: bytes) -> ExtractedDoc:
    parser = _TextOnlyHTML()
    parser.feed(data.decode("utf-8", errors="replace"))
    text = html.unescape("\n".join(part.strip() for part in parser.parts if part.strip()))
    return ExtractedDoc(
        text=text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(text))],
        tables=[],
        tier="T0",
        tool="html-text",
        tool_version=None,
    )


# --------------------------------------------------------------------------- T3 audit


def _item_key(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("key") or item.get("field") or item)
    if isinstance(item, tuple | list):
        return "\x1f".join(str(part) for part in item)
    return repr(item)


def sample_for_adjudication(
    items: list[Any], *, rate: float | None = None, policy: TierPolicy | None = None
) -> list[Any]:
    """Deterministic 1% sample keyed by sha256 — reruns select the same items."""
    policy = policy or TierPolicy.load()
    threshold = (rate if rate is not None else policy.adjudication_sample_rate) * 10_000
    if threshold <= 0:
        return []
    selected: list[Any] = []
    for item in items:
        digest = hashlib.sha256(_item_key(item).encode("utf-8")).hexdigest()
        if int(digest[:8], 16) % 10_000 < threshold:
            selected.append(item)
    return selected


class VLMAdjudicator:
    """T3 field adjudication through the router's ``ocr_vlm`` profile."""

    def __init__(
        self,
        router: Any,
        *,
        purpose: str = "ocr_vlm",
        prompt_version: str | None = None,
    ) -> None:
        from fmd.extract.prompts import load_prompt

        self.router = router
        self.purpose = purpose
        self.prompt = load_prompt("adjudicate", prompt_version)

    def select(self, items: list[Any], *, rate: float | None = None) -> list[Any]:
        return sample_for_adjudication(items, rate=rate)

    def adjudicate(
        self, items: list[Any], *, text: str, rate: float | None = None
    ) -> list[tuple[Any, Any]]:
        from fmd.extract.prompts import user_message
        from fmd.extract.schemas import ADJUDICATION

        decisions: list[tuple[Any, Any]] = []
        for item in self.select(items, rate=rate):
            user = user_message(text) + "\n\nFIELD UNDER REVIEW:\n" + f"{_item_key(item)}"
            verdict = self.router.call(
                self.purpose,
                ADJUDICATION,
                self.prompt.text,
                user,
                f"adjudicate:{self.prompt.ref}:{_item_key(item)}",
            )
            decisions.append((item, verdict))
        return decisions


# --------------------------------------------------------------------------- helpers


def _import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise TierUnavailable(
            f"T0 ({module}) is not installed. Install the extraction extra with "
            "`uv sync --extra extract`."
        ) from exc


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
