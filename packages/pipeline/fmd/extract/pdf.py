"""Tiered document text extraction (brief §9) — the extraction-side seam.

The acquisition layer lives in :mod:`fmd.acquire`. This module keeps the public
names the extraction pipeline and older scripts import, and decodes non-PDF
documents. PDF dispatch is a thin delegate to :func:`fmd.acquire.tiers.run_tiers`.

Tier summary (details and thresholds in ``config/acquire.yaml``):

* T0 ``pdftext`` — born-digital text, per-glyph provenance.
* T1 Docling — layout + tables, item and table-cell provenance.
* T2 PaddleOCR-VL — OCR, block-level provenance.
* T2b Qwen3-VL — Devanagari fallback when T2's Indic self-check fails.
* T3 human queue — repetition loops, entropy collapse, low confidence.

Heavy tiers are imported lazily inside :mod:`fmd.acquire.tiers`. When a tier's
dependency or endpoint is missing the tier raises :class:`TierUnavailable`;
dispatch falls back only when T0 already produced usable text.
"""

from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
from typing import Any

from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
from fmd.acquire.errors import TierUnavailable
from fmd.acquire.tiers import (
    TIER_ORDER,
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
from fmd.acquire.tiers import TierPolicy as TierPolicy
from fmd.acquire.types import ExtractedDoc, PageSpan, ScanProfile, Table

__all__ = [
    "TIER_ORDER",
    "AcquirePolicy",
    "ExtractedDoc",
    "PageSpan",
    "ScanProfile",
    "Table",
    "TierDecision",
    "TierPolicy",
    "TierUnavailable",
    "VLMAdjudicator",
    "apply_anomaly_guard",
    "extract_document",
    "extract_pdf",
    "indic_self_check",
    "plan_tiers",
    "probe",
    "sample_for_adjudication",
    "select_tier",
    "t0_extract",
    "t1_extract",
    "t2_extract",
    "t2b_extract",
]


# --------------------------------------------------------------------------- dispatch


def extract_pdf(
    data: bytes,
    *,
    policy: TierPolicy | None = None,
    t0: Any = None,
    t1: Any = None,
    t2: Any = None,
    t2b: Any = None,
) -> ExtractedDoc:
    """Probe with T0, then run the selected tier with explicit fallbacks."""
    return run_tiers(
        data,
        policy=policy,
        t0=t0,
        t1=t1,
        t2=t2,
        t2b=t2b,
    )


def extract_document(
    data: bytes,
    *,
    content_type: str | None = None,
    policy: TierPolicy | None = None,
    t0: Any = None,
    t1: Any = None,
    t2: Any = None,
    t2b: Any = None,
) -> ExtractedDoc:
    """Entry point used by the pipeline: PDFs are tiered, everything else decoded."""
    if _is_pdf(data, content_type):
        return extract_pdf(data, policy=policy, t0=t0, t1=t1, t2=t2, t2b=t2b)
    if content_type and "html" in content_type.lower():
        return _html_doc(data)
    return _text_doc(data)


def _is_pdf(data: bytes, content_type: str | None) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return data.lstrip()[:5] == b"%PDF-"


def _single_block_doc(text: str, *, tool: str) -> ExtractedDoc:
    block = TextBlock(page_no=1, text=text, bbox=None, granularity="block")
    canonical = build_canonical(
        [block],
        [
            PageGeometry(
                page_no=1,
                width=0.0,
                height=0.0,
                bbox=(0.0, 0.0, 0.0, 0.0),
                char_start=0,
                char_end=0,
            )
        ],
    )
    return ExtractedDoc(
        text=canonical.text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(canonical.text))],
        tables=[],
        tier="T0",
        tool=tool,
        tool_version=None,
        canonical=canonical,
        pages=canonical.pages,
        blocks=(block,),
        bbox_granularity="block",
    )


def _text_doc(data: bytes) -> ExtractedDoc:
    return _single_block_doc(data.decode("utf-8", errors="replace"), tool="text-decoder")


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
    return _single_block_doc(text, tool="html-text")


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
