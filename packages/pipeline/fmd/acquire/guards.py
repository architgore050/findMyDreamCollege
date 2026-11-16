"""Repetition-loop and entropy-collapse guard.

OCR and layout models fail in a recognisable way: a block is emitted many more
times than the page can hold, or the same short token sequence is repeated until
the character distribution collapses. That is a decoder loop, not evidence.

A flagged block/page is:

* marked ``ocr_anomaly`` on the page,
* routed to T3 (human queue),
* and **never** fed to claim extraction (the pipeline refuses to build claims
  from an anomalous document).

Thresholds are read from ``config/acquire.yaml`` (``anomaly:`` block). Nothing
here depends on a page's institution or language.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from fmd.acquire.canonical import TextBlock


@dataclass(frozen=True)
class BlockAnomaly:
    page_no: int
    block_index: int
    reasons: tuple[str, ...]
    length: int
    page_median_length: float
    entropy: float
    repetition_factor: int

    def as_dict(self) -> dict[str, object]:
        return {
            "page_no": self.page_no,
            "block_index": self.block_index,
            "reasons": list(self.reasons),
            "length": self.length,
            "page_median_length": self.page_median_length,
            "entropy": round(self.entropy, 6),
            "repetition_factor": self.repetition_factor,
        }


@dataclass(frozen=True)
class AnomalyReport:
    blocks: tuple[BlockAnomaly, ...] = ()
    pages: tuple[int, ...] = ()

    @property
    def ocr_anomaly(self) -> bool:
        return bool(self.blocks)

    @property
    def reasons(self) -> tuple[str, ...]:
        seen: list[str] = []
        for block in self.blocks:
            for reason in block.reasons:
                key = f"page_{block.page_no}:{reason}"
                if key not in seen:
                    seen.append(key)
        return tuple(seen)

    def as_dict(self) -> dict[str, object]:
        return {
            "ocr_anomaly": self.ocr_anomaly,
            "pages": list(self.pages),
            "blocks": [block.as_dict() for block in self.blocks],
        }


def entropy(text: str) -> float:
    """Shannon entropy in bits per character (0.0 for an empty string)."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def repetition_factor(text: str, *, max_unit: int = 4096) -> int:
    """How many times the smallest repeating unit repeats (1 if none).

    Uses the doubled-string period search (linear in practice), which is robust
    to the separator whitespace that survives ``split()`` normalisation. A block
    that is one unit repeated 71 times returns 71 even when the final separator
    was stripped.
    """
    normalised = " ".join(text.split())
    length = len(normalised)
    if length < 8:
        return 1
    seed_length = min(16, length // 2)
    period = normalised.find(normalised[:seed_length], 1)
    if period <= 0 or period > max_unit or period >= length:
        return 1
    if any(normalised[index] != normalised[index % period] for index in range(length)):
        return 1
    factor = max(1, round(length / period))
    return factor if factor >= 2 else 1


def _blocks_by_page(blocks: Sequence[TextBlock]) -> dict[int, list[TextBlock]]:
    grouped: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if block.text.strip():
            grouped.setdefault(block.page_no, []).append(block)
    return grouped


def detect_anomalies(
    blocks: Sequence[TextBlock],
    *,
    block_length_multiplier: float,
    entropy_ratio_floor: float,
    min_block_chars: int,
    min_page_chars: int,
    repeated_block_min_occurrences: int,
) -> AnomalyReport:
    """Flag blocks whose length explodes or whose entropy collapses."""
    grouped = _blocks_by_page(blocks)
    anomalies: list[BlockAnomaly] = []
    pages: list[int] = []

    for page_no in sorted(grouped):
        page_blocks = grouped[page_no]
        lengths = [len(block.text) for block in page_blocks]
        median_length = float(statistics.median(lengths)) if lengths else 0.0
        page_text = " ".join(block.text for block in page_blocks)
        page_entropy = entropy(page_text)
        page_is_substantial = len(page_text) >= min_page_chars
        flagged = False
        for index, block in enumerate(page_blocks):
            text = block.text
            length = len(text)
            if length < min_block_chars:
                continue
            reasons: list[str] = []
            if median_length > 0 and length > block_length_multiplier * median_length:
                reasons.append("block_length_spike")
            factor = repetition_factor(text)
            if factor >= repeated_block_min_occurrences:
                reasons.append("repetition_loop")
            block_entropy = entropy(text)
            others = [
                entropy(other.text)
                for position, other in enumerate(page_blocks)
                if position != index and len(other.text) >= min_block_chars
            ]
            reference = float(statistics.median(others)) if others else page_entropy
            if (
                page_is_substantial
                and reference > 0
                and block_entropy < entropy_ratio_floor * reference
            ):
                reasons.append("entropy_collapse")
            if not reasons:
                continue
            flagged = True
            anomalies.append(
                BlockAnomaly(
                    page_no=page_no,
                    block_index=index,
                    reasons=tuple(reasons),
                    length=length,
                    page_median_length=median_length,
                    entropy=block_entropy,
                    repetition_factor=factor,
                )
            )
        if flagged:
            pages.append(page_no)

    return AnomalyReport(blocks=tuple(anomalies), pages=tuple(pages))


__all__ = [
    "AnomalyReport",
    "BlockAnomaly",
    "detect_anomalies",
    "entropy",
    "repetition_factor",
]
