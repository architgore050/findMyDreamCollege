"""Repetition-loop and entropy-collapse guard for **OCR/VLM output only**.

OCR and layout models fail in a recognisable way: a decoder emits the same
short token sequence until the character distribution collapses, or it runs on
far past the amount of text a scanned page can physically hold. That is a
decoder loop, not evidence.

The guard is deliberately scoped:

* **T2 / T2b (OCR/VLM) only.** PaddleOCR-VL and Qwen3-VL return block-level
  text with no independent baseline, so their output is checked here.
* **T0 / T1 are never refused on length.** A born-digital text layer and a
  Docling layout are extracts of a real page; a long fee table is *evidence*,
  not an anomaly. This guard previously compared a block with the page median
  block length and flagged legitimate tables (3× the median prose block) as
  ``block_length_spike``. Long T0/T1 blocks now log a diagnostic warning and
  continue to extraction; they are never routed to T3.

For an OCR block the baselines are:

* the **page total** text length against ``ocr_page_chars_cap`` (a decoder that
  runs on makes the page itself implausibly long);
* the **block** length against ``ocr_block_chars_cap`` (no single OCR block on
  a scanned page can exceed this);
* the **repeated-substring ratio** (``ocr_repeated_substring_ratio``) plus the
  repeat count (``repeated_block_min_occurrences``): a synthetic 71× repeat
  trips this, a dense fee table with varied numbers does not;
* character-entropy collapse against the other blocks on the page.

Every refusal is a :class:`GuardTrip` carrying ``guard``, ``measured``,
``threshold``, ``granularity`` and a human-readable ``reason``. Nothing is
dropped silently: the trips are attached to the document (``guard_trips``) and
counted by name for the run summary.

Thresholds are read from ``config/acquire.yaml`` (``anomaly:`` block). Nothing
here depends on a page's institution or language.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from fmd.acquire.canonical import TextBlock

logger = logging.getLogger(__name__)

#: Tiers whose output is model-generated and therefore guarded.
OCR_TIERS = frozenset({"T2", "T2b"})


@dataclass(frozen=True)
class GuardTrip:
    """One refusal decision with everything needed to audit it."""

    guard: str
    measured: float
    threshold: float
    granularity: str
    reason: str
    page_no: int | None = None
    block_index: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "guard": self.guard,
            "measured": self.measured,
            "threshold": self.threshold,
            "granularity": self.granularity,
            "reason": self.reason,
            "page_no": self.page_no,
            "block_index": self.block_index,
        }


@dataclass(frozen=True)
class BlockAnomaly:
    page_no: int
    block_index: int
    reasons: tuple[str, ...]
    length: int
    page_total_length: int
    entropy: float
    repetition_factor: int
    trips: tuple[GuardTrip, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "page_no": self.page_no,
            "block_index": self.block_index,
            "reasons": list(self.reasons),
            "length": self.length,
            "page_total_length": self.page_total_length,
            "entropy": round(self.entropy, 6),
            "repetition_factor": self.repetition_factor,
            "trips": [trip.as_dict() for trip in self.trips],
        }


@dataclass(frozen=True)
class AnomalyReport:
    blocks: tuple[BlockAnomaly, ...] = ()
    page_trips: tuple[GuardTrip, ...] = ()
    pages: tuple[int, ...] = ()

    @property
    def trips(self) -> tuple[GuardTrip, ...]:
        """Every page- and block-level trip, in deterministic order."""
        collected = list(self.page_trips)
        for block in self.blocks:
            collected.extend(block.trips)
        return tuple(collected)

    @property
    def ocr_anomaly(self) -> bool:
        return bool(self.trips)

    @property
    def reasons(self) -> tuple[str, ...]:
        seen: list[str] = []
        for trip in self.trips:
            if trip.reason not in seen:
                seen.append(trip.reason)
        return tuple(seen)

    def guard_trips_by_name(self) -> dict[str, int]:
        """Counts for the run summary (``guard_trips`` keyed by guard name)."""
        counts: dict[str, int] = {}
        for trip in self.trips:
            counts[trip.guard] = counts.get(trip.guard, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, object]:
        return {
            "ocr_anomaly": self.ocr_anomaly,
            "pages": list(self.pages),
            "blocks": [block.as_dict() for block in self.blocks],
            "guard_trips": [trip.as_dict() for trip in self.trips],
            "guard_trips_by_name": self.guard_trips_by_name(),
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


def _smallest_period(text: str, *, max_unit: int) -> tuple[int, int]:
    """``(period, occurrences)`` of the smallest exact repeated unit, else (0, 0)."""
    normalised = " ".join(text.split())
    length = len(normalised)
    if length < 8:
        return 0, 0
    seed_length = min(16, length // 2)
    period = normalised.find(normalised[:seed_length], 1)
    if period <= 0 or period > max_unit or period >= length:
        return 0, 0
    if any(normalised[index] != normalised[index % period] for index in range(length)):
        return 0, 0
    return period, max(1, round(length / period))


def repetition_factor(text: str, *, max_unit: int = 4096) -> int:
    """How many times the smallest repeating unit repeats (1 if none).

    Uses the doubled-string period search (linear in practice), which is robust
    to the separator whitespace that survives ``split()`` normalisation. A block
    that is one unit repeated 71 times returns 71 even when the final separator
    was stripped.
    """
    period, occurrences = _smallest_period(text, max_unit=max_unit)
    if period == 0:
        return 1
    return occurrences if occurrences >= 2 else 1


def repeated_substring_ratio(text: str, *, max_unit: int = 4096) -> tuple[float, int]:
    """Share of the block covered by its most repeated unit, and its count.

    Two signals count as repetition, and the larger wins:

    * an exact periodic block (``"Fee 50000 " * 71``) covers 100% at 71 repeats;
    * a block whose lines are identical repeats the same line (a decoder loop
      that emits one line per iteration).

    A dense fee table has varied numbers and prose rows, so both ratios stay
    low; that is the difference between a table and a loop.
    """
    period, occurrences = _smallest_period(text, max_unit=max_unit)
    best_ratio = 0.0
    best_occurrences = 0
    if period and occurrences >= 2:
        best_ratio = min(1.0, (occurrences * period) / len(" ".join(text.split())))
        best_occurrences = occurrences

    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    total = sum(len(line) for line in lines)
    if lines and total:
        counts: dict[str, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        line_text, line_count = max(counts.items(), key=lambda item: (item[1], len(item[0])))
        if line_count >= 2:
            ratio = (line_count * len(line_text)) / total
            if ratio > best_ratio or (ratio == best_ratio and line_count > best_occurrences):
                best_ratio = min(1.0, ratio)
                best_occurrences = line_count

    return best_ratio, best_occurrences


def _blocks_by_page(blocks: Sequence[TextBlock]) -> dict[int, list[TextBlock]]:
    grouped: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if block.text.strip():
            grouped.setdefault(block.page_no, []).append(block)
    return grouped


def _reason(
    guard: str,
    measured: float,
    threshold: float,
    detail: str,
    *,
    page_no: int,
    block_index: int | None = None,
) -> str:
    where = f"page {page_no}" if block_index is None else f"page {page_no} block {block_index}"
    return (
        f"{where}: {guard} — measured {measured:g}, threshold {threshold:g} ({detail})"
    )


def _trip(
    guard: str,
    measured: float,
    threshold: float,
    granularity: str,
    detail: str,
    *,
    page_no: int,
    block_index: int | None = None,
) -> GuardTrip:
    return GuardTrip(
        guard=guard,
        measured=float(measured),
        threshold=float(threshold),
        granularity=granularity,
        reason=_reason(
            guard, measured, threshold, detail, page_no=page_no, block_index=block_index
        ),
        page_no=page_no,
        block_index=block_index,
    )


def detect_anomalies(
    blocks: Sequence[TextBlock],
    *,
    tier: str,
    entropy_ratio_floor: float,
    min_block_chars: int,
    min_page_chars: int,
    repeated_block_min_occurrences: int,
    block_chars_cap: int,
    page_chars_cap: int,
    repeated_substring_ratio_floor: float,
    truncated_page_min_chars: int,
    truncated_page_min_pages: int,
) -> AnomalyReport:
    """Flag OCR blocks/pages that look like decoder loops or truncation.

    Returns an empty report for any tier that is not OCR: born-digital T0/T1
    output is never refused by this function.
    """
    if tier not in OCR_TIERS:
        return AnomalyReport()

    grouped = _blocks_by_page(blocks)
    page_numbers = sorted(grouped)
    totals = {
        page_no: sum(len(block.text) for block in grouped[page_no]) for page_no in page_numbers
    }

    anomalies: list[BlockAnomaly] = []
    page_trips: list[GuardTrip] = []
    pages: list[int] = []

    for position, page_no in enumerate(page_numbers):
        page_blocks = grouped[page_no]
        page_total = totals[page_no]
        page_entropy = entropy(" ".join(block.text for block in page_blocks))
        page_is_substantial = page_total >= min_page_chars
        page_flagged = False

        if page_total > page_chars_cap:
            page_trips.append(
                _trip(
                    "page_length_cap",
                    page_total,
                    page_chars_cap,
                    "page",
                    "page total text exceeds what a scanned page can hold (runaway decoder)",
                    page_no=page_no,
                )
            )
            page_flagged = True

        if _is_truncated(
            position,
            page_numbers,
            totals,
            min_chars=truncated_page_min_chars,
            min_pages=truncated_page_min_pages,
            substantial_chars=min_page_chars,
        ):
            page_trips.append(
                _trip(
                    "page_truncated",
                    page_total,
                    truncated_page_min_chars,
                    "page",
                    "interior page carries almost no text between substantial pages",
                    page_no=page_no,
                )
            )
            page_flagged = True

        for index, block in enumerate(page_blocks):
            text = block.text
            length = len(text)
            if length < min_block_chars:
                continue
            trips: list[GuardTrip] = []
            factor = repetition_factor(text)
            ratio, occurrences = repeated_substring_ratio(text)

            if length > block_chars_cap:
                trips.append(
                    _trip(
                        "block_length_cap",
                        length,
                        block_chars_cap,
                        "block",
                        "single OCR block is longer than any scanned page block can be",
                        page_no=page_no,
                        block_index=index,
                    )
                )
            if factor >= repeated_block_min_occurrences or (
                ratio >= repeated_substring_ratio_floor
                and occurrences >= repeated_block_min_occurrences
            ):
                trips.append(
                    _trip(
                        "repetition_loop",
                        factor if factor >= repeated_block_min_occurrences else ratio,
                        repeated_block_min_occurrences
                        if factor >= repeated_block_min_occurrences
                        else repeated_substring_ratio_floor,
                        "block",
                        f"repeated unit {occurrences}× covering {ratio:.2f} of the block",
                        page_no=page_no,
                        block_index=index,
                    )
                )
            block_entropy = entropy(text)
            others = [
                entropy(other.text)
                for other_position, other in enumerate(page_blocks)
                if other_position != index and len(other.text) >= min_block_chars
            ]
            reference = float(statistics.median(others)) if others else page_entropy
            if (
                page_is_substantial
                and reference > 0
                and block_entropy < entropy_ratio_floor * reference
            ):
                trips.append(
                    _trip(
                        "entropy_collapse",
                        block_entropy,
                        entropy_ratio_floor * reference,
                        "block",
                        "character entropy collapsed against the page reference",
                        page_no=page_no,
                        block_index=index,
                    )
                )
            if not trips:
                continue
            page_flagged = True
            anomalies.append(
                BlockAnomaly(
                    page_no=page_no,
                    block_index=index,
                    reasons=tuple(trip.guard for trip in trips),
                    length=length,
                    page_total_length=page_total,
                    entropy=block_entropy,
                    repetition_factor=factor,
                    trips=tuple(trips),
                )
            )

        if page_flagged:
            pages.append(page_no)

    return AnomalyReport(blocks=tuple(anomalies), page_trips=tuple(page_trips), pages=tuple(pages))


def _is_truncated(
    position: int,
    page_numbers: Sequence[int],
    totals: dict[int, int],
    *,
    min_chars: int,
    min_pages: int,
    substantial_chars: int,
) -> bool:
    """An interior page far below its neighbours is a truncated decode.

    The first and last pages are exempt: cover pages and short end notes are
    legitimately sparse, so they cannot evidence truncation.
    """
    if len(page_numbers) < min_pages or position == 0 or position == len(page_numbers) - 1:
        return False
    page_total = totals[page_numbers[position]]
    if page_total >= min_chars:
        return False
    neighbours = (totals[page_numbers[position - 1]], totals[page_numbers[position + 1]])
    return max(neighbours) >= substantial_chars


def born_digital_length_warnings(
    blocks: Sequence[TextBlock],
    *,
    block_length_multiplier: float,
    min_block_chars: int,
) -> tuple[str, ...]:
    """Diagnostic only: long T0/T1 blocks are logged, never refused.

    The page-median baseline is retained for logs because it is a useful signal
    that a layout extractor merged a table into one item, but it is **not** a
    refusal: born-digital evidence must reach extraction.
    """
    messages: list[str] = []
    for page_no, page_blocks in sorted(_blocks_by_page(blocks).items()):
        lengths = [len(block.text) for block in page_blocks]
        median = float(statistics.median(lengths)) if lengths else 0.0
        if median <= 0:
            continue
        for index, block in enumerate(page_blocks):
            length = len(block.text)
            if length >= min_block_chars and length > block_length_multiplier * median:
                messages.append(
                    f"page {page_no} block {index}: block length {length} exceeds "
                    f"{block_length_multiplier:g}× the page median {median:g}; "
                    "born-digital evidence, continuing to extraction"
                )
    return messages


def guard_trip_summary(docs: Iterable[object]) -> dict[str, int]:
    """Aggregate ``guard_trips`` by guard name across extraction outcomes."""
    counts: dict[str, int] = {}
    for doc in docs:
        trips = getattr(doc, "guard_trips", ()) or ()
        for trip in trips:
            counts[trip.guard] = counts.get(trip.guard, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "AnomalyReport",
    "BlockAnomaly",
    "GuardTrip",
    "OCR_TIERS",
    "born_digital_length_warnings",
    "detect_anomalies",
    "entropy",
    "guard_trip_summary",
    "repeated_substring_ratio",
    "repetition_factor",
]
