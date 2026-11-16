from __future__ import annotations

import logging
from dataclasses import replace

from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
from fmd.acquire.guards import (
    detect_anomalies,
    entropy,
    guard_trip_summary,
    repeated_substring_ratio,
    repetition_factor,
)
from fmd.acquire.tiers import AcquirePolicy, apply_anomaly_guard
from fmd.acquire.types import ExtractedDoc, PageSpan

OCR_TIER = "T2"


def _policy() -> AcquirePolicy:
    return AcquirePolicy.load()


def _thresholds(policy: AcquirePolicy, tier: str = OCR_TIER) -> dict[str, object]:
    anomaly = policy.anomaly
    return {
        "tier": tier,
        "entropy_ratio_floor": anomaly.entropy_ratio_floor,
        "min_block_chars": anomaly.min_block_chars,
        "min_page_chars": anomaly.min_page_chars,
        "repeated_block_min_occurrences": anomaly.repeated_block_min_occurrences,
        "block_chars_cap": anomaly.ocr_block_chars_cap,
        "page_chars_cap": anomaly.ocr_page_chars_cap,
        "repeated_substring_ratio_floor": anomaly.ocr_repeated_substring_ratio,
        "truncated_page_min_chars": anomaly.ocr_truncated_page_min_chars,
        "truncated_page_min_pages": anomaly.ocr_truncated_page_min_pages,
    }


def _doc(
    blocks: list[TextBlock],
    *,
    tier: str = OCR_TIER,
    page_numbers: list[int] | None = None,
) -> ExtractedDoc:
    numbers = page_numbers or sorted({block.page_no for block in blocks})
    geoms = [
        PageGeometry(
            page_no=page_no,
            width=600.0,
            height=800.0,
            bbox=(0.0, 0.0, 600.0, 800.0),
            char_start=0,
            char_end=0,
        )
        for page_no in numbers
    ]
    canonical = build_canonical(blocks, geoms)
    return ExtractedDoc(
        text=canonical.text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=[],
        tier=tier,
        tool="stub",
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity=canonical.granularity,
    )


def _varied_text(min_chars: int) -> str:
    """Deterministic, non-periodic text of at least ``min_chars`` characters."""
    parts: list[str] = []
    index = 1
    while sum(len(part) + 1 for part in parts) < min_chars:
        parts.append(
            f"row {index} fee {10000 + index * 37} seats {index % 7} "
            f"note {index * index % 13} year {2000 + index % 25}"
        )
        index += 1
    return " ".join(parts)


def _fee_table_text(rows: int = 20) -> str:
    lines = ["Fee structure 2026 (INR per annum)"]
    for row in range(rows):
        lines.append(
            f"BSc MLT year {row % 4 + 1} semester {row + 1} tuition {45000 + row * 315} "
            f"lab {2500 + row * 17} exam {900 + row % 5 * 110} total {48400 + row * 349}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- repetition


def test_repetition_factor_counts_a_71x_block() -> None:
    unit = "Fee structure repeated by a stuck decoder. "
    assert repetition_factor(unit * 71) == 71


def test_repeated_substring_ratio_is_high_for_a_loop_and_low_for_a_table() -> None:
    ratio, occurrences = repeated_substring_ratio("Fee 50000 " * 71)
    assert ratio >= 0.99
    assert occurrences == 71

    table_ratio, _ = repeated_substring_ratio(_fee_table_text())
    assert table_ratio < 0.5


def test_synthetic_71x_repetition_in_ocr_is_refused_with_a_full_trip_record() -> None:
    normal = "Programme: BSc Medical Laboratory Technology. Sanctioned intake: 60 seats."
    repeated = "Fee 50000 " * 71
    blocks = [
        TextBlock(page_no=1, text=normal, bbox=(10, 10, 500, 20)),
        TextBlock(page_no=1, text=normal, bbox=(10, 30, 500, 40)),
        TextBlock(page_no=1, text=repeated, bbox=(10, 50, 500, 60)),
    ]

    report = detect_anomalies(blocks, **_thresholds(_policy()))

    assert report.ocr_anomaly is True
    assert report.pages == (1,)
    assert "repetition_loop" in report.blocks[0].reasons
    assert "block_length_cap" not in report.blocks[0].reasons
    trip = report.trips[0]
    assert trip.guard == "repetition_loop"
    assert trip.granularity == "block"
    assert trip.threshold == _policy().anomaly.repeated_block_min_occurrences
    assert "repeated unit 71×" in trip.reason

    guarded = apply_anomaly_guard(_doc(blocks), _policy())
    assert guarded.ocr_anomaly is True
    assert guarded.needs_review is True
    assert guarded.tier == "T3"
    assert guarded.anomalous_pages == (1,)
    assert guarded.guard_trips_by_name() == {"repetition_loop": 1}


def test_repetition_below_the_occurrence_threshold_is_not_a_loop() -> None:
    threshold = _policy().anomaly.repeated_block_min_occurrences
    reference = _varied_text(300)
    blocks = [
        TextBlock(page_no=1, text=reference, bbox=(10, 10, 500, 20)),
        TextBlock(page_no=1, text="Fee 50000 " * (threshold - 1), bbox=(10, 30, 500, 40)),
    ]

    report = detect_anomalies(blocks, **_thresholds(_policy()))

    assert report.ocr_anomaly is False

    blocks[-1] = TextBlock(page_no=1, text="Fee 50000 " * threshold, bbox=(10, 30, 500, 40))
    assert detect_anomalies(blocks, **_thresholds(_policy())).ocr_anomaly is True


# --------------------------------------------------------------------------- length caps


def test_block_at_the_ocr_cap_passes_and_one_character_over_is_refused() -> None:
    cap = _policy().anomaly.ocr_block_chars_cap
    reference = TextBlock(page_no=1, text=_varied_text(300), bbox=(10, 10, 500, 20))
    at_cap = TextBlock(page_no=1, text=_varied_text(cap)[:cap], bbox=(10, 30, 500, 40))
    over_cap = TextBlock(page_no=1, text=_varied_text(cap + 1)[: cap + 1], bbox=(10, 30, 500, 40))

    clean = detect_anomalies([reference, at_cap], **_thresholds(_policy()))
    flagged = detect_anomalies([reference, over_cap], **_thresholds(_policy()))

    assert clean.ocr_anomaly is False
    assert flagged.ocr_anomaly is True
    assert flagged.trips[0].guard == "block_length_cap"
    assert flagged.trips[0].measured == cap + 1
    assert flagged.trips[0].threshold == cap


# --------------------------------------------------------------------------- tiers


def test_legit_long_fee_table_in_t1_is_not_refused() -> None:
    # Docling merges the whole fee table into one item; it is 4× the page
    # median block length, which the old page-median rule called a spike.
    table = _fee_table_text()
    blocks = [
        TextBlock(page_no=1, text="Fees payable at admission", bbox=(10, 10, 500, 20)),
        TextBlock(page_no=1, text="See table below", bbox=(10, 30, 500, 40)),
        TextBlock(page_no=1, text=table, bbox=(10, 50, 500, 700)),
    ]
    assert len(table) > 4 * 40  # 4× the median of the two prose blocks

    guarded = apply_anomaly_guard(_doc(blocks, tier="T1"), _policy())

    assert guarded.ocr_anomaly is False
    assert guarded.needs_review is False
    assert guarded.tier == "T1"
    assert guarded.guard_trips == ()


def test_regression_ruhs_fee_pdf_page_medians_do_not_refuse_t1(
    caplog,
) -> None:
    # Real failing pattern (RUHS 11-page fee PDF, doc 7): page medians were
    # 169/32/18 chars against maxima of 558/435/718 under Docling (T1).
    blocks = [
        TextBlock(page_no=1, text=_varied_text(169), bbox=(10, 10, 500, 20)),
        TextBlock(page_no=1, text=_varied_text(32), bbox=(10, 30, 500, 40)),
        TextBlock(page_no=1, text=_varied_text(18), bbox=(10, 50, 500, 60)),
        TextBlock(page_no=1, text=_varied_text(718), bbox=(10, 70, 500, 700)),
    ]
    with caplog.at_level(logging.WARNING, logger="fmd.acquire.tiers"):
        guarded = apply_anomaly_guard(_doc(blocks, tier="T1"), _policy())

    assert guarded.tier == "T1"
    assert guarded.ocr_anomaly is False
    assert guarded.needs_review is False
    assert len(guarded.text) > 0
    assert any(
        "block length" in record.message and "continuing to extraction" in record.message
        for record in caplog.records
    )


def test_dense_fee_table_in_ocr_is_not_refused() -> None:
    blocks = [
        TextBlock(
            page_no=1,
            text="Fee structure for the BSc Medical Laboratory Technology programme.",
            bbox=(10, 10, 500, 20),
        ),
        TextBlock(
            page_no=1,
            text="All amounts are in Indian rupees and include tuition and laboratory fees.",
            bbox=(10, 30, 500, 40),
        ),
        TextBlock(page_no=1, text=_fee_table_text(), bbox=(10, 50, 500, 700)),
    ]

    report = detect_anomalies(blocks, **_thresholds(_policy()))

    assert report.ocr_anomaly is False
    assert apply_anomaly_guard(_doc(blocks), _policy()).tier == OCR_TIER


def test_born_digital_document_is_never_checked_for_repetition() -> None:
    blocks = [TextBlock(page_no=1, text="Fee 50000 " * 71, bbox=(10, 10, 500, 20))]

    report = detect_anomalies(blocks, **_thresholds(_policy(), tier="T1"))

    assert report.ocr_anomaly is False
    guarded = apply_anomaly_guard(_doc(blocks, tier="T1"), _policy())
    assert guarded.tier == "T1"
    assert guarded.guard_trips == ()


# --------------------------------------------------------------------------- entropy / truncation


def test_entropy_collapse_flags_a_degenerate_ocr_block() -> None:
    varied = (
        "Eligible students 120; students placed 96; median salary INR 420000 per annum. "
        "Internships are part of the curriculum and are not employment."
    )
    degenerate = "aaaa bbbb " * 40
    blocks = [
        TextBlock(page_no=1, text=varied, bbox=(10, 10, 500, 20)),
        TextBlock(page_no=1, text=degenerate, bbox=(10, 30, 500, 40)),
    ]

    report = detect_anomalies(blocks, **_thresholds(_policy()))

    assert report.ocr_anomaly is True
    assert "entropy_collapse" in report.blocks[0].reasons
    assert "entropy_collapse" in {trip.guard for trip in report.trips}


def test_genuinely_truncated_interior_page_is_flagged() -> None:
    blocks = [
        TextBlock(page_no=1, text=_varied_text(400), bbox=(10, 10, 500, 20)),
        TextBlock(page_no=2, text="Fee", bbox=(10, 10, 500, 20)),
        TextBlock(page_no=3, text=_varied_text(400), bbox=(10, 10, 500, 20)),
    ]

    report = detect_anomalies(blocks, **_thresholds(_policy()))
    guarded = apply_anomaly_guard(_doc(blocks), _policy())

    assert report.ocr_anomaly is True
    assert report.pages == (2,)
    trip = report.page_trips[0]
    assert trip.guard == "page_truncated"
    assert trip.granularity == "page"
    assert trip.measured == len("Fee")
    assert trip.threshold == _policy().anomaly.ocr_truncated_page_min_chars
    assert guarded.tier == "T3"
    assert guarded.guard_trips_by_name() == {"page_truncated": 1}


def test_short_cover_page_is_not_called_truncated() -> None:
    blocks = [
        TextBlock(page_no=1, text="Cover", bbox=(10, 10, 500, 20)),
        TextBlock(page_no=2, text=_varied_text(400), bbox=(10, 10, 500, 20)),
        TextBlock(page_no=3, text=_varied_text(400), bbox=(10, 10, 500, 20)),
    ]

    report = detect_anomalies(blocks, **_thresholds(_policy()))

    assert report.ocr_anomaly is False


# --------------------------------------------------------------------------- summary / clean


def test_guard_trips_appear_in_the_summary_by_name() -> None:
    loop = _doc(
        [
            TextBlock(page_no=1, text=_varied_text(300), bbox=(10, 10, 500, 20)),
            TextBlock(page_no=1, text="Fee 50000 " * 71, bbox=(10, 30, 500, 40)),
        ]
    )
    over_cap = _doc(
        [
            TextBlock(page_no=1, text=_varied_text(300), bbox=(10, 10, 500, 20)),
            TextBlock(
                page_no=1,
                text=_varied_text(_policy().anomaly.ocr_block_chars_cap + 1),
                bbox=(10, 30, 500, 40),
            ),
        ]
    )

    guarded_loop = apply_anomaly_guard(loop, _policy())
    guarded_cap = apply_anomaly_guard(over_cap, _policy())

    assert guard_trip_summary([guarded_loop, guarded_cap]) == {
        "block_length_cap": 1,
        "repetition_loop": 1,
    }
    outcome = guarded_loop.as_review_item()
    assert outcome["guard_trips_by_name"] == {"repetition_loop": 1}
    trip = outcome["guard_trips"][0]
    assert {"guard", "measured", "threshold", "granularity", "reason"} <= set(trip)
    assert guarded_loop.to_structure()["guard_trips"][0]["guard"] == "repetition_loop"


def test_clean_ocr_page_is_not_flagged() -> None:
    blocks = [
        TextBlock(
            page_no=1,
            text="Programme: BSc MLT. Sanctioned intake: 60 seats. Duration: 4 years.",
            bbox=(10, 10, 500, 20),
        ),
        TextBlock(
            page_no=1,
            text="Eligible students: 120. Students placed: 96. Median salary INR 420000.",
            bbox=(10, 30, 500, 40),
        ),
    ]

    report = detect_anomalies(blocks, **_thresholds(_policy()))

    assert report.ocr_anomaly is False
    assert entropy(blocks[0].text) > 0.0


def test_guard_does_not_mutate_a_clean_document() -> None:
    doc = _doc(
        [
            TextBlock(
                page_no=1,
                text="Short note about seats and fees that is not repetitive at all.",
                bbox=(10, 10, 500, 20),
            )
        ]
    )
    guarded = apply_anomaly_guard(doc, _policy())
    assert guarded is doc
    assert replace(guarded, tier="T0").tier == "T0"
