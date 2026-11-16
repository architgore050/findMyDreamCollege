from __future__ import annotations

from dataclasses import replace

from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
from fmd.acquire.guards import detect_anomalies, entropy, repetition_factor
from fmd.acquire.tiers import AcquirePolicy, apply_anomaly_guard
from fmd.acquire.types import ExtractedDoc, PageSpan


def _policy() -> AcquirePolicy:
    return AcquirePolicy.load()


def _thresholds(policy: AcquirePolicy) -> dict[str, float | int]:
    anomaly = policy.anomaly
    return {
        "block_length_multiplier": anomaly.block_length_multiplier,
        "entropy_ratio_floor": anomaly.entropy_ratio_floor,
        "min_block_chars": anomaly.min_block_chars,
        "min_page_chars": anomaly.min_page_chars,
        "repeated_block_min_occurrences": anomaly.repeated_block_min_occurrences,
    }


def _doc(blocks: list[TextBlock]) -> ExtractedDoc:
    page = PageGeometry(
        page_no=1,
        width=600.0,
        height=800.0,
        bbox=(0.0, 0.0, 600.0, 800.0),
        char_start=0,
        char_end=0,
    )
    canonical = build_canonical(blocks, [page])
    return ExtractedDoc(
        text=canonical.text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(canonical.text))],
        tables=[],
        tier="T0",
        tool="stub",
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity=canonical.granularity,
    )


def test_repetition_factor_counts_a_71x_block() -> None:
    unit = "Fee structure repeated by a stuck decoder. "
    assert repetition_factor(unit * 71) == 71


def test_synthetic_71x_repeated_block_is_routed_to_t3_and_never_claimed() -> None:
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
    assert "block_length_spike" in report.blocks[0].reasons

    guarded = apply_anomaly_guard(_doc(blocks), _policy())
    assert guarded.ocr_anomaly is True
    assert guarded.needs_review is True
    assert guarded.tier == "T3"
    assert guarded.anomalous_pages == (1,)


def test_entropy_collapse_flags_a_degenerate_block() -> None:
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


def test_clean_page_is_not_flagged() -> None:
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
