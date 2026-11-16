from __future__ import annotations

from fmd.acquire.canonical import (
    PageGeometry,
    TextBlock,
    box_inside,
    build_canonical,
)
from fmd.extract.validate import resolve_quote


def _page(page_no: int = 1, width: float = 600.0, height: float = 800.0) -> PageGeometry:
    return PageGeometry(
        page_no=page_no,
        width=width,
        height=height,
        bbox=(0.0, 0.0, width, height),
        char_start=0,
        char_end=0,
    )


def _span(
    text: str,
    *,
    page_no: int = 1,
    y0: float = 10.0,
    glyph: bool = False,
) -> TextBlock:
    x = 10.0
    char_boxes = []
    for _ in text:
        char_boxes.append((x, y0, x + 5.0, y0 + 10.0))
        x += 5.0
    return TextBlock(
        page_no=page_no,
        text=text,
        bbox=(10.0, y0, x, y0 + 10.0),
        granularity="glyph" if glyph else "span",
        char_boxes=tuple(char_boxes) if glyph else None,
    )


def test_quote_round_trip_offsets_bbox_and_back() -> None:
    blocks = [_span("Programme: BSc MLT", y0=10), _span("Sanctioned intake: 60", y0=30)]
    canonical = build_canonical(blocks, [_page()])

    quote = "Sanctioned intake: 60"
    hits = canonical.resolve(quote)

    assert len(hits) == 1
    hit = hits[0]
    assert canonical.text[hit.canonical_start : hit.canonical_end] == quote
    assert hit.bbox == (10.0, 30.0, 115.0, 40.0)
    assert hit.page_no == 1
    assert hit.granularity == "span"
    assert box_inside(hit.bbox, canonical.page_box(1))
    # The same offsets resolve back to the same quote (lossless map).
    assert canonical.bboxes(hit.canonical_start, hit.canonical_end)[0] == hit


def test_multi_span_quote_yields_multiple_bboxes() -> None:
    blocks = [_span("Total fee:", y0=10), _span("INR 50,000", y0=10)]
    canonical = build_canonical(blocks, [_page()])

    # A model quote collapses the block newline; validate maps it back over the
    # separator, so the projection must return both source boxes, not one.
    resolution = resolve_quote(canonical.text, "Total fee: INR 50,000")
    assert resolution.status == "resolved" and resolution.match is not None
    hits = canonical.bboxes(resolution.match.start, resolution.match.end)

    assert len(hits) == 2
    assert [hit.text for hit in hits] == ["Total fee:", "INR 50,000"]
    assert hits[0].bbox == (10.0, 10.0, 60.0, 20.0)
    assert hits[1].bbox == (10.0, 10.0, 60.0, 20.0)


def test_glyph_granularity_merges_only_covered_characters() -> None:
    canonical = build_canonical([_span("abcdefghij", glyph=True)], [_page()])

    hit = canonical.resolve("def")[0]

    assert canonical.granularity == "glyph"
    assert hit.granularity == "glyph"
    assert hit.bbox == (25.0, 10.0, 40.0, 20.0)


def test_ambiguous_quote_resolves_to_no_box() -> None:
    canonical = build_canonical([_span("Fee 100 Fee 100")], [_page()])
    assert canonical.resolve("Fee 100") == []


def test_devanagari_digits_and_rupee_span_resolve_to_a_bbox() -> None:
    text = "शुल्क: ₹५०,००० प्रति वर्ष"
    canonical = build_canonical([_span(text, glyph=True)], [_page()])

    resolution = resolve_quote(canonical.text, "₹50,000 प्रति वर्ष")
    assert resolution.status == "resolved" and resolution.match is not None

    hits = canonical.bboxes(resolution.match.start, resolution.match.end)
    assert hits
    assert all(box_inside(hit.bbox, canonical.page_box(1)) for hit in hits)
    for hit in hits:
        assert "प्रति वर्ष" in hit.text or "₹" in hit.text


def test_page_offsets_are_contiguous_and_cover_the_text() -> None:
    canonical = build_canonical(
        [_span("page one", page_no=1), _span("page two", page_no=2)],
        [_page(1), _page(2)],
    )
    assert canonical.text[canonical.pages[0].char_start : canonical.pages[0].char_end] == (
        "page one"
    )
    assert canonical.pages[1].char_start > canonical.pages[0].char_end
    assert canonical.pages[1].char_end == len(canonical.text)
