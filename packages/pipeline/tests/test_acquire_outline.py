"""The document section outline (W8): detection guards, levels, breadcrumbs."""

from __future__ import annotations

from fmd.acquire.canonical import TextBlock, build_canonical, outline_to_sections
from fmd.acquire.outline import (
    DIAGNOSTIC_NO_HEADINGS,
    EXTRACTOR_DOCLING,
    EXTRACTOR_HEURISTIC,
    OutlinePolicy,
    SectionIndex,
    heading_signal,
)


def _block(text: str, page_no: int = 1, label: str | None = None) -> TextBlock:
    return TextBlock(
        page_no=page_no,
        text=text,
        bbox=(10.0, 10.0, 200.0, 20.0),
        outline_label=label,
    )


def _page(page_no: int = 1):
    from fmd.acquire.canonical import PageGeometry

    return PageGeometry(
        page_no=page_no,
        width=600.0,
        height=800.0,
        bbox=(0.0, 0.0, 600.0, 800.0),
        char_start=0,
        char_end=0,
    )


# ------------------------------------------------------------------ detection


def test_nested_numbered_headings_get_levels_ranges_and_breadcrumbs() -> None:
    blocks = [
        _block("1. Governance"),
        _block("The governing body met four times during the year."),
        _block("1.1 Board of Studies"),
        _block("The board approved the revised syllabus."),
        _block("1.1.1 Lab Curriculum"),
        _block("B.Sc. MLT laboratory work was revised."),
        _block("2. Fees"),
        _block("Tuition fee Rs. 85,000 per year."),
    ]
    canonical = build_canonical(blocks, [_page()])

    assert canonical.outline_extractor == EXTRACTOR_HEURISTIC
    titles = [node.title for node in canonical.outline]
    assert titles == ["1. Governance", "1.1 Board of Studies", "1.1.1 Lab Curriculum", "2. Fees"]
    levels = [node.level for node in canonical.outline]
    assert levels == [1, 2, 3, 1]
    parents = [node.parent_key for node in canonical.outline]
    assert parents == [None, 1, 2, None]
    # Ranges: each section ends where the next heading at <= its level starts.
    fees = canonical.outline[3]
    body = canonical.text[fees.char_start : fees.char_end]
    assert body.startswith("2. Fees")
    assert "Tuition fee Rs. 85,000 per year." in body
    assert "1.1.1 Lab Curriculum" not in body
    lab = canonical.outline[2]
    assert "B.Sc. MLT laboratory work was revised." in canonical.text[
        lab.char_start : lab.char_end
    ]


def test_numbered_sentence_is_not_a_heading() -> None:
    # The acceptance guard: "1. The institute ..." is body prose, not a title.
    sentence = "1. The institute was established in 1998 to serve rural students."
    assert heading_signal(sentence, OutlinePolicy()) is None
    canonical = build_canonical(
        [_block("1. Governance"), _block(sentence), _block("2. Fees")], [_page()]
    )
    assert [node.title for node in canonical.outline] == ["1. Governance", "2. Fees"]
    # A lone numbered line is a list item or a row label, not a chapter.
    lone = build_canonical([_block("1. Medical Laboratory Technology")], [_page()])
    assert lone.outline == ()
    assert lone.outline_diagnostic == DIAGNOSTIC_NO_HEADINGS


def test_numbered_table_row_is_not_a_heading() -> None:
    assert heading_signal("1. Tuition fee 85000", OutlinePolicy()) is None
    assert heading_signal("2. Exam fee Rs. 5,000", OutlinePolicy()) is None
    assert heading_signal("1. B.Sc. MLT", OutlinePolicy()) is None


def test_devanagari_heading_is_detected() -> None:
    canonical = build_canonical(
        [_block("4.2 शुल्क संरचना"), _block("प्रवेश शुल्क प्रति वर्ष"), _block("१. प्रवेश शुल्क")],
        [_page()],
    )
    sources = [node.source for node in canonical.outline]
    assert "heuristic:devanagari" in sources
    numbered = [node for node in canonical.outline if node.source == "heuristic:numbered"]
    assert any(node.title == "१. प्रवेश शुल्क" for node in numbered)
    # A Devanagari sentence (danda terminator) is refused.
    assert heading_signal("प्रवेश शुल्क प्रति वर्ष।", OutlinePolicy()) is None


def test_all_caps_label_and_annexure_are_headings() -> None:
    canonical = build_canonical([_block("FEE STRUCTURE"), _block("Annexure-II")], [_page()])
    sources = [node.source for node in canonical.outline]
    assert "heuristic:caps" in sources
    assert "heuristic:annexure" in sources
    # Single all-caps words are table cells.
    assert heading_signal("TOTAL", OutlinePolicy()) is None
    assert heading_signal("INR", OutlinePolicy()) is None


# --------------------------------------------------------------------- docling


def test_docling_labels_preferred_and_page_furniture_excluded() -> None:
    blocks = [
        _block("Annual Report 2024-25", label="title"),
        _block("Annual Report 2024-25", label="page_header"),
        _block("4.2 Fees", label="section_header"),
        _block("Tuition fee Rs. 85,000 per year."),
    ]
    canonical = build_canonical(blocks, [_page()])

    assert canonical.outline_extractor == EXTRACTOR_DOCLING
    assert [node.title for node in canonical.outline] == [
        "Annual Report 2024-25",
        "4.2 Fees",
    ]
    assert canonical.outline[0].source == "docling:title"
    assert canonical.outline[1].source == "docling:section_header"
    # Labels carry page + bbox anchors.
    assert canonical.outline[0].page_no == 1
    assert canonical.outline[0].bbox == (10.0, 10.0, 200.0, 20.0)


def test_heuristics_take_over_when_docling_labels_are_absent() -> None:
    blocks = [_block("1. Governance"), _block("2. Fees")]
    canonical = build_canonical(blocks, [_page()])
    assert canonical.outline_extractor == EXTRACTOR_HEURISTIC
    assert len(canonical.outline) == 2


# ---------------------------------------------------------------------- levels


def test_levels_are_normalised_without_gaps() -> None:
    # A skip-level jump (1 -> 1.1.1) normalises to 1 -> 2: levels never gap.
    canonical = build_canonical(
        [_block("1. Governance"), _block("text"), _block("1.1.1 Labs"), _block("text")],
        [_page()],
    )
    assert [node.level for node in canonical.outline] == [1, 2]


# --------------------------------------------------------------------- lookup


def test_breadcrumb_lookup_mid_subsection() -> None:
    blocks = [
        _block("ADMISSIONS PROCESS"),
        _block("Intro text for admissions."),
        _block("4. Fees"),
        _block("Fee schedule intro."),
        _block("4.1 Hostel"),
        _block("Hostel rent is charged per semester."),
        _block("4.1.1 Mess Bill"),
        _block("The mess bill is billed monthly."),
        _block("5. Placements"),
    ]
    canonical = build_canonical(blocks, [_page()])
    index = SectionIndex(canonical.outline)
    mid_subsection = canonical.text.index("The mess bill is billed")
    node, chain = index.locate(mid_subsection)

    assert node is not None and node.title == "4.1.1 Mess Bill"
    # ADMISSIONS PROCESS is a level-1 *sibling* of "4. Fees", not an ancestor:
    # the breadcrumb is the ancestor chain, which starts at the chapter.
    assert [item.title for item in chain] == ["4. Fees", "4.1 Hostel", "4.1.1 Mess Bill"]
    assert index.breadcrumb(mid_subsection) == ("4. Fees", "4.1 Hostel", "4.1.1 Mess Bill")
    assert index.node_id(mid_subsection) == node.node_key
    # A span starting inside the hostel section resolves without a linear scan
    # and a span in no section (before the first heading) resolves to nothing.
    assert index.locate(-1)[0] is None


def test_breadcrumb_uses_narrowest_section_for_overlapping_ranges() -> None:
    blocks = [
        _block("4.2 Fees"),
        _block("4.2.1 Hostel"),
        _block("hostel body"),
    ]
    canonical = build_canonical(blocks, [_page()])
    index = SectionIndex(canonical.outline)
    hostel_start = canonical.text.index("4.2.1 Hostel")
    node, chain = index.locate(hostel_start)
    assert node.title == "4.2.1 Hostel"
    assert [item.title for item in chain] == ["4.2 Fees", "4.2.1 Hostel"]


def test_no_headings_document_degrades_with_a_diagnostic() -> None:
    canonical = build_canonical(
        [_block("Tuition fee Rs. 85,000 per year."), _block("hostel rent 12000")], [_page()]
    )
    assert canonical.outline == ()
    assert canonical.outline_extractor == ""
    assert canonical.outline_diagnostic == DIAGNOSTIC_NO_HEADINGS
    # W7 heuristic sections are empty too: the ladder simply has no section rung.
    assert canonical.sections == ()


def test_outline_to_sections_carries_level_and_breadcrumb() -> None:
    canonical = build_canonical(
        [
            _block("ADMISSIONS PROCESS"),
            _block("text"),
            _block("4. Fees"),
            _block("text"),
            _block("4.2 Hostel"),
        ],
        [_page()],
    )
    bounds = outline_to_sections(canonical.outline)
    fees = bounds[1]
    hostel = bounds[2]
    assert fees.level == 1
    assert fees.path == ("4. Fees",)
    assert fees.heading_text == "4. Fees"
    assert hostel.level == 2
    assert hostel.path == ("4. Fees", "4.2 Hostel")


def test_policy_thresholds_come_from_the_registry() -> None:
    from pathlib import Path

    from fmd.config import load_registry

    registry = load_registry(Path(__file__).resolve().parents[3] / "config", env={})
    policy = OutlinePolicy.from_acquire(registry.acquire)
    assert policy.heading_max_chars == 120
    assert policy.numbered_min_sequence == 2
    assert policy.devanagari_min_ratio == 0.5
    assert "section_header" in policy.docling_labels
    assert "page_header" in policy.furniture_labels
