"""Row-scoped quote contract: a table row is only meaningful with its header.

Every case here is offline and deterministic. The resolver accepts a value only
when the quote plus the row context identify exactly one location; anything else
is ``ambiguous_span`` → UNKNOWN.

W7 span-scope extends the contract with a narrowest-first ladder for quotes the
model supplied **no** row context for: document → row → page → section →
UNKNOWN. Scope affects uniqueness only — offsets stay absolute document
offsets — and the chosen scope plus the page number are recorded on the claim
qualifier.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fmd.acquire.canonical import TextBlock, build_canonical, looks_like_heading
from fmd.acquire.tables import associate_headers
from fmd.acquire.types import ExtractedDoc, PageSpan, Table
from fmd.db.repositories import ClaimCreate
from fmd.extract.chunking import TextWindow, dedupe_claims
from fmd.extract.pipeline import _window_scope_bounds
from fmd.extract.schemas import ExtractedField, SeatMatrix
from fmd.extract.validate import (
    CONTEXT_WINDOW_CHARS,
    resolve_quote,
    validate_document,
)
from fmd.verify.types import ClaimView
from fmd.verify.verifier import field_payload, seed_span

OBSERVED = datetime(2026, 11, 17, 19, 3, tzinfo=UTC)
HEADER = ["S.N", "Name of the Institution", "Course", "Intake"]
TESTS = Path(__file__).parent
SCOPE_REPORT = TESTS / "fixtures" / "extract" / "annual_report_scope.txt"
REAL_ANNUAL_REPORT = TESTS / "fixtures" / "extract" / "chunk" / "annual_report.txt"


def linearise(tables: list[Table]) -> str:
    """Docling writes one cell per line; reproduce that shape exactly."""
    lines: list[str] = []
    for table in tables:
        for row in table.rows:
            lines.extend(cell for cell in row if cell)
    return "\n".join(lines) + "\n"


def build_table(rows: list[list[str]], *, page: int | None = None) -> Table:
    """A table the way acquisition emits it: rows plus per-cell header links."""
    cells = [
        {
            "text": text,
            "start_row_offset_idx": row_index,
            "start_col_offset_idx": col_index,
            "column_header": row_index == 0,
        }
        for row_index, row in enumerate(rows)
        for col_index, text in enumerate(row)
        if text
    ]
    association = associate_headers(cells)
    return Table(
        rows=rows,
        page=page,
        cells=association.cells,
        column_header_uncertain=association.column_header_uncertain,
    )


def college_table(
    number: int, *, course: str = "1. B.Sc. Medical Laboratory Technology", seats: str = "20"
) -> Table:
    return build_table(
        [
            HEADER,
            [str(number), f"Institute {number} Medical College", course, seats],
            ["", "", "2. B.Sc. Nutrition", "15"],
        ],
        page=number,
    )


def repeated_tables(count: int = 18, per_table: int = 6) -> list[Table]:
    """Eighteen MLT rows with the same label and the same intake, across pages."""
    tables: list[Table] = []
    number = 1
    while number <= count:
        rows = [HEADER]
        for _ in range(per_table):
            if number > count:
                break
            rows.append(
                [
                    str(number),
                    f"Institute {number} Medical College",
                    "1. B.Sc. Medical Laboratory Technology",
                    "20",
                ]
            )
            rows.append(["", "", "2. B.Sc. Nutrition", "15"])
            number += 1
        tables.append(build_table(rows, page=len(tables) + 1))
    return tables


def validate(document: SeatMatrix, text: str, tables=None):
    return validate_document(
        document,
        text,
        extraction_id=1,
        entity_id=2,
        observed_at=OBSERVED,
        tables=tables,
    )


# ------------------------------------------------------------------ a. unique quote


def test_a_unique_quote_is_unchanged_by_the_context_contract() -> None:
    tables = [college_table(1)]
    text = linearise(tables)

    resolution = resolve_quote(text, "Institute 1 Medical College", tables=tables)

    assert resolution.status == "resolved"
    assert resolution.disambiguation == "unique"
    assert resolution.row is None
    assert text[resolution.match.start : resolution.match.end] == resolution.match.quote


# ------------------------------------------------------------------ b. context row


def test_ambiguous_quote_plus_row_context_resolves_to_one_row() -> None:
    tables = [college_table(1), college_table(2)]
    text = linearise(tables)
    quote = "B.Sc. Medical Laboratory Technology 20"

    without = resolve_quote(text, quote, tables=tables)
    assert without.status == "ambiguous"

    resolved = resolve_quote(text, quote, context="Institute 2 Medical College", tables=tables)
    assert resolved.status == "resolved"
    assert resolved.disambiguation == "context_row"
    assert resolved.row is not None
    assert resolved.row.row_label == "1. B.Sc. Medical Laboratory Technology"
    assert resolved.row.column_header == "Intake"
    assert resolved.row.value_cell == "20"
    assert resolved.context_quote == "Institute 2 Medical College"
    # The span really is in Institute 2's block, not Institute 1's.
    assert resolved.match.start > text.index("Institute 2 Medical College")


def test_row_identity_is_part_of_the_claim_qualifier() -> None:
    tables = [college_table(1), college_table(2)]
    text = linearise(tables)
    document = SeatMatrix(
        intake=ExtractedField(
            quote="B.Sc. Medical Laboratory Technology 20",
            context="Institute 2 Medical College",
            confidence=0.9,
        )
    )

    result = validate(document, text, tables=tables)

    assert result.rejections == []
    claim = next(claim for claim in result.claims if claim.predicate == "seats.intake")
    assert claim.value == 20
    assert claim.qualifier["disambiguation"] == "context_row"
    assert claim.qualifier["context_quote"] == "Institute 2 Medical College"
    assert claim.qualifier["row"]["row_label"] == "1. B.Sc. Medical Laboratory Technology"
    assert claim.qualifier["row"]["column_header"] == "Intake"
    assert claim.qualifier["row"]["table_page"] == 2
    assert claim.qualifier["row_char_start"] < claim.qualifier["row_char_end"]
    assert (
        text[claim.qualifier["row_char_start"] : claim.qualifier["row_char_end"]]
        == "1. B.Sc. Medical Laboratory Technology\n20"
    )


def test_wave2_seat_columns_stay_row_scoped_and_column_safe() -> None:
    """Wave-2 columns (sanctioned vs admitted) keep the row-context discipline.

    Both colleges print 20/18, so each quote is ambiguous document-wide; the
    context must resolve each to one institute block and the value must be
    parsed from that row's own numeric cell, never the serial number.
    """
    tables = [
        build_table(
            [
                ["S.N", "Name of the Institution", "Course", "Sanctioned", "Admitted"],
                [
                    "1",
                    "Institute 1 Medical College",
                    "1. B.Sc. Medical Laboratory Technology",
                    "20",
                    "18",
                ],
                [
                    "2",
                    "Institute 2 Medical College",
                    "1. B.Sc. Medical Laboratory Technology",
                    "20",
                    "18",
                ],
            ],
            page=1,
        )
    ]
    text = linearise(tables)
    document = SeatMatrix(
        intake=ExtractedField(
            quote="B.Sc. Medical Laboratory Technology 20",
            context="Institute 2 Medical College",
            confidence=0.9,
        ),
        admitted=ExtractedField(quote="18", context="Institute 2 Medical College", confidence=0.9),
    )

    result = validate(document, text, tables=tables)

    assert result.rejections == []
    by_predicate = {claim.predicate: claim for claim in result.claims}
    total = by_predicate["seats.intake"]
    admitted = by_predicate["seats.admitted"]
    assert total.value == 20
    assert admitted.value == 18
    for claim in (total, admitted):
        assert claim.qualifier["disambiguation"] == "context_row"
        assert claim.qualifier["row"]["table_page"] == 1
        assert claim.qualifier["row"]["row_label"] == "1. B.Sc. Medical Laboratory Technology"
        assert claim.qualifier["context_quote"] == "Institute 2 Medical College"
    assert total.qualifier["row"]["column_header"] == "Sanctioned"
    assert admitted.qualifier["row"]["column_header"] == "Admitted"
    # admitted within sanctioned: the wave-2 cross-guard does not fire
    assert total.qualifier["row"]["value_cell"] == "20"
    assert admitted.qualifier["row"]["value_cell"] == "18"


def test_context_that_names_a_block_without_the_quote_is_rejected() -> None:
    tables = [
        build_table(
            [
                HEADER,
                [
                    "1",
                    "Institute 1 Medical College",
                    "1. B.Sc. Medical Laboratory Technology",
                    "20",
                ],
            ],
            page=1,
        ),
        build_table(
            [
                HEADER,
                ["2", "Institute 2 Medical College", "1. B.Sc. Nutrition", "15"],
                ["", "", "2. B.Sc. Radiology", "10"],
            ],
            page=2,
        ),
        build_table(
            [
                HEADER,
                ["3", "Institute 3 Medical College", "1. B.Sc. Nutrition", "15"],
            ],
            page=3,
        ),
    ]
    text = linearise(tables)

    result = validate(
        SeatMatrix(
            intake=ExtractedField(
                quote="B.Sc. Nutrition 15",
                context="Institute 1 Medical College",
                confidence=0.9,
            )
        ),
        text,
        tables=tables,
    )

    assert result.claims == []
    rejection = result.rejections[0]
    assert rejection.reason == "ambiguous_span"
    assert "does not contain the quote" in rejection.detail

    # Positive control: the same ambiguous quote resolves for the block that does hold it.
    resolved = resolve_quote(
        text, "B.Sc. Nutrition 15", context="Institute 3 Medical College", tables=tables
    )
    assert resolved.status == "resolved"
    assert resolved.disambiguation == "context_row"
    assert resolved.row is not None
    assert resolved.row.table_page == 3


# ------------------------------------------------------------------ c. no context


def test_ambiguous_quote_without_context_stays_unknown() -> None:
    tables = [college_table(1), college_table(2)]
    text = linearise(tables)

    result = validate(
        SeatMatrix(
            intake=ExtractedField(quote="B.Sc. Medical Laboratory Technology 20", confidence=0.9)
        ),
        text,
        tables=tables,
    )

    assert result.claims == []
    rejection = result.rejections[0]
    assert rejection.reason == "ambiguous_span"
    assert "no row context" in rejection.detail
    assert result.outcomes[0].disambiguation == "none"


# ------------------------------------------------------------------ 18 repeated rows


def test_eighteen_repeated_rows_resolve_only_with_the_matching_context() -> None:
    tables = repeated_tables()
    text = linearise(tables)
    quote = "B.Sc. Medical Laboratory Technology 20"

    starts = set()
    for number in (1, 7, 13, 18):
        resolution = resolve_quote(
            text, quote, context=f"Institute {number} Medical College", tables=tables
        )
        assert resolution.status == "resolved", number
        assert resolution.disambiguation == "context_row"
        assert resolution.row is not None
        assert resolution.match.start > text.index(f"Institute {number} Medical College")
        starts.add(resolution.match.start)
    assert len(starts) == 4

    # A context that names no block cannot be used to pick a row.
    unknown = resolve_quote(text, quote, context="Somewhere Else College", tables=tables)
    assert unknown.status == "ambiguous"


# ------------------------------------------------------------------ d. header safety


def test_uncertain_column_header_rejects_a_numeric_claim() -> None:
    rows = [["Institute 1 Medical College", "1. B.Sc. Medical Laboratory Technology", "20"]]
    cells = associate_headers(
        [{"text": "20", "start_row_offset_idx": 0, "start_col_offset_idx": 2}]
    )
    assert cells.column_header_uncertain is True
    tables = [Table(rows=rows, page=1, cells=cells.cells, column_header_uncertain=True)]
    text = linearise(tables)

    result = validate(
        SeatMatrix(
            intake=ExtractedField(
                quote="B.Sc. Medical Laboratory Technology 20",
                context="Institute 1 Medical College",
                confidence=0.9,
            )
        ),
        text,
        tables=tables,
    )

    assert result.claims == []
    rejection = result.rejections[0]
    assert rejection.reason == "uncertain_column_header"
    outcome = next(item for item in result.outcomes if item.field == "intake")
    assert outcome.outcome == "rejected"


def test_uncertain_column_header_does_not_block_a_text_label() -> None:
    rows = [["Institute 1 Medical College", "1. B.Sc. Medical Laboratory Technology", "20"]]
    cells = associate_headers(
        [{"text": "20", "start_row_offset_idx": 0, "start_col_offset_idx": 2}]
    )
    tables = [Table(rows=rows, page=1, cells=cells.cells, column_header_uncertain=True)]
    text = linearise(tables)

    result = validate(
        SeatMatrix(
            programme_name=ExtractedField(
                quote="B.Sc. Medical Laboratory Technology",
                context="Institute 1 Medical College",
                confidence=0.9,
            )
        ),
        text,
        tables=tables,
    )

    assert result.rejections == []
    claim = next(claim for claim in result.claims if claim.predicate == "seats.programme_name")
    assert claim.value == "B.Sc. Medical Laboratory Technology"
    # A label has no column attribution, so an uncertain numeric header is irrelevant.
    assert claim.qualifier["disambiguation"] == "unique"


# ------------------------------------------------------------------ e. window route


def test_without_tables_a_local_context_disambiguates_by_window() -> None:
    filler = "x" * (CONTEXT_WINDOW_CHARS * 2)
    text = (
        "Institute 1 Medical College\n"
        "B.Sc. Medical Laboratory Technology 20\n"
        f"{filler}\n"
        "Institute 2 Medical College\n"
        "B.Sc. Medical Laboratory Technology 20\n"
    )
    quote = "B.Sc. Medical Laboratory Technology 20"

    resolved = resolve_quote(text, quote, context="Institute 2 Medical College")
    assert resolved.status == "resolved"
    assert resolved.disambiguation == "context_window"
    assert resolved.row is None
    assert resolved.match.start > text.index("Institute 2 Medical College")
    assert resolved.context_span is not None

    rejected = resolve_quote(text, quote, context="Not In This Document College")
    assert rejected.status == "ambiguous"

    first = resolve_quote(text, quote, context="Institute 1 Medical College")
    assert first.status == "resolved"
    assert first.match.start < text.index(filler)
    # Same quote, different context: a different location, never a first match.
    assert first.match.start != resolved.match.start


# ------------------------------------------------------------------ f. value cell


def test_numeric_value_comes_from_the_value_cell_not_the_serial_number() -> None:
    tables = [college_table(1), college_table(2)]
    text = linearise(tables)
    quote = "2 Institute 2 Medical College 1. B.Sc. Medical Laboratory Technology 20"
    # A row-scoped quote may include the serial and the institute; the value is
    # still the row's intake cell, never the first digit in the quote.
    resolution = resolve_quote(text, quote, context="Institute 2 Medical College", tables=tables)
    assert resolution.status == "resolved"

    document = SeatMatrix(
        intake=ExtractedField(quote=quote, context="Institute 2 Medical College", confidence=0.9)
    )
    result = validate(document, text, tables=tables)
    claim = next(claim for claim in result.claims if claim.predicate == "seats.intake")
    assert claim.value == 20, "the serial number must not become the intake"
    assert claim.qualifier["row"]["value_cell"] == "20"


@pytest.mark.parametrize("number", [1, 2])
def test_the_resolved_row_belongs_to_the_named_institute(number: int) -> None:
    tables = [college_table(1), college_table(2)]
    text = linearise(tables)

    resolution = resolve_quote(
        text,
        "B.Sc. Medical Laboratory Technology 20",
        context=f"Institute {number} Medical College",
        tables=tables,
    )

    assert resolution.row is not None
    assert resolution.row.table_page == number


# ------------------------------------------------------------------ W7 scope ladder
# The document-driven ladder for quotes without row context: page (exact
# geometry) then section (printed heading block), narrowest qualifying scope
# first. A scope instance qualifies only when it holds exactly ONE occurrence;
# two qualifying instances cancel out — never a first match.


def _page_map(text: str) -> list[PageSpan]:
    """Blank-line blocks are pages, exactly as the acquisition tiers emit them."""
    spans: list[PageSpan] = []
    start = 0
    page = 1
    while start < len(text):
        end = text.find("\n\n", start)
        end = len(text) if end == -1 else end + 2
        spans.append(PageSpan(page=page, char_start=start, char_end=end))
        start = end
        page += 1
    return spans


def _canonical_of(text: str):
    return build_canonical([TextBlock(page_no=1, text=line) for line in text.splitlines()])


def _report_canonical(text: str, pages: list[PageSpan]):
    """Canonical text byte-identical to the raw report, one block per line.

    The same shape the T0 tier produces: lines carry their page number,
    ``build_canonical`` inserts ``\\n`` between lines and ``\\n\\n`` between
    pages, so section offsets index the same text the page map does.
    """
    blocks: list[TextBlock] = []
    last_page = pages[-1].page
    for span in pages:
        body = text[span.char_start : span.char_end]
        if span.page < last_page and body.endswith("\n\n"):
            body = body[:-2]
        for line in body.split("\n"):
            if line:
                blocks.append(TextBlock(page_no=span.page, text=line))
    return build_canonical(blocks)


def _scope_report() -> tuple[str, list[PageSpan]]:
    text = SCOPE_REPORT.read_text(encoding="utf-8")
    return text, _page_map(text)


def _scope_claim(
    char_start: int,
    char_end: int,
    qualifier: dict,
    *,
    predicate: str = "seats.intake",
    confidence: float = 0.9,
) -> ClaimCreate:
    return ClaimCreate(
        extraction_id=1,
        entity_id=2,
        predicate=predicate,
        value=20,
        quote="B.Sc. Medical Laboratory Technology 20",
        char_start=char_start,
        char_end=char_end,
        confidence=confidence,
        qualifier=qualifier,
    )


def test_a_phrase_repeated_across_pages_resolves_on_its_unique_page() -> None:
    text, pages = _scope_report()
    quote = "Library audit: 55 books listed."

    resolution = resolve_quote(text, quote, pages=pages)

    assert resolution.status == "resolved"
    assert resolution.scope == "page"
    assert resolution.disambiguation == "unique_page"
    assert resolution.page_no == 6
    # Provenance stays absolute: the span indexes the full canonical text and
    # lands inside the recorded page, not inside a page-relative slice.
    assert text[resolution.match.start : resolution.match.end] == quote
    page6 = pages[5]
    assert page6.char_start <= resolution.match.start < page6.char_end
    assert resolution.scope_span == (page6.char_start, page6.char_end)


def test_two_pages_with_one_occurrence_each_stay_unknown() -> None:
    """Rule 2: two qualifying pages share the narrowest scope — no tie-break."""
    text, pages = _scope_report()

    resolution = resolve_quote(text, "Desk audit: 43 samples cleared.", pages=pages)

    assert resolution.status == "ambiguous"
    assert "2 pages hold exactly one occurrence" in resolution.detail


def test_two_occurrences_on_one_page_stay_unknown() -> None:
    text, pages = _scope_report()

    resolution = resolve_quote(text, "Students placed: 96", pages=pages)

    assert resolution.status == "ambiguous"
    assert "no page holds exactly one occurrence" in resolution.detail


def test_the_section_rung_cannot_rescue_when_no_section_is_unique() -> None:
    text, pages = _scope_report()
    canonical = _canonical_of(text)

    resolution = resolve_quote(
        text, "Students placed: 96", pages=pages, sections=canonical.sections
    )

    assert resolution.status == "ambiguous"
    assert "no section holds exactly one occurrence" in resolution.detail


def test_section_scope_resolves_when_the_page_cannot() -> None:
    """The chunked-window shape: page 5 alone, two headings, quote once in one."""
    text, pages = _scope_report()
    page5 = pages[4]
    window_text = text[page5.char_start : page5.char_end]
    canonical = _canonical_of(window_text)
    quote = "Lab audit: 87 samples cleared."

    resolution = resolve_quote(
        window_text,
        quote,
        pages=[PageSpan(page=1, char_start=0, char_end=len(window_text))],
        sections=canonical.sections,
    )

    assert resolution.status == "resolved"
    assert resolution.scope == "section"
    assert resolution.disambiguation == "unique_section"
    assert resolution.scope_label == "LAB AUDIT 2025-26"
    assert resolution.page_no == 1
    assert window_text[resolution.match.start : resolution.match.end] == quote
    section_text = window_text[resolution.scope_span[0] : resolution.scope_span[1]]
    assert section_text.startswith("LAB AUDIT 2025-26")
    assert section_text.count(quote) == 1


def test_page_scope_wins_over_section_scope_because_it_is_exact_geometry() -> None:
    text, pages = _scope_report()
    canonical = _canonical_of(text)

    resolution = resolve_quote(
        text, "Library audit: 55 books listed.", pages=pages, sections=canonical.sections
    )

    assert resolution.status == "resolved"
    assert resolution.scope == "page"
    assert resolution.page_no == 6


def test_a_failed_context_is_never_widened_past() -> None:
    """A context that does not hold the quote is a contradiction, not a hint.

    "Library audit: 55 books listed." would resolve on page 6 without a
    context; with a context that names a real line nowhere near any
    occurrence, the claim stays UNKNOWN — the ladder never widens over a
    failed context.
    """
    text, pages = _scope_report()

    resolution = resolve_quote(
        text,
        "Library audit: 55 books listed.",
        context="Institute Seven Medical College",
        pages=pages,
    )

    assert resolution.status == "ambiguous"
    assert resolution.match is None
    assert "no occurrence of the quote is next to exactly one occurrence of the context" in (
        resolution.detail
    )


def test_row_context_wins_even_when_a_page_would_resolve() -> None:
    """Table rows still win: the ladder stops at the row rung (acceptance 3).

    Page 1 carries two institutes (two occurrences), page 2 carries one
    institute only, and the prose page repeats the line twice. Without a
    context the page rung resolves page 2; with the row context the ladder
    must stop at the row rung and land inside Institute 1's block.
    """
    page1_tables = [college_table(1), college_table(2)]
    page2_tables = [college_table(3)]
    first_pages = linearise(page1_tables)
    second_page = linearise(page2_tables)
    prose = (
        "\nQuota applies to 1. B.Sc. Medical Laboratory Technology\n"
        "Quota applies to 1. B.Sc. Medical Laboratory Technology\n"
    )
    text = first_pages + second_page + prose
    pages = [
        PageSpan(page=1, char_start=0, char_end=len(first_pages)),
        PageSpan(page=2, char_start=len(first_pages), char_end=len(first_pages) + len(second_page)),
        PageSpan(page=3, char_start=len(first_pages) + len(second_page), char_end=len(text)),
    ]
    quote = "1. B.Sc. Medical Laboratory Technology"

    # Counterfactual: with no context, page 2 holds the only single occurrence.
    without = resolve_quote(text, quote, pages=pages)
    assert without.status == "resolved"
    assert without.scope == "page"
    assert without.page_no == 2

    resolved = resolve_quote(
        text,
        quote,
        context="Institute 1 Medical College",
        pages=pages,
        tables=page1_tables,
    )
    assert resolved.status == "resolved"
    assert resolved.scope == "row"
    assert resolved.disambiguation == "context_row"
    assert resolved.row is not None
    assert resolved.row.table_page == 1
    assert resolved.match.start < len(first_pages)
    assert text[resolved.match.start : resolved.match.end] == quote


def test_document_scope_is_recorded_when_the_quote_is_globally_unique() -> None:
    text, pages = _scope_report()

    resolution = resolve_quote(text, "Eligible students: 120 as on 31 March 2026.", pages=pages)

    assert resolution.status == "resolved"
    assert resolution.scope == "document"
    assert resolution.disambiguation == "unique"
    assert resolution.page_no == 10


# ------------------------------------------------------------------ W7 printed-heading sections


def test_printed_headings_are_detected_conservatively() -> None:
    assert looks_like_heading("LAB AUDIT 2025-26")
    assert looks_like_heading("PLACEMENT SUMMARY")
    assert looks_like_heading("Fee Structure:")
    assert looks_like_heading("ANNEXURE - II")
    assert looks_like_heading("Annexure-I")
    assert looks_like_heading("Schedule of Fees")

    # Table cells, running headers, row labels and sentences are not headings.
    assert not looks_like_heading("TOTAL")
    assert not looks_like_heading("INR")
    assert not looks_like_heading("Institute 2 Medical College")
    assert not looks_like_heading("1. B.Sc. Medical Laboratory Technology")
    assert not looks_like_heading(
        "Annual Report 2025-26 - Sri Ramachandra Institute of Medical Sciences - page 1 of 60"
    )
    assert not looks_like_heading(
        "Department review 1.1: teaching, assessment and lab upkeep were audited this term."
    )
    assert not looks_like_heading("Eligible students: 120 as on 31 March 2026.")


def test_build_canonical_derives_sections_with_absolute_offsets() -> None:
    lines = [
        "FEE STRUCTURE",
        "Tuition fee: INR 85000 per year",
        "Exam fee: INR 2000 per year",
        "TOTAL",
        "INR 87000 per year",
    ]
    text = "\n".join(lines)

    canonical = build_canonical([TextBlock(page_no=1, text=line) for line in lines])

    # One all-caps word ("TOTAL") is a cell value, not a heading.
    assert [section.heading_text for section in canonical.sections] == ["FEE STRUCTURE"]
    section = canonical.sections[0]
    assert section.page_no == 1
    assert text[section.heading_char_start : section.heading_char_end] == "FEE STRUCTURE"
    assert text[section.char_start : section.char_end] == "\n".join(lines)


def test_the_long_annual_report_fixture_derives_no_sections() -> None:
    """No printed heading in the real fixture: the section rung stays inert."""
    text = REAL_ANNUAL_REPORT.read_text(encoding="utf-8")

    assert _canonical_of(text).sections == ()


# ------------------------------------------------------------------ W7 chunked-window bounds


def test_window_scope_bounds_are_window_relative() -> None:
    text, pages = _scope_report()
    page5 = pages[4]
    window = TextWindow(
        index=1,
        char_start=page5.char_start,
        char_end=page5.char_end,
        core_start=page5.char_start,
        core_end=page5.char_end,
        text=text[page5.char_start : page5.char_end],
    )
    doc = ExtractedDoc(
        text=text,
        page_map=pages,
        tables=[],
        tier="T0",
        tool="pdftext",
        canonical=_report_canonical(text, pages),
    )

    window_pages, window_sections = _window_scope_bounds(doc, window)

    assert window_pages == [
        PageSpan(page=5, char_start=0, char_end=page5.char_end - page5.char_start)
    ]
    assert [section.heading_text for section in window_sections] == [
        "LAB AUDIT 2025-26",
        "STUDENT AFFAIRS",
    ]
    # PLACEMENT SUMMARY (page 1) is outside the window and must be dropped.
    window_text = window.text
    assert window_text[
        window_sections[0].char_start : window_sections[0].heading_char_end
    ] == "LAB AUDIT 2025-26"
    assert window_sections[1].char_end <= len(window_text)


def test_window_sections_without_their_heading_inside_are_dropped() -> None:
    text, pages = _scope_report()
    page9 = pages[8]
    window = TextWindow(
        index=2,
        char_start=page9.char_start,
        char_end=page9.char_end,
        core_start=page9.char_start,
        core_end=page9.char_end,
        text=text[page9.char_start : page9.char_end],
    )
    doc = ExtractedDoc(
        text=text,
        page_map=pages,
        tables=[],
        tier="T0",
        tool="pdftext",
        canonical=_report_canonical(text, pages),
    )

    window_pages, window_sections = _window_scope_bounds(doc, window)

    assert window_pages == [
        PageSpan(page=9, char_start=0, char_end=page9.char_end - page9.char_start)
    ]
    # Every heading sits on page 5, before this window: nothing is scoppable.
    assert window_sections is None


# ------------------------------------------------------------------ W7 dedupe rule
# Same predicate/value in the same row or section is ONE claim (overlapping
# windows re-quote the same fact); the same predicate/value on different pages
# is NOT one claim — a per-page repetition is genuine evidence.


def test_same_section_same_page_dedupes_to_one_claim() -> None:
    section = {
        "resolution_scope": "section",
        "section_heading": "LAB AUDIT 2025-26",
        "scope_page_no": 5,
    }
    first = _scope_claim(100, 115, dict(section), confidence=0.90)
    requoted = _scope_claim(130, 145, dict(section), confidence=0.95)

    kept, dropped = dedupe_claims([first, requoted])

    assert dropped == 1
    assert len(kept) == 1
    assert kept[0].confidence == pytest.approx(0.95)


def test_same_section_on_a_different_page_is_not_one_claim() -> None:
    page5 = {
        "resolution_scope": "section",
        "section_heading": "LAB AUDIT 2025-26",
        "scope_page_no": 5,
    }
    page9 = {
        "resolution_scope": "section",
        "section_heading": "LAB AUDIT 2025-26",
        "scope_page_no": 9,
    }

    kept, dropped = dedupe_claims(
        [_scope_claim(100, 115, page5), _scope_claim(200, 215, page9)]
    )

    assert dropped == 0
    assert len(kept) == 2


def test_page_scoped_repetition_on_different_pages_is_not_one_claim() -> None:
    page2 = {"resolution_scope": "page", "scope_page_no": 2}
    page5 = {"resolution_scope": "page", "scope_page_no": 5}

    kept, dropped = dedupe_claims(
        [_scope_claim(300, 315, page2), _scope_claim(400, 415, page5)]
    )

    assert dropped == 0
    assert len(kept) == 2


def test_the_same_row_still_dedupes_across_windows() -> None:
    row = {"row": {"table_index": 0, "row_index": 2, "row_label": "1. B.Sc. MLT"}}

    kept, dropped = dedupe_claims(
        [_scope_claim(500, 515, dict(row)), _scope_claim(520, 535, dict(row))]
    )

    assert dropped == 1
    assert len(kept) == 1


# ------------------------------------------------------------------ W7 verifier sees the scope


def test_claim_view_surfaces_the_resolution_scope() -> None:
    qualifier = {
        "resolution_scope": "page",
        "scope_page_no": 5,
        "scope_char_start": 4000,
        "scope_char_end": 5200,
    }
    view = ClaimView.from_claim(_scope_claim(4050, 4065, qualifier))

    assert view.resolution_scope == "page"
    assert view.scope_page == 5
    assert view.scope_span == (4000, 5200)
    payload = field_payload(view)
    assert payload["resolution_scope"] == "page"
    assert payload["page"] == 5

    bare = field_payload(ClaimView.from_claim(_scope_claim(0, 10, {})))
    assert "resolution_scope" not in bare
    assert "page" not in bare


def test_verifier_window_covers_a_page_scoped_claim_when_it_fits() -> None:
    qualifier = {
        "resolution_scope": "page",
        "scope_page_no": 5,
        "scope_char_start": 1000,
        "scope_char_end": 1400,
    }
    view = ClaimView.from_claim(_scope_claim(1200, 1215, qualifier))

    start, end = seed_span(view, chars=500)

    assert start <= 1000
    assert end >= 1400

    wide = ClaimView.from_claim(
        _scope_claim(
            1200,
            1215,
            {**qualifier, "scope_char_start": 0, "scope_char_end": 9000},
        )
    )

    assert seed_span(wide, chars=500) == (1200, 1215)


# ---------------------------------------------- W7 regression: real long annual report


def test_real_annual_report_repeated_phrase_stays_unknown_and_uniques_resolve() -> None:
    text = REAL_ANNUAL_REPORT.read_text(encoding="utf-8")
    pages = _page_map(text)
    assert len(pages) == 60

    # The running header repeats on every page; the audit sentence repeats
    # three times per page. No page and no section holds exactly one
    # occurrence, so the repeated phrase stays UNKNOWN — that is the safe
    # outcome, not a resolution.
    repeated = "teaching, assessment and lab upkeep were audited this term."
    assert text.count(repeated) == 180
    canonical = _canonical_of(text)
    assert canonical.sections == ()
    resolution = resolve_quote(text, repeated, pages=pages, sections=canonical.sections)
    assert resolution.status == "ambiguous"

    # "96" is printed once on page 20 and once on page 34: two qualifying
    # pages cancel out; the value is never resolved by position.
    value = resolve_quote(text, "96", pages=pages)
    assert value.status == "ambiguous"
    assert "2 pages hold exactly one occurrence" in value.detail

    # A unique printed line resolves document-scoped, with its page recorded.
    unique = resolve_quote(text, "Eligible students: 120 as on 31 March 2026.", pages=pages)
    assert unique.status == "resolved"
    assert unique.scope == "document"
    assert unique.page_no == 12
    expected = "Eligible students: 120 as on 31 March 2026."
    assert text[unique.match.start : unique.match.end] == expected
