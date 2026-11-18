"""Row-scoped quote contract: a table row is only meaningful with its header.

Every case here is offline and deterministic. The resolver accepts a value only
when the quote plus the row context identify exactly one location; anything else
is ``ambiguous_span`` → UNKNOWN.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fmd.acquire.tables import associate_headers
from fmd.acquire.types import Table
from fmd.extract.schemas import ExtractedField, SeatMatrix
from fmd.extract.validate import (
    CONTEXT_WINDOW_CHARS,
    resolve_quote,
    validate_document,
)

OBSERVED = datetime(2026, 11, 17, 19, 3, tzinfo=UTC)
HEADER = ["S.N", "Name of the Institution", "Course", "Intake"]


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
        total_seats=ExtractedField(
            quote="B.Sc. Medical Laboratory Technology 20",
            context="Institute 2 Medical College",
            confidence=0.9,
        )
    )

    result = validate(document, text, tables=tables)

    assert result.rejections == []
    claim = next(claim for claim in result.claims if claim.predicate == "seats.total")
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
        total_seats=ExtractedField(
            quote="B.Sc. Medical Laboratory Technology 20",
            context="Institute 2 Medical College",
            confidence=0.9,
        ),
        admitted=ExtractedField(quote="18", context="Institute 2 Medical College", confidence=0.9),
    )

    result = validate(document, text, tables=tables)

    assert result.rejections == []
    by_predicate = {claim.predicate: claim for claim in result.claims}
    total = by_predicate["seats.total"]
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
            total_seats=ExtractedField(
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
            total_seats=ExtractedField(
                quote="B.Sc. Medical Laboratory Technology 20", confidence=0.9
            )
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
            total_seats=ExtractedField(
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
    outcome = next(item for item in result.outcomes if item.field == "total_seats")
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
        total_seats=ExtractedField(
            quote=quote, context="Institute 2 Medical College", confidence=0.9
        )
    )
    result = validate(document, text, tables=tables)
    claim = next(claim for claim in result.claims if claim.predicate == "seats.total")
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
