from __future__ import annotations

from fmd.acquire.tables import associate_headers, is_numeric_cell, table_confidence


def _cell(
    row: int,
    col: int,
    text: str,
    *,
    row_span: int = 1,
    col_span: int = 1,
    header: bool = False,
) -> dict[str, object]:
    return {
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + row_span,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": row_span,
        "col_span": col_span,
        "text": text,
        "column_header": header,
    }


def test_numeric_detection_accepts_rupee_and_grouping() -> None:
    assert is_numeric_cell("₹50,000")
    assert is_numeric_cell("60")
    assert not is_numeric_cell("BSc MLT")
    assert not is_numeric_cell("")


def test_unambiguous_header_is_recorded_for_each_numeric_cell() -> None:
    cells = [
        _cell(0, 0, "Programme", header=True),
        _cell(0, 1, "Tuition fee", header=True),
        _cell(1, 0, "BSc MLT"),
        _cell(1, 1, "₹50,000"),
    ]

    association = associate_headers(cells)

    assert association.column_header_uncertain is False
    assert len(association.cells) == 1
    link = association.cells[0]
    assert (link.row, link.col, link.header_text) == (1, 1, "Tuition fee")
    assert link.ambiguous is False
    assert table_confidence(cells) == 1.0


def test_merged_header_marks_the_association_uncertain() -> None:
    # One header spans "Tuition" and "Hostel": a value under either column cannot
    # be attributed to a single header.
    cells = [
        _cell(0, 1, "Fees", col_span=2, header=True),
        _cell(1, 1, "50000"),
        _cell(1, 2, "20000"),
    ]

    association = associate_headers(cells)

    assert association.column_header_uncertain is True
    assert all(cell.ambiguous for cell in association.cells)
    assert all(cell.reason == "merged_header_span" for cell in association.cells)
    assert all(cell.header_text == "Fees" for cell in association.cells)
    assert table_confidence(cells) == 0.0


def test_missing_header_is_uncertain_not_guessed() -> None:
    cells = [_cell(0, 0, "60"), _cell(0, 1, "120")]

    association = associate_headers(cells)

    assert association.column_header_uncertain is True
    assert {cell.reason for cell in association.cells} == {"no_column_header"}
    assert all(cell.header_text is None for cell in association.cells)


def test_multiple_headers_above_a_value_are_ambiguous() -> None:
    # Two header cells overlap the same column (merged layout): neither can be
    # chosen as the single owner.
    cells = [
        _cell(0, 0, "Fee", col_span=2, header=True),
        _cell(0, 1, "Annual fee", header=True),
        _cell(1, 1, "50000"),
    ]

    association = associate_headers(cells)

    link = association.cells[0]
    assert link.ambiguous is True
    assert link.reason == "multiple_column_headers"
    assert link.header_text == "Annual fee | Fee"


def test_merged_numeric_cell_is_uncertain() -> None:
    cells = [
        _cell(0, 0, "Fee", header=True),
        _cell(1, 0, "50000", col_span=2),
    ]

    association = associate_headers(cells)

    assert association.cells[0].ambiguous is True
    assert association.cells[0].reason == "merged_numeric_cell"


def test_table_without_numbers_reports_no_confidence() -> None:
    cells = [_cell(0, 0, "Programme", header=True), _cell(1, 0, "BSc MLT")]
    assert table_confidence(cells) is None
    assert associate_headers(cells).column_header_uncertain is False
