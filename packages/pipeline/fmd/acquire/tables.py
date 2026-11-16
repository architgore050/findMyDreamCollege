"""Table cell to column-header association (merged-cell risk).

A quote resolving to a fee value proves the *pixels*; it does not prove the
value belongs to the "Tuition fee" column rather than "Hostel fee" next to it.
Merged or multi-row headers are exactly where a correct quote becomes a wrong
number.

For every **numeric** cell this module records ``(row, col, header_text)`` and
sets ``column_header_uncertain`` when the association is not unique:

* no header cell above the value;
* more than one distinct header text above it;
* the header cell spans columns (merged header) and therefore names several
  columns at once;
* the numeric cell itself spans rows/columns.

Nothing is repaired: an uncertain association is data the review console can
show, not a guess.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_NUMERIC = re.compile(
    r"^(?:[₹$£€]\s*)?[+-]?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|lakh|lakhs|crore|crores)?$",
    re.I,
)
_STRIP = re.compile(r"[\s\u00a0]+")


def _value(cell: Any, key: str, default: Any = None) -> Any:
    if isinstance(cell, Mapping):
        return cell.get(key, default)
    return getattr(cell, key, default)


def is_numeric_cell(text: str | None) -> bool:
    if not text:
        return False
    cleaned = _STRIP.sub(" ", str(text)).strip()
    return bool(_NUMERIC.match(cleaned))


@dataclass(frozen=True)
class CellHeaderLink:
    """One numeric cell and the header text it would be attributed to."""

    row: int
    col: int
    text: str
    header_text: str | None
    header_row: int | None
    ambiguous: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "text": self.text,
            "header_text": self.header_text,
            "header_row": self.header_row,
            "column_header_uncertain": self.ambiguous,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TableCellAssociation:
    """All numeric cells of one table plus the aggregate uncertainty flag."""

    cells: tuple[CellHeaderLink, ...]
    column_header_uncertain: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "cells": [cell.as_dict() for cell in self.cells],
            "column_header_uncertain": self.column_header_uncertain,
        }


def _grid(cells: Sequence[Any]) -> dict[tuple[int, int], Any]:
    """Map every covered grid position to a cell (spans expanded)."""
    grid: dict[tuple[int, int], Any] = {}
    for cell in cells:
        row = int(_value(cell, "start_row_offset_idx", 0) or 0)
        col = int(_value(cell, "start_col_offset_idx", 0) or 0)
        row_span = max(1, int(_value(cell, "row_span", 1) or 1))
        col_span = max(1, int(_value(cell, "col_span", 1) or 1))
        for row_offset in range(row, row + row_span):
            for col_offset in range(col, col + col_span):
                grid.setdefault((row_offset, col_offset), cell)
    return grid


def _cell_bounds(cell: Any) -> tuple[int, int, int, int]:
    row = int(_value(cell, "start_row_offset_idx", 0) or 0)
    col = int(_value(cell, "start_col_offset_idx", 0) or 0)
    end_row = int(_value(cell, "end_row_offset_idx", row + 1) or (row + 1))
    end_col = int(_value(cell, "end_col_offset_idx", col + 1) or (col + 1))
    return row, col, max(end_row, row + 1), max(end_col, col + 1)


def _is_header(cell: Any) -> bool:
    return bool(_value(cell, "column_header", False))


def _header_candidates(
    cell: Any, cells: Sequence[Any], grid: Mapping[tuple[int, int], Any]
) -> tuple[list[Any], str | None]:
    row, col, _, _ = _cell_bounds(cell)
    header_row = None
    candidates: list[Any] = []
    for candidate in cells:
        if candidate is cell or not _is_header(candidate):
            continue
        c_row, c_col, c_end_row, c_end_col = _cell_bounds(candidate)
        if c_end_row > row:
            continue  # not above the value
        if not (c_col <= col < c_end_col):
            continue
        candidates.append(candidate)
        header_row = c_row if header_row is None else max(header_row, c_row)
    if not candidates:
        # Fall back to the first row of the same column when no cell is flagged.
        for (grid_row, grid_col), candidate in grid.items():
            if grid_col != col or grid_row >= row:
                continue
            if grid_row == 0:
                candidates.append(candidate)
                header_row = 0
    # Keep only the nearest header row.
    if candidates and header_row is not None:
        nearest = [
            candidate for candidate in candidates if _cell_bounds(candidate)[0] == header_row
        ]
        if nearest:
            candidates = nearest
    return candidates, header_row


def associate_headers(cells: Sequence[Any]) -> TableCellAssociation:
    """Associate each numeric cell with its column header, flagging ambiguity."""
    if not cells:
        return TableCellAssociation(cells=(), column_header_uncertain=False)
    grid = _grid(cells)
    links: list[CellHeaderLink] = []
    uncertain = False

    for cell in cells:
        text = str(_value(cell, "text", "") or "").strip()
        if not is_numeric_cell(text):
            continue
        row, col, end_row, end_col = _cell_bounds(cell)
        candidates, header_row = _header_candidates(cell, cells, grid)
        headers = {
            _STRIP.sub(" ", str(_value(candidate, "text", "") or "")).strip()
            for candidate in candidates
        }
        headers.discard("")
        ambiguous = False
        reason: str | None = None
        header_text: str | None = None

        if (end_row - row) > 1 or (end_col - col) > 1:
            ambiguous = True
            reason = "merged_numeric_cell"
            header_text = next(iter(sorted(headers)), None)
        elif not headers:
            ambiguous = True
            reason = "no_column_header"
        elif len(headers) > 1:
            ambiguous = True
            reason = "multiple_column_headers"
            header_text = " | ".join(sorted(headers))
        else:
            header_text = next(iter(headers))
            spans_columns = any(
                _cell_bounds(candidate)[3] - _cell_bounds(candidate)[1] > 1
                for candidate in candidates
            )
            if spans_columns:
                ambiguous = True
                reason = "merged_header_span"

        uncertain = uncertain or ambiguous
        links.append(
            CellHeaderLink(
                row=row,
                col=col,
                text=text,
                header_text=header_text,
                header_row=header_row,
                ambiguous=ambiguous,
                reason=reason,
            )
        )

    return TableCellAssociation(cells=tuple(links), column_header_uncertain=uncertain)


def table_confidence(cells: Iterable[Any]) -> float | None:
    """Share of numeric cells whose header association is unambiguous.

    ``None`` when the table has no numeric cells: absence of numbers is not
    evidence of a clean table, so it must not count as confidence 1.0.
    """
    links = associate_headers(list(cells)).cells
    if not links:
        return None
    clean = sum(1 for link in links if not link.ambiguous)
    return clean / len(links)


__all__ = [
    "CellHeaderLink",
    "TableCellAssociation",
    "associate_headers",
    "is_numeric_cell",
    "table_confidence",
]
