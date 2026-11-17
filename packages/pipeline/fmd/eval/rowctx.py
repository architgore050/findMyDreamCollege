"""Row-context helpers shared by the golden loader and extraction metrics.

The extractor resolves an ambiguous quote against a table block whose anchor
cell is a supplied ``context`` (see :mod:`fmd.extract.validate`). The golden
loader cannot rebuild Docling tables from a committed text fixture, so it uses
the same deterministic text rules on the fixture itself:

* a needle resolves to *every* occurrence, exact first and then through the
  extractor's canonicalisation (NFKC, unified dashes/quotes, collapsed
  whitespace);
* a ''row block'' starts at the line that contains the context and ends at the
  next blank line, form feed, or new serial-numbered record
  (``1<TAB>`` / ``2      Institute``).

These helpers never guess: if the context is not unique, or the quote is not
unique inside the block, the caller rejects or marks attribution as wrong.
"""

from __future__ import annotations

import re

from fmd.extract.validate import canonicalise

CONTEXT_WINDOW_CHARS = 200

# A new record: a short serial number followed by a tab or at least two spaces.
# Course rows such as "3. B.Sc. ..." do not match because a period follows the
# digits.
_ANCHOR_LINE = re.compile(r"^[ \t]*\d{1,3}(?:\t|[ ]{2,})", re.MULTILINE)
_BLANK_LINE = re.compile(r"^[ \t]*$", re.MULTILINE)

Span = tuple[int, int]


def canonical(text: str) -> str:
    """Canonical form used by the extractor when exact text does not match."""

    value, _ = canonicalise(text)
    return value


def canonical_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    return canonical(left) == canonical(right)


def occurrence_spans(text: str | None, needle: str | None) -> list[Span]:
    """Every source span for ``needle``: exact hits, else canonical hits."""

    if not text or not needle:
        return []
    exact: list[Span] = []
    start = text.find(needle)
    while start != -1:
        exact.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    if exact:
        return exact

    canonical_text, mapping = canonicalise(text)
    canonical_needle = canonical(needle)
    if not canonical_text or not canonical_needle:
        return []
    hits: list[Span] = []
    start = canonical_text.find(canonical_needle)
    while start != -1:
        end = start + len(canonical_needle)
        if end <= len(mapping):
            span = (mapping[start][0], mapping[end - 1][1])
            if span not in hits:
                hits.append(span)
        start = canonical_text.find(canonical_needle, start + 1)
    return hits


def row_block(text: str, context_span: Span) -> Span:
    """The line/record block that the context anchors, as a source span."""

    line_start = text.rfind("\n", 0, context_span[0]) + 1
    line_end = text.find("\n", context_span[1])
    if line_end == -1:
        line_end = len(text)
    rest_start = min(line_end + 1, len(text))
    rest = text[rest_start:]

    boundary = len(text)
    for pattern in (_ANCHOR_LINE, _BLANK_LINE):
        match = pattern.search(rest)
        if match is not None:
            boundary = min(boundary, rest_start + match.start())
    form_feed = rest.find("\f")
    if form_feed != -1:
        boundary = min(boundary, rest_start + form_feed)
    return line_start, max(boundary, line_end)


def inside(spans: list[Span], block: Span) -> list[Span]:
    return [span for span in spans if block[0] <= span[0] and span[1] <= block[1]]


def same_line(text: str, start: int, end: int, position: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return line_start <= position < line_end


def window_occurrences(text: str, quote_spans: list[Span], context_span: Span) -> list[Span]:
    """Quote spans on the context line or within the extractor's window."""

    context_start, context_end = context_span
    hits: list[Span] = []
    for quote_span in quote_spans:
        near = (
            quote_span[0] - CONTEXT_WINDOW_CHARS <= context_start
            and context_end <= quote_span[1] + CONTEXT_WINDOW_CHARS
        )
        if same_line(text, quote_span[0], quote_span[1], context_start) or near:
            hits.append(quote_span)
    return hits


__all__ = [
    "CONTEXT_WINDOW_CHARS",
    "Span",
    "canonical",
    "canonical_equal",
    "inside",
    "occurrence_spans",
    "row_block",
    "same_line",
    "window_occurrences",
]
