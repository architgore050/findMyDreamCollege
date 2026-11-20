"""Document section outline (W8 section-structure).

Why this exists
---------------
A parsed PDF has a natural outline; before this module it was thrown away.
Canonical text was flattened, and the span resolver had to be "unique" inside a
heuristic ±200-char window or a flat heading block. Keeping a heading tree
gives real section boundaries for scope, a breadcrumb ("Admissions › Fees ›
Hostel") for the model, section-aligned chunk boundaries, and stable structure
for the planner and the review UI.

Extraction order (precision first; wrong attribution is worse than UNKNOWN):

1. **Docling labels** (T1). ``TextBlock.outline_label`` carries the Docling
   item label (``section_header``, ``title``); ``page_header``/``footer`` are
   page furniture and are excluded from the tree. Each label maps to its exact
   canonical offsets because :func:`fmd.acquire.canonical.build_canonical`
   collects the candidates while it emits the blocks. ``extractor`` is
   ``docling``.
2. **T0/T2 text heuristics** (any tier). Numbered headings (``1.``, ``4.2``,
   ``(iii)``, Devanagari ``१.२``), short ALL-CAPS lines, colon-terminated
   labels and annexure-family lead words. Every heuristic carries a
   false-positive guard: a numbered *sentence* (``1. The institute was
   established …``) is refused by the sentence-connector veto, the terminal
   punctuation veto and the length caps; a numbered *table row* (``1. Tuition
   fee 85000``) is refused by the amount veto; a lone numbered line is refused
   because chapters come in sequences. ``extractor`` is ``heuristic``.

Levels are assigned with an open-ancestor stack keyed on the heading's
numbering depth and are then normalised to ``1..N`` with **no gaps** (a
heading can be at most one level deeper than the previous one; an unnumbered
heading is a sibling of the previous heading). Node ranges are classic outline
ranges: from the heading's first character to the start of the next heading at
the same or a shallower level (or the end of the document), so children nest
inside parents and siblings never overlap.

Nothing here reads a clock, a network or a model. The same blocks produce the
same tree on every run. A missed heading only costs yield (narrower scope,
shallower breadcrumbs) — it never mis-attributes a span.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: Axis-aligned page box (same shape as ``fmd.acquire.canonical.BBox``).
BBox = tuple[float, float, float, float]

#: Stable extractor tags recorded on the tree and on every node.
EXTRACTOR_DOCLING = "docling"
EXTRACTOR_HEURISTIC = "heuristic"

#: Diagnostic recorded on the canonical text when no heading was found.
DIAGNOSTIC_NO_HEADINGS = "no_headings_detected"
#: Diagnostic when Docling emitted no usable outline labels and the tree was
#: derived from the text heuristics instead.
DIAGNOSTIC_DOCLING_LABELS_ABSENT = "docling_labels_absent"

DEVANAGARI_RANGE = range(0x0900, 0x0980)

_ROMAN = (
    r"i{1,3}|iv|v|vi{1,3}|ix|x|xi{1,3}|xiv|xv|xvi{1,3}|xix|xx"
)
#: Leading enumeration markers: ``4.``, ``4.2``, ``1)``, ``(iii)``, ``c.``,
#: Devanagari ``१.``/``१.२``. Depth = dot-separated part count.
_NUMBERED_PREFIX = re.compile(
    rf"^\(?\s*("
    rf"\d{{1,2}}(?:\.\d{{1,3}}){{0,3}}"
    rf"|[०-९]{{1,2}}(?:\.[०-९]{{1,3}}){{0,3}}"
    rf"|{_ROMAN}"
    rf"|[a-z]"
    rf")\s*[.)\]:]?\s+"
)
#: Sentence connectors — a remainder that reads as a sentence is not a title
#: (same vocabulary the planner's heading channel uses).
_SENTENCEISH = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|shall|will|would|"
    r"the|this|that|these|those|it|its|he|she|they)\b",
    re.IGNORECASE,
)
#: Table-row and amount shapes inside a would-be title.
_AMOUNTISH = re.compile(r"(?:₹|rs\.?|inr\b)|\d{3,}", re.IGNORECASE)
_HEADING_LEAD = re.compile(r"^(annexure|appendix|schedule)\b", re.I)
_TERMINAL_PUNCT = (".", "!", "?", ";", ",")
_DANDA = "\u0964"


@dataclass(frozen=True)
class OutlinePolicy:
    """Heading-detection thresholds (``config/acquire.yaml:outline``).

    Defaults mirror the shipped registry; :meth:`from_acquire` overrides them
    from config so no threshold is hardcoded away from the registry.
    """

    #: A heading line is short; a longer line is a sentence or a table row.
    heading_max_chars: int = 120
    heading_max_words: int = 14
    #: One all-caps word is a cell value ("TOTAL", "INR"), not a heading.
    all_caps_min_words: int = 2
    #: Numbered titles are short labels; the remainder after the marker caps.
    numbered_max_words: int = 10
    #: Numbered-heading sequences need at least this many members before the
    #: numbered heuristic fires: one lone numbered line is a list item or a
    #: row label, not a chapter.
    numbered_min_sequence: int = 2
    #: Devanagari headings need at least this share of Devanagari letters ...
    devanagari_min_ratio: float = 0.5
    #: ... and at least this many words (a single word is a cell, not a
    #: title), unless the line is a colon-terminated label.
    devanagari_min_words: int = 2
    #: Docling labels consumed as headings (never ``page_header``/``footer``).
    docling_labels: tuple[str, ...] = ("title", "section_header")
    #: Docling labels that are page furniture and never enter the tree.
    furniture_labels: tuple[str, ...] = (
        "page_header",
        "page_footer",
        "footer",
        "header",
        "footnote",
        "picture",
        "formula",
    )
    #: Safety cap on tree size for pathological documents.
    max_nodes: int = 1000

    @classmethod
    def from_acquire(cls, acquire: Any = None) -> OutlinePolicy:
        """Build from ``config/acquire.yaml:outline`` (unknown keys ignored)."""
        if acquire is None:
            return cls()
        raw = getattr(acquire, "outline", None)
        if not isinstance(raw, dict) or not raw:
            return cls()
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values: dict[str, Any] = {key: value for key, value in raw.items() if key in known}
        for key in ("docling_labels", "furniture_labels"):
            if isinstance(values.get(key), list):
                values[key] = tuple(str(item) for item in values[key])
        try:
            return cls(**values)
        except TypeError:
            return cls()


@dataclass(frozen=True)
class OutlineNode:
    """One heading and the text it governs, in canonical offsets.

    ``node_key`` is the 1-based ordinal in document order (the stable
    persistence key); ``parent_key`` is the parent's ``node_key`` (``None``
    for top-level nodes). ``char_start``/``char_end`` are the section's
    outline range — the heading plus everything under it down to the next
    heading at the same or a shallower level.
    """

    node_key: int
    level: int
    title: str
    char_start: int
    char_end: int
    heading_char_start: int
    heading_char_end: int
    page_no: int
    bbox: BBox | None
    source: str
    parent_key: int | None = None

    @property
    def heading_text(self) -> str:
        """Alias used by the span resolver's section bounds (validate.py)."""
        return self.title

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_key": self.node_key,
            "level": self.level,
            "title": self.title,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "heading_char_start": self.heading_char_start,
            "heading_char_end": self.heading_char_end,
            "page_no": self.page_no,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "source": self.source,
            "parent_key": self.parent_key,
        }


# ------------------------------------------------------------------- detection


def numbering_depth(text: str) -> tuple[int, str]:
    """The leading enumeration marker and its depth: ``(depth, remainder)``.

    ``"4.2 Fees"`` → ``(2, "Fees")``; ``"1. Introduction"`` → ``(1, ...)``;
    ``"(iii) Hostel"`` / ``"c) Fees"`` → ``(1, ...)``; ``"१.२ शुल्क"`` →
    ``(2, ...)``. ``(0, text)`` when the line carries no enumeration marker.
    """
    match = _NUMBERED_PREFIX.match(text)
    if match is None:
        return 0, text
    marker = match.group(1)
    return marker.count(".") + 1, text[match.end() :].strip()


def _is_devanagari_title(text: str, policy: OutlinePolicy) -> bool:
    """A short Devanagari line without sentence punctuation is a heading.

    Guarded like the Latin heuristics: length caps, no terminal punctuation,
    and the Devanagari danda (।) — a sentence ender — refused anywhere in the
    line.
    """
    if not 3 <= len(text) <= policy.heading_max_chars:
        return False
    if _DANDA in text or text[-1:] in _TERMINAL_PUNCT:
        return False
    words = text.split()
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    ratio = sum(1 for char in letters if ord(char) in DEVANAGARI_RANGE) / len(letters)
    if ratio < policy.devanagari_min_ratio:
        return False
    if len(words) >= policy.devanagari_min_words:
        return True
    return text[-1:] == ":"


def _remainder_is_title(remainder: str, policy: OutlinePolicy) -> bool:
    """False-positive guards for the text after an enumeration marker.

    Refused: sentences (sentence connectors), table rows and amounts (a
    3-digit number or a currency token inside the remainder), decimals and
    abbreviations that read as body text (``.``/``,``/``;`` inside), terminal
    sentence punctuation, and over-long remainders. ``"The institute …"``
    fails the connector veto; ``"Tuition fee 85000"`` fails the amount veto;
    ``"B.Sc. MLT"`` fails the interior-dot veto.
    """
    if not remainder:
        return False
    if len(remainder) > policy.heading_max_chars:
        return False
    if _DANDA in remainder:
        return False
    words = remainder.split()
    if not words or len(words) > policy.numbered_max_words:
        return False
    if remainder[-1:] in _TERMINAL_PUNCT:
        return False
    if any(char in remainder for char in ".,;"):
        return False
    if _AMOUNTISH.search(remainder):
        return False
    latin = any(char.isalpha() and ord(char) not in DEVANAGARI_RANGE for char in remainder)
    return not (latin and _SENTENCEISH.search(remainder))


def heading_signal(text: str, policy: OutlinePolicy) -> str | None:
    """Which heuristic marks this block a heading, or ``None``.

    The returned tag is recorded as the node's ``source``. Deliberately
    refused everywhere: title-case prose, single all-caps words (``TOTAL``),
    numbered sentences and numbered table rows.
    """
    stripped = text.strip()
    if not 3 <= len(stripped) <= policy.heading_max_chars:
        return None
    depth, remainder = numbering_depth(stripped)
    if depth >= 1 and _remainder_is_title(remainder, policy):
        return "heuristic:numbered"
    words = stripped.split()
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) < 3:
        return None
    terminal = stripped[-1]
    if terminal == ":":
        body = stripped[:-1]
        body_letters = [char for char in body if char.isalpha()]
        if len(body_letters) >= 3 and not any(char in body for char in ".!?"):
            return "heuristic:label"
        return None
    if terminal in ".!?":
        return None
    if len(words) >= policy.all_caps_min_words and all(char.isupper() for char in letters):
        return "heuristic:caps"
    if _HEADING_LEAD.match(stripped):
        return "heuristic:annexure"
    if _is_devanagari_title(stripped, policy):
        return "heuristic:devanagari"
    return None


# ----------------------------------------------------------------- tree build


def _levels_for(candidates: Sequence[tuple[int, str]]) -> list[int]:
    """Normalised levels (1..N, no gaps) for candidates in document order.

    ``candidates`` are ``(numbering_depth, title)``. Numbered headings open
    ancestors on a stack keyed by depth; an unnumbered heading is a sibling of
    the previous heading. The stack guarantees ``level <= previous + 1``,
    which is exactly the no-gap rule; the first heading is level 1.
    """
    levels: list[int] = []
    stack: list[tuple[int, int]] = []  # open (depth, level) numbered ancestors
    prev_level = 0
    for depth, _title in candidates:
        if depth >= 1:
            while stack and stack[-1][0] >= depth:
                stack.pop()
            level = stack[-1][1] + 1 if stack else 1
            stack.append((depth, level))
        else:
            while stack and stack[-1][1] >= prev_level:
                stack.pop()
            level = prev_level if prev_level >= 1 else 1
        levels.append(level)
        prev_level = level
    return levels


def _assemble(
    text: str,
    blocks: Sequence[tuple[int, int, int, BBox | None, str, str]],
    levels: Sequence[int],
    *,
    max_nodes: int,
) -> tuple[OutlineNode, ...]:
    """Ranges, parents and keys from document order + final levels.

    Range end: the start of the next heading at the same or a shallower level
    (else the end of the document) — children nest inside parents, siblings
    never overlap. Parent: the nearest previous heading one level up.
    """
    nodes: list[OutlineNode] = []
    parent_stack: list[int] = []  # node indexes one level up, deepest first
    total = len(text)
    count = min(len(blocks), len(levels))
    for index in range(count):
        if len(nodes) >= max_nodes:
            break
        start, end, page_no, bbox, _tag, title = blocks[index]
        level = int(levels[index])
        while parent_stack and nodes[parent_stack[-1]].level >= level:
            parent_stack.pop()
        section_end = total
        for follow in range(index + 1, count):
            if int(levels[follow]) <= level:
                section_end = blocks[follow][0]
                break
        if section_end <= start:
            section_end = total
        node = OutlineNode(
            node_key=len(nodes) + 1,
            level=level,
            title=title,
            char_start=start,
            char_end=section_end,
            heading_char_start=start,
            heading_char_end=end,
            page_no=page_no,
            bbox=bbox,
            source=blocks[index][4],
            parent_key=nodes[parent_stack[-1]].node_key if parent_stack else None,
        )
        nodes.append(node)
        parent_stack.append(len(nodes) - 1)
    return tuple(nodes)


def build_outline(
    text: str,
    runs: Sequence[tuple[int, int, int]],
    *,
    hints: Sequence[tuple[int, int, int, BBox | None, str]] = (),
    policy: OutlinePolicy | None = None,
) -> tuple[OutlineNode, ...]:
    """The document's heading tree over canonical text.

    ``runs`` are ``(start, end, page_no)`` per acquisition block in reading
    order (``canonical.block_runs`` shape). ``hints`` are Docling label
    candidates ``(start, end, page_no, bbox, label)`` collected by
    :func:`fmd.acquire.canonical.build_canonical`; when at least one usable
    label exists, the tree is Docling-derived and the text heuristics are not
    consulted (one extractor per tree, recorded on every node).
    """
    policy = policy or OutlinePolicy()
    hinted = _outline_from_hints(text, hints, policy)
    if hinted:
        return hinted
    return _outline_from_runs(text, runs, policy)


def _outline_from_hints(
    text: str,
    hints: Sequence[tuple[int, int, int, BBox | None, str]],
    policy: OutlinePolicy,
) -> tuple[OutlineNode, ...]:
    usable: list[tuple[int, int, int, BBox | None, str, str]] = []
    for start, end, page_no, bbox, label in hints:
        if label not in policy.docling_labels or label in policy.furniture_labels:
            continue
        if not 0 <= start < end <= len(text):
            continue
        title = text[start:end].strip()
        if not title:
            continue
        usable.append((start, end, page_no, bbox, f"docling:{label}", title))
    if not usable:
        return ()
    candidates: list[tuple[int, str]] = []
    for _start, _end, _page_no, _bbox, _tag, title in usable:
        depth, _remainder = numbering_depth(title)
        candidates.append((depth, title))
    levels = _levels_for(candidates)
    return _assemble(text, usable, levels, max_nodes=policy.max_nodes)


def _outline_from_runs(
    text: str,
    runs: Sequence[tuple[int, int, int]],
    policy: OutlinePolicy,
) -> tuple[OutlineNode, ...]:
    candidates: list[tuple[int, str]] = []
    blocks: list[tuple[int, int, int, BBox | None, str, str]] = []
    for start, end, page_no in runs:
        if not 0 <= start < end <= len(text):
            continue
        title = text[start:end].strip()
        if not title:
            continue
        signal = heading_signal(title, policy)
        if signal is None:
            continue
        depth, remainder = numbering_depth(title)
        candidates.append((depth, remainder))
        blocks.append((start, end, page_no, None, signal, title))
    numbered = sum(1 for block in blocks if block[4] == "heuristic:numbered")
    if numbered and numbered < policy.numbered_min_sequence:
        # One isolated numbered line is a list item or a row label, not a
        # chapter — and a bogus chapter would split a real section and
        # mis-attribute its quotes. Drop it; the other rungs still apply.
        candidates = [
            candidate
            for candidate, block in zip(candidates, blocks, strict=True)
            if block[4] != "heuristic:numbered"
        ]
        blocks = [block for block in blocks if block[4] != "heuristic:numbered"]
    if not candidates:
        return ()
    levels = _levels_for(candidates)
    return _assemble(text, blocks, levels, max_nodes=policy.max_nodes)


# -------------------------------------------------------------------- lookup


def breadcrumb_of(nodes: Sequence[OutlineNode], node: OutlineNode) -> tuple[str, ...]:
    """Root→leaf titles for ``node`` inside ``nodes`` (by parent keys)."""
    if not nodes:
        return ()
    by_key = {item.node_key: item for item in nodes}
    chain: list[str] = []
    current: OutlineNode | None = node
    seen: set[int] = set()
    while current is not None and current.node_key not in seen:
        seen.add(current.node_key)
        chain.append(current.title)
        current = by_key.get(current.parent_key) if current.parent_key else None
    chain.reverse()
    return tuple(chain)


class SectionIndex:
    """Breadcrumb lookup over an outline: bisect + parent chain, no linear scan.

    ``locate`` returns the deepest node whose range fully contains the span
    and its root→leaf chain. Containing nodes form a single ancestor chain in
    a proper outline, so the bisect position — the deepest node starting at or
    before the span start — is the only place a search can begin, and the
    chain walk is bounded by the tree depth.
    """

    def __init__(self, nodes: Sequence[OutlineNode]) -> None:
        ordered = sorted(nodes, key=lambda node: (node.char_start, -node.level))
        self._nodes: tuple[OutlineNode, ...] = tuple(ordered)
        self._starts: list[int] = [node.char_start for node in ordered]
        self._parents: list[int | None] = []
        stack: list[int] = []
        for index, node in enumerate(ordered):
            while stack and ordered[stack[-1]].level >= node.level:
                stack.pop()
            self._parents.append(stack[-1] if stack else None)
            stack.append(index)

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> tuple[OutlineNode, ...]:
        return self._nodes

    def locate(
        self, start: int, end: int | None = None
    ) -> tuple[OutlineNode | None, tuple[OutlineNode, ...]]:
        """Deepest section containing ``[start, end)`` plus its breadcrumb."""
        if end is None:
            end = start + 1
        index = bisect_right(self._starts, start) - 1
        if index < 0:
            return None, ()
        chain: list[OutlineNode] = []
        cursor: int | None = index
        while cursor is not None:
            node = self._nodes[cursor]
            if node.char_start <= start and end <= node.char_end:
                chain.append(node)
                cursor = self._parents[cursor]
                continue
            if chain:
                # An ancestor range always contains its child's range, so once
                # one node qualified the walk is pure breadcrumb assembly.
                break
            cursor = self._parents[cursor]
        if not chain:
            return None, ()
        chain.reverse()
        return chain[-1], tuple(chain)

    def breadcrumb(self, start: int, end: int | None = None) -> tuple[str, ...]:
        """Root→leaf titles for a span (empty when no section contains it)."""
        _node, chain = self.locate(start, end)
        return tuple(item.title for item in chain)

    def node_id(self, start: int, end: int | None = None) -> int | None:
        """The section node id (``node_key``) for a span, or ``None``."""
        node, _chain = self.locate(start, end)
        return node.node_key if node is not None else None

    def inside(self, char_start: int, char_end: int) -> list[OutlineNode]:
        """Nodes whose heading lies fully inside ``[char_start, char_end)``."""
        return [
            node
            for node in self._nodes
            if char_start <= node.heading_char_start and node.heading_char_end <= char_end
        ]


# ------------------------------------------------------------------ reporting


def outline_summary(
    nodes: Sequence[OutlineNode],
    *,
    extractor: str,
    diagnostic: str = "",
    max_listed: int = 2000,
) -> dict[str, Any]:
    """JSON-serialisable summary for ``extractions.structure_json``."""
    listed = list(nodes)[:max_listed]
    payload: dict[str, Any] = {
        "extractor": extractor,
        "node_count": len(nodes),
        "levels": sorted({node.level for node in nodes}),
        "nodes": [node.as_dict() for node in listed],
        "nodes_truncated": len(nodes) > len(listed),
    }
    if diagnostic:
        payload["diagnostic"] = diagnostic
    return payload


def section_starts(nodes: Iterable[OutlineNode]) -> tuple[int, ...]:
    """Heading char offsets, for chunk cuts at section boundaries."""
    return tuple(sorted({node.char_start for node in nodes}))


__all__ = [
    "BBox",
    "DIAGNOSTIC_DOCLING_LABELS_ABSENT",
    "DIAGNOSTIC_NO_HEADINGS",
    "EXTRACTOR_DOCLING",
    "EXTRACTOR_HEURISTIC",
    "OutlineNode",
    "OutlinePolicy",
    "SectionIndex",
    "breadcrumb_of",
    "build_outline",
    "heading_signal",
    "numbering_depth",
    "outline_summary",
    "section_starts",
]
