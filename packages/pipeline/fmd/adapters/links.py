"""Link classification and scoring shared by the crawl-style adapters.

Every heuristic here is deterministic and registry-driven: the document
extensions, filename patterns, anchor-text patterns, navigation patterns and
skip patterns all come from ``config/sources.yaml``. Nothing about a host, a
college, a state or a file name is hardcoded (invariant 7).

The three outcomes are deliberately boring:

``document``  a URL whose *format* is a document we snapshot (pdf/xlsx/...);
``navigate``  a URL worth one more crawl hop, and nothing else;
``skip``      a hard no: ``javascript:``/``mailto:``/``tel:``/fragment-only,
              login/registration walls, archives, images, executables.

Scoring only orders work. It never turns a page into evidence and it never
promotes a non-document to a document.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from fmd.adapters.base import SourceUnavailable
from fmd.fetch.politeness import registrable_domain

# Schemes we will ever follow. Anything else (javascript:, mailto:, tel:, data:)
# is a hard skip before any network call.
_HTTP_SCHEMES: frozenset[str] = frozenset({"", "http", "https"})

# Multi-part suffixes are compared on the full filename, not ``Path.suffix``.
_MULTI_SUFFIXES: tuple[str, ...] = (".tar.gz", ".tar.bz2", ".pdf.zip")


@dataclass(frozen=True)
class LinkDecision:
    """One classified anchor. ``score`` only orders a crawl queue."""

    url: str
    text: str
    kind: str
    score: int
    reason: str


def _suffix(path: str) -> str:
    lowered = path.lower()
    for suffix in _MULTI_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    tail = lowered.rsplit("/", 1)[-1]
    if "." not in tail:
        return ""
    return "." + tail.rsplit(".", 1)[-1]


def _patterns(values: Iterable[str] | None) -> list[re.Pattern[str]]:
    return [re.compile(value) for value in (values or []) if value]


def _matches_any(patterns: Sequence[re.Pattern[str]], value: str) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def classify_link(
    href: str,
    text: str,
    *,
    document_extensions: Sequence[str] = (".pdf",),
    page_extensions: Sequence[str] = (".html", ".htm", ".php", ".aspx", ".jsp"),
    document_patterns: Sequence[str] = (),
    anchor_patterns: Sequence[str] = (),
    navigate_patterns: Sequence[str] = (),
    skip_patterns: Sequence[str] = (),
) -> LinkDecision | None:
    """Classify one resolved-or-relative anchor.

    Returns ``None`` for a hard skip (empty href, non-HTTP scheme, login wall,
    archive/image/executable, or a link with no document and no navigation
    signal). The adapter never fetches a skipped URL.
    """
    href = (href or "").strip()
    text = (text or "").strip()
    if not href:
        return None

    parts = urlsplit(href)
    scheme = parts.scheme.lower()
    if scheme not in _HTTP_SCHEMES:
        return None
    # A bare "#section" or a query-only link resolves to the same page.
    path = parts.path
    if not path and not parts.netloc:
        return None

    lowered_path = path.lower()
    skip = _patterns(skip_patterns)
    if skip and (_matches_any(skip, href) or _matches_any(skip, text)):
        return None

    documents = [suffix.lower() for suffix in document_extensions]
    pages = [suffix.lower() for suffix in page_extensions]
    suffix = _suffix(lowered_path)

    doc_patterns = _patterns(document_patterns)
    anchor_match = _matches_any(_patterns(anchor_patterns), text)
    filename_match = _matches_any(doc_patterns, lowered_path)

    if suffix in documents:
        score = 10
        if filename_match:
            score += 5
        if anchor_match:
            score += 5
        return LinkDecision(href, text, "document", score, "document_extension")

    # Any other explicit file suffix is a hard skip unless it is a page we can
    # parse (a .php listing is still a page; a .zip is never a document).
    if suffix and suffix not in pages:
        return None

    nav_patterns = _patterns(navigate_patterns)
    nav_match = _matches_any(nav_patterns, lowered_path) or _matches_any(nav_patterns, text)
    score = 0
    if nav_match:
        score += 5
    if anchor_match:
        score += 3
    if filename_match:
        score += 2
    if score <= 0:
        return None
    return LinkDecision(href, text, "navigate", score, "navigation_signal")


def rank_decisions(decisions: Iterable[LinkDecision]) -> list[LinkDecision]:
    """Stable, deterministic order: score desc, then URL asc."""
    return sorted(decisions, key=lambda item: (-item.score, item.url))


def same_site(first: str, second: str) -> bool:
    """True when two URLs share a registrable domain (``ac.in`` aware)."""
    try:
        return registrable_domain(first) == registrable_domain(second)
    except Exception:  # pragma: no cover - tldextract missing in a bare install
        return (urlsplit(first).hostname or "") == (urlsplit(second).hostname or "")


def js_shell_signals(
    text: str,
    *,
    markers: Sequence[str] = (),
    min_anchors: int = 1,
) -> str | None:
    """Return a reason when a page looks like a client-rendered shell.

    A JavaScript shell is not a parse failure: the payload simply is not in the
    bytes we fetched. Parsing the chrome would invent a college list, so the
    adapter abstains with this reason instead.
    """
    lowered = text.lower()
    if markers and all(marker.lower() in lowered for marker in markers):
        anchor_count = lowered.count("<a ")
        if anchor_count < min_anchors:
            return "js_shell"
    return None


def require_param(value: object, *, name: str, source_id: str) -> object:
    """Registry parameters are mandatory; guessing a URL is a bug."""
    if value is None or value == "" or value == [] or value == {}:
        raise SourceUnavailable(
            "missing_registry_param",
            detail=f"{source_id} needs params.{name}",
        )
    return value
