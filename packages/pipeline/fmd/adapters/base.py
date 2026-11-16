"""Shared adapter machinery (brief §7).

Adapters are *fetch-layer* objects: they discover documents and yield
``DocumentRef`` values. They never parse a document into claims and never write
to the database. URL construction, query-parameter names and field whitelists
are read from the registry entry, never hardcoded (invariant 7).

A source that cannot be read for a policy or transport reason must abstain
loudly with :class:`SourceUnavailable`; an empty-but-valid listing returns an
empty list instead. That distinction is the difference between "no seats" and
"we could not look".
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

from fmd.discover.base import DocumentRef
from fmd.fetch.ladder import FetchLadder
from fmd.fetch.types import FetchResult


class SourceUnavailable(RuntimeError):
    """The source cannot produce evidence this run.

    ``reason`` is a stable machine-readable token recorded by the frontier:
    ``robots_disallow``, ``unreachable``, ``blocked``, ``verify_first_unverified``,
    ``abstained``, ``no_listing_url``, ``http_<status>``, etc. Never use an
    empty success in place of one of these.
    """

    def __init__(self, reason: str, *, url: str | None = None, detail: str | None = None) -> None:
        self.reason = reason
        self.url = url
        self.detail = detail
        message = reason if url is None else f"{reason} ({url})"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class _AnchorParser(HTMLParser):
    """Collect ``<a href>`` anchors and their text without a DOM dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[tuple[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.anchors.append((self._href.strip(), "".join(self._parts).strip()))
            self._href = None
            self._parts = []


def anchors(html: str) -> list[tuple[str, str]]:
    """Return ``(href, text)`` pairs from an HTML string (pure)."""
    parser = _AnchorParser()
    parser.feed(html)
    parser.close()
    return parser.anchors


def resolve(base: str, href: str) -> str:
    """Resolve a possibly-relative anchor against its listing page."""
    return urljoin(base, href.strip())


def matches(pattern: str | None, value: str) -> bool:
    """Optional regex filter; absent pattern means "do not filter"."""
    if not pattern:
        return True
    return re.search(pattern, value) is not None


def compact_record(
    row: Mapping[str, Any],
    *,
    field_whitelist: Iterable[str] = (),
    pii_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Project a raw record onto the whitelist, dropping PII unconditionally.

    A field named in ``pii_fields`` is never returned even if it also appears in
    the whitelist: the drop list wins (spec §2.6).
    """
    pii = set(pii_fields)
    allow = [name for name in field_whitelist if name not in pii]
    if allow:
        return {name: row.get(name) for name in allow if name not in pii}
    return {name: value for name, value in row.items() if name not in pii}


def _param_map(source: Any) -> dict[str, Any]:
    params = getattr(source, "params", None)
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        return params.model_dump(exclude_none=True)
    if isinstance(params, Mapping):
        return dict(params)
    return {}


class BaseAdapter:
    """Registry-backed adapter with a shared fetch/robots/politeness path."""

    name = "base"

    def __init__(self, source: Any, *, ladder: FetchLadder | None = None) -> None:
        self.source = source
        self.ladder = ladder or FetchLadder()

    # ------------------------------------------------------------------ params
    @property
    def source_id(self) -> str:
        return str(self.source.id)

    def param(self, key: str, default: Any = None) -> Any:
        return _param_map(self.source).get(key, default)

    @property
    def pii_fields(self) -> list[str]:
        return list(getattr(self.source, "pii_fields", None) or [])

    def ref(
        self,
        url: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        priority: int = 0,
    ) -> DocumentRef:
        return DocumentRef(
            url=url,
            source_id=self.source_id,
            discovery_method=self.name,
            metadata=dict(metadata or {}),
            priority=priority,
        )

    # ------------------------------------------------------------------- fetch
    def _fetch_ok(self, url: str) -> FetchResult:
        """Fetch through the ladder and return only a usable 2xx response."""
        result = self.ladder.fetch(url)
        robots = getattr(result, "robots", None)
        if result.status == 0 and result.error == "robots_disallowed":
            raise SourceUnavailable("robots_disallow", url=url, detail=result.error)
        if robots is not None and not robots.allowed:
            raise SourceUnavailable("robots_disallow", url=url, detail=robots.reason)
        if result.quarantined or result.error == "all_tiers_blocked":
            raise SourceUnavailable("blocked", url=url, detail=result.error)
        if result.status == 0:
            raise SourceUnavailable("unreachable", url=url, detail=result.error)
        if result.blocked:
            raise SourceUnavailable("blocked", url=url, detail=f"http_{result.status}")
        if not result.ok:
            raise SourceUnavailable(f"http_{result.status}", url=url)
        return result

    def fetch_text(self, url: str) -> str:
        return self._fetch_ok(url).text

    # -------------------------------------------------------------- subclass io
    def parse_listing(self, text: str) -> list[DocumentRef]:  # pragma: no cover - abstract
        raise NotImplementedError

    def fetch(self) -> Iterable[DocumentRef]:  # pragma: no cover - abstract
        raise NotImplementedError
