"""Affiliating state health-science universities (spec §4.1.4).

One generic adapter, five heterogeneous sites, zero hardcoded hosts. Each seed
domain is configured under ``params.per_domain`` with a discovery *mode*:

``html``     a college-list page whose anchors (and ``<option>`` lists) name
             affiliated colleges and link syllabus/seat-matrix PDFs;
``sitemap``  a WordPress/Joomla sitemap whose URLs are filtered by a path
             pattern (the sitemap advertises the affiliation pages);
``nav``      a bounded navigation crawl from the seed, filtered by anchor text.

A site that is not verified yet is skipped with a recorded reason rather than
guessed at. A client-rendered shell abstains with ``js_shell``: parsing its
chrome would invent colleges, which is worse than ``unknown``.

The college list itself is a *page* to snapshot; the adapter counts the parsed
entities and yields a ``DocumentRef`` for the page plus a ref per PDF it links.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, matches, resolve
from fmd.adapters.links import js_shell_signals, same_site
from fmd.discover.base import DocumentRef


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


class _OptionParser(HTMLParser):
    """Collect ``<option value="...">text</option>`` without a DOM dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.options: list[str] = []
        self._value: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "option":
            self._value = dict(attrs).get("value")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._value is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "option" and self._value is not None:
            label = " ".join("".join(self._text).split())
            self.options.append(label or (self._value or "").strip())
            self._value = None
            self._text = []


def options(html: str) -> list[str]:
    """Return the visible text of every ``<option>`` (pure)."""
    parser = _OptionParser()
    parser.feed(html)
    parser.close()
    return [value for value in parser.options if value]


class HealthUniversityAdapter(BaseAdapter):
    name = "health_university"

    # ------------------------------------------------------------------ config
    def _per_domain(self) -> dict[str, dict[str, Any]]:
        raw = self.param("per_domain") or {}
        if not isinstance(raw, dict):
            raise SourceUnavailable(
                "invalid_registry_param",
                detail=f"{self.source_id} params.per_domain must be a mapping",
            )
        return {
            str(key).lower().lstrip("."): dict(value or {})
            for key, value in raw.items()
            if isinstance(value, dict)
        }

    def site_config(self, url: str) -> dict[str, Any] | None:
        """Match a seed URL to its per-domain block by host suffix."""
        host = (urlsplit(url).hostname or "").lower()
        for key, entry in self._per_domain().items():
            if host == key or host.endswith("." + key):
                return entry
        return None

    # ------------------------------------------------------------------ parsing
    def parse_colleges(self, text: str) -> list[str]:
        """Affiliated-college names present as ``<option>`` values (pure).

        RUHS publishes its ~600-college list as a ``<select>``, not a table; a
        table-only parser would report zero colleges for a page that lists 600.
        """
        pattern = self.param("college_pattern") or r"(?i)college|institut"
        found: list[str] = []
        seen: set[str] = set()
        for label in options(text):
            if not matches(pattern, label):
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(label)
        return found

    def is_js_shell(self, text: str) -> str | None:
        return js_shell_signals(
            text,
            markers=tuple(self.param("js_shell_markers") or ()),
            min_anchors=int(self.param("js_shell_min_anchors", 1) or 1),
        )

    def _doc_refs(self, text: str, base_url: str, *, kind: str) -> list[DocumentRef]:
        href_pattern = self.param("pdf_href_pattern", r"(?i)\.pdf($|\?)")
        text_pattern = self.param("document_text_pattern")
        refs: list[DocumentRef] = []
        seen: set[str] = set()
        for href, label in anchors(text):
            if not href or not matches(href_pattern, href):
                continue
            if not matches(text_pattern, f"{href} {label}"):
                continue
            url = resolve(base_url, href)
            if url in seen or not same_site(base_url, url):
                continue
            seen.add(url)
            refs.append(
                self.ref(
                    url,
                    metadata={
                        "kind": kind,
                        "anchor_text": label,
                        # Role/domain emission only: the crawler re-classifies
                        # from the fetched bytes, but a ledger can already show
                        # that this ref is an affiliation document (evidence).
                        "document_role": "evidence",
                        "domain": _host(url),
                    },
                )
            )
        return refs

    def parse_listing(self, text: str, *, base_url: str | None = None) -> list[DocumentRef]:
        """HTML-mode listing: the college-list page itself plus its PDFs."""
        base = base_url or str(getattr(self.source, "base_url", "") or "")
        colleges = self.parse_colleges(text)
        refs: list[DocumentRef] = []
        if base:
            refs.append(
                self.ref(
                    base,
                    metadata={
                        "kind": "affiliated_college_list",
                        "college_count": len(colleges),
                        "college_names_sample": colleges[:10],
                        # A college list is inventory: it names institutions and
                        # is ingested, never extracted (migration 0004).
                        "document_role": "inventory",
                        "domain": _host(base),
                    },
                    priority=1,
                )
            )
        refs.extend(self._doc_refs(text, base, kind="affiliation_document"))
        return refs

    def parse_sitemap(self, text: str, *, sitemap_url: str) -> list[DocumentRef]:
        """Sitemap-mode listing: affiliation pages filtered by path pattern (pure)."""
        path_pattern = self.param("sitemap_path_pattern") or r"(?i)affiliat|college"
        refs: list[DocumentRef] = []
        seen: set[str] = set()
        for match in re.finditer(r"<loc>\s*(.*?)\s*</loc>", text, re.IGNORECASE | re.DOTALL):
            url = match.group(1).strip()
            if not url or url in seen:
                continue
            if not matches(path_pattern, url):
                continue
            seen.add(url)
            refs.append(
                self.ref(
                    url,
                    metadata={"kind": "affiliation_page", "found_in": sitemap_url},
                )
            )
        return refs

    # -------------------------------------------------------------------- fetch
    def _fetch_site(self, seed: str, config: dict[str, Any]) -> list[DocumentRef]:
        mode = str(config.get("mode") or self.param("default_mode") or "html").lower()
        if mode == "sitemap":
            sitemap_url = config.get("sitemap_url") or seed
            text = self.fetch_text(str(sitemap_url))
            return self.parse_sitemap(text, sitemap_url=str(sitemap_url))
        if mode == "html":
            listing_url = str(config.get("listing_url") or seed)
            text = self.fetch_text(listing_url)
            reason = self.is_js_shell(text)
            if reason is not None:
                raise SourceUnavailable(reason, url=listing_url, detail="client-rendered shell")
            return self.parse_listing(text, base_url=listing_url)
        if mode == "nav":
            return self._crawl_nav(seed)
        raise SourceUnavailable(
            "invalid_registry_param",
            url=seed,
            detail=f"unknown per_domain mode {mode!r} (expected html, sitemap or nav)",
        )

    def _crawl_nav(self, seed: str) -> list[DocumentRef]:
        max_pages = int(self.param("max_pages", 5) or 5)
        nav_pattern = self.param("navigate_pattern") or r"(?i)affiliat|college|institution"
        queue = [seed]
        visited: set[str] = {seed}
        refs: list[DocumentRef] = []
        pages = 0
        while queue and pages < max_pages:
            url = queue.pop(0)
            text = self.fetch_text(url)
            pages += 1
            reason = self.is_js_shell(text)
            if reason is not None:
                raise SourceUnavailable(reason, url=url, detail="client-rendered shell")
            refs.extend(self.parse_listing(text, base_url=url))
            for href, label in anchors(text):
                if not href or not matches(nav_pattern, f"{href} {label}"):
                    continue
                candidate = resolve(url, href)
                if candidate in visited or not same_site(seed, candidate):
                    continue
                visited.add(candidate)
                queue.append(candidate)
        return refs

    def fetch(self) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")
        seeds = [str(seed) for seed in (getattr(self.source, "seeds", None) or [])]
        if not seeds:
            raise SourceUnavailable(
                "no_seed",
                detail=f"{self.source_id} has no seeds; add university domains to sources.yaml",
            )

        refs: list[DocumentRef] = []
        seen: set[str] = set()
        skipped: list[tuple[str, str]] = []
        first_reason: str | None = None
        verified_sites = 0
        for seed in seeds:
            config = self.site_config(seed)
            if config is None:
                skipped.append((seed, "no_site_config"))
                continue
            if not config.get("verified", False):
                skipped.append((seed, "verify_first_unverified"))
                first_reason = first_reason or "verify_first_unverified"
                continue
            verified_sites += 1
            try:
                found = self._fetch_site(seed, config)
            except SourceUnavailable as exc:
                skipped.append((seed, exc.reason))
                first_reason = first_reason or exc.reason
                continue
            for ref in found:
                if ref.url in seen:
                    continue
                seen.add(ref.url)
                ref.metadata["seed"] = seed
                ref.metadata["skipped_sites"] = [f"{url}:{reason}" for url, reason in skipped]
                refs.append(ref)

        if not refs and verified_sites == 0:
            raise SourceUnavailable(
                first_reason or "no_verified_site",
                detail="no configured health-university site is verified for crawling",
            )
        if not refs and first_reason is not None:
            raise SourceUnavailable(first_reason, detail="every verified site failed this run")
        return refs
