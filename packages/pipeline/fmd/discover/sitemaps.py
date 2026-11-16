"""Bounded sitemap discovery (brief §7).

Sitemaps are expanded per domain with hard bounds on depth, child sitemaps and
emitted URLs. The parser is stdlib-only and never touches the network directly:
callers inject a byte fetcher (normally the fetch ladder) so robots/politeness
still apply.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit

from fmd.discover.base import DocumentRef, dedupe_refs

logger = logging.getLogger(__name__)

Fetcher = Callable[[str], bytes | str | None]
MAX_SITEMAPS_PER_DOMAIN = 25
MAX_URLS_PER_DOMAIN = 2_000
MAX_DEPTH = 2


def _parse_locs(text: str) -> list[str]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        logger.warning("invalid sitemap XML: %s", exc)
        return []
    locs: list[str] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "loc" and element.text:
            locs.append(element.text.strip())
    return locs


def _decode(body: bytes | str) -> str:
    if isinstance(body, str):
        return body
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return body.decode("utf-8", errors="replace")


class SitemapDiscovery:
    def __init__(
        self,
        fetch: Fetcher,
        *,
        source_id: str = "institution_sites",
        max_sitemaps: int = MAX_SITEMAPS_PER_DOMAIN,
        max_urls: int = MAX_URLS_PER_DOMAIN,
        max_depth: int = MAX_DEPTH,
    ) -> None:
        self.fetch = fetch
        self.source_id = source_id
        self.max_sitemaps = max_sitemaps
        self.max_urls = max_urls
        self.max_depth = max_depth

    def _fetch_text(self, url: str) -> str | None:
        try:
            body = self.fetch(url)
        except Exception as exc:
            logger.warning("sitemap fetch failed for %s: %s", url, exc)
            return None
        if body is None:
            return None
        return _decode(body)

    def discover(
        self,
        sitemap_urls: list[str],
        *,
        domain: str | None = None,
    ) -> list[DocumentRef]:
        """Expand sitemap URLs; if ``domain`` is set, keep only that host."""
        queue: list[tuple[str, int]] = [(url, 0) for url in sitemap_urls]
        visited: set[str] = set()
        refs: list[DocumentRef] = []

        while queue and len(visited) < self.max_sitemaps and len(refs) < self.max_urls:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            text = self._fetch_text(url)
            if text is None:
                continue
            locs = _parse_locs(text)
            is_index = "<sitemapindex" in text
            for loc in locs:
                absolute = urljoin(url, loc)
                if is_index:
                    if depth < self.max_depth:
                        queue.append((absolute, depth + 1))
                    continue
                parts = urlsplit(absolute)
                if parts.scheme not in ("http", "https"):
                    continue
                if domain and not (parts.hostname or "").endswith(domain):
                    continue
                refs.append(
                    DocumentRef(
                        url=absolute,
                        source_id=self.source_id,
                        discovery_method="sitemap",
                        metadata={"sitemap": url},
                    )
                )
                if len(refs) >= self.max_urls:
                    break
        return dedupe_refs(refs)
