"""NAAC accreditation-results listing (spec §4.3.1).

One Joomla page links ~1,539 PDFs. ``http://`` times out; only ``https://`` works
and the certificate chain is incomplete, which is a fetch-layer trust problem,
not an adapter one. Filenames are irregular, so classification is by anchor
text/table context; the adapter only resolves and filters the links.
"""

from __future__ import annotations

from collections.abc import Iterable

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, matches, resolve
from fmd.discover.base import DocumentRef


class NaacAdapter(BaseAdapter):
    name = "naac"

    def parse_listing(self, text: str) -> list[DocumentRef]:
        base = self.param("listing_url") or getattr(self.source, "base_url", "")
        href_pattern = self.param("pdf_href_pattern", r"\.pdf$")
        text_pattern = self.param("link_text_pattern")
        refs: list[DocumentRef] = []
        seen: set[str] = set()
        for href, label in anchors(text):
            if not href or not matches(href_pattern, href):
                continue
            if not matches(text_pattern, label):
                continue
            url = resolve(base, href)
            if url in seen:
                continue
            seen.add(url)
            refs.append(self.ref(url, metadata={"anchor_text": label}))
        return refs

    def fetch(self) -> Iterable[DocumentRef]:
        url = self.param("listing_url") or getattr(self.source, "base_url", None)
        if not url:
            raise SourceUnavailable("no_listing_url", detail="naac has no listing URL")
        return self.parse_listing(self.fetch_text(url))
