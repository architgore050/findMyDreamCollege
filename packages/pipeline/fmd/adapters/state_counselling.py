"""State counselling notice listings (spec §4.5).

Every state row is the same shape: a notice listing page (HTML) linking dated
PDFs (seat matrix, cutoff, brochure, schedule). The per-state URLs, href regex,
link-text regex and date regex live in ``config/sources.yaml``; nothing about a
state, course or host is hardcoded here. The application portals are auth-walled
and out of scope; this adapter only yields public notices.

An empty-but-valid listing returns ``[]``. A robots refusal, block, transport
failure or missing registry URL raises :class:`SourceUnavailable`, because
"could not look" is not "no seats".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, matches, resolve
from fmd.discover.base import DocumentRef


class StateCounsellingAdapter(BaseAdapter):
    name = "state_counselling"

    def _listing_urls(self) -> list[str]:
        urls = list(self.param("listing_urls") or [])
        if not urls:
            listing = self.param("listing_url")
            if listing:
                urls.append(listing)
        if not urls:
            base = getattr(self.source, "base_url", None)
            if base:
                urls.append(base)
        return urls

    def parse_listing(self, text: str, *, base_url: str | None = None) -> list[DocumentRef]:
        base = base_url or (self._listing_urls() or [""])[0]
        href_pattern = self.param("pdf_href_pattern", r"\.pdf$")
        text_pattern = self.param("link_text_pattern")
        date_pattern = self.param("date_pattern")
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
            metadata: dict[str, object] = {"anchor_text": label}
            if date_pattern:
                found = re.search(date_pattern, href) or re.search(date_pattern, label)
                if found:
                    metadata["notice_date_token"] = (
                        found.group("date") if "date" in found.groupdict() else found.group(0)
                    )
            refs.append(self.ref(url, metadata=metadata))
        return refs

    def fetch(self) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")
        if getattr(self.source, "verify_first", False) and not self.param("verified", False):
            raise SourceUnavailable(
                "verify_first_unverified",
                detail="listing URL is unverified; a human must confirm it before crawling",
            )
        urls = self._listing_urls()
        if not urls:
            raise SourceUnavailable(
                "no_listing_url", detail=f"{self.source_id} has no verified listing URL"
            )
        refs: list[DocumentRef] = []
        seen: set[str] = set()
        for url in urls:
            for ref in self.parse_listing(self.fetch_text(url), base_url=url):
                if ref.url not in seen:
                    seen.add(ref.url)
                    refs.append(ref)
        return refs
