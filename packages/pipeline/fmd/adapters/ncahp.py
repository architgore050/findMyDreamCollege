"""National Commission for Allied and Healthcare Professions (NCAHP).

AHP Registration under the Act 2021 keeps an official recognising list of
institutions with the MLS 2025 curriculum. This adapter walks the listing page
at the registry's ``base_url`` and yields refs for the PDF recognition
notices; the listing shape follows the structural samples used by the other
recognising-authority adapters (fee_regulator).

Discovery-only: the adapter never parses notice text into claims (the
extraction layer owns that), so published recognition status stays
evidence-grade.
"""

from __future__ import annotations

from collections.abc import Iterable

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, matches, resolve
from fmd.discover.base import DocumentRef


class NcahpAdapter(BaseAdapter):
    name = "ncahp"

    # ------------------------------------------------------------------ parsing
    def parse_listing(self, text: str, *, base_url: str | None = None) -> list[DocumentRef]:
        """Return refs for recognising PDF anchors; nothing else is read."""
        base = base_url or str(getattr(self.source, "base_url", "") or "")
        href_pattern = self.param("pdf_href_pattern") or r"(?i)\.pdf($|\?)"
        source = self.source
        refs: list[DocumentRef] = []
        seen: set[str] = set()
        for href, label in anchors(text):
            if not href or not matches(href_pattern, href):
                continue
            url = resolve(base, href)
            if url in seen:
                continue
            seen.add(url)
            refs.append(
                self.ref(
                    url,
                    metadata={
                        "kind": "recognition_document",
                        "anchor_text": label,
                        "source_id": self.source_id,
                        "source_name": str(getattr(source, "name", "") or ""),
                        "source_notes": str(getattr(source, "notes", "") or ""),
                    },
                )
            )
        return refs

    # -------------------------------------------------------------------- fetch
    def fetch(self) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")
        listing_url = str(getattr(self.source, "base_url", "") or "")
        if not listing_url:
            raise SourceUnavailable(
                "no_listing_url",
                detail=f"{self.source_id} has no base_url configured",
            )
        text = self.fetch_text(listing_url)
        # Empty-but-valid: a loaded listing with no PDFs is a successful crawl.
        return self.parse_listing(text, base_url=listing_url)
