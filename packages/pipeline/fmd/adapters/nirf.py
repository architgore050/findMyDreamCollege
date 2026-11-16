"""NIRF ranking pages (spec §4.3.2).

Two hard limits are encoded here, not left to downstream discipline:

* **No programme-level claim.** NIRF publishes no paramedical/MLT category, so
  every ref carries ``institution_level_only=True``. A rank belongs to an
  institution and may never be attributed to a programme.
* **Browser UA is mandatory.** ``nirfindia.org`` answers the default crawler UA
  with a CloudFront 403. The fetch ladder escalates on that block signal; the
  refs carry ``fetch_tier_required="escalated"`` so the frontier records it and
  a persistent 403 is an abstention, not an empty success.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, matches, resolve
from fmd.discover.base import DocumentRef


class NirfAdapter(BaseAdapter):
    name = "nirf"

    #: NIRF evidence is institution-level by construction (spec §4.3.2).
    allowed_claim_scope = "institution"

    def _listing_urls(self) -> list[str]:
        urls = [str(url) for url in (self.param("listing_urls") or [])]
        if not urls and self.param("listing_url"):
            urls.append(str(self.param("listing_url")))
        if not urls:
            base = getattr(self.source, "base_url", None)
            if base:
                urls.append(str(base))
        return urls

    def _scope(self, url: str) -> dict[str, Any]:
        """Ranking year and category read from the URL path (registry patterns)."""
        metadata: dict[str, Any] = {
            "institution_level_only": True,
            "programme_claims_allowed": False,
            "fetch_tier_required": "escalated",
        }
        year_pattern = self.param("year_pattern")
        category_pattern = self.param("category_pattern")
        pdf_category_pattern = self.param("pdf_category_pattern")
        if year_pattern:
            found = re.search(year_pattern, url)
            if found:
                metadata["ranking_year"] = found.groupdict().get("year") or found.group(0)
        for pattern in (category_pattern, pdf_category_pattern):
            if not pattern:
                continue
            found = re.search(str(pattern), url)
            if found:
                metadata["ranking_category"] = found.groupdict().get("category") or found.group(0)
                break
        return metadata

    def parse_listing(self, text: str, *, base_url: str | None = None) -> list[DocumentRef]:
        """Yield the ranking page itself plus per-institute data-sheet PDFs (pure)."""
        base = base_url or (self._listing_urls() or [""])[0]
        pdf_pattern = self.param("pdf_href_pattern") or r"(?i)nirfpdfcdn/.*\.pdf$"
        refs: list[DocumentRef] = []
        if base:
            refs.append(
                self.ref(
                    base,
                    metadata={"kind": "ranking_page", **self._scope(base)},
                    priority=1,
                )
            )
        seen: set[str] = {base}
        for href, label in anchors(text):
            if not href or not matches(pdf_pattern, href):
                continue
            url = resolve(base, href)
            if url in seen:
                continue
            seen.add(url)
            metadata = {"kind": "institute_data_sheet", "anchor_text": label, **self._scope(url)}
            stem = url.rsplit("/", 1)[-1]
            institute_id = re.sub(r"(?i)\.pdf$", "", stem)
            if institute_id:
                metadata["institute_id"] = institute_id
            refs.append(self.ref(url, metadata=metadata))
        return refs

    def fetch(self) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")
        urls = self._listing_urls()
        if not urls:
            raise SourceUnavailable(
                "no_listing_url", detail=f"{self.source_id} has no ranking page configured"
            )
        refs: list[DocumentRef] = []
        seen: set[str] = set()
        read_any = False
        first_reason: str | None = None
        for url in urls:
            try:
                text = self.fetch_text(url)
            except SourceUnavailable as exc:
                first_reason = first_reason or exc.reason
                continue
            read_any = True
            for ref in self.parse_listing(text, base_url=url):
                if ref.url in seen:
                    continue
                seen.add(ref.url)
                refs.append(ref)
        if not read_any:
            # CloudFront 403 with a crawler UA is a block, never "no rankings".
            raise SourceUnavailable(
                first_reason or "blocked",
                detail="NIRF needs a browser-UA tier; a persistent block is an abstention",
            )
        return refs
