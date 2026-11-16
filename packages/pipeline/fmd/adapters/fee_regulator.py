"""State fee regulatory commission orders (spec §4.6).

Fee evidence is a *document*, never a number parsed here. This adapter walks a
state's order listing and yields refs for the order PDFs; WS-D extracts the
tuition heads. Orders are usually gazette scans, so the format claim travels as
``metadata["scanned"]`` rather than being guessed later.

Some states publish no order listing at all. Maharashtra's known portal is a
search form whose ``sub_type`` vocabulary is JS-populated and unknown today
(spec §4.6.1): its results cannot be requested honestly, so the adapter records
``form_replay_unverified`` instead of returning an empty success.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, matches, resolve
from fmd.discover.base import DocumentRef


class FeeRegulatorAdapter(BaseAdapter):
    name = "fee_regulator"

    # ------------------------------------------------------------------ config
    def _portals(self) -> list[dict[str, Any]]:
        raw = self.param("portals") or []
        if not isinstance(raw, list):
            raise SourceUnavailable(
                "invalid_registry_param",
                detail=f"{self.source_id} params.portals must be a list",
            )
        return [dict(entry) for entry in raw if isinstance(entry, dict)]

    def _scanned(self, url: str, label: str) -> bool:
        hint = self.param("scan_hint_pattern")
        if hint and matches(hint, f"{url} {label}"):
            return True
        return bool(self.param("assume_scanned", False))

    # ------------------------------------------------------------------ parsing
    def parse_listing(
        self,
        text: str,
        *,
        base_url: str | None = None,
        portal_id: str | None = None,
    ) -> list[DocumentRef]:
        """Extract order-PDF refs only; amounts stay in the PDF (pure)."""
        base = base_url or str(getattr(self.source, "base_url", "") or "")
        href_pattern = self.param("order_href_pattern") or r"(?i)\.pdf($|\?)"
        text_pattern = self.param("order_text_pattern")
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
            metadata: dict[str, Any] = {
                "kind": "fee_order",
                "anchor_text": label,
                "scanned": self._scanned(url, label),
            }
            if portal_id:
                metadata["portal_id"] = portal_id
            refs.append(self.ref(url, metadata=metadata))
        return refs

    def is_js_shell(self, text: str) -> bool:
        """A listing that is a form shell has no orders to read (spec §4.6.1)."""
        lowered = text.lower()
        return bool(self.param("form_markers")) and all(
            marker.lower() in lowered for marker in self.param("form_markers")
        )

    # -------------------------------------------------------------------- fetch
    def fetch(self) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")
        portals = self._portals()
        if not portals:
            raise SourceUnavailable(
                "no_listing_url",
                detail=f"{self.source_id} has no fee-order portal configured",
            )

        refs: list[DocumentRef] = []
        seen: set[str] = set()
        skipped: list[tuple[str, str]] = []
        form_only = False
        verified_reads = 0
        first_reason: str | None = None

        for portal in portals:
            portal_id = str(portal.get("id") or portal.get("state") or "unknown")
            listing_url = str(portal.get("listing_url") or "")
            if not listing_url:
                skipped.append((portal_id, "no_listing_url"))
                continue
            if not portal.get("verified", False):
                skipped.append((portal_id, "verify_first_unverified"))
                first_reason = first_reason or "verify_first_unverified"
                continue
            if str(portal.get("mode") or "html").lower() == "form":
                # The search is a POST whose sub_type vocabulary is unpublished.
                skipped.append((portal_id, "form_replay_unverified"))
                form_only = True
                first_reason = first_reason or "form_replay_unverified"
                continue
            text = self.fetch_text(listing_url)
            verified_reads += 1
            if self.is_js_shell(text):
                skipped.append((portal_id, "js_shell"))
                first_reason = first_reason or "js_shell"
                continue
            for ref in self.parse_listing(text, base_url=listing_url, portal_id=portal_id):
                if ref.url in seen:
                    continue
                seen.add(ref.url)
                ref.metadata["skipped_portals"] = [f"{key}:{why}" for key, why in skipped]
                refs.append(ref)

        if refs:
            return refs
        if form_only:
            raise SourceUnavailable(
                "form_replay_unverified",
                detail="the only verified fee portal needs a POST vocabulary a human must supply",
            )
        if verified_reads:
            return []  # empty-but-valid: an order listing with no orders this run
        raise SourceUnavailable(
            first_reason or "no_listing_url",
            detail="no fee-order listing could be read",
        )


def portal_host(portal: dict[str, Any]) -> str:
    """Metadata helper for provenance; never used to filter fetches."""
    return (urlsplit(str(portal.get("listing_url") or "")).hostname or "").lower()
