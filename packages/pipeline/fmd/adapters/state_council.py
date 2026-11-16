"""State allied-health / paramedical councils (spec §4.2.2).

The authoritative fact is the **NCAHP constituting notification** per state.
Only Kerala publishes anything resembling a council site, and it publishes no
institution register; ~29 states have no public site at all. This adapter
therefore yields document refs and provenance, never a recognition claim.

Two safety rules are enforced *before* any fetch:

1. a URL on ``denylisted_domains`` is refused outright — the confirmed
   impersonators registered under the Indian Trusts Act may never be crawled
   or cited;
2. any document whose origin is neither the NCAHP notification list nor a
   human-approved statutory host (``*.gov.in``/``*.nic.in``/``*.ac.in``) is
   flagged ``untrusted_origin`` so downstream claim code cannot mistake it for
   statutory evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import unquote, urlsplit

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, matches, resolve
from fmd.config.loader import denied_domain
from fmd.discover.base import DocumentRef


def _iso_date(token: str) -> str | None:
    """Normalise ``D.M.YYYY`` / ``D-M-YYYY`` to ISO, else ``None``."""
    found = re.fullmatch(r"(\d{1,2})[.\-](\d{1,2})[.\-](\d{4})", token.strip())
    if not found:
        return None
    day, month, year = (int(part) for part in found.groups())
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


class StateCouncilAdapter(BaseAdapter):
    name = "state_council"

    def __init__(
        self,
        source: Any,
        *,
        ladder: Any = None,
        denylisted_domains: Iterable[str] | None = None,
    ) -> None:
        super().__init__(source, ladder=ladder)
        if denylisted_domains is not None:
            domains = list(denylisted_domains)
        else:
            domains = list(self.param("denylisted_domains") or [])
        self.denylisted_domains = tuple(str(domain) for domain in domains)

    # ------------------------------------------------------------------ policy
    def refused_domain(self, url: str) -> str | None:
        """Return the denylisted suffix matching ``url``, or ``None``."""
        return denied_domain(url, self.denylisted_domains)

    def _refuse_denylisted(self, urls: Iterable[str]) -> None:
        for url in urls:
            if not url:
                continue
            match = self.refused_domain(url)
            if match is not None:
                raise SourceUnavailable(
                    "denylisted_domain",
                    url=url,
                    detail=f"{match!r} is a confirmed impersonator; never fetch it (spec §4.2.2)",
                )

    def origin(self, url: str) -> str:
        """Classify a URL's authority for a recognition claim."""
        host = (urlsplit(url).hostname or "").lower()
        ncahp_host = str(self.param("ncahp_host_suffix") or "").lower()
        if ncahp_host and (host == ncahp_host or host.endswith("." + ncahp_host)):
            return "ncahp_notification"
        suffixes = [
            str(suffix).lower().lstrip(".")
            for suffix in (self.param("statutory_host_suffixes") or [])
        ]
        if any(host == suffix or host.endswith("." + suffix) for suffix in suffixes):
            return "statutory_host"
        return "untrusted_origin"

    def _origin_metadata(self, url: str) -> dict[str, Any]:
        authority = self.origin(url)
        return {
            "authority": authority,
            "untrusted_origin": authority == "untrusted_origin",
        }

    # ------------------------------------------------------------------ parsing
    def _state_token(self, url: str) -> str | None:
        """Derive a state *token* from the filename; canonicalisation is ER's job."""
        name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        name = re.sub(r"(?i)\.pdf$", "", name)
        words = re.split(r"[^A-Za-z]+", name)
        strip = {str(word).lower() for word in (self.param("state_token_strip") or [])}
        for word in words:
            lowered = word.lower()
            if len(lowered) < 2 or lowered in strip or lowered.isdigit():
                continue
            return lowered
        return None

    def parse_listing(self, text: str, *, base_url: str | None = None) -> list[DocumentRef]:
        """Extract notification/rules PDF refs with state token + date (pure)."""
        base = base_url or str(self.param("listing_url") or getattr(self.source, "base_url", ""))
        href_pattern = self.param("notification_href_pattern") or r"(?i)/statecouncil/.*\.pdf$"
        rules_pattern = self.param("rules_href_pattern")
        date_pattern = self.param("notification_date_pattern")
        require_ncahp = bool(self.param("require_ncahp_host", False))
        refs: list[DocumentRef] = []
        seen: set[str] = set()
        for href, label in anchors(text):
            if not href or not matches(href_pattern, href):
                continue
            url = resolve(base, href)
            if url in seen or self.refused_domain(url) is not None:
                continue
            if require_ncahp and self.origin(url) != "ncahp_notification":
                continue
            seen.add(url)
            metadata: dict[str, Any] = {
                "kind": "council_rules"
                if rules_pattern and matches(rules_pattern, href)
                else "council_notification",
                "anchor_text": label,
                **self._origin_metadata(url),
            }
            token = self._state_token(url)
            if token:
                metadata["state_token"] = token
            if date_pattern:
                found = re.search(date_pattern, label) or re.search(date_pattern, href)
                if found:
                    token = found.group("date") if "date" in found.groupdict() else found.group(0)
                    metadata["notification_date_token"] = token
                    iso = _iso_date(token)
                    if iso:
                        metadata["notification_date"] = iso
            refs.append(self.ref(url, metadata=metadata))
        return refs

    def _council_site_refs(self) -> list[DocumentRef]:
        refs: list[DocumentRef] = []
        for entry in self.param("council_sites") or []:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or "")
            if not url or self.refused_domain(url) is not None:
                continue
            source_url = str(entry.get("source_url") or "")
            admitted = bool(source_url) and self.origin(source_url) == "ncahp_notification"
            origin_metadata = self._origin_metadata(url)
            metadata: dict[str, Any] = {
                "kind": "council_site",
                "source_url": source_url or None,
                "authority": "ncahp_admitted" if admitted else origin_metadata["authority"],
                "untrusted_origin": not admitted and origin_metadata["untrusted_origin"],
            }
            if entry.get("state_token"):
                metadata["state_token"] = str(entry["state_token"])
            refs.append(self.ref(url, metadata=metadata, priority=1))
        return refs

    # -------------------------------------------------------------------- fetch
    def _declared_urls(self) -> list[str]:
        urls: list[str] = []
        base = getattr(self.source, "base_url", None)
        if base:
            urls.append(str(base))
        urls.extend(str(seed) for seed in (getattr(self.source, "seeds", None) or []))
        listing = self.param("listing_url")
        if listing:
            urls.append(str(listing))
        urls.extend(str(url) for url in (self.param("listing_urls") or []))
        urls.extend(
            str(entry.get("url"))
            for entry in (self.param("council_sites") or [])
            if isinstance(entry, dict) and entry.get("url")
        )
        return urls

    def fetch(self) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")

        declared = self._declared_urls()
        # Refuse before the first byte leaves the process.
        self._refuse_denylisted(declared)

        refs: list[DocumentRef] = []
        seen: set[str] = set()
        for entry in self._council_site_refs():
            if entry.url not in seen:
                seen.add(entry.url)
                refs.append(entry)

        listing_urls = [str(self.param("listing_url"))] if self.param("listing_url") else []
        listing_urls.extend(str(url) for url in (self.param("listing_urls") or []))
        if not listing_urls:
            base = getattr(self.source, "base_url", None)
            if base:
                listing_urls.append(str(base))

        first_reason: str | None = None
        read_any = False
        for listing_url in listing_urls:
            try:
                text = self.fetch_text(listing_url)
            except SourceUnavailable as exc:
                first_reason = first_reason or exc.reason
                continue
            read_any = True
            for ref in self.parse_listing(text, base_url=listing_url):
                if ref.url in seen:
                    continue
                seen.add(ref.url)
                refs.append(ref)

        if not refs and not read_any:
            raise SourceUnavailable(
                first_reason or "no_listing_url",
                detail="no NCAHP state-council listing could be read",
            )
        return refs
