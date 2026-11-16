"""AISHE final reports (spec §4.1.3).

The index page is WordPress; the payload is a static ``.xlsx`` whose path is
built from the report id (``assets/excel/<YYYY-YY>.xlsx``). The path was recorded
from the verified Angular route, but a fetch from the dev host returned
connection code **000** (DNS/TLS/egress, not 404). This source is therefore
``verify_first: true`` / ``enabled: false`` and the adapter **abstains with an
explicit status** until a human flips ``params.verified``. It never turns an
unreachable source into an empty success.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors
from fmd.discover.base import DocumentRef


class AisheAdapter(BaseAdapter):
    name = "aishe"

    def report_ids(self, text: str) -> list[str]:
        pattern = self.param("report_link_pattern")
        if not pattern:
            raise SourceUnavailable(
                "missing_registry_param", detail="aishe needs params.report_link_pattern"
            )
        ids: list[str] = []
        for href, label in anchors(text):
            for value in (href, label):
                for match in re.findall(pattern, value):
                    report_id = match if isinstance(match, str) else match[0]
                    if report_id not in ids:
                        ids.append(report_id)
        return ids

    def parse_listing(self, text: str) -> list[DocumentRef]:
        template = self.param("asset_url_template")
        if not template:
            raise SourceUnavailable(
                "missing_registry_param", detail="aishe needs params.asset_url_template"
            )
        return [
            self.ref(template.format(report_id=report_id), metadata={"report_id": report_id})
            for report_id in self.report_ids(text)
        ]

    def fetch(self) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")
        verified = bool(self.param("verified", False))
        if getattr(self.source, "verify_first", False) and not verified:
            raise SourceUnavailable(
                "verify_first_unverified",
                detail=(
                    "AISHE static xlsx path returned connection code 000 from the dev host; "
                    "set params.verified after a successful fetch"
                ),
            )
        url = self.param("listing_url") or getattr(self.source, "base_url", None)
        if not url:
            raise SourceUnavailable("no_listing_url", detail="aishe has no index URL")
        return self.parse_listing(self.fetch_text(url))
