"""7th CPC pay-matrix anchor (spec §4.7).

This adapter emits exactly one document ref and stops. The known PDF
(``l4Paymatrix.pdf``) is **Level 4**, one page, and scanned: the text layer is
empty, so it must go to the T2/VLM path rather than a born-digital PDF reader.
The adapter never parses a pay cell, never multiplies a matrix cell by a DA
rate, and never presents basic pay as gross pay.

The Level 5/6 matrix is not verified — ``l5``/``l6``/``level5``/``level6`` all
404 (spec §4.7) — so only Level 4 is configured. A missing or unverified URL is
an abstention.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fmd.adapters.base import BaseAdapter, SourceUnavailable
from fmd.discover.base import DocumentRef


class PayMatrixAdapter(BaseAdapter):
    name = "pay_matrix"

    def document_url(self) -> str | None:
        return str(self.param("document_url") or getattr(self.source, "base_url", "") or "") or None

    def parse_listing(self, text: str, *, base_url: str | None = None) -> list[DocumentRef]:
        """No listing exists; the source *is* one document (pure, for symmetry)."""
        url = base_url or self.document_url()
        return [self._ref(url)] if url else []

    def _ref(self, url: str) -> DocumentRef:
        metadata: dict[str, Any] = {
            "kind": "pay_matrix",
            "level": getattr(self.source, "level", None),
            "scanned": bool(getattr(self.source, "scanned", False)),
            "verify_first": bool(getattr(self.source, "verify_first", False)),
            "do_not_parse": True,
            "format": getattr(self.source, "format", None),
        }
        return self.ref(url, metadata=metadata, priority=1)

    def fetch(self) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")
        url = self.document_url()
        if not url:
            raise SourceUnavailable(
                "no_document_url",
                detail=f"{self.source_id} has no pay-matrix PDF configured",
            )
        if getattr(self.source, "verify_first", False) and not self.param("verified", False):
            raise SourceUnavailable(
                "verify_first_unverified",
                url=url,
                detail="the pay-matrix URL is unverified; a human must confirm it first",
            )
        return [self._ref(url)]
