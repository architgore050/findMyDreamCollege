"""UGC recognised-university JSON XHR (spec §4.1.1).

The endpoint is a JSON XHR, not an HTML directory. UGC returns **HTTP 200 with a
~51 KB HTML error shell** for unknown paths, so a 200 is not evidence of a
document (spec §1, §4.1.2). The response also leaks ``user_password``,
``user_username`` and contact fields; only the registry's ``field_whitelist`` may
enter a ``DocumentRef`` and every ``pii_fields`` key is dropped unconditionally.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from urllib.parse import urlencode, urlsplit

from fmd.adapters.base import BaseAdapter, SourceUnavailable, compact_record
from fmd.discover.base import DocumentRef


def _host(url: str) -> str:
    value = url if "://" in url else f"https://{url.lstrip('/')}"
    return (urlsplit(value).hostname or "").lower()


class UgcErrorShell(SourceUnavailable):
    """UGC served its 200-status HTML error shell instead of JSON."""

    def __init__(self, *, url: str | None = None, detail: str | None = None) -> None:
        super().__init__("ugc_error_shell", url=url, detail=detail)


class UgcAdapter(BaseAdapter):
    name = "ugc"

    # ------------------------------------------------------------------ parsing
    def _payload(self, text: str, *, url: str | None = None) -> dict:
        stripped = text.lstrip()
        if stripped.startswith("<"):
            raise UgcErrorShell(url=url, detail="response is HTML, not JSON")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if "<html" in text.lower() or "<!doctype" in text.lower():
                raise UgcErrorShell(url=url, detail="HTML error shell") from exc
            raise SourceUnavailable("invalid_json", url=url, detail=str(exc)) from exc
        if not isinstance(payload, dict) or "List" not in payload:
            raise UgcErrorShell(url=url, detail="JSON has no List member")
        if payload.get("c") not in (None, "success"):
            raise UgcErrorShell(url=url, detail=f"status flag c={payload.get('c')!r}")
        return payload

    @property
    def _field_whitelist(self) -> list[str]:
        return list(self.param("field_whitelist", []) or [])

    def normalise_row(self, row: dict) -> dict:
        record = compact_record(
            row,
            field_whitelist=self._field_whitelist,
            pii_fields=self.pii_fields,
        )
        for key in ("state", "uni_name", "uni_type", "status", "url"):
            if isinstance(record.get(key), str):
                record[key] = record[key].strip()
        return record

    def normalise_rows(self, text: str) -> list[dict]:
        return [self.normalise_row(row) for row in self._payload(text).get("List") or []]

    def parse_listing(self, text: str) -> list[DocumentRef]:
        """Return one seed ref per university row that publishes a website.

        Rows without a website stay in :meth:`normalise_rows` (so the inventory
        is complete) but cannot become fetch targets. ``metadata`` is the
        whitelisted record only; PII is already gone.
        """
        refs: list[DocumentRef] = []
        for record in self.normalise_rows(text):
            url = record.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            refs.append(
                self.ref(
                    url.strip(),
                    metadata={
                        "ugc": record,
                        # Role/domain emission: a UGC row names an institution,
                        # it is not evidence (migration 0004).
                        "document_role": "inventory",
                        "domain": _host(url),
                    },
                )
            )
        return refs

    # -------------------------------------------------------------------- fetch
    def fetch(self) -> Iterable[DocumentRef]:
        endpoint = self.param("api_endpoint") or getattr(self.source, "base_url", None)
        unitype_param = self.param("unitype_param")
        unitype_ids = self.param("unitype_ids") or []
        if not endpoint or not unitype_param or not unitype_ids:
            raise SourceUnavailable(
                "missing_registry_param",
                detail="ugc needs params.api_endpoint/unitype_param/unitype_ids",
            )
        refs: list[DocumentRef] = []
        for unitype_id in unitype_ids:
            url = f"{endpoint}?{urlencode({unitype_param: unitype_id})}"
            body = self.fetch_text(url)
            # Validate now so a 200 error shell aborts the run instead of
            # entering the frontier as a "document".
            payload = self._payload(body, url=url)
            if not payload.get("List"):
                # A valid but empty listing yields nothing; it is not a failure.
                continue
            refs.append(self.ref(url, metadata={"unitype_id": unitype_id}))
        return refs
