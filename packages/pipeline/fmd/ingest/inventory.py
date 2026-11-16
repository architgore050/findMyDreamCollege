"""Parse an inventory payload into institution records (pure, offline).

The UGC ``Getuniversity_details`` XHR is a JSON object with a ``List`` member
(one object per university). AISHE/other inventories publish workbooks or JSON
with different keys. The shape is not hardcoded: each registry source declares
its field map under ``params.inventory``::

    params:
      inventory:
        country: IN
        fields:
          name: uni_name
          state: state
        # optional
          type: uni_type
          website: url
          status: status
        keep_statuses: ["2(f) & 12B"]

Parsing is defensive: a row without a name is dropped (never invented), values
are whitespace-collapsed, and the payload's own error shell is rejected loudly.
No PII is read: the field map can only name whitelisted columns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from fmd.discover.base import DocumentRef, params_from_source
from fmd.er.normalise import normalise_name


class InventoryPayloadError(ValueError):
    """The payload is not an inventory document (HTML shell, bad JSON, no rows)."""


@dataclass(frozen=True)
class InventoryFieldMap:
    """Registry-declared mapping from a payload row to an inventory record."""

    name: str
    state: str | None = None
    institution_type: str | None = None
    website: str | None = None
    status: str | None = None
    list_key: str | None = None
    country: str | None = None
    keep_statuses: tuple[str, ...] = ()

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> InventoryFieldMap:
        fields = params.get("fields") or {}
        if not isinstance(fields, Mapping) or not fields.get("name"):
            raise InventoryPayloadError(
                "params.inventory.fields.name is required; the ingester never guesses "
                "which column holds the institution name"
            )
        return cls(
            name=str(fields["name"]),
            state=_opt(fields.get("state")),
            institution_type=_opt(fields.get("type")),
            website=_opt(fields.get("website")),
            status=_opt(fields.get("status")),
            list_key=_opt(params.get("list_key")),
            country=_opt(params.get("country")),
            keep_statuses=tuple(str(value) for value in (params.get("keep_statuses") or ())),
        )

    @classmethod
    def from_source(cls, source: object) -> InventoryFieldMap:
        params = params_from_source(source)
        block = params.get("inventory")
        if not isinstance(block, Mapping):
            raise InventoryPayloadError(
                f"{getattr(source, 'id', 'source')} declares no params.inventory field map"
            )
        return cls.from_params(block)


@dataclass(frozen=True)
class InventoryRecord:
    """One institution named by an inventory payload."""

    name: str
    state: str | None = None
    institution_type: str | None = None
    website: str | None = None
    status: str | None = None
    country: str | None = None

    @property
    def normalised_name(self) -> str:
        return normalise_name(self.name).text

    @property
    def acronyms(self) -> tuple[str, ...]:
        """Aliases the name itself derives (``King George's Medical University`` → ``kgmu``)."""
        canonical = normalise_name(self.name)
        return tuple(
            acronym
            for acronym in canonical.acronyms
            if acronym and acronym.casefold() != canonical.text.replace(" ", "")
        )

    @property
    def domain(self) -> str | None:
        return domain_from_website(self.website)


def _opt(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _string(value: Any) -> str:
    return " ".join(str(value or "").split())


def parse_inventory(text: str, *, field_map: InventoryFieldMap) -> list[dict[str, Any]]:
    """Return the raw row dicts from a JSON inventory payload.

    Rejects the UGC-style 200 HTML error shell: a shell is not an empty
    inventory, and treating it as one is how a run reports success while
    learning nothing.
    """
    stripped = text.lstrip()
    if stripped.startswith("<"):
        raise InventoryPayloadError("payload is HTML, not an inventory document")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if "<html" in text.lower() or "<!doctype" in text.lower():
            raise InventoryPayloadError("payload is an HTML error shell") from exc
        raise InventoryPayloadError(f"payload is not valid JSON: {exc}") from exc

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        key = field_map.list_key
        if key is None:
            key = next((k for k, v in payload.items() if isinstance(v, list)), None)
        rows = payload.get(key) if key else None
        if rows is None:
            raise InventoryPayloadError(
                "payload has no row list; declare params.inventory.list_key"
            )
    else:
        raise InventoryPayloadError("payload is not a JSON object or list")
    return [row for row in rows if isinstance(row, Mapping)]


def records_from_payload(
    text: str, *, field_map: InventoryFieldMap
) -> tuple[list[InventoryRecord], list[dict[str, Any]]]:
    """Parse rows into records, returning ``(kept, dropped)`` with a reason each."""
    kept: list[InventoryRecord] = []
    dropped: list[dict[str, Any]] = []
    for raw in parse_inventory(text, field_map=field_map):
        name = _string(raw.get(field_map.name))
        if not name:
            dropped.append({"reason": "missing_name", "row": dict(raw)})
            continue
        status = _string(raw.get(field_map.status)) if field_map.status else None
        if field_map.keep_statuses and (status or "") not in field_map.keep_statuses:
            dropped.append({"reason": "status_filtered", "name": name, "status": status})
            continue
        kept.append(
            InventoryRecord(
                name=name,
                state=_opt(raw.get(field_map.state)) if field_map.state else None,
                institution_type=(
                    _opt(raw.get(field_map.institution_type))
                    if field_map.institution_type
                    else None
                ),
                website=_opt(raw.get(field_map.website)) if field_map.website else None,
                status=status,
                country=field_map.country,
            )
        )
    return kept, dropped


_EXPLICIT_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")


def _http_value(value: str) -> str | None:
    """Return an http(s) URL string, or ``None`` for mailto/tel/javascript/..."""
    candidate = str(value).strip()
    if not candidate:
        return None
    match = _EXPLICIT_SCHEME.match(candidate)
    if match:
        if match.group(1).lower() not in {"http", "https"}:
            return None
        return candidate
    return f"https://{candidate.lstrip('/')}"


def domain_from_website(value: str | None) -> str | None:
    """Registrable host from an inventory website field, or ``None``.

    The UGC payload mixes bare hosts (``www.example.ac.in``), schemes and
    trailing paths. We keep only an http(s) host; a mailto or javascript value is
    not a website and must not become a crawl seed.
    """
    if not value:
        return None
    candidate = _http_value(str(value))
    if candidate is None:
        return None
    host = (urlsplit(candidate).hostname or "").lower().strip(".")
    if not host or "." not in host:
        return None
    return host


def seed_url(value: str | None) -> str | None:
    if not value:
        return None
    candidate = _http_value(str(value))
    if candidate is None:
        return None
    parts = urlsplit(candidate)
    if not parts.hostname:
        return None
    return f"{parts.scheme}://{parts.netloc}/"


def document_refs(records: Iterable[InventoryRecord], *, source_id: str) -> list[DocumentRef]:
    """Website roots as crawl leads (never evidence; ``is_lead`` stays True)."""
    refs: list[DocumentRef] = []
    for record in records:
        url = seed_url(record.website)
        if url is None:
            continue
        refs.append(DocumentRef(url=url, source_id=source_id, metadata={"inventory": record.name}))
    return refs


__all__ = [
    "InventoryFieldMap",
    "InventoryPayloadError",
    "InventoryRecord",
    "document_refs",
    "domain_from_website",
    "parse_inventory",
    "records_from_payload",
    "seed_url",
]
