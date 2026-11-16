"""Discovery lead types (brief §7).

Discovery emits *leads*, never evidence. A lead is a URL plus its provenance
(the registry entry or query template that produced it). Parsing and claim
extraction happen downstream, behind the snapshot store.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class DiscoveryUnavailable(RuntimeError):
    """An optional discovery backend (duckdb, network) is not available."""


@dataclass(frozen=True)
class DocumentRef:
    """Satisfies ``fmd.adapters.DocumentRef`` and adds provenance.

    The optional ``etag`` / ``last_modified`` fields let a caller issue a
    conditional GET without re-reading ``DocumentsRepo``; ``last_modified`` is
    already an HTTP-date string so it can go straight into the header.
    """

    url: str
    source_id: str | None = None
    discovery_method: str | None = None
    query: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
    etag: str | None = None
    last_modified: str | None = None
    priority: int = 0

    @property
    def is_lead(self) -> bool:
        """Discovery output is never evidence (brief §7)."""
        return True


@runtime_checkable
class DiscoverySource(Protocol):
    name: str

    def discover(self, *args: object, **kwargs: object):  # pragma: no cover - protocol
        ...


def params_from_source(source: Any) -> dict[str, Any]:
    """Read the structured ``params`` block from a registry or seeded source.

    Accepts a pydantic ``Source`` (attribute ``params``), a raw YAML mapping, or
    a seeded ``SourceRecord`` whose config JSON nests the block under
    ``config.params``. Never invents defaults: discovery params are registry data.
    """
    raw: Any = None
    if isinstance(source, Mapping):
        raw = source.get("params")
        if raw is None:
            config = source.get("config")
            if isinstance(config, Mapping):
                raw = config.get("params")
    else:
        raw = getattr(source, "params", None)
        if raw is None:
            config = getattr(source, "config", None)
            if isinstance(config, Mapping):
                raw = config.get("params")
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(exclude_none=True)
    return dict(raw) if isinstance(raw, Mapping) else {}


def dedupe_refs(refs: list[DocumentRef]) -> list[DocumentRef]:
    """Stable first-seen dedupe by URL."""
    seen: set[str] = set()
    out: list[DocumentRef] = []
    for ref in refs:
        key = ref.url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out
