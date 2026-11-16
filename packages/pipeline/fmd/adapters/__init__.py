"""Source adapter registry (workstream C owns implementations, brief §8).

``config/sources.yaml`` maps every source to an ``adapter`` name. This module is
the code-side declaration of which adapter implementations exist. Adding a
source family means adding an implementation module here *and* updating this
registry; a config entry pointing at an unknown name fails registry loading.

No adapters are implemented in WS-A. Each listed adapter lives in
``fmd/adapters/<name>.py`` once WS-C lands and must expose
``fetch() -> Iterable[DocumentRef]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentRef(Protocol):
    """Minimal fetch-layer contract (implemented by WS-B/WS-C, brief §7).

    ``etag`` / ``last_modified`` carry a previous response's validators so a
    caller can issue a conditional GET without re-reading ``DocumentsRepo``.
    ``last_modified`` is an HTTP-date string. ``priority`` is the frontier hint.
    """

    url: str
    source_id: str | None
    etag: str | None
    last_modified: str | None
    priority: int


@runtime_checkable
class SourceAdapter(Protocol):
    """Contract from AGENTS.md: no parsing in the fetch layer."""

    name: str

    def fetch(self) -> Iterable[DocumentRef]:  # pragma: no cover - protocol
        ...


# Adapter identifiers backed by an implementation module. Keep alphabetical.
KNOWN_ADAPTERS: frozenset[str] = frozenset(
    {
        "aishe",
        "common_crawl",
        "fee_regulator",
        "financing",
        "health_university",
        "institution_site",
        "nabh",
        "nabl",
        "naac",
        "ncahp",
        "nirf",
        "pay_matrix",
        "searxng",
        "state_council",
        "state_counselling",
        "ugc",
        "ugc_pdf",
    }
)


def is_known_adapter(name: str) -> bool:
    return name in KNOWN_ADAPTERS
