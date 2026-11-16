"""Postgres-backed crawl frontier — workstream B (brief §7, §21).

Queue/budget primitives live in :mod:`fmd.frontier.queue`; persisted per-domain
state lives in :mod:`fmd.frontier.state`. Imports are lazy so
``fmd.frontier.state`` can be used by the fetch package without a cycle.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Frontier",
    "InMemoryCrawlStats",
    "InMemoryDomainStateStore",
    "InMemoryFrontierRepo",
    "PostgresCrawlStats",
    "PostgresDomainStateStore",
]


def __getattr__(name: str) -> Any:
    if name == "Frontier":
        from fmd.frontier.queue import Frontier

        return Frontier
    if name in {"InMemoryCrawlStats", "InMemoryFrontierRepo", "PostgresCrawlStats"}:
        from fmd.frontier import queue

        return getattr(queue, name)
    if name in {"InMemoryDomainStateStore", "PostgresDomainStateStore"}:
        from fmd.frontier import state

        return getattr(state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
