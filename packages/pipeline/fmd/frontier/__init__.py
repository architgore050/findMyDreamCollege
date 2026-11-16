"""Postgres-backed crawl frontier — workstream B (brief §7, §21).

TODO(WS-B): scheduler, per-domain budgets, ban/yield scores, quarantine.
WS-A provides only the thin ``CrawlFrontierRepo`` persistence primitives.
"""

from __future__ import annotations


def run_frontier(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("crawl frontier is workstream B (see brief §7)")
