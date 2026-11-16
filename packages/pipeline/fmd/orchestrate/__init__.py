"""Orchestration layer: wires existing components into runnable stages.

Public surface:

* :class:`Orchestrator` — ``crawl``, ``extract``, ``score`` and ``worker``.
* :class:`RunSummary` — one JSON-serialisable run record.
* :class:`JsonFileLedger` / :class:`SqlLedger` — run persistence.
"""

from __future__ import annotations

from fmd.orchestrate.ledger import (
    JsonFileLedger,
    NullLedger,
    RunLedger,
    RunSummary,
    SqlLedger,
)
from fmd.orchestrate.orchestrator import OrchestrationError, Orchestrator

__all__ = [
    "JsonFileLedger",
    "NullLedger",
    "OrchestrationError",
    "Orchestrator",
    "RunLedger",
    "RunSummary",
    "SqlLedger",
]
