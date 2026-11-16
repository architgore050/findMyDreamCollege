"""Run ledger for the orchestration layer (brief §7, §22).

Every ``fmd crawl|extract|score|worker`` run produces one :class:`RunSummary`
that is printed to stdout as a single JSON object and persisted by a
:class:`RunLedger`. No schema currently models a pipeline run, and migrations are
out of this workstream's scope, so the default ledger is a JSON file under the
snapshot root (``runs/<run_id>.json`` plus ``heartbeat.json`` for the worker).
The interface is deliberately a protocol: a future ``pipeline_runs`` migration
can implement :class:`RunLedger` without touching the orchestrator.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


@dataclass
class RunSummary:
    """One stage execution: what it was asked to do and what happened."""

    run_id: str
    stage: str
    started_at: datetime
    finished_at: datetime | None = None
    dry_run: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    planned_urls: list[str] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    failed: bool = False

    @property
    def duration_ms(self) -> int:
        if self.finished_at is None:
            return 0
        return max(0, int((self.finished_at - self.started_at).total_seconds() * 1000))

    def bump(self, key: str, amount: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + amount

    def fail(self, stage: str, reason: str, **detail: Any) -> None:
        """Record an isolated failure; never raises by itself."""
        entry: dict[str, Any] = {"stage": stage, "reason": reason}
        entry.update({key: value for key, value in detail.items() if value is not None})
        self.failures.append(entry)
        self.failed = True

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "stage": self.stage,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "duration_ms": self.duration_ms,
            "dry_run": self.dry_run,
            "ok": not self.failed,
            "counts": dict(sorted(self.counts.items())),
            "failures": self.failures,
        }
        if self.sources:
            payload["sources"] = self.sources
        if self.planned_urls:
            payload["planned_urls"] = self.planned_urls
        if self.outcomes:
            payload["outcomes"] = self.outcomes
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    def finish(self) -> RunSummary:
        self.finished_at = utcnow()
        return self


class RunLedger(Protocol):
    def record(self, summary: RunSummary) -> Path | None:  # pragma: no cover - protocol
        ...

    def heartbeat(self, payload: dict[str, Any]) -> Path | None:  # pragma: no cover - protocol
        ...


class NullLedger:
    """Ledger used by dry runs and unit tests; persists nothing."""

    def record(self, summary: RunSummary) -> Path | None:
        return None

    def heartbeat(self, payload: dict[str, Any]) -> Path | None:
        return None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


class JsonFileLedger:
    """Persist one JSON document per run plus a worker heartbeat file."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def heartbeat_path(self) -> Path:
        return self.root / "heartbeat.json"

    def record(self, summary: RunSummary) -> Path | None:
        if summary.dry_run:
            return None
        path = self.runs_dir / f"{summary.run_id}.json"
        _atomic_write(path, json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def heartbeat(self, payload: dict[str, Any]) -> Path | None:
        body = {"observed_at": utcnow().isoformat(), **payload}
        _atomic_write(self.heartbeat_path, json.dumps(body, sort_keys=True) + "\n")
        return self.heartbeat_path
