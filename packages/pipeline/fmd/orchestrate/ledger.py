"""Run ledger for the orchestration layer (brief §7, §22).

Every ``fmd crawl|extract|score|worker`` run produces one :class:`RunSummary`
that is printed to stdout as a single JSON object and persisted by a
:class:`RunLedger`.

Two implementations, one interface:

* :class:`SqlLedger` writes a ``pipeline_runs`` row (migration 0003). This is
  the default whenever a database is available: the ledger is queryable in SQL
  (counts and errors are JSONB, status is a closed set).
* :class:`JsonFileLedger` writes ``runs/<run_id>.json`` plus
  ``heartbeat.json`` under the snapshot root for offline runs and tests.

The interface stays a protocol so a caller can inject either.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fmd.db.repositories import PipelineRunCreate, PipelineRunsRepo


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
    """Ledger used by unit tests; persists nothing."""

    def record(self, summary: RunSummary) -> Path | None:
        return None

    def heartbeat(self, payload: dict[str, Any]) -> Path | None:
        return None


class SqlLedger:
    """Persist each run as a ``pipeline_runs`` row (the production default).

    The row is written even for a dry run, with ``status='dry_run'``: knowing
    that a dry run happened is observability, not evidence, and it never enters
    the evidence tables. ``record`` returns ``None`` because there is no file.
    """

    def __init__(self, database: Any = None, *, repo: Any | None = None) -> None:
        if repo is None and database is None:
            raise ValueError("SqlLedger needs a database or a PipelineRunsRepo")
        self.repo = repo if repo is not None else PipelineRunsRepo(database)

    def record(self, summary: RunSummary) -> Path | None:
        if summary.finished_at is None:
            summary.finish()
        status = "dry_run" if summary.dry_run else ("failed" if summary.failed else "succeeded")
        self.repo.record(
            PipelineRunCreate(
                run_id=summary.run_id,
                stage=summary.stage,
                started_at=summary.started_at,
                finished_at=summary.finished_at,
                counts=dict(summary.counts),
                errors=list(summary.failures),
                status=status,
            )
        )
        return None

    def heartbeat(self, payload: dict[str, Any]) -> Path | None:
        # Worker liveness is process-local; the run rows are the durable record.
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
