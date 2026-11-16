"""Orchestration facade: the one object the CLI and worker drive.

``Orchestrator`` wires the components that already exist — registry, adapters,
frontier, fetch ladder, snapshot store, extract pipeline and scorer — and emits a
:class:`RunSummary` per stage. Dependencies are injectable so the whole thing runs
offline with a fake adapter, fake fetcher, in-memory repositories and a tmp
snapshot store; the production constructor uses Postgres repositories.
"""

from __future__ import annotations

import logging
import signal
import threading
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from fmd.config import Registry
from fmd.db.connection import Database
from fmd.db.repositories import (
    AliasesRepo,
    ClaimsRepo,
    DocumentsRepo,
    EntitiesRepo,
    ExtractionsRepo,
    ProgrammesRepo,
    ScoresRepo,
    SnapshotsRepo,
)
from fmd.fetch.ladder import FetchLadder
from fmd.frontier.queue import Frontier
from fmd.orchestrate.adapters import build_adapter
from fmd.orchestrate.crawl import DEFAULT_SOURCE_FAILURE_BUDGET, Crawler
from fmd.orchestrate.extract import DEFAULT_TOOL_VERSION, Extractor, build_extract_router
from fmd.orchestrate.ledger import JsonFileLedger, RunLedger, RunSummary, SqlLedger, utcnow
from fmd.orchestrate.score import score_programmes
from fmd.paths import resolve_snapshot_dir
from fmd.plan import DocumentPlanner
from fmd.store import SnapshotStore

logger = logging.getLogger(__name__)


class OrchestrationError(RuntimeError):
    """The orchestrator cannot start with the given inputs."""


class Orchestrator:
    def __init__(
        self,
        *,
        registry: Registry,
        database: Database | None = None,
        store: Any | None = None,
        frontier: Frontier | None = None,
        documents: Any | None = None,
        snapshots: Any | None = None,
        extractions: Any | None = None,
        claims: Any | None = None,
        programmes: Any | None = None,
        scores: Any | None = None,
        entities: Any | None = None,
        sources: Iterable[Any] | None = None,
        adapter_factory: Callable[[Any], Any] | None = None,
        fetcher: Any | None = None,
        ledger: RunLedger | None = None,
        router: Any | None = None,
        run_id_factory: Callable[[], str] | None = None,
        max_source_failures: int = DEFAULT_SOURCE_FAILURE_BUDGET,
        tool_version: str = DEFAULT_TOOL_VERSION,
        score_persist: Callable[[Any, Any], Any] | None = None,
        planner: Any | None = None,
    ) -> None:
        self.registry = registry
        self.database = database
        self.score_persist = score_persist
        self._sources = list(sources) if sources is not None else None
        self.run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)
        self.tool_version = tool_version
        self.max_source_failures = max_source_failures

        self.store = store
        self.documents = (
            documents if documents is not None else self._require_database(DocumentsRepo)
        )
        self.snapshots = (
            snapshots if snapshots is not None else self._require_database(SnapshotsRepo)
        )
        self.extractions = (
            extractions if extractions is not None else self._require_database(ExtractionsRepo)
        )
        self.claims = claims if claims is not None else self._require_database(ClaimsRepo)
        self.programmes = (
            programmes if programmes is not None else self._require_database(ProgrammesRepo)
        )
        self.scores = scores if scores is not None else self._require_database(ScoresRepo)
        self.entities = entities if entities is not None else self._optional(EntitiesRepo)
        if frontier is None:
            if database is None:
                raise OrchestrationError("crawl requires a database or an injected frontier")
            frontier = Frontier.from_database(database)
        self.frontier = frontier
        self.fetcher = fetcher if fetcher is not None else FetchLadder()
        self.adapter_factory = adapter_factory or (
            lambda source: build_adapter(
                source,
                ladder=self.fetcher,
                denylisted_domains=self.registry.sources.denied_domains(),
            )
        )
        self.ledger: RunLedger = ledger if ledger is not None else self._default_ledger()
        self.router = router or build_extract_router(registry)
        self.planner = (
            planner
            if planner is not None
            else DocumentPlanner(
                entities=self.entities,
                aliases=AliasesRepo(database) if database is not None else None,
                er_config=registry.er,
            )
        )

    # ------------------------------------------------------------------ helpers
    def _require_database(self, repo: type) -> Any:
        if self.database is None:
            raise OrchestrationError(
                f"{repo.__name__} is required: pass a database or inject the repository"
            )
        return repo(self.database)

    def _optional(self, repo: type) -> Any | None:
        return None if self.database is None else repo(self.database)

    def _store(self) -> Any:
        if self.store is None:
            self.store = SnapshotStore(resolve_snapshot_dir())
        return self.store

    def _default_ledger(self) -> RunLedger:
        # With Postgres available the run ledger is the ``pipeline_runs`` table;
        # offline runs keep the JSON file under the snapshot root.
        if self.database is not None:
            return SqlLedger(self.database)
        root = getattr(self.store, "root", None)
        return JsonFileLedger(root) if root is not None else JsonFileLedger(resolve_snapshot_dir())

    def _summary(self, stage: str, *, dry_run: bool = False) -> RunSummary:
        return RunSummary(
            run_id=self.run_id_factory(),
            stage=stage,
            started_at=utcnow(),
            dry_run=dry_run,
        )

    def _all_sources(self) -> list[Any]:
        if self._sources is not None:
            return [source for source in self._sources if getattr(source, "enabled", True)]
        return list(self.registry.enabled_sources())

    def _selected_sources(self, source_ids: Iterable[str] | None) -> list[Any]:
        available = self._all_sources()
        if not source_ids:
            return available
        wanted = list(dict.fromkeys(source_ids))
        found = [source for source in available if str(getattr(source, "id", "")) in wanted]
        missing = [
            source_id
            for source_id in wanted
            if source_id not in {str(getattr(source, "id", "")) for source in found}
        ]
        if missing:
            raise OrchestrationError(
                "unknown or disabled source(s): "
                + ", ".join(missing)
                + ". Enabled: "
                + ", ".join(str(getattr(source, "id", "")) for source in available)
            )
        return found

    def _record(self, summary: RunSummary, *, dry_run: bool) -> RunSummary:
        summary.finish()
        if not dry_run:
            try:
                path = self.ledger.record(summary)
            except Exception as exc:  # observability must not fail a successful run
                logger.warning("could not persist run %s: %s", summary.run_id, exc)
            else:
                if path is not None:
                    logger.info("run %s recorded at %s", summary.run_id, path)
        return summary

    # --------------------------------------------------------------------- crawl
    def crawl(
        self,
        sources: Iterable[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> RunSummary:
        summary = self._summary("crawl", dry_run=dry_run)
        selected = self._selected_sources(sources)
        summary.metadata["sources"] = [str(getattr(source, "id", "")) for source in selected]
        crawler = Crawler(
            frontier=self.frontier,
            fetcher=self.fetcher,
            store=None if dry_run else self._store(),
            documents=self.documents,
            snapshots=self.snapshots,
            adapter_factory=self.adapter_factory,
            denylisted_domains=self.registry.sources.denied_domains(),
            max_source_failures=self.max_source_failures,
            planner=None if dry_run else self.planner,
            sources=selected,
        )
        crawler.plan(selected, summary=summary, limit=limit, dry_run=dry_run)
        if not dry_run:
            crawler.drain(summary=summary, limit=limit)

        if dry_run:
            summary.failed = bool(summary.failures) and summary.counts.get("planned", 0) == 0
        else:
            effective = sum(
                summary.counts.get(key, 0)
                for key in ("fetched", "stored", "deduped", "not_modified", "rejected")
            )
            # A single unimplemented or unreachable source is not a failed run,
            # but a run where nothing at all could be fetched is.
            summary.failed = bool(summary.failures) and effective == 0
        if not dry_run:
            summary.metadata["frontier"] = self._frontier_counts()
        return self._record(summary, dry_run=dry_run)

    def _frontier_counts(self) -> dict[str, int]:
        try:
            return dict(self.frontier.counts())
        except Exception:  # pragma: no cover - stats must never fail a crawl
            return {}

    # ------------------------------------------------------------------- extract
    def extract(self, limit: int | None = None, force: bool = False) -> RunSummary:
        summary = self._summary("extract")
        if self.router is None:
            raise OrchestrationError("extract requires a model router")
        extractor = Extractor(
            snapshots=self.snapshots,
            documents=self.documents,
            extractions=self.extractions,
            claims=self.claims,
            entities=self.entities,
            store=self._store(),
            router=self.router,
            tool_version=self.tool_version,
        )
        extractor.run(summary=summary, limit=limit, force=force)
        summary.metadata["tool_version"] = self.tool_version
        summary.metadata["force"] = force
        attempted = summary.counts.get("extracted", 0) + summary.counts.get("abstained", 0)
        summary.failed = (
            attempted > 0
            and summary.counts.get("extracted", 0) == 0
            and summary.counts.get("skipped", 0) == 0
        )
        return self._record(summary, dry_run=False)

    # --------------------------------------------------------------------- score
    def score(
        self,
        profile: dict[str, Any] | None = None,
        dry_run: bool = False,
        *,
        limit: int = 500,
        force: bool = False,
        persist: Any = None,
    ) -> RunSummary:
        summary = self._summary("score", dry_run=dry_run)
        outcome = score_programmes(
            registry=self.registry,
            profile=profile,
            programmes=self.programmes,
            claims=self.claims,
            scores=self.scores,
            limit=limit,
            dry_run=dry_run,
            force=force,
            persist=persist or self.score_persist,
        )
        summary.bump("programmes", outcome.programmes)
        summary.bump("written", outcome.written)
        summary.bump("skipped", outcome.skipped)
        summary.bump("unknown", outcome.unknown)
        for failure in outcome.failures:
            summary.fail("score", str(failure.get("reason")), **failure)
        summary.outcomes.extend({"line": line} for line in outcome.lines)
        summary.failed = outcome.programmes == 0 and bool(outcome.failures)
        return self._record(summary, dry_run=dry_run)

    # -------------------------------------------------------------------- worker
    def worker(
        self,
        *,
        limit: int | None = None,
        idle_seconds: float = 5.0,
        max_ticks: int | None = None,
        sleep: Callable[[float], None] | None = None,
        stop_event: threading.Event | None = None,
        on_summary: Callable[[RunSummary], None] | None = None,
    ) -> list[RunSummary]:
        stop = stop_event or threading.Event()
        wait = sleep if sleep is not None else stop.wait
        restore = _install_signal_handlers(stop)
        last_tick: list[RunSummary] = []
        ticks = 0

        def emit(summary: RunSummary) -> None:
            # Only the current tick is retained: a long-running worker must not
            # accumulate one summary per stage per tick.
            last_tick.append(summary)
            if on_summary is not None:
                on_summary(summary)

        try:
            while not stop.is_set():
                if max_ticks is not None and ticks >= max_ticks:
                    break
                ticks += 1
                last_tick = []
                crawl_summary = self.crawl(limit=limit)
                emit(crawl_summary)
                self._heartbeat(ticks, crawl_summary)
                if stop.is_set():
                    break
                extract_summary = self.extract(limit=limit)
                emit(extract_summary)
                self._heartbeat(ticks, extract_summary)
                if stop.is_set():
                    break
                score_summary = self.score(limit=limit or 500)
                emit(score_summary)
                self._heartbeat(ticks, score_summary)
                if max_ticks is None or ticks < max_ticks:
                    wait(idle_seconds)
        finally:
            restore()
            self._heartbeat(ticks, None, stopped=True)
        return last_tick

    def _heartbeat(self, tick: int, summary: RunSummary | None, *, stopped: bool = False) -> None:
        payload: dict[str, Any] = {
            "stage": "worker",
            "state": "stopped" if stopped else "running",
            "tick": tick,
            "process_id": _process_id(),
        }
        if summary is not None:
            payload.update(
                {
                    "run_id": summary.run_id,
                    "last_stage": summary.stage,
                    "counts": summary.counts,
                    "ok": not summary.failed,
                }
            )
        try:
            self.ledger.heartbeat(payload)
        except Exception as exc:  # heartbeat must never kill the worker
            logger.warning("heartbeat write failed: %s", exc)


def _process_id() -> int:
    import os

    return os.getpid()


def _install_signal_handlers(stop: threading.Event) -> Callable[[], None]:
    """Install SIGTERM/SIGINT handlers; return a restore callback.

    Only the main thread may install handlers; tests and embedded callers get a
    no-op restore.
    """

    def handler(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
        logger.info("received signal %s; stopping after the current stage", signum)
        stop.set()

    previous: dict[int, Any] = {}
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
    except (ValueError, OSError):  # pragma: no cover - non-main interpreter
        return lambda: None

    def restore() -> None:
        for signum, old in previous.items():
            signal.signal(signum, old)

    return restore
