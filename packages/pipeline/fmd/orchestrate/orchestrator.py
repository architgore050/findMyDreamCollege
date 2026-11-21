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
    InstitutionDomainsRepo,
    ProgrammesRepo,
    ScoresRepo,
    SectionsRepo,
    SnapshotsRepo,
)
from fmd.fetch.ladder import FetchLadder
from fmd.frontier.queue import Frontier
from fmd.orchestrate.acquire import AcquisitionStage
from fmd.orchestrate.adapters import build_adapter
from fmd.orchestrate.crawl import DEFAULT_SOURCE_FAILURE_BUDGET, Crawler
from fmd.orchestrate.extract import DEFAULT_TOOL_VERSION, Extractor, build_extract_router
from fmd.orchestrate.ingest import INSTITUTION_ADAPTER, InventoryStage
from fmd.orchestrate.ledger import JsonFileLedger, RunLedger, RunSummary, SqlLedger, utcnow
from fmd.orchestrate.plan import PlanStage
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
        aliases: Any | None = None,
        institution_domains: Any | None = None,
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
        sections: Any | None = None,
        parse_cache: Any | None = None,
        t0: Any | None = None,
        t1: Any | None = None,
    ) -> None:
        self.registry = registry
        self.database = database
        self.score_persist = score_persist
        self._sources = list(sources) if sources is not None else None
        self.run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)
        self.tool_version = tool_version
        self.max_source_failures = max_source_failures
        self._parse_cache_instance = parse_cache
        # Injected tier callables (test seams), handed to the extract stage.
        self.t0 = t0
        self.t1 = t1

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
        self.sections = sections if sections is not None else self._optional(SectionsRepo)
        self.programmes = (
            programmes if programmes is not None else self._require_database(ProgrammesRepo)
        )
        self.scores = scores if scores is not None else self._require_database(ScoresRepo)
        self.entities = entities if entities is not None else self._optional(EntitiesRepo)
        self.aliases = aliases if aliases is not None else self._optional(AliasesRepo)
        self.institution_domains = (
            institution_domains
            if institution_domains is not None
            else self._optional(InstitutionDomainsRepo)
        )
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
                aliases=self.aliases,
                er_config=registry.er,
                plan_config=registry.plan,
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

    def _parse_cache(self) -> Any | None:
        """The persistent T1 parse cache (W10), or ``None`` without a store.

        Shares the snapshot object store: parse payloads are content-addressed
        objects beside the evidence bytes, looked up through the
        ``document_parses`` table (migration 0011). Explicitly injected caches
        win; a database-less orchestrator runs the legacy fused path.
        """
        if self._parse_cache_instance is not None:
            return self._parse_cache_instance
        if self.database is None:
            return None
        from fmd.acquire.parse_cache import ParseCache
        from fmd.db.repositories import DocumentParsesRepo

        self._parse_cache_instance = ParseCache(
            DocumentParsesRepo(self.database), self._store()
        )
        return self._parse_cache_instance

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
        queued = {} if dry_run else self._queued_domains(selected)
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
        crawler.plan(
            selected,
            summary=summary,
            limit=limit,
            dry_run=dry_run,
            domains={
                source_id: [row.seed_url for row in rows] for source_id, rows in queued.items()
            },
            entity_hints={
                row.seed_url: row.entity_id
                for rows in queued.values()
                for row in rows
                if row.entity_id is not None
            },
        )
        if not dry_run:
            crawler.drain(summary=summary, limit=limit)
            self._mark_domains_planned(queued)

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
            summary.metadata["roles"] = self._document_roles()
        return self._record(summary, dry_run=dry_run)

    def _queued_domains(self, selected: Iterable[Any]) -> dict[str, list[Any]]:
        """Domains selected by ``ingest`` mapped to their institution-site source."""
        if self.institution_domains is None:
            return {}
        source_ids = [
            str(getattr(source, "id", ""))
            for source in selected
            if getattr(source, "adapter", None) == INSTITUTION_ADAPTER
        ]
        if not source_ids:
            return {}
        try:
            rows = self.institution_domains.list_by_state("queued", limit=500)
        except Exception as exc:  # observability must not abort a crawl
            logger.warning("could not read queued institution domains: %s", exc)
            return {}
        return {source_id: list(rows) for source_id in source_ids} if rows else {}

    def _mark_domains_planned(self, queued: dict[str, list[Any]]) -> None:
        domains = [row.domain for rows in queued.values() for row in rows]
        if not domains or self.institution_domains is None:
            return
        try:
            self.institution_domains.mark(domains, "planned", reason="adapter_planned")
        except Exception as exc:  # pragma: no cover - queue bookkeeping only
            logger.warning("could not mark institution domains planned: %s", exc)

    def _document_roles(self) -> dict[str, int]:
        counter = getattr(self.documents, "counts_by_role", None)
        if counter is None:
            return {}
        try:
            return dict(counter())
        except Exception:  # pragma: no cover - stats must never fail a crawl
            return {}

    # --------------------------------------------------------------------- ingest
    def ingest(self, limit: int | None = None) -> RunSummary:
        """Inventory documents -> entities -> prioritised institution domains."""
        summary = self._summary("ingest")
        if self.entities is None or self.institution_domains is None:
            summary.bump("skipped")
            summary.metadata["reason"] = "ingest requires a database or injected repositories"
            return self._record(summary, dry_run=False)
        stage = InventoryStage(
            documents=self.documents,
            snapshots=self.snapshots,
            store=self._store(),
            entities=self.entities,
            aliases=self.aliases,
            domains=self.institution_domains,
            sources=self._all_sources(),
            registry=self.registry,
        )
        stage.run(summary=summary, limit=limit)
        summary.failed = bool(summary.failures)
        return self._record(summary, dry_run=False)

    def _frontier_counts(self) -> dict[str, int]:
        try:
            return dict(self.frontier.counts())
        except Exception:  # pragma: no cover - stats must never fail a crawl
            return {}

    # ------------------------------------------------------------------- extract
    def extract(
        self,
        limit: int | None = None,
        force: bool = False,
        *,
        from_cache: bool = False,
        no_acquire: bool = False,
        dry_run: bool = False,
        record: bool = True,
    ) -> RunSummary:
        summary = self._summary("extract", dry_run=dry_run)
        if self.router is None:
            raise OrchestrationError("extract requires a model router")
        parse_cache = self._parse_cache()
        summary.metadata["parse_cache"] = parse_cache is not None
        if dry_run:
            # The plan: documents, known pages, expected cache hits. Reads the
            # queue and the cache table; writes nothing and never touches the
            # model URL (which may be unreachable by design in a dry run).
            stage = AcquisitionStage(
                snapshots=self.snapshots,
                documents=self.documents,
                store=self._store(),
                parse_cache=parse_cache,
                tool_version=self.tool_version,
            )
            items = stage.plan(limit=limit)
            summary.bump("selected", len(items))
            for item in items:
                if item.cache_hit:
                    summary.bump("cache_hits_expected")
                    summary.bump("parsed")
                else:
                    summary.bump("to_convert")
                summary.outcomes.append(
                    {
                        **item.as_dict(),
                        "outcome": "cache_hit" if item.cache_hit else "would_convert",
                    }
                )
            summary.metadata["mode"] = "dry_run"
            return self._record(summary, dry_run=True) if record else summary.finish()
        # Evidence documents that Phase A deferred must reach text acquisition
        # before the extract queue is read. This is the only place the sample is
        # bought; the stage's cache makes a re-run idempotent.
        stage = PlanStage(
            planner=self.planner,
            documents=self.documents,
            snapshots=self.snapshots,
            store=self._store(),
            sources=self._all_sources(),
            text_sample_chars=self.registry.plan.phase_b.text_sample_chars,
        )
        stage.run(summary=summary, limit=limit)
        allow_convert = not (from_cache or no_acquire)
        extractor = Extractor(
            snapshots=self.snapshots,
            documents=self.documents,
            extractions=self.extractions,
            claims=self.claims,
            entities=self.entities,
            store=self._store(),
            router=self.router,
            tool_version=self.tool_version,
            predicates=self.registry.predicates,
            sections=self.sections,
            parse_cache=parse_cache,
            t0=self.t0,
            t1=self.t1,
            allow_convert=allow_convert,
        )
        extractor.run(summary=summary, limit=limit, force=force)
        summary.metadata["tool_version"] = self.tool_version
        summary.metadata["force"] = force
        summary.metadata["mode"] = (
            "from_cache" if from_cache else ("no_acquire" if no_acquire else "live")
        )
        attempted = summary.counts.get("extracted", 0) + summary.counts.get("abstained", 0)
        # Preserve failures already recorded per document (e.g. an acquisition
        # OOM): a run that also extracted something is still a failed run.
        summary.failed = summary.failed or (
            attempted > 0
            and summary.counts.get("extracted", 0) == 0
            and summary.counts.get("skipped", 0) == 0
        )
        return self._record(summary, dry_run=False) if record else summary.finish()

    # ------------------------------------------------------------------- acquire
    def acquire(
        self,
        *,
        limit: int | None = None,
        pages: int | None = None,
        device: str | None = None,
        dry_run: bool = False,
    ) -> RunSummary:
        """The acquisition stage (W10): convert and cache, zero LLM calls.

        Selection is the extractor's own queue in the extractor's own order;
        the orchestrator's router is deliberately never handed to the stage.
        ``pages`` is a page budget evaluated only at document boundaries — the
        document that crosses it finishes in full, then acquisition stops.
        """
        summary = self._summary("acquire", dry_run=dry_run)
        from fmd.acquire.tiers import AcquirePolicy
        from fmd.orchestrate.acquire import AcquisitionStage

        stage = AcquisitionStage(
            snapshots=self.snapshots,
            documents=self.documents,
            store=self._store(),
            parse_cache=self._parse_cache(),
            policy=AcquirePolicy.load(acquire=self.registry.acquire),
            tool_version=self.tool_version,
            device=device,
        )
        stage.run(summary=summary, limit=limit, pages_budget=pages, dry_run=dry_run)
        summary.metadata["zero_llm_calls"] = True
        summary.failed = summary.failed or (
            summary.counts.get("selected", 0) > 0
            and summary.counts.get("parsed", 0) == 0
        )
        return self._record(summary, dry_run=dry_run)

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
        summary.bump("out_of_scope", outcome.out_of_scope)
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
                # Inventory -> entities -> queued institution domains. The next
                # tick's crawl consumes the queue; running it here keeps the
                # pipeline's missing link inside one worker loop.
                ingest_summary = self.ingest(limit=limit)
                emit(ingest_summary)
                self._heartbeat(ticks, ingest_summary)
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
