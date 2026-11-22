"""Batched orchestrator (W10): stop server → acquire page-batch on CUDA →
start server → extract that batch from cache → repeat.

Why batching exists (measured, runbook §6): Docling converts 2–6× faster on
CUDA (0.166 vs 1.31 s/page) but OOMs while the 26B text model holds its 64k KV
cache (589 MiB free after the CUDA context; Docling needs ~1.0–1.3 GiB), so the
fused loop ran conversions on CPU while the GPU idled. With the persistent
parse cache the two phases separate cleanly and the one resident GPU resource
is time-shared instead:

    per cycle:  ensure server stopped
                → acquire one page-batch on CUDA (zero LLM calls)
                → ensure server started, /health ready (poll with timeout)
                → extract that batch from cache
                → repeat

Batch sizing is **by pages, evaluated only at document boundaries**. Completed
documents accumulate until the converted page total reaches ``batch_pages``;
the document that crosses the limit is always finished — converted in full and
extracted in the same cycle — and no further document starts. A document larger
than ``batch_pages`` runs alone. A document is never truncated, never dropped,
never split across cycles, and never cached or extracted while incomplete.

Server control is injected, not hardcoded: any object with ``stop()``,
``start()`` and ``wait_ready(timeout_s)`` works; the default adapter shells out
to ``scripts/serve_local.sh`` (``config/acquire.yaml:batch.server_script``).
Every exit path — conversion failure, LLM failure, SIGINT, a start failure —
leaves the server **stopped** (try/finally) unless ``keep_server`` is set.

Degraded mode: when server control is unavailable (``FMD_SERVER_CONTROL=off``
or no script), the run falls back to CPU acquisition + inline extraction and
says so in the summary (``server_control: degraded``) rather than failing.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from fmd.acquire.errors import AcquisitionFailed
from fmd.extract.pipeline import target_for_document
from fmd.orchestrate.acquire import DEVICE_ENV, AcquisitionStage
from fmd.orchestrate.extract import DEFAULT_TOOL_VERSION, Extractor
from fmd.orchestrate.ledger import RunSummary, utcnow

#: Forces degraded mode for one run (no server control, no GPU batch window).
SERVER_CONTROL_OFF = "off"
DEGRADED = "degraded"
SCRIPTED = "scripted"


class ServerControlError(RuntimeError):
    """A stop/start/readiness operation failed; the cycle aborts, server stopped."""


class ServerController(Protocol):
    """The seam the batched runner drives. Tests inject a fake.

    ``stop``/``start`` are "ensure" operations: calling them twice in a row is
    legal. ``wait_ready`` polls until the server answers ``/health`` or raises
    :class:`ServerControlError` after ``timeout_s``.
    """

    def stop(self) -> None: ...

    def start(self) -> None: ...

    def wait_ready(self, timeout_s: float) -> None: ...


def resolve_script(path: str | Path) -> Path:
    """Resolve the server script: absolute, cwd-relative, then repo-relative."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.is_file():
        return candidate
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        resolved = parent / candidate
        if resolved.is_file():
            return resolved
    return candidate


class ScriptServerController:
    """Default adapter: shells out to a start|stop|status|probe script and
    polls ``/health`` on the local text model's URL.

    The script contract is ``scripts/serve_local.sh``'s: ``stop`` SIGTERMs the
    process group, ``start`` backgrounds the server (no-op when already
    healthy), and readiness is the model server's own ``/health`` endpoint —
    derived from the registry's local provider base URL with any ``/v1``
    suffix stripped.
    """

    def __init__(
        self,
        script: str | Path,
        *,
        base_url: str | None = None,
        poll_interval_s: float = 2.0,
        timeout_s: float = 60.0,
    ) -> None:
        self.script = resolve_script(script)
        self.base_url = (base_url or "").rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ helpers
    def _run(self, action: str) -> None:
        if not self.script.is_file():
            raise ServerControlError(
                f"server script {str(self.script)!r} does not exist; set "
                "acquire.yaml:batch.server_script or run with FMD_SERVER_CONTROL=off "
                "for degraded CPU mode"
            )
        try:
            result = subprocess.run(
                [str(self.script), action],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ServerControlError(f"server script {action!r} timed out: {exc}") from exc
        if result.returncode != 0:
            raise ServerControlError(
                f"server script {action!r} exited {result.returncode}: "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )

    @property
    def health_url(self) -> str:
        root = self.base_url
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return f"{root}/health"

    # ------------------------------------------------------------------ control
    def stop(self) -> None:
        self._run("stop")

    def start(self) -> None:
        self._run("start")

    def wait_ready(self, timeout_s: float) -> None:
        if not self.base_url:
            raise ServerControlError(
                "no local model URL configured; cannot poll /health (set "
                "FMD_LOCAL_LLM_URL or providers.local.base_url)"
            )
        deadline = time.monotonic() + max(timeout_s, 0.0)
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                response = httpx.get(self.health_url, timeout=min(self.poll_interval_s, 10.0))
                if response.status_code == 200:
                    return
                last_error = f"/health answered {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(self.poll_interval_s)
        raise ServerControlError(
            f"server not ready after {timeout_s:g}s at {self.health_url}"
            + (f" (last: {last_error})" if last_error else "")
        )


class NullServerController:
    """Degraded-mode controller: server control is explicitly off."""

    def stop(self) -> None:
        return None

    def start(self) -> None:
        return None

    def wait_ready(self, timeout_s: float) -> None:
        return None


@dataclass(frozen=True)
class WorkItem:
    """One queued snapshot resolved against its document plan."""

    snapshot: Any
    document: Any
    url: str | None
    cache_hit: bool
    known_pages: int | None


def _install_sigint(stop: threading.Event):
    """SIGINT sets the stop flag; the runner checks it between documents.

    Only the main thread can install handlers; embedded callers and tests on
    worker threads get a no-op. The previous handler is always restored.
    """
    previous: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        stop.set()

    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    try:
        previous[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):  # pragma: no cover - non-main interpreter
        return lambda: None

    def restore() -> None:
        with contextlib.suppress(ValueError, OSError):  # interpreter teardown
            signal.signal(signal.SIGINT, previous[signal.SIGINT])

    return restore


def take_batch(
    work: list[WorkItem],
    *,
    batch_pages: int,
    pages_for: Any,
) -> list[WorkItem]:
    """A planned page-batch, sized only at document boundaries (dry runs).

    Same rule the live cycle applies, minus the pages only conversion can
    learn: a document is taken while the total is below ``batch_pages``; a
    document with a *known* page count (cached parse row) that would cross
    the limit is deferred, whole; an uncached document is taken (its pages
    are unknown until converted — in the live cycle it is converted in full
    and, if it crosses, ends the batch). The first document of an empty
    batch is always taken — that is how a 434-page document runs alone.
    """
    batch: list[WorkItem] = []
    total = 0
    for item in work:
        if total >= batch_pages:
            break
        known = item.known_pages
        if known is None and pages_for is not None:
            known = pages_for(item.snapshot.sha256)
        if known is not None and total > 0 and total + int(known) > batch_pages:
            break  # known-size document would cross: never started, deferred
        if known:
            total += int(known)
        batch.append(item)
    return batch


logger = logging.getLogger(__name__)


class PrefetchBuffer:
    """CPU-side double buffer (wave v1-w4): reads the *next* batch's snapshot
    bytes off the main thread while the current cycle's extraction phase keeps
    the GPU busy. The worker touches nothing but the object store — no
    converter, no model server, no network — so the one-CUDA-consumer rule
    (runbook §6) is untouched and every conversion still runs inside the
    server-stopped window.

    ``take`` is non-blocking: a byte set that has not landed yet simply falls
    back to a live read in the acquire loop. Correctness never depends on the
    prefetch winning the race; only the wall clock does.
    """

    def __init__(self, *, store: Any, depth: int = 4, stop: threading.Event | None = None) -> None:
        self._store = store
        self._depth = max(int(depth), 1)
        self._stop = stop if stop is not None else threading.Event()
        self._queue: queue.Queue[str] = queue.Queue()
        self._submitted: set[str] = set()
        self._bytes: dict[str, bytes] = {}
        self._thread: threading.Thread | None = None
        #: Observable read order (sha256 of every byte set landed off-thread).
        self.reads: list[str] = []

    def submit(self, sha256s: list[str]) -> None:
        """Queue byte reads for the given snapshots, skipping known ones."""
        fresh: list[str] = []
        for sha in sha256s[: self._depth]:
            if sha in self._submitted:
                continue
            self._submitted.add(sha)
            fresh.append(sha)
        if not fresh:
            return
        for sha in fresh:
            self._queue.put(sha)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="fmd-prefetch", daemon=True)
            self._thread.start()

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                sha = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if sha is None:
                return
            try:
                data = self._store.get(sha)
            except Exception as exc:
                # A prefetch miss is never a run failure — take() returns None
                # and the acquire loop reads live — but it must not be silent:
                # a persistently failing store otherwise degrades to 100% live
                # reads with zero observability.
                logger.warning("prefetch read failed for %s: %s", sha[:12], exc)
                continue
            self._bytes[sha] = data
            self.reads.append(sha)

    def take(self, sha256: str) -> bytes | None:
        """The prefetched bytes, or ``None`` when not ready (live read wins)."""
        return self._bytes.pop(sha256, None)

    def close(self) -> None:
        """Stop the worker and join it; an interrupted run never leaks a thread."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)


class BatchedRunner:
    """Drives the stop → acquire → start → extract cycles."""

    def __init__(
        self,
        *,
        orchestrator: Any,
        controller: ServerController | None = None,
        batch_pages: int = 100,
        device: str = "cuda",
        keep_server: bool = False,
        poll_timeout_s: float = 600,
        server_script: str | Path | None = None,
        tool_version: str = DEFAULT_TOOL_VERSION,
        line_sink: Any | None = None,
        t0: Any = None,
        t1: Any = None,
        prefetch_depth: int = 4,
    ) -> None:
        self.orchestrator = orchestrator
        #: Double buffering (wave v1-w4): while one cycle's extraction phase
        #: keeps the GPU busy, this many next-batch documents' bytes are read
        #: off-thread, so the next server-stopped window spends its seconds on
        #: conversion, not on disk I/O.
        self.prefetch_depth = max(int(prefetch_depth), 0)
        self._prefetch_buffer: PrefetchBuffer | None = None
        self.batch_pages = max(int(batch_pages), 1)
        self.device = device
        self.keep_server = bool(keep_server)
        self.poll_timeout_s = float(poll_timeout_s)
        self.tool_version = tool_version
        # Injected tier callables (test seams and the native text-only probe);
        # passed straight through to the acquisition stage and the extractor.
        self.t0 = t0
        self.t1 = t1
        self._sizing_policy_instance: Any = None
        self.line_sink = line_sink or _stderr_line
        self.controller = (
            controller if controller is not None else self._default_controller(server_script)
        )

    # ------------------------------------------------------------------ wiring
    def _default_controller(self, server_script: str | Path | None) -> ServerController | None:
        if os.environ.get("FMD_SERVER_CONTROL", "").strip().lower() == SERVER_CONTROL_OFF:
            return None
        script = server_script
        if not script:
            batch = getattr(getattr(self.orchestrator, "registry", None), "acquire", None)
            script = getattr(getattr(batch, "batch", None), "server_script", None)
        if not script:
            return None
        registry = getattr(self.orchestrator, "registry", None)
        base_url = _local_base_url(registry)
        return ScriptServerController(script, base_url=base_url)

    def _parse_cache(self) -> Any | None:
        getter = getattr(self.orchestrator, "_parse_cache", None)
        if getter is not None:
            return getter()
        return None

    def _t2_parse_cache(self) -> Any | None:
        getter = getattr(self.orchestrator, "_t2_parse_cache", None)
        if getter is not None:
            return getter()
        return None

    def _store_for_prefetch(self) -> Any:
        store = self.orchestrator.store
        if store is None:
            store = self.orchestrator._store()
        return store

    def _tier_caches(self) -> list[Any]:
        """The tier caches in lookup order (T1 first, then T2).

        Sizing consults every tier cache a document's parse may live in: since
        wave v1-w3 a scanned document's parse is a T2 row, and a sizer that
        reads T1 only would treat it as unknown-pages forever (converted in
        full every cycle it appears, never deferred whole).
        """
        caches = [self._parse_cache(), self._t2_parse_cache()]
        return [cache for cache in caches if cache is not None]

    def _has_parse(self, snapshot_sha256: str) -> bool:
        policy = self._sizing_policy()
        for cache in self._tier_caches():
            try:
                if cache.has_parse(snapshot_sha256, policy):
                    return True
            except Exception:  # a sizing peek must never fail a cycle
                continue
        return False

    def _pages_for(self, snapshot_sha256: str) -> int | None:
        """The cached page count, first hit across the tier caches (T1→T2)."""
        policy = self._sizing_policy()
        for cache in self._tier_caches():
            try:
                if cache.has_parse(snapshot_sha256, policy):
                    return cache.pages_for(snapshot_sha256, policy)
            except Exception:
                continue
        return None

    def _extractor(self, *, allow_convert: bool) -> Extractor:
        orchestrator = self.orchestrator
        return Extractor(
            snapshots=orchestrator.snapshots,
            documents=orchestrator.documents,
            extractions=orchestrator.extractions,
            claims=orchestrator.claims,
            entities=orchestrator.entities,
            store=orchestrator.store if orchestrator.store is not None else orchestrator._store(),
            router=orchestrator.router,
            tool_version=self.tool_version,
            predicates=getattr(orchestrator.registry, "predicates", None),
            sections=orchestrator.sections,
            parse_cache=self._parse_cache(),
            t2_parse_cache=self._t2_parse_cache(),
            t0=self.t0,
            t1=self.t1,
            allow_convert=allow_convert,
        )

    def _acquisition_stage(self) -> AcquisitionStage:
        orchestrator = self.orchestrator
        from fmd.acquire.tiers import AcquirePolicy

        policy = AcquirePolicy.load(acquire=orchestrator.registry.acquire)
        return AcquisitionStage(
            snapshots=orchestrator.snapshots,
            documents=orchestrator.documents,
            store=orchestrator.store if orchestrator.store is not None else orchestrator._store(),
            parse_cache=self._parse_cache(),
            t2_parse_cache=self._t2_parse_cache(),
            policy=policy,
            tool_version=self.tool_version,
            device=self.device,
            t0=self.t0,
            t1=self.t1,
        )

    # ------------------------------------------------------------------ public
    def run(
        self,
        *,
        limit: int | None = None,
        dry_run: bool = False,
        stop_event: threading.Event | None = None,
    ) -> RunSummary:
        orchestrator = self.orchestrator
        summary = RunSummary(
            run_id=orchestrator.run_id_factory(),
            stage="extract_batched",
            started_at=utcnow(),
            dry_run=dry_run,
        )
        if os.environ.get("FMD_SERVER_CONTROL", "").strip().lower() == SERVER_CONTROL_OFF:
            # The operator's switch wins, even over an injected controller.
            return self._run_degraded(summary, limit=limit, dry_run=dry_run)
        if self.controller is None:
            return self._run_degraded(summary, limit=limit, dry_run=dry_run)
        work = self._resolve_queue(limit=limit, summary=summary)
        if dry_run:
            return self._run_dry(summary, work)
        return self._run_cycles(summary, work, stop_event=stop_event)

    # ------------------------------------------------------------------ degraded
    def _run_degraded(self, summary: RunSummary, *, limit: int | None, dry_run: bool) -> RunSummary:
        """No server control ⇒ CPU acquisition + inline extraction, said out loud.

        The fallback is the fused path that existed before this patch — one
        pass, conversions inline on CPU (``FMD_DOCLING_DEVICE=cpu``), extraction
        immediately after each conversion. Nothing about the evidence changes;
        the summary records the degraded control mode so the operator knows
        this run gained nothing from the GPU.
        """
        summary.metadata["server_control"] = DEGRADED
        if dry_run:
            work = self._resolve_queue(limit=limit, summary=summary)
            return self._run_dry(summary, work, degraded=True)
        restore_device = _pin_device_env("cpu")
        try:
            inner = self.orchestrator.extract(limit=limit, record=False)
        finally:
            restore_device()
        _merge(summary, inner)
        summary.failed = summary.failed or inner.failed
        return self._finish(summary, dry_run=False)

    # ------------------------------------------------------------------ dry run
    def _resolve_queue(self, *, limit: int | None, summary: RunSummary) -> list[WorkItem]:
        orchestrator = self.orchestrator
        pending = orchestrator.snapshots.list_extractable(
            self.tool_version, limit if limit is not None else 500
        )
        cache = self._parse_cache()
        work: list[WorkItem] = []
        for snapshot in pending:
            document = orchestrator.documents.get(snapshot.document_id)
            target, reason = (
                (None, "missing_document")
                if document is None
                else target_for_document(document, snapshot.id)
            )
            if target is None:
                summary.bump("abstained")
                summary.outcomes.append(
                    {
                        "snapshot_id": snapshot.id,
                        "document_id": snapshot.document_id,
                        "url": getattr(document, "url", None),
                        "outcome": "abstained",
                        "reason": reason,
                    }
                )
                continue
            cache_hit = False
            known_pages: int | None = None
            if cache is not None:
                try:
                    cache_hit = self._has_parse(snapshot.sha256)
                    if cache_hit:
                        known_pages = self._pages_for(snapshot.sha256)
                except Exception:  # plan only: cache trouble is not a plan failure
                    cache_hit = False
            work.append(
                WorkItem(
                    snapshot=snapshot,
                    document=document,
                    url=getattr(document, "url", None),
                    cache_hit=cache_hit,
                    known_pages=known_pages,
                )
            )
        return work

    def _run_dry(
        self, summary: RunSummary, work: list[WorkItem], *, degraded: bool = False
    ) -> RunSummary:
        """The cycle plan: documents, known pages, expected cache hits.

        Nothing is written and no model URL is contacted: dry-run reads the
        queue and the cache table only. Pages of uncached documents are
        reported as unknown (they become known exactly when conversion does).
        """
        if degraded:
            summary.metadata["server_control"] = DEGRADED
        cycles: list[dict[str, Any]] = []
        remaining = list(work)
        cycle_index = 0
        while remaining:
            cycle_index += 1
            batch = take_batch(remaining, batch_pages=self.batch_pages, pages_for=None)
            if not batch:
                break
            remaining = remaining[len(batch) :]
            known = sum(int(item.known_pages or 0) for item in batch)
            cycles.append(
                {
                    "cycle": cycle_index,
                    "documents": [
                        {
                            "snapshot_id": item.snapshot.id,
                            "document_id": item.snapshot.document_id,
                            "url": item.url,
                            "cache_hit": item.cache_hit,
                            "known_pages": item.known_pages,
                        }
                        for item in batch
                    ],
                    "known_pages": known,
                    "unknown_pages_documents": sum(1 for item in batch if not item.known_pages),
                    "expected_cache_hits": sum(1 for item in batch if item.cache_hit),
                    "expected_conversions": sum(1 for item in batch if not item.cache_hit),
                }
            )
            summary.bump("planned", len(batch))
            summary.bump("cache_hits_expected", sum(1 for item in batch if item.cache_hit))
            summary.bump("to_convert", sum(1 for item in batch if not item.cache_hit))
        summary.bump("cycles", len(cycles))
        summary.metadata["cycles"] = cycles
        summary.metadata["batch_pages"] = self.batch_pages
        summary.metadata["server_control"] = summary.metadata.get("server_control") or SCRIPTED
        return self._finish(summary, dry_run=True)

    # ------------------------------------------------------------------ cycles
    def _run_cycles(
        self,
        summary: RunSummary,
        work: list[WorkItem],
        *,
        stop_event: threading.Event | None = None,
    ) -> RunSummary:
        # SIGINT (or an injected stop event, e.g. a worker shutting down) ends
        # the run between documents and cycles; the finally below still stops
        # the server, so an interrupt never leaves one running.
        stop = stop_event if stop_event is not None else threading.Event()
        restore_signal = _install_sigint(stop)
        controller = self.controller
        assert controller is not None
        stage = self._acquisition_stage()
        self._prefetch_buffer = (
            PrefetchBuffer(store=self._store_for_prefetch(), depth=self.prefetch_depth, stop=stop)
            if self.prefetch_depth > 0
            else None
        )
        cycle_index = 0
        try:
            remaining = list(work)
            while remaining and not stop.is_set():
                cycle_index += 1
                cycle_summary = self._run_cycle(
                    cycle_index, remaining, stage, stop, summary=summary
                )
                _merge(summary, cycle_summary)
                if cycle_summary.metadata.get("abort"):
                    summary.fail(
                        "extract_batched",
                        str(cycle_summary.metadata.get("abort_reason") or "cycle aborted"),
                        cycle=cycle_index,
                    )
                    break
        finally:
            restore_signal()
            if self._prefetch_buffer is not None:
                self._prefetch_buffer.close()
                self._prefetch_buffer = None
            if not self.keep_server:
                try:
                    controller.stop()
                except Exception as exc:
                    summary.fail("server_control", f"final stop failed: {exc}")
        summary.bump("cycles", cycle_index)
        summary.metadata["batch_pages"] = self.batch_pages
        summary.metadata["device"] = self.device
        summary.metadata["keep_server"] = self.keep_server
        summary.metadata["server_control"] = SCRIPTED
        attempted = summary.counts.get("extracted", 0) + summary.counts.get("abstained", 0)
        summary.failed = summary.failed or (
            attempted > 0
            and summary.counts.get("extracted", 0) == 0
            and summary.counts.get("skipped", 0) == 0
        )
        return self._finish(summary, dry_run=False)

    def _sizing_policy(self) -> Any:
        """The acquire policy the cache keys are derived from (lazy, cached)."""
        if self._sizing_policy_instance is None:
            from fmd.acquire.tiers import AcquirePolicy

            self._sizing_policy_instance = AcquirePolicy.load(
                acquire=self.orchestrator.registry.acquire
            )
        return self._sizing_policy_instance

    def _submit_prefetch(self, remaining: list[WorkItem], buffer: PrefetchBuffer) -> None:
        """Candidates for the next cycle's acquire phase, read off-thread.

        Only *uncached* documents need bytes: a cached parse goes straight to
        the extract phase without touching the store. Batch sizing follows the
        same boundary rule the live cycle applies, so the prefetch covers what
        the next window will plausibly convert (a bounded guess; a miss only
        costs a live read later).
        """
        candidates = take_batch(remaining, batch_pages=self.batch_pages, pages_for=self._pages_for)
        shas = [
            item.snapshot.sha256
            for item in candidates
            if self._pages_for(item.snapshot.sha256) is None
        ]
        buffer.submit(shas[: self.prefetch_depth])

    def _run_cycle(
        self,
        cycle_index: int,
        remaining: list[WorkItem],
        stage: AcquisitionStage,
        stop: threading.Event,
        *,
        summary: RunSummary,
    ) -> RunSummary:
        """One cycle: stop → acquire a page-batch → start → extract from cache.

        The batch grows **during the acquisition phase**, one document at a
        time: a document is taken only while the converted page total is below
        the limit; its pages join the total exactly when its conversion (or
        cache hit) completes. A document with a *known* page count (cached
        row) that would cross the limit is never started — it stays at the
        head of the queue for the next cycle. An uncached document's size is
        unknowable before conversion, so it is taken, converted **in full**
        (never truncated), and if it crosses the limit it is the last document
        of the cycle. No document is ever split, and extraction covers exactly
        the documents that acquired a parse this cycle.
        """
        controller = self.controller
        assert controller is not None
        cycle = RunSummary(
            run_id=self.orchestrator.run_id_factory(),
            stage=f"extract_batched:{cycle_index}",
            started_at=utcnow(),
        )
        started = time.perf_counter()
        abort: str | None = None
        acquired: list[WorkItem] = []
        total_pages = 0
        line_pages = 0
        try:
            # 1) ensure stopped — Docling needs the VRAM the 26B holds at 64k.
            controller.stop()
            # 2) acquire the batch on the configured device (zero LLM calls),
            #    taking documents one at a time until the page total is reached.
            while remaining and not stop.is_set():
                item = remaining[0]
                known = self._pages_for(item.snapshot.sha256)
                if total_pages >= self.batch_pages:
                    break
                if known is not None and total_pages > 0 and total_pages + known > self.batch_pages:
                    # A known-size document that would cross the limit is never
                    # started: it is deferred to the next cycle, whole.
                    break
                if known is not None:
                    # The parse already exists (converted on an earlier cycle
                    # or run): nothing to acquire, the document goes straight
                    # to the extract phase — no second cache hit counted.
                    remaining.pop(0)
                    total_pages += int(known)
                    line_pages += int(known)
                    acquired.append(item)
                    continue
                buffer = self._prefetch_buffer
                data_for = (lambda sha, b=buffer: b.take(sha)) if buffer is not None else None
                stage.run(
                    summary=cycle,
                    selected=[item.snapshot],
                    dry_run=False,
                    data_for=data_for,
                )
                pages_after = self._pages_for(item.snapshot.sha256)
                remaining.pop(0)
                if pages_after is None:
                    # Conversion failed for this document: nothing cached, the
                    # batch continues with the next one. The stage recorded the
                    # reason; extraction simply does not see this document.
                    continue
                total_pages += int(pages_after)
                line_pages += int(pages_after)
                acquired.append(item)
            # 3) ensure started and /health ready — skipped when an interrupt
            #    already ended the run: never start a server nobody will use.
            if not stop.is_set():
                controller.start()
                # 3b) double buffering: the GPU window is now open (the server
                #     owns the card and the poll loop leaves the CPU idle) —
                #     read the *next* batch's bytes off-thread so the next
                #     server-stopped window spends its time converting, not
                #     on disk I/O. The worker touches the object store only.
                buffer = self._prefetch_buffer
                if buffer is not None:
                    self._submit_prefetch(remaining, buffer)
                controller.wait_ready(self.poll_timeout_s)
            # 4) extract exactly the documents that acquired a parse this cycle
            #    (allow_convert=True: a payload that failed verification falls
            #    back to inline CPU conversion rather than losing the doc).
            if acquired and not stop.is_set():
                extractor = self._extractor(allow_convert=True)
                selected = [item.snapshot for item in acquired]
                extractor.run(summary=cycle, selected=selected)
        except ServerControlError as exc:
            abort = f"server_control_failed: {exc}"
            cycle.fail("server_control", str(exc), cycle=cycle_index)
            cycle.metadata["abort"] = True
            cycle.metadata["abort_reason"] = abort
        except AcquisitionFailed as exc:
            abort = f"acquisition_failed: {getattr(exc, 'reason', None) or exc}"
            cycle.fail("acquire", abort, cycle=cycle_index)
            cycle.metadata["abort"] = True
            cycle.metadata["abort_reason"] = abort
        finally:
            # Every exit path stops the server unless explicitly kept: a start
            # failure, a conversion abort, an LLM failure, Ctrl-C mid-cycle.
            if not self.keep_server:
                try:
                    controller.stop()
                except Exception as exc:
                    cycle.fail("server_control", f"stop after cycle failed: {exc}")
        seconds = time.perf_counter() - started
        self._emit_line(
            cycle_index,
            documents=len(acquired),
            pages=line_pages,
            cache_hits=cycle.counts.get("cache_hits", 0),
            seconds=seconds,
            device=self.device,
        )
        cycle.metadata["abort"] = cycle.metadata.get("abort", False)
        if stop.is_set():
            cycle.metadata["interrupted"] = True
        summary.bump("llm_calls", cycle.counts.get("llm_calls", 0))
        return cycle

    def _emit_line(
        self,
        cycle: int,
        *,
        documents: int,
        pages: int,
        cache_hits: int,
        seconds: float,
        device: str,
    ) -> None:
        per_min = (pages / (seconds / 60.0)) if seconds > 0 and pages else 0.0
        self.line_sink(
            f"[batch] cycle={cycle} documents={documents} pages={pages} "
            f"cache_hits={cache_hits} seconds={seconds:.1f} "
            f"pages_per_min={per_min:.1f} device={device}"
        )

    def _finish(self, summary: RunSummary, *, dry_run: bool) -> RunSummary:
        summary.finish()
        recorder = getattr(self.orchestrator, "_record", None)
        if recorder is not None:
            return recorder(summary, dry_run=dry_run)
        return summary


def _merge(master: RunSummary, cycle: RunSummary) -> None:
    for key, value in cycle.counts.items():
        if key == "llm_calls":
            continue  # already forwarded by the cycle
        master.counts[key] = master.counts.get(key, 0) + value
    master.outcomes.extend(cycle.outcomes)
    master.failures.extend(cycle.failures)
    if cycle.failed:
        master.failed = True


def _stderr_line(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _pin_device_env(device: str):
    """Pin ``FMD_DOCLING_DEVICE`` for a degraded run; restore after."""
    previous = os.environ.get(DEVICE_ENV)

    def restore() -> None:
        if previous is None:
            os.environ.pop(DEVICE_ENV, None)
        else:
            os.environ[DEVICE_ENV] = previous

    os.environ[DEVICE_ENV] = device
    return restore


def _local_base_url(registry: Any) -> str | None:
    """The local text model's base URL from the registry (env-interpolated)."""
    models = getattr(registry, "models", None)
    providers = getattr(models, "providers", None) or {}
    local = providers.get("local")
    url = getattr(local, "base_url", None)
    return str(url) if url else None


__all__ = [
    "BatchedRunner",
    "DEGRADED",
    "NullServerController",
    "SCRIPTED",
    "SERVER_CONTROL_OFF",
    "ScriptServerController",
    "ServerControlError",
    "ServerController",
    "WorkItem",
    "resolve_script",
    "take_batch",
]
