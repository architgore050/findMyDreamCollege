"""Compute metrics (FMD-CLOUD-030): logging that answers "what is the bottleneck?".

The 50-college E2E run spent ~2 h on CPU Docling acquisition while a 26B server
sat on the GPU, and its logs only said "still running". This module is the
measurement half of the fix:

* :class:`ComputeMetrics` — append-only JSONL records under
  ``data/compute-metrics/`` (gitignored except a README), one file per
  ``run_id``, every record carrying stage/phase/device/duration/tokens/throughput
  plus the sampler's GPU + process aggregates.
* :class:`ResourceSampler` — a daemon thread sampling ``nvidia-smi``
  (utilization.gpu, memory.used, power.draw) and ``ps`` (cpu%, rss) every
  ``interval_s`` around a phase. Absent tools degrade to ``None`` fields,
  never a crash.
* :func:`metrics_scope` / :func:`emit` — a process-wide active sink so deep
  layers (router, acquisition, embeddings) can emit without every constructor
  growing a parameter.

No user data in metrics: ``input_ref`` is the pipeline's own content-hash
reference (``sha256:family:prompt-version``), never a URL or prompt text.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Fields every record carries (sorted key order keeps the JSONL diffable).
RECORD_FIELDS = (
    "batch_size",
    "cache_hit",
    "cache_hits",
    "cpu_percent",
    "decode_tps",
    "device",
    "documents",
    "duration_ms",
    "gpu_samples",
    "gpu_util_pct",
    "input_ref",
    "model",
    "ocr_device",
    "ocr_engine",
    "pages",
    "phase",
    "power_draw_w",
    "prefill_tps",
    "provider",
    "reason",
    "rss_mib",
    "run_id",
    "stage",
    "status",
    "tokens_in",
    "tokens_out",
    "total_tps",
    "vram_peak_mib",
    "vram_used_mib",
)


def default_metrics_root() -> Path:
    """``data/compute-metrics`` of the surrounding repository checkout.

    Resolution: ``FMD_METRICS_DIR`` → nearest ``data/`` directory walking up
    from this file (the repo layout) → ``data/compute-metrics`` relative to the
    working directory (container). Created on first write, never before.
    """
    env = os.environ.get("FMD_METRICS_DIR")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "data").is_dir():
            return parent / "data" / "compute-metrics"
    return Path("data") / "compute-metrics"


def metrics_disabled_by_env() -> bool:
    """``FMD_COMPUTE_METRICS=off`` disables recording for one process."""
    return os.environ.get("FMD_COMPUTE_METRICS", "").strip().lower() in {"off", "0", "false"}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of ``values`` for ``q`` in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, min(len(ordered), round(q / 100.0 * len(ordered))))
    return ordered[rank - 1]


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# ------------------------------------------------------------------ nvidia-smi
# One availability probe per process: an absent binary must not be re-spawned
# every sample. Tests inject their own probes instead of relying on this cache.

_NVIDIA_SMI: str | None = None
_NVIDIA_PROBED = False
_NVIDIA_LOCK = threading.Lock()


def nvidia_smi_path() -> str | None:
    """The nvidia-smi binary path, probed once per process (None when absent)."""
    global _NVIDIA_SMI, _NVIDIA_PROBED
    with _NVIDIA_LOCK:
        if not _NVIDIA_PROBED:
            _NVIDIA_SMI = shutil.which("nvidia-smi")
            _NVIDIA_PROBED = True
        return _NVIDIA_SMI


def _smi_query(fields: list[str]) -> list[list[str]] | None:
    """One ``nvidia-smi --query-gpu`` invocation as CSV rows (or None)."""
    binary = nvidia_smi_path()
    if not binary:
        return None
    try:
        result = subprocess.run(
            [
                binary,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    rows: list[list[str]] = []
    for line in result.stdout.strip().splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == len(fields):
            rows.append(cells)
    return rows or None


def _cell_float(cell: str) -> float | None:
    """A CSV cell as a float; ``N/A``/``[Not Supported]`` become None."""
    text = cell.strip()
    if not text or text.startswith("[") or "N/A" in text or "not supported" in text.lower():
        return None
    try:
        return float(text)
    except ValueError:
        return None


def query_gpu_telemetry() -> dict[str, float] | None:
    """utilization.gpu (%), memory.used (MiB), power.draw (W) of GPU 0.

    ``None`` when nvidia-smi is absent or fails — a missing tool is a ``null``
    metric, never a crash.
    """
    rows = _smi_query(["utilization.gpu", "memory.used", "power.draw"])
    if not rows:
        return None
    util, mem, power = (_cell_float(cell) for cell in rows[0])
    if util is None and mem is None and power is None:
        return None
    return {"gpu_util_pct": util, "vram_used_mib": mem, "power_draw_w": power}


def query_gpu_memory() -> dict[str, float] | None:
    """memory.used / memory.free / memory.total (MiB) of GPU 0, or None."""
    rows = _smi_query(["memory.used", "memory.free", "memory.total"])
    if not rows:
        return None
    used, free, total = (_cell_float(cell) for cell in rows[0])
    if used is None and free is None and total is None:
        return None
    return {"vram_used_mib": used, "vram_free_mib": free, "vram_total_mib": total}


def query_process_usage(pid: int | None = None) -> dict[str, float] | None:
    """``ps`` cpu%% + rss (MiB) for one pid, or None when ps is unavailable."""
    binary = shutil.which("ps")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "-o", "%cpu=,rss=", "-p", str(pid if pid is not None else os.getpid())],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    cells = result.stdout.strip().split()
    if len(cells) < 2:
        return None
    cpu, rss_kb = _cell_float(cells[0]), _cell_float(cells[1])
    if cpu is None or rss_kb is None:
        return None
    return {"cpu_percent": cpu, "rss_mib": rss_kb / 1024.0}


class _NullSampler:
    """Stand-in when sampling is off or the factory fails; all fields None."""

    def __enter__(self) -> _NullSampler:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {}


class ResourceSampler:
    """Sample GPU + process usage every ``interval_s`` around one phase.

    The thread touches nothing but the probe callables; a probe failure is
    counted and skipped (the phase still completes, its metrics say ``null``).
    Long phases pay one subprocess call per interval — cheap next to a Docling
    conversion or an LLM decode.
    """

    def __init__(
        self,
        *,
        interval_s: float = 5.0,
        pid: int | None = None,
        probe_gpu: Any | None = None,
        probe_proc: Any | None = None,
    ) -> None:
        self.interval_s = max(float(interval_s), 0.2)
        self.pid = pid if pid is not None else os.getpid()
        self._probe_gpu = probe_gpu or query_gpu_telemetry
        self._probe_proc = probe_proc or query_process_usage
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu: list[dict[str, float]] = []
        self._proc: list[dict[str, float]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._work, name="fmd-metrics-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.interval_s * 2 + 1.0)
        self._thread = None

    def __enter__(self) -> ResourceSampler:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ----------------------------------------------------------------- sampling
    def _work(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)
        # One final sample so a phase shorter than the interval still records.
        self._sample_once()

    def _sample_once(self) -> None:
        with self._lock:
            gpu = self._safe(self._probe_gpu)
            if gpu is not None:
                self._gpu.append(gpu)
            proc = self._safe(self._probe_proc)
            if proc is not None:
                self._proc.append(proc)

    @staticmethod
    def _safe(probe: Any) -> dict[str, float] | None:
        try:
            return probe()
        except Exception:  # a probe must never kill the sampler thread
            return None

    # ------------------------------------------------------------------ results
    def summary(self) -> dict[str, Any]:
        """Aggregates for one record: means + peaks, ``null`` when unsampled."""
        with self._lock:
            gpu, proc = list(self._gpu), list(self._proc)

        def pick(rows: list[dict[str, float]], key: str) -> list[float]:
            return [row[key] for row in rows if row.get(key) is not None]

        util = pick(gpu, "gpu_util_pct")
        vram = pick(gpu, "vram_used_mib")
        power = pick(gpu, "power_draw_w")
        cpu = pick(proc, "cpu_percent")
        rss = pick(proc, "rss_mib")
        return {
            "gpu_util_pct": _mean(util),
            "vram_used_mib": _mean(vram),
            "vram_peak_mib": max(vram) if vram else None,
            "power_draw_w": _mean(power),
            "cpu_percent": _mean(cpu),
            "rss_mib": max(rss) if rss else None,
            "gpu_samples": len(gpu),
        }


# ------------------------------------------------------------------- the sink


class ComputeMetrics:
    """Append-only JSONL metrics writer for one run (one ``run_id``).

    Writes ``<root>/<run_id>.jsonl``; one JSON object per line, keys sorted.
    ``record`` is safe to call from any thread (the file is opened per write
    and flushed) so a crashed run keeps everything it already measured.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        root: str | Path | None = None,
        sampler_factory: Any | None = None,
        interval_s: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.root = Path(root) if root is not None else default_metrics_root()
        self.path = self.root / f"{self.run_id}.jsonl"
        self.interval_s = interval_s
        self.enabled = enabled and not metrics_disabled_by_env()
        self._sampler_factory = sampler_factory
        self._lock = threading.Lock()
        self.count = 0

    # ------------------------------------------------------------------ writers
    def _make_sampler(self) -> Any:
        if self._sampler_factory is None:
            return ResourceSampler(interval_s=self.interval_s)
        try:
            return self._sampler_factory()
        except Exception:  # a broken factory degrades to no sampling
            return _NullSampler()

    def record(self, **fields: Any) -> dict[str, Any]:
        """Normalise and append one record; returns what was written."""
        rec: dict[str, Any] = {key: fields.get(key) for key in RECORD_FIELDS}
        rec["run_id"] = self.run_id
        # Keep unknown-but-useful fields (e.g. probe diagnostics) at the tail.
        extras = {
            key: value
            for key, value in fields.items()
            if key not in RECORD_FIELDS and value is not None
        }
        if extras:
            rec["extra"] = extras
        rec["observed_at"] = utcnow_iso()
        if not self.enabled:
            return rec
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self.count += 1
        return rec

    @contextlib.contextmanager
    def phase(
        self,
        stage: str,
        phase: str,
        *,
        device: str | None = None,
        sample: bool = True,
        **fields: Any,
    ) -> Iterator[dict[str, Any]]:
        """Time one phase, sample resources around it, write the record on exit.

        The yielded dict is the record under construction: the caller fills
        ``documents``/``pages``/``tokens_in``/... as they become known. A body
        exception records ``status=error:<type>`` and re-raises.
        """
        rec: dict[str, Any] = {**fields, "stage": stage, "phase": phase, "device": device}
        sampler: Any = _NullSampler()
        if sample:
            sampler = self._make_sampler()
        started = time.perf_counter()
        try:
            sampler.__enter__()
        except Exception:  # a failing sampler start must not fail the phase
            sampler = _NullSampler()
        try:
            yield rec
        except BaseException as exc:
            rec["status"] = rec.get("status") or f"error:{type(exc).__name__}"
            raise
        finally:
            rec["duration_ms"] = int((time.perf_counter() - started) * 1000)
            try:
                sampler.__exit__(None, None, None)
                rec.update(sampler.summary())
            except Exception:  # sampling stays best-effort on the way out too
                pass
            rec.setdefault("status", "ok")
            with contextlib.suppress(Exception):  # metrics never fail the phase
                self.record(**rec)

    # ------------------------------------------------------------------ reading
    def read(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """The records of one run (default: this sink's own run_id)."""
        return load_records(self.root, run_id=run_id or self.run_id)


def load_records(root: str | Path, *, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every JSONL record under ``root``, optionally filtered to one run."""
    root = Path(root)
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # a torn tail line after a kill
                        continue
                    if run_id is None or rec.get("run_id") == run_id:
                        records.append(rec)
        except OSError:
            continue
    return records


# ------------------------------------------------------- module-level activation
# Deep layers (router, acquisition, embeddings) emit through :func:`emit`
# without every constructor growing a metrics parameter. CLI commands wrap a
# run in :func:`metrics_scope`; tests activate their own scoped sink.

_active: ComputeMetrics | None = None


def active() -> ComputeMetrics | None:
    return _active


def emit(stage: str, phase: str, *, device: str | None = None, **fields: Any) -> None:
    """Record one event on the active sink (no-op when none is active)."""
    sink = _active
    if sink is None:
        return
    with contextlib.suppress(Exception):  # metrics never break the measured layer
        sink.record(stage=stage, phase=phase, device=device, **fields)


@contextlib.contextmanager
def metrics_scope(
    *,
    run_id: str | None = None,
    root: str | Path | None = None,
    sampler_factory: Any | None = None,
    interval_s: float = 5.0,
    enabled: bool = True,
) -> Iterator[ComputeMetrics]:
    """Activate a run-scoped metrics sink for the wrapped block."""
    global _active
    sink = ComputeMetrics(
        run_id=run_id,
        root=root,
        sampler_factory=sampler_factory,
        interval_s=interval_s,
        enabled=enabled,
    )
    previous = _active
    _active = sink
    try:
        yield sink
    finally:
        _active = previous


# --------------------------------------------------------------------- summary


def summarize(
    root: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate records into the ``fmd metrics summary`` payload.

    Groups by (stage, device): p50/p95 duration and tok/s, call counts, cache
    hit rates; totals GPU vs CPU wall time; and one ranked bottleneck line —
    the group with the largest total duration.
    """
    records = load_records(root, run_id=run_id)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (str(rec.get("stage")), str(rec.get("device") or "unknown"))
        groups.setdefault(key, []).append(rec)

    rows: list[dict[str, Any]] = []
    for (stage, device), items in sorted(groups.items()):
        durations = [
            float(rec["duration_ms"]) for rec in items if rec.get("duration_ms") is not None
        ]
        tps = [
            float(rec["total_tps"])
            for rec in items
            if rec.get("total_tps") is not None and float(rec["total_tps"]) > 0
        ]
        cache_flags = [bool(rec["cache_hit"]) for rec in items if rec.get("cache_hit") is not None]
        cache_hits = [int(rec["cache_hits"]) for rec in items if rec.get("cache_hits") is not None]
        row: dict[str, Any] = {
            "stage": stage,
            "device": device,
            "calls": len(items),
            "total_ms": int(sum(durations)),
            "p50_ms": percentile(durations, 50),
            "p95_ms": percentile(durations, 95),
            "p50_tps": percentile(tps, 50),
            "p95_tps": percentile(tps, 95),
        }
        if cache_flags:
            row["cache_hit_rate"] = sum(cache_flags) / len(cache_flags)
        elif cache_hits and sum(int(rec.get("documents") or 0) for rec in items) > 0:
            docs = sum(int(rec.get("documents") or 0) for rec in items)
            row["cache_hit_rate"] = sum(cache_hits) / docs if docs else None
        rows.append(row)

    gpu_ms = sum(row["total_ms"] for row in rows if row["device"].startswith("cuda"))
    cpu_ms = sum(row["total_ms"] for row in rows if not row["device"].startswith("cuda"))
    bottleneck = max(rows, key=lambda row: row["total_ms"]) if rows else None
    return {
        "run_id": run_id,
        "records": len(records),
        "groups": rows,
        "gpu_time_ms": gpu_ms,
        "cpu_time_ms": cpu_ms,
        "bottleneck": bottleneck,
        "bottleneck_line": (
            "bottleneck: stage={stage} device={device} calls={calls} "
            "total_s={total_s:.1f} p50_ms={p50:.0f} p95_ms={p95:.0f}".format(
                stage=bottleneck["stage"],
                device=bottleneck["device"],
                calls=bottleneck["calls"],
                total_s=bottleneck["total_ms"] / 1000.0,
                p50=bottleneck["p50_ms"] or 0.0,
                p95=bottleneck["p95_ms"] or 0.0,
            )
            if bottleneck
            else None
        ),
    }


def render_summary(summary: dict[str, Any]) -> str:
    """Plain-text table for ``fmd metrics summary`` (no --json)."""
    lines: list[str] = []
    run_label = summary.get("run_id") or "(all runs)"
    lines.append(f"compute metrics — run_id={run_label} records={summary['records']}")
    header = (
        f"{'stage':<22} {'device':<8} {'calls':>6} {'total_s':>9} "
        f"{'p50_ms':>9} {'p95_ms':>9} {'p50_tps':>8} {'p95_tps':>8} {'cache%':>7}"
    )
    lines.append(header)
    for row in summary["groups"]:
        cache = row.get("cache_hit_rate")
        cache_text = f"{cache * 100:.0f}" if cache is not None else "-"
        lines.append(
            f"{row['stage']:<22} {row['device']:<8} {row['calls']:>6} "
            f"{row['total_ms'] / 1000.0:>9.1f} "
            f"{row['p50_ms']:>9.0f} {row['p95_ms']:>9.0f} "
            f"{row['p50_tps'] if row['p50_tps'] is not None else float('nan'):>8.1f} "
            f"{row['p95_tps'] if row['p95_tps'] is not None else float('nan'):>8.1f} "
            f"{cache_text:>7}"
        )
    lines.append(
        "gpu_time_s={:.1f} cpu_time_s={:.1f}".format(
            summary["gpu_time_ms"] / 1000.0, summary["cpu_time_ms"] / 1000.0
        )
    )
    if summary.get("bottleneck_line"):
        lines.append(summary["bottleneck_line"])
    return "\n".join(lines)


__all__ = [
    "ComputeMetrics",
    "RECORD_FIELDS",
    "ResourceSampler",
    "default_metrics_root",
    "emit",
    "load_records",
    "metrics_disabled_by_env",
    "metrics_scope",
    "percentile",
    "query_gpu_memory",
    "query_gpu_telemetry",
    "query_process_usage",
    "render_summary",
    "summarize",
]
