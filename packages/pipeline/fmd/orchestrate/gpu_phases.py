"""Automatic resource phase manager (FMD-CLOUD-030): one GPU, time-shared.

The repo already had the correct pattern — ``fmd extract --batched`` stops the
model server, converts one page-batch on CUDA, restarts the server, extracts —
but the capability was trapped inside that one command. This module extracts
the server lifecycle into a reusable :class:`ResourcePhaseManager` so every
acquisition path (``fmd acquire --phased``, the worker's batched route, the
batched extractor) drives the same stop → pin → acquire → restart window with
zero manual care.

Why (measured, runbook §6): Docling converts 2–6× faster on CUDA
(0.166 vs 1.31 s/page) but OOMs while the 26B text model holds its 64k KV
cache (~589 MiB free after the CUDA context; Docling needs ~1.0–1.3 GiB), so
the two GPU consumers must be time-multiplexed.

Degradation contract: a run never fails purely for missing GPU control. No
controller (``FMD_SERVER_CONTROL=off`` or no server script) or a starved card
(nvidia-smi probe below ``min_free_vram_gib`` even after the stop) resolves the
phase to CPU with an explicit recorded reason — matching the batched runner's
historical degraded mode. The VRAM verdict is taken **after** the stop: with a
resident 26B the card always looks starved, and the stop is exactly what frees
it.

Server control stays injected: any object with ``stop()``/``start()``
/``wait_ready(timeout_s)`` works; :class:`ScriptServerController` shells out to
``scripts/serve_local.sh``. This module reuses those controllers — there is no
second lifecycle implementation.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from fmd.obs.metrics import query_gpu_memory

#: Forces degraded mode for one run (no server control, no GPU batch window).
SERVER_CONTROL_OFF = "off"
DEGRADED = "degraded"
SCRIPTED = "scripted"

#: Device env vars pinned for the duration of an acquire phase. Docling reads
#: ``FMD_DOCLING_DEVICE`` (tiers.docling_device); the in-process OCR runner
#: reads ``FMD_OCR_DEVICE`` (rapidocr_backend). Both must follow the window's
#: resolved device, or a CUDA window would still convert on CPU.
DEVICE_ENV = "FMD_DOCLING_DEVICE"
OCR_DEVICE_ENV = "FMD_OCR_DEVICE"

#: Conservative floor when no registry is available for the T1 backend's own
#: ``min_free_vram_gib``: Docling needs ~1.0–1.3 GiB (runbook §6); 1.5 leaves
#: margin without demanding the whole card.
DEFAULT_MIN_FREE_VRAM_GIB = 1.5


class ServerControlError(RuntimeError):
    """A stop/start/readiness operation failed; the phase aborts, server stopped."""


class ServerController(Protocol):
    """The seam the phase manager drives. Tests inject a fake.

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


def local_base_url(registry: Any) -> str | None:
    """The local text model's base URL from the registry (env-interpolated)."""
    models = getattr(registry, "models", None)
    providers = getattr(models, "providers", None) or {}
    local = providers.get("local")
    url = getattr(local, "base_url", None)
    return str(url) if url else None


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
                f"server script {action} exited {result.returncode}: "
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


def control_off() -> bool:
    """Is server control explicitly disabled for this process?"""
    return os.environ.get("FMD_SERVER_CONTROL", "").strip().lower() == SERVER_CONTROL_OFF


def pin_device_env(device: str):
    """Pin both acquisition device env vars for a phase; restore after.

    The phase sets ``FMD_DOCLING_DEVICE`` and ``FMD_OCR_DEVICE`` around the
    window so converters follow the resolved device without editing config.
    The previous values are always restored.
    """
    previous = {env: os.environ.get(env) for env in (DEVICE_ENV, OCR_DEVICE_ENV)}

    def restore() -> None:
        for env, value in previous.items():
            if value is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = value

    os.environ[DEVICE_ENV] = device
    os.environ[OCR_DEVICE_ENV] = device
    return restore


@dataclass
class PhaseWindow:
    """What one acquire-phase window decided and did (mutable by the caller).

    ``restart`` is the caller's escape hatch: a driver that knows the run is
    ending (SIGINT, drained queue) sets it to ``False`` so the manager never
    starts a server nobody will use. The manager itself also honours the
    ``stop_event`` it was given.
    """

    requested: str
    device: str
    control: str = "none"  # scripted | degraded | none
    reason: str | None = None
    degraded: bool = False
    stopped_server: bool = False
    restart: bool = True
    ready: bool | None = None
    stop_s: float | None = None
    start_s: float | None = None
    wait_ready_s: float | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "device": self.device,
            "control": self.control,
            "reason": self.reason,
            "degraded": self.degraded,
            "stopped_server": self.stopped_server,
            "restart": self.restart,
            "ready": self.ready,
            "stop_s": self.stop_s,
            "start_s": self.start_s,
            "wait_ready_s": self.wait_ready_s,
            "error": self.error,
        }


@contextlib.contextmanager
def _null_phase() -> Iterator[dict[str, Any]]:
    """Stand-in for ``ComputeMetrics.phase`` when no metrics sink is active."""
    yield {}


def resolve_controller(
    *,
    controller: ServerController | None = None,
    server_script: str | Path | None = None,
    registry: Any | None = None,
) -> ServerController | None:
    """The effective controller: injected wins, then the registry's script.

    ``FMD_SERVER_CONTROL=off`` (the operator's switch) wins over everything and
    yields ``None`` — degraded mode. A script that cannot be resolved also
    yields ``None`` rather than a controller that would fail at every stop.
    """
    if control_off():
        return None
    if controller is not None:
        return controller
    script = server_script
    if not script and registry is not None:
        batch = getattr(getattr(registry, "acquire", None), "batch", None)
        script = getattr(batch, "server_script", None)
    if not script:
        return None
    if not resolve_script(script).is_file():
        return None
    return ScriptServerController(script, base_url=local_base_url(registry))


class ResourcePhaseManager:
    """Reusable automatic GPU phase manager.

    One instance per run; ``acquire_phase`` may be entered many times (the
    batched runner opens one window per cycle). The manager never calls the
    model and never touches documents — it owns only the device env and the
    server lifecycle around the yielded window.
    """

    def __init__(
        self,
        *,
        controller: ServerController | None = None,
        server_script: str | Path | None = None,
        registry: Any | None = None,
        device: str = "cuda",
        poll_timeout_s: float = 600.0,
        keep_server: bool = False,
        line_sink: Any | None = None,
        gpu_memory_probe: Any | None = None,
        min_free_vram_gib: float | None = None,
        on_restarted: Any | None = None,
    ) -> None:
        self.controller = resolve_controller(
            controller=controller, server_script=server_script, registry=registry
        )
        self.device = device
        self.poll_timeout_s = float(poll_timeout_s)
        self.keep_server = bool(keep_server)
        self.line_sink = line_sink or _stderr_line
        self._gpu_memory_probe = gpu_memory_probe or query_gpu_memory
        self.min_free_vram_gib = (
            float(min_free_vram_gib)
            if min_free_vram_gib is not None
            else self._configured_min_vram(registry)
        )
        self.on_restarted = on_restarted

    # ------------------------------------------------------------------ probes
    @staticmethod
    def _configured_min_vram(registry: Any) -> float:
        """T1's own CUDA floor from the registry, when one is configured."""
        backends = getattr(getattr(registry, "acquire", None), "backends", None) or {}
        spec = backends.get("T1")
        raw = getattr(spec, "min_free_vram_gib", None)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_MIN_FREE_VRAM_GIB
        return value if value > 0 else DEFAULT_MIN_FREE_VRAM_GIB

    def _free_vram_gib(self) -> float | None:
        """Free VRAM (GiB) via the injected probe; None when unprobeable."""
        try:
            memory = self._gpu_memory_probe()
        except Exception:
            return None
        if not memory:
            return None
        free = memory.get("vram_free_mib")
        return float(free) / 1024.0 if free is not None else None

    # ------------------------------------------------------------------ public
    @property
    def has_control(self) -> bool:
        return self.controller is not None

    def resolve_device(
        self, requested: str | None = None, *, free_gib: float | None = None
    ) -> tuple[str, str | None]:
        """Pre-flight device resolution (no server interaction).

        Returns ``(device, reason)``. ``auto`` becomes cuda only when the GPU
        probe can see a card with enough free VRAM; anything else degrades to
        cpu with the reason recorded. A cuda request without a controller is a
        degradation (``no_server_control``), matching the historical
        run-level fallback.

        ``free_gib`` is an override for callers that probed themselves; the
        manager does **not** probe before stopping the server — with a resident
        26B the card always looks starved, and the stop is what frees it.
        """
        value = str(requested or self.device).strip().lower()
        base = value.split(":", 1)[0]
        if base == "cpu":
            return "cpu", None
        if base == "auto":
            free = self._free_vram_gib() if free_gib is None else free_gib
            if free is None:
                return "cpu", "auto:gpu_unprobeable (nvidia-smi absent)"
            if free < self.min_free_vram_gib:
                return "cpu", f"auto:insufficient_gpu_memory({free:.2f}GiB free)"
            return "cuda", None
        # cuda / cuda:N — the VRAM verdict is taken after the stop instead.
        if self.controller is None:
            return "cpu", "no_server_control"
        if free_gib is not None and free_gib < self.min_free_vram_gib:
            return "cpu", f"insufficient_gpu_memory({free_gib:.2f}GiB free)"
        return value, None

    @contextlib.contextmanager
    def acquire_phase(
        self,
        *,
        device: str | None = None,
        stage: str = "acquire",
        stop_event: threading.Event | None = None,
    ) -> Iterator[PhaseWindow]:
        """One automatic acquisition window: stop server → pin devices → yield
        → restart + wait_ready (try/finally).

        Degrades to CPU — with an explicit reason on the yielded window and in
        the metrics record — when there is no controller or the card cannot
        fit the converters even after the stop. Every exit path is safe: the
        env pin is always restored, and a restart failure is re-raised (as
        ``ServerControlError``) only when the body itself was not already
        failing.
        """
        requested = str(device or self.device).strip().lower()
        sink = _active_sink()
        if sink is not None:
            phase_cm = sink.phase(stage, "acquire_window", device=requested, control=None)
        else:
            phase_cm = _null_phase()
        with phase_cm as rec:
            window = self._enter_window(requested, rec)
            restore = pin_device_env(window.device)
            body_error: BaseException | None = None
            try:
                yield window
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                self._exit_window(window, restore, stop_event, body_error, rec)

    # ----------------------------------------------------------------- windows
    def _enter_window(self, requested: str, rec: dict[str, Any]) -> PhaseWindow:
        resolved, reason = self.resolve_device(requested)
        window = PhaseWindow(requested=requested, device=resolved, reason=reason)
        if not resolved.startswith("cuda"):
            # cpu (explicit or degraded): no server interaction needed.
            window.control = DEGRADED if requested.startswith("cuda") else "none"
            window.degraded = requested.startswith("cuda")
            self._announce(window)
            rec.update(window.as_dict())
            return window
        controller = self.controller
        assert controller is not None  # resolve_device guarantees this
        started = time.perf_counter()
        controller.stop()
        window.stop_s = time.perf_counter() - started
        window.stopped_server = True
        window.control = SCRIPTED
        # The stop was the whole point: probe with the card now free. A card
        # that is still starved degrades the window to CPU instead of OOM-ing
        # Docling after it loads weights (the runbook's old failure mode).
        free = self._free_vram_gib()
        if free is not None and free < self.min_free_vram_gib:
            window.device = "cpu"
            window.reason = f"insufficient_gpu_memory({free:.2f}GiB free after stop)"
            window.control = DEGRADED
            window.degraded = True
            self._announce(window)
        rec.update(window.as_dict())
        return window

    def _exit_window(
        self,
        window: PhaseWindow,
        restore: Any,
        stop_event: threading.Event | None,
        body_error: BaseException | None,
        rec: dict[str, Any],
    ) -> None:
        restore()
        interrupted = stop_event is not None and stop_event.is_set()
        restart = (
            window.stopped_server and window.restart and not interrupted and body_error is None
        )
        if restart:
            self._restart(window, body_error)
        elif window.stopped_server and (interrupted or body_error is None):
            # Interrupted, or a caller-set restart=False: leave it stopped, but
            # make "stopped" explicit (the controller ops are idempotent).
            with contextlib.suppress(Exception):
                self.controller.stop()  # type: ignore[union-attr]
        rec.update(window.as_dict())

    def _restart(self, window: PhaseWindow, body_error: BaseException | None) -> None:
        """Restart the server and poll readiness (the window's exit ramp)."""
        controller = self.controller
        assert controller is not None
        try:
            started = time.perf_counter()
            controller.start()
            window.start_s = time.perf_counter() - started
            if self.on_restarted is not None:
                self.on_restarted(window)
            started = time.perf_counter()
            controller.wait_ready(self.poll_timeout_s)
            window.wait_ready_s = time.perf_counter() - started
            window.ready = True
        except Exception as exc:
            window.ready = False
            window.error = str(exc)
            # Leave the box safe even when the error propagates below.
            with contextlib.suppress(Exception):
                controller.stop()
            if body_error is None:
                if isinstance(exc, ServerControlError):
                    raise
                raise ServerControlError(f"server restart failed: {exc}") from exc
            self._announce(window, stage=f"restart failed after {type(body_error).__name__}")

    def _announce(self, window: PhaseWindow, *, stage: str | None = None) -> None:
        """One flushed stderr line per degraded decision (logs that say why)."""
        if not window.degraded:
            return
        suffix = f" ({stage})" if stage else ""
        self.line_sink(
            f"[gpu-phases] device={window.device} control={window.control} "
            f"reason={window.reason}{suffix}"
        )


def _active_sink() -> Any:
    """The active compute-metrics sink (no-op recording when none is active)."""
    from fmd.obs.metrics import active

    return active()


def _stderr_line(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


__all__ = [
    "DEFAULT_MIN_FREE_VRAM_GIB",
    "DEGRADED",
    "DEVICE_ENV",
    "OCR_DEVICE_ENV",
    "NullServerController",
    "PhaseWindow",
    "ResourcePhaseManager",
    "SCRIPTED",
    "SERVER_CONTROL_OFF",
    "ScriptServerController",
    "ServerControlError",
    "ServerController",
    "control_off",
    "local_base_url",
    "pin_device_env",
    "resolve_controller",
    "resolve_script",
]
