"""ResourcePhaseManager (FMD-CLOUD-030): automatic GPU phases, offline.

The "model server" is a fake controller recording its call sequence; the GPU
probe is injected; no subprocess, no nvidia-smi, no Docling. Covers the
stop → pin → acquire → restart ordering, degradation reasons, restart/body
failure paths, the interrupt contract, ``fmd acquire``'s phased route and the
worker's automatic phased batched route.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from test_orchestrate_batched import BatchedHarness, _registry

from fmd.orchestrate.gpu_phases import (
    DEGRADED,
    SCRIPTED,
    ResourcePhaseManager,
    ServerControlError,
    control_off,
    pin_device_env,
    resolve_controller,
)
from fmd.orchestrate.ledger import utcnow


class FakeController:
    """Records the control sequence; readiness is a flag."""

    def __init__(self, *, events: list[str], fail_start: bool = False) -> None:
        self.events = events
        self.fail_start = fail_start
        self.stopped = True

    def stop(self) -> None:
        self.events.append("stop")
        self.stopped = True

    def start(self) -> None:
        self.events.append("start")
        if self.fail_start:
            raise ServerControlError("simulated server start failure")
        self.stopped = False

    def wait_ready(self, timeout_s: float) -> None:
        self.events.append("wait_ready")
        if self.fail_start:
            raise ServerControlError("server not ready (fake)")
        self.stopped = False


def _memory_probe(free_mib: float | None):
    def probe():
        if free_mib is None:
            return None
        return {"vram_used_mib": 0.0, "vram_free_mib": free_mib, "vram_total_mib": 16303.0}

    return probe


# --------------------------------------------------------------------- manager


def test_acquire_phase_stops_pins_restarts_and_waits() -> None:
    events: list[str] = []
    manager = ResourcePhaseManager(
        controller=FakeController(events=events),
        device="cuda",
        gpu_memory_probe=_memory_probe(None),  # unprobeable: keep cuda
    )
    with manager.acquire_phase(device="cuda", stage="acquire") as window:
        assert window.device == "cuda"
        assert window.control == SCRIPTED
        assert window.stopped_server is True
        assert events == ["stop"]  # stopped before the body runs
    assert events == ["stop", "start", "wait_ready"]  # restarted ready after
    assert window.ready is True


def test_acquire_phase_pins_both_device_envs_and_restores(monkeypatch) -> None:
    monkeypatch.delenv("FMD_DOCLING_DEVICE", raising=False)
    monkeypatch.delenv("FMD_OCR_DEVICE", raising=False)
    manager = ResourcePhaseManager(
        controller=FakeController(events=[]), device="cuda", gpu_memory_probe=_memory_probe(None)
    )
    seen: dict[str, str] = {}
    with manager.acquire_phase(device="cuda") as window:
        seen["docling"] = window.device
        import os

        seen["pinned"] = os.environ.get("FMD_DOCLING_DEVICE", "")
        seen["pinned_ocr"] = os.environ.get("FMD_OCR_DEVICE", "")
    import os

    assert seen["pinned"] == "cuda"
    assert seen["pinned_ocr"] == "cuda"
    assert "FMD_DOCLING_DEVICE" not in os.environ  # restored (was unset)
    assert "FMD_OCR_DEVICE" not in os.environ


def test_no_controller_degrades_to_cpu_with_reason() -> None:
    manager = ResourcePhaseManager(controller=None, device="cuda")
    assert manager.has_control is False
    device, reason = manager.resolve_device("cuda")
    assert device == "cpu" and reason == "no_server_control"
    with manager.acquire_phase(device="cuda") as window:
        assert window.device == "cpu"
        assert window.degraded is True
        assert window.reason == "no_server_control"
        assert window.stopped_server is False


def test_fmd_server_control_off_disables_everything(monkeypatch) -> None:
    monkeypatch.setenv("FMD_SERVER_CONTROL", "off")
    assert control_off() is True
    assert resolve_controller(server_script="scripts/serve_local.sh") is None
    manager = ResourcePhaseManager(
        controller=FakeController(events=[]), device="cuda", gpu_memory_probe=_memory_probe(None)
    )
    # resolve_controller re-checks the env switch: the operator's switch wins
    # even over an injected controller (the run-level degraded contract).
    assert manager.has_control is False
    device, reason = manager.resolve_device("cuda")
    assert device == "cpu" and reason == "no_server_control"


def test_insufficient_vram_after_stop_degrades_to_cpu() -> None:
    """The VRAM verdict is taken after the stop: a card still starved (some
    other consumer) degrades the window to CPU instead of OOM-ing Docling."""
    events: list[str] = []
    manager = ResourcePhaseManager(
        controller=FakeController(events=events),
        device="cuda",
        gpu_memory_probe=_memory_probe(500.0),  # 0.5 GiB free < 1.0 floor
        min_free_vram_gib=1.0,
    )
    with manager.acquire_phase(device="cuda") as window:
        assert window.device == "cpu"
        assert window.degraded is True
        assert window.reason is not None and "insufficient_gpu_memory" in window.reason
        assert events == ["stop"]  # the stop happened before the verdict
    assert events == ["stop", "start", "wait_ready"]  # restarted for extraction


def test_sufficient_vram_keeps_cuda() -> None:
    manager = ResourcePhaseManager(
        controller=FakeController(events=[]),
        device="cuda",
        gpu_memory_probe=_memory_probe(4000.0),
        min_free_vram_gib=1.0,
    )
    with manager.acquire_phase(device="cuda") as window:
        assert window.device == "cuda" and window.degraded is False


def test_auto_resolves_by_probe(monkeypatch) -> None:
    monkeypatch.delenv("FMD_SERVER_CONTROL", raising=False)
    manager = ResourcePhaseManager(
        controller=FakeController(events=[]),
        device="auto",
        gpu_memory_probe=_memory_probe(2000.0),
        min_free_vram_gib=1.0,
    )
    assert manager.resolve_device("auto") == ("cuda", None)
    starved = ResourcePhaseManager(
        controller=None, device="auto", gpu_memory_probe=_memory_probe(500.0), min_free_vram_gib=1.0
    )
    device, reason = starved.resolve_device("auto")
    assert device == "cpu" and "insufficient_gpu_memory" in (reason or "")
    blind = ResourcePhaseManager(
        controller=None, device="auto", gpu_memory_probe=_memory_probe(None)
    )
    device, reason = blind.resolve_device("auto")
    assert device == "cpu" and "unprobeable" in (reason or "")


def test_explicit_cpu_never_touches_the_server() -> None:
    events: list[str] = []
    manager = ResourcePhaseManager(
        controller=FakeController(events=events),
        device="cuda",
        gpu_memory_probe=_memory_probe(None),
    )
    with manager.acquire_phase(device="cpu") as window:
        assert window.device == "cpu"
        assert window.control == "none"  # not a degradation: nothing asked for GPU
        assert window.degraded is False
    assert events == []  # no stop, no start


def test_start_failure_aborts_window_and_leaves_server_stopped() -> None:
    events: list[str] = []
    manager = ResourcePhaseManager(
        controller=FakeController(events=events, fail_start=True),
        device="cuda",
        gpu_memory_probe=_memory_probe(None),
    )
    with (
        pytest.raises(ServerControlError),
        manager.acquire_phase(device="cuda"),
    ):
        pass
    # stop (enter) → start attempt → the failure path forces a stop
    assert events.count("stop") >= 2
    controller = manager.controller
    assert controller is not None and controller.stopped is True


def test_body_error_propagates_unchanged_and_skips_restart() -> None:
    events: list[str] = []
    manager = ResourcePhaseManager(
        controller=FakeController(events=events),
        device="cuda",
        gpu_memory_probe=_memory_probe(None),
    )

    class Boom(RuntimeError):
        pass

    with (
        pytest.raises(Boom),
        manager.acquire_phase(device="cuda"),
    ):
        raise Boom("conversion exploded")
    assert "start" not in events  # never start a server nobody will use
    assert "wait_ready" not in events
    import os

    assert os.environ.get("FMD_DOCLING_DEVICE") != "cuda"  # env restored


def test_stop_event_skips_restart() -> None:
    events: list[str] = []
    manager = ResourcePhaseManager(
        controller=FakeController(events=events),
        device="cuda",
        gpu_memory_probe=_memory_probe(None),
    )
    stop = threading.Event()
    stop.set()
    with manager.acquire_phase(device="cuda", stop_event=stop) as window:
        pass
    assert "start" not in events
    assert window.ready is None


def test_caller_can_set_restart_false() -> None:
    events: list[str] = []
    manager = ResourcePhaseManager(
        controller=FakeController(events=events),
        device="cuda",
        gpu_memory_probe=_memory_probe(None),
    )
    with manager.acquire_phase(device="cuda") as window:
        window.restart = False
    assert "start" not in events


def test_pin_device_env_round_trip(monkeypatch) -> None:
    monkeypatch.setenv("FMD_DOCLING_DEVICE", "auto")
    monkeypatch.delenv("FMD_OCR_DEVICE", raising=False)
    restore = pin_device_env("cuda")
    import os

    assert os.environ["FMD_DOCLING_DEVICE"] == "cuda"
    assert os.environ["FMD_OCR_DEVICE"] == "cuda"
    restore()
    assert os.environ["FMD_DOCLING_DEVICE"] == "auto"
    assert "FMD_OCR_DEVICE" not in os.environ


# ------------------------------------------------------- acquire --phased route


def test_orchestrator_acquire_phased_drives_the_server_lifecycle(tmp_path) -> None:
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/r.pdf", "r", 3)

    summary = harness.orchestrator.acquire(
        limit=5,
        phased=True,
        line_sink=harness.lines.append,
        controller=harness.controller,
        gpu_memory_probe=harness.gpu_memory_probe,  # host-independent: keep SCRIPTED
    )

    assert summary.metadata["zero_llm_calls"] is True
    assert summary.metadata["phased"] is True
    assert summary.metadata["phase_control"] == SCRIPTED
    assert summary.metadata["device"] == "cuda"
    # stop → convert → start → wait_ready: the window restarts the server and
    # leaves it running/ready — the next `fmd extract` (or worker tick)
    # consumes the parses immediately, no manual start in between.
    assert harness.controller.events[0] == "stop"
    assert "convert:r" in harness.controller.events
    assert "start" in harness.controller.events and "wait_ready" in harness.controller.events
    assert harness.controller.stopped is False  # left running, /health-ready
    assert summary.counts["parsed"] == 1
    assert summary.metadata["device"] == "cuda"


def test_orchestrator_acquire_phased_degraded_without_control(tmp_path, monkeypatch) -> None:
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/a.pdf", "a", 3)
    monkeypatch.setenv("FMD_SERVER_CONTROL", "off")

    summary = harness.orchestrator.acquire(limit=5, phased=True, controller=harness.controller)

    assert summary.failed is False
    assert summary.metadata["phased"] is True
    assert summary.metadata["phase_control"] == DEGRADED
    assert summary.metadata["phase_reason"] == "no_server_control"
    # No server control was attempted; the conversion still happened.
    assert not {"stop", "start", "wait_ready"} & set(harness.controller.events)
    assert summary.counts["parsed"] == 1


def test_orchestrator_acquire_phased_dry_run_touches_nothing(tmp_path, monkeypatch) -> None:
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/a.pdf", "a", 3)

    summary = harness.orchestrator.acquire(limit=5, phased=True, dry_run=True)

    assert summary.metadata["phase_reason"] == "dry_run"
    assert harness.controller.events == []
    assert summary.counts.get("to_convert") == 1


def test_acquire_emits_per_document_progress_lines(tmp_path) -> None:
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    harness.add_document("https://college.ac.in/r.pdf", "r", 3)
    lines: list[str] = []

    stage_events = harness.controller.events
    harness.orchestrator.acquire(
        limit=5, phased=False, line_sink=lines.append, controller=harness.controller
    )

    assert lines, "progress lines must be emitted per document"
    line = lines[0]
    assert line.startswith("[acquire] doc=")
    assert "pages=3" in line
    assert "cache_hit=False" in line
    assert "seconds=" in line
    assert not {"stop", "start", "wait_ready"} & set(stage_events)  # unphased: no server


# ------------------------------------------------------------ worker route


def _worker_orchestrator(tmp_path, registry):
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    return harness


def test_worker_phased_route_disabled_for_replay_router(tmp_path) -> None:
    """Offline tests (mock fixtures) never stop/start a real server."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    orchestrator = harness.orchestrator

    class _ReplayModels:
        profiles = {"bulk_extract": type("P", (), {"provider": "replay"})()}

    orchestrator.router.models = _ReplayModels()
    assert orchestrator._router_is_replay() is True
    assert orchestrator._phased_extract_enabled() is False
    summary = orchestrator._worker_extract(limit=5)
    assert summary.stage == "extract"
    assert harness.controller.events == []


def test_worker_phased_route_env_override(monkeypatch, tmp_path) -> None:
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    orchestrator = harness.orchestrator
    # Make the router look real (a non-replay provider list).
    orchestrator.router = type("FakeRealRouter", (), {})()
    orchestrator.router.models = None
    orchestrator._registry_override = True

    # models None ⇒ _router_is_replay returns True; give it a models stub:
    class _Models:
        profiles = {"bulk_extract": type("P", (), {"provider": "local"})()}

    orchestrator.router.models = _Models()
    assert orchestrator._router_is_replay() is False
    assert orchestrator._phased_extract_enabled() is True

    monkeypatch.setenv("FMD_PHASED_EXTRACT", "off")
    assert orchestrator._phased_extract_enabled() is False


def test_worker_phased_route_uses_batched_runner(tmp_path, monkeypatch) -> None:
    """With a real router + controller available, the tick routes phased."""
    registry = _registry()
    harness = BatchedHarness(tmp_path, registry, batch_pages=100)
    orchestrator = harness.orchestrator

    class _Models:
        profiles = {"bulk_extract": type("P", (), {"provider": "local"})()}

    class _Router:
        models = _Models()

    orchestrator.router = _Router()

    constructed: list[dict[str, Any]] = []

    class FakeRunner:
        def __init__(self, *, orchestrator, **kwargs) -> None:
            constructed.append({"device": kwargs.get("device"), "pages": kwargs.get("batch_pages")})

        def run(self, *, limit=None, dry_run=False, stop_event=None):
            from fmd.orchestrate.ledger import RunSummary

            return RunSummary(run_id="phased", stage="extract_batched", started_at=utcnow())

    monkeypatch.setattr("fmd.orchestrate.batched.BatchedRunner", FakeRunner)
    summary = orchestrator._worker_extract(limit=5)
    assert summary.stage == "extract_batched"
    assert summary.metadata["route"] == "phased_batched"
    assert constructed and constructed[0]["device"] == "cuda"  # config device
    assert constructed[0]["pages"] == 100  # acquire.yaml batch.pages
