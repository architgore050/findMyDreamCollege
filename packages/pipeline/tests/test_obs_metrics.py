"""Compute metrics (FMD-CLOUD-030): sink, sampler, summary and CLI — offline.

All sampling uses injected fakes (no GPU, no nvidia-smi, no network). The
router emission test drives a real ModelRouter with an injected fake provider.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from fmd.config import load_registry
from fmd.llm.providers import ProviderResult
from fmd.llm.router import ModelRouter
from fmd.obs.metrics import (
    ComputeMetrics,
    ResourceSampler,
    load_records,
    metrics_disabled_by_env,
    metrics_scope,
    percentile,
    query_gpu_memory,
    query_gpu_telemetry,
    query_process_usage,
    render_summary,
    summarize,
)

REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"


def _registry():
    return load_registry(REPO_CONFIG, env={})


class FakeSampler:
    """Deterministic sampler: canned aggregates, no subprocesses."""

    def __init__(self, summary: dict[str, Any] | None = None) -> None:
        self.summary_value = summary or {
            "gpu_util_pct": 87.5,
            "vram_used_mib": 4096.0,
            "vram_peak_mib": 8192.0,
            "power_draw_w": 120.0,
            "cpu_percent": 40.0,
            "rss_mib": 2048.0,
            "gpu_samples": 12,
        }

    def __enter__(self) -> FakeSampler:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return dict(self.summary_value)


def test_metrics_scope_records_jsonl(tmp_path) -> None:
    with metrics_scope(run_id="run-1", root=tmp_path, sampler_factory=FakeSampler) as metrics:
        with metrics.phase("acquire", "convert", device="cuda") as rec:
            rec["documents"] = 1
            rec["pages"] = 140
        metrics_emit_ok = metrics.record(
            stage="llm",
            phase="bulk_extract",
            device="cuda",
            duration_ms=1500,
            tokens_in=800,
            tokens_out=200,
            total_tps=25.0,
            input_ref="abc:family:v1",
        )
    path = tmp_path / "run-1.jsonl"
    assert path.is_file()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    convert = lines[0]
    assert convert["stage"] == "acquire" and convert["phase"] == "convert"
    assert convert["device"] == "cuda"
    assert convert["pages"] == 140
    assert convert["status"] == "ok"
    assert convert["gpu_util_pct"] == pytest.approx(87.5)  # fake sampler merged
    assert convert["vram_peak_mib"] == 8192.0
    assert convert["run_id"] == "run-1"
    assert "observed_at" in convert
    llm = lines[1]
    assert llm["input_ref"] == "abc:family:v1"  # the content-hash reference, not a URL
    assert metrics.count == 2
    assert metrics_emit_ok["run_id"] == "run-1"


def test_metrics_phase_records_body_errors(tmp_path) -> None:
    with (
        metrics_scope(run_id="run-err", root=tmp_path) as metrics,
        pytest.raises(RuntimeError),
        metrics.phase("acquire", "convert", device="cuda"),
    ):
        raise RuntimeError("docling exploded")
    (record,) = load_records(tmp_path, run_id="run-err")
    assert record["status"] == "error:RuntimeError"
    assert record["duration_ms"] >= 0


def test_metrics_disabled_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FMD_COMPUTE_METRICS", "off")
    assert metrics_disabled_by_env() is True
    metrics = ComputeMetrics(run_id="off", root=tmp_path)
    metrics.record(stage="acquire", phase="convert", device="cpu", duration_ms=1)
    assert not (tmp_path / "off.jsonl").exists()  # nothing written
    assert metrics.count == 0


def test_sampler_with_fake_probes() -> None:
    samples = iter(
        [
            {"gpu_util_pct": 10.0, "vram_used_mib": 1000.0, "power_draw_w": 50.0},
            {"gpu_util_pct": 90.0, "vram_used_mib": 3000.0, "power_draw_w": 150.0},
        ]
    )
    procs = iter([{"cpu_percent": 10.0, "rss_mib": 500.0}, {"cpu_percent": 30.0, "rss_mib": 900.0}])

    sampler = ResourceSampler(
        interval_s=0.05, probe_gpu=lambda: next(samples), probe_proc=lambda: next(procs)
    )
    with sampler:
        while len(sampler._gpu) < 2 or len(sampler._proc) < 2:  # determinism, not timing
            if not sampler._stop.wait(0.01):
                break
    summary = sampler.summary()
    assert summary["gpu_util_pct"] == pytest.approx(50.0)
    assert summary["vram_peak_mib"] == 3000.0
    assert summary["power_draw_w"] == pytest.approx(100.0)
    assert summary["cpu_percent"] == pytest.approx(20.0)
    assert summary["rss_mib"] == 900.0  # peak
    assert summary["gpu_samples"] == 2


def test_sampler_degrades_when_probes_fail() -> None:
    def broken():
        raise RuntimeError("nvidia-smi exploded")

    sampler = ResourceSampler(interval_s=0.05, probe_gpu=broken, probe_proc=lambda: None)
    with sampler:
        pass
    summary = sampler.summary()
    assert summary["gpu_util_pct"] is None
    assert summary["gpu_samples"] == 0
    assert summary["cpu_percent"] is None  # None-returning probe records nothing


def test_query_helpers_absent_tools_are_none(monkeypatch) -> None:
    import fmd.obs.metrics as mod

    monkeypatch.setattr(mod, "nvidia_smi_path", lambda: None)
    assert query_gpu_telemetry() is None
    assert query_gpu_memory() is None
    assert query_process_usage(pid=999999999) is None  # no such process


def test_summarize_and_bottleneck(tmp_path) -> None:
    with metrics_scope(run_id="run-s", root=tmp_path) as metrics:
        with metrics.phase("acquire", "convert", device="cuda") as rec:
            rec["duration_ms"] = 3000
            rec["pages"] = 100
            rec["cache_hit"] = True
        with metrics.phase("acquire", "convert", device="cuda") as rec:
            rec["duration_ms"] = 1000
            rec["pages"] = 30
            rec["cache_hit"] = False  # a real convert, exactly as the stage emits
        with metrics.phase("acquire", "convert", device="cpu") as rec:
            rec["duration_ms"] = 500
            rec["pages"] = 5
        metrics.record(
            stage="llm",
            phase="bulk_extract",
            device="cuda",
            duration_ms=1500,
            total_tps=25.0,
            cache_hit=False,
        )
    summary = summarize(tmp_path, run_id="run-s")
    assert summary["records"] == 4
    by_key = {(row["stage"], row["device"]): row for row in summary["groups"]}
    cuda = by_key[("acquire", "cuda")]
    assert cuda["calls"] == 2
    assert cuda["total_ms"] == 4000
    assert cuda["p50_ms"] == 1000 and cuda["p95_ms"] == 3000
    assert cuda["cache_hit_rate"] == pytest.approx(0.5)
    assert summary["gpu_time_ms"] == 5500
    assert summary["cpu_time_ms"] == 500
    assert summary["bottleneck"]["stage"] == "acquire"
    assert summary["bottleneck"]["device"] == "cuda"
    assert "bottleneck: stage=acquire device=cuda" in summary["bottleneck_line"]
    text = render_summary(summary)
    assert "acquire" in text and "bottleneck" in text


def test_percentile_nearest_rank() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    values = [float(i) for i in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0


# --------------------------------------------------------------------- CLI


def test_metrics_summary_cli_table_and_json(tmp_path, capsys) -> None:
    from fmd.cli import _build_parser

    with (
        metrics_scope(run_id="run-cli", root=tmp_path) as metrics,
        metrics.phase("acquire", "convert", device="cuda") as rec,
    ):
        rec["duration_ms"] = 1200
        rec["pages"] = 50

    parser = _build_parser()
    args = parser.parse_args(
        ["metrics", "summary", "--run-id", "run-cli", "--metrics-dir", str(tmp_path)]
    )
    from fmd.cli import _cmd_metrics

    assert _cmd_metrics(args) == 0
    out = capsys.readouterr().out
    assert "compute metrics" in out and "bottleneck" in out

    args_json = parser.parse_args(["metrics", "summary", "--json", "--metrics-dir", str(tmp_path)])
    assert _cmd_metrics(args_json) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 1
    assert payload["groups"][0]["stage"] == "acquire"


def test_metrics_summary_cli_empty_dir(tmp_path, capsys) -> None:
    from fmd.cli import _build_parser, _cmd_metrics

    parser = _build_parser()
    args = parser.parse_args(["metrics", "summary", "--metrics-dir", str(tmp_path)])
    assert _cmd_metrics(args) == 1
    assert "no compute metrics" in capsys.readouterr().out


# ----------------------------------------------------------- router emission


class _Out(BaseModel):
    ok: bool = True


class FakeProvider:
    """A provider that never touches the network."""

    name = "local"
    kind = "openai_compatible"
    supports_guided_json = True
    config = None

    def complete(self, **kwargs: Any) -> ProviderResult:
        time.sleep(0.002)  # a real (tiny) latency so tok/s are derivable
        return ProviderResult(
            content={"ok": True},
            raw="{}",
            model="gemma-4-26b-a4b-gguf",
            provider="local",
            tokens_in=100,
            tokens_out=20,
        )


def test_router_emits_compute_metric_per_call(tmp_path) -> None:
    models = _registry().models
    router = ModelRouter(models, providers={"local": FakeProvider()})
    profile = models.profiles["bulk_extract"]
    profile = profile.model_copy(update={"provider": "local", "required": True})
    models = models.model_copy(update={"profiles": {**models.profiles, "bulk_extract": profile}})
    router = ModelRouter(models, providers={"local": FakeProvider()})

    with metrics_scope(run_id="run-router", root=tmp_path):
        router.call("bulk_extract", _Out, "system", "user", "abc:family:v1")

    (record,) = [r for r in load_records(tmp_path, run_id="run-router") if r["stage"] == "llm"]
    assert record["phase"] == "bulk_extract"
    assert record["provider"] == "local"
    assert record["model"] == "gemma-4-26b-a4b-gguf"
    assert record["tokens_in"] == 100 and record["tokens_out"] == 20
    assert record["duration_ms"] >= 0
    assert record["input_ref"] == "abc:family:v1"
    assert record["total_tps"] is not None and record["total_tps"] > 0


def test_router_without_active_metrics_still_works(tmp_path) -> None:
    models = _registry().models
    profile = models.profiles["bulk_extract"].model_copy(
        update={"provider": "local", "required": True}
    )
    models = models.model_copy(update={"profiles": {**models.profiles, "bulk_extract": profile}})
    router = ModelRouter(models, providers={"local": FakeProvider()})
    assert router.call("bulk_extract", _Out, "system", "user", "ref").ok is True


def test_router_cache_hit_emits_zero_duration_metric(tmp_path) -> None:
    models = _registry().models
    profile = models.profiles["bulk_extract"].model_copy(
        update={"provider": "local", "required": True}
    )
    models = models.model_copy(update={"profiles": {**models.profiles, "bulk_extract": profile}})
    provider = FakeProvider()
    router = ModelRouter(models, providers={"local": provider})
    with metrics_scope(run_id="run-cache", root=tmp_path):
        models.budgets = models.budgets.model_copy(update={"cache": True})
        router.budgets = models.budgets
        router.call("bulk_extract", _Out, "system", "user", "ref")
        router.call("bulk_extract", _Out, "system", "user", "ref")  # same prompt+input ⇒ cache
    records = [r for r in load_records(tmp_path, run_id="run-cache") if r["stage"] == "llm"]
    assert len(records) == 2
    assert records[0]["cache_hit"] is None and records[1]["cache_hit"] is True
    assert records[1]["duration_ms"] == 0


def test_sampler_factory_failure_degrades_to_null(tmp_path) -> None:
    def broken_factory():
        raise RuntimeError("no sampler today")

    metrics = ComputeMetrics(run_id="f", root=tmp_path, sampler_factory=broken_factory)
    with metrics.phase("acquire", "convert", device="cuda") as rec:
        rec["pages"] = 3
    (record,) = load_records(tmp_path, run_id="f")
    assert record["pages"] == 3
    assert record["gpu_util_pct"] is None
    assert record["status"] == "ok"


def test_concurrent_records_are_atomic_lines(tmp_path) -> None:
    metrics = ComputeMetrics(run_id="threads", root=tmp_path)

    def writer(n: int) -> None:
        for _ in range(25):
            metrics.record(stage="t", phase=f"p{n}", device="cpu", duration_ms=n)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = load_records(tmp_path, run_id="threads")
    assert len(records) == 100
    assert all(isinstance(r["duration_ms"], int) for r in records)
