"""Embedding backend measurement — the operator runs this on the host.

``uv run python -m fmd.llm.embed_bench`` (in ``packages/pipeline``, weights
already cached for the configured model).

The agent that wrote this module cannot run it (no GPU, no weights), so every
number it prints is measured here, live, on the operator's machine. Nothing in
this file invents a throughput or a VRAM figure: a metric that cannot be
captured is printed as ``[not measured]`` with the reason.

What it reports, per backend kind:

* environment: backend/model/dim/device/batch, CPU count, torch threads;
* VRAM before/after via ``nvidia-smi`` when present (the CPU path must show a
  ~0 delta — that is the point of the decision; an HTTP backend reports the
  *client* view only and says so);
* throughput and per-text latency (p50/p95) over synthetic name-sized and
  sentence-sized inputs at 1/32/256/1024 texts (warmup excluded);
* a recommendation computed by :func:`recommend`, a deterministic rule over
  the measured numbers — never a canned answer.

Modes: default = full measurement; ``--probe`` = exactly one embedding call
(the runbook's "one embedding call" verification step); ``--json`` = machine-
readable report for pasting into the runbook.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

#: Runbook §8 ER budget: one resolve_with_audit batch of ~500 mention/candidate
#: name strings should embed in under this many seconds. Latency above it means
#: the ER stage, not the LLM, becomes the bottleneck and FMD_EMBED_BATCH
#: tuning (or the embed_openai backend) needs a look.
ER_NAME_BUDGET_S = 60.0

#: Sentences per scoring run (insights + dedup) should embed in under this.
INSIGHT_BUDGET_S = 10.0

_NAME_PROMPTS = [
    "King George's Medical University",
    "Government Medical College Nagpur",
    "Baba Farid University of Health Sciences",
    "Rajendra Institute of Medical Technology",
    "Maharashtra University of Health Sciences",
    "Christian Medical College Vellore",
    "Seth GS Medical College",
    "Kasturba Medical College Manipal",
]
_SENTENCE_PROMPTS = [
    "Placement cell reports a median package of 4.2 LPA for 2024 graduates.",
    "The institution holds NAAC accreditation at the A grade since 2019.",
    "Hostel accommodation is available for 60 percent of enrolled students.",
    "Internships are arranged at partner diagnostic laboratories in Pune.",
    "Admission requires the state CET with a biology aggregate of 50 percent.",
    "The clinical laboratory is NABL accredited and open to BSc students.",
]


def _synthetic_texts(kind: str, count: int) -> list[str]:
    base = _NAME_PROMPTS if kind == "names" else _SENTENCE_PROMPTS
    return [base[index % len(base)] for index in range(count)]


def _percentiles(samples: Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    mid = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {"p50_ms": round(mid * 1000, 3), "p95_ms": round(p95 * 1000, 3)}


def _vram_mib() -> dict[str, Any] | None:
    """nvidia-smi snapshot, or None with a reason (desktops without GPU tooling)."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        out = subprocess.run(
            [
                binary,
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    used, _, total = out.partition(",")
    try:
        return {"used_mib": int(used.strip()), "total_mib": int(total.strip())}
    except ValueError:
        return None


def _torch_threads() -> dict[str, Any]:
    try:
        import torch

        return {
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        }
    except Exception:  # noqa: BLE001 - measurement must not depend on torch
        return {"torch_version": "[not measured: torch not importable]"}


def _timed_call(backend: Any, texts: list[str]) -> tuple[float, int | None]:
    started = time.perf_counter()
    batch = backend.embed(texts)
    elapsed = time.perf_counter() - started
    if len(batch.vectors) != len(texts):  # defensive; contract tested elsewhere
        raise SystemExit(f"backend returned {len(batch.vectors)} vectors for {len(texts)} inputs")
    return elapsed, batch.tokens_in


def measure(backend: Any, *, counts: Sequence[int] = (1, 32, 256, 1024)) -> dict[str, Any]:
    """Warmup, then per-size throughput/latency for names and sentences."""
    report: dict[str, Any] = {"backend": backend.name, "kind": backend.kind}
    vram_before = _vram_mib()
    report["vram_before"] = vram_before or "[not measured: nvidia-smi unavailable]"

    # Warmup: first call pays model load (CPU) or connection setup (HTTP);
    # it is excluded from every timed row and its duration is reported as-is.
    warm_started = time.perf_counter()
    backend.embed(_synthetic_texts("names", 2))
    report["warmup_s"] = round(time.perf_counter() - warm_started, 3)

    rows: list[dict[str, Any]] = []
    for kind in ("names", "sentences"):
        for count in counts:
            texts = _synthetic_texts(kind, count)
            samples = []
            tokens_in = None
            for _ in range(3):
                elapsed, tokens_in = _timed_call(backend, texts)
                samples.append(elapsed / count)
            per_text = statistics.mean(samples)
            rows.append(
                {
                    "kind": kind,
                    "texts": count,
                    "wall_s": round(per_text * count, 3),
                    "texts_per_s": round(1.0 / per_text, 2),
                    **_percentiles(samples),
                    "tokens_in": tokens_in if tokens_in is not None else "[not reported]",
                }
            )
    report["rows"] = rows

    vram_after = _vram_mib()
    report["vram_after"] = vram_after or "[not measured: nvidia-smi unavailable]"
    if vram_before and vram_after:
        report["vram_delta_mib"] = vram_after["used_mib"] - vram_before["used_mib"]
    else:
        report["vram_delta_mib"] = "[not measured: nvidia-smi unavailable]"
    report["environment"] = {"cpu_count": _cpu_count(), **_torch_threads()}
    report["recommendation"] = recommend(report)
    return report


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 0


def recommend(report: dict[str, Any]) -> dict[str, Any]:
    """Deterministic decision rule over measured numbers only.

    The rule (runbook §8): the CPU backend is fit for purpose when the largest
    measured ER-shaped batch fits the ER name budget and the sentence batch
    fits the insight budget. It is never overridden by an expectation: if the
    numbers miss the budgets, the recommendation says so and names the knobs.
    """
    rows = report.get("rows", [])
    names = next((row for row in rows if row["kind"] == "names" and row["texts"] == 1024), None)
    sentences = next(
        (row for row in rows if row["kind"] == "sentences" and row["texts"] == 256), None
    )
    checks: list[dict[str, Any]] = []
    if names is None:
        checks.append({"check": "er_names_within_budget", "result": "[not measured]"})
    else:
        checks.append(
            {
                "check": "er_names_within_budget",
                "measured_s": names["wall_s"],
                "budget_s": ER_NAME_BUDGET_S,
                "result": "pass" if names["wall_s"] <= ER_NAME_BUDGET_S else "fail",
            }
        )
    if sentences is None:
        checks.append({"check": "insight_sentences_within_budget", "result": "[not measured]"})
    else:
        checks.append(
            {
                "check": "insight_sentences_within_budget",
                "measured_s": sentences["wall_s"],
                "budget_s": INSIGHT_BUDGET_S,
                "result": "pass" if sentences["wall_s"] <= INSIGHT_BUDGET_S else "fail",
            }
        )
    verdict = (
        "cpu_backend_fit_for_purpose"
        if all(check.get("result") == "pass" for check in checks)
        else "cpu_backend_misses_budget"
    )
    return {
        "verdict": verdict,
        "checks": checks,
        "note": (
            "verdict is computed from the measured rows above; a fail names the "
            "knobs (FMD_EMBED_BATCH, FMD_EMBED_DEVICE) before any backend switch"
        ),
    }


def probe() -> dict[str, Any]:
    """One embedding call: proves the backend is real and prints the dimension."""
    from fmd.config import get_registry
    from fmd.llm.embeddings import build_embedding_backend

    backend = build_embedding_backend(get_registry().models)
    batch = backend.embed(["verify: one embedding call"])
    return {
        "backend": backend.name,
        "kind": backend.kind,
        "model": batch.model,
        "dim": batch.dim,
        "vector_dim": len(batch.vectors[0]) if batch.vectors else 0,
        "head": [round(value, 6) for value in batch.vectors[0][:3]] if batch.vectors else [],
        "ok": bool(batch.vectors) and len(batch.vectors[0]) == batch.dim,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="one embedding call, then exit")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)

    from fmd.config import get_registry
    from fmd.llm.embeddings import EmbeddingError, EmbeddingsUnavailable, build_embedding_backend

    try:
        backend = build_embedding_backend(get_registry().models)
    except EmbeddingsUnavailable as exc:
        print(json.dumps({"ok": False, "reason": exc.reason, "error": str(exc)}, indent=2))
        return 1
    if args.probe:
        try:
            result = probe()
        except EmbeddingsUnavailable as exc:
            print(json.dumps({"ok": False, "reason": exc.reason, "error": str(exc)}, indent=2))
            return 1
        except EmbeddingError as exc:
            print(json.dumps({"ok": False, "reason": exc.reason, "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1

    report = measure(backend)
    print(json.dumps(report, indent=2, sort_keys=True))
    checks = report["recommendation"]["checks"]
    return 0 if all(check.get("result") in {"pass", "[not measured]"} for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
