# data/compute-metrics/

Append-only JSONL compute metrics — one file per `run_id`
(`<run_id>.jsonl`), written by `fmd/obs/metrics.py` (`ComputeMetrics`).
This directory exists so the path is stable; **only this README is
committed** — every `.jsonl` record stays local (gitignored) because it is
host telemetry, not evidence.

`fmd metrics summary --run-id <id>` aggregates a run into p50/p95 duration
and tok/s by stage+device, total GPU vs CPU time, cache-hit rates, and one
ranked bottleneck line.

No applicant or document user data is recorded: `input_ref` is the
pipeline's content-hash reference (`sha256:family:prompt-version`), and the
sampler records GPU utilisation / VRAM / power and process CPU / RSS only.
Set `FMD_METRICS_DIR` to relocate, `FMD_COMPUTE_METRICS=off` to disable.
