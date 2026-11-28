# FMD-CLOUD-030-gpu-phase-metrics · automatic VRAM phases + compute metrics · wave w16

~~~~
You are the one-shot cloud coding agent implementing automatic resource phasing and compute metrics.

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 6f95fea769bfd78ec4b17d8d94c5160131b4847a
All output — English-only.

BRANCHING
1. `git switch -c v1/gpu-phase-metrics` from main; verify `git rev-parse HEAD`; if it
   differs from BASE SHA, set BASE SHA := `git rev-parse HEAD`.
2. `git push origin v1/gpu-phase-metrics` immediately.
3. Commit + push after every logical step. Subagents allowed. End: PR → main.

WHY
The 50-college E2E run spent ~2 h doing CPU Docling acquisition (21 then 2 parses
per ~100 min) while a 26B server sat on the GPU, and its logs only said "still
running". The repo ALREADY has the correct pattern in `fmd/orchestrate/batched.py`
(`BatchedRunner`: stop server → acquire one page-batch on CUDA → start server →
extract from cache; `ServerController` protocol + `ScriptServerController` +
degraded CPU mode). That capability is trapped inside `fmd extract --batched` and
a standalone `fmd acquire` still defaults to CPU. Runs must work end-to-end on
their own: no manual server stop/start, no custom scoped CPU runner.

READ FIRST
- packages/pipeline/fmd/orchestrate/batched.py (ServerController, ScriptServerController,
  NullServerController, BatchedRunner._run_cycle/_run_cycles/degraded path)
- packages/pipeline/fmd/orchestrate/acquire.py (device pinning / AcquisitionStage)
- packages/pipeline/fmd/cli.py (extract --batched/--batch-pages/--keep-server/--device;
  acquire parser; worker command)
- packages/pipeline/fmd/config/models.py BatchConfig / config/acquire.yaml batch block
- packages/pipeline/fmd/llm/router.py (ModelRouter.call + LLMCallSink: latency/tokens/cost)
- docs/product-development-cycle/11-inference-runbook.md §6 (measured VRAM: CUDA Docling
  OOMs alongside 64k KV; 2–6× faster per page when time-multiplexed)
- tests for batched/acquire/router

TASK A — reusable automatic GPU phase manager
1. New `fmd/orchestrate/gpu_phases.py`: extract/refactor the server lifecycle from
   `BatchedRunner` into a reusable `ResourcePhaseManager` (or `GpuPhaseManager`):
   - `acquire_phase(device)` context manager: if device resolves to cuda and a
     server controller is configured, stop server → pin FMD_DOCLING_DEVICE/
     FMD_OCR_DEVICE=cuda → yield → finally restart + wait_ready; if no controller
     or insufficient VRAM, degrade to CPU with an explicit recorded reason and
     never fail the run purely for missing GPU control (match current degraded mode).
   - Reuse the existing `ServerController`/`ScriptServerController`, not a second
     implementation. Keep all exit paths safe (try/finally).
2. `fmd extract --batched` uses the shared manager; current tests/behavior stay green.
3. `fmd acquire` gains `--device auto|cpu|cuda` and `--phased` (config-driven default:
   phased CUDA when a controller exists); when phased, the server lifecycle is
   automatic and zero LLM calls remains true during acquisition. Add a concise
   per-document progress line (id, pages, device, seconds, cache hit) with flushed
   output so logs say what happened, not only "still running".
4. `fmd worker`/pipeline path uses the phased batched route automatically when
   config+GPU controller are available (config flag, default on); no manual care.

TASK B — compute metrics (logging that answers "what is the bottleneck?")
1. New `fmd/obs/metrics.py`: append-only JSONL records under `data/compute-metrics/`
   (gitignored except a README; versioned run_id). Record per stage/phase/task:
   run_id, stage, phase, device, model/provider, duration_ms, documents, pages,
   cache_hits, tokens_in/out, prefill tok/s, decode tok/s, total tok/s, batch size,
   OCR engine/device, gpu_util, vram_used/peak, power_draw, cpu_percent, rss, status.
2. `ResourceSampler` context manager/thread: sample nvidia-smi (utilization.gpu,
   memory.used, power.draw) + ps (cpu%, rss) every 5s around each phase.
   Degrade gracefully when nvidia-smi is absent (record null, never crash).
3. Integrate: `ModelRouter.call` (surface existing sink latency/tokens/cost per
   stage+input_ref), Docling/acquisition conversions, OCR tiers, embedding provider,
   `BatchedRunner` cycles, `ResourcePhaseManager` phases. No user data in metrics.
4. New `fmd metrics` CLI: `fmd metrics summary --run-id <id>` → table/JSON with
   p50/p95 duration and tok/s by stage+device, total GPU vs CPU time, cache-hit
   rates, and a ranked bottleneck line. Add tests with a fake sampler (no GPU).
5. Improve `run_logged.sh` heartbeat to include GPU util/VRAM/power + CPU/RSS when
   tools exist, without changing its START/END/`.status` contract.

TASK C — docs/tests/gates
- Update runbook §6, batch docs, and the free-stack docs for automatic phases +
  metrics; state that no manual server stop/start is needed.
- Tests: phase stop/start ordering + failure/degraded paths, acquire --phased,
  router metric emission, sampler fakes, metrics CLI, run_logged heartbeat,
  existing batched/acquire suites green.
- Gates: ruff, offline pytest, DB pytest (report counts), golden provenance 1.0,
  verdict-live replay + dry-runs.

SCOPE: fmd/orchestrate/{batched,acquire,gpu_phases}.py, fmd/cli.py,
fmd/config/{models,loader}.py, config/acquire.yaml, fmd/llm/router.py,
fmd/obs/metrics.py (new), docs, tests, scripts/run_logged.sh.
Forbidden: paid SDKs/APIs, real .env, apps/infra/db migrations, fmd/extract/**

REPORT (chat ≤ 400 words): changes, test/gate counts, PR URL, and exactly what the
operator must do (ideally nothing beyond a normal command) for a phased run.
Do NOT claim host GPU validation from the sandbox; parent validates on the RTX 5080.
~~~~
