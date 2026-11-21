# FMD-CLOUD-020 · batched acquire/extract · wave w10

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege). You have no GPU and no
model server. Everything GPU-dependent must be exercised through fakes and dry runs; the operator runs
the real pass locally afterwards.

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 68a593df2a3c385d59b1b821e4373342812573de
BRANCH: feature/fmd-w10-batched-acquire-68a593d-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.
Clone the repo, `git checkout -b <branch> 68a593d...`, and verify `git rev-parse HEAD` equals the base.

READ FIRST
- AGENTS.md (invariants, layout, test rules — obey them)
- docs/product-development-cycle/11-inference-runbook.md §6 (measured VRAM table; it is why this patch exists)
- docs/product-development-cycle/04-phase-3-implementation-brief.md §6/§11
- fmd/acquire/tiers.py (Docling conversion, device selection, artifact guards, OOM handling)
- fmd/extract/pdf.py, fmd/extract/pipeline.py (chunking, per-document flow, engine provenance)
- fmd/orchestrate/*.py (stage pattern, RunSummary)
- fmd/cli.py (`extract`, `verify`, `score`, `crawl`)
- fmd/db/repositories.py, fmd/store/** (content-addressed object store)
- config/acquire.yaml, config/models.yaml (do not change model settings)
- scripts/serve_local.sh (start|stop|status|probe)

PROBLEM (measured on the operator's machine, 2026-11-20)
Extraction is a single fused loop: for each document it runs Docling T1 conversion on CPU, then calls
the LLM. Consequences:
- Docling is 2–6× faster on CUDA (0.166 vs 1.31 s/page on a 161-page report) but CUDA OOMs while the
  26B LLM holds 64k KV on the GPU (589 MiB free after CUDA context; Docling needs ~1.0–1.3 GiB).
- So acquisition runs on CPU while the GPU idles between LLM calls (observed 6 W during a run, all cores
  at 100%).
- There is no parse cache, so acquisition and extraction cannot be split: re-running extraction
  re-converts every document.

REQUIRED SOLUTION — phase split with a persistent parse cache, batched by pages.

A. Parse cache (the enabling piece)
1. Persist Docling's T1 parse in a form sufficient to reconstruct the object the extractor consumes
   (pages → blocks with text/label/bbox, tables, outline/section info), as versioned JSON.
2. Key = (snapshot sha256, tier, policy digest, parse-schema version). **Device is not part of the key**:
   a CUDA parse is reusable by a CPU run and vice versa. A policy/schema change invalidates; a device
   change must not.
3. Store the payload content-addressed (reuse the existing object store) plus a lookup table
   `document_parses` (migration `0011`, forward-only) mapping key → object sha, pages, parser, created_at,
   and the policy digest. Do not edit migrations 0001–0010.
4. Lookup before conversion. Cache hit ⇒ **zero** Docling invocations (test with a counting fake
   converter). Corrupt, truncated, or schema-mismatched cache ⇒ reconvert and record a reason; never
   crash, never guess.
5. Writes are atomic (temp object → move, then row). An interrupted write must never be read back.
6. Live-convert and cached-parse paths must produce **identical extraction output** for the same document.
   Prove it with a committed parse fixture: extract from a mocked live conversion and from the cache,
   assert identical claim tuples and identical spans.
7. T2/OCR output: out of scope for this patch (do not change OCR); note it as a known gap. The cache
   covers T1 only.

B. Stage split (CLI)
8. `fmd acquire [--limit N] [--pages N] [--device cpu|cuda] [--dry-run]` converts documents and writes
   the cache, and makes **zero LLM calls** (assert with a mock router that would raise on call).
   Selection order = the same priority the extractor uses today.
9. `fmd extract [--from-cache] [--no-acquire] [--limit N] [--dry-run]`:
   - default behaviour unchanged (acquire inline if no cached parse);
   - `--from-cache` uses only cached parses and is **guaranteed to invoke no Docling**;
   - `--no-acquire` with a missing parse records `missing_parse_cache` for that document and continues,
     with a non-zero exit only if nothing was extractable;
   - JSON summary gains `parsed`, `cache_hits`, `pages_converted`, `llm_calls` (existing keys preserved).
10. `--dry-run` for both stages prints the plan (documents, known pages, expected cache hits) and writes
    nothing, with the model URL unreachable.

C. Batched orchestrator
11. `fmd extract --batched [--batch-pages 100] [--dry-run] [--keep-server] [--server-script PATH]`:
    per cycle — ensure server stopped → acquire one page-batch on CUDA → ensure server started and
    `/health` is ready (poll with timeout) → extract that batch from cache → repeat.
12. **Batch sizing is by pages, and the limit is a soft one**: accumulate documents until the converted
    page total reaches `batch_pages`, and always process at least one document even if it alone exceeds
    the limit. Default `100` (break-even measured at ~20–25 pages/cycle; 100 leaves margin).
13. Server control is injected, not hardcoded: an interface with `stop()`, `start()`, `wait_ready()`;
    the default adapter shells out to `scripts/serve_local.sh` from `config/acquire.yaml`. Tests use a
    fake controller and assert call order and that the server is **stopped at the end** (try/finally),
    even on conversion failure, LLM failure, or Ctrl-C (SIGINT handler).
14. Failure isolation: one document failing to convert is recorded and the batch continues; a document
    whose parse is cached but whose extraction failed is retried on the next run; a server start failure
    aborts the cycle with a clear error and leaves the server stopped. Never leave a model server
    running on any exit path unless `--keep-server`.
15. Resume/idempotency: re-running skips documents that already have a cached parse **and** a completed
    extraction. Interrupting a cycle loses only in-flight work; persisted claims and parses stay.
16. Degraded mode: if server control is unavailable (`FMD_SERVER_CONTROL=off` or no script), fall back to
    CPU acquisition + inline extraction and say so in the summary, rather than failing.

D. Config, docs, visibility
17. `config/acquire.yaml` gains a validated `batch` block: `pages`, `device`, `server_script`,
    `keep_server`, `poll_timeout_s`, plus cache `schema_version`. Loader rejects unknown keys, zero/negative
    page limits, and a device outside the known set.
18. Minimal per-batch line on stderr/stdout (no logging framework): cycle index, documents, pages,
    seconds, pages/min, cache hits, device. One line per cycle — not a log pipeline.
19. Docs in the same patch: runbook §6 gets the batched command sequence and the break-even rule
    (~20–25 pages/cycle) with the measured constants; the implementation brief gets one paragraph.

E. Tests (offline; no network, no model download, no real Docling conversion, no GPU)
- cache hit ⇒ zero converter calls; device change ⇒ still a hit; policy/schema bump ⇒ reconvert
- corrupt/truncated/mismatched cache ⇒ reconvert with recorded reason, no crash
- atomic write: a failed write leaves no usable cache row
- live-vs-cache extraction equivalence on the committed parse fixture (identical claims + spans)
- `acquire` makes zero LLM calls; `extract --from-cache` makes zero Docling calls
- `--no-acquire` with missing parse ⇒ `missing_parse_cache`, run continues
- soft page limit: 100-page limit with a 434-page document processes it alone
- fake controller call order (stop → acquire → start → wait_ready → extract) and server stopped on
  converter exception, on LLM exception, and on SIGINT
- batch dry-run plan with an unreachable model URL, writing nothing
- resume skips completed documents; retries failed extraction
- JSON summary keys and non-zero exit only when nothing extractable

F. Explicitly out of scope
- Do not change extraction prompts, prompt versions, chunking, scoring, weights, or predicates.
- Do not change OCR/T2, or `config/models.yaml` model settings.
- Do not add a metrics/logging system, no Prometheus, no telemetry.
- No new network access, no new model downloads, no paid APIs.

SCOPE — files you own: `packages/pipeline/fmd/acquire/**`, `fmd/extract/**`, `fmd/orchestrate/**`,
`fmd/cli.py`, `fmd/config/**`, `fmd/db/repositories.py` + `db/migrations/0011_*.sql`,
`config/acquire.yaml`, `scripts/batch_extract.sh` (new, thin wrapper over the CLI), tests and fixtures,
`docs/product-development-cycle/11-inference-runbook.md`, `docs/product-development-cycle/04-…-brief.md`
(one paragraph). Nothing else.

RUN (all must pass; note in the report if the DB is unavailable, but it was available in CI for w9):
cd packages/pipeline && uv sync --all-extras
uv run ruff check .
uv run pytest -q
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q      (if a DB is present)
uv run python -m fmd.cli eval --golden ../../data/golden                   (provenance must be 1.0)
FMD_LOCAL_LLM_URL=http://127.0.0.1:9/v1 uv run python -m fmd.cli extract --batched --dry-run --limit 5
  (must succeed with no model server, print the cycle plan, write nothing)

OUTPUT (repo root, UTC timestamp): 
fmd-patch-w10-batched-acquire-68a593d-<UTC>-01.patch
fmd-patch-w10-batched-acquire-68a593d-<UTC>-01.json
fmd-patch-w10-batched-acquire-68a593d-<UTC>-01.md
fmd-patch-w10-batched-acquire-68a593d-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: git diff --binary --full-index --find-renames 68a593df2a3c385d59b1b821e4373342812573de...HEAD > <patch>

MANIFEST: format fmd-patch-v1, wave w10, workstream batched-acquire, base_sha as above,
branch feature/<patch_id>, commit_message, summary, files_changed, tests_run,
evals {"result":"pass","notes":"provenance 1.0"}, known_gaps, created_at, agent.
Report the concrete new test names and their outcomes, and any acceptance item you could only verify
by fake.

findings.json must state: the cache key and invalidation rules, the atomic-write scheme, the batch-sizing
rule, the controller call order, the exact operator command sequence for a real run, the measured
constants the defaults are based on (CUDA 0.166–0.9 s/page vs CPU 1.0–3.8 s/page; ~20–25 s per server
restart; break-even ~20–25 pages), the expected cycle count for a ~1,000-page backlog, and known gaps
(T2 not cached, GPU path validated only via fakes, Docling library-version invalidation policy).
~~~~
