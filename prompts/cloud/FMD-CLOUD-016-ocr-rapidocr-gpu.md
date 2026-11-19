# FMD-CLOUD-016 · ocr-rapidocr-gpu · wave w7 (FINAL)

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 4e7f36c62331433ea05c261c011cdcda5eb64e01
BRANCH: feature/fmd-w7-ocr-rapidocr-gpu-4e7f36c-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.

READ FIRST:
- AGENTS.md
- docs/product-development-cycle/14-deferred-items.md (D18 — OCR must ship)
- docs/product-development-cycle/04-phase-3-implementation-brief.md §9
- docs/product-development-cycle/11-inference-runbook.md (T2/T2b, budgets, bbox granularity)
- config/acquire.yaml (T2 Qwen3-VL, T2b PaddleOCR-VL, guards, device policy)
- fmd/acquire/tiers.py (T2 contract, tier selection, FMD_DOCLING_OCR hotfix)
- fmd/acquire/types.py, fmd/acquire/guards.py
- infra/ocr/**

CONTEXT / DECISION (treat as fixed):
- **RapidOCR is the RUNNER, not the model.** It is an ONNX-runtime wrapper; the models it loads are
  **PaddleOCR PP-OCRv5 server det+rec**, including a Devanagari/Hindi recognition model and a Latin
  one. Pin model ids in config; no hardcoded paths.
- **Primary path: PP-OCRv5 via RapidOCR on onnxruntime-gpu (CUDA)**, CPU as bounded fallback.
  Purpose-built detection+recognition is far cheaper per page than a VLM; a VLM over a 200-page
  scan is exactly what stalled us.
- **Table structure is NOT OCR's job.** Docling/TableFormer keeps it (already in T1).
- **Qwen3-VL-8B stays FULLY WIRED and selectable** — as the Devanagari fallback and as a T2
  backend option. It is not removed, not demoted out of the codebase, and never silently used.
  PaddleOCR-VL stays as the alternate full pipeline. No engine is deleted by this patch.
- **One engine per page, one transcript.** Engine name, version and model ids recorded in the
  claim qualifier, so a mixed-engine corpus stays reproducible.
- **This preference is PROVISIONAL UNTIL MEASURED.** Do not present PP-OCRv5 as proven more
  accurate than Qwen3-VL. Ship the measurement and the decision rule; the numbers decide.
- LLM work stays on **Gemma 4 26B A4B** (already locked); do not introduce other LLMs here.

SCOPE — FILES YOU OWN:
- fmd/acquire/tiers.py (T2 backend selection), fmd/acquire/types.py (engine metadata)
- new fmd/acquire/rapidocr_backend.py
- config/acquire.yaml (T2 backend, device, models, budgets)
- infra/ocr/** (only if the service contract must change)
- tests/test_acquire*.py, tests/fixtures/acquire/**
- new scripts/ocr_validate.py
- docs/product-development-cycle/11-inference-runbook.md (OCR section)
- docs/product-development-cycle/14-deferred-items.md (D18 status only)

FORBIDDEN: fmd/extract/** fmd/verify/** fmd/score/** fmd/plan/** fmd/fetch/** fmd/db/** db/migrations/** apps/** .env

ACCEPTANCE
1. **RapidOCR on GPU** as `T2.backend: rapidocr` (default on this host):
   - `device: cuda` when onnxruntime-gpu + a CUDA provider is available, else `cpu` with a recorded reason;
   - PP-OCRv5 server det+rec models for **Devanagari/Hindi and Latin**, ids in config;
   - block-level bboxes, page number, engine metadata (`engine`, `engine_version`, `model_ids`)
     persisted on the claim qualifier.
2. **Bounded**: `max_pages`, `max_seconds`, per-page progress line, and D18 abort semantics
   (`acquisition_budget_exceeded`, no partial claims, continue to the next document). A stall must
   be visible within one page.
3. **Fail visibly**: GPU unavailable → CPU (recorded); models missing → `TierUnavailable` naming
   the prefetch command; OCR empty on an image-only page → `no_text_layer` → T3.
4. **Selection matrix in config**: `T2.backend ∈ rapidocr | qwen3vl | paddleocr`. Qwen is never the
   default, never removed. Rationale + cost matrix in the runbook.
5. **`scripts/ocr_validate.py --ground-truth`** (required deliverable): runs PP-OCRv5 and Qwen3-VL
   over the SAME 5–10 real scans; reports CER, chrF++ and pages/min per engine, per document and
   aggregate; exits non-zero if fewer than 3 documents have ground truth; supports `--engine` and
   `--limit` and refuses >10 documents without `--i-know`.
6. **Committed decision rule** (document it; do not claim a winner in the patch):
   - PP-OCRv5 within ~5 chrF++ of Qwen on Devanagari → keep PP-OCRv5 everywhere;
   - PP-OCRv5 loses by more → route Devanagari pages to Qwen, Latin stays on PP-OCRv5, per-page
     engine recorded.
7. **Repetition/entropy guard** still applies to OCR output; aborted OCR never publishes claims.
8. **Devanagari capability must be proven, not assumed**: add a test that the configured
   Devanagari recognition model id is present in the model manifest and that a mock Devanagari
   input routes to it (script char class), plus a documented command to run the real check on GPU.
9. Tests offline (mock onnx session): backend selection; CUDA-unavailable → CPU with reason;
   missing models → TierUnavailable; engine metadata on claims; page/time budget; progress line;
   empty page → no_text_layer; guard trips on synthetic 71× repetition; config rejects unknown backend.
10. Runbook: serving/verifying PP-OCRv5 on GPU, VRAM expectation, the small-batch validation, and
    the explicit statement that shipping requires OCR enabled and measured.

CONSTRAINTS: no network in tests; no model downloads in tests; no secrets; scoped diff; visibility
is non-negotiable — every scanned page records engine + version + granularity.

RUN (all must pass):
cd packages/pipeline && uv sync --all-extras
uv run ruff check .
uv run pytest -q
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden     (provenance must be 1.0)

OUTPUT (root, UTC): fmd-patch-w7-ocr-rapidocr-gpu-4e7f36c-<UTC>-01.patch / .json / .md
fmd-patch-w7-ocr-rapidocr-gpu-4e7f36c-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: git diff --binary --full-index --find-renames 4e7f36c62331433ea05c261c011cdcda5eb64e01...HEAD > <patch>

MANIFEST: format fmd-patch-v1, wave w7, workstream ocr-rapidocr-gpu, base_sha as above,
branch feature/<patch_id>, commit_message, summary, files_changed, tests_run,
evals {"result":"pass","notes":"provenance 1.0"}, known_gaps, created_at, agent.

findings.json must list: backend matrix, model ids with licences, measured-or-expected pages/min,
what the guard protects, the exact Devanagari verification command, and what remains unmeasured
until the small-batch run on real scans.
~~~~
