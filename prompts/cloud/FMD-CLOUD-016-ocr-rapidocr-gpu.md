# FMD-CLOUD-016 · ocr-rapidocr-gpu · wave w7

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 4e7f36c62331433ea05c261c011cdcda5eb64e01
BRANCH: feature/fmd-w7-ocr-rapidocr-gpu-4e7f36c-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.

READ FIRST:
- AGENTS.md
- docs/product-development-cycle/14-deferred-items.md (D18 — OCR must ship; this implements it)
- docs/product-development-cycle/04-phase-3-implementation-brief.md §9
- docs/product-development-cycle/11-inference-runbook.md (T2/T2b, budgets, bbox granularity)
- config/acquire.yaml (T2 Qwen3-VL, T2b PaddleOCR-VL, guards, device policy)
- fmd/acquire/tiers.py (T2 contract, tier selection, hotfix FMD_DOCLING_OCR)
- fmd/acquire/types.py, fmd/acquire/guards.py
- infra/ocr/** (existing OCR service contract)

CONTEXT / DECISION (from the operator, treat as fixed):
- **RapidOCR (PaddleOCR-family models) on GPU is the primary OCR engine for T2.** It is
  purpose-built for detection+recognition, far cheaper per page than a VLM, and the operator
  prefers it. Run it on CUDA via onnxruntime-gpu; CPU is the bounded fallback.
- **Qwen3-VL is demoted to an evidence-only fallback** for pages whose Devanagari output fails a
  measured gate. It must not be the default and must not be silently used.
- **One engine per page.** Engine name/version/model id is recorded in the claim qualifier so the
  transcript is reproducible. Never leave Docling's implicit RapidOCR instance unbounded.
- LLM work stays on **Gemma 4 26B A4B** (already locked); do not introduce other LLMs here.

SCOPE — FILES YOU OWN:
- fmd/acquire/tiers.py (T2 backend selection), fmd/acquire/types.py (engine metadata)
- new fmd/acquire/rapidocr_backend.py
- config/acquire.yaml (T2 backend, device, models, budgets)
- infra/ocr/** (service contract if the in-process path needs a shape change)
- tests/test_acquire*.py, tests/fixtures/acquire/**
- new scripts/ocr_validate.py (small-batch measurement harness)
- docs/product-development-cycle/11-inference-runbook.md (OCR section), 14-deferred-items.md (D18 status)

FORBIDDEN: fmd/extract/** fmd/verify/** fmd/score/** fmd/plan/** fmd/fetch/** fmd/db/** db/migrations/** apps/** .env

ACCEPTANCE
1. **RapidOCR on GPU** as `T2.backend: rapidocr` (default on this host), with:
   - `device: cuda` when onnxruntime-gpu and a CUDA provider are available, else `cpu` with a
     recorded reason;
   - detection + recognition models suitable for **Devanagari/Hindi and Latin**, ids recorded in
     config (no hardcoded paths);
   - block-level bboxes (`bbox_granularity: block`), page number, and engine metadata
     (`engine`, `engine_version`, `model_ids`) persisted on the claim qualifier.
2. **Bounded**: `max_pages`, `max_seconds`, per-page progress line, and the abort semantics from
   D18 (`acquisition_budget_exceeded`, no partial claims, continue to next document). A stall must
   be visible within one page.
3. **Fail visibly**: GPU unavailable → CPU (recorded); models missing → `TierUnavailable` naming
   the prefetch command; OCR empty on an image-only page → `no_text_layer` → T3, never silent.
4. **Selection matrix in config**: `T2.backend ∈ rapidocr | qwen3vl | paddleocr`. Qwen3-VL stays
   selectable but is not the default. The matrix and its cost/quality rationale go in the runbook.
5. **`scripts/ocr_validate.py`** — small-batch harness: run N documents (default 5, cap 10), emit
   pages/min, pages OCR'd, chars/page, empty-page rate, mean confidence if available, and a
   `--ground-truth` mode computing CER and chrF++ against a transcript fixture. It must refuse to
   run on more than 10 documents without `--i-know`.
6. **Repetition/entropy guard still applies** to OCR output (existing T2 guard), and abort still
   prevents publishing claims from a looping decoder.
7. Tests offline (mock onnx session): backend selection; CUDA-unavailable → CPU with reason;
   missing models → TierUnavailable; engine metadata on claims; page/time budget; progress line;
   empty page → no_text_layer; guard trips on synthetic 71× repetition; config matrix validation
   rejects an unknown backend.
8. Runbook: how to serve/verify RapidOCR on GPU, VRAM expectation, how to run the small-batch
   validation, and the explicit statement that shipping requires OCR enabled and measured.
9. Do NOT download models in tests. Do NOT change LLM profiles.

CONSTRAINTS: no network in tests; no secrets; scoped diff; visibility is non-negotiable — every
scanned page must record engine + version + granularity.

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

findings.json must list: the backend matrix, model ids chosen (with licence), the measured-or-expected
pages/min, what the guard protects, and what remains unmeasured until the small-batch run on real
scans (e.g. Devanagari CER) — with the exact command to measure it.
~~~~
