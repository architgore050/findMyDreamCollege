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

CONTEXT / DECISION (treat as fixed):
- **RapidOCR is the RUNNER, not the model.** It is an ONNX-runtime wrapper. The models it loads are
  **PaddleOCR PP-OCRv5 server det+rec**, including a Devanagari/Hindi recognition model and a Latin
  one. Pin model ids in config; no hardcoded paths.
- **Primary path: PP-OCRv5 via RapidOCR on onnxruntime-gpu (CUDA)**, CPU as bounded fallback.
  Purpose-built detection+recognition is far cheaper per page than a VLM; a VLM on a 200-page scan
  is exactly what stalled us before.
- **Table structure is NOT OCR's job.** Docling/TableFormer keeps it (already in T1).
- **Qwen3-VL-8B stays fully wired and selectable**, as the Devanagari fallback and as a T2 backend
  option. It is not removed and not silently used. PaddleOCR-VL stays as the alternate full
  pipeline. No engine is deleted by this patch.
- **One engine per page, one transcript.** Engine name, version and model ids recorded in the
  claim qualifier, so a mixed-engine corpus is still reproducible.
- **This preference is provisional until measured.** Ship `scripts/ocr_validate.py --ground-truth`
  and the decision rule below; do not present PP-OCRv5 as proven more accurate than Qwen3-VL.
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
9. **Validation gate + decision rule (required deliverable).** `scripts/ocr_validate.py --ground-truth`
   must compare PP-OCRv5 and Qwen3-VL on the same 5–10 real scans, reporting CER, chrF++ and
   pages/min per engine, and exiting non-zero if fewer than 3 documents had ground truth. Document
   the committed rule: if PP-OCRv5 is within ~5 chrF++ of Qwen on Devanagari, keep PP-OCRv5
   everywhere; if it loses by more, route Devanagari pages to Qwen while Latin stays on PP-OCRv5 —
   with the per-page engine recorded so the corpus stays reproducible. Do not claim a winner in the
   patch; the numbers decide.
10. Do NOT download models in tests. Do NOT change LLM profiles.

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
