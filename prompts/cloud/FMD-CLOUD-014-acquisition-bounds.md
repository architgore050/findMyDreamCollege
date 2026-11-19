# FMD-CLOUD-014 · acquisition-bounds · wave w6

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: a4921d28a08a1682a681a3078bdd861010deef60
BRANCH: feature/fmd-w6-acquisition-bounds-a4921d2-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.

READ FIRST:
- AGENTS.md
- docs/product-development-cycle/04-phase-3-implementation-brief.md §9
- docs/product-development-cycle/11-inference-runbook.md
- config/acquire.yaml (T0/T1/T2/T3, max_pages, guards)
- fmd/acquire/tiers.py, fmd/acquire/types.py, fmd/acquire/errors.py
- fmd/orchestrate/extract.py (how a document is acquired then extracted)
- tests/test_acquire_tiers.py, tests/test_acquire_guards.py

PROBLEM (measured): an extraction run over 28 linked documents **stalled for 25+ minutes on a single
scanned PDF**. Log ends at "RapidOCR returned empty result" with no further progress; the process was
alive, GPU idle, and no claim/extraction was written. Cause: Docling's built-in OCR runs on CPU for
image-only pages, page by page, with **no page cap and no time budget**, duplicating the purpose of
T2 OCR and blocking the entire run. We killed it manually. This is the second time a single
document has consumed the run.

SCOPE — FILES YOU OWN:
- fmd/acquire/** (tier policy, budgets, progress logging)
- config/acquire.yaml
- fmd/orchestrate/extract.py (acquisition budget plumbing only)
- tests/test_acquire_*.py, tests/test_orchestrate_extract*.py, tests/fixtures/acquire/**
- docs/product-development-cycle/11-inference-runbook.md

FORBIDDEN: fmd/extract/schemas/** fmd/extract/prompts/** fmd/extract/validate.py fmd/verify/** fmd/score/** fmd/plan/** fmd/db/** db/migrations/** apps/** infra/** .env scripts/**

ACCEPTANCE
1. **Docling OCR policy is explicit.** `T1.do_ocr` config: `false` (default on this host), `true`, or
   `auto` (only when the page has no text layer AND T2 is unavailable). With `do_ocr: false`, an
   image-only page produces no text and the document routes to **T2 if configured, else T3** with
   reason `no_text_layer` — it must NOT silently spend 25 minutes on CPU OCR.
2. **Per-document acquisition budget** (config): `max_pages` for OCR-bearing documents and
   `max_seconds` wall-clock. On breach: abort the document with `acquisition_budget_exceeded`
   (include `pages_done` / `pages_total`), persist no claims from the partial pass, and **continue
   to the next document**. The run summary counts it.
3. **Progress visibility**: log one line per N pages (config, default 10) with page index, elapsed,
   and tier. A future stall must be obvious from the log within a minute, not 25.
4. **Global run budget**: optional `max_seconds` for the whole extraction stage; when hit, stop
   cleanly, mark the remaining documents `not_attempted`, and exit 0 with a summary (no partial
   claim writes).
5. **No regression** for text-layer documents: they take T0/T1 text path, no page cap applied
   (or a generous cap), single-pass, same results as today.
6. Tests (offline, fake clock/backend):
   - OCR disabled + image-only page → `no_text_layer` → T3/T2, fast;
   - page cap honoured, remaining pages not OCR'd, document cost bounded;
   - wall-clock budget aborts the document and the run continues to the next;
   - progress line emitted every N pages;
   - global budget marks remaining docs `not_attempted`;
   - text-layer document unaffected;
   - no claims persisted from an aborted partial acquisition.
7. Docs updated: the runbook gets a table of acquisition budgets and the OCR policy, plus a note
   that T2 is the only OCR path intended on this host.

CONSTRAINTS: precision and honesty over completeness — an aborted document must be visibly aborted,
never a silent zero. No network in tests; no live model; scoped diff.

RUN (all must pass):
cd packages/pipeline && uv sync --all-extras
uv run ruff check .
uv run pytest -q
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden     (provenance must be 1.0)

OUTPUT (root, UTC): fmd-patch-w6-acquisition-bounds-a4921d2-<UTC>-01.patch / .json / .md
fmd-patch-w6-acquisition-bounds-a4921d2-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: git diff --binary --full-index --find-renames a4921d28a08a1682a681a3078bdd861010deef60...HEAD > <patch>

MANIFEST: format fmd-patch-v1, wave w6, workstream acquisition-bounds, base_sha as above,
branch feature/<patch_id>, commit_message, summary, files_changed, tests_run,
evals {"result":"pass","notes":"provenance 1.0"}, known_gaps, created_at, agent.

findings.json must list: the budget defaults chosen, the OCR policy matrix, the reason codes added,
and what a legitimate long document now loses (pages beyond the cap) — with the trade-off stated.
~~~~
