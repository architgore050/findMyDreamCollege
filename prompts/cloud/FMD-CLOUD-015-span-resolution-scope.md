# FMD-CLOUD-015 · span-resolution-scope · wave w7

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 4e7f36c62331433ea05c261c011cdcda5eb64e01
BRANCH: feature/fmd-w7-span-scope-4e7f36c-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.

READ FIRST:
- AGENTS.md
- docs/product-development-cycle/04-phase-3-implementation-brief.md §9
- docs/product-development-cycle/11-inference-runbook.md
- fmd/extract/validate.py (resolve_quote, canonicalisation, guards, disambiguation)
- fmd/acquire/canonical.py (page anchors, bboxes)
- fmd/extract/prompts/seat_matrix/v4.md, fee_structure/v4.md (context contract)
- data/golden/** (must not regress)

PROBLEM (measured): a 17-document run produced **5 claims and 14 rejections, all `ambiguous_span`**.
Every rejected claim was correct in substance; it failed because the resolver demands **document-wide
uniqueness**. In a 100-page annual report, "A" grade, "2024", or a programme name appears many times,
so a correct value is discarded. That is our design flaw, not the document's.

SCOPE — FILES YOU OWN:
- fmd/extract/validate.py (resolution scope only)
- fmd/extract/schemas/** (only if a scope field must be surfaced)
- fmd/extract/prompts/** (new version only if the model must supply finer context; keep old)
- fmd/verify/** (pass scope to the verifier; verdict semantics unchanged)
- fmd/acquire/canonical.py (expose section/page boundaries for scoping)
- tests/test_extract_validate.py, tests/test_extract_context.py, tests/fixtures/**
- docs/product-development-cycle/11-inference-runbook.md (resolution section)

FORBIDDEN: fmd/score/** fmd/fetch/** fmd/plan/** fmd/ingest/** fmd/db/** db/migrations/** apps/** infra/** .env scripts/**

ACCEPTANCE
1. **Scoped resolution ladder**, narrowest-first:
   a. globally unique → `resolution_scope: document`;
   b. else unique within the claim's **row context** (existing behaviour) → `row`;
   c. else unique within its **page** → `page`;
   d. else unique within its **section/heading block** → `section`;
   e. else `ambiguous_span` → UNKNOWN.
   The chosen scope and the page number are recorded in the claim qualifier.
2. **Never first-match.** If two occurrences share the narrowest scope that would otherwise resolve,
   the claim is UNKNOWN — no positional tie-breaks, no "closest to top".
3. **Table rows still win.** A row-scoped quote resolves at `row` even if a page-level match exists,
   because the row is the semantic unit.
4. **Provenance stays absolute**: `char_start/char_end` remain offsets into the canonical document
   text; scope affects *uniqueness*, never offsets.
5. **Dedupe interaction**: the same predicate/value resolving on different pages is NOT
   automatically one claim (it may be a genuine per-page repetition, e.g. a header) — but the same
   predicate/value within the same row or section is. State the rule you implement and test it.
6. **Verifier receives the scope** so it does not treat a page-scoped quote as suspicious.
7. Tests offline:
   - phrase repeated on pages 2, 5, 9 but unique on page 5 → resolves with `scope=page`, page recorded;
   - same phrase twice on one page with no row context → UNKNOWN;
   - row context resolves even when the phrase appears on other pages;
   - section-scoped case (heading + unique-in-section);
   - offsets unchanged and quotes resolve against the full canonical text;
   - golden eval provenance still 1.0;
   - a regression test using a real repeated phrase from a long annual report fixture.
8. Docs updated with the ladder and the dedupe rule.

CONSTRAINTS: precision first; widening scope must never produce a wrong-attribution claim. If a
scope-based resolution cannot be made safe for a field family, keep UNKNOWN and say so. No network;
no model; scoped diff.

RUN (all must pass):
cd packages/pipeline && uv sync --all-extras
uv run ruff check .
uv run pytest -q
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden     (provenance must be 1.0)

OUTPUT (root, UTC): fmd-patch-w7-span-scope-4e7f36c-<UTC>-01.patch / .json / .md
fmd-patch-w7-span-scope-4e7f36c-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: git diff --binary --full-index --find-renames 4e7f36c62331433ea05c261c011cdcda5eb64e01...HEAD > <patch>

MANIFEST: format fmd-patch-v1, wave w7, workstream span-resolution-scope, base_sha as above,
branch feature/<patch_id>, commit_message, summary, files_changed, tests_run,
evals {"result":"pass","notes":"provenance 1.0"}, known_gaps, created_at, agent.

findings.json must list: the ladder, the dedupe rule, any family where scoping stayed UNKNOWN,
and the expected change in claim yield on long documents (with the reasoning, not a guess).
~~~~
