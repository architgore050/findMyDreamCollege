# FMD-CLOUD-018 · section-structure · wave w8

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: c5cf4ad2eab479886dfdf28e61508ef64c1198cf
BRANCH: feature/fmd-w8-section-structure-c5cf4ad-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.

READ FIRST:
- AGENTS.md
- docs/product-development-cycle/04-phase-3-implementation-brief.md §9 (extraction), §10 (planning)
- docs/product-development-cycle/11-inference-runbook.md (acquisition, section scope)
- fmd/acquire/canonical.py (canonical text + page/bbox anchors)
- fmd/acquire/tiers.py (Docling path; DoclingDocument items have labels such as section_header/title and page provenance)
- fmd/extract/pipeline.py (chunking, prompt assembly, claim assembly)
- fmd/extract/validate.py (resolution ladder: document → row → page → section → UNKNOWN)
- fmd/plan/rules.py (heading signals already used for classification)
- data/golden/**

MOTIVATION (operator proposal, endorse and implement):
A parsed PDF has a natural outline. We currently throw that structure away and flatten everything
into canonical text, then ask the resolver to be "unique" within a heuristic +/-200 char window or a
table block. That is why a correct claim gets rejected when the same phrase appears once per page
across 11 pages ("probably under section 4.2 Fees"). Keeping the document's heading tree gives us:
- real section boundaries for resolution scope,
- a breadcrumb ("Admissions › Fees › Hostel") to give the model when reading a chunk,
- section-aligned chunk boundaries instead of arbitrary token windows,
- a stable structure for the planner and for review UIs.

GOAL: extract and persist a **document section tree**, use it everywhere the pipeline currently
guesses at structure.

SCOPE — FILES YOU OWN:
- fmd/acquire/** (outline extraction alongside canonical text)
- db/migrations/0009_document_sections.sql (new; never edit 0001–0008)
- fmd/db/repositories.py (sections repo only)
- fmd/extract/pipeline.py (breadcrumb injection + chunk boundaries)
- fmd/extract/validate.py (real section scope)
- fmd/extract/prompts/** (new versions only if the prompt text changes; keep old)
- fmd/plan/rules.py (consume the same structure; no precedence change)
- config/acquire.yaml (heading-detection thresholds)
- tests/test_acquire*.py, tests/test_extract_*.py, tests/test_plan*.py, tests/fixtures/**
- docs/product-development-cycle/11-inference-runbook.md, 04-…-brief.md (§9 short paragraph)

FORBIDDEN: fmd/verify/** fmd/score/** fmd/fetch/** fmd/ingest/** apps/** infra/** .env scripts/** config/sources.yaml

ACCEPTANCE
1. **Outline extraction** (T1/Docling first, T0 text as fallback):
   - prefer Docling structural labels (`section_header`, `title`, `page_header`) with their page and bbox;
   - fallback heuristics for born-digital text: numbered headings (`1.`, `1.2`, `(iii)`), short
     ALL-CAPS lines, and Devanagari headings; each heuristic must have a test and a false-positive
     guard (e.g. a sentence that merely begins "1. The institute ..." is NOT a heading);
   - produce a tree with `level`, `title`, `start`, `end` (char offsets into canonical text), `page`,
     `bbox`; levels normalised to 1..N (no gaps).
2. **Persistence**: `document_sections` table keyed by snapshot/document with parent_id, level, title,
   char range, page, bbox, engine/version of the extractor. Migration 0009 only; idempotent on re-run.
3. **Section lookup API**: given `[char_start, char_end)`, return the heading breadcrumb (list of
   titles, root→leaf) and the section node id. Must be O(log n)-ish, not a linear scan per claim.
4. **Resolver integration**: `section` scope uses real section boundaries instead of the +/− heuristic;
   ladder unchanged (document → row → page → section → UNKNOWN). Record `section_path` (breadcrumb) in
   the claim qualifier alongside `resolution_scope`.
5. **Prompt integration**: each extraction window carries its breadcrumb as metadata
   (`document_sections: [...]`), not as document text. Bump the prompt version only if the prompt text
   changes; the breadcrumb must never be quotable as evidence (test that a claim cannot resolve to a
   heading string used as context).
6. **Chunking integration**: prefer splitting at section boundaries; if a section is larger than the
   window budget, split inside it and carry the same breadcrumb. Short documents remain single-pass.
7. **Planner integration**: planner may consume heading titles as classification signals (it already
   has heading signals) via the persisted tree, without changing precedence.
8. **No headings case**: documents with no detectable structure produce an empty tree; resolution falls
   back to the existing heuristic and says `section: heuristic` in diagnostics. Never fail a document
   because headings were not found.
9. Tests (offline):
   - nested headings (1 / 1.1 / 1.1.1) with correct ranges and breadcrumbs;
   - false positive "1. The institute …" not treated as a heading;
   - Devanagari heading detected;
   - breadcrumb lookup at an offset in the middle of a subsection;
   - a repeated phrase unique within a section resolves at `section` scope with `section_path` recorded;
     the same phrase in two sections stays UNKNOWN;
   - heading strings cannot be used as claim quotes;
   - chunk splits at a heading boundary, not mid-paragraph, when possible;
   - no-headings document degrades to heuristic with a diagnostic;
   - golden eval provenance stays 1.0.
10. Docs: the runbook describes the outline, its persistence, and the resolution ladder with real
    sections; brief §9 gains one paragraph on structure retention.

CONSTRAINTS: precision first; a wrong attribution is worse than UNKNOWN. No network; no model; no
schema edits to applied migrations; scoped diff. If Docling's labels are unreliable for a document
type, degrade to heuristics and record which extractor produced the tree.

RUN (all must pass):
cd packages/pipeline && uv sync --all-extras
uv run ruff check .
uv run pytest -q
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden     (provenance must be 1.0)

OUTPUT (root, UTC): fmd-patch-w8-section-structure-c5cf4ad-<UTC>-01.patch / .json / .md
fmd-patch-w8-section-structure-c5cf4ad-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: git diff --binary --full-index --find-renames c5cf4ad2eab479886dfdf28e61508ef64c1198cf...HEAD > <patch>

MANIFEST: format fmd-patch-v1, wave w8, workstream section-structure, base_sha as above,
branch feature/<patch_id>, commit_message, summary, files_changed, tests_run,
evals {"result":"pass","notes":"provenance 1.0"}, known_gaps, created_at, agent.

findings.json must list: the heading extractor(s) used and their confidence, the level
normalisation rule, how breadcrumbs are passed to the model, what happens when structure is absent,
and the expected effect on the currently-rejected `ambiguous_span` claims (reasoned, not guessed).
~~~~
