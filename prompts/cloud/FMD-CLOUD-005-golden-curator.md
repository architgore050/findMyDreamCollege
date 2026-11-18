# FMD-CLOUD-005 · golden-curator · wave w2

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 78cd8483d69641118d5e1b080d34a86ed16ec11f
BRANCH: feature/fmd-w2-golden-curator-78cd848-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read —
produce the .md and .json files.

READ FIRST: AGENTS.md · 04-phase-3-implementation-brief.md §19 · 08-eval-methodology.md
· 12-phase-4-test-plan.md · 15-patch-workflow.md · data/golden/** (schema + _TEMPLATE)
· fmd/eval/golden.py · fmd/eval/metrics.py · fmd/eval/run.py

PROBLEM: the golden set has 5 entries / 16 verified rows and must reach ~50 for the
bake-off to mean anything. Curating them by hand-editing YAML is slow and error-prone,
and the loader's strictest rule (a verified quote must resolve in a committed fixture,
and a `context_row` row must be ambiguous globally but unique in its row) is easy to
violate accidentally. Humans need a safe tool, not a schema lecture.

SCOPE — FILES YOU OWN:
- packages/pipeline/fmd/eval/** (golden tooling only)
- packages/pipeline/fmd/cli.py (a `fmd golden` command group only)
- data/golden/** (schema docs/template only; add no invented rows)
- packages/pipeline/tests/test_eval_golden*.py
- docs/product-development-cycle/08-eval-methodology.md (curation workflow section)

FORBIDDEN: fmd/extract/** fmd/score/** fmd/fetch/** fmd/db/** db/** apps/** infra/**
config/** .env scripts/**

ACCEPTANCE
1. `fmd golden add` (interactive or flag-driven) that:
   - takes a source PDF path/URL + page + predicate + expected value/unit;
   - **extracts the quote from the source itself** (pdftext) and shows the exact span,
     refusing to accept a quote the operator typed from memory;
   - verifies uniqueness; for tables requires `row_label`/`column_header` and proves
     the value is ambiguous without context and unique with it;
   - writes the row as `verified` only if every check passes, otherwise writes
     `unverified` with `expected: null` and a reason — never a silent promotion;
   - records `source_url`, `retrieved_at`, and the fixture path it created.
2. `fmd golden check` — re-validate the whole set (offline): context resolution,
   fixture presence, sha256 of fixtures, duplicate predicate/entity rows, and a report
   of verified vs unverified counts by family.
3. `fmd golden list` — table of entries with status and what each one gates.
4. Fixture creation is deterministic and small (page-scoped TSV/text), committed under
   `data/golden/fixtures/`, with the source PDF's sha256 recorded.
5. Tests: quote-from-source (typed quote rejected), uniqueness failure, row-context
   proof, unverified fallback, deterministic fixture bytes, check detecting a tampered
   fixture, list output.
6. Do not add any golden row yourself; the tool is the deliverable.

CONSTRAINTS: no network required for checks; no secrets; scoped diff.

RUN: cd packages/pipeline && uv sync --all-extras && uv run ruff check . && uv run pytest -q
     uv run python -m fmd.cli eval --golden ../../data/golden  (provenance must stay 1.0)

OUTPUT: fmd-patch-w2-golden-curator-78cd848-<UTC>-01.patch/.json/.md
Patch: git diff --binary --full-index --find-renames 78cd8483d69641118d5e1b080d34a86ed16ec11f...HEAD > <patch>
Manifest format fmd-patch-v1, wave w2, workstream golden-curator.
ALSO write findings.json with the command surface and any rule you had to relax (state why).
~~~~
