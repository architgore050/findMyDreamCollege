# FMD-CLOUD-013 · fee-veto-precision · wave w6

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 8ca5caca168c643e1b8c9a4fa0a8d29e672d9b1a
BRANCH: feature/fmd-w6-fee-veto-precision-8ca5cac-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.

READ FIRST:
- AGENTS.md
- docs/product-development-cycle/04-phase-3-implementation-brief.md §9, §10
- docs/product-development-cycle/11-inference-runbook.md (classification section)
- fmd/plan/rules.py (the text-channel veto and the `_AMOUNT` pattern)
- fmd/plan/planner.py
- config/plan.yaml
- tests/test_plan_family_coverage.py, tests/fixtures/plan/**

TWO MEASURED PRECISION DEFECTS (found at intake, reproduced):

1. **The all-text-channel veto over-fires.** A genuine fee circular containing phrases like
   "revaluation fee" or "affiliation fee" now scores only 0.35 and lands in
   `family_undetected` — the veto cancels *every* text-derived channel including new heading
   evidence. This is a false negative on exactly the documents the product lacks (we have 1
   fee_structure document in the whole corpus).

2. **Headings alone can declare fee_structure with no amounts.** A prospectus whose headings
   read "Fee Structure 2025-26 / Fee payment 2026 / Fee refund 2025" resolves to
   `fee_structure` even when no fee amount appears anywhere in the sampled text.

SCOPE — FILES YOU OWN:
- fmd/plan/rules.py
- fmd/plan/planner.py (channel scoring only; precedence unchanged)
- config/plan.yaml (veto vocabulary, thresholds, amount-evidence rules)
- tests/test_plan_family_coverage.py, tests/test_plan*.py, tests/fixtures/plan/**
- docs/product-development-cycle/11-inference-runbook.md (veto semantics)

FORBIDDEN: fmd/extract/** fmd/verify/** fmd/acquire/** fmd/fetch/** fmd/score/**
fmd/ingest/** fmd/db/** db/migrations/** apps/** infra/** .env scripts/** config/sources.yaml

ACCEPTANCE
1. **Narrow the veto to administrative meaning, not the word "fee".** Veto only when the document
   is an administrative instrument about fee *administration*: recognition/affiliation approval
   orders, fixation/approval of fees by a regulator, refund rules, late-fee penalty notices, and
   the known case "Fee for recognition of Institution as Research Centre". Do NOT veto a document
   that publishes what students pay (tuition/hostel/mess/lab/exam fees).
2. **Require amount evidence for `fee_structure`.** Classify as `fee_structure` only if the sampled
   text also contains a money-like token (₹, Rs., INR, "per annum", "per semester", digit-grouped
   amounts) within the sampled window. Headings alone cap the score below the acceptance threshold
   and yield `ambiguous_family` instead of a confident family.
3. **Keep every existing negative guard passing.** Tender, convocation, staff circular, event/news,
   exam schedule, noise office order, and the research-centre fee order must remain
   `family_undetected`.
4. **Precedence unchanged**: registry family > text evidence > URL evidence.
5. **Fixtures/tests** (offline, quoted expectations):
   - genuine fee circular with "revaluation fee" in body and a tuition table → `fee_structure`;
   - affiliation/recognition approval order mentioning fees → `family_undetected`;
   - fee-headings-only prospectus with no amounts → not confident (`ambiguous_family`);
   - the same prospectus with a real fee table → `fee_structure`;
   - regression: every negative corpus fixture still abstains.
6. Document the veto vocabulary and the amount-evidence rule in the runbook.
7. No model calls, no OCR; bounded sample from config.

CONSTRAINTS: precision guards are non-negotiable; do not trade a false negative for a false
positive. No network in tests; no secrets; scoped diff.

RUN (all must pass):
cd packages/pipeline && uv sync --all-extras
uv run ruff check .
uv run pytest -q
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden     (provenance must be 1.0)

OUTPUT (root, UTC): fmd-patch-w6-fee-veto-precision-8ca5cac-<UTC>-01.patch / .json / .md
fmd-patch-w6-fee-veto-precision-8ca5cac-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: git diff --binary --full-index --find-renames 8ca5caca168c643e1b8c9a4fa0a8d29e672d9b1a...HEAD > <patch>

MANIFEST: format fmd-patch-v1, wave w6, workstream fee-veto-precision, base_sha as above,
branch feature/<patch_id>, commit_message, summary, files_changed, tests_run,
evals {"result":"pass","notes":"provenance 1.0"}, known_gaps, created_at, agent.

findings.json must list: the narrowed veto vocabulary, the amount-evidence rule and threshold,
and every case where you had to choose between a false positive and a false negative (with the
choice you made and why).
~~~~
