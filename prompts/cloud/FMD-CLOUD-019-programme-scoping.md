# FMD-CLOUD-019 · programme-scoping · wave w9 (FINAL, semantic tier)

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: ac6abcf6bb4133510c2901ad51326a9db2fb33db
BRANCH: feature/fmd-w9-programme-scoping-ac6abcf-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.

READ FIRST:
- AGENTS.md
- docs/product-development-cycle/04-phase-3-implementation-brief.md §6 (programmes), §11 (scoring)
- docs/product-development-cycle/14-deferred-items.md §B2 (cross-country equivalence — the design
  constraint behind this patch)
- docs/product-development-cycle/11-inference-runbook.md
- fmd/ingest/** (programme materialisation, normalisation, field_sources)
- fmd/cli.py (`fmd programmes ...`)
- fmd/llm/** (provider-agnostic router; use it, do not call providers directly)
- config/predicates.yaml, config/sources.yaml (existing name_aliases)

PROBLEM (measured on the live DB after materialisation): 7 programmes exist, including rows that are
not this product:
```
 1  TN Dr. M.G.R. Medical University | B.Sc. Medical Laboratory Technology | seats 20 (claim 47)
 5  TN Dr. M.G.R. Medical University | "3. B.Sc. Medical Laboratory Technology"   <-- duplicate
 8  Adesh University                  | MBBS - Bachelor of Medicine and Bachelor of Surgery  <-- out
11  Rani Lakshmi Bai Central Agri Univ | B.Sc (Hons) Agriculture                          <-- out
```
Two defects: (a) no target-field scoping, so MBBS/Agriculture enter the pool; (b) no name normalisation,
so row numbering ("3. ") created a duplicate.

DESIGN CONSTRAINT (operator, non-negotiable): **exact alias lists are insufficient.** Universities invent
one-off names, and the same degree is named differently per country. The system must be able to *decide*
equivalence from evidence, not only match known strings — while remaining conservative and auditable.

SCOPE — FILES YOU OWN:
- new config/programmes.yaml (country-keyed taxonomy + conceptual definition + classifier settings);
  register it in fmd/config (10th registry file)
- fmd/ingest/** (normalisation + three-tier scoping in materialisation)
- new fmd/programmes/** (equivalence classifier, cache, review queue) OR fmd/ingest/equivalence.py
- fmd/cli.py (`fmd programmes materialise|list|inspect|requalify`)
- fmd/db/repositories.py (Programme queries only; migration allowed as 0010 if a column is needed)
- fmd/score/** (eligibility only; out-of-scope must never be scored)
- tests/test_programmes*.py, tests/fixtures/programmes/**
- docs/product-development-cycle/11-inference-runbook.md, 04-…-brief.md §6 (one paragraph)

FORBIDDEN: fmd/extract/** fmd/verify/** fmd/acquire/** fmd/fetch/** fmd/plan/** apps/** infra/** .env scripts/**

ACCEPTANCE

A. Tier 1 — rules (fast, deterministic, high precision)
1. `config/programmes.yaml` is **country-keyed**: `countries: { IN: {...}, AU: {...stub}, US: {...stub}, ... }`
   plus `default` and a `definition` block. No hardcoded lists in Python.
2. Per country: `include` patterns, `exclude` patterns, `alias` canonicalisation. **exclude beats include.**
   India: B.Sc./BMLT/BMLS/MLS/B.MLS/Medical Laboratory Technology/Clinical Laboratory Science;
   explicit excludes for MBBS, BDS, BAMS, BHMS, B.Sc Nursing, GNM, BPT, B.Pharm, Agriculture.
3. Other countries ship as **empty stubs** (no invented patterns). An unknown country falls back to
   `default` and flags `country_taxonomy_missing`.
4. Name normalisation before matching: strip leading row numbering (`"3. "`, `"3)"`), collapse whitespace,
   normalise punctuation/case, apply aliases. Prove `"3. B.Sc. ..."` merges rather than duplicating.

B. Tier 2 — semantic equivalence (for names no rule matches)
5. Use the existing **model router** (provider-agnostic, default local Gemma; never call a provider
   directly). Given: programme name, institution, department/affiliation if known, and any
   curriculum/regulator claims; and the **conceptual definition** (diagnostic laboratory science:
   clinical biochemistry, haematology, microbiology, histopathology, blood bank, immunology), decide
   `equivalent: yes | no | uncertain` + `confidence` + `rationale` + **evidence list** (claim ids or
   quoted strings it relied on).
6. **Conservative by construction**: no evidence → `uncertain`. `uncertain` never enters the ranking;
   it becomes `needs_review`. A false positive (admitting MBBS/Nursing) is worse than `uncertain`.
7. Batched, cached by hash of (normalised name, institution type, definition version); one call per new
   name, not per row. Record model, prompt version, and cache flag per decision for audit.
8. Offline tests with the mock/replay provider: yes/no/uncertain paths; MBBS → no; Nursing → no;
   "Bachelor of Biomedical Science (Laboratory Medicine)" → yes (AU stub or classifier); unknown country
   → uncertain + flag; no-evidence row → uncertain; cache hit makes zero provider calls.

C. Tier 3 — learning loop
9. A human `needs_review` decision (`fmd programmes decide <id> --equivalent|--not-equivalent --note`)
   writes a `manual` alias/decision that Tier 1 consumes on the next run, so the classifier is asked
   less over time. Manual decisions always win over the classifier.
10. `fmd programmes requalify` re-evaluates existing rows (merge the row-numbering duplicate, mark
    MBBS/Agriculture out of scope, apply manual decisions), idempotent, with `--dry-run`.

D. Visibility and scoring
11. Out-of-scope rows are **kept** with `in_scope=false` + `scope_reason` (for audit), excluded from
    scoring/ranking/shortlist; `fmd programmes list --all|--in-scope` and `inspect` show the tier that
    decided (rule | semantic | manual) and the evidence.
12. `fmd score` skips out-of-scope programmes and reports the skipped count, so an MBBS can never appear
    in a ranked set.
13. Docs updated in the same patch.

CONSTRAINTS: no invented equivalences; every semantic admission cites evidence; no deletion of claims or
evidence; no scoring weight changes; no network in tests; no model downloads; scoped diff; do not edit
applied migrations 0001–0009.

RUN (all must pass):
cd packages/pipeline && uv sync --all-extras
uv run ruff check .
uv run pytest -q
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden     (provenance must be 1.0)

OUTPUT (root, UTC): fmd-patch-w9-programme-scoping-ac6abcf-<UTC>-01.patch / .json / .md
fmd-patch-w9-programme-scoping-ac6abcf-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: git diff --binary --full-index --find-renames ac6abcf6bb4133510c2901ad51326a9db2fb33db...HEAD > <patch>

MANIFEST: format fmd-patch-v1, wave w9, workstream programme-scoping, base_sha as above,
branch feature/<patch_id>, commit_message, summary, files_changed, tests_run,
evals {"result":"pass","notes":"provenance 1.0"}, known_gaps, created_at, agent.

findings.json must list: the country-keyed taxonomy, the normalisation steps, the precedence rule
(exclude > include > semantic > uncertain), the classifier prompt/definition version, cache key, the
before/after programme counts expected on the operator's DB (reasoned), and every name the classifier
could not decide (with what evidence is missing).
~~~~
