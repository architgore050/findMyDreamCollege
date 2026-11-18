# FMD-CLOUD-001 · crawler-targeting · wave w1

You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

**Repo:** https://github.com/architgore050/findMyDreamCollege.git
**Base SHA:** `ff7d13e2192d22ae25f666f69f03b73a1d9f6063`
**Branch:** `feature/fmd-w1-crawler-targeting-ff7d13e-<UTC>-01`
Do not push to GitHub. Do not open a PR. Do not merge. No secrets.

## Read first
- `AGENTS.md`
- `docs/product-development-cycle/04-phase-3-implementation-brief.md` §7 (discovery), §8 (sources), §9 (extraction)
- `docs/product-development-cycle/11-inference-runbook.md`
- `docs/product-development-cycle/13-network-robustness.md`
- `config/sources.yaml` → `institution_sites.params`, `state_health_universities.params`
- `packages/pipeline/fmd/adapters/institution_site.py`, `fmd/plan/rules.py`, `fmd/plan/roles.py`, `fmd/ingest/targeting.py`

## Problem
The crawler finds institutions and accreditation documents, but almost no **fee / seat-matrix / placement** evidence — the highest-value documents for this product. Measured: 27 evidence docs; 47 linked as accreditation 27 / placement 10 / curriculum 9 / **seat_matrix 1**, **fee_structure 0**.

## Scope — files you own
`config/sources.yaml` (only `institution_sites.params` and `state_health_universities.params`), `packages/pipeline/fmd/adapters/institution_site.py`, `packages/pipeline/fmd/plan/rules.py`, `packages/pipeline/fmd/plan/roles.py`, `packages/pipeline/fmd/ingest/targeting.py`, `packages/pipeline/tests/test_adapters_institution_site.py`, `packages/pipeline/tests/test_plan*.py`, `packages/pipeline/tests/fixtures/adapters/**`, `docs/product-development-cycle/06-india-adapter-spec.md` (targeting section only).

**Forbidden:** `fmd/fetch/**`, `fmd/extract/**`, `fmd/verify/**`, `fmd/score/**`, `fmd/db/**`, `db/**`, `apps/**`, `infra/**`, `.env`, `scripts/**`.

## Acceptance criteria
1. **Evidence-type link scoring**, data-driven and config-overridable: boost filenames/anchors matching `seat|intake|sanctioned|seat matrix`, `fee|tuition|hostel|mess|fee structure`, `prospectus`, `admission brochure`, `placement|annual report|AQAR|SSR`, `NIRF`. Deprioritise `news|event|gallery|tender|staff circular`.
2. **Negative pattern is mandatory:** an administrative order whose URL contains "Fee" in an unrelated sense — the real case is `Guidelines … Fee for recognition of Institution as Research Centre for Ph.D. Course` — must NOT classify as `fee_structure`. Tests must pin this.
3. **PDF-first budget reserve per domain:** a configurable share of each domain's document budget is reserved for candidate evidence PDFs so navigation pages cannot consume it.
4. **Family hints at discovery** propagate to the planner for seat_matrix/fee_structure, without overriding registry-declared families.
5. **Offline tests only** — no live crawling, no network. Fixtures prove: seat/fee PDFs outrank news and galleries; the negative case is rejected; budget reserve respects caps; scoring tables are config-driven.
6. Docs you own updated in the same patch.

## Constraints
- No user-specific values, no secrets, no binaries, no `tmp/`, `runs/`, `.venv`, `node_modules`.
- Do not edit applied migrations; no schema changes expected.
- Keep the diff scoped; no unrelated refactors.
- If you need a forbidden file or hit a design ambiguity, STOP and record it in `known_gaps`.

## Run (must pass)
```
cd packages/pipeline
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## Output — exactly three files at workspace root (UTC, e.g. `20261118T2200`)
```
fmd-patch-w1-crawler-targeting-ff7d13e-<UTC>-01.patch
fmd-patch-w1-crawler-targeting-ff7d13e-<UTC>-01.json
fmd-patch-w1-crawler-targeting-ff7d13e-<UTC>-01.md
```
Patch: `git diff --binary --full-index --find-renames ff7d13e2192d22ae25f666f69f03b73a1d9f6063...HEAD > <patch>`

Manifest (`format: fmd-patch-v1`):
```json
{"format":"fmd-patch-v1","patch_id":"...","wave":"w1","workstream":"crawler-targeting",
 "base_sha":"ff7d13e2192d22ae25f666f69f03b73a1d9f6063","branch":"...","commit_message":"...",
 "summary":"...","files_changed":["..."],"tests_run":[{"command":"...","result":"pass","notes":""}],
 "evals":{"result":"not-run"},"known_gaps":[],"created_at":"...","agent":"..."}
```

Final reply: output paths, the scoring table, and known gaps (max 200 words).
