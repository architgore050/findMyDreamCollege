# FMD-CLOUD-003 · e2e-acceptance · wave w1

You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

**Repo:** https://github.com/architgore050/findMyDreamCollege.git
**Base SHA:** `ff7d13e2192d22ae25f666f69f03b73a1d9f6063`
**Branch:** `feature/fmd-w1-e2e-acceptance-ff7d13e-<UTC>-01`
Do not push, open PRs, merge, or use secrets.

## Read first
- `AGENTS.md`
- `docs/product-development-cycle/04-phase-3-implementation-brief.md` §17 (API), §18 (UI), §22 (definition of done)
- `docs/product-development-cycle/11-inference-runbook.md`
- `docs/product-development-cycle/13-network-robustness.md`
- `docs/product-development-cycle/07-workstream-status.md`
- `docs/product-development-cycle/12-phase-4-test-plan.md`
- `packages/pipeline/CONTRACT.md`

## Problem
Every component works in isolation; nothing is strung together. Build the acceptance path and fix **only** the integration seams it exposes.

## Environment you do NOT have
No live DB, no model server, no GPU, no crawling. Your job is to produce the script + fixes + unit tests; **live steps must be runnable later by the operator and clearly marked.** Never fake a pass.

## Scope — files you own
new `scripts/acceptance.sh`, new `docs/product-development-cycle/16-acceptance-checklist.md`, `packages/pipeline/fmd/api.py` (integration fixes only), `fmd/cli.py` (wiring fixes only), `apps/web/lib/api.ts` + `apps/web/app/**` (only fixes needed for a real end-to-end render), `packages/pipeline/tests/test_api*.py`, `apps/web/**/__tests__/**`.

**Forbidden:** `fmd/fetch/**`, `fmd/extract/**`, `fmd/verify/**`, `fmd/score/**`, `fmd/plan/**`, `fmd/db/**`, `db/**`, `infra/**`, `.env`, `config/**`.

## Acceptance script requirements (`scripts/acceptance.sh`)
Offline-safe, idempotent, non-zero exit on failure, PASS/FAIL table at the end.

1. **Preflight:** DB reachable, migrations applied, registry seeded, model reachable (`fmd llm doctor --live`), `scripts/serve_local.sh probe` passes. Each check prints SKIP with a reason when the dependency is absent.
2. **API contract sweep** with JSON assertions:
   - `GET /api/health`, `GET /api/sources`, `GET /api/programmes`, `GET /api/programmes/{id}`
   - `POST /api/programmes/{id}/cost` with a **synthetic profile used in-process only, never persisted**
   - `POST /api/programmes/{id}/filters`
   - `POST /api/rankings` with a minimal profile `{nationality, target_intake_year, target_field}`
   - `GET /api/compare?ids=...`
   - `POST /api/exams`, `POST /api/exclusions`
   Assert the envelope keys (`scorer_version`, `weights_version`, `claims_hash`, `generated_at`); UNKNOWN is never rendered as 0 or 50; every money/salary value carries `method` and, for ranges, `lo`/`hi`.
3. **Web gates:** `pnpm lint`, `pnpm build`, `pnpm test`, then smoke-render `/`, `/shortlist`, `/programmes/<id>`, `/compare`, `/exams`, `/exclusions`, `/sources` and assert HTTP 200 with no raw `NaN`, `undefined`, or bare composite decimal in the HTML.
4. **Privacy assertion:** grep logs and DB writes to prove no profile field is persisted.

## Fixes allowed
Only what the acceptance run proves broken — wrong verb, missing envelope field, route 500, cost endpoint ignoring the profile, filter endpoint returning UNKNOWN without a reason. If a defect lives in a forbidden file, add it to `known_gaps` and mark that check SKIP with the reason.

## Constraints
No secrets, no `.env`, no binaries, no `tmp/`, `runs/`, `.venv`, `node_modules`. Scoped diff. No schema changes.

## Run (must pass)
```
cd packages/pipeline && uv sync --all-extras && uv run ruff check . && uv run ruff format --check . && uv run pytest -q
cd apps/web && pnpm install --frozen-lockfile && pnpm lint && pnpm build && pnpm test
bash -n scripts/acceptance.sh
```
Also add unit tests for the script's assertion helpers so they can be verified without a live stack.

## Output — three files at workspace root (UTC)
```
fmd-patch-w1-e2e-acceptance-ff7d13e-<UTC>-01.patch
fmd-patch-w1-e2e-acceptance-ff7d13e-<UTC>-01.json
fmd-patch-w1-e2e-acceptance-ff7d13e-<UTC>-01.md
```
Patch: `git diff --binary --full-index --find-renames ff7d13e2192d22ae25f666f69f03b73a1d9f6063...HEAD > <patch>`
Manifest `format: fmd-patch-v1`, wave `w1`, workstream `e2e-acceptance`, evals `not-run`, known_gaps = every SKIP.

Final reply (max 250 words): checks implemented, defects fixed, SKIPs with reasons, output paths.
