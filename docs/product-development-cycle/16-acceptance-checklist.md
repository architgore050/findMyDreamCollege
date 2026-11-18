# 16 — Acceptance checklist (end-to-end path)

**Status:** implemented 2026-11-18 (wave w1, workstream e2e-acceptance, base
`ff7d13e`). Owner: `scripts/acceptance.sh`. Companion docs:
`12-phase-4-test-plan.md` §4 (the live-corpus phase-4 script — this one is the
offline-safe daily gate), `07-workstream-status.md` ("0 capabilities are
proven end-to-end"), `packages/pipeline/CONTRACT.md`.

## 1. Why this exists

Every component was proven in isolation (698 pytest + 45+ vitest green at the
snapshot) while **nothing was strung together**: the API had never been swept
endpoint-by-endpoint, the web app had never been smoke-rendered against the
built API, and no automated check proved that a submitted profile is never
persisted. This script runs that path, fails loudly on any seam it exposes,
and is honest about what it could not check: every absent dependency prints
**SKIP with a reason** and is never counted as a pass.

## 2. Running it

```bash
# full stack (compose): DB + API + web already up
API_BASE_URL=http://127.0.0.1:8000 scripts/acceptance.sh

# against a bare host: the script starts uvicorn itself on a scratch port
# (port 8765) when no API answers, and builds/serves the web app on 3100
scripts/acceptance.sh --json-out reports/acceptance.json

# offline (no DB, no model): everything dependency-shaped prints SKIP
scripts/acceptance.sh
```

Environment: `DATABASE_URL` (default compose port 5433), `API_BASE_URL`,
`WEB_BASE_URL`, `API_PORT`, `WEB_PORT`. Exit code is non-zero **iff at least
one check FAILED**; SKIPs are listed as known gaps in the final table.

## 3. Check inventory

| # | Check | Verdicts | Enforced invariant |
|---|---|---|---|
| 1 | `preflight/db-reachable` | PASS (LIVE) / SKIP | operator runnability |
| 2 | `preflight/migrations` | PASS (LIVE) / FAIL / SKIP | `schema_migrations` count == `db/migrations/*.sql` count |
| 3 | `preflight/registry-seeded` | PASS (LIVE) / FAIL / SKIP | sources & countries rows exist (`fmd migrate && fmd seed`) |
| 4 | `preflight/model-reachable` | PASS (LIVE) / SKIP | `fmd llm doctor --live` exit 0 (required profiles usable) |
| 5 | `preflight/serve-local-probe` | PASS (LIVE) / SKIP | `scripts/serve_local.sh probe`: text, no thought leak, guided JSON incl. nullable |
| 6 | `api/server` | PASS / SKIP | API answers, or uvicorn started on a scratch port |
| 7 | `api/health`, `api/sources`, `api/programmes` | PASS / FAIL / SKIP | 200 + evidence envelope (§17) |
| 8 | `api/programme-detail` | PASS / FAIL / SKIP | 200 + envelope + money blocks carry method/lo/hi |
| 9 | `api/programme-cost` (POST, synthetic profile) | PASS / FAIL / SKIP | envelope + UNKNOWN never 0/50 + **profile honoured** (`loan_required=false` ⇒ principal 0; income ⇒ gap computed) |
| 10 | `api/programme-filters` (POST) | PASS / FAIL / SKIP | 8 decisions, `fail`/`unknown` always carry a reason |
| 11 | `api/rankings` (minimal profile `{nationality, target_intake_year, target_field}`) | PASS / FAIL / SKIP | envelope + intervals + filter reasons |
| 12 | `api/compare` | PASS / FAIL / SKIP | Pareto matrix, ≥2 programme ids |
| 13 | `api/exams`, `api/exclusions` (POST) | PASS / FAIL / SKIP | status ∈ open/closed/unverified, deciding filters carry reasons |
| 14 | `web/lint`, `web/build`, `web/test` | PASS / FAIL / SKIP | `pnpm lint`, `pnpm build` (`NEXT_PUBLIC_API_URL` inlined), `pnpm test` |
| 15 | `web/render /…` (7 routes) | PASS / FAIL / SKIP | HTTP 200 and rendered markup (scripts/styles stripped) has no raw `NaN`/`undefined`/`Infinity`/`[object Object]` and no bare composite decimal — the composite renders as a band |
| 16 | `privacy/response-bodies` | PASS / FAIL / SKIP | the synthetic-profile sentinel appears in no response body or rendered page |
| 17 | `privacy/api-logs` | PASS / FAIL / SKIP | sentinel absent from the API server log |
| 18 | `privacy/db-rows` | PASS / FAIL / SKIP | sentinel absent from **every** table (read-only `::text LIKE` scan, as in `test_profile_is_never_persisted`) |

The synthetic profile lives only in the request bodies of one run: `regions`
and `acceptance_marker` carry a per-run sentinel, `financing` states
`loan_required=false` plus an explicit income point so checks 9/16 have
deterministic expectations. It is never written to disk outside the script's
`mktemp -d` work dir, which is removed on exit.

## 4. Where the assertions live

`packages/pipeline/fmd/acceptance.py` — pure, standard-library-only functions
(`envelope_failures`, `unknown_render_failures`, `money_method_failures`,
`filter_reason_failures`, `cost_respects_profile_failures`,
`html_artifact_failures`, `privacy_failures`) behind a small CLI
(`python -m fmd.acceptance <check>`). `tests/test_api_acceptance.py` pins
every helper's pass/fail behaviour without a live stack; the shell script is
only an orchestrator, so a check cannot drift from its test silently.

## 5. Defects the first run exposed (all fixed in this patch)

| Defect | Seam | Fix |
|---|---|---|
| `costCategorySchema` omitted `"tuition"` — every seeded programme detail payload failed zod in the browser and rendered the failure panel | web ↔ API | added `tuition` to `apps/web/lib/types.ts` (+ contract test) |
| `sourceSchema.adapter` required while the registry permits `null` | web ↔ API | schema nullable; `/sources` renders "no adapter" |
| `/api/scoring/rerun` resolved `DatabaseDep` **before** the admin gate — without Postgres the 503 `database_unavailable` shadowed 401/503 auth verdicts, and unauthorised callers made the API probe the DB first | API dependency order | admin gate declared first; `test_admin_write_endpoints_are_gated` now passes with no DB and asserts the DB-down path |
| FastAPI's 422 body echoed `exc.errors()` **including the submitted `input`** — a malformed POST reflected the caller's profile (family financing data) into the response and any response-logging | privacy | `_validation_message` strips `input`; `test_validation_errors_never_echo_the_profile` pins it; the script's sentinel scan now guards every response body |
| GET `/api/programmes/{id}/cost` probed the programme twice | API | single `_programme_or_404` |

## 6. SKIP semantics and known gaps

A SKIP names its missing dependency and the operator command that would turn
it into a real check. Known gaps when no stack is up: no DB (checks 1–3, 8–13
partial, 18), no model server (4–5), no API (6–13, 16–17), no pnpm/web build
(14–15). **A SKIP is never a pass** — the verdict line prints both counts, and
the manifest lists every SKIP from a run as a known gap.

Live steps deliberately left to the operator (no faked passes): a real
`make up` compose stack, `fmd llm doctor --live` against a served model,
`scripts/serve_local.sh probe` against real GGUF weights, and the phase-4
live-corpus run of doc 12 §4.

## 7. What this script does not prove

It proves the seams hold: envelope discipline, UNKNOWN/interval honesty,
filter reasons, cost-profile coupling, HTML hygiene, profile non-persistence.
It does **not** prove the shortlist is good, that crawled evidence is the
right evidence, that extraction quality gates pass, or that the DB is
populated (an empty registry yields PASSes for shape and SKIPs for content —
doc 12's rule that "a DB-count assertion that passes at 0 is a failure"
belongs to the phase-4 live run, where the corpus is a precondition).
