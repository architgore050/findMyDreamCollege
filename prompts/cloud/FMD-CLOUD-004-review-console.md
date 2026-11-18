# FMD-CLOUD-004 · review-console · wave w2 (pasted into chat as the canonical prompt)

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 78cd8483d69641118d5e1b080d34a86ed16ec11f
BRANCH: feature/fmd-w2-review-console-78cd848-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read by the
operator — the .md and .json files are your only report.

READ FIRST: AGENTS.md · docs/product-development-cycle/04-phase-3-implementation-brief.md
§17 (API), §18 (UI) · 11-inference-runbook.md · 15-patch-workflow.md · 07-workstream-status.md
· packages/pipeline/CONTRACT.md · db/migrations/0005_claim_verification.sql · fmd/api.py
· fmd/db/repositories.py (ClaimsRepo, aliases) · apps/web/lib/api.ts

PROBLEM: the review console is the last 501 surface. Everything it needs already
exists in the schema: `claims.verification_status/verified_by/verified_at/notes`,
`er_audit`, `entity_aliases` with `manual` type, `documents.plan_state=needs_review`.
There is no way for a human to accept/reject/correct a claim, confirm an entity
merge, or resolve an ambiguous document — so auto-accept errors are unrecoverable.

SCOPE — FILES YOU OWN:
- packages/pipeline/fmd/api.py (review endpoints only)
- packages/pipeline/fmd/db/repositories.py (claim/alias review methods only)
- db/migrations/0008_review_actions.sql (new; do not edit 0001–0007)
- apps/web/app/review/** (new)
- apps/web/lib/api.ts, apps/web/lib/types.ts (review client)
- packages/pipeline/tests/test_review_api.py, apps/web/**/__tests__/** (review)
- docs/product-development-cycle/11-inference-runbook.md (review section)

FORBIDDEN: fmd/fetch/** fmd/extract/** fmd/verify/** fmd/acquire/** fmd/score/**
fmd/plan/** fmd/ingest/** config/** infra/** .env scripts/**

ACCEPTANCE
1. Endpoints (admin-token gated, no user accounts):
   GET  /api/review/queue?kind=claim|alias|document   paginated, low-confidence first
   POST /api/review/claims/{id}   {action: accept|reject|correct, corrected_quote?, note?}
   POST /api/review/aliases/{id}  {action: confirm|reject}
   POST /api/review/documents/{id} {action: link_entity|mark_not_applicable, entity_id?}
   Every action writes an append-only audit row with `decided_by='human'`, timestamp,
   and the previous state. Never mutate claim value/quote/span — corrections create a
   superseding claim (`superseded_by`) instead.
2. `manual` aliases are authoritative: confirming an alias blocks future auto-merges
   on that pair (ER already reads this — prove it with a test).
3. The web page `/review` lists the queue with: predicate, quote, page/bbox, source URL,
   evidence tier, verification status, and the deciding claim. Accept/reject/correct
   work; every action shows the audit trail afterwards. No profile data on this page.
4. Wrong-action safety: rejecting a `supported` claim requires a note; the UI warns.
5. Tests: endpoint contract + RBAC (missing token → 403), supersede-not-mutate,
   manual alias blocks auto-merge, queue ordering, idempotent double-accept, and an
   offline component test for the queue rendering.
6. Docs updated in the same patch.

CONSTRAINTS: no secrets, no .env, no binaries, no tmp/ runs/ .venv node_modules.
Scoped diff. If a needed file is forbidden, STOP and record it in known_gaps.

RUN (all must pass):
cd packages/pipeline && uv sync --all-extras && uv run ruff check . && uv run pytest -q
cd apps/web && pnpm install --frozen-lockfile && pnpm lint && pnpm build && pnpm test

OUTPUT (three files at workspace root, UTC like 20261118T2300):
fmd-patch-w2-review-console-78cd848-<UTC>-01.patch / .json / .md
Patch: git diff --binary --full-index --find-renames 78cd8483d69641118d5e1b080d34a86ed16ec11f...HEAD > <patch>
Manifest format fmd-patch-v1, wave w2, workstream review-console, base_sha as above.
ALSO write `findings.json`: {"patch_id":"...","decisions":[...],"gaps":[...],"endpoints":[...]}.
~~~~
