# findMyDreamCollege — v1.0.0-rc1

Release candidate. The build-out waves (PLAN v1 #1–#10) are merged to main;
what separates this candidate from a final v1.0.0 is the operator-side data
work in `docs/v1/LOCAL-QUEUE.md` (LQ-1…LQ-7) — docker, GPUs, model weights,
the real crawl. None of it can run in the build sandbox; all of it is
documented with exact commands and expected outputs.

## What ships in this candidate

**Pipeline (`packages/pipeline`, 1477 offline + 1570 DB-gated tests green,
eval provenance 1.0):**

- Ingest → discover → crawl → acquire (T0–T3 with a parse cache) → plan
  (deterministic family/entity lanes) → extract (schema'd, span'd, evidence-
  linked) → verify (second-pass claim judging) → score (7 sub-scores,
  composite bands, no fabricated numbers) → review (supersede/reject, history
  preserved).
- Classification throughput: `classify-report` replays the deterministic
  family chain per evidence document and buckets every missing_family row;
  the semantic tier is an admission CONTRACT (`fmd.plan.semantic`) — model
  proposals are refused unless the cited quote is verbatim in the fetched
  sample, the family is known, and no second family is proposed.
- Scoring coverage diagnostics: `coverage-report` answers "why is this
  programme 1/7" with the feeding claim ids per component.
- Adapters: the fee/seat/placement/admission lanes under contract
  (span-resolution invariant, administrative-fee veto, robots inherited from
  the fetch ladder).
- Trust surface: `/api/methodology` generated from `config/` (tests pin every
  value to the registry), dispute path in the payload, append-only source
  exclusion registry (`AppendOnlyViolation` on rewrite attempts).

**Web (`apps/web`, 14 files / 71 vitest tests green):**

- 7 routes + methodology: profile (client-side only), shortlist, compare,
  programme detail with provenance click-through (claim → quote → span →
  sha256 → source document), exams, exclusions, sources, review console.
- No NaN/undefined/bare composite decimals — enforced mechanically by
  `expectCleanRender` on every tested surface (it caught and killed a real
  `#undefined` render bug in the deadline table).

**Docs:** `docs/v1/{PLAN,STATE,LOCAL-QUEUE,HANDOVER,RELEASE}.md` — the plan,
the state, the operator's queue, every autonomous decision with rationale,
and these notes.

## Primary target

The v1 target is MLT (Medical Laboratory Technology) university selection in
India — the registry carries the B.Sc. MLT/MLS, BMLT, BMLS, DMLT vocabulary
and the ranking flow covers it. The same machinery generalises to other
programme families by registry entries, not code.

## Operator queue (what only the operator can do)

LQ-1 docker compose up · LQ-2 the real crawl (robots + rate limits; the four
family lanes) · LQ-3 extraction+scoring on the crawled corpus · LQ-4 LLM
fixtures · LQ-5 verify pass · LQ-6 coverage re-check · LQ-7 the 26B semantic
classification pass over `classify-report`'s residual, applied through
`plan --admit-evidence`. Exact commands in `docs/v1/LOCAL-QUEUE.md`.

## Known limits (deliberate, recorded in HANDOVER)

- The semantic tier computes nothing semantic in the cloud; its audit trail
  is the CLI's `admitted_detail` JSON (a DB-level signals column was
  deferred to avoid a migration in the release window).
- `/sources` has no dedicated render test (server component, no computed
  numbers; failure path covered) — recorded as an accepted gap.
- `family_set_by_source` in the classification report cannot split registry
  vs reference declarations (no family_source column — documented in
  `fmd/plan/report.py`).
- The Nav "Methodology" link is pinned by review, not a router-mocked test.
