# v1 STATE — done / in-PR / blocked / next

Updated at the end of every wave. Read together with `PLAN.md` (per-item
acceptance), `LOCAL-QUEUE.md` (operator-only runs), `HANDOVER.md` (decisions &
risks).

## Bootstrap (2026-11-21)

- Cloned repo at main `117373fdde777b1deead0aeda2a405d6141036f8`.
- **Discrepancy:** the task brief's START SHA `03c62c149993a50c590b1b09b0b1711a238995a4`
  does **not exist** in this repository (`git cat-file` fails). Main has moved
  past the state the brief describes: patch branches w1–w10 are merged, W9
  programme scoping is in. Decision (recorded, per "adapt only with a written
  reason"): proceed from current main; the brief's SHA is treated as stale.
- Environment measured:
  - Docker: **absent** in this sandbox → no compose run possible here; compose
    work is code+tests+script only, real run is LQ-1.
  - GPU / 26B model: absent → all ML inference is LQ.
  - pnpm 10.34.9 via corepack (PATH `~/.local/bin`).
  - Local Postgres 16.2 + pgvector via `pgserver` (bundled binaries,
    data under `/home/z/my-project/.tools/pgdata`), port 5433, user/pass fmd/fmd
    — matches the gate `DATABASE_URL`. Start: `bash /home/z/my-project/scripts/start_pg.sh`.
- Baseline gates on main (this state):
  - `uv run ruff check .` → All checks passed
  - `uv run pytest -q` → 1355 passed, 83 skipped
  - `DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q` → 1435 passed, 3 skipped
    (skips: Qwen3-VL weights e2e, duckdb-environment, +1 interface-complete T2)
  - `uv run python -m fmd.cli eval --golden ../../data/golden` → provenance 1.0 PASS
    (golden: 5 entries / 16 verified rows — small; non-vacuous proof needs LQ-2)
  - web: `pnpm lint` ✓ · `pnpm test` 61 passed ✓ · `pnpm build` ✓

## Done

- **Wave 4 (PLAN #4) — double-buffered batched runner + tier-aware sizing
  (branch `v1/w4-double-buffer`).**
  - `PrefetchBuffer` (one worker, object store only): next batch's snapshot
    bytes are read off-thread while the GPU window is open; the acquire loop
    consumes them via the new `AcquisitionStage.run(data_for=...)` seam and
    falls back to a live read when a byte set has not landed. `prefetch_depth`
    → registry knob `batch.prefetch_documents` (default 4, 0 disables,
    negative rejected by the loader; CLI-wired like every sibling knob).
  - Sizing across tiers: T2-cached scans are known-pages to the sizer (the
    wave-3 sizing note, implemented; T2-cached scan ⇒ zero conversions in its
    first cycle, tested).
  - Verifier: **request-changes → fixed** (registry wiring, restored the
    unjustified call-order test edit, ruff format on changed files, docs,
    prefetch-failure logging). Stress scenarios: no thread-safety blocker.
  - Gates: ruff clean + format clean on changed files · pytest offline 1451 ✓
    · DB-gated 1540 ✓ · eval provenance 1.0 ✓ · web untouched (diff empty).
  - Residual: real-GPU validation = LQ-4 (unchanged).

- (bootstrap) docs/v1 state files created and committed.
- **Wave 3 (PLAN #3) — T2 parse cache, engine-pinned keys (branch
  `v1/w3-t2-parse-cache`). Foundation: the never-merged W10 branch
  ("persistent T1 parse cache + batched page-window acquire") was verified
  (approve) and merged as PR #3 first — items 3–4 presuppose it.**
  - Key: T2 rows carry an `engine_fingerprint` {engine, engine_version,
    model_ids(det/rec/rec_keys), model_source, model_sha256} inside the key;
    an engine-version bump or model-id change invalidates. Device excluded for
    every tier. T1 key layout byte-identical to W10 (fingerprint=None).
  - Scope decision (recorded in HANDOVER): only engines whose version+model
    ids are locally pinnable without a service (rapidocr via importlib
    metadata) are cached; qwen3vl/paddleocr T2 results are refused with a
    stable reason rather than cached under an aliasable identity.
  - The T2 policy digest projects the T2 block + RapidOCRSettings parse-shaping
    fields; store gate refuses docs whose own engine_meta disagrees with the
    locally derived fingerprint (env drift). Atomicity/completeness inherit
    from the W10 scheme (object first, row second; complete documents only).
  - Wiring: AcquisitionStage/Extractor/ExtractPipeline/BatchedRunner/
    Orchestrator thread `t2_parse_cache`; acquire+extract consult T1 then T2;
    live T2 results store into the T2 cache; batched sizing unchanged.
  - Tests: 11 new T2 tests (invalidation on engine bump + model swap, device
    exclusion, live-vs-cached claim equality, roundtrip identity, refusals,
    zero-tier-call hits) + a gate fix the tests caught (store gate must use
    the run's policy, not a reloaded default registry). All 13 T1 cache tests
    and 41 W10 tests untouched and green.
  - Gates: ruff ✓ · pytest offline 1432 ✓ · DB-gated 1521 ✓ · eval
    provenance 1.0 ✓. Web untouched this wave.
  - **Verifier verdict: approve** (independent subagent; scope decision judged
    as satisfying the brief's intent, not a dodge). Hardened from its findings
    before merge: store_parse refuses a parse whose tier differs from the
    cache tier (cross-tier aliasing via an injected T1-only cache); store
    refusals now reach run-summary outcome rows via CacheMiss.as_dict; doc 05
    gains the G8.1 parse-cache scope row. New test pins the cross-tier refusal
    (12 T2 tests total, 53 cache-suite tests green).
- **Wave 2 (PLAN #2) — acceptance harness offline-complete + JSON manifest
  (branch `v1/w2-acceptance-complete`).**
  - `scripts/acceptance.sh` now carries a canonical `EXPECTED_CHECKS` manifest
    (32 ids ≥ the doc-16 rows) and a safety net before the verdict: any manifest
    id that reached no verdict gets an explicit SKIP naming the harness bug.
    Dependency branches record honest SKIPs for everything they gate
    (pnpm-absent ⇒ web/lint,build,test,server + 7 renders; live web with empty
    DB ⇒ `web/render /programmes/<id>` SKIP).
  - `--json-out` report (format `fmd-acceptance-v1`) now carries `verdict` and
    `manifest {expected, recorded, missing, complete}` for CI.
  - Pins: `packages/pipeline/tests/test_acceptance_manifest.py` (6 tests:
    manifest == canonical inventory, record-site coverage incl. variable sites,
    safety-net iteration, JSON fields, writer parses).
  - **The safety net proved itself in its first run**: it caught `web/pnpm`
    never being recorded when pnpm *is* present — fixed by recording PASS in
    that branch.
  - Functional proof: full offline run → **24 passed / 0 failed / 8 skipped,
    manifest complete=true (expected 32, missing [])**.
  - Gates: ruff ✓ · pytest offline 1372 ✓ · DB-gated 1452 ✓ · eval
    provenance 1.0 ✓ · web lint/test/build ✓ (exercised in the run).
  - **Verifier verdict: approve** (independent subagent; full path trace of all
    32 ids, no invariant violation, no verdict-flip path). Fixed from its one
    actionable finding: the live render loop mirrors the concrete programme
    route's verdict onto the manifest placeholder `web/render /programmes/<id>`
    so a healthy populated run no longer draws a false "check never ran" net row
    (its simulated populated run had shown exactly that false alarm).
  - Granular commits pushed to `v1/w2-acceptance-complete` as work progressed
    (319ff62 script, 8a6a7d3 pins, eafbd5d web/pnpm fix, mirror fix + docs) —
    per operator protocol of 2026-11-21: everything is on GitHub before any
    summary.
- **Wave 1 (PLAN #1) — compose network wiring + acceptance compose smoke
  (PR #1, merged 2026-11-21).**
  - Root cause of the observed `Temporary failure in name resolution` loop:
    compose DNS resolves service names only among containers attached to the
    *same* network; the project-default network name varies with the compose
    project, so a db started under a different project/`-p`/checkout left api
    resolving nothing until its entrypoint gave up and `restart: unless-stopped`
    looped it forever.
  - Fix: every service (db, searxng, ocr, ocr-paddle, api, worker, web) now
    attaches to one declared network `fmd-net` with a fixed *physical* name
    (`networks: {fmd-net: {name: fmd-net}}`), so every invocation of this file
    converges on one segment and service aliases (`db`, `api`, `web`) follow the
    containers. db gained `restart: unless-stopped`. Runtime `network_mode:
    host` is forbidden (build-time `build.network: host` kept — documented
    BuildKit DNS workaround).
  - `scripts/acceptance.sh` gained section 0 `compose/smoke`: docker absent →
    SKIP (reason); daemon down → SKIP; compose config invalid → FAIL; stack up →
    decisive probe (api container connects to db at DATABASE_URL via psycopg) +
    web port probe. Offline invariants pinned by
    `packages/pipeline/tests/test_compose_wiring.py` (11 tests).
  - Doc 16 check inventory: `compose/smoke` recorded as check #0 (19 checks total).
  - Offline functional proof: `scripts/acceptance.sh` full run in this sandbox →
    **23 passed / 0 failed / 7 skipped, VERDICT PASS** (with a migrated+seeded
    local Postgres: countries=13, sources=30, source_exclusions=4, exams=8);
    with docker absent the new check SKIPs with its reason; `--json-out` mode
    exercised. The two earlier FAILs were the harness honestly reporting an
    empty DB — fixed by migrate+seed, not by code.
  - Gates: ruff ✓ · pytest offline 1366 ✓ · DB-gated 1446 ✓ · eval provenance 1.0 ✓ ·
    web lint/test/build ✓.
  - **Verifier verdict: approve** (independent subagent). Fixed from its findings
    before merge: compose/smoke gates the live probe on db AND api (partial
    stack → SKIP, not false FAIL); web running-but-not-answering → FAIL
    (matches the doc-16 row); service-set test uses exact equality; doc-16 row
    lists the FAIL verdict. Noted, not fixed (pre-existing, out of scope):
    `http_code` returns "" not "000" when curl is missing — helper-wide
    pattern, tracked for wave 2's harness pass.
  - Residual: the live `make up` boot itself needs docker → LQ-1 (unchanged).

## Merged to main (verified via GitHub API)

- PR #5 (wave 4: double-buffered batched runner + tier-aware sizing), merged
  as be0fcfa. Cumulative: waves 1-4 all merged; verifier-driven fixes applied
  pre-merge in every wave.

- PR #1 (wave 1 compose wiring), PR #2 (wave 2 acceptance manifest), PR #3
  (W10 foundation: T1 parse cache + batched acquire), PR #4 (wave 3 T2 parse
  cache, engine-pinned keys). Local main synced to 33eb59f.

## In-PR

- (none)

## Blocked (cloud)

- Any run of `docker compose` (no docker binary) — LQ-1.
- Any model/GPU/real-crawl run — LQ-2..LQ-5.

## Next

- Wave 5 (PLAN #5): plan/classification throughput — deterministic family
  rules from URL/path/source signals first, semantic classifier second; every
  semantic admission still needs cited evidence; uncertain never admitted; a
  dry-run report command shows rule-hit counts; any model run = LQ.
  Then waves 6-10 per PLAN order. Sandbox notes: pgdata survives recycles but
  loses its empty runtime dirs (recovery documented in
  /home/z/my-project/scripts/start_pg.sh); local main must be re-cloned after
  a recycle — state lives here, in the repo.
