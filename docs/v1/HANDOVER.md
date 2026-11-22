# v1 HANDOVER — decisions, broken invariants, risks, operator runbook

Append-only. One section per wave + final runbook. Architectural decisions and
invariant findings live here so the next agent does not re-derive them.

## Bootstrap (2026-11-21)

- **Stale START SHA.** Task brief says base `03c62c1…`; that object does not
  exist in the repo. Actual main at takeover: `117373f` (W9 programme scoping
  merged). All waves base on current main. Recorded as a concern, not lowered
  any target.
- **Sandbox capability map** (what the cloud can/cannot do — drives every
  "done" claim):
  - CAN: uv/pytest/ruff, pnpm 10 (corepack), local Postgres 16.2+pgvector
    (`pgserver` wheel; `bash /home/z/my-project/scripts/start_pg.sh`),
    static analysis of compose, network fetch for GitHub.
  - CANNOT: docker/compose (no daemon), GPU, llama-server / 26B weights,
    real OCR engines, polite real crawls at scale (left to operator anyway).
  - Therefore: "done" = implemented + offline-gated; every real-world proof is
    a LOCAL-QUEUE entry. Nothing is marked done on the strength of an unrun
    gate.
- **Environment for the operator** (if resuming in this sandbox): pnpm needs
  `export PATH=~/.local/bin:$PATH` (corepack shim). Postgres data:
  `/home/z/my-project/.tools/pgdata`.

## Wave 4 — double-buffered batched runner (2026-11-22)

- **Interpretation of the brief:** "CPU work for the next batch" = the disk
  I/O + cache peeks the acquire loop would otherwise do serially inside the
  server-stopped window. Docling/OCR conversions stay inside that window (the
  brief's second clause); the prefetch worker touches the object store only.
  Running conversions outside the window would have *downgraded* them from
  CUDA (0.166–0.9 s/page) to CPU (1.0–3.8 s/page) — rejected.
- **Correctness never depends on the prefetch winning the race:** `take()` is
  non-blocking; a byte set that has not landed falls back to a live read.
  The worker logs store failures (a silent degrade to 100% live reads was the
  verifier's observability finding) and a failed sha is simply not retried
  this run.
- **Registry-driven (verifier finding, fixed):** `batch.prefetch_documents`
  lives in `AcquireBatchConfig`/`acquire.yaml`/loader validation/CLI wiring
  like every sibling batch knob (default 4, 0 disables, negative rejected).
- **Verifier-caught regression, reverted:** the wave-4b commit had replaced
  the pre-existing call-order test's `counts["pages"] == 3` assertion with a
  line-based duplicate (a `str.replace` with an unanchored pattern hit both
  tests). Restored; the line-based assertion remains only in the T2 sizing
  test, where the stage never runs and the counter genuinely does not exist.
- **Wave-3 sizing note (HANDOVER §Wave 3) implemented:** the batch sizer now
  reads pages across tier caches (T1 then T2) — a T2-cached scan is a
  zero-page hit in its first cycle instead of being reconverted whole.
- Thread-safety verdict (verifier stress scenarios): no blocker; a lost
  prefetch after close is wasted work only, by design.

## Wave 3 — T2 parse cache (2026-11-21)

- **Foundation first:** backlog items 3–4 presuppose the W10 branch, which had
  never been merged. Verified (approve) and merged as PR #3 before extending.
- **Scope decision — cache only locally pinnable engines.** The brief's key
  requirements (engine + engine version + model ids) can only be satisfied
  honestly when the engine identity is derivable *without the engine*. RapidOCR
  reports its version via importlib.metadata and pins model ids in config —
  cacheable. qwen3vl/paddleocr report their version only through the service;
  caching under `version=None` would alias a service-side model swap — exactly
  the failure the key requirements exist to prevent — so those T2 results are
  refused (`t2_engine_fingerprint_unavailable`) and OCR'd live every time.
  If the operator wants qwen3vl caching, the service contract needs a
  model-identity field in `/extract` responses first (a backend change, not a
  cache change).
- **No schema bump.** The T2 payload reuses the W10 schema (same
  ExtractedDoc shape; engine provenance already in `tool`/`tool_version`/
  `engine_meta`). `PARSE_SCHEMA_VERSION` stays 1 — no T1 invalidation.
- **Two caches, one store/table.** `document_parses` already carries `tier`;
  T2 rows are just rows with tier="T2" and a fingerprint-bearing key. Lookup
  order is T1 then T2 (a T2 hit skips both Docling probing and OCR).
- **Bug the tests caught** (kept as a lesson): the store gate initially
  reloaded the default registry instead of using the run's policy — the
  model-swap invalidation test failed exactly because the gate compared the
  doc against the *committed* config, not the *run's* config. Gates must use
  the identity the run actually had.
- **Sizing note:** the batched runner's page sizing still reads the T1 cache
  (`pages_for`). T2-cached scans report unknown pages to the sizer until their
  first convert-then-cache cycle; they are never truncated or split (documents
  finish whole), so the cost is one conservative cycle, not correctness.
  Candidate follow-up: sizer reads max(pages across tier caches).

## Wave 2 — acceptance harness completeness (2026-11-21)

- **Decision: a canonical `EXPECTED_CHECKS` id manifest + a safety net**, not
  per-branch record-site audits alone. The doc-16 "18 checks" are rows that
  expand to 32 check ids (`compose/smoke`, 5 preflight, 11 api, 5 web gates,
  7 renders, 3 privacy); the manifest is the finer-grained truth and the doc
  rows stay the human grouping. The net (records SKIP for any id that reached
  no verdict) is the invariant-carrying layer; the pins
  (`test_acceptance_manifest.py`) freeze the manifest and the record-site
  coverage, including the two variable sites (`api/$label`, `web/render $route`).
- **The net caught a real bug on its first full run**: `web/pnpm` was only
  recorded on its SKIP branch; with pnpm present the id never reached the
  verdict table. This is exactly the failure mode the brief bans ("no check
  silently omitted") and it had survived every previous green run. Keep the net.
- JSON contract for CI (`--json-out`, format `fmd-acceptance-v1`): `totals`,
  `verdict` (`PASS`/`FAIL` — FAIL iff any check FAILED; SKIP never fails),
  `manifest.complete` must be true for a trustworthy run, per-check
  `name`/`result`/`detail`. A check may appear more than once (contract probes
  after a PASS add rows); `manifest.recorded` is name-deduplicated.
- `web/pnpm-install` is an auxiliary setup step (records only failures) and is
  intentionally outside the manifest; the pins whitelist it.
- Known limitation (recorded, not fixed): `http_code` returns "" (not "000")
  when curl itself is missing — pre-existing helper-wide pattern; out of
  wave-2 scope (keep diffs scoped per wave).

## Wave 1 — compose wiring (2026-11-21)

- **Decision: declared network with fixed physical name `fmd-net`.** Rejected
  alternatives: (a) relying on the pinned project `name:` alone — it fixes
  invocations of *this* file, but not a db already running under an older
  project name, which is exactly how the failure was observed; (b) `external`
  network pre-created by the operator — adds a manual bootstrap step to the
  fresh-clone path (definition-of-done #1) for no benefit here. The fixed-name
  declared network keeps `make up` zero-touch while converging split
  invocations.
- **Kept the api `dns: [1.1.1.1, 8.8.8.8]` override** (systemd-resolved stub
  hosts). On user-defined networks the override configures *forwarders* of
  Docker's embedded DNS; service names still resolve internally. If an operator
  host blocks outbound UDP/53, external crawl fetches degrade — that is a
  crawl-time concern, not the db-DNS failure this wave fixed.
- **`db` now has `restart: unless-stopped`**: the api/worker health-gate on it,
  so a crashed db previously wedged the stack silently.
- **New acceptance check `compose/smoke` is read-only** (never `up`s the stack):
  config validation + live probes only when the stack is already running.
  A full boot remains the operator's LQ-1 — pulling/building images in an
  acceptance gate would make the daily path 10+ minutes and network-dependent.
- Risk noted: `docker compose ps --format/--status` flags need a compose v2
  new enough to support them; on an old compose the smoke check degrades to
  SKIP (never a false PASS).

## Wave 1 operator verification pending

LQ-1 in LOCAL-QUEUE.md: run `make up` + `scripts/acceptance.sh` on a docker
host and commit back `reports/acceptance.json`. Until then, wave 1 is
"done (code + offline gates)", not "verified live" — the distinction is kept
explicit everywhere.
