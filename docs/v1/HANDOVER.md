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
