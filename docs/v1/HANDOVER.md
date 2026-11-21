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

## Wave 1 — compose wiring (in progress)

(decision record pending)
