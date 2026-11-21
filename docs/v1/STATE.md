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

- (bootstrap) docs/v1 state files created and committed.

## In-PR

- (none)

## Blocked (cloud)

- Any run of `docker compose` (no docker binary) — LQ-1.
- Any model/GPU/real-crawl run — LQ-2..LQ-5.

## Next

- Wave 1 (PLAN #1): compose network wiring + acceptance compose smoke.
