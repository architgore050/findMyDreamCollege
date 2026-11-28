# v1 Agent Handoff (AGENT-README.md) — handoff for the cloud V1 driver

State lives in `docs/v1/`: `STATE.md`, `PLAN.md`, `HANDOVER.md`,
`LOCAL-QUEUE.md`. Read them together; this file is the entry point. All four
state files were re-verified against `origin/main` on 2026-11-22 and are
byte-identical to the `~/Downloads` copies.

## Base

- Repo: `https://github.com/architgore050/findMyDreamCollege.git` (origin).
- Main tip: `09284e41c12e6f91ae539d0c3f955898044a2462` — `git rev-parse main`
  and `git rev-parse origin/main` both return it on 2026-11-22.
- Cloud agent starts: clone origin and checkout main at the tip above.

## New since the driver's last run

- AGENTS.md **hard rule** (commit `10bb390`): everything must be pushed to
  main locally *before* you tell a cloud agent to pull from main. The driver
  prompt defers to AGENTS.md ("working mode, long-running protocol.
  Non-negotiable") but its own HARD RULES list has no equivalent line — state
  this rule explicitly at kickoff: `git fetch origin`, pull to tip, verify
  with `git rev-parse HEAD`.
- Driver prompt START SHA is `10bb390ee3325ce03af5c7d33d879796bd8c898c`; the
  current tip is `09284e41c12e6f91ae539d0c3f955898044a2462`. Start from the tip.
- W10 (T1 parse cache + batched acquire) and wave 3 (T2 cache) verified
  present: `packages/pipeline/fmd/orchestrate/batched.py` (`BatchedRunner`,
  `_run_cycle`/`_run_cycles`), `db/migrations/0011_document_parses.sql` (W10),
  `packages/pipeline/fmd/acquire/parse_cache.py` (`PARSE_SCHEMA_VERSION = 1`,
  `engine_fingerprint`), tests `test_orchestrate_batched.py`,
  `test_parse_cache.py`, `test_parse_cache_t2.py`, `test_parse_cache_db.py`.
  No correction needed.
- FMD-CLOUD-030 (2026-11-28) verified present: automatic resource phases
  (`packages/pipeline/fmd/orchestrate/gpu_phases.py`, `ResourcePhaseManager`,
  used by `BatchedRunner` and `fmd acquire --phased`, worker auto-route) and
  compute metrics (`packages/pipeline/fmd/obs/metrics.py`, JSONL under
  `data/compute-metrics/`, `fmd metrics summary`), tests
  `test_orchestrate_gpu_phases.py`, `test_obs_metrics.py`,
  `test_run_logged_heartbeat.py`. Real-GPU validation remains LQ-4.

## V1 state as of now

- **Done**: wave 1 (compose `fmd-net` wiring on db/api/web — verified;
  wave-1 `compose/smoke`), wave 2 (`EXPECTED_CHECKS` 32-id manifest in
  `scripts/acceptance.sh` + `--json-out` — verified), W10 foundation, wave 3
  (T2 parse cache — `tier="T2"` rows, fingerprint-bearing key,
  `t2_engine_fingerprint_unavailable` refusals).
- **In-PR**: none.
- **Blocked (cloud)**: LQ-1..LQ-6 — reasons quoted under "Local queue".
- **Next**: wave 4 (PLAN #4, `pending`): resource-phased acquisition double
  buffering — the W10 batched runner already time-shares server-stopped
  acquire with server-up extract; wave 4 overlaps next-batch CPU prep with the
  GPU window (real-GPU validation = LQ-4). Uncategorized TODO from STATE:
  wave-3 sizing note (HANDOVER) — sizer reads T1 pages only; candidate: read
  max pages across tier caches.

## Local queue — why local + exact commands

- **LQ-1: real `make up` compose verification (incl. wave-1 DNS fix)** — cloud
  lacks a docker daemon.
  ```
  cd /path/to/findMyDreamCollege
  docker compose -f infra/docker-compose.yml down -v
  make up          # or: docker compose -f infra/docker-compose.yml up -d --build
  docker compose -f infra/docker-compose.yml ps
  docker compose -f infra/docker-compose.yml logs api | grep -i "name resolution"
  make migrate && make seed
  curl -sf localhost:8000/api/health
  curl -sfI localhost:3000 | head -1
  API_BASE_URL=http://127.0.0.1:8000 WEB_BASE_URL=http://localhost:3000 scripts/acceptance.sh --json-out reports/acceptance.json
  ```
  Commit back: `reports/acceptance.json` (+ any compose correction as new PR).
  Blocks: wave-1 verification, DoD #1.
- **LQ-2: real crawl + extract + verify + score (GPU + model + real network)**
  — cloud has no GPU, no llama-server 26B weights, no real egress for live runs.
  ```
  cd packages/pipeline && set -a && . ../../.env && set +a
  export FMD_EXTRACT_PROVIDER=local FMD_SNAPSHOT_DIR=$HOME/fmd-snap2
  scripts/serve_local.sh start && scripts/serve_local.sh probe
  docker compose -f ../../infra/docker-compose.yml up -d db api
  uv run python -m fmd.cli migrate && uv run python -m fmd.cli seed
  uv run python -m fmd.cli crawl --limit 50
  uv run python -m fmd.cli extract --limit 20
  uv run python -m fmd.cli verify --limit 50
  uv run python -m fmd.cli score
  uv run python -m fmd.cli eval --golden ../../data/golden
  ```
  Commit back: golden rows for newly-verified docs, `reports/` eval report,
  counts in STATE.md.
- **LQ-3: operator evidence targets** — needs populated corpus (model + real network).
  ```
  psql "$DATABASE_URL" -c "select count(*) from programmes where in_scope"
  psql "$DATABASE_URL" -c "select entity_id, count(distinct component) from component_coverage group by 1 having count(distinct component) >= 3"
  ```
  Targets: in-scope ≥ 150; ≥ 50 programmes with ≥ 3 of 7 components. Never
  fabricate counts.
- **LQ-4: real-GPU validation of the wave-4 batched runner** — GPU only; all
  GPU paths are faked in tests.
  ```
  set -a && . .env && set +a
  export FMD_SNAPSHOT_DIR=$HOME/fmd-snap2
  uv run python -m fmd.cli acquire --batched --batch-size 8
  ```
  Expected: `nvidia-smi` shows never more than one CUDA consumer; run summary
  `ok:true`; no `docling_cuda_oom`.
- **LQ-5: T2 OCR real-engine cache validation (wave 3)** — model/OCR weights.
  ```
  set -a && . .env && set +a
  export FMD_SNAPSHOT_DIR=$HOME/fmd-snap2
  uv run python -m fmd.cli acquire --limit 5 --tier T2
  uv run python -m fmd.cli acquire --limit 5 --tier T2
  ```
  Expected: second run reports cache hits, identical results.
- **LQ-6: operator acceptance gate (phase-4, doc 12 §4)** — real stack +
  real corpus.
  ```
  docker compose -f infra/docker-compose.yml down -v
  docker compose -f infra/docker-compose.yml up -d --build
  curl -sf localhost:8000/api/health                       # status ok
  make migrate && make seed
  psql "$DATABASE_URL" -c "select count(*) from sources where enabled"
  psql "$DATABASE_URL" -c "select count(*) from countries"
  make crawl ARGS='--limit 50'
  make extract ARGS='--limit 20'
  make verify && make score
  psql "$DATABASE_URL" -c "select count(*) from programmes"
  psql "$DATABASE_URL" -c "select verification_status, count(*) from claims group by 1"
  make eval                    # exit 0, provenance 1.0, vacuous: false
  cd apps/web && pnpm test     # 45+ pass, no 501 on a data screen
  ```
  Commit back: `reports/phase4-acceptance.md`.
  Blocks: acceptance-checklist "all PASS or explicitly BLOCKED" claim, release.

Environment facts (STATE.md bootstrap): cloud sandbox — Docker: **absent**
(no compose); GPU/26B model: absent; pnpm 10.34.9 via corepack (`export
PATH=~/.local/bin:$PATH`); local Postgres 16.2+pgvector via `pgserver`
(`bash /home/z/my-project/scripts/start_pg.sh`, data `/home/z/my-project/.tools/pgdata`,
port 5433, fmd/fmd) — the cloud CAN run DB-gated pytest.

## Verified inconsistencies (driver docs vs reality; neutral, one line each)

1. Driver prompt START SHA `10bb390ee3325ce03af5c7d33d879796bd8c898c` is one
   commit behind origin/main tip `09284e41c12e6f91ae539d0c3f955898044a2462`
   (`dbb47a3`). Driver starts at tip; if the prompt is updated, set it to `dbb47a31…`.
2. Task brief's START SHA `03c62c149993a50c590b1b09b0b1711a238995a4` does not
   exist in the repo (recorded in STATE.md bootstrap).
3. Minor: STATE.md says "Local main synced to 33eb59f" — actual tip is
   `dbb47a31…`.

## Commands

- Launch driver: paste `prompts/cloud/FMD-CLOUD-V1-DRIVER.md` into the cloud
  agent session; it reads AGENTS.md + `docs/v1/*` and follows its WAVE PROTOCOL.
- Gates (per wave; never merge red):
  ```
  cd packages/pipeline && uv sync --all-extras
  uv run ruff check .
  uv run pytest -q
  DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
  uv run python -m fmd.cli eval --golden ../../data/golden        # provenance 1.0
  cd ../../apps/web && pnpm lint && pnpm build && pnpm test       # if node_modules installable
  ```

## Operator to-dos

- Re-verify the four files against origin/main: `docs/v1/STATE.md`,
  `docs/v1/PLAN.md`, `docs/v1/HANDOVER.md`, `docs/v1/LOCAL-QUEUE.md`.
- ~/Downloads copies exist: `~/Downloads/STATE.md`, `~/Downloads/PLAN.md`,
  `~/Downloads/HANDOVER.md`, `~/Downloads/LOCAL-QUEUE.md` (verified
  byte-identical on 2026-11-22). Also in ~/Downloads:
  `fmd-patch-w10-batched-acquire-*` (.patch/.json/.md/findings.json) — evidence
  of the W10 work.
