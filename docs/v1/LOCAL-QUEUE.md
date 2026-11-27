# v1 LOCAL-QUEUE — runs only the operator's machine can do

Each entry: why local, exact command + env, expected output, what to commit
back, and the waves it blocks. The cloud sandbox has **no Docker, no GPU, no
model weights**; everything here needs one of those.

### LQ-1: real `make up` compose verification (incl. DNS-wiring fix of wave 1)
Why local: **real Docker** — the cloud sandbox has no docker daemon.
Command:
  cd /path/to/findMyDreamCollege
  docker compose -f infra/docker-compose.yml down -v
  make up          # or: docker compose -f infra/docker-compose.yml up -d --build
  docker compose -f infra/docker-compose.yml ps
  docker compose -f infra/docker-compose.yml logs api | grep -i "name resolution"
  make migrate && make seed
  curl -sf localhost:8000/api/health
  curl -sfI localhost:3000 | head -1
  API_BASE_URL=http://127.0.0.1:8000 WEB_BASE_URL=http://localhost:3000 scripts/acceptance.sh --json-out reports/acceptance.json
Expected: `api` status Up (healthy); zero `Temporary failure in name resolution`
lines; `/api/health` → `{"status":"ok"}`; web answers HTTP 200 on :3000;
acceptance report has no FAILED (SKIPs allowed with reasons).
Commit back: `reports/acceptance.json` plus any compose correction the run
forces (as a new PR; do not hand-edit the wave-1 commit).
Blocks: wave 1 verification, definition-of-done #1.

### LQ-2: real crawl + extraction + scoring pass (GPU + model + real network)
Why local: **GPU (RTX 5080) + llama-server 26B-A4B weights + real egress.**
(The family lanes this crawl exercises: fee_regulator, state_counselling x12, institution_sites — the fee/seat/placement/admission lanes of PLAN #6.)
Command:
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
Expected: crawl yields real snapshots (frontier non-empty); extract produces
claims with provenance 1.0 non-vacuous; `programmes` count > 0 in SQL.
Commit back: golden-set rows for any newly-verified documents
(`data/golden/...`), updated eval report under `reports/`, counts in
`docs/v1/STATE.md` (claim/programme totals).
Blocks: definition-of-done #1–#5, #7 (evidence targets), wave 5 real reduction.

### LQ-3: operator evidence targets (need a populated corpus)
Why local: depends on LQ-2's corpus; **model + real network**.
Command: (after LQ-2)
  psql "$DATABASE_URL" -c "select count(*) from programmes where in_scope"
  psql "$DATABASE_URL" -c "select entity_id, count(distinct component) from component_coverage group by 1 having count(distinct component) >= 3"
Expected: in-scope programmes ≥ 150; ≥ 50 programmes with ≥ 3 of 7 scored
components. If below target, crawl more (never fabricate).
Commit back: counts recorded in `docs/v1/STATE.md` § evidence targets.
Blocks: definition-of-done #7, release decision (rc vs final).

### LQ-4: real-GPU validation of the resource-phased batched runner (wave 4)
Why local: **GPU** — tests fake all GPU paths; only a real run proves the
windowing is safe (one CUDA consumer at a time).
Command:
  set -a && . .env && set +a
  export FMD_SNAPSHOT_DIR=$HOME/fmd-snap2
  uv run python -m fmd.cli extract --batched --batch-pages 8   # Docling+OCR inside the server-stopped GPU window
Expected: `nvidia-smi` shows never more than one CUDA consumer; run summary
`ok:true`; no `docling_cuda_oom`.
Commit back: run summary JSON (`--json`? check `fmd extract --help` for summary output; stdout JSON) +
any timing constants adjusted from measurement.
Blocks: wave 4 verification.

### LQ-5: T2 OCR real-engine cache validation (wave 3)
Why local: **model/OCR weights** (Qwen3-VL ~10.6 GB or RapidOCR engine on real scans).
Command:
  set -a && . .env && set +a
  export FMD_SNAPSHOT_DIR=$HOME/fmd-snap2
  uv run python -m fmd.cli acquire --limit 5 --tier T2   # first run: live
  uv run python -m fmd.cli acquire --limit 5 --tier T2   # second run: cache hits
Expected: second run reports cache hits (identical results, ~0 engine time);
`fmd doctor`/cache stats show hit count ≥ documents cached.
Commit back: `reports/` cache stats, any bug fix discovered.
Blocks: wave 3 real-world verification.

### LQ-6: operator acceptance gate run (phase-4 live, doc 12 §4)
Why local: real stack + real corpus (doc 12: "a DB-count assertion that passes
at 0 is a failure").
Command: doc 12 §4 script verbatim (compose down -v → up → migrate/seed →
crawl/extract/verify/score → eval → web test), outputs to
`reports/phase4-acceptance.md`.
Expected: all commands exit 0; programmes > 0; claims > 0; provenance 1.0
non-vacuous.
Commit back: `reports/phase4-acceptance.md`.
Blocks: acceptance-checklist "all PASS or explicitly BLOCKED" claim, release.

### LQ-7: 26B semantic classification pass over `classify-report` residual (wave 5)
Why local: **26B model** — GPU + model weights (~12.7 GB GGUF); the cloud has neither.
The deterministic family chain and the admission contract are merged (PLAN #5); the model
itself must run here to propose admissions for `below_floor`/`ambiguous`/`no_signal` rows.

Command:
```
cd packages/pipeline && set -a && . ../../.env && set +a
export FMD_EXTRACT_PROVIDER=local FMD_SNAPSHOT_DIR=$HOME/fmd-snap2
scripts/serve_local.sh start && scripts/serve_local.sh probe
uv run python -m fmd.cli classify-report --with-text --json-out /tmp/classify-report.json
# Review the residual buckets in /tmp/classify-report.json.
# Validate the plausible admissions you want to apply (verbatim quote from the
# document's text sample, known family, no second family proposed) into a JSON
# list of {"document_url", "family", "quote"} at /tmp/admissions.json.
uv run python -m fmd.cli plan --admit-evidence /tmp/admissions.json
```
Expected: report lists `would_classify_with_text` / `below_floor` / `ambiguous` /
`no_signal` per bucket; `plan --admit-evidence` applies every admitted document as
`family_source=semantic` (`ambiguous` and `no_signal` are refused, never guessed).
`scripts/serve_local.sh stop` at the end.
Commit back: `reports/classify-report.json` + `/tmp/admissions.json` (review by operator);
DB counts in `docs/v1/STATE.md` if updated.
Blocks: wave-5 residual reduction, DoD #2/#7 "useful corpus" from real admitted rows.

### LQ-8: NAAC truth ingestion (network-blocked; opportunistic retry)
Why local: naac.gov.in was unreachable from the cloud sandbox (PR #11 wave) **and**
from this host (probe 2026-11-23: HTTP 000 after 15s). Ingestion needs a working
egress path to naac.gov.in; retry opportunistically from here (or via another route).
Command: (once reachable; contract in `docs/v1/eval/03-truth-ingestion.md` +
`data/truth/naac/README.md`)
```
curl -sI --max-time 25 https://www.naac.gov.in/   # probe first
# fetch the grades list page, commit the raw copy, build data/truth/naac/naac_grades.yaml
# with truth_version>=1 + source_sha256 + fetched_at per the ingestion contract, then:
cd packages/pipeline && uv run python -m fmd.cli eval --eval-kind product --stream naac-proxy
```
Expected: `naac_grades.yaml` with real entries + raw page copy whose sha256 the loader
re-verifies; naac-proxy report with non-trivial coverage (currently honest unknown=1.0).
Commit back: `data/truth/naac/**` + `reports/eval-naac-proxy-*.json`.
Blocks: naac-proxy signal, slow-stream ladder step 2. Do NOT fabricate grades.

### LQ-9: verdict live variant + human-audit gate
**Status (2026-11-28):** post-fix2 live run completed: **4/5 emitted,
0 turn-cap stops**, 5 searches / 4 maps, schema retries work, no paid fallback.
Two cases still failed the external-citation audit; all labels are `unknown`.
A **scoreable 2-entry worksheet** (`canary-01`, `canary-02`) is ready at
`reports/audit-verdict-live-post-fix2.json`; `operator_agreement` is empty.
**Operator human audit is pending.**
Why local: real 26B model + real web research + operator judgment; the ≥70%
human-agreement gate (docs/v1/eval/04-verdict-audit-protocol.md) is an operator act.
Command: (with SearXNG + Firecrawl Local + MCP running per
`docs/v1/eval/09-free-web-discovery-runbook.md`)
```
cd packages/pipeline && set -a && . ../../.env && set +a
scripts/serve_local.sh start && scripts/serve_local.sh probe
uv run python -m fmd.cli eval --eval-kind verdict --stream verdict-live \
  --audit-sample 0.4 --report-out ../../reports/eval-verdict-live-<ts>.json
scripts/serve_local.sh stop
# operator: complete operator_agreement in the worksheet for the sampled cases
uv run python -m fmd.cli eval --eval-kind verdict --audit-score ../../reports/audit-verdict-live-<ts>.json
```
Expected: live verdict report; operator agreement ≥70% on the audited subset;
circularity guard (≥1 external source) passes on every audited case.
Commit back: `reports/eval-verdict-live-*.json` + audit notes in `docs/v1/STATE.md`.
Blocks: verdict stream becoming a product-quality signal (human agreement + a
non-unknown verdict distribution still need to be demonstrated).
