# FMD-CLOUD-024-eval-live-orchestrator · live verdict backend + audit tooling · wave w13 · eval-live

~~~~
You are the ORCHESTRATOR cloud agent for one long run. You may spawn subagents and
coordinate them in parallel; everything lands on one branch and one PR.

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: ebef0f06061b1f1509a1d225bfe6f222735f1ad1
All output — code, comments, docs, commit messages, PR text, your report — English-only.

BRANCHING PROTOCOL
1. `git switch -c v1/eval-live` from main at the base SHA; verify `git rev-parse HEAD`;
   if it differs from BASE SHA, set BASE SHA := `git rev-parse HEAD` (true base for this run).
2. `git push origin v1/eval-live` immediately, before anything else.
3. Commit + push after every logical step. Never batch everything to the end.
4. Subagents (allowed, encouraged for parallel pieces) must branch off `v1/eval-live`
   (e.g. `v1/eval-live/backend`, `v1/eval-live/audit-tooling`) and open PRs INTO
   `v1/eval-live`. You review each sub-PR, verify gates at the branch tip, merge it, and
   resolve conflicts yourself. Keep `v1/eval-live` green at all times.
5. End: open the PR `v1/eval-live` → `main`.

READ FIRST
- AGENTS.md — invariants apply to eval code too: #1 no user-specific logic (profiles are
  fixtures/runtime input), #2 numbers computed by the harness, never emitted by the model,
  #3 UNKNOWN is a value, #9 politeness beats throughput, #10 no paid APIs.
- docs/v1/eval/01-eval-designs.md §3 (verdict-style eval design)
- docs/v1/eval/04-verdict-audit-protocol.md (the audit protocol you must make executable)
- packages/pipeline/fmd/eval/verdict.py — the `VerdictBackend` protocol, MockVerdictRouter,
  harness scoring, circularity guard. Scoring code is FROZEN: the live backend plugs into
  the same protocol without touching it.
- packages/pipeline/fmd/llm/** — the provider-agnostic model router. Model access goes
  through it, always.
- packages/pipeline/fmd/fetch/** — the politeness layer (robots.txt, per-domain rate limits,
  truthful UA). All live fetching goes through it.
- packages/pipeline/fmd/cli.py — `eval` parser + `_cmd_eval` (extend, don't restructure).

WHAT TO BUILD (the live verdict stream + the audit tooling that makes it a product signal)

**1. `fmd/eval/verdict_live.py` — LiveVerdictBackend**
- Implements the same `VerdictBackend` protocol as MockVerdictRouter.
- A BOUNDED research loop through the router: system prompt + profile + the product
  shortlist rows + a tool schema (fetch_url, cite_source); at most N model turns
  (configurable, default small, e.g. 8) and at most M fetches per profile. Hard limits,
  enforced in code.
- Every fetch goes through `fmd/fetch` (robots.txt honoured, rate limits, truthful UA with
  contact). No auth/CAPTCHA bypass. No paid APIs.
- Evidence pointers must carry url + quote + sha256 of the fetched content; the harness
  re-verifies sha256 against what the fetch layer actually stored.
- The model emits ONLY the structured verdict object (verdict label, reference_top3,
  evidence, notes). It never emits scores; harness scoring is untouched.
- UNKNOWN is a first-class output: the backend must be able to return `unknown` when its
  research is thin — and the prompt must instruct it to prefer `unknown` over guessing.
- The circularity guard (≥1 source outside the product's ingested evidence) stays
  harness-side, unchanged.

**2. CLI wiring**
- `fmd eval --eval-kind verdict --stream verdict-live [--report-out reports/eval-verdict-live-<ts>.json]`
- In tests/CI the router resolves to the mock provider; on the operator's host it resolves
  to the configured local profile. No new model config unless strictly required — prefer
  the existing router profiles.

**3. Audit tooling (make doc 04 executable)**
- `--audit-sample <fraction>` (e.g. 0.15): seeded RNG (seed recorded in the report) selects
  10–20% of verdict cases and writes an audit worksheet JSON: per sampled case the full
  evidence graph (URLs, quotes, sha256s, external flags) + an empty `operator_agreement`
  field.
- `fmd eval --eval-kind verdict --audit-score <worksheet.json>`: reads the completed
  worksheet, computes the human-agreement rate (harness-computed), compares against the
  ≥70% gate from doc 04, and reports PASS/FAIL/INSUFFICIENT. The gate is REPORTED, never
  enforced in CI (no operator exists in CI).

**4. Replay fixtures + tests (fully offline)**
- Committed scripted live sessions under `packages/pipeline/tests/fixtures/eval_verdict/live/`:
  router turns + fake fetch responses, replayed by a test backend. Tests never touch network
  or a real model.
- Required test coverage:
  - loop terminates at the turn/fetch cap;
  - evidence sha256 re-verification passes on fixture content and FAILS on tampered content;
  - circularity guard rejects a verdict whose evidence is all internal;
  - schema validation rejects a model output that tries to emit a score;
  - `unknown` verdicts are accepted and counted, never coerced;
  - audit worksheet sampling is deterministic under a fixed seed;
  - audit-score computes the agreement rate correctly on a completed fixture worksheet
    (one PASS case, one FAIL case);
  - CLI: `verdict-live` parses and runs end-to-end on the mock provider; exit codes.

**5. Docs**
- Append an "Implemented" section to `docs/v1/eval/04-verdict-audit-protocol.md` with the
  exact commands (do not rewrite its history).
- New `docs/v1/eval/07-verdict-live-runbook.md`: the operator's host sequence —
  `scripts/serve_local.sh start && probe`, env (`FMD_EXTRACT_PROVIDER=local`,
  `FMD_SNAPSHOT_DIR=$HOME/fmd-snap2`), the verdict-live command, the audit worksheet loop,
  `scripts/serve_local.sh stop`, and the rule that the GPU must be left free (≤100 MiB)
  after the run. Note explicitly: the live stream has NEVER run against a real model yet;
  the first real run is the operator's (LQ-9).

GATES (all must pass at the branch tip; report counts)
- `cd packages/pipeline && uv sync --all-extras`
- `uv run ruff check .`
- `uv run pytest -q` (offline; report passed/skipped counts)
- `uv run python -m fmd.cli eval --golden ../../data/golden` (provenance coverage 1.0)
- dry-runs exit 0: nirf-proxy, naac-proxy, self-drift, cv-stability, verdict-fast,
  expert-panel, AND verdict-live on the mock provider.
- The DB-backed pytest gate cannot run in your sandbox (no postgres) — say so honestly;
  the operator runs it at intake.

SCOPE
You may touch: `packages/pipeline/fmd/eval/**`, `packages/pipeline/fmd/cli.py`,
`packages/pipeline/fmd/llm/**` (small additive surfaces only — prefer adapters in
`fmd/eval/` over changing the router), `packages/pipeline/tests/**`,
`packages/pipeline/tests/fixtures/eval_verdict/**`, `docs/v1/eval/**`, `reports/**`,
`config/models.yaml` ONLY if the router needs a verdict profile entry (avoid if possible).
Read-only: `fmd/fetch/**`, `data/truth/**`, `docs/product-development-cycle/**`.
Do NOT touch: `fmd/extract/**`, `fmd/fetch/**` (modify), `apps/**`, `infra/**`, `.env`,
`config/predicates.yaml`, `config/sources.yaml`, `db/migrations/*`, `docs/v1/LOCAL-QUEUE.md`
(host-owned).

HONESTY RULES
- No live model exists in your sandbox: do not claim any live-model behavior was observed.
  Everything you prove, prove on the mock/replay path, and label it as such.
- If a subagent's PR breaks a gate, you fix or reject it — the branch tip is your
  responsibility.

REPORT (chat, max 400 words): what was built (backend / CLI / audit tooling / docs),
test names + counts, gates status, PR URL, and the top 3 honest gaps you noticed while
building (e.g. limits that will bite on real research loops, fetch-layer friction, prompt
design risks).
~~~~
