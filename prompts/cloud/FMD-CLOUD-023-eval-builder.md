# FMD-CLOUD-023-eval-builder · build the v1 product-eval streams · wave w12 · eval-builder

~~~~
You are the one-shot cloud coding agent building the v1 **product evals**. Research is done
(see the four files in `docs/v1/eval/`: `00-research-summary.md`, `01-eval-designs.md`,
`02-findings-viable.md`, `findings.json`). This wave builds the evals as code + docs.

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: 86d3248431a94970ed99e6d9f742f2af516cc4b5
Do not push until you open the main PR at the end; commit and push often.
All output must be English-only.

BRANCHING
1. `git switch -c v1/eval-builder` from main at the base SHA, verify `git rev-parse HEAD`.
2. `git push origin v1/eval-builder` immediately (before doing anything else) — you will get PRs
   into this branch later.
3. Build in this branch; commit + push after every logical step. Do not wait for the end.
4. Subagents (allowed) may branch off `v1/eval-builder` and open PRs into it.
5. End: open the PR from `v1/eval-builder` → `main`.

READ FIRST
- AGENTS.md (invariants — they apply to evals; esp. #1 no user-specific logic in eval harnesses,
  #2 numbers computed by harness, #3 UNKNOWN is a value, #4 truth data carries provenance)
- docs/v1/eval/00-research-summary.md, 01-eval-designs.md, 02-findings-viable.md, findings.json
- docs/product-development-cycle/08-eval-methodology.md (existing harness: golden set, provenance, drift)
- packages/pipeline/fmd/eval/** (golden.py, metrics.py, run.py, drift.py — extend this code)
- fmd/cli.py `eval` parser + `_cmd_eval` (extend CLI with `--eval-kind/--stream`)
- fmd/orchestrate/orchestrator.py (extend the pipeline wiring if needed)
- AGENTS.md "applying changes to main" — commit often to `v1/eval-builder`.

BUILD TARGETS (from `docs/v1/eval/01-eval-designs.md` + `02-findings-viable.md`)

You build what makes the product answerable. The research said no mature public outcome dataset
exists for Indian B.Sc. MLT; the honest eval ladder is proxies first, then audited research, then
opt-in/expert data later. Build all of:

**1. Fast stream — proxy ranking + self-drift + cross-validation (data/truth/, `fmd eval`)**
- `data/truth/nirf/`: yearly medical + overall NIRF tables with provenance (`truth_version`,
  `scope`, `as_of`, `source_url`, `sha256`, `fetched_at`) + `data/truth/naac/` grades table
  (5-year cycle). Include an acquisition README (how to re-fetch: NIRF rank lists from
  nirfindia.org PDFs, NAAC grade lists from naac.gov.in) and a `docs/v1/eval/03-truth-ingestion.md`
  describing the ingestion contract. Ingest the most recent usable year as the committed truth file.
- `fmd/eval/truth.py`: loader + top-k/Spearman/Kendall metrics on institution→programme overlap.
- `fmd eval --eval-kind product --stream nirf-proxy`: for a product shortlist, build reference
  ranking from NIRF/NAAC and score: top-k overlap (k=3,5), rank correlation on overlap,
  "best available" hit rate; UNKNOWN when no entry.
- `fmd eval --eval-kind product --stream naac-proxy`: same on NAAC grades (CGFund proxy on the
  Institutional programme, medium maturity — same rule, honest label).
- `fmd eval --stream self-drift`: versioned shortlist snapshots (file path + sha256 of claims),
  diff since last run (rank change magnitude, % evidence-driven vs unexplained).
- `fmd eval --stream cv-stability`: leave-N-out over the product's own candidate set (Kendall-tau
  under leave-N-out and ±ε weight noise; fraction of top-k surviving).
- Reports: `reports/eval-<stream>-<ts>.json` per run (schema in `docs/v1/eval/` sketches).

**2. Verdict stream — the product-quality stream (the audited research-agent eval)**
- `fmd/eval/verdict.py`: implements §3 of `01-eval-designs.md` — for each profile, the verdict
  agent (model; test with the mock router in CI) retrieves the product shortlist, selects
  evidence-rich + thin/no-evidence Programme rows, does fast research (citable web sources),
  builds `reference_top3`, and emits structured `verdict: correct|partially_correct|incorrect|unknown`.
- Harness-side scoring: agreement rate of verdict==correct; top-3 overlap with the agent's
  reference_top3; UNKNOWN coverage fraction; evidence audit trail.
- `fmd eval --eval-kind verdict --stream verdict-fast`: the canary mode (fast, deterministic,
  no live web). Profile fixtures: `tests/fixtures/eval_verdict/profiles/*.json`.
- **Audit/anti-hallucination:** no verification step in cloud CI; document the human-audit protocol
  (`docs/v1/eval/04-verdict-audit-protocol.md`): human spot-check of a random 10–20% of verdicts;
  report includes full evidence graph for every non-correct case; circularity guard: agent must
  cite ≥1 source not in the product's ingested evidence.

**3. Slow stream — expert panel + opt-in survey instrument (the "deals/proprietary" path)**
- `fmd/eval/expert_panel.py`: the panel-evaluation kit per `02-findings-viable.md` (Krippendorff /
  ICC, top-k overlap; interval for effectiveness) with fixtures + audit protocol.
- `docs/v1/eval/05-opt-in-survey-instrument.md`: the opt-in survey *design* (DPDP-compliant,
  programme + graduation year only, never user identity; wording for graduates asking about their
  own programme; incentive options; retention approach). NOT implementation — the design is the
  deliverable.
- `docs/v1/eval/06-acquisition-paths.md`: honest research — who could provide programme-level
  outcome data to an MLT-focused tool (hospitals/departments, state bathing bodies?, Industry
  bodies, MOHFW)? What MOUs/compliances are needed; realistic timeline. Written by the agent
  (its research capabilities; keep it honest — that data largely doesn't exist).

**Integration (`fmd eval` CLI — §4 sketch from `01-eval-designs.md`):**
```
fmd eval --eval-kind pipeline|product|verdict --stream golden|nirf-proxy|naac-proxy|self-drift|cv-stability|verdict-fast|expert-panel --fixture-path ... --report-out ...
```
`--eval-kind pipeline` (default) = current behaviour. Scores always harness-computed; the model
emits only the structured verdict object.

**TESTS (offline; no network; no real model — the verdict agent runs on the mock/replay router):**
- Truth tables load/validate (schema, provenance fields, forward-compat), and the truth-ingestion
  contract from `03-truth-ingestion.md` is exercised by fixtures.
- nirf/naac proxy metrics on a committed example: sanity — top-k overlap ≥ 1 on a constructed case,
  UNKNOWN when entry missing.
- self-drift: snapshot diff logic (same snapshot → 0 changes).
- cv-stability: on a small synthetic candidate set, leave-N-out computes a number in [0,1].
- verdict-fast on mocked router: produces correct verdict JSON; harness scores; audit trail
  (evidence pointers) includes URL + sha256.
- expert-panel scoring on a fixture.
- CLI: all three eval-kinds parse; exit codes OK.

**GATES (all must pass; report counts):**
- `cd packages/pipeline && uv sync --all-extras`
- `uv run ruff check .`
- `uv run pytest -q` (offline)
- `DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q`
- `uv run python -m fmd.cli eval --golden ../../data/golden` (provenance 1.0)
- `uv run python -m fmd.cli eval --eval-kind product --stream nirf-proxy --dry-run` (offline parse check)

**PROTOCOL:**
- Commit + push after every logical step (loader, metrics, CLI, each stream, tests).
- End: open PR `v1/eval-builder` → `main` via gh. Include: summary of what was built per stream,
  verification table (test names + pass counts), "where each number came from", and honest notes:
  slow-stream acquisition paths are designs, not reality; no stream currently proves high-confidence
  programme-level outcomes — this is consistent with the research and we will NOT claim otherwise.

RUN OUTPUT (repo root, UTC): not needed — agent commits to `v1/eval-builder`.

SCOPE: you may touch `packages/pipeline/**` (fmd/eval/**, fmd/cli.py, fmd/orchestrate/**),
`config/**` (bulk_extract prompt already has context; don't change models.yaml unless a verdict
profile is needed — use existing local profile), `data/truth/**`, `docs/product-development-cycle/`
(nothing except new eval docs), new `docs/v1/eval/*` additions, `reports/**` (eval outputs),
`packages/pipeline/tests/**`, `scripts/` (add optional helpers? — allowed but optional).
Do NOT touch: `fmd/extract/**` (except nothing), `fmd/fetch/**` (read-only OK), `apps/**`, `infra/**`,
`.env`, `config/predicates.yaml`, `config/sources.yaml` (truth tables are separate), `db/migrations/*`.

REPORT (chat, max 400 words): what each stream does, test names + counts, gates status, PR URL,
and the top 3 honest gaps you noted while building (e.g., "nirf/naac proxy has no programme-level
coverage for paramedical; verdict-fast is the only high-signal stream").
~~~~
