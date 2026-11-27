# 07 — Verdict-live operator runbook (LQ-9)

**Wave:** FMD-CLOUD-024 / w13 (verdict-live)
**Audience:** the operator of the local host (GPU + model weights). This stream does not run in CI
beyond the offline replay path.
**Status: live runs exist, but the human audit gate has not passed.** Committed reports under
`reports/eval-verdict-live-*.json` record the operator runs; replay tests prove only the plumbing
and guards. Treat every live result as advisory until the worksheet gate in §4 passes.

The backend and its guarantees live in `fmd/eval/verdict_live.py`; the audit protocol this
implements is `04-verdict-audit-protocol.md` (§5 records what is now executable). The design of
the verdict stream is `01-eval-designs.md` §3.

## 1. Preconditions

- Workstation with the local model weights and ≥1 GiB VRAM headroom at 64k context, per
  `scripts/serve_local.sh` (Gemma 4 26B-A4B UD-IQ4_XS, llama.cpp, port 8010). The verdict-live
  profile reuses the same served model as extraction — no second server, no extra VRAM.
- `cd packages/pipeline && uv sync --all-extras` (the fetch ladder needs the crawl extras).
- Environment, per `docs/v1/LOCAL-QUEUE.md`:
  `export FMD_EXTRACT_PROVIDER=local FMD_SNAPSHOT_DIR=$HOME/fmd-snap2`
  The router profile for this stream is `verdict_live` in `config/models.yaml`; it defaults to the
  local Gemma and is switched with `FMD_VERDICT_LIVE_PROVIDER` / `FMD_VERDICT_LIVE_MODEL` only if
  you must. Do not point it at a paid API (invariant 10). The profile also sets
  `allow_fallback: false`: a local failure is re-raised unchanged and the router never tries the
  global `api` fallback. This preserves the backend's `json_schema_violation` retry instead of
  masking it as `unresolved_provider`.
- GPU free before you start: `nvidia-smi` shows no other tenant.

## 2. Start and probe the model server

```
scripts/serve_local.sh start
scripts/serve_local.sh probe    # 3 assertions: text, no thought leak, guided JSON (+nullable)
```

If the probe fails, stop here — a guided-JSON failure means every verdict turn will fall back to
JSON-instruct (which works but is weaker for the nested verdict schema). Fix the server first.

## 3. Run the live stream

From `packages/pipeline/` (`.env` first, per the local-queue convention):

```
set -a && . ../../.env && set +a
uv run python -m fmd.cli eval --eval-kind verdict --stream verdict-live \
  --report-out reports/eval-verdict-live-<ts>.json \
  --audit-sample 0.15 --audit-seed 13
```

- Hard caps (defaults): `--max-turns 12` model turns, `--max-fetches 6` fetches,
  `--max-searches 1` free SearXNG search, and `--max-maps 2` Firecrawl Local site
  maps per profile. They are enforced in code; raise them only with a reason
  recorded next to the report (doc 01 §3.5: cost and audit hours are the limiting
  factors, not compute).
- On the **penultimate turn**, if `citations == 0` and the fetched ledger is non-empty, the prompt
  issues a citation checkpoint: `cite_source` now, with no `search_web`, `map_site`, or `fetch_url`.
  The final turn is reserved for emit.
- The **final model turn is enforced in code as emit-only**. Its prompt forbids `search_web`,
  `map_site`, `fetch_url`, and `cite_source`, and asks for a verdict now using the evidence and
  citations already collected (or an honest `unknown` if they are insufficient). A non-emit action
  is rejected before dispatch, without execution, and the case degrades to `unknown` with
  `backend_run.stop_reason = final_turn_action_rejected`. A final `emit_verdict` with zero
  citations is rejected as well and degrades with `final_turn_zero_citations`, so an ignored
  citation checkpoint cannot publish an uncited success.
- Every fetch the model requests goes through the fetch ladder (robots.txt honoured, per-domain
  rate limits, truthful UA with contact). If a fetch is refused or fails, the model is told and
  the page cannot be cited.
- Expect minutes-to-tens-of-minutes per profile. Five canary profiles are committed; a pilot is
  exactly that — do not scale up before the audit gate below has passed once.
- UNKNOWN verdicts are normal and honest. A case whose `backend_run.stop_reason` is
  `final_turn_action_rejected`, `final_turn_zero_citations`, `turn_cap_reached`, or `model_error:*`
  degraded to unknown; do not re-label it.

The same invocation writes the audit worksheet `reports/audit-verdict-live-<ts>.json` (§5 of doc
04): the 10–20 % fraction is a target band, with a minimum of `min(2, n_cases)` entries. Sampling is
seeded and stratified so every `incorrect` and `partially_correct` verdict is included; each entry
carries the full evidence graph, the fetch ledger (url, status, sha256, fetched_at), and an empty
`operator_agreement` field. A one-case worksheet can contain only one entry and remains
`INSUFFICIENT` at the formal gate.

## 4. The audit worksheet loop (doc 04 §2)

Second-person adjudication: the auditor must not be the person who configured the run, and must
re-do the research independently (citable sources only) **before** reading the agent's rationale.

For each sampled case:

1. Re-derive a reference top-3 from public, citable sources.
2. Check every evidence pointer: fetch the URL, confirm the quote appears verbatim, compare the
   `sha256` against what you fetch (the worksheet's `fetch_ledger` records what the run stored —
   `sha256` over the raw response body). The harness already re-verified each pointer against its
   own ledger; your check is the independent second pass.
3. Fill `operator_agreement`:
   `{"agrees": true|false, "citation_ok": true|false, "reason": "short reason"}`
   `citation_ok: false` means you found a fabricated citation — that fails the whole run (doc 04
   §2.3): quarantine the verdict set, fix, re-run.

Then score the completed worksheet:

```
uv run python -m fmd.cli eval --eval-kind verdict --audit-score reports/audit-verdict-live-<ts>.json
```

Exit codes: `0` PASS (agreement ≥ 0.70), `1` FAIL (below the gate, or a fabricated-citation
finding), `3` INSUFFICIENT (fewer than two completed audits). The gate is REPORTED, never enforced
in CI — deciding what to do with a FAIL is a human act: below the gate the stream stays advisory
(its numbers may be published as diagnostics but must not feed a product claim).

## 5. Stop and leave the GPU free

```
scripts/serve_local.sh stop
nvidia-smi   # the GPU must be left free: ≤100 MiB used after the run
```

Nothing you started may outlive the run (model server, crawl processes, temporary worktrees). If
`nvidia-smi` shows more than 100 MiB in use after `stop`, kill the leftover process and record why
it survived in your run notes.

## 6. Honesty rules for live validation

- Label the run as live-model validation; do not present replay-path results as live-model behavior
  anywhere.
- Record the model profile actually served (`scripts/serve_local.sh status`), the report path, the
  worksheet seed, and the audit outcome in `docs/v1/STATE.md` per the local-queue convention.
- Commit back `reports/eval-verdict-live-<ts>.json` and the completed worksheet; never commit
  `.env` or any user-specific profile (invariant 1).
- If the run trips an invariant (a pointer the fetch layer never stored, a score in the model
  output, a circular citation), the harness catches it — fix the prompt or the cap, re-run, and
  note the fix in the report. Do not edit the scored cases by hand.
