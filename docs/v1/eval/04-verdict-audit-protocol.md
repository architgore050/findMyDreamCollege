# 04 — Verdict-stream audit protocol (human audit, anti-hallucination, circularity)

**Wave:** FMD-CLOUD-024 / v1 eval builder
**Covers:** the verdict stream built in `fmd/eval/verdict.py` (§3 of `01-eval-designs.md`)
**Status:** the fast canary (`verdict-fast`) and the live-research variant are built. Replay tests
prove the plumbing and guards; committed live reports prove the loop runs, but research quality
remains advisory until the human audit gate below passes.

## 1. What exists today

`fmd eval --eval-kind verdict --stream verdict-fast` runs the canary:

1. Loads the committed profile fixtures (`packages/pipeline/tests/fixtures/eval_verdict/profiles/*.json`) —
   five synthetic profiles covering `correct`, `partially_correct`, `incorrect`, `unknown`, and a
   correct case with a thin/no-evidence honest negative.
2. The verdict agent runs on the **mock/replay router** (`MockVerdictRouter`): it replays the
   committed research record and validates it against the verdict schema. No network, no live model,
   deterministic in CI.
3. The harness computes every score: agreement rate (`verdict == correct`), top-3 overlap between the
   product shortlist and the agent's `reference_top3`, UNKNOWN coverage, and per-profile evidence
   audit. The model never scores anything (invariant 2).

**What the canary proves:** the plumbing, schemas, scoring and audit machinery work, and the audit
trail is structurally complete. **What it does not prove:** that a real model produces non-hallucinated
verdicts. That claim requires the live variant plus the human audit below — until then no
product-quality claim may cite this stream.

## 2. Human audit protocol (for the live variant)

### 2.1 Sampling

- Spot-check a **random 10–20 % of verdicts per run** as the target band, stratified so that every
  `incorrect` and every `partially_correct` verdict in the run is reviewed (failures are the
  interesting cases; the report already carries their full evidence graph).
- Always sample at least **`min(2, n_cases)`** entries. Small runs therefore oversample the target
  band: five cases at a clamped 20 % still yield two entries, while a one-case run yields one and
  the formal agreement gate remains `INSUFFICIENT` because it needs two completed audits.
- Second-person adjudication, matching the golden-set practice (doc 08 §12): the auditor must not be
  the person who configured the run, and must re-do the research independently (citable sources only)
  before reading the agent's rationale.

### 2.2 Auditor tasks, per sampled verdict

1. Re-derive a reference top-3 for the profile from public, citable sources.
2. Check every evidence pointer: does the URL resolve to the claimed quote? Does the recorded
   `sha256` match the content? (The report stores `url + quote + sha256 + fetched_at` per pointer so
   this is mechanical.)
3. Agree/disagree with the verdict label, with a short reason. Disagreements are recorded per verdict,
   not averaged away.

### 2.3 Acceptance and what happens on failure

- Per-run inter-rater agreement (auditor vs agent) is reported next to the agreement rate.
- If agreement on the correct/incorrect split falls below **70 %** (the threshold from
  `02-findings-viable.md`), the stream stays **advisory**: its numbers may be published as diagnostics
  but must not feed any product claim, and the next run needs better tooling or a smaller profile set.
- An audit finding of a fabricated citation (URL does not contain the quote, or sha256 mismatch) fails
  the whole run: the verdict set is quarantined and re-run after the fix.

### 2.4 Reporting

Every run report (`reports/eval-verdict-fast-<ts>.json`) carries, per profile:

- the full structured verdict (label, rationale, `reference_top3`, `selected_programmes` with their
  evidence tiers, `unknown_aspects`);
- the complete evidence graph — every pointer with `url`, `title`, `quote`, `sha256`, `fetched_at`,
  `source_kind`, and the `external` flag — for **every** non-correct case, and sampled correct cases;
- the audit block: `n_evidence`, `n_external`, and the list of audit failures.

## 3. Anti-hallucination and circularity guards (enforced in code)

- **Every pointer must carry URL + quote + sha256** — a pointer without them is schema-invalid, so a
  verdict that invents an uncitable source cannot pass validation.
- **Circularity guard:** each profile fixture lists the product's own ingested evidence URLs; the
  verdict must cite **at least one source marked `external`**, and an `external` pointer whose URL is
  in the ingested set is an audit failure. This is the mitigation for the trap the research flagged
  (the agent re-using the product's mined claims and confirming it).
- **UNKNOWN is a value with content:** a `verdict: unknown` must name its `unknown_aspects`; it counts
  toward UNKNOWN coverage and never toward agreement.
- **Leapfrog / style drift:** the live variant must pin the model profile, `prompt_sha256`, and the
  fixture `as_of` per verdict, and store the full verdict object, so a later model cannot silently
  rewrite history. The fast variant already stores everything; the live variant inherits the schema.

## 4. Slow variant (original design; implementation status in §5)

The slow variant replaces the mock backend with a `VerdictBackend` backed by the model router (the
protocol is already defined in `fmd/eval/verdict.py`), runs live research under the pipeline's
robots/politeness rules, and costs minutes-to-tens-of-minutes per profile. Build order when it is
funded: 5-profile pilot → human audit (this protocol) → only then expansion. Cost per profile and
audit hours are the limiting factor, not compute.

## 5. Implemented (FMD-CLOUD-024 / w13) — this protocol is now executable

The slow variant above is built: `fmd/eval/verdict_live.py` (`LiveVerdictBackend`) implements the
same `VerdictBackend` protocol as the fast canary, so §2 above is no longer a paper protocol — it
has CLI surfaces. The original design above is retained for context; this section records the
executable behavior.

What exists (harness-side guarantees unchanged):

- **Bounded research loop** through the provider-agnostic router: one action per model turn —
  `fetch_url` (routed through `fmd/fetch`: robots.txt, per-domain rate limits, truthful UA; no
  auth/CAPTCHA bypass), free/leads-only `search_web` and `map_site`, `cite_source` (verbatim quote of
  a page fetched *in that run*), or `emit_verdict`. Hard caps in code: `--max-turns` (default 12),
  `--max-fetches` (default 6), `--max-searches` (default 1), and `--max-maps` (default 2).
- **No paid model fallback:** `profiles.verdict_live.allow_fallback: false` in `config/models.yaml`
  disables the global local-to-API fallback for this profile (invariant 10). The router re-raises
  the original `ProviderError`, so a local `json_schema_violation` reaches the backend's existing
  schema-retry instead of being masked by an unresolved API provider.
- **Citation checkpoint and hard final turn:** on the penultimate turn, if no citation exists but
  the fetched ledger is non-empty, the prompt requires `cite_source` immediately and forbids
  search/map/fetch. The final turn is enforced in code: only `emit_verdict` is dispatched. Any
  other final-turn action is rejected without execution and the case degrades to `unknown` with
  `stop_reason: final_turn_action_rejected`. A final `emit_verdict` with zero citations is also
  rejected rather than published as a successful emission; it degrades with
  `stop_reason: final_turn_zero_citations`.
- **The model never scores and never hashes.** The turn schema is `extra="forbid"`, so a response
  carrying a score or an undeclared field is rejected as a schema violation. `sha256` of every
  evidence pointer is computed by the harness over the exact bytes the fetch layer stored, and
  `run_verdict_live` re-verifies every pointer against that ledger (a tampered page or a pointer
  the fetch layer never stored fails the case).
- **UNKNOWN is a first-class output** (§3): the system prompt instructs the model to prefer
  `unknown` over guessing, and a loop that hits a cap or a fatal model error degrades to an honest
  `unknown` verdict whose `unknown_aspects` name the stop reason. It is never coerced to a label.
- **Circularity guard untouched**: still enforced by `audit_verdict` in `fmd/eval/verdict.py`.
- **Replay first**: in tests/CI the `verdict_live` router profile resolves to the mock provider and
  the stream replays committed scripted sessions
  (`packages/pipeline/tests/fixtures/eval_verdict/live/sessions/`). Operator live runs are committed
  under `reports/eval-verdict-live-*.json`; replay results alone are not evidence of live-model
  research quality, and the stream stays advisory until the human audit passes.

Exact commands (all from `packages/pipeline/`; offline ones need no model host):

1. Offline replay run (CI parity; exit 0, no network, no model):
   `FMD_VERDICT_LIVE_PROVIDER=mock uv run python -m fmd.cli eval --eval-kind verdict --stream verdict-live --dry-run`
2. Live run with report (operator host, model server up — see `07-verdict-live-runbook.md`):
   `uv run python -m fmd.cli eval --eval-kind verdict --stream verdict-live --report-out reports/eval-verdict-live-<ts>.json`
3. §2.1 sampling — pass `--audit-sample 0.15 --audit-seed 13` on the run invocation (the worksheet
   is built from that run's cases; sampling is deterministic for a given seed, the seed is recorded
   in the worksheet). Writes `reports/audit-verdict-live-<ts>.json`: per sampled case the full
   evidence graph (url, title, quote, sha256, fetched_at, source_kind, external) plus the fetch
   ledger, with an empty `operator_agreement` field. Every `incorrect` and `partially_correct`
   verdict is always sampled; the seeded RNG fills the rest up to the clamped 10–20 % target band,
   with a minimum of `min(2, n_cases)` entries. Small runs may therefore exceed the band, and a
   one-case run remains `INSUFFICIENT` at the formal gate.
4. §2.3 gate — after the auditor completes the worksheet (fill `operator_agreement` with
   `{"agrees": bool, "citation_ok": bool, "reason": str}`):
   `uv run python -m fmd.cli eval --eval-kind verdict --audit-score reports/audit-verdict-live-<ts>.json`
   prints the harness-computed agreement rate and PASS/FAIL/INSUFFICIENT (exit 0/1/3) against the
   ≥70 % gate. A `citation_ok: false` finding fails the whole run regardless of the rate (§2.3).
   The gate is REPORTED, never enforced in CI (no operator exists in CI).

Limitation, stated honestly: the worksheet must be sampled in the same invocation as the run
(`--audit-sample` reads that run's cases); regenerating a worksheet from an already-written report
JSON is not implemented — re-run with the flag if the report was written without it.
