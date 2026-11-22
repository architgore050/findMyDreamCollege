# 04 — Verdict-stream audit protocol (human audit, anti-hallucination, circularity)

**Wave:** FMD-CLOUD-024 / v1 eval builder
**Covers:** the verdict stream built in `fmd/eval/verdict.py` (§3 of `01-eval-designs.md`)
**Status:** the fast canary (`verdict-fast`) is built and runs offline on the mock router. The slow,
live-research variant is designed here but **not built** in this wave.

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

- Spot-check a **random 10–20 % of verdicts per run**, stratified so that every `incorrect` and every
  `partially_correct` verdict in the run is reviewed (failures are the interesting cases; the report
  already carries their full evidence graph).
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

## 4. Slow variant (designed, not built)

The slow variant replaces the mock backend with a `VerdictBackend` backed by the model router (the
protocol is already defined in `fmd/eval/verdict.py`), runs live research under the pipeline's
robots/politeness rules, and costs minutes-to-tens-of-minutes per profile. Build order when it is
funded: 5-profile pilot → human audit (this protocol) → only then expansion. Cost per profile and
audit hours are the limiting factor, not compute.
