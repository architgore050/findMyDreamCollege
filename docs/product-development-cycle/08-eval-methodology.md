# 08 — Evaluation methodology

Status: WS-EVAL harness for brief §19. Offline, deterministic, no network, no
database, no model call. Run it with:

```
make eval
# equivalently
cd packages/pipeline && uv run python -m fmd.cli eval \
  --golden ../../data/golden --report-out ../../reports/eval.json
```

`--baseline <previous-report.json>` adds drift diagnostics. `--fail-under`
(default `1.0`) is the provenance hard gate; a failure exits non-zero and the
report records `ok: false`.

The harness replays the real validation path on committed fixtures:
recorded guided JSON → family schema → `validate_document` → claims. It does not
re-extract with a model, so a score here is a property of the fixtures and the
gates, not a measurement of model quality.

## Golden set

`data/golden/programmes/*.yaml`. An entry names an institution/programme and a
committed fixture (or a cited, dated source). Each expected field carries
`value`, `unit`, `quote`, `status`, and `expectation`.

**Current committed state (2026-11-16): 4 entries, 14 verified rows, 1
unverified row, 0 citation-only rows.** Every verified quote was checked
verbatim against its committed fixture by the loader. Six verified rows are
inventory/ranking facts with no extraction replay (UGC, RUHS, NIRF); only the
placement-report entry is measured for extraction accuracy.

A row is one of:

| `status` | `expectation` | Meaning |
|---|---|---|
| verified | value | The fixture/citation contains the value and quote. |
| verified | abstain | The document is silent; `absent_terms` must really be absent. |
| verified | reject | The value must be rejected; `reason` is recorded. |
| unverified | value (value null) | Not checked by a human yet. Counted, excluded from accuracy. |

`status: unverified` with a non-null value is a loader error. So is a verified
quote that does not occur in its fixture. That is the point: a typo cannot
become "ground truth". A row that cannot meet either traceability route must
stay unverified; the harness will report fewer metrics rather than a prettier
number.

## Metrics

**Extraction span accuracy / P/R/F1.** A field is a true positive only when
(a) the predicted quote resolves exactly at `[char_start, char_end)` in the
extraction text, (b) it equals the golden quote, (c) the value matches
(numeric tolerance `1e-9`), and (d) the normalised unit matches. Precision =
TP/(TP+FP), recall = TP/(TP+FN). Missing predictions are false negatives, never
zeros.

**Unit normalisation accuracy.** Among predicted fields with an expected unit,
the share whose unit normalises to the golden unit (`percent` → `ratio`, `inr` →
`INR`) using the same vocabulary as the extraction gate. This proves the
normaliser agrees with the curated expectation; it does not prove the document's
unit was interpreted correctly beyond the quoted span.

**Provenance coverage.** Resolved published numbers ÷ all published numbers. A
published number is a numeric claim with `method` in `{extracted, derived}`. It
resolves when its quote is non-empty and `text[start:end]` is exactly that quote.
`modelled` numbers are excluded and counted; they are labelled, not passed off as
evidence. A run with zero published numbers reports `1.0` and `vacuous: true`, so
an empty pipeline cannot look healthy by accident. The gate compares this value
to `--fail-under`; 1.0 is the default and the only acceptable release value.

`--require-snapshot` is available to the metric (URL + `fetched_at` + sha256); the
offline fixture run does not use it because a fixture is not a retrieved URL.
That distinction is reported (`require_snapshot: false`), not hidden.

**Rejection correctness.** Using golden `expectation: reject` rows and the
pipeline's rejection records: TP = expected and rejected, FP = not expected but
rejected, FN = expected but accepted. The harness currently has zero reject rows,
so P/R/F1 are `null` — not `1.0`.

**Interval coverage.** Share of `(truth, lo, hi)` observations where
`lo ≤ truth ≤ hi`, with nominal 0.8. It reports `measurable: false` when there are
no interval predictions. The committed fixtures contain a median but no spread
and no `n`, so the offline run correctly reports `n=0`; inventing an interval
would defeat the metric.

**Abstention rate / correctness.** Rate = missing predictions ÷ scored fields.
Correctness = abstentions that were *supposed* to happen (golden
`expectation: abstain`) ÷ all abstentions. A pipeline that abstains everywhere
gets recall 0 and correctness 0 on value rows; abstention is not a free pass.

**Drift.** `fmd/eval/drift.py` computes PSI (10 equal-width bins; <0.10 stable,
≥0.25 significant) and two-sample KS (asymptotic p-value) across sub-score
distributions, plus top-20 Jaccard churn. Every golden entry that is extractable
is a **canary**, replayed on every run with a stable `claims_hash`. A canary
metric that moves while the hash is unchanged (and versions did not change) is
flagged `unexplained`; a changed hash is labelled `claims_diff`. Reports are
JSON so consecutive runs diff cleanly.

## What this proves

- Every claim the harness publishes from the committed fixtures has a quote that
  resolves at its exact span, and extraction agrees with 14 curated rows.
- The golden loader cannot silently accept a fabricated verified value.
- The provenance gate is wired to the process exit code.
- Drift logic distinguishes a claims change from an unexplained movement.
- The whole run is reproducible offline and byte-identical for a fixed checkout.

## What it does not prove

- It does **not** measure model extraction quality: only one recorded response is
  replayed. Six verified rows are not measured at all.
- It is **not** the brief's 120–150 programme, two-annotator, α ≥ 0.7 golden set.
  It is a truthful seed and a working harness; the set must grow before release.
- It does **not** prove salary intervals are calibrated; there are no interval
  predictions yet.
- It does **not** prove URL-level retrievability offline. Citation-only rows are
  counted separately and cannot be re-checked without network.
- It does **not** prove scoring correctness or Kendall τ against a human panel.

## Adding a row without lying

1. Find the exact sentence/number in a committed fixture or a public source.
2. Copy the quote character-for-character. Expected values are the normalised
   claim values; put the printed form in `printed_value`/`printed_unit`.
3. Fill `fixture:` (repo-relative) or `source_url` + a real `retrieved_at`.
4. Leave `status: unverified`, `value: null` if you cannot do 2–3. Do not guess.
5. Run `make eval`; the loader fails on a quote that is not verbatim.
6. Update the verified/unverified counts in this document and the committed
   fixture citation if needed.

If a metric is `null`, the harness is saying it could not measure it. That is
intended behaviour, not a defect to paper over.
