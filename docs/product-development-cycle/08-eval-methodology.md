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

**Current committed state (2026-11-18): 5 entries, 16 verified rows, 1
unverified row, 0 citation-only rows.** Every verified quote was checked
verbatim against its committed fixture by the loader. Six verified rows are
inventory/ranking facts with no extraction replay (UGC, RUHS, NIRF); only the
placement-report entry is measured for extraction accuracy. The two Stanley
seat-matrix rows are fixture-checked but not replayed. One of them
(`tnmgrmu-ahs-ug-seat-matrix-2024-25.seats.total`) carries row context, so its
attribution metric is `null`.

A row is one of:

| `status` | `expectation` | Meaning |
|---|---|---|
| verified | value | The fixture/citation contains the value and quote. |
| verified | abstain | The document is silent; `absent_terms` must really be absent. |
| verified | reject | The value must be rejected; `reason` is recorded. |
| unverified | value (value null) | Not checked by a human yet. Counted, excluded from accuracy. |

A row may also carry optional row context, matching what the extractor emits
from the seat-matrix contract:

```yaml
context: "Govt. Stanley Medical College & Hospital, Chennai - 600 001"
disambiguation: context_row          # unique | context_row | context_window
row:
  row_label: "3. B.Sc. Medical Laboratory Technology"
  column_header: "Intake"
  value_cell: "20"
  table_page: 1
```

`context` is the anchor cell (institution/heading/row label). `disambiguation`
says how the extractor proved the span: `unique` (one occurrence), `context_row`
(ambiguous quote, exactly one occurrence inside the context's row block), or
`context_window` (same line / ±200 characters). `row` records the extractor's
`RowIdentity`; `table_page` is 1-based. All three fields are optional and default
to absent, so older rows load unchanged.

The loader is stricter than the old presence check. For a verified row with a
non-empty `context` it requires:

1. exactly one occurrence of the context in the committed fixture (exact first,
   then the extractor's canonicalisation);
2. `disambiguation` present and consistent — a contextless row may not claim
   `context_row`, and `unique` may not carry a context;
3. for `context_row`: the quote must be ambiguous globally (otherwise it is
   mislabelled) and exactly once inside the context's row block; if `row_label`
   is present it must also resolve exactly once inside that block;
4. for `context_window`: exactly one quote occurrence on the context line or
   within the same ±200-character window.

A committed fixture is text, not a Docling table, so the block is the anchor
line plus its following lines until a blank line, form feed, or next serial
number. This is deliberately conservative: a context that does not disambiguate
fails the load rather than becoming ground truth.

`status: unverified` with a non-null value is a loader error. So is a verified
quote that does not occur in its fixture. That is the point: a typo cannot
become "ground truth". A row that cannot meet either traceability route must
stay unverified; the harness will report fewer metrics rather than a prettier
number.

## Curation workflow (`fmd golden`)

Hand-editing YAML against the loader's rules is slow and easy to get wrong, so
the golden tooling ships the rules as commands. Everything is offline and
deterministic; no golden rows are invented by the tool — it only writes what a
curator picked from a source document it hashed and extracted itself.

### `fmd golden add` — curate one expected row

```
fmd golden add \
  --source <path-or-URL.pdf> --page 1 \
  --predicate seats.total --value 20 --unit count --printed-value 20 \
  --institution "Govt. Stanley Medical College & Hospital" \
  --programme "B.Sc. Medical Laboratory Technology" --state "Tamil Nadu" \
  --context "Govt. Stanley Medical College &" \
  --row-label "3. B.Sc. Medical Laboratory Technology" --column-header "Intake" \
  --trim-left 1 --pick 2
```

What it does, in order:

1. **Extracts, never accepts, the quote.** pdftext (the same T0 library the
   pipeline uses) reads exactly one page of the source PDF (local path, or a
   URL fetched once with a truthful UA after a robots.txt check — a disallowed
   host is never fetched; download the file and pass the path instead). A
   `--quote` flag exists only to be refused: a typed quote can
   never be verified, so the command exits non-zero when it is passed. The
   curator instead picks a candidate *line* (`--pick N`, or the only candidate
   is chosen automatically, or prompts when `--interactive`); `--trim-left` /
   `--trim-right` drop whole words (e.g. a serial number) from the extracted
   line. The quote is always a span of the extracted page text, and the chosen
   span is printed.
2. **Builds the fixture from the source.** The page becomes a deterministic,
   page-scoped text fixture under `data/golden/fixtures/<entry>_pageNN.txt`:
   one line per pdftext line, with a blank line before every record header (a
   serial-numbered line at the page's left margin) so `rowctx.row_block` can
   resolve table rows. Identical PDF bytes produce identical fixture bytes;
   `fmd golden check` re-hashes them.
3. **Proves the quote with the loader's own rules.** Without a context, the
   quote must occur exactly once. With `--context`, the command requires
   `--row-label` and `--column-header` and proves both loader facts: the quote
   is ambiguous without the context (>= 2 occurrences in the fixture) and
   unique inside the context's row block. A quote that is already globally
   unique cannot carry a table-row context (the loader calls that mislabelled)
   — the command refuses and explains how to re-run.
4. **Writes `verified` only when every check passes.** Otherwise it writes
   `status: unverified` with `value: null` and the failing check as the row
   note, exits non-zero, and never silently promotes. The assembled entry is
   validated against the loader's pydantic schema before anything is written,
   so a generated entry cannot fail `fmd eval`.
5. **Records provenance.** The entry carries `source_url` (the URL when the
   source is one, or `--source-url` for a local file) and `retrieved_at`
   (UTC date). A JSON sidecar next to the fixture (`..._pageNN.json`) records
   the fixture sha256, the source PDF sha256, the URL, the page, and the
   pdftext version used at curation time. The entry header comment repeats the
   same facts for human readers.

Further predicates on the same document merge into the existing entry (one
entry has exactly one fixture, so rows from another page need their own entry
id). Entries not generated by the tool are never auto-edited; hand-curated
rows stay hand-curated. `--family` is refused: replay entries need a recorded
guided-JSON fixture that curation cannot produce.

### `fmd golden check` — offline re-validation

Re-runs every rule the curator proved, plus state it cannot see at curation
time: fixture presence and sha256 against the sidecar (tampered fixture →
failure), row-context resolution for every verified value row, duplicate
predicate/entity rows (exact duplicates across entries fail; the same
institution + predicate + value on different documents is a warning, because
re-publication is legitimate), and a verified/unverified breakdown by family.
Exit 0 only with no failures; `--json` for the machine-readable report. Fixtures
written before this tool existed are reported as `unrecorded` rather than
assumed clean.

### `fmd golden list` — status and gates

Prints every entry with its measured flag and per-row status plus what each row
gates: verified value rows gate extraction accuracy (with a context they also
gate row attribution, where a wrong-row value is a false positive); verified
`abstain` rows gate abstention correctness; verified `reject` rows gate
rejection correctness; unverified rows gate nothing until a human verifies
them. Entries with `family` + recorded guided JSON are replayed canaries.
`--json` carries the same payload.

### What the tool proves and what it does not

It proves span traceability: the quote came from the hashed source, the
fixture is byte-identical to what pdftext produced, and the uniqueness /
row-block facts hold. It does not prove the source is the right document, that
the value is *correct* (a curator can still pick the wrong line — that is why
§12 keeps second-person adjudication), or that pdftext recovered every cell.
A scan, or a page whose records have no left-margin serial numbers, simply
fails the proof and falls back to `unverified`.

## Metrics

**Extraction span accuracy / P/R/F1.** A field is a true positive only when
(a) the predicted quote resolves exactly at `[char_start, char_end)` in the
extraction text, (b) it equals the golden quote, (c) the value matches
(numeric tolerance `1e-9`), (d) the normalised unit matches, and (e) for a
row-context claim, the span lands in the golden row's block. Precision =
TP/(TP+FP), recall = TP/(TP+FN). Missing predictions are false negatives, never
zeros.

**Row-attribution accuracy.** Only applies to verified value rows with a
`context`. For each such row that received a prediction, the prediction is
correct when its span is inside the unique context block; if the prediction
publishes `context_quote` or a row label, that metadata must also match. Report
`row_context_rows`, `row_attribution_checks`, `row_attribution_correct`,
`row_attribution_accuracy` (null until checks > 0), and `wrong_attributions`.

`wrong_attributions` is the catastrophic case: the value is real but the span is
in another row. It is a false positive, not a false negative — the run found the
number, it attributed it to the wrong institution. This is the error the seat
matrix previously could not measure.

**What row attribution does not prove.** The loader can only check the
committed text fixture, not the original table structure; a block boundary is a
deterministic text convention. And if a row-context golden row has no recorded
prediction (as with the Stanley seat row today), attribution is `null`, not
`1.0`. The metric proves attribution only for replayed entries, never for the
curated row by itself.

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
  resolves at its exact span, and extraction agrees with 16 curated rows.
- A verified row that names a context has that context re-checked against the
  fixture; a fabricated or non-disambiguating context fails the load.
- The golden loader cannot silently accept a fabricated verified value.
- The provenance gate is wired to the process exit code.
- Drift logic distinguishes a claims change from an unexplained movement.
- The whole run is reproducible offline and byte-identical for a fixed checkout.

## What it does not prove

- It does **not** measure model extraction quality: only one recorded response is
  replayed. Eight verified rows are not measured at all, including the
  row-context Stanley seat row.
- It does **not** measure row attribution for entries that are not replayed. The
  Stanley row is load-checked, but its `row_attribution_accuracy` is `null`.
- It is **not** the brief's 120–150 programme, two-annotator, α ≥ 0.7 golden set.
  It is a truthful seed and a working harness; the set must grow before release.
- It does **not** prove salary intervals are calibrated; there are no interval
  predictions yet.
- It does **not** prove URL-level retrievability offline. Citation-only rows are
  counted separately and cannot be re-checked without network.
- It does **not** prove scoring correctness or Kendall τ against a human panel.

## Adding a row without lying

The preferred route is `fmd golden add` (see the curation workflow above): it
extracts the quote from the source PDF itself, proves it, and writes the entry.
The manual route below stays for non-PDF sources (committed HTML/JSON
fixtures, cited pages) and for rows the tool refuses; the loader enforces the
same rules on both.

1. Find the exact sentence/number in a committed fixture or a public source.
2. Copy the quote character-for-character. Expected values are the normalised
   claim values; put the printed form in `printed_value`/`printed_unit`.
3. Fill `fixture:` (repo-relative) or `source_url` + a real `retrieved_at`.
4. If the value comes from a table row, add `context`, `disambiguation`, and
   `row` exactly as the extractor reported them. Do not hand-write a context
   that the fixture does not contain; the loader will reject it.
5. Leave `status: unverified`, `value: null` if you cannot do 2–4. Do not guess.
6. Run `make eval`; the loader fails on a quote/context that is not provable.
7. Update the verified/unverified counts in this document and the committed
   fixture citation if needed.

If a metric is `null`, the harness is saying it could not measure it. That is
intended behaviour, not a defect to paper over.
