# Golden set — provenance rules

This directory is the repo's only human-curated ground truth. It exists so the
eval harness can measure extraction against something other than itself.

**Hard rule: do not invent ground truth.** A row may only assert a value that:

1. appears verbatim in a committed fixture (state `fixture:` on the entry),
   normally under `packages/pipeline/tests/fixtures/`; table transcriptions may
   live under `data/golden/fixtures/`, **or**
2. comes from a cited public source with a `source_url` and `retrieved_at`
   (offline runs cannot re-check it; it is reported as `citation` traceability).

If neither holds, set `status: unverified`, `value: null`, and leave a `note`
describing what a human must check. Unverified rows are counted in the report
and **excluded** from every accuracy metric. A fabricated salary is worse than a
missing one.

For a table-row claim, also record the extractor's `context`, `disambiguation`
(`unique` | `context_row` | `context_window`), and `row` identity
(`row_label`, `column_header`, `value_cell`, `table_page`). A verified context
must resolve in the fixture and, for `context_row`, leave exactly one quote in
its row block. The loader rejects a fabricated or non-disambiguating context.

Expected values are the *normalised* claim values the extraction gate emits
(percentages become ratios, currencies are upper-case ISO codes). Use
`printed_value` / `printed_unit` to record what the document actually printed.

Rows are one YAML file per entry (or a list of entries in one file). Files whose
name starts with `_` are templates and are not loaded.

## Adding a row

The fast, safe route for PDF sources is the curation tool (see
`docs/product-development-cycle/08-eval-methodology.md` § "Curation workflow"):

```
cd packages/pipeline
uv run python -m fmd.cli golden add \
  --source <path-or-URL.pdf> --page 1 \
  --predicate seats.total --value 20 --unit count --printed-value 20 \
  --institution "..." --programme "..." \
  --context "..." --row-label "..." --column-header "..." --trim-left 1 --pick N
```

It extracts the quote from the source PDF itself (a typed `--quote` is
rejected), proves uniqueness and row-context disambiguation with the loader's
own rules, writes `verified` only when every check passes (otherwise
`unverified` with `value: null` and a reason), and records the fixture path,
`source_url`, `retrieved_at`, and both the PDF and fixture sha256. Then:

1. `uv run python -m fmd.cli golden check --golden ../../data/golden` must pass.
2. Run `uv run python -m fmd.cli eval --golden ../../data/golden`; the loader
   fails if the quote or its row context is not provable, so a typo cannot
   quietly become "verified".
3. If the document is genuinely silent on a field, use
   `expectation: abstain` with `absent_terms` rather than inventing a value.
4. If a value must be rejected by the gates, use `expectation: reject` and a
   `reason`.
5. Update the verified/unverified counts in
   `docs/product-development-cycle/08-eval-methodology.md`.

Non-PDF sources (committed HTML/JSON fixtures, cited pages) are still curated
by hand below.

### Manual route

1. Open the fixture and copy the **exact** quote (including punctuation).
2. Run `uv run python -m fmd.cli eval --golden data/golden`; the loader fails if
   the quote or its row context is not provable, so a typo cannot quietly become
   "verified".
3. If the document is genuinely silent on a field, use
   `expectation: abstain` with `absent_terms` rather than inventing a value.
4. If a value must be rejected by the gates, use `expectation: reject` and a
   `reason`.
5. Update the verified/unverified counts in `docs/product-development-cycle/08-eval-methodology.md`.
