# data/truth/nirf/ — NIRF ranking truth tables

## What is committed

| File | Content |
|------|---------|
| `nirf_2025_medical.yaml` | NIRF 2025 Medical category, ranks 1–50, with TLR/RPC/GO/OI/PR parameter scores |
| `nirf_2025_overall.yaml` | NIRF 2025 Overall category, ranks 1–100 (tied ranks occur), parameter scores |
| `raw/<Year>_<Category>Ranking.html` | Byte-exact copy of the ranking page each truth file was parsed from |

Every YAML carries `source_url`, `source_sha256` (hash of the raw file bytes —
re-verified by `fmd/eval/truth.py` on every load), `fetched_at`, `as_of`
(the ranking edition year), `truth_version` and `scope: institution`.

## Where the data comes from

NIRF publishes each edition's rank lists on `nirfindia.org` as interactive
ranking pages (and PDFs). The pages for a `(year, category)` pair live at:

```
https://www.nirfindia.org/Rankings/<year>/<Category>Ranking.html
# e.g. https://www.nirfindia.org/Rankings/2025/MedicalRanking.html
```

Known quirks (encoded in the fetcher and the adapter):

- The default crawler UA receives a **CloudFront 403**; a browser UA works
  (same escalation the pipeline's NIRF adapter documents in
  `fmd/adapters/nirf.py`).
- The rank table is embedded in the page HTML (DataTables), including a hidden
  per-institution parameter table (TLR/RPC/GO/OI/PERCEPTION) and a
  per-institute PDF link. Institutional IDs look like `IR-D-N-15`.
- Ranks can **tie** (e.g. two institutions at rank 27 in Overall 2025).
- The HTML page lists a subset of the full PDF list (Medical 2025: top 50 on
  the page; Overall: top 100). The truth file describes exactly what the
  committed page contains — never treat it as the complete category list.
- NIRF has **no paramedical / allied-health category**. Medical-category rank
  is a weak institutional proxy for an MLT programme inside a medical college;
  Overall rank is the only signal for standalone paramedical colleges.

## How to re-fetch a new edition

```
cd packages/pipeline && uv run python ../../scripts/eval/fetch_nirf_truth.py \
    --year 2026 --category medical --out-dir ../../data/truth/nirf
# repeat for --category overall
```

The script fetches with a truthful UA (escalating on 403), parses the rows,
aborts without writing anything if the page structure changes, writes the raw
page copy and the provenance-carrying YAML, and prints the sha256. Re-run with
`--raw-file <cached copy>` to parse a saved page without a second network hit
(ranking pages have rate-limit moods; one clean fetch per edition is enough).

Bump `--truth-version` only when re-ingesting the same edition with a changed
parser; a new edition year is a new file, not a new version.
