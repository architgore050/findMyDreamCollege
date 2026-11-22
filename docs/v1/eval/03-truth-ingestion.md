# 03 — Truth ingestion contract (data/truth/ → fmd/eval/truth.py)

**Wave:** FMD-CLOUD-024 / v1 eval builder
**Depends on:** `01-eval-designs.md` §1.2/§4.1, `08-eval-methodology.md` (golden provenance rules)

This document is the contract between the files under `data/truth/` and the
loader/metrics in `fmd/eval/truth.py`. The loader enforces it mechanically; a
truth file that violates it fails the load instead of becoming "ground truth"
(the same stance `fmd/eval/golden.py` takes toward golden rows).

## 1. File layout

```
data/truth/
  README.md                 layout + honest limits
  nirf/
    README.md               acquisition README (how to re-fetch)
    nirf_<year>_<category>.yaml
    raw/<Year>_<Category>Ranking.html   byte-exact page copy behind each YAML
  naac/
    README.md               acquisition README + current status
    naac_grades.yaml
```

A truth YAML is one of two kinds: `truth_kind: nirf_ranking` (one edition, one
category) or `truth_kind: naac_grades` (one rolling table). There is no third
kind; the loader refuses unknown kinds.

## 2. Provenance block (mandatory, checked on every load)

| Field | Rule enforced by the loader |
|-------|-----------------------------|
| `schema_version` | integer, currently 1 |
| `truth_kind` | `nirf_ranking` or `naac_grades` |
| `truth_version` | integer ≥ 1 (0 = explicitly "nothing ingested yet", used by the empty NAAC table) |
| `scope` | must be `institution` — both NIRF and NAAC are institution-level; a `programme` scope is rejected because neither source can see programmes |
| `as_of` | NIRF: the ranking edition year, as `"YYYY"`. NAAC: null while unfetched; once filled, the latest `valid_until` across rows. Never a made-up date |
| `source_url` | the page the rows were read from |
| `source_sha256` | sha256 of the bytes at `raw_file` (null only together with `entries: []` on a not-yet-ingested table) |
| `fetched_at` | UTC ISO-8601 timestamp of the fetch (null only when nothing ingested) |
| `raw_file` | repo-relative path to the raw page copy |

The loader recomputes `sha256(raw_file)` and compares it to `source_sha256`.
A missing raw file, a byte change, or a stale hash is a load error — truth
data that cannot prove its own provenance never reaches a metric. The
committed NIRF raw pages are small (~180–360 KB each); this is the same
"commit the fixture, hash it" trade-off the golden set already makes.

## 3. Entry rows

**NIRF entry** (per institution on the page):

```yaml
- institution_id: IR-D-N-15          # NIRF Institute ID, unique per file
  institution_key: all india institute of medical sciences delhi
  name: All India Institute of Medical Sciences, Delhi
  city: New Delhi
  state: Delhi
  rank: 1                            # int; ties allowed (two entries may share a rank)
  score: 91.8                        # published overall score (0-100)
  params: {tlr: 93.51, rpc: 91.87, go: 98.28, oi: 65.34, pr: 100.0}
  source_pdf: https://www.nirfindia.org/nirfpdfcdn/2025/pdf/Medical/IR-D-N-15.pdf
```

Loader checks: ranks form a non-decreasing series starting at 1 (ties are
legal NIRF output); every row carries all five parameter scores; `name` and
`institution_key` are non-empty; `institution_id` is unique; `institution_key`
must equal `normalise_institution_key(name)` (recomputed, not trusted).

**NAAC entry** (per institution, once ingested):

```yaml
- institution_key: kasturba medical college manipal
  aishe_code: "C-12345"
  grade: A+
  cgpa: 3.34
  accreditation_cycle: 2
  valid_until: "2027-03-31"
  regime: graded                     # graded | binary
  source_pdf: <accreditation-results PDF URL>
```

Loader checks: `grade` ∈ {A++, A+, A, B++, B+, B, C, D} (binary-regime rows
use `regime: binary` with the binary label in `grade`); `cgpa` ∈ [0, 4];
`institution_key` unique per file; `valid_until` an ISO date. An NAAC file
whose `truth_version` is 0 must have `entries: []` — a table that claims
"not ingested" and rows at the same time is a contradiction the loader rejects.

## 4. The join key (`institution_key`)

Product shortlists and truth tables live in different vocabularies, so both
sides are reduced to the same normalised key:

1. NFKD-ASCII-fold (strip diacritics);
2. lowercase;
3. `&` → `and`;
4. every non-alphanumeric run → one space;
5. strip.

The rule is duplicated in the ingestion script
(`scripts/eval/fetch_nirf_truth.py`) and in `fmd/eval/truth.py`; the loader
recomputes the key from `name` on load, so a divergence between the two
implementations fails ingestion rather than silently mismatching joins.
Matching stays deliberately conservative: no fuzzy matching, no alias
resolution at eval time. If a product shortlist names an institution in a form
that does not normalise identically, that row scores as UNKNOWN — a coverage
gap to fix in the shortlist's entity resolution, not in the metric.

## 5. Ingestion tooling and the write path

`scripts/eval/fetch_nirf_truth.py` is the only sanctioned writer for NIRF
truth files. It fetches once (truthful UA, escalating on the documented
CloudFront 403), parses, validates ranks/scores, writes the raw page copy and
the YAML, and prints the sha256. It aborts without writing anything when the
page structure changes — a half-parsed page must never become truth. The NAAC
writer is not built until the first successful reachability window; the
procedure is in `data/truth/naac/README.md` and its rows will be hand-curated
per PDF (with per-PDF provenance) rather than bulk-parsed on first ingest.

## 6. What a truth file proves — and what it never proves

- It proves: the rows exist verbatim on a specific public page, hashed at
  fetch time, re-checkable offline forever.
- It never proves: that institutional rank/grade is the right signal for an
  MLT programme choice; that a rank covers the full category (pages publish
  subsets); that a NAAC grade says anything about a specific programme's
  employment outcomes. Every consumer (nirf-proxy, naac-proxy, verdict
  reference ranking) must repeat the proxy label in its report — the metric
  code refuses to emit a bare "accuracy" number for a proxy stream.
