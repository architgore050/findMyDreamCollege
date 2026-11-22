# data/truth/naac/ — NAAC grade truth table (5-year cycle)

## Status: schema committed, grades NOT ingested yet

`naac_grades.yaml` below carries the ingestion schema with `entries: []`.
During this wave's ingestion (2026-11-22) the NAAC site was unreachable from
the build environment: both `https://naac.gov.in/` and the accreditation-
results listing page timed out repeatedly (60 s and 280 s connection timeouts,
matching the flakiness already recorded for this host in `config/sources.yaml`
and `fmd/adapters/naac.py`). No grades were ingested, and none were invented.

**UNKNOWN is a value (invariant 5):** until the table is filled, the
`naac-proxy` stream reports UNKNOWN coverage of 100% on real shortlists and the
metric numbers in its reports stay `null` on truth-table level. The stream's
logic is exercised offline by a committed synthetic fixture under
`packages/pipeline/tests/fixtures/eval_product/` that is clearly marked
`synthetic: true` and never ships as truth.

## What the table will contain (per docs/v1/eval/03-truth-ingestion.md)

One row per accredited institution:

- `institution_key` — the shared normalised join key (same rule as NIRF rows)
- `aishe_code` — the AISHE code printed on the grade certificate PDF
- `grade` — `A++ | A+ | A | B++ | B+ | B | C | D` (or the 2025 binary-regime
  equivalent, with `regime` recorded per row)
- `cgpa` — the institutional CGPA on NAAC's published scale
- `accreditation_cycle` — 1st/2nd/3rd/… cycle
- `valid_until` — the grade's expiry date (grades run on 5-year cycles)
- `source_pdf` — the accreditation-results PDF URL the row was read from

## How to ingest (when the site is reachable)

1. Fetch the listing page
   `https://naac.gov.in/index.php/en/19-quick-links/62-accreditationresults`
   over **https** with a truthful UA (http times out; the certificate chain is
   incomplete, which is a fetch-layer trust question — see the notes on the
   `naac_results` source in `config/sources.yaml`; do not disable TLS, handle
   the intermediate certificate instead).
2. The page links ~1,539 result PDFs. Classify anchors by text + table context
   (the adapter's documented rule — do not pattern-match filenames).
3. Extract AISHE code, CGPA, grade, cycle and validity from each health-
   sciences PDF; write one row per institution with the PDF URL as
   `source_pdf`; record `fetched_at` and the listing page's sha256 in the
   header block; commit the PDFs' sha256s (not the PDFs themselves) unless a
   fixture is needed for a disputed row.
4. Run `fmd eval --eval-kind product --stream naac-proxy --dry-run` after
   filling the table: the loader re-verifies provenance and the stream reports
   coverage honestly.

Realistic effort: the PDF corpus is born-digital but irregular; expect a
curated subset (e.g. institutions that also appear in the product's shortlists
for committed profiles) before a full ingest. A full blind ingest is not worth
it until the naac-proxy stream has a consumer that reads its output.
