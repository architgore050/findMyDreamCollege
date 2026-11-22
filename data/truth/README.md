# data/truth/ — external reference tables for product evals

This directory holds **proxy ground truth** the product streams in
`fmd eval --eval-kind product` rank shortlists against. It is analogous to
`data/golden/`: curated, provenance-carrying, committed. It is NOT a raw crawl
and never contains user data.

Every truth file carries the provenance block defined in
[docs/v1/eval/03-truth-ingestion.md](../../docs/v1/eval/03-truth-ingestion.md):
`truth_kind`, `truth_version`, `scope`, `as_of`, `source_url`, `source_sha256`,
`fetched_at`, and a `raw_file` pointer whose bytes the sha256 hashes. The loader
(`fmd/eval/truth.py`) re-verifies the hash offline on every run — a truth file
whose raw copy is missing or tampered with fails the load instead of silently
becoming "ground truth".

| Directory | Contents | Status |
|-----------|----------|--------|
| `nirf/`   | NIRF ranking editions (medical, overall) | 2025 editions ingested |
| `naac/`   | NAAC institutional grades/CGPA (5-year cycle) | schema committed; **not ingested yet** — naac.gov.in was unreachable from the ingestion environment (see `naac/README.md`) |

Honest limits, repeated in every doc:

- NIRF publishes **no paramedical/MLT category**. A NIRF rank is an
  institution-level signal and may never be attributed to a specific B.Sc. MLT
  programme (AGENTS.md invariant 3; `fmd/adapters/nirf.py` enforces the same
  rule for pipeline claims).
- NAAC grades are institutional and on a 5-year accreditation cycle; a grade
  may pre-date the intake year a user cares about by up to five years.
- Neither source is an outcome measure. They cannot answer "is this MLT
  programme good" — only "is the institution recognised as strong". The
  product streams that consume these tables label themselves proxies, and the
  verdict stream (docs/v1/eval/04-verdict-audit-protocol.md) exists because no
  programme-level outcome golden set exists for Indian B.Sc. MLT.
