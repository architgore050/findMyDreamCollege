# 07 — Workstream status: evidence-based readiness dossier

**Snapshot:** 2026-11-17, 39 commits (`git log --oneline | wc -l`). Written by
WS-STATUS from the tree, tests, committed fixtures and a read-only query of the
running `findmydreamcollege-db-1` (Postgres 17 + pgvector, healthy). No network
was used. **A capability is "proven" only when a test or a real run demonstrates
it; code existing is not proof.**

**How to read the evidence tags:**

- `[T]` test run: `pytest` and `vitest` executed at this snapshot.
- `[DB]` live `psql` query against the compose database.
- `[E]` `fmd eval --golden ../../data/golden` executed at this snapshot.
- `[C]` config/code inspection; not a behavioural proof.
- `[R]` a prior report/commit message, **not reproduced here** — treat as stale.

---

## 1. Capability ledger — Proven / Unproven / Blocked

| Capability | Proven (evidence) | Unproven (why) | Blocked (on what) |
|---|---|---|---|
| Config registry + validation | Loader validates countries/sources/weights/exams (`test_config.py`; 13 countries, 32 sources, 8 exams `[DB]`; 29 of 32 sources enabled `[DB]`). | That the registry's claims about live sources are still true — most were verified 2026-11-16 `[C]`, no recheck. | Human approval of crawl scale and of `verify_first` sources. |
| Migrations + DB schema | 5 migrations apply; `schema_migrations`, 23 tables `[DB]`; idempotency test in `test_db.py`. | Migration from a genuinely fresh volume (not rerun) at this commit — current volume predates the latest containers. | — |
| Snapshot/evidence store | 8 snapshots with sha256, 3 PDFs + 5 JSON/HTML `[DB]`; content-addressed store tested (`test_store.py`). | That fetched bytes are the *right* documents: 7 of 12 documents are evidence, 5 abstained `[DB]`; no extractor has read them. | — |
| Fetch ladder / robots / politeness | Ladder, robots policies, POST replay, pagination and dedupe tested (`test_fetch_*.py`, 22 ladder tests). | Live crawl through the full HTTP stack at this commit — the API container is crash-looping (`ModuleNotFoundError: yaml`), so compose-run crawl is not proven now. | Rebuild the API image; production crawl approval. |
| India adapters (UGC/NABH/NAAC/NIRF/etc.) | Adapter parsers tested against committed fixtures; 12 documents from 4 domains were actually fetched `[DB]`; 5 crawl runs succeeded, 2 failed `[DB]`. | That inventory ingestion produced programmes: DB has **1 entity, 0 programmes, 0 institution_domains** `[DB]`. The commit "1246 entities, 1218 domains" (5f7ddb5) is `[R]` and not reproducible from this DB. | UGC/NABH ingestion rerun against the persisted DB. |
| Extraction: text PDFs → claims | Pipeline, schemas, span resolution and Excel-free validation tested on fixtures (`test_extract_*.py`; 9 pipeline tests; quote-first span tests). | **Extraction from a real PDF: 0 extractions, 0 claims** `[DB]`; only one recorded LLM response is replayed `[E]`; no model run exists. | A live local model (vLLM) and one human-run `fmd extract`. |
| Extraction: tables / merged cells | Header-association logic and "merged cell is uncertain" tested on synthetic fixtures (`test_acquire_tables.py`). | **Table structure on merged cells in a real document: 0 cells observed**, no TableFormer artifacts, no Docling table run. | Docling artifacts (`config/acquire.yaml` T1) and a real table document. |
| Devanagari / scanned OCR | OCR service contract, fail-fast and offline 503 tested; canonical span→bbox tested (`test_ocr_service.py`, `test_acquire_canonical.py`). | Live Qwen3-VL page: skipped — weights absent (~10.6 GB); `11-inference-runbook.md` §5 marks OCR blocks/pages **unverified**. | Vision weights download (out of scope offline). |
| Verification second pass | Quote-only verdict schema, value-free prompt, repair, spend caps, persistence tested (`test_verify.py`; 7 field rejection test). | That any real claim moves to `supported`/`rejected`: **0 claims, 0 verification_status rows** `[DB]`. | Extraction must produce claims first. |
| Entity resolution | Normalisation, blocking, similarity bands, aliases and audit persistence tested (`test_er_*.py`); `er_audit` table exists. | Real merge on the corpus: **0 er_audit rows**, 1 entity `[DB]`. | Ingested inventory. |
| Scoring / filters / cost | 7 sub-scores and 8 binary filters tested (`test_score_*.py`, `test_filters.py`); UNKNOWN→[0,100], DSR₂₅, loan flags tested. | That a score is produced from real claims: **0 scores and 0 programmes** `[DB]`; no calibration vs humans (Kendall τ absent). | Claims + programmes. |
| Eval harness | `fmd eval` runs and gates: 4 entries, **14 verified rows, 1 unverified**; precision/recall/F1 = 1.0 on 8 replayed fields; provenance 1.0 (4/4) `[E]`. | Model quality: only one recorded response; 6 verified rows unmeasured; interval coverage n=0; not the brief's 120–150 set. | 20-document bake-off and a larger golden set. |
| 20-document bake-off | Harness and hard gates tested on synthetic candidates (`test_bakeoff.py`, 6 tests). | **Never run on 20 real documents** (no `/data/bakeoff`, no report). | Curated 20 docs + model access. |
| API surface | 12 API tests; POST profile body, cost/filter/ranking/exclusion endpoints, evidence reads. | A full ranking against real rows: DB empty of programmes. Review endpoints still 501. | Populated DB; WS-G review build. |
| Web app (7 screens) | 9 vitest files, **45 tests pass**; honest 501/503 states; profile in browser storage only. | Every number clicking through to a real quote: no numbers exist in the DB; no browser E2E. | Populated API. |
| Review console | 501 rendering tested. | The console itself: `GET /api/review/queue` and `POST /api/review/claims/{id}` return 501 (`api.py:1277-1284`). | Decision on admin UX + build. |
| Compose stack | 4 compose files/services; db healthy 11 h; migrations in place. | `make up` healthy end-to-end: **api is Restarting (1)** with `ModuleNotFoundError: yaml`; no web container running `[DB/docker ps]`. | Rebuild the pipeline image with runtime deps. |
| LLM router / offline mode | Providers, mock/replay, doctor-never-prints-secrets, offline env tested (`test_llm.py`, `test_bakeoff.py`). | Any real model call: no live inference evidence in this checkout. | Local vLLM serve. |

**Split:** 13 component behaviours are proven by tests or a real run (registry,
migrations, snapshot store, fetch primitives, adapter parsing, fixture
extraction, table-header logic, verification logic, ER logic, scoring/filters,
eval harness, bake-off harness, API/web contracts). 5 product behaviours are
unproven (real PDF extraction, real merged-cell tables, live OCR, the 20-doc
bake-off, a populated ranked recommendation). 6 are hard-blocked (live model
extraction, real tables, OCR weights, bake-off corpus, populated DB, review
console). **0 capabilities are proven end-to-end for a recommendation.**

---

## 2. Milestones (brief §20)

| ID | Status | Exact remaining acceptance criteria |
|---|---|---|
| **M0 Foundation** | **Partial** | Fix API container (yaml import) and start web; prove `docker compose up` → healthchecks on a fresh clone; rerun `make test` via compose. Offline pytest (698 pass/3 skip) and ruff (clean) already pass. |
| **M1 Inventory** | **Not met** | ≥300 institutions and ≥150 BSc MLT programmes with resolvable recognition chain. Current: **1 entity, 0 programmes, 0 domains** `[DB]`. Re-run UGC/state ingest and persist; report the actual counts (the "1246/1218" claim is unreproduced). |
| **M2 Evidence** | **Not met** | ≥1,000 snapshots; claims with **100% span resolution on a 200-doc sample**; audited `llm_calls`. Current: **8 snapshots, 12 documents, 0 claims, 0 extractions** `[DB]`; only 4 fixture claims resolve `[E]`. |
| **M3 Scoring + filters** | **Code-complete, data-unproven** | All 7 sub-scores emit intervals on real programmes; 8 filters with UNKNOWN; DSR₂₅ computed; **golden set scored** (not just extraction). Current: **0 scores** `[DB]`; eval does not score a programme. |
| **M4 Web app** | **Partial** | 7 screens against real data; every number clicks to a quote; exclusions explainable. Current: screens render and 45 tests pass, but against an empty DB; no E2E. |
| **M5 Review + eval** | **Not met** | Review console with source-span viewer (currently 501); drift/coverage reports from a real run; docs complete. Drift code is tested, never run on scale distributions. |

---

## 3. Invariant audit (AGENTS.md)

| # | Invariant | Verdict | Evidence / soft spot |
|---|---|---|---|
| 1 | No user-specific logic | **Compliant** | `ProfileRequest.profile` is a free dict with no defaults (`api.py:386`); grep for budget/intake/income finds only runtime profile handling and `config/`. No applicant identifiers. |
| 2 | Numbers computed; prose generated | **Compliant in code, unproven in run** | All weights/versions from `config/weights.yaml`; prompts repeatedly say "No scores or recommendations" and "Never compute" (`extract/prompts/*/v2.md`). No model has actually emitted a claim, so the end-to-end property is untested. |
| 3 | Every published number resolves to a span | **Partial** | Gate passes 1.0 (4/4) on fixtures `[E]`; `--require-snapshot` is **off** offline; DB has **0 published claims**, so the real-corpus property is vacuous. `verified_at` exists (migration 0005) but no row uses it. |
| 4 | Filters binary; scores weighted | **Compliant** | 8 implementations match `config/weights.yaml:filters`; each has pass/fail/unknown tests (`test_filters.py`, 10 tests). UNKNOWN ranks as "verify", never a penalty (`filter_policy.unknown_blocks_ranking: false`). |
| 5 | UNKNOWN is a value, not 0/50 | **Compliant** | `score/aggregate.py`: UNKNOWN contributes [0,100], point is `None`; `min_claims_for_score: 2`; tests in `test_score_aggregate.py`. No real scores to confirm propagation. |
| 6 | Internships ≠ employment | **Compliant at schema/score/prompt** | `curriculum.internship_hours/months` are separate predicates; `placement.internship_counted_as_employment` is its own field; `flags.internship_counted_as_placement` fires on intern outcomes (`test_score_flags.py:41`). Real placement reports never extracted. |
| 7 | Registry-driven | **Compliant with soft spots** | Countries/sources/exams/weights/cost/plan/acquire are YAML. Hardcoded operational constants remain: `DEFAULT_SOURCE_FAILURE_BUDGET=5`, unknown-sigma fallback 0.55, default limit 200. They are documented defaults, not user data. |
| 8 | Append-only evidence; versioned scores | **Partial** | Claims content is never updated in place, but there is **no DB trigger/revoke**; repositories do `UPDATE claims SET entity_id` (reparent), `superseded_by`, and verification columns (`repositories.py:998,1897,2031`). Convention, not enforcement. |
| 9 | Politeness, truthful UA | **Partial** | Robots policies + per-domain limits + retry tests exist; fetches use `settings.user_agent`. But `.env.example` ships `contact: set-me@example.com` — a placeholder contact, not a truthful one, until configured. |
| 10 | No paid APIs in v1 | **Partial / at risk** | `bulk_extract` defaults local, but `hard_pdf` defaults to provider `api` and `adjudicate` defaults to `gemini` (`config/models.yaml:106,123`). Keyless/unset degrades, and tests use mock; still, the default path for hard PDFs points at a paid provider. |

**Profile-in-URL:** none. Profile travels only in POST bodies (`api.py` cost/filter/rank models; `apps/web/lib/api.ts`); `compare?ids=` carries programme ids only. LocalStorage is the only persistence (`apps/web/lib/use-profile.ts`).

---

## 4. Numbers that exist (all from this snapshot)

| Metric | Value | Source |
|---|---:|---|
| Commits | 39 | `git log --oneline \| wc -l` |
| Institutions (`entities`) | **1** (RUHS, Rajasthan) | `[DB]` |
| Institution domains | **0** | `[DB]` |
| Programmes | **0** | `[DB]` |
| Sources registered / enabled | 32 / 29 | `[DB]` |
| Exams / countries | 8 / 13 | `[DB]` |
| Documents | **12** (4 domains) | `[DB]` |
| Documents by role | evidence 7, inventory 4, navigation 1 | `[DB]` |
| Documents by family | unset 10, fee_structure 1, exam_notification 1 | `[DB]` |
| Documents by plan_state | not_applicable 5, abstained 5, planned 1, linked 1 | `[DB]` |
| Evidence PDFs | **3** (all fetched via httpx) | `[DB]` |
| Snapshots | **8** (sha256 present) | `[DB]` |
| Extractions | **0** | `[DB]` |
| Claims / verified claims | **0 / 0** | `[DB]` |
| Scores | **0** | `[DB]` |
| `er_audit` rows | **0** | `[DB]` |
| Frontier URLs | 12 | `[DB]` |
| Pipeline runs | crawl ok 5, crawl failed 2, extract failed 1, ingest ok 1 | `[DB]` |
| Golden entries / verified rows / unverified | 4 / **14** / 1 | `[E]` |
| Golden extraction accuracy | P=R=F1=1.0 on 8 replayed fields; 6 rows unmeasured | `[E]` |
| Provenance coverage | 1.0 (4/4, modelled excluded 0) | `[E]` |
| Interval coverage | n=0 (not measurable) | `[E]` |
| Python tests | **698 passed, 3 skipped** (DB-backed; 701 collected) | `[T]` with `DATABASE_URL=:5433` |
| Python tests, no DB | 657 passed, 44 skipped | `[T]` |
| Web tests | **45 passed** (9 files) | `[T]` |
| Ruff | clean | `[T]` |

**Stale / unverified numbers:** "1246 entities, 1218 domains" (commit 5f7ddb5)
and "NABH 25→100 hospitals" (commit a06d287) are commit-message claims `[R]`,
not present in the current DB. The old status doc's "316 passed" is superseded.
`11-inference-runbook.md` §5 still marks decode throughput, guided JSON and OCR
throughput **unverified**.

---

## 5. Top 10 risks (ranked by damage to a recommendation)

| # | Risk | Why it invalidates a recommendation | Cheapest settling test |
|---|---|---|---|
| 1 | **No claims exist** | Every score/shortlist is empty; nothing can be recommended honestly. | Run `fmd extract --limit 3` on the 3 real PDFs with the local model, then count claims. |
| 2 | **Extraction never ran on a real PDF** | Fixture replay proves the harness, not the model or PDF text layer. | One RUHS PDF → inspect 10 claims, quote and `char_start/end` in `extractions.text`. |
| 3 | **Inventory is 1 institution / 0 programmes** | A "shortlist" from one entity is not a product; M1 is not met. | Rerun UGC ingest into the persisted DB; `select count(*) from programmes`. |
| 4 | **Merged-cell tables untested** | A fee/seat number can attach to the wrong programme — a confidently wrong recommendation. | Two real fee/seat tables; compare extracted cells to manual transcription. |
| 5 | **Outcomes are modelled or absent** | Salary/placement evidence is the core of `outcomes` (30% weight); G2.1 says no verified salary microdata. | Query how many salary claims are `measured` vs `modelled` after extraction; expect 0 measured. |
| 6 | **Exam windows may be stale** | Closed/open is a hard filter; a stale window excludes or admits wrongly. | Inspect all 8 exam rows: `observed_at`, TTL (3 days), source URL; re-fetch one live. |
| 7 | **Compose stack broken** | Brief §22 DoD starts with `make up`; api is crash-looping now. | Rebuild image, `curl localhost:8000/api/health`, start web. |
| 8 | **Scanned/Devanagari OCR absent** | Those documents silently abstain or lose rows, biasing thin-evidence programmes. | After weights: one scanned fixture through `/extract`; assert block count > 0. |
| 9 | **Verification may reject or rubber-stamp** | A false "supported" publishes a bad number; a mass reject empties the shortlist. | Verify 20 extracted claims; inspect 5 supports and 5 rejects by hand. |
| 10 | **Review console 501** | No correction path; a wrong claim cannot be fixed before publishing. | Build/read `/api/review/queue`; assert it returns real low-confidence claims. |

---

## 6. Single next action

**Bring up local vLLM and run one bounded real extraction + verification on the
three already-snapshotted PDFs (`fmd extract --limit 3` then `fmd verify`), then
record the claim/verified counts — this converts the empty evidence pipeline
into the first real claims and unblocks M2, M3 and every downstream gate.**
