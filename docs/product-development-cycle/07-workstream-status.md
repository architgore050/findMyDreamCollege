# Workstream status — phase 3 integration snapshot

Captured **2026-11-16** by WS-I. Derived from the code, `config/` and the test
suite — not from workstream self-reports. Several workstream outputs were
**uncommitted** when this snapshot was taken, so names may move; re-verify before
phase 4 sign-off.

**Legend:** *real* = implemented and exercised by tests · *gated* = implemented
but depends on an optional dependency/access and degrades with a named fallback ·
*stub* = raises / exits non-zero / returns 501 on purpose · *blocked* = needs a
human decision or data we do not have.

## Per-workstream

| WS | Real today | Capability-gated | Stubbed | Blocked on a decision |
|---|---|---|---|---|
| **A — foundation** | Config loader + registry validation; `0001_init.sql` and repositories; content-addressed snapshot store; model router with mock/replay; `fmd migrate/seed/llm-ping`; FastAPI reads (`health`, `sources`, `programmes`, evidence); envelope (`scorer_version`, `weights_version`, `claims_hash`, `generated_at`) | — | — | `make up` end-to-end still waits on the crawl/extract CLI verbs |
| **B — fetch/crawl** | Fetch ladder, robots + per-domain politeness, retry/backoff primitives, Postgres `crawl_frontier`, discovery (sitemaps, Common Crawl, optional SearXNG) | `curl_cffi`/`patchright` browser tiers; Common Crawl index; SearXNG only with the `discovery` compose profile | `fmd crawl`, `fmd worker` exit 2 (`WS-B`); no scheduler; no live crawl orchestration CLI | Budget defaults and crawl cadence policy are config, but a human must approve first production crawl scale |
| **C — India adapters** | UGC JSON XHR, NAAC, NABH, AISHE and a state-counselling adapter covering twelve states; `config/sources.yaml` (27 enabled sources); PII whitelist on UGC; registry loader rejects denylisted hosts | AISHE disabled (`verify_first`, unverified); UGC consolidated PDFs disabled (200 error shell); `nabh.co` AJAX (no nonce/CAPTCHA) | No fetch orchestration: adapters parse fixtures, the CLI does not run them | NABL excluded by robots; AISHE xlsx needs a successful fetch from a permitted network; UGC XHR credential fields must never leave the whitelist |
| **D — extraction** | `ExtractPipeline.run/run_batch`; tiered PDF text (`pypdfium2`/`pdftext`/`docling`); versioned prompts + schema registry; span/unit/rate validation; append-only claim writes; rejections + rewrites + optional VLM adjudication | `docling` heavy tier; VLM adjudication only when a model profile is configured; Devanagari/merged-cell OCR remains weak | `fmd extract` CLI verb exits 2 (`WS-D`); `run_extract` is the working seam | T3 VLM audit sampling (1%) and docling row-count reconciliation need an operational policy |
| **E — entity resolution** | `normalise_name` → deterministic blocking → similarity scoring → accept/reject/adjudicate bands → `resolve`/`resolve_with_audit`; manual aliases and canonical locks; audit record per mention | DuckDB/Splink blocking; sentence-transformers embeddings (deterministic fallback with `embedding_unavailable`); transliteration currently raises | Persistence of `er_audit`/alias provenance; no CLI verb; **current tree has `NameError: AliasMiningConfig`** in `fmd/er/config.py`, blocking all collection | `0002_er_audit_and_replay.sql` now exists (uncommitted) but **no repository writes it**; `test_db.py` still asserts exactly one migration |
| **F — score/filter/cost/API** | `score_programme` + `ScoreCreate` persistence + insight sets; `shrinkage_inactive` flag; eight binary filters with `FilterDecision`; cost/financing model with rendered snapshot; `POST /api/rankings`, `/api/exams`, `/api/exclusions`; `GET /api/programmes/{id}/cost`, `/api/compare`; `POST /api/scoring/rerun`; `fmd score --dry-run/--force/--limit` | Salary quantiles fall back to `modelled`; loan rate may be assumed (flagged); cost returns `available=false` when evidence is incomplete | `fmd eval` exits 2; `fmd/eval` raises `NotImplementedError` | Golden set, provenance-coverage 100%, interval coverage and drift diff are not machine-enforced yet |
| **G — web/review** | Seven Next.js screens + API client; honest 501/503 states; profile kept in browser storage, never persisted server-side; tests for 501 rendering | Rankings/shortlist depend on WS-F endpoints and a populated DB | Review console is a placeholder; `GET /api/review/queue` and `POST /api/review/claims/{id}` return 501 | Human claim adjudication workflow and admin-token UX are undecided |

## Known open items (with evidence)

| Owner | Item | Evidence in repo | Effect |
|---|---|---|---|
| WS-C | **NABL is not a source**; robots.txt is `Disallow: /` | `config/sources.yaml:exclusions.nabl_labs`; no `nabl` adapter wired despite the name in `KNOWN_ADAPTERS` | Clinical-training evidence relies on NABH only. Do not add a crawler; request an official dump if needed. |
| WS-C | **AISHE xlsx unverified** — static path returned connection code `000` from the dev host (DNS/TLS/egress, not 404) | `aishe_reports.enabled: false`, `verify_first: true`, `params.verified: false`; adapter raises `verify_first_unverified` | AISHE contributes nothing until someone fetches it from an allowed network and sets `params.verified`. The adapter abstains rather than reporting empty success. |
| WS-C | **UGC XHR exposes credentials/PII** in the raw JSON | `ugc_university_list.field_whitelist` + `pii_fields` in `config/sources.yaml`; adapter `compact_record` drops them | Safe only while every ingest path uses the whitelist. Never log raw payloads; a new adapter that bypasses `compact_record` is a leak. |
| WS-C | **Fake councils denylisted** | `denylisted_domains`: `statealliedandhealthcarecouncil.com`, `sahstc.com`; loader + `validate_recognition_origin` reject claims | Statutory recognition may only come from NCAHP; these hosts must never be fetched or cited. |
| WS-F | **`make eval` is not implemented** | `fmd/eval/__init__.py` raises `NotImplementedError`; `fmd eval` exits 2 | The AGENTS.md definition of done (“provenance coverage 100%”) is currently verified only by unit tests, not by an eval report. |
| WS-E | **ER audit migration landed, persistence did not** | `db/migrations/0002_er_audit_and_replay.sql`; `fmd/er/service.py` is pure; no `er_audit` repository | `test_db.py::test_migrations_are_idempotent` fails on `len(applied) == 1` (now 2). Repo wiring + test update belong to WS-E/WS-A. |
| WS-G | **`POST /api/review/*` unbuilt** | API returns 501 with `workstream: WS-G`; web review page renders the 501 state | No human claim-adjudication path. Do not publish corrections until this exists. |
| WS-B/D | **`crawl`/`extract`/`worker` CLI verbs exit 2** | `fmd/cli.py:UNIMPLEMENTED` | The phase-3 DoD path `make up → seed → crawl → score → dev` cannot complete. Extraction works only through `run_extract(...)`. |
| WS-D | **Devanagari/merged-cell OCR is not trustworthy** | `05-gaps-…` G6.3; docling optional tier | Any published number still needs span validation; a 1% VLM audit is a policy decision, not code. |

## Verification snapshot

Commands run from `packages/pipeline`:

| Command | Result |
|---|---|
| `uv sync --all-extras` | **succeeds** (composes crawl+extract+er) |
| `uv run ruff check .` | **3 errors**, all in a concurrent WS-E edit to `fmd/er/config.py` (`F821 Undefined name 'AliasMiningConfig'`, plus unused import). WS-I-owned files pass. |
| `uv run pytest -q` (no `DATABASE_URL`) | last complete run: **306 passed, 13 skipped**; current tree cannot collect (`NameError: AliasMiningConfig` in `fmd/er/config.py`) |
| `DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q` | last complete run: **316 passed, 2 skipped, 1 failed**; current tree cannot collect. The failure was `test_db.py::test_migrations_are_idempotent` (WS-E migration 0002), not WS-I. |

The DB-backed run uses throw-away schemas per fixture; no test touches shared
tables. Skips are explicit: `DATABASE_URL` unset, or DuckDB/Splink intentionally
installed (their fallback tests are skipped to prove the deterministic path).
