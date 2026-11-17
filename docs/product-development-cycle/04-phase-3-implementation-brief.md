# Phase 3 — Implementation Brief (v1: India, maximum depth)

**Status:** approved for build. Supersedes the country mix proposed in doc 03; all other doc 03 principles stand.
**Owner:** implementation agents. **Consumer:** the web app + any future agent picking this up.
**Companion docs:** `05-gaps-and-full-solution-backlog.md` (what v1 deliberately omits), `AGENTS.md` (repo conventions).

---

## 1. What v1 is

A generalized, self-hostable tool that inventories bachelor's-level Medical Laboratory Technology / Medical Laboratory Science programmes in **India**, mines public evidence about each one, and produces a **ranked, filterable, fully sourced shortlist** for a user-supplied profile.

v1 is **not** a global tool. It is India-only at full depth, with every country/field/source assumption living in config so that adding a country is data work, not code work.

**v1 is done when** a stranger can `docker compose up`, load the seed registry, run the pipeline, open the web app, enter a profile, and get a defensible shortlist where every number clicks through to a verbatim source span with a retrieval timestamp.

## 2. Non-goals for v1

Deployment (VM/Vercel/cold storage/public DB), auth, multi-user tenancy, countries beyond India, non-MLT fields, mobile app, loan underwriting, live exam-registration scraping at scale, and any form of paid API. All are tracked in doc 05.

## 3. Invariants (violate these and the build is wrong)

1. **No user-specific logic.** Intake year, budget, marks, nationality, travel tolerance, risk appetite are *profile inputs*. No defaults that encode one person. No fixtures named after a real applicant.
2. **Numbers are computed; prose is generated.** The LLM extracts claims, it never emits a score.
3. **Every published number resolves to a span.** `claim -> extraction -> snapshot -> URL + fetched_at + sha256`. A number without a resolvable span is either dropped or downgraded to `modelled` and labelled as such in the UI.
4. **Filters are binary, scores are weighted.** Closed registration, exam-requires-travel, unmet eligibility, unrecognised institute, unmet language requirement are hard cutoffs. They are never penalties inside a weighted sum.
5. **`UNKNOWN` is a value.** Missing evidence produces `UNKNOWN`, which propagates as a wide interval, never as 0 and never as 50.
6. **Internship is not employment.** Enforced in schema and scoring; an internship counted as placement invalidates the record.
7. **Registry-driven.** Countries, sources, weights, exam calendars, fee rules live in `config/*.yaml`. Adding a state source is a YAML entry plus at most a generic adapter.
8. **Append-only evidence, versioned scores.** Claims are never mutated in place. Scores carry `scorer_version` + `weights_version` + `claims_hash` and are recomputable.

## 4. Domain model (the part most likely to be built wrong)

Ground truth from research, encoded as constraints:

| Fact | Consequence for the build |
|---|---|
| NCAHP 2025 MLS curriculum: 4 years, 1,800-hour internship in semesters 7–8, degree awarded only on completion | Programme duration and internship presence are extractable, comparable fields |
| NCAHP: "internships cannot be reflected as work experience as those are part of the academic program" | `outcome_type` enum must separate `employment`, `internship`, `higher_studies`, `unknown`. Internship rows can never contribute to an employment rate |
| Registration of AHPs Regulations 2026: practice requires registration | `recognised_institute` is a **hard filter**. Unrecognised institute ⇒ excluded regardless of score |
| No central paramedical institution registry exists | Inventory is assembled: UGC universities → affiliating health-science universities → state counselling seat matrices → college sites |
| NIRF has no paramedical category | NIRF is a *proxy* signal at institution level, never programme-level evidence |
| DMLT (diploma) is being regulated down into an assistant cadre | `credential_level` filter must distinguish `degree` / `diploma` / `lateral_entry` |

**Primary outcome metric (v1):** `internship_to_offer_conversion_12m` + `registration_conversion_12m`, both `modelled` unless directly sourced. **Secondary:** `p25_first_year_salary` for the placement city, plus `salary_growth_3y`.

## 5. Architecture

```
config/*.yaml ──► registry loader ──► crawl planner
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                   Scrapy frontier            discovery
                   (per-domain budget)   (SearXNG, CC index,
                          │               sitemaps, seeds)
                          ▼
                   fetch ladder: httpx → curl_cffi → patchright
                          │
                          ▼
                   snapshot store (sha256 on disk) + documents/snapshots rows
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
        PDF tiering               HTML extract
   pypdfium2 → Docling → MinerU → VLM
              │
              ▼
     extraction (guided JSON via model router) ──► llm_calls audit
              │
              ▼
     claims (append-only, span-anchored, confidence)
              │
              ├──► entity resolution (splink + bge-m3 + LLM band)
              │
              ▼
     scoring engine (pure functions, versioned)
              │
              ▼
  scores + intervals + flags + insight sets ──► API ──► web app
```

**Repo layout**

```
config/            countries, sources, weights, exams, models, fee-rules
db/migrations/     SQL migrations (plain SQL, no ORM magic)
packages/pipeline/ Python package `fmd`
  fmd/config       registry loading + validation (pydantic)
  fmd/db           connection, repositories, migrations runner
  fmd/store        content-addressed snapshot store
  fmd/fetch        ladder, robots, rate limiter, circuit breaker
  fmd/frontier     Postgres-backed crawl queue + budgets
  fmd/discover     SearXNG, Common Crawl index, sitemaps, seed adapters
  fmd/adapters     one module per source family (ugc, aishe, nirf, naac,
                   state_counselling, fee_regulator, nabl, nabh, college)
  fmd/extract      pdf tiers, html, schema registry, prompt versions
  fmd/llm          provider-agnostic router, caching, cost accounting
  fmd/er           normalisation, blocking, splink, LLM adjudication
  fmd/score        sub-scores, aggregation, intervals, insight selection
  fmd/filters      hard cutoffs (eligibility, deadlines, travel, recognition)
  fmd/cost         TCOA, hostel/independent toggles, DSR25, FX bands
  fmd/eval         golden set, extraction metrics, drift
  fmd/cli          entrypoints
apps/web/          Next.js 15 + TS + Tailwind + shadcn
infra/             docker-compose, Dockerfiles
scripts/           seed, backfill, eval runners
docs/              this folder
```

## 6. Data model

Plain Postgres 17 + pgvector. Migrations in `db/migrations`, applied by the CLI.

```sql
entities(id, kind, canonical_name, country, state, city, status, created_at)
   -- kind: institution | programme | hospital | exam | lender | city
entity_aliases(id, entity_id, alias, alias_type, source_id, confidence)
   -- alias_type: acronym | former_name | transliteration | misspelling | local_name

documents(id, url, domain, kind, first_seen, last_seen, etag, last_modified,
          robots_decision, robots_checked_at, http_status)
snapshots(id, document_id, sha256, bytes_path, fetched_at, fetcher_tier,
          content_type, headers_json)
extractions(id, snapshot_id, tool, tool_version, model, prompt_version,
            page_range, text, structure_json, created_at)
claims(id, extraction_id, entity_id, predicate, value_json, unit,
       qualifier_json, quote, char_start, char_end, confidence,
       observed_at, method, superseded_by)
   -- method: extracted | derived | modelled | user_provided
llm_calls(id, purpose, prompt_sha256, prompt_path, model, provider, params_json,
          raw_response, tokens_in, tokens_out, latency_ms, cost_usd, cache_hit)
programmes(id, institution_id, name, credential_level, duration_months,
           internship_months, affiliated_university_id, recognition_status,
           seats, medium, created_at)
   -- credential_level: degree | diploma | lateral_entry | certificate
scores(id, entity_id, scorer_version, weights_version, claims_hash, computed_at,
       sub_scores_json, composite, composite_lo, composite_hi, evidence_tier)
insight_sets(id, entity_id, scores_id, insight_ids_json, coverage_json)
crawl_frontier(domain, url, priority, next_allowed_at, attempts, state,
               tier_used, ban_score, yield_score)
domain_politeness(domain, rps, crawl_delay, robots_json, robots_fetched_at,
                  quarantined_until)
exams(id, name, country, scope, provider, registration_opens, registration_closes,
      earliest_sitting, latest_sitting, requires_travel, remote_options,
      fee_amount, currency, source_claim_id)
exam_windows(id, exam_id, cycle_year, registration_opens, registration_closes,
             sitting_dates_json, source_claim_id, observed_at)
cost_items(id, programme_id, category, label, amount, currency, period,
           basis, source_claim_id, confidence)
   -- category: tuition | hostel | mess | rent | utilities | food | transport |
   --           equipment | exam_fee | visa | insurance | misc
   -- period: one_time | per_year | per_month
financing_options(id, scope, lender, scheme, rate_min, rate_max, collateral_rules,
                  max_amount, moratorium_months, tenure_months, subsidy_rules,
                  source_claim_id, observed_at)
```

**Hard rules**
- `claims` is append-only. Corrections insert a new row and set `superseded_by`.
- `claims.quote` is stored verbatim. `char_start/char_end` index into `extractions.text`.
- `method='modelled'` claims can never be cited as if extracted; the API returns them in a separate array.
- Every score row stores `claims_hash` so a score can be tied to the exact evidence snapshot.

## 7. Ingestion & crawling

**Fetch ladder** (`fmd/fetch`), escalate only on block signals (403/429/503/`cf-mitigated`), never on 404/410:
`httpx` (HTTP/2, pooled) → `curl_cffi` impersonate → `patchright` (headful under Xvfb, persistent context) → quarantine + retry later. Record `fetcher_tier` per snapshot. Tier usage per domain becomes the ban-risk metric.

**Politeness:** per *registrable* domain via `tldextract` (critical for `ac.in`/`gov.in`), default 1 rps + jitter, honour `Crawl-delay` and `Retry-After`, `protego` for robots with the decision logged per URL, 5 consecutive 5xx ⇒ 6h quarantine, `yield_score` < 2% after 200 pages ⇒ stop.

**Orchestration:** Scrapy 2.20 as scheduler/politeness engine, with the frontier in Postgres so a restart loses nothing and crawl budget is inspectable in SQL.

**Conditional re-crawl:** store and send `If-None-Match`/`If-Modified-Since`; a 304 costs nothing and keeps staleness honest.

**Dedupe:** sha256 of normalised bytes for snapshots; simhash of extracted text to catch the same PDF mirrored across portals.

**Discovery order for India:** seed registry (UGC universities → state health-science universities → affiliated colleges) → sitemaps on college domains → SearXNG for "placement report", "BSc MLT", "seat matrix" → Common Crawl URL index (DuckDB over the Parquet index) for `*.ac.in`/`*.gov.in` PDF sweeps. Do not blind-crawl the open web; every URL must be attributable to a registry entry or a query template.

## 8. Source registry (India v1)

`config/sources.yaml` entries must carry: `id, name, base_url, kind, format, cadence_days, trust, adapter, enabled, notes`.

| Family | Concrete sources | Adapter notes |
|---|---|---|
| University inventory | UGC filterable list + consolidated PDFs | Paginated HTML; tokenised query params |
| Enrolment context | AISHE final report, AISHE dashboard, data.gov.in extracts | Dashboard is JS; prefer published Excel |
| Regulator | NCAHP site, MLS 2025 curriculum, AHP Registration Regulations 2026 | PDF, low volume, high trust — these drive hard filters |
| State councils | Per-state paramedical councils (patchy, sometimes unofficial domains) | Generic HTML/PDF adapter; `requires_manual_review` flag |
| Affiliating universities | RUHS, WBUHS, TNMGRMU, BFUHS, KNRUHS, and peers | College lists, seat matrices, syllabi, exam calendars |
| Quality | NAAC results, NIRF rankings, UGC autonomous list | NIRF = institution-level proxy only |
| Clinical infrastructure | NABL accredited labs, NABH accredited hospitals | Proximity proxy for training quality |
| Admissions | State counselling: MHT CET, KCET/KEA, TN health, LBS Kerala, JENPAS (WBJEEB), BFUHS, OJEE, DME MP, DGME UP, EAPCET/KNRUHS, RUHS | Seat matrices + eligibility + calendars as PDF/HTML |
| Fees | State fee regulatory commissions (MH, RJ, KL, TN, MP) | Scanned PDF orders; partial coverage by design |
| Pay anchor | 7th CPC pay matrix (Level 5/6) | Static PDF, anchors the government-salary band |
| Financing | Vidya Lakshmi, CSIS, PM-Vidyalaxmi scheme pages, bank rate pages | Public terms only; never present as approval |
| Institution sites | Discovered per college | Placement PDFs usually live in NAAC SSR/AQAR/NIRF DCS annexures |

**Do not scrape:** NIRF DCS behind login, paywalled job portals, anything requiring CAPTCHA solving or auth, or sites whose robots.txt forbids it. Record the exclusion in the source registry with a reason.

## 9. Extraction

Tiered, cheapest-first, with a mandatory VLM audit pass:

| Tier | Tool | Use |
|---|---|---|
| T0 | `pypdfium2` / `pdftext` | born-digital text; ~30–50% of corpus |
| T1 | Docling | layout + tables on clean scans |
| T2 | MinerU / dots.ocr on vLLM | scanned tables, Devanagari headers |
| T3 | 7B VLM (olmOCR-class) | field-level adjudication on 1% sample + failures |

**Guided JSON only.** Pydantic schema → xgrammar via vLLM, or provider structured output. Every field includes `value`, `unit`, `quote`, `char_start`, `char_end`, `confidence`. The schema must permit `null` — schema-constrained decoding otherwise forces a plausible-but-wrong value.

**Validation gates** (a claim is rejected, not "fixed", if):
- quote not found at the given span (fuzzy match ≥ 0.95 allowed, else reject);
- unit missing or unparseable;
- denominator missing on any rate;
- value outside a configured plausibility band for that predicate.

**Prompt/eval contract:** prompts live in `fmd/extract/prompts/<name>/<version>.md`, referenced by `prompt_version`, hashed into `llm_calls`.

## 10. Entity resolution

`normalise` (NFKC, strip "University of", expand IIT/NIT/IIIT/BITS-style lexicons, transliterate Devanagari) → `block` (splink with DuckDB backend; rules on `(state, first_token)` and `(pincode, soundex)`) → `compare` (bge-m3 embeddings + pgvector ANN for alias candidates) → `score` (Fellegi–Sunter `m`/`u` priors) → `adjudicate` with LLM **only** in the 0.40–0.85 band, batched, cached by pair hash.

Alias types are first-class because the original brief demands acronym/alternate-name handling: `acronym | former_name | transliteration | misspelling | local_name`. Aliases discovered from headers/footers of placement PDFs are high-value — that is how you find "KGMU" in a document that only says "King George's Medical University".

## 11. Scoring

**Sub-scores** (each returns `{point, lo, hi, n_claims, n_sources, evidence_tier}`), weights in `config/weights.yaml`, user-adjustable at query time:

| Sub-score | v1 weight | Built from |
|---|---|---|
| `outcomes` | 30 | internship→offer conversion, registration conversion, P25 salary vs city CoL, 3y growth, report-integrity flags |
| `clinical_training` | 15 | NABL/NABH-linked hospital access, internship hours, departments rotated, equipment/recency claims |
| `academic_quality` | 10 | NAAC grade, university affiliation, faculty counts, autonomous status, NIRF proxy |
| `affordability_roi` | 20 | TCOA, net price, DSR₂₅, CoL-adjusted discretionary income |
| `financing_feasibility` | 5 | published loan terms, subsidy eligibility, collateral thresholds, legal student-work offset |
| `student_life_safety` | 10 | city safety, hostel availability, gender-specific signals, support services, review NLP |
| `admission_realism` | 10 | cutoff percentiles vs user's marks, eligibility margin, seat competition |

**Aggregation:** documented monotone transforms + published weights + empirical-Bayes shrinkage toward peer-group prior (ownership × state × affiliation type). Display `measured` and `prior_adjusted` separately so shrinkage never hides a real signal.

**Predicate registry (`config/predicates.yaml`):** the ninth registry file is the single authoritative map from an emitted claim predicate (`seats.total`, `curriculum.internship_months`, `accreditation.naac_grade`, …) to its sub-score component, unit kind, and denominator/period requirements, plus an optional scoring method. `fmd/score` never hardcodes a predicate name. Loader validation rejects a missing extraction predicate, a unit-kind disagreement with the extraction schema, a dangling component reference, and an alias that points nowhere; predicates absent from both the map and the declared aliases are reported as `unknown_predicates`, never silently dropped. A claim whose unit or period contradicts the registry is rejected (`SubScore.rejected`) rather than coerced.

**Component and composite minimums:** a component is scored from `n ≥ 1` usable claim, but its interval is widened by `sqrt(min_claims_for_confidence / n)`; a single perfectly-copied span is not a complete score. A composite is published from at least `min_components_for_composite` known components (default 2). A subset composite is `partial`: its point is the weighted mean of known components, while `lo`/`hi` still carry the full `[0, 100]` mass of every unknown component, and `components_used` / `components_unknown` are reported explicitly. Zero known components is `UNKNOWN` `[0, 100]`. A subset composite is never tiered `well_evidenced`.

**Missing data:** `UNKNOWN` propagates to `[0, 100]` for that sub-score. Comparisons use **Pareto dominance**: A is shown as better than B only if A ≥ B on every known sub-score and strictly greater on one. Undominated pairs are rendered "not comparable on current evidence". No imputation of informative non-disclosure.

**Salary/quantile estimation** (`fmd/score/quantiles.py`):
- median + IQR ⇒ direct quantiles;
- median + range + n ⇒ Hozo (2005) / Wan (2014) estimators;
- mean + median ⇒ log-normal ⇒ `Q1 = m·exp(−0.6745σ)`, `σ² = 2 ln(μ/m)`;
- mean only ⇒ report the Cantelli bound `[0, 1.33μ]` and refuse a point estimate;
- n < 10 ⇒ no empirical quantile; log-normal with sector prior on σ, interval widened by t/√n.
Every salary output carries `n`, method, and its interval. Never a naked point estimate.

**Gaming resistance:** an additive red-flag score attached to the *claim*, not the university: no denominator; no distribution; "highest package" as headline; suspiciously round figures; YoY jump > 2σ of peers; programme-level data aggregated across degrees; recruiter list without counts; offer letters presented as joinings. Flags lower claim confidence and are displayed verbatim in the UI.

**Insight selection** (the "top few that form the complete picture" requirement): greedy max-coverage over claim/predicate embeddings — pick the smallest set of insights such that (a) each adds a predicate not already represented, (b) marginal information gain decays, (c) every hard-filter failure is included regardless. Store as `insight_sets` with the coverage report so it is auditable.

## 12. Hard filters (`fmd/filters`)

Evaluated before ranking, each returning `pass | fail | unknown` plus the claim that decided it:

1. `credential_level == degree` (and `lateral_entry` only if the profile allows it).
2. `recognition_status == recognised` (NCAHP/state council/university affiliation chain resolvable).
3. `eligibility`: 10+2 PCB, minimum aggregate, age — compared against profile.
4. `language`: required test + minimum score vs profile, or medium-of-instruction exemption.
5. `exam_registration_open`: `now < registration_closes` using the *observed* window with its `observed_at`, plus TTL. Stale windows return `unknown` and render as "unverified", never as "open".
6. `exam_travel`: if `requires_travel` and the profile forbids travel ⇒ fail. `unknown` renders as "verify".
7. `intake_year_available`: profile's target intake must exist in the programme's intake cycles.
8. `budget_ceiling`: profile ceiling vs TCOA range; fail only when `lo > ceiling`.

`unknown` is never silently treated as `pass`. The UI shows "N programmes excluded" with the reason histogram, and lets the user inspect each exclusion.

## 13. Cost & financing model (`fmd/cost`)

**Definitions, each rupee counted once:**
1. `TCOA` = tuition + mandatory fees + living + books/equipment + internship travel. No opportunity cost.
2. `net_price` = TCOA − scholarships/waivers/fee reimbursement.
3. `family_out_of_pocket` = net_price − loan principal disbursed.
4. `debt_at_repayment` = principal + capitalised moratorium interest.
5. `EMI` at contractual rate and tenure.
6. **`DSR25` = first-year EMI ÷ P25 first-year gross salary** for the placement city.
   Bands: `<0.20` comfortable · `0.20–0.30` manageable · `0.30–0.40` stressed · `>0.40` not financeable.
7. `affordability_gap` = family OOP ÷ annual family income (flag > 1.0).

**Living toggle:** two complete, non-additive scenarios — `hostel` (institution hostel + mess) and `independent` (PG rent + food + utilities + transport). Presented side by side, never summed. Where a component is unpublished, use the city basket with `method='modelled'` and a ±20% band.

**Student-work offset:** represent statutory hours (India: effectively none — cap realistic informal earnings at ≤10% of TCOA, labelled "informal"), effective net wage, term vs vacation split, and currency mismatch when applicable. Never let an offset reduce `TCOA`; it reduces `family_out_of_pocket` only.

**FX:** any non-INR component carries a ±10% sensitivity band. No point FX rates in stored scores.

## 14. Student-life / review NLP (`fmd/extract/nlp`)

Aspect taxonomy fixed in config: `clinical_exposure, lab_infrastructure, faculty, placement_support, internship_quality, administration, safety, hostel, cost_transparency`.

Passes: ABSA for polarity → BERTopic for *discovery only* (surfacing unanticipated aspects, never scoring) → LLM rubric scoring with mandatory quoted evidence.

Guards: no aspect score below n≈5 mentions; shrink 5–30 toward peer prior with interval; near-duplicate/burst detection for astroturfing; per-source fixed effect (never pool Collegedunia/Reddit/Quora into one number); weight by specificity, not stars. Report `n` and source mix beside every aspect score.

## 15. Model router (`fmd/llm`)

Provider-agnostic. Config in `config/models.yaml`:

```yaml
profiles:
  bulk_extract:   {provider: local|api, model: ..., max_tokens: ..., guided_json: true}
  hard_pdf:       {provider: ..., model: ...}
  adjudicate:     {provider: ..., model: ..., temperature: 0}
  nlp_rubric:     {provider: ..., model: ...}
  embed:          {provider: ..., model: bge-m3}
budgets: {daily_usd: ..., per_run_usd: ...}
```

Requirements: deterministic sampling for extraction/adjudication, prompt caching keyed by `(purpose, prompt_sha256, input_sha256, model)`, per-run cost accounting, hard budget stop, and a `--mock` provider that replays recorded responses so the whole pipeline runs offline in CI.

**Hardware note:** 16 GB VRAM ⇒ local serving is ≤14B at int4. Use local for bulk schema extraction and 7B OCR; escalate to API for adjudication and hard PDFs. Do not design around a 70B local model.

## 16. User profile (generalized)

`config/profile.schema.json`, all fields optional with explicit "prefer not to say":

```json
{
  "nationality": "IN",
  "target_intake_year": 2027,
  "target_field": "mlt",
  "credential_levels": ["degree"],
  "marks": {"board": "CBSE", "class12_aggregate_pct": 78, "pcb_pct": 76},
  "english_test": {"type": "IELTS", "score": null, "status": "not_taken"},
  "entrance_exams": [{"exam": "MHT_CET", "score": null, "status": "planned"}],
  "budget": {"currency": "INR", "annual_ceiling": 400000, "total_ceiling": 2000000},
  "financing": {"loan_required": true, "family_income_band": "4.5-8L", "collateral_available": false},
  "living_preference": ["hostel", "independent"],
  "travel_for_exams": "no",
  "regions": ["MH", "KA", "TN"],
  "qualitative": {"extracurriculars": 3, "leadership": 2, "clinical_exposure": 1},
  "risk_appetite": "estimated_ok"
}
```

`qualitative.*` are 1–5 self-ratings, used only where a programme's admission model actually weighs them (most Indian MLT admissions do not) — the UI must say so rather than pretend.

## 17. API surface (server-side, consumed by the web app)

```
GET  /api/health
GET  /api/programmes?filters...        list + facets + exclusion histogram
GET  /api/programmes/{id}              full record, claims by predicate, costs, exams
GET  /api/programmes/{id}/evidence     paginated claims with quotes + source links
GET  /api/programmes/{id}/cost?living=hostel|independent
GET  /api/rankings?profile=...&weights=...   ranked + dominance + confidence + insights
GET  /api/compare?ids=a,b,c            Pareto matrix + differing claims
GET  /api/exams?profile=...            open/closed/travel-required with observed_at
GET  /api/sources                      registry + last crawl + trust
POST /api/scoring/rerun                admin: recompute with a given weights version
GET  /api/review/queue                 low-confidence claims needing human adjudication
POST /api/review/claims/{id}           accept / reject / correct (human label)
```

Every response includes `scorer_version`, `weights_version`, `claims_hash`, and `generated_at`.

## 18. Web app (v1 screens)

1. **Profile** — the form above, with "unknown" as a first-class choice.
2. **Shortlist** — ranked cards: composite band (not a fake decimal), evidence tier, DSR₂₅, TCOA range, living toggle, hard-filter status, and the selected insight set.
3. **Drill-down** — sub-score bars with intervals and `n`, every number clickable to its quote; separate "measured" vs "modelled" tabs; red flags list; cost scenarios side by side.
4. **Compare** — Pareto matrix, "not comparable" states made explicit.
5. **Exams** — open/closed/unverified, registration windows with `observed_at`, travel requirement, past-paper links where public.
6. **Exclusions** — why each programme was filtered out, with the deciding claim.
7. **Review console** — low-confidence claim queue, accept/reject/correct, source snapshot viewer with the highlighted span.

Designed for distrust: no spinner-until-you-agree, no dark patterns, no "best university" headline. The UI's job is to make the evidence legible.

## 19. Evaluation (`fmd/eval`)

- **Golden set:** 120–150 programmes, stratified by state, ownership, recognition status, and *data thinness* (deliberately over-sample thin evidence). Two annotators, adjudication, target Krippendorff α ≥ 0.7. Frozen and versioned.
- **Extraction:** span-level P/R/F1, value accuracy with unit tolerance, and **provenance coverage = 100%** (every published number resolves to a retrieved quote; anything less is a bug).
- **Scoring:** Kendall τ vs human panel; interval coverage (do 80% intervals contain truth ~80% of the time?); Brier score for evidence tiers; abstention correctness.
- **Drift:** PSI/KS on sub-score distributions per release, top-20 Jaccard churn, canary set re-scored every pipeline version. Any movement must be attributable to a claim diff or a versioned scorer change.
- **Offline CI:** `--mock` model provider + 20 recorded snapshots ⇒ full pipeline runs in minutes with no network.

## 20. Milestones & acceptance criteria

| ID | Deliverable | Done when |
|---|---|---|
| M0 | Foundation | `docker compose up` gives Postgres+pgvector+API+web; migrations apply; config registry validates; CI green on `--mock` |
| M1 | Inventory | ≥300 Indian institutions and ≥150 BSc MLT programmes with recognition chain resolvable; UGC + state-university + counselling adapters live |
| M2 | Evidence | ≥1,000 snapshots; claims extracted with 100% span resolution on a 200-doc sample; `llm_calls` fully audited |
| M3 | Scoring + filters | All 7 sub-scores produce intervals; 8 hard filters implemented with `unknown` handling; DSR₂₅ computed; golden set scored |
| M4 | Web app | 7 screens functional against real data; every number click-throughs to a quote; exclusions explainable |
| M5 | Review + eval | Review console with source-span viewer; drift + coverage reports; docs complete |

## 21. Workstreams for delegation

| WS | Scope | Depends on | Parallel-safe |
|---|---|---|---|
| A | Foundation: compose, migrations, config loader, DB repos, snapshot store, model router + mock provider, CLI skeleton | — | blocks all |
| B | Fetch ladder + robots/politeness + frontier + discovery (SearXNG, CC index) | A | yes |
| C | Adapters: UGC, AISHE, NCAHP, NAAC, NIRF, NABL/NABH, state counselling, fee regulators, college-site generic | A, B | yes |
| D | Extraction: PDF tiers, schema registry, guided JSON, claim validation, prompts | A | yes |
| E | ER: normalisation, splink, alias mining | A, D | yes |
| F | Scoring, filters, cost/financing, insight selection, eval harness | A, D | yes |
| G | Web app: 7 screens + API client + review console | A, F | partial |

**Contract discipline:** B–G may not invent schema. Any schema change goes through a migration added by A's owner-of-record (or a new numbered migration reviewed against §6). Any new source is a YAML entry + adapter implementing `SourceAdapter.fetch() -> Iterable[DocumentRef]`.

## 22. Definition of done (phase 3)

1. Fresh clone → `make up` → `make seed` → `make crawl` → `make score` → `make dev` works on a clean machine.
2. A profile with only `{nationality, target_intake_year, target_field}` returns a ranked shortlist with honest `unknown` states.
3. Every displayed number resolves to a verbatim quote, URL, and retrieval timestamp.
4. Excluding a programme always shows the deciding claim and the filter that fired.
5. `make eval` prints extraction accuracy, provenance coverage, interval coverage, and drift vs the previous run.
6. The repo contains no reference to any specific applicant, family, budget, or intake year outside `config/` and test fixtures.
