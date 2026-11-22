# Gaps & Full-Solution Backlog

Everything v1 deliberately does **not** do, ordered by how much it blocks the original vision. Future agents: pick from here, do not silently expand v1.

**Legend:** `P0` needed before the tool is useful to a stranger · `P1` needed for the full original brief · `P2` scale/polish · `BLOCKED` needs data/access we do not have.

---

## 1. Geography

| ID | Gap | Why it matters | Pri | Notes |
|---|---|---|---|---|
| G1.1 | v1 is India-only | Original brief wants global scoring | P1 | Architecture already country-scoped; add adapters, not forks |
| G1.2 | Tier B country adapters | Cheap value plays: Philippines, Malaysia, UAE/Saudi, South Africa | P1 | Each needs: regulator, programme inventory, licensure gate, CoL city basket |
| G1.3 | Tier C: AU/CA/UK/IE/NZ/US | High-outcome destinations inside the ₹15–35L band | P1 | Best public data exists here: QILT, HESA Graduate Outcomes, College Scorecard |
| G1.4 | Tier D: DE/NL/Nordics | Low tuition but language-gated for MLT | P2 | Filter by medium-of-instruction, not country |
| G1.5 | Country welcome / visa reality scoring | Original brief requirement; not implemented | P1 | Needs: visa grant rates for Indian students, post-study work rules, PR pathways, MLT shortage-list status |
| G1.6 | Country onboarding checklist | Adding a country is more than YAML | P2 | Write `docs/adding-a-country.md` before G1.2 starts |

## 2. Outcomes data

| ID | Gap | Why | Pri | Notes |
|---|---|---|---|---|
| G2.1 | No verified salary microdata | P25 salary is `modelled`, not measured | BLOCKED | Best proxies: EPFO establishment records, NIRF DCS public mirrors, RTI |
| G2.2 | No registration-conversion data | The strongest honest outcome metric | BLOCKED | Depends on state registers going live + public (NCAHP 2026 rollout) |
| G2.3 | Internship stipend compliance unknown | Materially changes net cost | P1 | Needs per-college policy extraction; frequently unpublished |
| G2.4 | Employer-verified placement | Defeats gaming best | P2 | Requires partnerships; long-term moat |
| G2.5 | Multi-year trend mining | Single-year snapshots hide manipulation | P1 | Crawl historical placement PDFs; compute YoY deltas |
| G2.6 | Hospital tie-up mapping | Best proxy for clinical training quality | P1 | NABH/NABL proximity is a weak proxy; distance + bed count + pathology staffing is better |

## 3. Admissions & exams

| ID | Gap | Why | Pri | Notes |
|---|---|---|---|---|
| G3.1 | Exam calendars are partial | Only state counselling cycles covered in v1 | P1 | Need: university-level tests, CUET, per-state paramedical notifications |
| G3.2 | No live registration-status verification | TTL-based staleness only | P1 | Requires per-exam adapters; some portals are JS + captcha → `unknown` by design |
| G3.3 | No past-paper linkage | Original brief requirement | P1 | Curate public repositories; never mirror copyrighted papers — link only |
| G3.4 | Qualitative self-assessment unused | Most Indian MLT admissions ignore extracurriculars | P2 | Keep the input; show "not applicable here" rather than faking a score |
| G3.5 | No cutoff-history model | `admission_realism` needs real cutoff distributions | P1 | Mine 3 years of state counselling cutoff PDFs |
| G3.6 | No "chance of admission" calibration | Would be the most requested feature, easiest to get wrong | P2 | Needs labelled outcomes; ship dominance instead until then |

## 4. Cost & financing

| ID | Gap | Why | Pri | Notes |
|---|---|---|---|---|
| G4.1 | Overseas loan products unscored | PM-Vidyalaxmi excludes foreign universities; NBFC terms differ structurally | P1 | Separate `financing_scope` axis: domestic vs overseas |
| G4.2 | No lender sanction-rate data | Published terms ≠ access | BLOCKED | Would need lender partnerships or FOI |
| G4.3 | CSIS eligibility for BSc MLT unresolved | Materially changes DSR for low-income families | P1 | Track per-lender checklists; mark `conditional` |
| G4.4 | Fee-regulator coverage is partial | Tuition sometimes `modelled` | P1 | Only MH/RJ/KL/TN/MP have usable orders; other states = estimate |
| G4.5 | Equipment/consumable costs unpublished | Under-reported line item | P1 | Mine college prospectuses + student mentions; label `modelled` |
| G4.6 | Scholarship inventory shallow | NSP + state schemes + institutional waivers | P1 | Needs per-state NSP scrapers |
| G4.7 | No FX hedging guidance | FX bands exist, advice does not | P2 | Product decision, not pipeline |

## 5. Data platform & operations

| ID | Gap | Why | Pri | Notes |
|---|---|---|---|---|
| G5.1 | No deployment | v1 is local docker only | P1 | Target: VM for pipeline/GPU, managed Postgres, object storage for snapshots, Next.js on Vercel |
| G5.2 | No public read-only DB/API | User wants a world-readable DB | P1 | Ship dumps + read replica + rate limits before opening it |
| G5.3 | No public data dump / versioning | Reproducibility + trust | P1 | Nightly Parquet export + immutable release tags |
| G5.4 | No auth or tenancy | Public tool, but review console must be protected | P1 | Minimal: single admin token for write paths |
| G5.5 | No scheduled crawling | v1 is manual `make crawl` | P1 | Cron + per-source cadence from the registry |
| G5.6 | No observability | Crawl yield, block rates, cost per claim | P2 | Metrics into Postgres first; dashboards later |
| G5.7 | No backup/restore rehearsal | Snapshots are the crown jewels | P1 | Test restore before any public launch |
| G5.8 | No legal review of scraping posture | robots-compliant, but still a risk | P1 | Document policy; honour takedowns; publish exclusion list |

## 6. ML/NLP quality

| ID | Gap | Why | Pri | Notes |
|---|---|---|---|---|
| G6.1 | No hierarchical model | v1 uses heuristics + shrinkage | P2 | Replace shrinkage without schema change, per doc 02 research |
| G6.2 | No learned ranker | No labels yet | BLOCKED | Requires multi-year outcomes + human panel |
| G6.3 | Devanagari OCR is shaky | Docling silently drops rows on merged-cell tables | P1 | Mandatory VLM audit pass on 1% sample; row-count reconciliation |
| G6.4 | Review corpus is thin | Few MLT-specific reviews exist | P1 | Widen sources: Reddit, Quora, YouTube transcripts, Telegram (public), LinkedIn posts |
| G6.5 | No astroturfing model beyond heuristics | Duplicate/burst detection only | P2 | Needs labelled spam set |
| G6.6 | No multilingual corpus | Hindi/Marathi/Tamil mentions ignored | P2 | bge-m3 covers embeddings; extraction needs prompt variants |
| G6.7 | Insight selection is greedy | Good enough, not optimal | P2 | Submodular maximisation would be a strict improvement |
| G6.8 | No causal claims | Correctly avoided in v1 | — | Do not add without experiment design |

## 7. Product surface

| ID | Gap | Why | Pri | Notes |
|---|---|---|---|---|
| G7.1 | No saved comparisons / shareable links | Core UX for family decisions | P1 | Stateless URL encoding, no accounts needed |
| G7.2 | No PDF/print dossier | Families share documents | P1 | Server-rendered report with provenance appendix |
| G7.3 | No alerts | Deadlines move | P1 | Email/Telegram on registration-window changes for saved programmes |
| G7.4 | No mobile layout | Most Indian users are mobile-first | P1 | Responsive first; native later |
| G7.5 | No Hindi/regional UI | Access barrier | P2 | i18n scaffolding early, translations later |
| G7.6 | No explainer content | Users do not know what DSR₂₅ or P25 means | P1 | Glossary + inline tooltips, plain language |

## 8. Trust & governance

| ID | Gap | Why | Pri | Notes |
|---|---|---|---|---|
| G8.1 | No correction/dispute channel | Universities and students must be able to contest | P1 | Public issue template + claim-level dispute flag |
| G8.2 | No data-source takedown log | Legal hygiene | P1 | Append-only exclusion registry |
| G8.3 | No versioned public methodology page | Trust depends on it | P1 | Generate from `config/weights.yaml` + scorer version |
| G8.4 | No privacy policy / DPDP posture | Profile data is personal | P1 | v1 stores nothing server-side; keep it that way until needed |
| G8.5 | No audit of prior-agent claims | Docs contain estimates | P2 | Golden set is the mitigation; cite it in methodology |

---

## Explicitly rejected (do not build)

- Scraping login-walled NIRF DCS, job portals, or paywalled review sites.
- CAPTCHA solving or proxy rotation to defeat protections.
- LLM-generated final scores.
- Single-number "best university" rankings without intervals.
- Imputing placement data for programmes that publish nothing.
- Treating internships as placements.
- Hardcoding any applicant's country, budget, intake year, or exam list.

## 8. Parse-cache scope (wave v1-w3, 2026-11-21)

| ID | Gap | Why | Pri | Notes |
|---|---|---|---|---|
| G8.1 | T2 parse cache covers in-process RapidOCR only | qwen3vl/paddleocr engine versions are only knowable by asking the service; caching under an unpinned identity would alias a service-side model swap | P1 | Cache key pins {engine, engine_version, model_ids}; device excluded. To cache HTTP engines, `/extract` must report model identity first (backend change). Refusals recorded as `t2_engine_fingerprint_unavailable`. |
