# 01 — Eval designs for v1 product quality

**Wave:** FMD-CLOUD-022 / w11 · eval-strategy  
**Companion:** `00-research-summary.md` (source catalogue), `08-eval-methodology.md` (existing golden harness)  
**Invariants:** AGENTS.md four invariants + “score computed by harness, model may only verdict”.

This document designs three families of eval:

1. **Golden-dataset evals** (extend the existing pipeline eval world).  
2. **Evals that do not need a golden dataset** (survey, crowd, role-model, self-eval, time-asymmetric).  
3. **Verdict-style evals** (research agent per profile → audited “product is right / partially right / wrong”).

Plus an integration sketch for how streams plug into `fmd eval`.

---

## 1. Golden-dataset evals (taxonomy)

These require external or curated ground truth that can be stored under `data/golden/` (or a parallel `data/truth/`) with full provenance.

### 1.1 Extraction / provenance / drift (existing — baseline)

- **Data shape:** `data/golden/programmes/*.yaml` + fixtures (already defined in doc 08).  
- **Score rule:** span accuracy P/R/F1, row-attribution accuracy, provenance coverage %, drift vs previous report.  
- **Honest gaps:** Measures pipeline fidelity, not product usefulness.  
- **Cost / time:** Free, offline, deterministic.  
- **Status:** Implemented; keep as the hard gate.

### 1.2 Institutional proxy ranking (NIRF / NAAC)

- **Data shape:** Yeared NIRF parameter scores + rank lists (institution → Medical / Overall / College category); NAAC grade + CGPA. Stored as YAML or Parquet with `source_url`, `fetched_at`, `sha256`, `as_of`.  
- **Score rule (product-facing):** For a test profile, compute the product’s ranked shortlist. Build a reference ranking of the same candidate set ordered by NIRF Medical (or Overall) score, or by NAAC CGPA. Metrics:  
  - Top-k overlap (k=3,5)  
  - Rank correlation (Spearman / Kendall) on the overlapping set  
  - “Best available” hit rate: does the product’s #1 appear in the reference top-3?  
  Report UNKNOWN when the institution has no NIRF/NAAC entry.  
- **Honest gaps:** Proxy only; no MLT programme signal; voluntary participation; GO salary incomplete; 1–2 year lag.  
- **Cost / time:** Low (annual scrape or manual download of public PDFs/CSVs). Fast.  
- **Risk:** Circular if the product later starts weighting NIRF heavily; document as external reference only.

### 1.3 Recognition / inventory fidelity

- **Data shape:** UGC / NCAHP / state seat-matrix ground truth already partially goldened.  
- **Score rule:** Precision/recall of “recognised_institute” filter and seat totals against golden seat matrices.  
- **Gaps:** Already partially covered; expand to more states.  
- **Cost:** Low.

### 1.4 Placement-report extraction accuracy (fragile golden)

- **Data shape:** Curated placement PDFs → expected `internship_to_offer_conversion_12m`, `p25_first_year_salary`, etc., with strict outcome_type separation.  
- **Score rule:** Same span + value accuracy as 1.1, plus invariant-6 checks (internship never counted as employment).  
- **Honest gaps:** Self-reported, sparse, often gamed; small N. Treat as “extraction correctness on placement family”, never as “true outcomes”.  
- **Cost:** Medium (curation labour).  

### 1.5 Role-model / alumni outcome golden (slow, sparse)

- **Data shape:** Small set of anonymised or public alumni trajectories (LinkedIn public profiles, published case studies) mapped to programme + graduation year + first-job title / sector.  
- **Score rule:** Does the product’s shortlist contain the programme the role-model attended? Or rank-percentile of that programme.  
- **Gaps:** Extreme selection bias, privacy, ToS, tiny N. Use only as a qualitative canary, never as a primary metric.  
- **Cost:** High (manual research + legal review).

---

## 2. Evals that do NOT need a golden dataset

These produce a verdict or score from process, cross-validation, or external research that is not a fixed labelled set.

### 2.1 Cross-validation / leave-N-out on the product’s own candidate set

- **Idea:** For a fixed profile, hold out N programmes that the product ranked highly; re-score the remaining set; check rank stability and whether the held-out programmes would still surface under reasonable weight perturbations.  
- **Data shape:** None external; uses the live (or fixture) candidate pool.  
- **Score rule:** Rank stability (Kendall-tau under leave-N-out and under ±ε weight noise); fraction of top-k that survive.  
- **Gaps:** Measures internal consistency, not external correctness. Useful as a “does the ranking collapse under small changes?” canary.  
- **Cost / time:** Fast, pure computation.

### 2.2 Expert-judgment / AHP-style panels (NAAC, domain experts)

- **Idea:** Small panel of domain experts (or structured use of NAAC/NIRF criteria as expert proxies) produces pairwise or ranked preferences for a handful of profiles. Product ranking is compared to the panel.  
- **Data shape:** Panel responses (YAML or form export) with provenance (who, when, conflict disclosures).  
- **Score rule:** Agreement metrics (top-k overlap, rank correlation); inter-rater reliability (Krippendorff / ICC) reported separately.  
- **Gaps:** Expensive, slow, small coverage; panel composition bias.  
- **Cost:** High (human time). Slow stream.

### 2.3 Wisdom-of-the-crowd / public review aggregation

- **Idea:** Aggregate public signals (Shiksha, Collegedunia, Reddit, Quora, YouTube transcripts) for “would recommend / placement satisfaction” per programme. Treat as soft labels.  
- **Data shape:** Extracted sentiment / rating distributions with source URLs and timestamps.  
- **Score rule:** Correlation of product rank with mean review score or % positive; coverage of programmes that have ≥K reviews.  
- **Gaps:** Severe selection and astroturfing bias; sparse for pure MLT programmes; ToS.  
- **Cost:** Medium (crawl + NLP already partially in pipeline).  

### 2.4 Self-eval / temporal drift of the product itself

- **Idea:** Freeze a set of profiles and the product’s shortlists at time T. At T+Δ re-run the same profiles against the updated evidence base. Measure how much the shortlist changed and whether changes are explained by new evidence (provenance) or by scorer drift.  
- **Data shape:** Versioned shortlist snapshots + claims_hash / weights_version.  
- **Score rule:** Rank change magnitude; % of changes that cite new claims with fetched_at > T; unexplained drift rate.  
- **Gaps:** Does not prove correctness; only that the product is stable or that changes are evidence-driven. Essential for “maturity over years”.  
- **Cost:** Low once snapshots are stored. Fast/slow hybrid (cron).

### 2.5 Time-asymmetric / rolling “did we get it right?” ladder

- **Idea:** Use N-year-old evidence to produce a shortlist for intake year N+k; later compare against what actually happened (placement reports, registration data, or retrospective expert judgment for that cohort). Builds a ladder of predictive accuracy.  
- **Data shape:** Historical evidence snapshots + later outcome labels (when they become available).  
- **Score rule:** Predictive hit rate of top-k vs realised outcomes; calibration of intervals.  
- **Gaps:** Requires waiting for outcomes; most MLT outcome data will remain sparse. Start with institutional proxies (did the high-NIRF programmes still rank high?).  
- **Cost:** High latency; low marginal compute.  

### 2.6 Survey-driven / opt-in student outcome waves (slow acquisition)

- **Idea:** Future opt-in surveys of graduates (or of users who accepted a recommendation) producing self-reported first-job / registration status.  
- **Data shape:** Anonymised survey responses linked only to programme + graduation year (never to user identity).  
- **Score rule:** Programme-level outcome rates vs product rank order.  
- **Gaps:** Cold-start (no data yet); response bias; privacy (GDPR-like / Indian DPDP compliance).  
- **Cost:** High (design, recruitment, incentives, legal). Explicitly a post-v1 / “deals study” item.

---

## 3. Verdict-style evals (core product-quality stream)

This is the main novel design requested. The eval itself is a specialised research agent that, for each profile in a test set, produces an audited verdict about the product’s shortlist.

### 3.1 What “research” means per profile

For a profile such as:

the verdict agent:

1. Retrieves the product’s ranked shortlist (top-N programmes) for that profile from a frozen fixture or a live but version-pinned API.  
2. Selects 1–3 programmes with the densest evidence in the product’s own claim store and 1–2 programmes that the product ranked highly but that have thin/no evidence (honest negatives).  
3. Conducts specialised research (public web, official PDFs, NIRF/NAAC, placement reports, curriculum pages, alumni signals) constrained to **citable** sources only. The agent is forbidden from inventing facts; every claim it uses must carry a URL + quote or be marked UNKNOWN.  
4. Builds a small reference ranking or preference set for that profile using only the research evidence + the external proxies in §1–2.  
5. Emits a structured verdict:

### 3.2 Scoring rule (harness-computed)

- The model is allowed only to emit the structured verdict + rationale + evidence pointers.  
- The harness computes:  
  - Agreement rate of `verdict == correct` across the test set.  
  - Partial-credit scoring (e.g. top-3 overlap with the agent’s reference_top3).  
  - Coverage: fraction of profiles for which the agent returned `unknown` or flagged sparse evidence.  
  - Auditability: every evidence pointer must resolve (URL reachable at evaluation time or archived sha256).  
- **Product is “correct” for a profile iff** the product’s recommended programmes are a subset of (or high-overlap with) the reference ranking built from the agent’s research + any available golden proxies, and no strong counter-evidence was found.  
- UNKNOWN is reported when evidence is insufficient; it is never coerced to a numeric score of 0 or 50.

### 3.3 Audit & anti-hallucination

- All agent-produced claims are re-checked by the same span/provenance machinery used in the golden harness where possible.  
- Human spot-check of a random 10–20 % of verdicts (second-person adjudication, as already practised for golden curation).  
- The eval report includes the full evidence graph for every incorrect or partially_correct case so failures are diagnosable.

### 3.4 Fast vs slow variants

- **Fast:** Agent restricted to already-ingested claims + NIRF/NAAC tables + a small set of high-trust PDFs. Deterministic retrieval, no live web.  
- **Slow:** Agent may perform live research (within robots/politeness rules) and may request new acquisitions. Used for periodic deep dives or for the “viable options” prioritisation.

### 3.5 Cost / time / risks

- Cost: model tokens + human audit time. Higher than pure golden.  
- Time: minutes to tens of minutes per profile (fast); hours for deep research (slow).  
- Risks:  
  - Circular-data trap (agent uses the same sources the product already mined → over-optimistic). Mitigate by requiring at least one external or held-out source.  
  - Leapfrog / model drift (newer model changes verdict style). Mitigate by pinning model version and by human audit.  
  - Staleness (research done in 2026 for 2027 intake). Explicit `as_of` on every verdict.  
  - Privacy (if any personal alumni data is used). Prefer public aggregate or institutional signals.

---

## 4. Integration sketch (design only)

### 4.1 Data layout

Every truth file carries:

### 4.2 CLI / harness hooks (sketch)

Extend `fmd eval` (or a sibling `fmd eval-product`):

- `--eval-kind pipeline` = today’s behaviour.  
- `--eval-kind product` = proxy ranking + cross-validation streams.  
- `--eval-kind verdict` = research-agent stream (may call model under pinned profile).  

Scoring remains harness-side. Model output is only the structured verdict object.

### 4.3 Scheduling & thresholds

- Fast streams (pipeline, NIRF-proxy, self-drift): CI or nightly cron.  
- Slow streams (verdict-deep, expert panel, survey): quarterly or on-demand.  
- Drift report: any stream can be compared to its previous run; unexplained rank changes raise a warning.  
- Hard gates: provenance coverage and extraction F1 stay as today. Product streams start as soft / advisory until coverage and inter-rater reliability are characterised.

### 4.4 How “maturity” is observed

Maturity is not a single number. It is the accumulation of:

1. Multi-year self-drift reports showing evidence-driven (not random) change.  
2. Growing coverage of proxy streams (more institutions with NIRF/NAAC).  
3. A slowly growing set of audited verdicts with high human agreement.  
4. (Future) first survey or administrative outcome labels that close the loop on a subset of programmes.

Until those exist, the product must surface coverage and UNKNOWN honestly rather than claiming validated recommendations.

---

## 5. Feasibility / risk notes (cross-cutting)

| Risk | Mitigation |
|------|------------|
| Circular data (product & eval share the same mined claims) | Require external or held-out evidence for “correct”; document source overlap |
| Privacy / ToS (LinkedIn, reviews) | Prefer public institutional data; anonymise; legal review before any bulk alumni scrape |
| Leapfrog (model or weights change changes historical verdicts) | Pin model + weights_version + claims_hash; store full verdict artefacts |
| Staleness | Every truth and every verdict carries `as_of`; rolling window reported |
| Sparse MLT data | Explicit UNKNOWN; never impute; prioritise coverage metrics |
| Cost of slow streams | Start with fast proxy + small verdict canary set; expand only after ROI is clear |

---

## 6. What to build first (see also 02)

1. Extend golden layout for NIRF/NAAC truth tables + a simple top-k / rank-correlation metric against product shortlists.  
2. Implement the self-drift / snapshot comparison stream (purely internal, zero external data risk).  
3. Design and pilot a 5–10 profile verdict-fast canary with human audit.  
4. Defer survey / LinkedIn / large role-model sets until legal and acquisition paths are clear.

All of the above respect the B2 / HANDOVER preference for world-data evals before proprietary data.