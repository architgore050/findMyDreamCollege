# 00 — Research summary: external ground-truth sources for product evals

**Wave:** FMD-CLOUD-022 / w11 · eval-strategy  
**Scope:** research only. No code.  
**Date:** 2026-11-22  
**Base SHA:** 66702b20b603bc36babbca76be359ae3a3a8209e  

This document catalogues external sources that could serve as ground truth (or proxy ground truth) for the product question: *given a profile ({nationality, target_intake_year, target_field}), does the ranked shortlist the product produces actually match where students go / what is defensible?*

Existing `fmd eval` (doc 08) measures pipeline correctness (extraction span accuracy, provenance coverage, drift). It does **not** measure product usefulness. The sources below are candidates for closing that gap.

All four invariants from AGENTS.md apply to evals themselves:

1. No user-specific logic in the eval harness.  
2. Numbers are computed by the harness, never emitted by a model.  
3. UNKNOWN is a first-class value; coverage gaps must be reported, never defaulted.  
4. Every golden or truth datum carries provenance (URL, fetched_at, sha256).

---

## A. Indian institutional quality & ranking sources

### A.1 NIRF (National Institutional Ranking Framework)

- **URL:** https://www.nirfindia.org/ (rank lists, methodology PDFs per category/year)
- **What it provides:** Annual ranks and parameter scores for Overall, Universities, Colleges, Medical, Engineering, etc. Five parameters (TLR, RP/RPII, GO, OI, PR) with published weightages (typical: 30/30/20/10/10; Colleges and some categories differ). Graduation Outcomes (GO) includes exam performance, median salary (where reported), progression.
- **Maturity:** High for medical colleges (AIIMS etc.); **low for paramedical / B.Sc. MLT specifically**. NIRF has no dedicated paramedical or allied-health category. Institution-level Medical rank is a weak proxy for an MLT programme inside a medical college or a pure paramedical college.
- **Licensing / usability:** Public data; methodology documents are open. Rank lists are reproducible from submitted data + third-party verification (Scopus etc.). Suitable as a *proxy signal* at institution level, never as programme-level ground truth for MLT.
- **Honest gaps:** No programme-level MLT data; GO median salary is self-reported and often incomplete for non-engineering/non-medical-core programmes; participation is voluntary. Intake-year mismatch is large (rank year vs student cohort year).
- **Eval use:** Fast, data-driven proxy for “institution quality” in a reference ranking. Cannot answer “best MLT programme for profile X”.

### A.2 NAAC (National Assessment and Accreditation Council)

- **URL:** http://naac.gov.in/ (grade lists, manuals, CGPA bands)
- **What it provides:** Institutional grades (A++ … C) and CGPA on seven criteria (Curricular Aspects, Teaching-Learning & Evaluation, Research, Infrastructure, Student Support & Progression, Governance, Institutional Values). Revised frameworks (RAF, Binary 2025) change process but grades remain public.
- **Maturity:** Medium–high for institutional quality; Student Support & Progression criterion is the closest to outcomes. Not programme-specific.
- **Licensing / usability:** Public. Grades are stable for 5-year cycles; useful as a slow-changing quality filter or expert-judgment proxy.
- **Honest gaps:** Binary vs graded regimes; many paramedical colleges unaccredited or low-grade; no MLT-specific metrics; peer-team visit introduces subjectivity.
- **Eval use:** Expert-judgment benchmark / hard filter for “recognised quality institution”. Not outcome ground truth.

### A.3 AICTE / NCAHP / UGC / state counselling matrices

- **URLs:** AICTE portal, NCAHP regulations (2025 MLS curriculum), UGC university list (already in `config/sources.yaml`), state seat matrices (RUHS, TNMGRMU etc. — already mined).
- **What they provide:** Recognition status, approved intake, curriculum structure (4-year B.Sc. MLT + 1800-hour internship), hard eligibility filters.
- **Maturity:** High for *recognition* and *inventory*; zero for employment outcomes.
- **Eval use:** Already used as hard filters and inventory. Essential for “is this programme even eligible?” but not for ranking quality of outcomes.

### A.4 MoHFW / MoSES statistics, DIET, placement-report archives

- Limited public programme-level placement or registration-conversion data for MLT. Placement reports exist on individual college sites (already targeted by the pipeline) but are self-reported, often gamed, and rarely distinguish internship vs employment (invariant 6).
- Newspaper / industry HR surveys exist but are sparse, non-longitudinal for MLT, and hard to attribute to specific programmes.

---

## B. Outcome / employment data (India)

### B.1 PLFS / NSS (Periodic Labour Force Survey, National Sample Survey)

- **Source:** MoSPI / NSO annual PLFS reports (e.g. 2025: graduate unemployment ~11.2 %).
- **What it provides:** Aggregate unemployment / labour-force status by education level, sex, rural/urban. No programme or institution identifiers.
- **Maturity:** High for national aggregates; **unusable for programme-level product eval**. Useful only as a background prior or for “is the field employable at all?”.
- **Licensing:** Public government data.
- **Honest gaps:** No microdata linkage to specific colleges or MLT programmes; sampling frames do not support college-level inference.

### B.2 India Skills Report / Wheebox / CII employability surveys

- Aggregate employability percentages by broad degree family. Not MLT-specific at usable resolution.
- Useful as a slow prior; not as per-profile ground truth.

### B.3 College placement reports & LinkedIn / CV corpora

- **Placement reports:** Already in the pipeline’s evidence model. Self-reported; internship vs employment distinction is fragile (see invariant 6). Coverage is sparse and biased toward colleges that publish.
- **LinkedIn / public CVs / Shiksha / Collegedunia alumni snippets:** “Wisdom-of-the-crowd” signal. High noise, selection bias, privacy constraints, ToS risk. Can support role-model or crowd benchmarks at low volume, but cannot be the primary golden set.
- **Honest gaps:** No reliable, large-N, programme-linked, longitudinal placement microdata for Indian B.Sc. MLT exists in the public domain. Any claim of “mature outcome dataset” for this field is currently false.

### B.4 Corporate / hospital recruitment data, batching events

- Occasional public hiring announcements or campus-drive reports. Extremely sparse; not systematic. Feasible only as qualitative evidence inside a verdict-style research agent, never as a scored golden set.

---

## C. International precedent (for method, not direct data)

### C.1 US College Scorecard (NCES / Dept of Education)

- **URL:** https://collegescorecard.ed.gov/ (data files, field-of-study earnings, technical documentation)
- **What it provides:** Median earnings 1–10 years after entry/completion, debt, completion rates, programme-level (CIP) earnings where N is sufficient. Linked administrative tax + NSLDS data.
- **Maturity:** High. Value-added models and ROI rankings are published (Brookings, Georgetown CEW, etc.). Clear statistical-power caveats and suppression for small N.
- **Usability for fmd:** Methodological template only. US data cannot evaluate Indian MLT programmes. Demonstrates what “mature outcome product” looks like: multi-year cohorts, administrative linkage, programme-level where possible, honest UNKNOWN for small cells.

### C.2 QILT (Australia), THE, other national systems

- Student experience and graduate outcome surveys. Similar methodological lessons: longitudinal, survey + administrative hybrid, field-of-study breakdowns, clear coverage statements.

### C.3 How products claim maturity

- Years of successive cohort data + published methodology papers + statistical-power disclosures + external validation (third-party audits or academic replications). No Indian MLT product currently meets this bar; most rely on NIRF/NAAC proxies + self-reported placement %.

---

## D. Summary table (usability for the product question)

| Source | Level | Outcome signal | Coverage for B.Sc. MLT | Licensing | Usable as golden? | Notes |
|--------|-------|----------------|------------------------|-----------|-------------------|-------|
| NIRF ranks / GO | Institution | Weak (median salary, progression) | Proxy only | Public | Proxy only | No paramedical category |
| NAAC CGPA / grade | Institution | Indirect (Student Support) | Medium | Public | Filter / expert proxy | 5-year cycle |
| Seat matrices / NCAHP | Programme | None (recognition/intake) | High (inventory) | Public | Already used | Hard filter |
| PLFS / NSS | National | Aggregate unemployment | None | Public | Prior only | No college link |
| College placement reports | Programme | Self-reported rates/salaries | Sparse, biased | Public (scraped) | Fragile golden | Internship vs employment risk |
| LinkedIn / CV / review sites | Individual | Employment / salary anecdotes | Very sparse | ToS / privacy | Crowd / role-model only | High noise |
| College Scorecard (US) | Programme | Earnings, debt | N/A (US) | Public | Method template | Gold standard for design |

**Bottom line:** There is **no mature, public, programme-level outcome golden dataset for Indian B.Sc. MLT**. Product evals must therefore lean heavily on (1) proxy institutional signals, (2) verdict-style research agents that produce audited decisions per profile, and (3) carefully designed no-golden evals (cross-year calibration, expert panels, opt-in surveys) that treat UNKNOWN honestly.

See `01-eval-designs.md` for the concrete designs that follow from these findings.