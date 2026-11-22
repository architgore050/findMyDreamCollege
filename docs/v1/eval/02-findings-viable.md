# 02 — Viable options summary: what to build first

**Wave:** FMD-CLOUD-022 / w11 · eval-strategy  
**Depends on:** `00-research-summary.md`, `01-eval-designs.md`

## Viable streams ranked by readiness

| Priority | Stream | Kind | External data needed? | Cost to stand up | Riskiest assumption | Recommended next step |
|----------|--------|------|-----------------------|------------------|---------------------|-----------------------|
| P0 | Pipeline golden (existing) | Golden | No (fixtures) | Done | None | Keep as hard gate |
| P1 | NIRF / NAAC proxy ranking | Golden-proxy | Yes (public PDFs/CSVs) | Low | That institution rank is a useful signal for MLT programme choice | Ingest 2024–2025 NIRF Medical + Overall + NAAC grades into `data/truth/`; add top-k / Spearman metric |
| P1 | Self-drift / shortlist snapshots | No-golden | No | Low | That we store versioned shortlists | Add snapshot writer + diff report to `fmd eval` |
| P2 | Verdict-fast canary (5–10 profiles) | Verdict | Minimal (reuse ingested claims + NIRF) | Medium | That a research agent can produce auditable, non-hallucinated verdicts | Write 5 profile fixtures + agent prompt + human audit protocol; run once |
| P3 | Cross-validation / leave-N-out stability | No-golden | No | Low | Internal consistency ≈ usefulness | Implement after P1 |
| P4 | Expert panel (small) | No-golden | Human experts | High | Panel bias / availability | Defer until product has a stable shortlist surface |
| Later | Survey / opt-in outcomes | No-golden (acquisition) | Yes (new data) | High | Response rate + legal | “Deals study”: design instrument + DPDP compliance path |
| Later | LinkedIn / CV role-model | Golden-sparse | Yes (ToS risk) | High | Selection bias + privacy | Legal review first; do not build yet |
| Later | Time-asymmetric predictive ladder | Hybrid | Historical + future outcomes | High latency | That outcomes become observable | Start logging historical shortlists now |

## Cost & coverage honesty

- **P1 streams** can be built with existing public data and the current pipeline; coverage will still be incomplete for pure paramedical colleges that never appear in NIRF Medical.  
- **Verdict stream** is the only design that can say “the product is wrong for this profile because a better programme exists with evidence X”. It is also the most expensive per profile and the most vulnerable to circularity.  
- No stream currently gives high-confidence programme-level employment outcomes for Indian B.Sc. MLT. Any dashboard that implies otherwise is over-claiming.

## Riskiest assumptions to verify before larger commitment

1. **NIRF/NAAC utility:** Does rank-ordering by NIRF Medical score actually correlate with anything students or employers care about for MLT programmes? Spot-check 10–20 programmes with known placement or registration signals.  
2. **Verdict agent reliability:** On a 5-profile pilot, what is inter-rater agreement between agent and a human researcher? If <70 % on “correct/incorrect”, the stream needs better tooling or stays advisory.  
3. **Data acquisition path for true outcomes:** Is there any realistic path (government MoU, hospital consortium, opt-in alumni) to programme-level first-job data within 2–3 years? If not, the product must stay in “proxy + coverage honesty” mode indefinitely.

## Alignment with B2 / HANDOVER

World-data (NIRF, NAAC, seat matrices, public placement PDFs) before proprietary or survey data. Verdict-style eval is the bridge that lets us claim product quality without waiting for mature outcome microdata.

## Recommended build order (concrete)

1. `data/truth/nirf/` + `data/truth/naac/` loaders and a `--stream nirf-proxy` metric.  
2. Versioned shortlist snapshots + `--stream self-drift`.  
3. 5-profile verdict-fast pilot with full audit artefacts.  
4. Only then consider expanding the verdict set or investing in survey design.