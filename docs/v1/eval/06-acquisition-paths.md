# 06 — Programme-level outcome data: acquisition paths (honest research)

**Wave:** FMD-CLOUD-024 / v1 eval builder
**Author:** written by the build agent from the sources catalogued in `00-research-summary.md` plus
direct checks during this wave (NIRF pages fetched and ingested; NAAC site unreachable — see
`data/truth/naac/README.md`).
**Honesty first:** this document describes *paths*, not *facts in hand*. As of 2026-11-22 no
programme-level employment or registration dataset for Indian B.Sc. MLT is publicly available, and
none of the paths below has been opened. If a path stalls, the product stays in "proxy + coverage
honesty" mode (doc 02) — there is no fallback that manufactures outcomes.

## 1. Who could hold programme-level outcome data, and what they can actually share

| Holder | What they have | Realistic share | Main blockers |
|---|---|---|---|
| **Teaching hospitals / lab departments** (hiring side) | Which institutions' graduates they hire, intern-to-offer conversion per campus | Aggregate counts per source institution, per year — never individual records | HR data is internal; no obligation to share; hospital legal teams default to no; the data is about *their* hiring, not the programme |
| **State health directorates / allied-health councils** (registration side) | Registered practitioners by qualification and (sometimes) institution of training | Aggregate registration counts per institution, periodic | States differ in what they record; many registers are paper-era; RTI requests get aggregates after months, if at all; no MLT-specific register in most states |
| **NCAHP / national commission bodies** | Programme recognition + curriculum, not outcomes | Nothing outcome-shaped | Mandate is education standards, not employment tracking |
| **Universities / colleges (examination cells, alumni cells)** | Placement reports, alumni traces | Self-reported placement stats (already mined by the pipeline — fragile, invariant 6) | Self-reported, gamed, internship≠employment; independent verification impossible without cohort lists, which they will not share |
| **Industry bodies / federations** (e.g. hospital associations, diagnostics chains) | Sector workforce surveys | Broad aggregates by region, not programme | Membership data; nothing at programme grain; survey waves irregular |
| **Assessment/employability vendors** (Wheebox-style) | Employability test results by institution | Licenced, paid, institution-level but not MLT-specific | Paid APIs (violates invariant 10 for v1); coverage of B.Sc. MLT test-takers is thin |
| **The graduates themselves** | First-job reality | The opt-in survey (doc 05) | Cold start, selection bias, needs the legal review first |

The one-line conclusion: **no single holder owns programme-level MLT outcome microdata.** Every real
path is an aggregation partnership built slowly, or the opt-in survey.

## 2. The three paths worth attempting, in order of realism

### Path A — opt-in graduate survey (highest realism, lowest data value per year)

- **What:** doc 05's instrument, run as a persistent low-volume survey behind the product.
- **What it yields:** programme × graduation-year self-reported field-employment and registration
  rates, with published response composition.
- **Prerequisites:** DPDP review of the instrument (the four open questions in doc 05 §6); a
  moderation policy; a published methodology page.
- **Timeline:** instrument approval 1–2 months; first analysable wave 12–18 months after launch;
  minimum-cell thresholds mean the first *evaluable* programme cells appear around month 18–24.
- **Cost:** mostly design/legal/moderation time; negligible infrastructure.
- **Risk:** selection bias caps what the numbers can claim; the mitigation is publishing composition
  and never ranking institutions on survey rates alone.

### Path B — state registration aggregates via RTI / routine disclosure (medium realism, slow)

- **What:** file RTI applications (or use routine-disclosure pages where they exist) with state
  health directorates / councils for *aggregate* counts of registered medical-laboratory practitioners
  by qualifying institution and year.
- **What it yields:** a slow, patchy, but **administrative** (non-self-reported) signal. Even
  institution-level registration counts — not programme-level — would beat anything else available.
- **Prerequisites:** per-state research on which body holds the register; an RTI template that asks
  for aggregates only (no personal data — this materially improves compliance); patience.
- **Timeline:** 3–9 months per state realistically, with refusals and partial answers expected; a
  3–5 state pilot is the right scope, not all states.
- **Cost:** low money, high persistence.
- **Risk:** most registers do not record the qualifying institution for MLT; several states' allied-
  health councils are themselves the bodies whose websites are unreachable (naac.gov.in-style
  flakiness is a known pattern — the `naac_results` source notes in `config/sources.yaml`).

### Path C — hospital-hiring aggregation partnerships (lowest realism, highest value if it lands)

- **What:** convince 2–3 teaching hospitals or a diagnostics chain to share *aggregate* hiring
  counts by source institution ("in 2026 we hired 14 MLT graduates: 6 from X, 5 from Y, 3 from Z"),
  under a one-page data-sharing note.
- **What it yields:** the closest thing to demand-side outcome ground truth that exists.
- **Prerequisites:** a real relationship (this is a founder/BD task, not an agent task); a written
  commitment that the data is aggregate-only, attributed to the hospital in the methodology page;
  no per-candidate records ever.
- **Timeline:** 6–12 months to a first signed note if the relationship exists; otherwise do not
  plan on it.
- **Cost:** relationship capital; light legal.
- **Risk:** selection (hospitals hire from institutions they already trust — the data measures their
  network, not programme quality); tiny N; revocable at any time.

## 3. What does NOT work (so nobody re-litigates it)

- **Scraping LinkedIn/CV corpora** for alumni outcomes: ToS + privacy + selection bias; the research
  wave already filed it as legal-review-first, do-not-build (doc 02, "Later").
- **College placement reports as ground truth:** already in the pipeline as *claims*, correctly
  labelled self-reported and fragile; promoting them to truth would violate invariant 6 in practice
  (internship vs employment) even when the numbers look clean.
- **Buying a commercial dataset:** nothing MLT-specific at programme grain is on the market in India,
  and paid APIs are excluded in v1 (invariant 10).
- **NIRF/NAAC as outcomes:** they measure institutional reputation, not graduate outcomes; this wave
  ingests them as *proxy* signals with explicit labels for exactly that reason.

## 4. Decision rules (so the paths stay honest)

1. Every path publishes its **status and last-checked date** on the methodology page; a stalled path
   says "stalled", not "in progress".
2. No outcome number enters the product or an eval report without: aggregate-only provenance, a named
   holder, a date, and a minimum-cell rule (doc 05 §4).
3. If Paths A–C are all stalled by month 24, the product's answer to "outcomes" remains what it is
   today: **coverage honesty, UNKNOWN intervals, and the audited verdict stream** — and every
   surface says so.
