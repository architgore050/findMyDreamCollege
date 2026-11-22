# FMD-CLOUD-022-eval-strategy-research · research a product-eval design for v1 · wave w11

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege). This wave is
RESEARCH + PROPOSAL ONLY. You are not implementing anything. Do not edit code. Your job is to
produce a well-evidenced answer to: **"how should this product be evaluated against external
ground truth — and how do we get external ground truth that this pipeline cannot produce?"**

Base SHA: cdae03f71a
Do not push. Chat reply is NOT read — the deliverables are the files this wave asks for.

READ FIRST
- AGENTS.md
- docs/product-development-cycle/04-phase-3-implementation-brief.md §1–§5, §11 (what v1 claims to do; the two DoD statements)
- docs/product-development-cycle/16-acceptance-checklist.md (what "done" already means — know the boundary of what's missing)
- docs/product-development-cycle/08-eval-methodology.md — the existing eval world in this repo (golden set, provenance, drift). You are extending it.
- docs/product-development-cycle/02-phase-2-scoring-criteria-and-existing-solutions.md — phase-2 research on what other products did. This is your starting map of "already solved?" territory — quote it where relevant.
- docs/product-development-cycle/05-gaps-and-full-solution-backlog.md (any D-items about evals, data sources, deployment, public data)
- config/sources.yaml, config/predicates.yaml (what evidence types we already mine)

THE PRODUCT QUESTION WE NEED EVAL STREAMS FOR

Today `fmd eval` = golden-set extraction accuracy + provenance coverage + drift. That proves the
pipeline is correct, not that the product is useful. The product question: *given a profile
({nationality, target_intake_year, target_field}), does the ranked shortlist the product produces
actually match where students go?* An eval stream answers that against ground truth someone else
measured — outcomes of students, not our claims. The product we want is "a genuinely mature
product" — one whose recommendations have been validated over years. Research: how do real
outcome datasets become "maturity"? (Probit/survey cohorts, longitudinal placement data, college
outcome benchmarks.)

SCOPE OF YOUR RESEARCH (be exhaustive; use the web tool)
A. **Existing external ground-truth datasets for Indian university outcome evaluation**
   1. Indian regulators and agencies: NIRF (rank lists, institutional accreditation, maybe outcome
      data?); NCWEB (NIRF); MoHFW statistics; Ministry of Skills & Entrepreneurship; AICTE (accred.);
      NAAC scores (colleges, district, profiles) — how does NAAC's methodology work, what data do they
      measure vs rank?
   2. Outcome/employment datasets: NSS/NARIVS? (NSS 2M 'Activities & Employment Status' — but it's an
      unknown-era sample), EDU-GGP (MHRD DIET?), DIET2/ICAR for diabetic safety — no; focus: employment
      status of graduates per degree per state (there are reports, e.g., NSS surveys by NSSIC). Also:
      UIDAI (Skillsinfra?), NSF GIFS? (no), THE CAREER, LinkedIn 'Outlook Report'? (private), Google
      Careers? (no), employment reports from seekers; ShastraOnline / Sulekha / iSpray — fringe.
   3. Rankings that legitimately rank programmes: NIRF (billing: SSR/SCS methodologies), Careers360?
      or 'IKEA'? (no) — better: Shiksha, Careers360 India (private, but has per-programme data + user
      reviews — DOES THIS COUNT AS "PRIVATE FEEDBACK"?), CollegeDunia (private, similar), Shiksha data
      feeds (API?), TheWorld University Rankings? (billing), Employment scores: newspaper HR
      survey by Hindustan Times / DNA / The Hindu / Economic Times — historically published (across
      years) — check archival access!
   4. US/int'l outcome data we might use as a template for maturity signals: College Scorecard (NCES —
      debt, grad rates), QILT (quondam of OECD — quality/job stability), NCCIIC? no; THE StEP —EEDb —
      US methodologies: NCHEpie (niche?), ' bom ' GE - es? NO — be concrete and honest: dig into
      NCES, QILT, HEA 'Employability measures'.
   5. Maturity: how do products claim maturity (sources citing years of data, methodology papers,
      statistical-power caveats)? What would we need (a) to produce — N samples, time series per
      programme; (b) a "truth source" — self-eval of outcome claims vs exit surveys (self-consistency).
B. **The two eval streams we need to propose**
   - **Stream A (fast, data-driven).** From public datasets: design a benchmark — e.g., "given profile P
     and location S, for two candidate programmes A and B (both accepted/intersected), which does the
     dataset say students actually complete/later-employ?" — i.e., a *consistency score* between our
     ranking and external outcome rankings over the same dataset. Requirements: (i) be honest: external
     data likely has gaps — study gaps, say how to handle (sparse, missing-not-zero, date mismatch with
     intake year, scope of programme breadth: compare full names not B.Sc.)
     (ii) avoid circularity: split methods — use public data for eval of rankings, but still source claims
     from our own pipeline? Hmm — think about what counts as ground truth vs what we generate.
   - **Stream B (slow, proprietary data acquisition).** Research how to acquire real outcome data:
     partnerships with consultancies (names, fee ranges, typical engagement: roadshows, white papers,
     past services), NO-CAP — interview older batches of students via these sources?
     Governance/ethics of student-outcome feedback (privacy, NDAs, GDPR, India Data Protection Authority
     Bill 2024, encryption).
   C. **Harness-ready design for both.** What data schema do they feed our pipeline? How to score and
      report (precision@k, recall, interval overlap, "genuinely correct" = interval-overlap rule, not
      point estimates). How to version and freeze ground-truth snapshots (immutable, time-stamped,
      retractable).
D. **An honest feasibility/risk report.** Circular-data traps (learning on what we extract vs using it
    as truth), time-to-data, privacy, dark-funnel leakage of user IDs, A/B testing, the maturity of the
    public data. Where the product is currently guaranteed correct by construction (registry-driven) —
    how to validate there too.

DELIVERABLES (files in the repo, in the branch, commit + push to main):
1. `docs/v1/eval/00-research-summary.md` — your findings, each source's URL, maturity, licensing, whether
   it's usable for our product question (honest: most are not fit for purpose, with reasons).
2. `docs/v1/eval/01-stream-a-fast-data-driven.md` — Stream A design: source(s), ground-truth table design,
   the scoring rule (what "correct" means for our product), data availability analysis (years of data,
   sample sizes), how it behaves on our current corpus (expected), data licensing + acquisition (read-only
   public sources only; robots; rate limits; caching; costs).
3. `docs/v1/eval/02-stream-b-slow-proprietary.md` — Stream B design: acquisition path to private data
   (how to make deals, what to ask, alternatives to private data: self-collected via LinkedIn ("Google
   Products"? "Mission"?), opt-in graduate surveys we could run ourselves — wait: we can only claim data we
   legitimately acquire; the prompt should say: PROPOSE acquisition paths for third-party proprietary
   outcome data (consultancies, AGROM?..), PLUS note that self-collected student feedback is impossible
   under privacy norms without opt-in machinery — propose opt-in web survey framework + data minimization
   + "harbor" for opt-out, and note FMD invariant 14 (no user-specific logic — these are scalars, OK)).
4. `docs/v1/eval/03-integration-plan.md` — how the two streams hook into `fmd eval` (the CLI contract:
   where truth data lives — new SQL tables, data/golden/ layout, `--fixture-path` contract), how scores are
   compared, how A/B results are reported (small YAML-driven report with pass/fail thresholds), how
   stream-B mode is flagged (`--mode data-driven|proprietary`), and how maturity is observed over time
   (time series of score by stream).
5. `prompts/cloud/FMD-CLOUD-022-eval-strategy.md`? No — that's this patch.

Notes:
- NO — "golder truth data of what university is actually the best option" — the agent should research
  sources of "best options" literature: the web has things like NIRF PCI methodologies, ICAR (agri),
  NAAC scores, MoHFW staffing in graduation (KVPY?), Google Search? (public).
- Honesty and honesty is a MUST (AGENTS.md): an external dataset that is scraped-ad filtered or small
  says so. Proposing no-SQL? No — extend Postgres: new tables `eval_truth` (referral hashes), schema
  versioning; note that data-driven truth is data, so it should be as well-sourced as other claims
  (URL, fetched_at, sha256).

Formats: follow the repo's existing conventions: Data under data/ (parquet via duckdb? existing `data/golden/` has JSON), config-driven YAML, YAML/MD docs.

RUN (all must pass; note in the report if the DB is unavailable, but it was available in CI for w9):
  - N/A: research + docs; but confirm offline tests green on main first (uv sync, ruff, pytest).
  - `uv run ruff check .`
  - `uv run pytest -q` (offline)
  - Doc: `ls docs/v1/eval/`; verify the new docs render (text only).
  - Also confirm `git rev-parse HEAD` == the recorded base SHA.

OUTPUT (repo root, UTC timestamp):
  fmd-patch-w11-eval-strategy-<UTC>-01.patch
  fmd-patch-w11-eval-strategy-<UTC>-01.json
  fmd-patch-w11-eval-strategy-<UTC>-01.md
  fmd-patch-w11-eval-strategy-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: `git diff --binary --full-index --find-renames <base>...HEAD > <patch>`
MANIFEST: format fmd-patch-v1, wave w11, workstream eval-strategy, base_sha as above,
  branch feature/<patch_id>, commit_message, summary, files_changed, tests_run (assert >= green
  earlier results), evals {"result":"pass","notes":"see docs/v1/eval/00-research-summary.md"},
  known_gaps, created_at, agent.
findings.json must state: every external dataset considered and its blocker for Stream A or B;
  the full acquisition-path options for Stream B (consultancies, surveys, NHA/exams);
  the two streams as concretely designable (data shape, score rule, integration into fmd eval);
  which public datasets are directly usable today and which need NDAs/agreements.
~~~~

You (the operator) gave me three things this prompt must not contain:
1. The framing "mature product" — the agent should *define* maturity itself; your position is just that a real product needs evidence from the outside, not generated inside.
2. The "fast/slow" dichotomy — the agent should think from scratch about what makes evals fast vs slow, not absorb your categories.
3. The "we will implement both" — propose; don't design both fully; the user decides which to build after seeing the research.
