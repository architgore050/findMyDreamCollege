# FMD-CLOUD-022-eval-strategy-research · research a product-eval design for v1 · wave w11

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege). This wave is
**research + design only. No code, no execution, no PRs.** Your job is to produce:

1. A catalog of "golden-dataset" evals (external ground truth about unis) — where they exist,
   and which are usable here.
2. A catalog of **evals that do NOT need a golden dataset at all** — novel product-quality evals
   where a specialized research agent, done per profile/use case, produces a verdict of the form
   "the product recommends the right thing for this profile" (e.g., "the best uni for B.Sc. MLT in
   State X in this profile," or "the recommendation is likely wrong" with why). This is the core
   of what we need — product evals, not pipeline evals.
3. Integration sketches (where data lives, scoring rule, when it runs), but as *design sketches*.

REPO: https://github.com/architgore050/findMyDreamCollege.git
Base SHA: 66702b20b603bc36babbca76be359ae3a3a8209e. Do not push. Chat reply is NOT read —
the deliverables are the files this wave asks for.

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

YOUR JOB, EXACTLY
- Research "how do evals work when there is no golden dataset?" — this is the main deliverable.
  Examples to investigate and design around: cross-validation (leave-N-out), expert judgment
  benchmarks (AHP, NAAC scores as evaluators of quality), "wisdom of the crowd" evals (LinkedIn,
  resumes, course platforms), "role-model benchmarks" (the best students per profile, replayed),
  self-eval/drift over time (what does it mean for the product to be right when the world changes),
  time-asymmetric evals (an eval on N year-old data against profile year N+k, like a ladder of
  rolling "did we get it right in the N years since?" questions).
- Design "verdict evals": evals where the agent conducts *specialized research* for each profile
  (e.g., profile: "B.Sc. MLT, plus 10th-grade intelligence, state X, year 2027") and produces a
  decision: (a) the product is right ("these N programmes genuinely match", with evidence from the
  dataset if any and from research otherwise), or (b) it's wrong ("a genuinely better uni exists —
  here it is, with what it does differently"). The point: the product evals a *verdict product* —
  so the eval productizes the eval.
- Design fast and slow eval kinds. Fast = data-driven, repeatable, no new acquisition. Slow =
  acquiring things we can't get yet (outcome data via consultancies, proprietary student data,
  survey programs with opt-ins).
- For every eval idea: data shape, scoring rule, honest coverage/gaps (sparse data, missing values,
  intake-year mismatch, breadth of programme names), cost, time-to-run.
- Feasibility/risk notes: circular-data traps, privacy, leapfrog risk (changing model over time),
  staleness windows, how "maturity" is observed over time.

SCOPE OF YOUR RESEARCH (be exhaustive; use the web tool)
A. **Where we can find ground truth and how mature it is.**
   - Indian: NIRF (rank lists, methodology), MoHFW statistics, MoSES, AICTE (accred.), NAAC scores
     (colleges, district, profiles) — methodology?
   - Outcome/employment data: NSS (NARIVS), MoHFW (DIET), LinkedIn (salaries/ LinkedIn 'Outlook' — private),
     newspaper HR surveys (across years — archival!),shiksha, careerc 364 with historical data?
   - Web + public: KHARCH of hiring events? "Batching"? Yes! Corporate recruitment data over years.
   - Existing eval precedent papers on Indian uni evals (search: "Indian university ranking methodology",
     "NAAC score study", "Employment generation Indian college papers").
   - US/int'l: College Scorecard (NCES), QILT, THE, NCHEpie — NCES IPEDS methods, earnings vs graduates.
   - Maturity: how do products claim maturity? (years of data, method papers, statistical-power caveats).
B. **Design doc: "evals that don't need golden datasets."**
   - Survey-driven: student-perceived outcomes as eval labels (opt-in, waves of students).
   - NSS/MICRO employment status as eval labels for "where did our programme's students end up?"
   - LinkedIn/CVs/SSDR studies as expert benchmarks.
   - Cross-year calibration: v1 evals (2026 data), v2 evals (2028), etc.
   - Behavior-oriented eval: what does a "success" mean — into median BSc.MLT of year Y+2? into
     recognized programme? into job? ("placement outcomes" as the gold: placement_report claims).
C. **Design doc: verdict-style evals.** — the "what would a research agent do for one profile" design:
   - What "research" means per profile (deep dive on 1-3 unis with the most evidence, some with no evidence
     — mark as honest negatives).
   - How to produce a verdict with the product: "correct (X unis match), partially (Y unis match but Z
     underperform), incorrect (A's top rec is B, and B is listed with...)" — careful to not let the model
     invent facts; define an eval "product is correct iff for all profiles in a test set, the rec is in the
     top-3 of a reference ranking built from the dataset used by the existing `fmd eval`?).
D. **Harness sketch (integration plan, but as sketch):**
   - How new truth data would live: new SQL tables, `data/golden/` layout, fixture format, `--fixture-path`
     contract, versions (truth_version, scope, as_of, sha256 of the data).
   - How the two eval kinds hook into `fmd eval` (CLI flags like `--eval-kind`, `--stream`, golden vs
     proprietary streams).
   - How to run them: cron/scheduler, thresholds, drift reports (drift vs previous stream run).

DELIVERABLES (files in the repo, in the branch, commit + push to main):
1. `docs/v1/eval/00-research-summary.md` — findings: each external source's URL, maturity, licensing,
   usability for our product question.
2. `docs/v1/eval/01-eval-designs.md` — designs: (a) golden-dataset evals (the full taxonomy of current
   kinds — dataset, score rule, honest gaps); (b) evals without golden datasets (survey/crowd/role-model/
   self-eval kinds, as above, with data shapes + score rules); (c) verdict-style eval (the "research agent
   per profile" design — what it does, how the verdict is produced and how to audit it); (d) integration
   sketch (how streams plug into fmd eval).
3. `docs/v1/eval/02-findings-viable.md`? — or fold into 01. Keep: "viable options summary — which streams
   we'd build first, cost, and the riskiest assumption to verify before committing."
4. A proposal prompt? No.

Notes:
- **The four invariants (AGENTS.md) are hard bounds on your evals, not just on the pipeline:**
  1. No user-specific logic (no applicant marks/budget in evals).
  2. Numbers computed, never model-emitted — for evals: "the score we report is computed by the
     harness, not by the eval agent's claims" — the model is allowed to *verdict*, not *evidence*.
  3. UNKNOWN is a value — the eval has to report when it doesn't know (coverage gaps) instead of
     defaulting to 0/50.
  4. Trust in the eval source itself (golden datasets need provenance: URL, fetched_at, sha256).
  5. The B2/D19 guidance from `docs/v1/HANDOVER` — evals for an Outcome-focused product benefit most
     from world-data evals before proprietary data; keep the B2/HANDOVER deferred-items vision aligned.
- **Honesty:** say where an external dataset can't be used (e.g., proprietary, small N, US-only).
   Say where proposed eval costs are wishes, not estimates — it should say "this needs a deals study:
   how would we get data X?"
- Formats: YAML/MD, follow repo conventions.

RUN (all must pass; note in the report if the DB is unavailable, but it was available in CI for w9):
  - N/A: research + docs; but confirm offline tests green on main first (uv sync, ruff, pytest).
  - `uv run ruff check .`
  - `uv run pytest -q` (offline)
  - Doc: `ls docs/v1/eval/`; verify the new docs render (text only).
  - Also confirm `git rev-parse HEAD` == the recorded base SHA (66702b2).

OUTPUT (repo root, UTC timestamp):
  fmd-patch-w11-eval-strategy-<UTC>-01.patch
  fmd-patch-w11-eval-strategy-<UTC>-01.json
  fmd-patch-w11-eval-strategy-<UTC>-01.md
  fmd-patch-w11-eval-strategy-<UTC>-01-findings.json   (prefixed; never bare findings.json)

Patch: `git diff --binary --full-index --find-renames <base>...HEAD > <patch>`
MANIFEST: format fmd-patch-v1, wave w11, workstream eval-strategy, base_sha as above,
  branch feature/<patch_id>, commit_message, summary, files_changed, tests_run (assert >= green
  earlier results), evals {"result":"pass","notes":"see docs/v1/eval/01-eval-designs.md"},
  known_gaps, created_at, agent.
findings.json must state: the full eval catalog (golden-dataset evals + no-golden-dataset evals +
  verdict-style evals), the honest blockers for each (data availability, size, licensing, privacy),
  the integration sketch summary, and what we should build first.
~~~~
