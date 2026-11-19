# FMD-CLOUD-017 · programme-materialisation · wave w8

~~~~
You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: <fill with current main from the operator>
BRANCH: feature/fmd-w8-programme-materialisation-<base7>-<UTC>-01
Do not push, open PRs, merge, or use secrets. Your chat reply is NOT read — the .md and .json are the report.

READ FIRST:
- AGENTS.md
- docs/product-development-cycle/04-phase-3-implementation-brief.md §6, §11
- docs/product-development-cycle/11-inference-runbook.md
- config/predicates.yaml (seat/curriculum/fee predicates and their descriptions)
- fmd/ingest/** (programme materialisation stage)
- fmd/db/repositories.py (Programmes repo)
- fmd/score/** (what a programme needs to be scoreable)

PROBLEM (measured): the corpus now has 61 verified claims — 28 accreditation, 14 seats, 7 fee,
7 curriculum, 5 placement — but only **2 programmes** exist in the DB, and both are `partial` at best.
Programme materialisation only runs inside the crawl/ingest path, so claims discovered later (or by
re-extraction after prompt/scope fixes) never become programme rows. The result: the tool cannot
rank anything despite having seat data.

SCOPE — FILES YOU OWN:
- fmd/ingest/** (materialisation logic; extract a reusable function)
- new fmd/cli.py subcommand `fmd programmes` (materialise | list | inspect)
- fmd/db/repositories.py (Programme queries only; migration allowed as 0009 if a column is truly needed)
- tests/test_programmes*.py, tests/fixtures/programmes/**
- docs/product-development-cycle/11-inference-runbook.md

FORBIDDEN: fmd/extract/** fmd/verify/** fmd/acquire/** fmd/fetch/** fmd/plan/** fmd/score/** apps/** infra/** .env

ACCEPTANCE
1. `fmd programmes materialise [--force] [--dry-run]` builds/updates programme rows from **existing
   verified claims**, without crawling or extracting:
   - one programme per (institution, normalised programme name); `field_sources` maps every field to
     the claim id that produced it;
   - fields: programme name, credential level, sanctioned intake (`seats.intake`), aggregate
     (`seats.total` — never used as a per-programme intake), duration (`seats.duration_months`), internship
     months (from curriculum claims, months only — no semesters→months derivation);
   - `NULL` where no claim exists; no invented values; no derivation.
2. **Idempotent and resumable**: rerunning creates no duplicates and updates only fields whose source
   claim changed (track claim id per field). `--dry-run` writes nothing and prints the plan.
3. `fmd programmes list` prints id, institution, programme, seats (with source claim id), evidence
   completeness (how many of the 6 scoring components are backed by at least one claim), and
   `needs_review` reasons.
4. `fmd programmes inspect <id>` shows every field, its claim id, quote, and page — a reviewer can
   audit a programme end to end.
5. **Guard against the seats.total trap**: aggregate totals must never populate a per-programme intake
   field. Add a test with a document that has both `seats.intake=20` (row) and `seats.total=195`
   (aggregate) proving only 20 lands on the programme and 195 is stored as an institution-level total
   (or discarded with a reason).
6. Tests offline (DB-gated skip cleanly): materialise from a claim set; idempotent rerun; changed claim
   updates field_sources; missing fields stay NULL; aggregate vs row separation; dry-run writes
   nothing; inspect output cites claims; ER-normalised name grouping (no duplicates for B.Sc. MLT vs
   B.Sc. Medical Laboratory Technology when the alias is confirmed — otherwise separate with
   `needs_review`).
7. Docs updated in the same patch.

CONSTRAINTS: no network; no model; no crawling; no invented values; scoped diff; do not change
scoring weights. If the DB schema needs a column, add migration 0009 and never edit 0001–0008.

RUN (all must pass):
cd packages/pipeline && uv sync --all-extras
uv run ruff check .
uv run pytest -q
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden     (provenance must be 1.0)

OUTPUT (root, UTC): fmd-patch-w8-programme-materialisation-<base7>-<UTC>-01.patch / .json / .md
fmd-patch-w8-programme-materialisation-<base7>-<UTC>-01-findings.json   (prefixed; never bare findings.json)

MANIFEST: format fmd-patch-v1, wave w8, workstream programme-materialisation, base_sha as given,
branch feature/<patch_id>, commit_message, summary, files_changed, tests_run,
evals {"result":"pass","notes":"provenance 1.0"}, known_gaps, created_at, agent.

findings.json must list: the field→predicate mapping, the aggregate-vs-row rule, what remains
underivable (e.g. duration when only semesters exist), and the expected programme count on the
operator's current DB (reasoned, not guessed).
~~~~
