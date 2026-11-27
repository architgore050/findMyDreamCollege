# AGENTS.md

Orientation for any agent (human or model) working in this repo. Read this before touching code.

## Agent working mode: orchestration by default

You are an orchestrator by default. You do not investigate or write code unless the user explicitly says to. Your job is to:

- turn a goal into a self-contained prompt for a one-shot cloud agent: it clones the repo, reads this file, implements the change, and emits a `.patch` + manifest + report + findings;
- run local subagents for work that can only happen on this machine — patch intake/merge, GPU or host measurements, DB inspection, applying a patch set;
- hold the invariants and acceptance criteria inside every prompt: exact base SHA, files in scope, forbidden paths, required tests, gates, and the artifacts to emit;
- verify what comes back (gates, quoted assertions, scoped diff) and report the verdict, not a retelling of the diff.

Investigate or code directly only when the user asks for it, or when one lookup is needed to write a correct prompt (a CLI flag, a file path). Do not "just fix it" in passing.

**Hard rule for cloud agents:** everything must be pushed to `main` locally **before** you tell a cloud agent to pull from `main`. Do not launch a cloud agent against a stale or divergent local `main` — refresh (`git fetch origin` and fast-forward local `main` to `origin/main`) and merge any local work first.

## Long-running commands

Any command that can exceed ~2 minutes (crawls, acquisition, model runs, full suites against the DB) follows this protocol:

1. Run it once, in the background (harness background execution or `nohup … &`). Never run it in the foreground and never poll or tail it while waiting — the harness notifies on completion.
2. Make the script self-contained: log to `/tmp/opencode/<task>/run.log` via
   `scripts/run_logged.sh <log> <command...>` (START + exact command + PID,
   30-second `ps` HEARTBEATs, END + exit code, sibling `.status` file), print a
   one-screen final summary as the last output, exit non-zero on failure.
3. Leave the machine as you found it: stop anything you started (`trap`/`finally`: model server, containers, temporary worktrees) and report free GPU memory at the end.
4. Persist progress per item where the pipeline supports it, so a kill loses only in-flight work.
5. Never run the same long task twice in parallel.

## What this is

A data-mining + scoring tool that inventories bachelor's-level Medical Laboratory Technology / Medical Laboratory Science programmes, mines public evidence, and produces a sourced, filterable shortlist for a user-supplied profile.

**v1 scope: India only, maximum depth.** Global coverage is tracked in `docs/product-development-cycle/05-gaps-and-full-solution-backlog.md`. Do not expand scope silently.

## Read in this order

1. `docs/product-development-cycle/04-phase-3-implementation-brief.md` — the build contract. Authoritative.
2. `docs/product-development-cycle/01..03-*.md` — why the product exists.
3. `docs/product-development-cycle/05-gaps-and-full-solution-backlog.md` — what is intentionally missing.
4. `config/` — the registry. If behaviour is not driven by config, it is probably wrong.
5. `docs/product-development-cycle/18-scrape-failures.md` — reading the scraping-failure log (what we lose to blocked/failed fetches).
6. `docs/product-development-cycle/14-deferred-items.md` — deferred/future items (e.g. the prompt-injection prevention pass, §E); consult when starting new work.
7. `docs/product-development-cycle/20-future-paid-web-services.md` — the paid-service OPPORTUNITY list (Firecrawl Cloud, Tavily, proxies, Alexandria, …): read it so you know what exists, but DO NOT implement any of it by default — invariant 10 keeps the code free/self-hosted; the implemented path is `19-free-web-data-stack.md`.

## Invariants (a PR that violates one is rejected)

1. **No user-specific logic.** No applicant's marks, budget, intake year, country, or family situation in code, defaults, seeds, or fixtures. Profiles are runtime input.
2. **Numbers are computed; prose is generated.** LLMs extract claims. They never emit scores.
3. **Every published number resolves to a span** — `claim → extraction → snapshot → URL + fetched_at + sha256`. Unresolvable ⇒ drop or mark `modelled`.
4. **Filters are binary; scores are weighted.** Closed registration, travel-required exams, unmet eligibility, unrecognised institutes are cutoffs, never penalties.
5. **`UNKNOWN` is a value**, never 0 and never 50. It propagates as an interval.
6. **Internships are not employment.** Enforced in schema and scoring.
7. **Registry-driven.** Countries, sources, weights, exams, fee rules ⇒ `config/*.yaml`.
8. **Append-only evidence, versioned scores.** Never mutate a claim; supersede it.
9. **Politeness beats throughput.** robots.txt is honoured, no auth/CAPTCHA bypass, per-registrable-domain rate limits, truthful UA with contact.
10. **No paid APIs in v1.** Open-source crawling only; model access goes through the provider-agnostic router.

## Layout

```
config/            countries, sources, weights, exams, models, cost rules
db/migrations/     plain SQL, numbered, forward-only
packages/pipeline/ Python 3.12 package `fmd` (+ tests beside modules)
apps/web/          Next.js app (TS, Tailwind, shadcn)
infra/             docker-compose, Dockerfiles
scripts/           seed, crawl, score, eval entrypoints
docs/              product-development-cycle/
```

Python: `uv` for env/deps, `ruff` for lint+format, `pytest` for tests, `pydantic` for all I/O boundaries.
TypeScript: `pnpm`, `eslint`, `prettier`, `zod` for runtime schemas.

## Common commands

```
make up          # postgres + pgvector + api + web
make migrate     # apply SQL migrations
make seed        # load config registry into DB
make crawl       # run the frontier (respects budgets)
make extract     # snapshots -> extractions -> claims
make score       # claims -> scores + insights
make eval        # golden set, provenance coverage, drift
make dev         # web app against local API
make test        # pytest + vitest, mock model provider, no network
```

`make test` must never hit the network or a real model. Use the `--mock` provider and committed fixtures.

## How to add a source

1. Add a `config/sources.yaml` entry: `id, name, base_url, kind, format, cadence_days, trust, adapter, enabled`.
2. Implement `SourceAdapter.fetch() -> Iterable[DocumentRef]` in `fmd/adapters/<family>.py`. No parsing in the fetch layer.
3. Add an extraction schema in `fmd/extract/schemas/<source>.py` with `value/unit/quote/char_start/char_end/confidence` on every field.
4. Add a fixture + a test asserting span resolution on the fixture.
5. Record `trust` honestly. Low-trust sources may inform `modelled` claims only.

## How to add a country (post-v1)

Read `05-gaps-…` §G1.6 first. Minimum: programme inventory source, regulator/recognition source, admissions calendar source, city CoL basket, currency, licensure gate, student-work rules. Country files are data, not branches.

## Data rules

- Claims: append-only; corrections insert + set `superseded_by`.
- Quotes are verbatim; spans must resolve in `extractions.text`.
- Rates require a denominator in the same claim's `qualifier_json`, or they are rejected.
- Money always carries `currency`, `period`, and `as_of`.
- Salaries: prefer median → Q1 → mean; mean-only cannot produce a point estimate.
- Every modelled value carries method + confidence + interval.

## Testing expectations

- Pipeline unit tests per module; golden-set tests for extraction on committed fixtures.
- `provenance_coverage` must stay at 100%: any published number whose span does not resolve is a failing test, not a warning.
- Score changes require a version bump (`scorer_version` / `weights_version`) and an eval report diff.

## Documentation expectations

- Prompt changes: bump `prompt_version`, keep the old file.
- Schema changes: new migration, never edit an applied one.
- Any deviation from the brief: note it in `05-gaps-…` with a reason.

## Definition of done for any task

`make test` green, `make eval` provenance coverage 100%, docs updated, and no new user-specific assumptions. If a task cannot meet that, say so instead of shipping it.
