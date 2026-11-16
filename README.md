# findMyDreamCollege

A data-mining and scoring engine for bachelor's-level Medical Laboratory Technology / Medical Laboratory Science programmes. It inventories institutions and programmes from public registries, mines government PDFs and institutional disclosures, and produces a **sourced, filterable shortlist** for a user-supplied profile — where every number resolves to a verbatim quote, URL, and retrieval timestamp.

**v1 scope: India only, maximum depth.** Global coverage is deliberately deferred and tracked in the backlog.

## Why this exists

Study-abroad comparison tools rank universities on prestige and published fees. For allied-health degrees the decisive facts are different and mostly hidden:

- **Internships are not placements.** NCAHP's 2025 MLS curriculum makes a 1,800-hour internship a mandatory part of the degree, so counting it as employment inflates every "placement rate" in the sector.
- **Outcome data is sparse and gamed.** India publishes no programme-level BSc MLT placement or salary data. Where PDFs exist, denominators are often redefined after the fact.
- **Recognition is binary.** Since the AHP Registration Regulations 2026, practising requires registration; a degree from a non-recognised institute is a dead end, not a low score.
- **Affordability is a loan question, not a fee question.** The honest metric is debt service against the bottom-quartile salary, not tuition alone.

So this tool scores what it can prove, marks the rest `modelled` with intervals, and refuses to rank what it cannot support.

## Quick start

```bash
cp .env.example .env      # fill in model endpoints (or use --mock)
make up                   # db + api + web
make migrate
make seed
make crawl               # respects robots.txt, per-domain budgets, no paid APIs
make extract
make score
open http://localhost:3000
```

Everything runs locally in Docker. No paid scraping services, no hosted database.

## Documentation

| Doc | Contents |
|---|---|
| `docs/product-development-cycle/01-*` | User & problem research |
| `docs/product-development-cycle/02-*` | Scoring criteria + evaluation of existing solutions |
| `docs/product-development-cycle/03-*` | Original design brief (global vision) |
| `docs/product-development-cycle/04-*` | **Phase 3 implementation brief — authoritative build contract** |
| `docs/product-development-cycle/05-*` | Gaps and full-solution backlog |
| `AGENTS.md` | Repo conventions, invariants, how to add sources/countries |

## Design commitments

1. No user-specific logic. Profiles are runtime input, never code.
2. Numbers are computed; LLM output is evidence, never a score.
3. Binary filters (closed registration, travel-required exams, unrecognised institutes) are never blended into a weighted score.
4. `UNKNOWN` is a real value and propagates as an interval.
5. Every artefact is reviewable: raw snapshot, prompt version, model, extracted claim, quote span.
6. Politeness over throughput: robots.txt honoured, no CAPTCHA bypass, no login-walled scraping.

## Status

Pre-M0: documentation and configuration registry are in place; the pipeline, database schema, and web app are being built against `04-phase-3-implementation-brief.md`.
