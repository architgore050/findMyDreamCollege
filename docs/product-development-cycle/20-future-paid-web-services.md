# 20 — Future paid web services (opportunities YET TO BE TAPPED — DOC ONLY)

Status: written 2026-11-27 (FMD-CLOUD-027). **Nothing in this document is
implemented.** No SDK, API key, config flag, or code path in this repository
calls any service listed here. Invariant 10 ("no paid APIs in v1") governs the
codebase until an explicit product decision amends it; this document exists so
that decision, if it ever comes, starts from a written map instead of a
sales page.

Read-order note for agents: this file is an *opportunity list*, not a todo
list. Do not implement any of it by default; the free stack in
`19-free-web-data-stack.md` is the implemented path.

## 1. Firecrawl Cloud (Search / Scrape / Crawl)

- **What we would use:** managed search endpoint, high-volume scrape/crawl
  with anti-bot handling, structured extraction, PDF/Office parsing at scale.
- **Potential value:** removes the operator-host dependency for stealth
  fetching; local `FIRECRAWL_CLOUD=off` currently refuses cloud fallbacks, so
  blocked-domain coverage would jump without running our own proxy fleet.
- **Cost model / free-tier caveat:** credit-based (per page/action), with a
  small free tier; search and blocked-page fallback are the parts that go to
  Cloud by default today — which is exactly why the client fails closed.
- **Adoption trigger:** measured local failure rate on the scrape-failure log
  (`fmd failures`) exceeding what the MCP browser tier recovers, sustained
  crawl volume where politeness delays dominate wall-clock time, and an
  approved monthly budget.
- **Provenance/privacy risks:** pages fetched by a third party change what
  `documents.fetched_at` means (we record what the vendor saw); their cache
  (`maxAge`) can serve stale content; source IPs and footprints leave our
  control. Requires honest provenance fields (`via: firecrawl-cloud`) and an
  invariant-10 amendment.

## 2. Firecrawl Semantic Index / extraction add-ons

- **What we would use:** managed vector index over crawled pages, semantic
  queries instead of re-crawling, structured JSON extraction endpoints.
- **Potential value:** cheaper re-queries of the same institutional corpus;
  our pgvector/embedding stack stays local but is operator-operated.
- **Cost model:** index storage + query credits; free tier is small.
- **Adoption trigger:** repeated re-extraction passes over unchanged pages
  (measured via snapshot dedupe stats) and a cost comparison against the local
  stack.
- **Risks:** index drift vs. our append-only evidence model; another party
  holds our corpus-derived data.

## 3. Managed anti-bot / proxy providers (e.g. residential proxy fleets)

- **What we would use:** rotating residential/mobile egress for the ~5–10 % of
  `.ac.in`/council domains that block datacenter IPs outright.
- **Potential value:** converts `all_tiers_blocked` quarantines into fetches.
- **Cost model:** per-GB, typically the most expensive line item at scale.
- **Adoption trigger:** quarantine counts per institution from the
  scrape-failure log reaching a level where manual or on-site collection is
  costlier; approved budget.
- **Risks:** politeness optics (invariant 9: robots is honoured; a proxy does
  not change that, but it changes how we appear); truthful UA/contact stays
  mandatory; legal review per jurisdiction (India-first).

## 4. Alexandria (providers / datasets)

- **What we would use:** curated structured datasets (education registries,
  rankings) instead of scraping primaries.
- **Potential value:** fast bootstrapping of the registry; cross-checks for
  the verifier.
- **Cost model:** cloud-only, subscription/credit based. Alexandria is
  Cloud-only and is never called by the current code.
- **Adoption trigger:** a concrete gap where no free primary/regulatory source
  exists (see `05-gaps-…` §G1) and the dataset's license allows our use.
- **Risks:** provenance becomes second-hand by definition — such data can
  only ever inform `modelled` claims, never direct evidence.

## 5. Interact / Agent / Monitor (managed runtimes)

- **What we would use:** managed headless-browser sessions, scheduled
  monitoring of exam-notice pages, agentized navigation.
- **Potential value:** exam windows and notification scraping without keeping
  a warm host.
- **Cost model:** per-session/per-monitor subscriptions.
- **Adoption trigger:** cadence-driven work (exam notifications) outgrowing
  the single-host scheduler; an operator-paid reliability SLO.
- **Risks:** least aligned with the self-hosted design; vendor-side
  fingerprinting/proxying is indistinguishable from evasion if misused —
  needs the same politeness audit as #3.

## 6. Enterprise controls (SOC2-style vendor governance)

- **What we would use:** vendor risk review, DPAs, audit logs for whichever
  paid vendor is eventually adopted.
- **Potential value:** makes #1–#5 adoptable inside an institution/investor
  context rather than a personal project.
- **Cost model:** direct fees plus process cost.
- **Adoption trigger:** the first contract that requires handling real
  applicant data (v1 stores none — profiles are runtime input).

## 7. Tavily search

- **What we would use:** an LLM-oriented search API as an alternative to
  self-hosted SearXNG.
- **Potential value:** higher result quality for niche queries
  ("<college> BSc MLT fee structure 2026"), no SearXNG engine rate limits.
- **Cost model:** credit-based with a monthly free tier; keys are secrets.
- **Adoption trigger:** SearXNG provider_errors dominating the verdict-live
  journal after tuning, and a measured quality delta on the canary profiles.
- **Risks:** results shape (title/url/snippet) overlaps our lead model, but
  provenance and rate/budget accounting would need the same harness-side
  caps as the free providers; another external dependency for a step that is
  discovery-only today.

## 8. Decision rule (unchanged)

For every row above: measure the free stack first (`fmd failures`,
`provider_errors`, quarantine counts), write the comparison next to the
relevant eval report, and only then take invariant-10 to a product decision.
The default answer stays "the free stack in doc 19".
