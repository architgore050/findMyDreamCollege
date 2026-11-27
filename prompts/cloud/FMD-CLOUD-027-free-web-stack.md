# FMD-CLOUD-027-free-web-stack · invisible MCP + Firecrawl Local + SearXNG · wave w15

~~~~
You are the ORCHESTRATOR cloud coding agent for this wave. You may spawn subagents
for parallel pieces; everything lands on one branch and one PR.

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: f9991d6f67dd1c29a2bb713a004b40031463f7c6
All output (code, comments, docs, commit messages, PR text, report) — English-only.

NON-NEGOTIABLE SCOPE: FREE / SELF-HOSTED ONLY
- NO Tavily, Firecrawl Cloud, Alexandria, Agent, Interact, Monitor, paid proxies, or any
  paid API. Do not add their SDKs or API keys. Invariant #10 stays intact.
- Firecrawl Local MUST run with `FIRECRAWL_CLOUD=off`. The client must fail closed if a
  response says `source=firecrawl-cloud` (no silent cloud hand-off).
- Paid/managed future opportunities go in a DOC only (Task D), never in code.

BRANCHING
1. `git switch -c v1/free-web-stack` from main at the base SHA; verify `git rev-parse HEAD`.
2. `git push origin v1/free-web-stack` immediately.
3. Commit + push after every logical step. Subagents branch off this branch and PR into it.
4. End: PR `v1/free-web-stack` → `main`.

READ FIRST
- AGENTS.md (invariants; esp. #9 politeness, #10 no paid APIs)
- packages/pipeline/fmd/fetch/ladder.py, types.py, failure_log.py
- packages/pipeline/fmd/discover/searxng.py + discover/base.py (REUSE this client; do not
  write a second SearXNG client)
- packages/pipeline/fmd/eval/verdict_live.py (SYSTEM_PROMPT, ResearchAction, loop, ledger)
- packages/pipeline/fmd/eval/verdict.py (VerdictObject, audit_verdict)
- tests/test_fetch_ladder.py + test_eval_verdict_live.py + tests/test_discover_searxng.py
- infra/docker-compose.yml `searxng` service + infra/searxng/settings.yml (JSON format enabled)
- pyproject.toml + uv.lock (patchright is the current browser dependency)
- .env.example (SEARXNG_PORT already exists)
- docs/v1/eval/07-verdict-live-runbook.md, docs/product-development-cycle/18-scrape-failures.md
- reports/eval-verdict-live-20261126T150000Z.json (diagnosis: guessed URLs → 404, Google →
  robots, citations sparse, all verdicts unknown)

================================================================================
TASK A — replace PatchrightFetcher with InvisiblePlaywrightMCPFetcher
================================================================================
Exact server (verified upstream):
- GitHub https://github.com/oss-invisible/playwright_mcp ; PyPI `invisible-playwright-mcp`;
  MIT; Python >=3.11; start over stdio: `uvx invisible-playwright-mcp`.
- Tools: `browser_open`, `browser_navigate(url, wait_until)`,
  `browser_read_html(mode="form"|"text"|"full")`, `browser_read_text(selector="body")`,
  `browser_status`, `browser_close`, `browser_list`. `browser_navigate` returns HTTP status +
  final URL (structuredContent; text fallback). Server is stateful: one page.
- Env: STEALTHFOX_PROXY, STEALTHFOX_NO_PROXY, STEALTHFOX_SEED, STEALTHFOX_PROFILE_DIR,
  STEALTHFOX_BINARY, STEALTHFOX_HEADLESS, INVISIBLE_MCP_HOME.
- Contract: https://raw.githubusercontent.com/oss-invisible/playwright_mcp/main/docs/mcp-server.md

1. Remove PatchrightFetcher and its in-process Chromium helpers from `fmd/fetch/ladder.py`.
   Remove `patchright` from pyproject extras; regenerate uv.lock.
2. New `fmd/fetch/mcp_browser.py`:
   - `InvisiblePlaywrightMCPFetcher` implements `TierFetcher`.
   - Official Python `mcp` SDK (`mcp>=1.19,<2`; add to `all` + `crawl` extras);
     `mcp.client.stdio.stdio_client` + `ClientSession` + `StdioServerParameters`.
   - Lazy startup; `close()` terminates client/server.
   - `fetch(request)`: non-GET → UnsupportedReplay; ensure `browser_open` once;
     `browser_navigate(url, wait_until="domcontentloaded")`; parse status/final URL from
     structuredContent (text fallback); body = UTF-8 bytes of `browser_read_html(mode="full")`;
     `fetcher_tier="invisible_playwright_mcp"`.
   - `browser_open` `opened=false` / `cause=downloading|launch_failed|missing_libraries|...`
     → honest FetchResult(status=0, error="mcp_browser_<cause>") or TierUnavailable for a
     missing command/SDK. Serialize MCP calls with a threading.Lock.
   - Config: `FMD_BROWSER_MCP_ENABLED` (default "1"; "0" → TierUnavailable),
     `FMD_BROWSER_MCP_COMMAND` (default "uvx"),
     `FMD_BROWSER_MCP_ARGS` (default "invisible-playwright-mcp"); pass STEALTHFOX_* /
     INVISIBLE_MCP_HOME env unchanged.
3. `default_fetchers()` order: HttpxFetcher → CurlCffiFetcher → InvisiblePlaywrightMCPFetcher
   → FirecrawlLocalFetcher (Task B). Add `FetchLadder.close()` and wire the orchestrator
   lifecycle where appropriate (do not break injected test fetchers).
4. Politeness unchanged: robots decision precedes browser calls; no auth/CAPTCHA bypass;
   truthful UA where possible; failures still flow through failure_log.py.

Tests (offline; no real engine):
- Inject a fake MCP session: exact call order open → navigate → read_html; FetchResult shape.
- 404/redirect/status parsing; non-GET → UnsupportedReplay; disabled/missing command →
  TierUnavailable; lock serializes; close terminates.
- A tiny stdio JSON-RPC fake-server fixture (initialize, tools/list, tools/call) proves the
  spawn/wire path without downloading an engine or touching the network.

================================================================================
TASK B — Firecrawl Local (free, local-only) client + fallback tier + site map
================================================================================
Verified upstream:
- Firecrawl Local = one Rust binary (`cargo install firecrawl-local`), serves Firecrawl v2
  API on localhost:3002; no DB/queue/container; scrape/crawl/map/batch/PDF+Office parsing
  run locally. Search and blocked-page fallback go to Cloud by default — MUST disable with
  `FIRECRAWL_CLOUD=off`.
- Docs: https://docs.firecrawl.dev/quickstarts/firecrawl-local and
  https://raw.githubusercontent.com/firecrawl/firecrawl/main/README.md
- Local supports markdown/html/rawHtml/links; JSON/summary skipped; browser actions ignored.
- Alexandria is Cloud-only and must never be called.

1. New `fmd/fetch/firecrawl_local.py` (httpx only; do NOT add the firecrawl SDK):
   - `FirecrawlLocalClient`: base URL from `FIRECRAWL_LOCAL_URL` (default
     http://127.0.0.1:3002), timeout from `FIRECRAWL_LOCAL_TIMEOUT_S` (default 60),
     `FIRECRAWL_LOCAL_ENABLED` (default "0" until operator starts the local server).
   - `scrape(url) -> FetchResult`: POST /v2/scrape with formats ["html","markdown"],
     `maxAge` configurable (default 0 = fresh; local only); parse `data.metadata.statusCode`
     + `sourceURL`; body = `data.html` (fallback markdown encoded UTF-8). If metadata/source
     says `firecrawl-cloud` OR `blocked=true` → fail closed with a typed error; never accept
     a cloud result.
   - `map(url, query=None, limit=50) -> list[MapLink]`: POST /v2/map; parse links
     [{url,title,description}]. Local only.
   - Typed errors: disabled, unavailable, cloud_handoff_forbidden, bad_response,
     http_status, transport_error. `close()`.
2. `FirecrawlLocalFetcher` implements TierFetcher around the client; non-GET →
   UnsupportedReplay; disabled/unreachable → TierUnavailable; failures → honest FetchResult
   so failure_log records them. Last local tier in `default_fetchers()`.
3. `scripts/firecrawl_local.sh start|stop|probe`:
   - check for `firecrawl-local`; if absent print `cargo install firecrawl-local` (do not
     auto-install during tests).
   - start with `FIRECRAWL_CLOUD=off firecrawl-local serve`, log/pid under
     /tmp/opencode/firecrawl-local; probe the local health endpoint (verify the exact route
     from the pinned docs/source; if no health route, document a local-only scrape probe).
   - stop kills only the PID it started. Print the no-cloud guarantee.
4. Tests: httpx MockTransport for scrape/map payloads + response parsing; cloud-sourced
   result rejected; disabled/unreachable → TierUnavailable; non-GET; FetchResult metadata;
   script syntax check. No real Firecrawl process/network in pytest.

================================================================================
TASK C — free discovery in verdict-live (SearXNG search + Firecrawl Local map)
================================================================================
Reuse `fmd/discover/searxng.py` (`SearxngClient.search(query) -> list[DocumentRef]`,
metadata title/engine/score). Compose already runs SearXNG on ${SEARXNG_PORT:-8888}:8080
and JSON format is enabled. No new SearXNG client.

1. `fmd/eval/verdict_live.py`:
   - Add `search_web` (query) and `map_site` (url, optional query) to ResearchAction.
   - `LiveVerdictBackend` gains optional `search_provider` + `map_provider` and caps
     `max_searches=2`, `max_maps=2`. `_handle_search` journals SearXNG results
     (title/url/engine); `_handle_map` journals Firecrawl Local map links
     (url/title/description). DISCOVERY ONLY — neither enters the evidence ledger; the model
     must still fetch_url + cite_source (sha256/verbatim audit unchanged).
   - Missing/disabled provider → honest journal note; never guess or call a paid fallback.
   - Stats/report: searches_used, maps_used, provider_errors (no credits — everything free).
   - SYSTEM_PROMPT: document both actions; prefer map_site on an institution domain instead
     of guessing course URLs; use search_web for discovery; NEVER construct Google search
     URLs; paid/cloud services are unavailable.
   - Defaults: max_turns 12 (was 8; the diagnosis showed the budget too tight),
     max_fetches 6, max_searches 2, max_maps 2. CLI flags `--max-searches`, `--max-maps`;
     wire through. Existing mock canary sessions still valid.
2. CLI builds providers from env only: SearXNG base URL default
   `http://127.0.0.1:${SEARXNG_PORT:-8888}`; Firecrawl Local URL from
   `FIRECRAWL_LOCAL_URL`; no API keys.
3. Tests:
   - Extend scripted live sessions with a search_web turn and a map_site turn; journal
     contains discovered URLs; report stats present.
   - Search cap + map cap enforced; provider unavailable → journal note, no crash.
   - Firecrawl Local cloud-source rejection propagates as an honest failure (no evidence).
   - Existing verdict-live tests + all product/verdict dry-runs stay green.

================================================================================
TASK D — env, docs, future opportunities (DOC ONLY)
================================================================================
1. `.env.example` additions (no secrets):
   ```
   # --- Invisible Playwright MCP browser tier ---
   FMD_BROWSER_MCP_ENABLED=1
   FMD_BROWSER_MCP_COMMAND=uvx
   FMD_BROWSER_MCP_ARGS=invisible-playwright-mcp
   STEALTHFOX_BINARY=
   STEALTHFOX_PROXY=
   STEALTHFOX_NO_PROXY=0
   STEALTHFOX_PROFILE_DIR=
   STEALTHFOX_SEED=
   STEALTHFOX_HEADLESS=1
   INVISIBLE_MCP_HOME=
   # --- Firecrawl Local (FREE; server must run with FIRECRAWL_CLOUD=off) ---
   FIRECRAWL_LOCAL_ENABLED=0
   FIRECRAWL_LOCAL_URL=http://127.0.0.1:3002
   FIRECRAWL_LOCAL_TIMEOUT_S=60
   FIRECRAWL_CLOUD=off
   # --- SearXNG (already in compose; FREE) ---
   SEARXNG_URL=http://127.0.0.1:8888
   ```
   Do NOT edit the real `.env`.
2. `docs/product-development-cycle/19-free-web-data-stack.md`: current free stack, install
   (uv/uvx, cargo install firecrawl-local), start/probe/stop commands, `FIRECRAWL_CLOUD=off`
   guarantee, SearXNG reuse, politeness/no-auth rules, and the no-paid-API policy.
3. `docs/v1/eval/09-free-web-discovery-runbook.md`: verdict-live actions/caps/env/commands,
   report fields, and how to read searches/maps/provider_errors.
4. `docs/product-development-cycle/20-future-paid-web-services.md`: opportunities YET TO BE
   TAPPED, explicitly NOT implemented: Firecrawl Cloud Search/Scrape/Crawl, Semantic Index,
   managed anti-bot/proxies, Alexandria providers/datasets, Interact/Agent/Monitor,
   enterprise controls; Tavily search. For each: potential value, cost model/free-tier
   caveat, adoption trigger (measured local failure rate, volume, approved budget), and
   provenance/privacy risks. State invariant #10 remains until an explicit product decision.
5. AGENTS.md: add a read-order line for `20-future-paid-web-services.md` so agents see the
   opportunity list but know not to implement paid services by default.

GATES (all at branch tip; report counts)
- `uv sync --all-extras`; `uv run ruff check .`
- `uv run pytest -q` (offline; fake MCP, httpx mocks, no real engine/Firecrawl/SearXNG)
- `uv run python -m fmd.cli eval --golden ../../data/golden` (provenance 1.0)
- verdict-live replay canary + all product/verdict dry-runs exit 0
- DB-backed pytest cannot run in your sandbox (no postgres) — say so; operator runs at intake.

SCOPE
Allowed: `packages/pipeline/fmd/fetch/**`, `fmd/eval/verdict_live.py`,
`fmd/eval/verdict.py` (read-only unless a minimal shared type is needed),
`fmd/discover/searxng.py` (read-only; reuse), `fmd/cli.py` (max-searches/max-maps wiring),
`tests/**`, `pyproject.toml`, `uv.lock`, `.env.example`,
`docs/product-development-cycle/{19,20}-*.md`, `docs/v1/eval/09-*`,
`scripts/firecrawl_local.sh` (+ optional probe helper), `reports/**` (mock canary).
Forbidden: real `.env`, any paid SDK/key/config, `fmd/extract/**`, `apps/**`, `infra/**`
(SearXNG is already wired), `config/predicates.yaml`, `config/sources.yaml`,
`db/migrations/*`, `docs/v1/LOCAL-QUEUE.md`.

REPORT (chat ≤ 400 words): what was built, test names + counts, gates, PR URL, and top 3
honest gaps (host setup + first-run MCP engine download; Firecrawl Local needs Rust/cargo;
SearXNG upstream engines can rate-limit; no live provider validation in sandbox).
~~~~
