# 19 — Free web data stack (invisible MCP browser + Firecrawl Local + SearXNG)

Status: implemented 2026-11-27 (FMD-CLOUD-027, branch `v1/free-web-stack`).
Scope: the fetch ladder's stealth tiers, the verdict-live discovery actions,
and the lifecycle tooling around them.

The diagnosis in `reports/eval-verdict-live-20261126T150000Z.json` was clear:
the live verdict agent guessed course URLs (404s), tried Google (robots), and
ran out of turns — every verdict came back `unknown`. The fix is a **free,
self-hosted web stack**: a stealth browser tier over an MCP server, a local
Firecrawl scrape/map tier, and SearXNG search for discovery. No paid API, no
cloud hand-off, ever (invariant 10).

## 1. What runs where

```
fetch ladder (fmd/fetch/ladder.py)
  1. HttpxFetcher            pooled httpx, HTTP/2, conditional GETs
  2. CurlCffiFetcher         browser TLS fingerprint impersonation
  3. InvisiblePlaywrightMCPFetcher   fmd/fetch/mcp_browser.py (stdio MCP)
  4. FirecrawlLocalFetcher   fmd/fetch/firecrawl_local.py (v2 API, local only)
  -> quarantine              every tier blocked; politeness wins (invariant 9)
```

Escalation is unchanged: only a block signal (403/429/503, `cf-mitigated`)
moves down the ladder; 404/410 stays terminal on the first tier; robots.txt is
honoured before any tier runs; failures land in the append-only scrape-failure
log (`docs/product-development-cycle/18-scrape-failures.md`).

## 2. Invisible Playwright MCP browser tier

The in-process PatchrightFetcher is gone. The third tier now drives the
external `invisible-playwright-mcp` server (MIT, Python >= 3.11) over stdio
with the official `mcp` SDK — this process never launches a browser.

- Server: <https://github.com/oss-invisible/playwright_mcp> (PyPI:
  `invisible-playwright-mcp`, pinned to **0.74.0** — the version the
  2026-11-27 real-server matrix validated across all 20 tools). Tools used:
  `browser_open`, `browser_navigate(url, wait_until)`,
  `browser_read_html(mode="full")`. `browser_navigate` answers with one
  sentence — `navigated to <final-url> (HTTP <status>)` — in the text
  blocks and `structuredContent.result`, so 404s and login-wall redirects
  are visible to the ladder (explicit `status:`/`url:` field answers are
  still parsed for older shapes). The navigation waits for `networkidle`
  by default (`FMD_BROWSER_MCP_WAIT_UNTIL`:
  `domcontentloaded|load|networkidle`) so a JS-rendered page is read after
  the network settles, never as its static shell; a networkidle timeout
  falls back ONCE to `domcontentloaded` (logged, never silent).
- The server is stateful with one page, so every MCP call is serialized with a
  lock. `FetchLadder.close()` (and the orchestrator/CLI shutdown paths)
  terminates it.
- First run downloads the engine; `browser_open` answers
  `opened=false, cause=downloading`. The tier reports **honest failures**
  (`mcp_browser_downloading`, `mcp_browser_launch_failed`,
  `mcp_browser_missing_libraries`, …) instead of pretending the page was
  fetched. A missing command or SDK is a `TierUnavailable`, and the ladder
  escalates past the tier.
- Config: `FMD_BROWSER_MCP_ENABLED` (default 1), `FMD_BROWSER_MCP_COMMAND`
  (default `uvx`), `FMD_BROWSER_MCP_ARGS` (default
  `invisible-playwright-mcp`); `STEALTHFOX_*` and `INVISIBLE_MCP_HOME` are
  passed through to the server unchanged (see `.env.example`).
- Install: `uv sync --extra crawl` (brings the `mcp` SDK); the server itself
  is fetched by `uvx` on first use. No `patchright install` step anymore.
- Display (2026-11-27 real-host finding #3): the server is **headed by
  default** (`STEALTHFOX_HEADLESS=0` in `.env.example`); `DISPLAY`,
  `XAUTHORITY`, `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR` and
  `DBUS_SESSION_BUS_ADDRESS` are passed through to the spawned server when
  set. `STEALTHFOX_HEADLESS=1` needs Xvfb — without it the tier fails a
  preflight naming the fix (install `xvfb` or set `STEALTHFOX_HEADLESS=0`)
  instead of a bare browser error. `browser_open` refusals arrive as
  `isError=true` with `structuredContent {opened: false, cause: ...}` and
  are reported verbatim (`mcp_browser_no_display`, …).
- Opt-in real-server contract tests (`FMD_MCP_INTEGRATION=1`,
  `FMD_FIRECRAWL_INTEGRATION=1`; skipped offline) replay the 2026-11-27
  matrix against a localhost-only origin so upgrades fail loudly on wire
  drift. The captured real wire shapes live in
  `packages/pipeline/tests/fixtures/mcp/` and
  `packages/pipeline/tests/fixtures/firecrawl_local/`.

## 3. Firecrawl Local (free, local-only)

`firecrawl-local` is a single Rust binary that serves the Firecrawl v2 API on
`127.0.0.1:3002` — no database, no queue, no container. Scrape, crawl, map and
PDF/Office parsing run locally.

- Install (never automatic; needs Rust/cargo): `cargo install --locked
  --version 0.1.1 firecrawl-local` — 0.1.1 is the tested version
  (`FIRECRAWL_LOCAL_VERSION` in `scripts/firecrawl_local.sh`;
  env-overridable so upgrades are deliberate).
- Lifecycle: `scripts/firecrawl_local.sh start|stop|probe`. `start` always
  runs `FIRECRAWL_CLOUD=off firecrawl-local serve`, waits for the pinned
  health route (`GET /health`), writes the PID/log under
  `/tmp/opencode/firecrawl-local`, and **prints the no-cloud guarantee**.
  `stop` kills only the PID it started.
- `FIRECRAWL_CLOUD=off` means "nothing goes to Firecrawl Cloud" upstream
  (unrecognized values are also treated as off). The Python client adds a
  second, independent guard: a deep provenance scan (top level, `data`,
  `metadata`, nested containers) **fails closed** on any
  `source: firecrawl-cloud` — for scrape AND map — with
  `cloud_handoff_forbidden`; a cloud result is never silently accepted, and
  Alexandria (cloud-only) is never called. A **local** block (`blocked:
  true` with a local source) is honest instead: scrape reports the real
  status with `target_blocked` and an empty body (the ladder escalates or
  quarantines), and map refuses the leads (`FirecrawlTargetBlocked`) —
  never content, never mistaken for cloud taint.
- Local PDF/DOCX parsing runs through **`/v2/scrape`** (validated 0.1.1):
  the parsed text arrives as `data.markdown` while `metadata.contentType`
  still says `application/pdf`. The client labels every body by what it
  actually is (`text/html` for HTML, `text/markdown` for parsed documents
  and the markdown fallback), so downstream consumers never get a
  content-type that lies about the bytes. **`/v2/parse` is not usable
  locally** — it answers 501 `CLOUD_ONLY` with the cloud off; do not call it.
- Client config: `FIRECRAWL_LOCAL_ENABLED` (default 0 until the operator
  starts the server), `FIRECRAWL_LOCAL_URL`, `FIRECRAWL_LOCAL_TIMEOUT_S`,
  `FIRECRAWL_LOCAL_MAX_AGE` (default 0 = always fresh). httpx only — the
  `firecrawl` SDK is deliberately not used.
- Local supports `markdown`/`html`/`rawHtml`/`links`; JSON/summary formats are
  skipped and browser actions are ignored.

## 4. SearXNG (already wired)

Compose runs SearXNG on `${SEARXNG_PORT:-8888}:8080` with the JSON format
enabled (`infra/searxng/settings.yml`). The pipeline reuses the existing
client — `fmd/discover/searxng.py` (`SearxngClient.search(query) ->
list[DocumentRef]` with title/engine/score metadata). Do not write a second
SearXNG client. Host env: `SEARXNG_URL=http://127.0.0.1:8888` (or set
`SEARXNG_PORT`; the default matches compose).

Upstream engines can rate-limit a self-hosted instance; the limiter is off in
the committed settings for local use. If SearXNG is unreachable or returns
non-JSON, `SearxngClient` raises its typed unavailable error and verdict-live
journals it as a provider error — it never falls back to a paid search.

## 5. verdict-live discovery actions

The verdict-live research loop (doc 07 is the operator runbook) has two extra
actions, both **discovery only — leads never enter the evidence ledger**:
evidence still requires `fetch_url` + `cite_source`, and the sha256/verbatim
audit is unchanged.

- `{"action": "search_web", "query": "..."}` — free SearXNG search
  (`max_searches`, default 2). Results are title + url + engine leads.
- `{"action": "map_site", "url": "...", "query": "optional focus"}` — free
  Firecrawl Local site map (`max_maps`, default 2). Prefer it on an
  institution's own domain instead of guessing course URLs.

Hard caps are enforced in code (`--max-searches` / `--max-maps`; default
`--max-turns` is now 12). The system prompt forbids constructing Google (or
any search-engine) URLs and states that paid/cloud services are unavailable.
A missing or failing provider is an honest journal note plus a
`provider_errors` counter — there is no paid fallback to try.

## 6. Commands (operator host)

```
uv sync --all-extras                       # mcp SDK + everything else
scripts/firecrawl_local.sh start           # FIRECRAWL_CLOUD=off, waits for /health
scripts/firecrawl_local.sh probe
scripts/firecrawl_local.sh stop            # kills only its own PID

# verdict-live with discovery (model server first, per docs/v1/LOCAL-QUEUE.md):
set -a && . .env && set +a
uv run python -m fmd.cli eval --eval-kind verdict --stream verdict-live \
  --report-out reports/eval-verdict-live-<ts>.json
```

## 7. The no-paid-API policy (invariant 10)

Nothing in this stack talks to a paid endpoint: the browser tier is a local
MCP server, Firecrawl Local runs on localhost with `FIRECRAWL_CLOUD=off`,
SearXNG is a compose service, and the model router stays provider-agnostic
with local defaults. The paid/managed opportunities that exist upstream are
**documented, not implemented** — see
`docs/product-development-cycle/20-future-paid-web-services.md` before
considering any of them. Invariant 10 stands until an explicit product
decision says otherwise.

Politeness note (invariant 9): the crawl preflight warns loudly when
`FMD_USER_AGENT` still carries the placeholder contact
(`set-me@example.com`). The operator must set a real address in `.env`
before pointing the crawler at the public web; nothing in the codebase
invents one, and the warning (not an error) keeps dry-runs and tests
usable.
