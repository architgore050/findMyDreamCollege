# FMD-CLOUD-028-free-web-stack-fixes · fix real-provider contract failures · wave w15-fix

~~~~
You are the one-shot cloud coding agent fixing PR #15 after full local real-provider testing.

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE BRANCH: `v1/free-web-stack`
BASE SHA: b06400a20113df5edb6fb092300d1b10b26cb0e2
All output — English-only.

BRANCHING
1. `git switch -c v1/free-web-stack-fixes` from `v1/free-web-stack` at the base SHA.
2. `git push origin v1/free-web-stack-fixes` immediately.
3. Commit + push after every logical step. End: PR `v1/free-web-stack-fixes` → `v1/free-web-stack`
   (NOT main; the parent will merge this into the existing PR #15 branch).

READ FIRST
- AGENTS.md (invariants #9/#10; no paid APIs)
- packages/pipeline/fmd/fetch/mcp_browser.py (whole file)
- packages/pipeline/fmd/fetch/firecrawl_local.py (whole file)
- tests/test_fetch_mcp_browser.py + tests/test_fetch_firecrawl_local.py
- docs/product-development-cycle/19-free-web-data-stack.md
- docs/v1/eval/09-free-web-discovery-runbook.md
- .env.example + scripts/firecrawl_local.sh

REAL-SERVER FINDINGS (local host, 2026-11-27)
The official invisible-playwright-mcp 0.74.0 server passed all 20 tools. Firecrawl Local
0.1.1 passed real scrape/map/crawl/batch/PDF/DOCX. The repo adapters failed these contracts:

1. MCP navigate wire shape (release blocker)
   Actual upstream answer:
   - text: `navigated to http://127.0.0.1:18765/ok (HTTP 200)`
   - structuredContent: `{"result": "navigated to http://127.0.0.1:18765/ok (HTTP 200)"}`
   Repo `_parse_navigation` only understands `status:` / `url:` fields, so every real GET raised
   `BrowserNavigationError: navigate answer carries no parseable HTTP status`.
   Fix: parse the actual text (and structuredContent.result) with a strict regex for
   `navigated to <url> (HTTP <3-digit status>)`; retain backward-compatible field parsing.
   Add a committed fixture with this exact real wire shape + tests for 200/redirect/404.

2. MCP browser_open error shape (medium)
   Actual upstream error: `isError=true` with structuredContent
   `{"opened": false, "cause": "no_display"}`. Repo `_call` raises before
   `_ensure_browser_open` can preserve the cause.
   Fix: preserve structured error causes; return honest
   `FetchResult(status=0, error="mcp_browser_no_display")` (and other causes:
   missing_libraries, sandbox_blocked, launch_failed, downloading, download_failed,
   profile_locked). Add fixtures/tests.

3. MCP spawn display environment (release blocker on this host)
   `_passthrough_env()` only forwards STEALTHFOX_/INVISIBLE_MCP_ prefixes, so a headed server
   loses DISPLAY/XAUTHORITY and Firefox reports no DISPLAY. The committed
   `STEALTHFOX_HEADLESS=1` default requires Xvfb, which is not installed on this host.
   Fix:
   - pass through DISPLAY, XAUTHORITY, WAYLAND_DISPLAY, XDG_RUNTIME_DIR,
     DBUS_SESSION_BUS_ADDRESS when present;
   - change `.env.example` default to `STEALTHFOX_HEADLESS=0` (headed on the operator's
     existing display); keep headless configurable;
   - if headless=1 and no Xvfb is available, fail with a clear TierUnavailable/preflight
     message (`install xvfb` or set STEALTHFOX_HEADLESS=0), not a generic browser error.
   Add env-passthrough tests.

4. Firecrawl redirect provenance (high)
   Real API: `metadata.sourceURL=/redirect`, `metadata.url=/ok`, statusCode=200.
   Repo uses sourceURL as final_url. Fix: prefer `metadata.url`, fall back to sourceURL/request.
   Add real-shape fixture + test.

5. Firecrawl blocked target status (medium)
   Real API: `data.source=local`, `blocked=true`, `metadata.statusCode=403`.
   Repo raises `FirecrawlCloudHandoffForbidden` before reading status, then fetcher returns
   status 0. Fix: distinguish local blocked target from cloud taint:
   - local blocked → honest FetchResult(status=403, error="target_blocked", empty body) so the
     ladder can escalate/quarantine;
   - cloud source → fail closed (cloud_handoff_forbidden), never return content.
   Add fixture/test for both.

6. Firecrawl content-type (medium)
   PDF/DOCX parsed markdown is returned with `content-type: text/html`.
   Fix: HTML body → text/html; markdown fallback/parsed document → text/markdown; only use
   metadata.contentType when it matches the bytes actually returned. Add PDF/DOCX fixtures.

7. Firecrawl no-cloud guard (medium, defense in depth)
   `_is_cloud_or_blocked` checks metadata/data only; top-level `source=firecrawl-cloud` is
   ignored; map() has no source/blocked check.
   Fix: a single provenance helper checks top-level + data + metadata (and nested known
   containers) for `source=firecrawl-cloud`; apply it to scrape AND map; reject blocked map
   responses. Add top-level cloud fixtures for both endpoints.

8. Version pins + contract drift (low but required)
   Tested: invisible-playwright-mcp 0.74.0; Firecrawl Local 0.1.1.
   Fix: make versions explicit/tested defaults — `FMD_BROWSER_MCP_ARGS` default
   `invisible-playwright-mcp==0.74.0`; Firecrawl install uses `--locked` and a documented
   tested version (0.1.1). Keep env-overridable. Add an opt-in real-server integration test
   (skipped unless `FMD_MCP_INTEGRATION=1` / `FMD_FIRECRAWL_INTEGRATION=1`) so upgrades fail
   loudly on contract drift.

9. Docs/ops clarifications
   - Local document parsing is `/v2/scrape` (PDF/DOCX passed); `/v2/parse` returned 501
     `CLOUD_ONLY` with cloud off. Say this explicitly in doc 19 + runbook 09.
   - Document headed/headless + Xvfb requirements and the display passthrough.
   - Add a public-crawl preflight warning/error when FMD_USER_AGENT still contains
     `set-me@example.com` / placeholder contact. Do NOT invent a real address; the operator
     must configure it. Tests/local fixtures must remain usable.

REQUIRED TESTS (offline in CI; real-server tests opt-in)
- Commit compact real-wire fixtures under tests/fixtures/mcp/ and
  tests/fixtures/firecrawl_local/ for: navigate text+structuredContent, browser_open error,
  redirect metadata, local blocked 403, PDF/DOCX markdown, top-level cloud scrape/map.
- Contract tests feed those fixtures through the real parser/client code.
- Env passthrough test; navigate 200/redirect/404; open cause preservation; Firecrawl
  final_url/status/content-type/provenance; map cloud rejection.
- Opt-in integration tests (default skip) for actual MCP navigate and actual Firecrawl
  scrape redirect/blocked/cloud-off. They must never call a paid service.
- Existing full suite stays green.

GATES (branch tip; report counts)
- `uv sync --all-extras`; `uv run ruff check .`
- `uv run pytest -q` (offline; report counts)
- golden eval provenance 1.0; verdict-live replay + all dry-runs exit 0
- DB pytest cannot run in your sandbox — say so; parent runs it at intake.

SCOPE
Allowed: `fmd/fetch/mcp_browser.py`, `fmd/fetch/firecrawl_local.py`, related tests/fixtures,
`.env.example`, `scripts/firecrawl_local.sh`, docs 19/09, and a minimal crawl preflight in
`fmd/cli.py` or settings if needed. Forbidden: paid SDKs/APIs, real `.env`, `apps/**`,
`infra/**`, `config/sources.yaml`, `config/predicates.yaml`, `db/migrations/*`, main branch.

REPORT (chat ≤ 400 words): each fix + test, gate counts, PR URL, and any remaining gap.
Do NOT claim real-server validation in the cloud sandbox; the parent reruns the local matrix.
~~~~
