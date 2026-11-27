# FMD-CLOUD-029-mcp-runtime-defaults · two residual MCP fixes · update PR #16 directly

~~~~
You are the cloud coding agent applying two small follow-up fixes to the EXISTING fix branch.

REPO: https://github.com/architgore050/findMyDreamCollege.git
BRANCH: `v1/free-web-stack-fixes` (currently `9d117c4febf84ff9058065177293ba5db97463d2`)
All output — English-only.

WORK DIRECTLY ON THIS BRANCH (no new branch, no new PR):
1. `git switch v1/free-web-stack-fixes`; `git pull --ff-only`; verify HEAD.
2. Make the two fixes below, commit + push after each. PR #16 updates automatically.

READ FIRST
- packages/pipeline/fmd/fetch/mcp_browser.py (env passthrough, navigate call, tests)
- tests/test_fetch_mcp_browser.py
- tests/test_real_server_integration.py
- docs/product-development-cycle/19-free-web-data-stack.md

REAL-HOST FINDINGS (2026-11-27, after PR #16 fixes)
Everything else passed: MCP 20/20 tools, Firecrawl 30/30, ladder 403 through all four tiers,
opt-in integration 2/2 with STEALTHFOX_HEADLESS=0. Two residual blockers:

1. RUNTIME HEADLESS DEFAULT
   `.env.example` says STEALTHFOX_HEADLESS=0, but when the variable is UNSET the spawned
   upstream server still defaults to headless=1 and fails on this host (no Xvfb):
   `mcp_browser_no_display`. The exact opt-in integration command without the env fails.
   Fix: in `_passthrough_env()` (or the single spawn-env construction point), use
   `env.setdefault("STEALTHFOX_HEADLESS", "0")` AFTER copying explicit host values, so:
   - unset → spawned env has 0 (headed, matching the documented default);
   - explicit 0 → 0;
   - explicit 1 → 1 (and the existing no-Xvfb preflight still fires).
   Add a unit test for unset/explicit-0/explicit-1.

2. JS READ RACE
   `browser_navigate(wait_until="domcontentloaded")` + immediate `browser_read_html` returned
   the stale static shell in 1 of 10 `/js` fetches (9/10 rendered).
   Fix: make the navigate wait strategy deterministic for JS pages:
   - default `wait_until="networkidle"` (configurable via `FMD_BROWSER_MCP_WAIT_UNTIL`,
     accepted values domcontentloaded|load|networkidle);
   - if networkidle times out / navigation errors, one bounded fallback to
     `domcontentloaded`, then read (do not silently return a known-stale shell);
   - keep the existing call timeout and non-GET refusal.
   Add unit tests for networkidle default + fallback. Extend the opt-in real-server
   integration fixture with delayed JS (e.g. setTimeout 250ms) and repeat the fetch 10x,
   asserting rendered content every time (test is skipped unless FMD_MCP_INTEGRATION=1).

GATES
- `uv sync --all-extras`; `uv run ruff check .`
- `uv run pytest -q` (offline; report counts)
- golden provenance 1.0; verdict-live replay + dry-runs exit 0
- DB pytest cannot run in sandbox — parent runs it.
- Do NOT claim real-server validation here; parent reruns the two real-provider checks.

REPORT (chat ≤ 250 words): two fixes, test names/counts, gates, pushed commit SHAs, any gap.
~~~~
