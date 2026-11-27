# 09 — Free web discovery runbook (verdict-live with SearXNG + Firecrawl Local)

Status: written 2026-11-27 (FMD-CLOUD-027). Companion to
`docs/v1/eval/07-verdict-live-runbook.md` (the original LQ-9 sequence) and
`docs/product-development-cycle/19-free-web-data-stack.md` (the stack itself).

The 2026-11-26 live run (`reports/eval-verdict-live-20261126T150000Z.json`)
degraded all five canaries to `unknown`: the agent guessed course URLs (404s),
tried to fetch Google search pages (robots), and hit the 8-turn budget. This
runbook is the re-run sequence with the free discovery stack enabled.

## 1. Preconditions (host)

- Model server per doc 07 §1–2 (`scripts/serve_local.sh start && … probe`).
- Free discovery, all local:
  - SearXNG: `make up` (or the compose service) — `${SEARXNG_PORT:-8888}:8080`,
    JSON format enabled. Probe: `curl -s 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 300`.
  - Firecrawl Local (optional but recommended):
    `cargo install firecrawl-local` (Rust/cargo required), then
    `scripts/firecrawl_local.sh start` — it always starts the server with
    `FIRECRAWL_CLOUD=off` and waits for `GET /health` on :3002.
  - MCP browser tier: `uv sync --all-extras`; the server
    (`uvx invisible-playwright-mcp`) downloads its engine on first
    `browser_open` (honest `mcp_browser_downloading` failure if a fetch races
    the download; retry afterwards).
- `.env`: keep `FMD_BROWSER_MCP_ENABLED=1`, set `FIRECRAWL_LOCAL_ENABLED=1`
  once the local server is up, `SEARXNG_URL=http://127.0.0.1:8888`
  (see `.env.example`). No API keys — none of these services take one.

## 2. Actions and caps

The tool schema the model works with (enforced in code, never negotiable):

| Action | Free backend | Cap (default) | Output |
| --- | --- | --- | --- |
| `fetch_url` | fetch ladder (httpx → curl_cffi → MCP browser → Firecrawl Local) | `--max-fetches 6` | stored page (ledger) |
| `search_web` | SearXNG (`SEARXNG_URL`) | `--max-searches 2` | title/url/engine leads |
| `map_site` | Firecrawl Local (`FIRECRAWL_LOCAL_URL`) | `--max-maps 2` | url/title/description links |
| `cite_source` | — | — | verbatim-verified citation |
| `emit_verdict` | — | — | structured verdict |

`--max-turns` defaults to **12** (was 8; the diagnosis showed the budget too
tight once discovery actions are in the mix). Discovery results are **leads
only**: they never enter the evidence ledger, and a cited URL must still be
`fetch_url`-ed in the same run (`sha256` + verbatim audit unchanged). The
prompt also forbids constructing Google/search-engine URLs and any paid or
cloud call; a missing provider produces an honest journal note instead.

## 3. Run

```
set -a && . .env && set +a
cd packages/pipeline
uv run python -m fmd.cli eval --eval-kind verdict --stream verdict-live \
  --report-out reports/eval-verdict-live-<ts>.json \
  --audit-sample 0.15 --audit-seed 13
```

The command prints, among the usual lines:

```
  limits: max_turns 12 max_fetches 6 max_searches 2 max_maps 2 (hard, enforced in code)
  discovery (free providers): searches 2 maps 1 provider_errors 0 (search: searxng, map: firecrawl_local; no paid services)
```

## 4. Reading the report

Per case (`cases[].backend_run`) and in the aggregate:

- `searches_used` / `maps_used` — how often the agent spent its discovery
  budget. Persistent 0s with an available provider usually mean the model
  ignored the guidance; many caps hit (`SEARCH/MAP BUDGET EXHAUSTED` in the
  journal) mean the caps, not the stack, are the limiting factor.
- `provider_errors` — SearXNG/Firecrawl failures (typed, e.g.
  `cloud_handoff_forbidden`, `SearxngUnavailable`). Everything is free, so an
  error is an infrastructure problem to fix, never a billing problem. A
  cloud-handoff rejection means the Firecrawl server was started WITHOUT
  `FIRECRAWL_CLOUD=off` — restart it with `scripts/firecrawl_local.sh start`.
- Discovery URLs appear only in the journal text; `fetch_ledger` still lists
  exactly the pages the fetch layer stored, and `reverify.failures` must stay
  `[]`.

Interpretation guidance: `agreement_rate` and `unknown_coverage` are read
exactly as in doc 07. The honest expectation for a first discovery-enabled
run is fewer guessed-URL 404s (leads replace guesses) and fewer
`turn_cap_reached` stops, not perfect verdicts.

## 5. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `SEARCH UNAVAILABLE` in every journal | SearXNG not running or `SEARXNG_URL` wrong; probe the `/search?format=json` route |
| `MAP UNAVAILABLE` | `FIRECRAWL_LOCAL_ENABLED=0` or server down; `scripts/firecrawl_local.sh start` |
| `SEARCH FAILED (SearxngUnavailable)` | upstream engines rate-limited; retry later or add engines in `infra/searxng/settings.yml` |
| `MAP FAILED (… cloud_handoff_forbidden)` | server started without `FIRECRAWL_CLOUD=off`; restart via the script |
| `FETCH FAILED … mcp_browser_downloading` | engine still downloading on first use; run again |
| `FETCH FAILED … mcp_browser_launch_failed` / `missing_libraries` | host can't run the browser; check `STEALTHFOX_*` env and system libs |

## 6. What stays frozen

Scoring, the audit worksheet flow, `reverify_evidence`, the circularity guard,
and the honest-`unknown` degradation are untouched. The discovery actions
widen what the agent can *find*; nothing about what counts as *evidence*
changed.
