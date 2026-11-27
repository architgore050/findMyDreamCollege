# 09 — Free web discovery runbook (verdict-live with SearXNG + Firecrawl Local)

Status: written 2026-11-27 (FMD-CLOUD-027). Companion to
`docs/v1/eval/07-verdict-live-runbook.md` (the original LQ-9 sequence) and
`docs/product-development-cycle/19-free-web-data-stack.md` (the stack itself).

The 2026-11-26 live run (`reports/eval-verdict-live-20261126T150000Z.json`)
degraded all five canaries to `unknown`: the agent guessed course URLs (404s),
tried to fetch Google search pages (robots), and hit the 8-turn budget. This
runbook is the re-run sequence with the free discovery stack enabled.

The 2026-11-27 live run (`reports/eval-verdict-live-20261127T172435Z.json`)
then reached `max_turns=12` on all five profiles: each used two searches, no
maps, and cases 1–4 cited external evidence without emitting a verdict. The
final-turn directive and one-search default below address that failure mode.

A subsequent post-tuning live run exposed two more defects addressed here: three of five cases
emitted but had zero external citations, while two stopped on turn 1 because a local schema error
triggered the global local-to-API fallback and the unresolved API provider masked
`json_schema_violation`. The fixes are a penultimate citation checkpoint, an emit-only final turn
enforced in code, and `allow_fallback: false` for `verdict_live`.

## 1. Preconditions (host)

- Model server per doc 07 §1–2 (`scripts/serve_local.sh start && … probe`).
- Free discovery, all local:
  - SearXNG: `make up` (or the compose service) — `${SEARXNG_PORT:-8888}:8080`,
    JSON format enabled. Probe: `curl -s 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 300`.
  - Firecrawl Local (optional but recommended): install the tested version
    (`cargo install --locked --version 0.1.1 firecrawl-local`, Rust/cargo
    required), then `scripts/firecrawl_local.sh start` — it always starts
    the server with `FIRECRAWL_CLOUD=off` and waits for `GET /health` on
    :3002. Local PDF/DOCX parsing goes through **`/v2/scrape`** (parsed
    text arrives as `data.markdown`); `/v2/parse` answers 501 `CLOUD_ONLY`
    with the cloud off — do not use it.
  - MCP browser tier: `uv sync --all-extras`; the server
    (`uvx invisible-playwright-mcp==0.74.0`, the tested version) downloads
    its engine on first `browser_open` (honest `mcp_browser_downloading`
    failure if a fetch races the download; retry afterwards). The tier is
    **headed by default** (`STEALTHFOX_HEADLESS=0`): `DISPLAY`,
    `XAUTHORITY`, `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR` and
    `DBUS_SESSION_BUS_ADDRESS` are passed through to the server. For
    headless (`STEALTHFOX_HEADLESS=1`) install `xvfb` first — without it
    the tier fails a preflight naming that fix instead of a bare browser
    error.
- `.env`: keep `FMD_BROWSER_MCP_ENABLED=1`, set `FIRECRAWL_LOCAL_ENABLED=1`
  once the local server is up, `SEARXNG_URL=http://127.0.0.1:8888`
  (see `.env.example`). No API keys — none of these services take one.

## 2. Actions and caps

The tool schema the model works with (enforced in code, never negotiable):

| Action | Free backend | Cap (default) | Output |
| --- | --- | --- | --- |
| `fetch_url` | fetch ladder (httpx → curl_cffi → MCP browser → Firecrawl Local) | `--max-fetches 6` | stored page (ledger) |
| `search_web` | SearXNG (`SEARXNG_URL`) | `--max-searches 1` | title/url/engine leads |
| `map_site` | Firecrawl Local (`FIRECRAWL_LOCAL_URL`) | `--max-maps 2` | url/title/description links |
| `cite_source` | — | — | verbatim-verified citation |
| `emit_verdict` | — | — | structured verdict |

`--max-turns` defaults to **12** (was 8; the diagnosis showed the budget too
tight once discovery actions are in the mix). On the **penultimate turn**, if no
citation exists but the fetched ledger is non-empty, the prompt requires
`cite_source` now and forbids `search_web`, `map_site`, and `fetch_url`. The
**final turn is reserved for `emit_verdict` and enforced in code**: a non-emit
action is rejected before any handler runs, so it cannot fetch, search, map, or
cite. The case degrades to `unknown` with
`stop_reason = final_turn_action_rejected`; the directive never relaxes validation. A final
`emit_verdict` with zero citations is rejected too (`final_turn_zero_citations`) and degrades to
`unknown`, so ignoring the checkpoint cannot publish an uncited success.

The `verdict_live` model profile sets **`allow_fallback: false`** in
`config/models.yaml`. A local provider error is re-raised unchanged; the router
never tries the global `api` fallback (invariant 10). In particular, a local
`json_schema_violation` reaches the backend's schema-retry instead of being
masked by `model_error:unresolved_provider` from an unconfigured paid provider.

`--max-searches` now defaults to **1** (was 2) so discovery cannot consume the
emit turn; `--max-maps` remains **2**. Both remain overridable with the CLI
flags. After `search_web` identifies an official institution domain — or one
is visible in `ingested_evidence_urls` — call `map_site` on that domain with
the programme name as the query **before guessing course paths**, then fetch a
returned programme link. Discovery results are **leads only**: they never
enter the evidence ledger, and a cited URL must still be `fetch_url`-ed in
the same run (`sha256` + verbatim audit unchanged). The prompt also forbids
constructing Google/search-engine URLs and any paid or cloud call; a missing
provider produces an honest journal note instead.

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
  limits: max_turns 12 max_fetches 6 max_searches 1 max_maps 2 (hard, enforced in code)
  discovery (free providers): searches 1 maps 1 provider_errors 0 (search: searxng, map: firecrawl_local; no paid services)
```

## 4. Reading the report

Per case (`cases[].backend_run`) and in the aggregate:

- `searches_used` / `maps_used` — how often the agent spent its discovery
  budget. Persistent 0s with an available provider usually mean the model
  ignored the guidance. One search is the default and may be followed by
  `SEARCH BUDGET EXHAUSTED`; that is expected, not a provider failure. Use
  `map_site` on the discovered official domain, then fetch/cite or emit.
  `final_turn_action_rejected` means the model chose a non-emit action on the
  final turn; the harness rejected it without execution and kept the honest
  `unknown` degradation. `final_turn_zero_citations` means the model ignored the
  penultimate checkpoint and tried to emit with no citations; the harness kept
  `unknown` instead of publishing an uncited success.
  `model_error:json_schema_violation` is the local schema
  error itself (retried by the backend while turns remain), never an API
  fallback failure.
- `provider_errors` — SearXNG/Firecrawl failures (typed, e.g.
  `cloud_handoff_forbidden`, `SearxngUnavailable`). Everything is free, so an
  error is an infrastructure problem to fix, never a billing problem. A
  cloud-handoff rejection means the Firecrawl server was started WITHOUT
  `FIRECRAWL_CLOUD=off` — restart it with `scripts/firecrawl_local.sh start`.
- Discovery URLs appear only in the journal text; `fetch_ledger` still lists
  exactly the pages the fetch layer stored, and `reverify.failures` must stay
  `[]`.

Interpretation guidance: `agreement_rate` and `unknown_coverage` are read
exactly as in doc 07. The honest expectation for the next discovery-enabled
run is fewer guessed-URL 404s (leads replace guesses) and fewer
`turn_cap_reached` stops, not perfect verdicts.

## 5. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `SEARCH UNAVAILABLE` in every journal | SearXNG not running or `SEARXNG_URL` wrong; probe the `/search?format=json` route |
| `SEARCH BUDGET EXHAUSTED` after one search | Expected with the default cap; map the official institution domain, then fetch/cite or emit rather than searching again |
| `final_turn_action_rejected` | The model chose search/map/fetch/cite on the final turn; the harness rejected it without execution. Inspect the prompt/journal, keep the honest `unknown`, and retune before another live run |
| `final_turn_zero_citations` | The model ignored the penultimate citation checkpoint and emitted with no citations; the harness rejected the emission and kept `unknown`. Inspect fetched-page excerpts and quote selection before retrying |
| `model_error:unresolved_provider` on turn 1 | Check for an env override pointing `verdict_live` at a missing provider. It must not be the global fallback: `allow_fallback: false` is required |
| `model_error:json_schema_violation` | The local model returned an invalid action/verdict schema. The backend retries while turns remain; if exhausted, keep `unknown` and inspect prompt/schema compatibility |
| `MAP UNAVAILABLE` | `FIRECRAWL_LOCAL_ENABLED=0` or server down; `scripts/firecrawl_local.sh start` |
| `SEARCH FAILED (SearxngUnavailable)` | upstream engines rate-limited; retry later or add engines in `infra/searxng/settings.yml` |
| `MAP FAILED (… cloud_handoff_forbidden)` | server started without `FIRECRAWL_CLOUD=off`; restart via the script |
| `FETCH FAILED … mcp_browser_downloading` | engine still downloading on first use; run again |
| `FETCH FAILED … mcp_browser_launch_failed` / `missing_libraries` | host can't run the browser; check `STEALTHFOX_*` env and system libs |
| `FETCH FAILED … mcp_browser_no_display` | the server lost its display; keep the tier headed (`STEALTHFOX_HEADLESS=0`) and pass the display env through, or install `xvfb` for headless |
| `FETCH FAILED … target_blocked` | the site blocked the local renderer (honest 403, no cloud involved); the ladder escalates or quarantines |
| `MAP FAILED (target_blocked)` | the site blocked the mapping request; retry later or map a narrower domain |

## 6. What stays frozen

Scoring, `reverify_evidence`, the circularity guard, and the honest-`unknown`
degradation are untouched. The audit worksheet keeps its seeded stratification
and 10–20 % target band, with a `min(2, n_cases)` entry floor; the formal gate
still needs two completed audits. Discovery and the citation/final-turn guards
widen what the agent can *find* and constrain when it acts; nothing about what
counts as *evidence* changed.
