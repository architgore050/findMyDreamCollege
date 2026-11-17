# 13 — Network robustness (slow/choppy link)

Status: implemented 2026-11-18 (WS-NETROBUST). Scope: every outbound interaction
in the repo, made fail-fast, retried sanely, and resumable.

Operator context: hostel wifi at ~100–800 KiB/s, frequent drops, DNS timeouts,
occasional total outages. The failure this document fixes is not "slow" — it is
*one dead link turning into hundreds of sequential timeouts* and *a DNS hang
looking like a model failure*.

## 1. The shared policy

One module owns HTTP behaviour: `packages/pipeline/fmd/net/policy.py`. Call
sites import it; **no call site constructs a bare `httpx.Client`** in the files
this workstream owns. The factory is `make_client()`, which always sets
timeouts, a pooled-connection limit and an HTTP/2 attempt (falling back to
HTTP/1.1 when the optional `h2` package is absent).

| knob | default | why |
|---|---|---|
| connect timeout | **4 s** | covers DNS + TCP + TLS; a DNS black hole fails in ≤4 s, not 8 s per host |
| read / write timeout | 20 s | page fetch and discovery default |
| pool timeout | 4 s | waiting for a pooled socket must not hang |
| attempts | **3 total** (2 retries) | transient-only; the fetch ladder adds its own 3 tiers |
| backoff | full jitter, 0.5 s → 30 s | decorrelates parallel retries after an outage |
| `Retry-After` | honoured, numeric **and** HTTP-date, capped 300 s | a wrong header cannot park a run |
| outage breaker | **3 consecutive** DNS/connect failures → `network_unavailable` | abort the stage, never fail item-by-item for minutes |
| HTTP/2 | enabled when `h2` is installed | fewer connections on a lossy link |

`request_with_retries()` is the only retry loop. It retries
`{408, 425, 429, 500, 502, 503, 504}` and the retryable exceptions below; it
does **not** retry 401/403/404/410.

## 2. Error taxonomy

`classify_exception()` / `classify_browser_error()` produce a `NetworkFault`.
Reason strings are what stages record and what users see.

| fault | trigger | retryable | attempts | user-facing reason |
|---|---|---|---|---|
| `dns_failure` | `socket.gaierror`, "name or service not known" | **no** | 1 | "DNS resolution failed … the network is down" |
| `connect_timeout` | `httpx.ConnectTimeout`, `net::ERR_CONNECTION_TIMED_OUT` | yes | 3 | "connect timed out" |
| `read_timeout` | `httpx.ReadTimeout` (incl. mid-body) | yes | 3 | "read timed out" |
| `write_timeout` | `httpx.WriteTimeout` | yes | 3 | "write timed out" |
| `tls_error` | `ssl.SSLError`, certificate text | no | 1 | "TLS/certificate error" |
| `rate_limited` | 429 | yes, `Retry-After` first | 3 | "rate limited" |
| `server_error` | 5xx | yes | 3 | "server error" |
| `truncated_body` | `RemoteProtocolError` "incomplete/peer closed" | yes | 3 | "response truncated" |
| `connection_reset` | `ReadError`/`WriteError` | yes | 3 | "connection reset" |
| `network_unreachable` | `ERR_INTERNET_DISCONNECTED`, "network is unreachable" | yes | 3 → breaker | `network_unavailable` |
| `browser_crash` / `detached_frame` / `page_too_large` | Playwright strings, size cap | crash no; detached yes | 1 / 2 | "browser crashed" / "frame detached" / "page too large" |

**DNS fail-fast fix.** `dns_failure` is non-retryable and counts toward the
breaker. A host that cannot resolve costs one connect timeout (≤4 s) once, and
three such failures abort the stage with `network_unavailable`. Previously each
host paid a fresh 8 s connect budget and every URL retried the same dead
lookup.

## 3. Inventory (16 outbound paths)

| # | module / function | protocol | timeout before | retry before | failure mode | policy now |
|---|---|---|---|---|---|---|
| 1 | `fmd.fetch.ladder.HttpxFetcher` (tier 1) | HTTPS/1.1+2 | fixed 20 s all phases | none (tier escalation only) | DNS/TLS hang, truncated body | shared client, 4 s connect, 20 s read, 2 attempts, breaker |
| 2 | `CurlCffiFetcher` (tier 2) | HTTPS (impersonate) | 25 s | none | timeout, DNS | same guard/breaker; no internal retry (ladder escalates) |
| 3 | `PatchrightFetcher` (tier 3, browser) | Chromium/CDP | 30 s navigation, unbounded close | none | zombie Chromium, dead subresources | config-driven `BrowserPolicy`, asset blocking, page cap, per-domain lock, hard kill (§6) |
| 4 | `fmd.fetch.robots.RobotsCache` | HTTPS | inherits tier | none | unreachable robots treated per-domain | shares ladder/guard; `NetworkUnavailable` propagates instead of marking every domain blocked |
| 5 | `fmd.discover.sitemaps` | injected fetch (ladder) | caller's | unbounded queue | walks every sitemap on a dead host | `max_consecutive_failures=3`, `aborted_reason` |
| 6 | `fmd.discover.searxng` | HTTP JSON | 20 s | none | non-JSON, 5xx, no results | shared client, 3 attempts, `Retry-After`, breaker |
| 7 | `fmd.discover.common_crawl` | duckdb httpfs | 30 s (duckdb default) | none | parquet read stalls | `http_timeout=20`, `http_retries=2`, `http_retry_wait_ms=1000` |
| 8 | local llama.cpp / vLLM (`openai_compatible`) | HTTP JSON | **fixed 60 s** | none | 4k completion cut off / dead endpoint retried per doc | read = 30 s + `max_tokens/16` (cap 600 s), 3 attempts, **single slot**, breaker |
| 9 | hosted API (`api`, `anthropic`, `gemini`) | HTTPS JSON | fixed 60 s | none | same plus 429 | same scaling; `Retry-After`; breaker after 3 |
| 10 | `scripts/hf_download.py` | HTTPS range | 30 s connect / 120 s read | 20 per file, 5→120 s | stalled body forever, Range ignored | stall detector (8 KiB/s × 60 s), validates `Content-Range`, `Retry-After`, non-retryable 401/403/404 |
| 11 | `scripts/fetch_model.sh` | curl / aria2 | 30 s connect, curl speed probe | curl 20, aria2 **infinite** | aria2 hammers a dead link | aria2 `--max-tries=5`; curl `Retry-After`; Range-ignore already restarts |
| 12 | Docling/HF artifact fetch (`fmd.acquire.tiers`, Hugging Face Hub) | HTTPS | HF defaults (~10 min) | HF internal | first-use stall with no cache | offline guard already returns `artifacts_unavailable_offline` <1 s; prefetch policy `HF_HUB_ETAG_TIMEOUT=10`, `HF_HUB_DOWNLOAD_TIMEOUT=30` (documented; acquisition file is another workstream's) |
| 13 | web UI → API (`apps/web/lib/api.ts`) | fetch | none (could hang forever) | none | infinite spinner, retry storm on POST | 20 s `AbortController`, GET-only retries (2), POST never retried, `network_unavailable` / `network_timeout` distinct from 5xx |
| 14 | **Playwright MCP server** (agent browsing, outside pipeline) | Chromium/CDP via MCP | inherits MCP defaults; can hang indefinitely | MCP-internal/agent | agent waits on a dead page; zombie contexts | **documented recommendations only** (§7) — not active unless a human configures it |
| 15 | `fmd.adapters.nabh.NabhAdapter` (out of edit scope) | HTTPS POST | 30 s | none | bare `httpx.post`, default pool | flagged: should call `fmd.net.policy.make_client`; owner notified in §9 |
| 16 | `fmd.acquire.tiers` OCR POST (out of edit scope) | HTTP JSON | configured 120/300 s | none | bare `httpx.post` | flagged: same; local single-slot discipline would also apply |

Bare clients found: #15 and #16. Both have explicit timeouts but no connection
limits; they are outside this workstream's file scope and listed as follow-ups.

## 4. LLM path

* **Timeout scales with `max_tokens`**: `llm_timeouts(n).read = 30 + n/16`
  seconds, capped at 600. 128 tokens → 38 s; 4096 → 286 s. No fixed 120 s.
* **Retries only transient faults**, `Retry-After` first, max 3 attempts.
* **Single-slot local endpoint**: `_endpoint_slot(url)` serialises every request
  to loopback/`host.docker.internal`. One GPU, one completion.
* **Mid-batch abort**: `ModelRouter` counts consecutive transient failures per
  run. On the 3rd, `self._outage` is set and `NetworkOutage`
  (`reason="network_unavailable"`) is raised. Subsequent calls fail immediately
  without touching the socket. Results already returned/persisted are untouched;
  the caller must mark the remaining items `unverified` (it never fabricates a
  result). `router.network_state()` exposes the outage for the run summary.
* `fmd llm doctor --live` probes use 2 s connect / 5 s read, one attempt, so a
  dead endpoint reports in seconds.

## 5. Downloads and artifacts

* `hf_download.py`: resume from the real file size; a `206` must carry a
  `Content-Range` whose start equals the local size and whose total equals the
  Hub size, otherwise the file is **not** appended. A `200` to a ranged request
  restarts from zero. A transfer below 8 KiB/s for 60 s is aborted and resumed
  (curl `--speed-limit/--speed-time` equivalent). `Retry-After` is honoured;
  401/403/404 fail immediately.
* `fetch_model.sh`: already used `-C -`, `--speed-limit 10K`,
  `--speed-time 60`, Range-ignore detection. aria2's `--max-tries=0` was
  unbounded and is now 5 per attempt (the watchdog and wall budget still bound
  the outer loop).
* Hugging Face Hub (Docling artifact prefetch): set `HF_HUB_ETAG_TIMEOUT=10`
  and `HF_HUB_DOWNLOAD_TIMEOUT=30` before a prefetch. `FMD_OFFLINE=1` with no
  cache still fails in <1 s with `artifacts_unavailable_offline`; this
  workstream did not edit `fmd/acquire/**`.

No chunk is assembled from an unverified offset: the size check, the
`Content-Range` start check and the per-file size/hash verification are all
before the `.part` rename.

## 6. Browser tier (patchright)

Config-driven through `BrowserPolicy.from_env()`; every value is an env var so
a host can tune without code changes.

| setting | env var | default |
|---|---|---|
| navigation timeout | `FMD_BROWSER_NAV_TIMEOUT_MS` | 20 000 ms |
| action timeout | `FMD_BROWSER_ACTION_TIMEOUT_MS` | 5 000 ms |
| selector wait | `FMD_BROWSER_SELECTOR_TIMEOUT_MS` | 5 000 ms |
| network-idle extra wait (off by default) | `FMD_BROWSER_WAIT_NETWORK_IDLE`, `FMD_BROWSER_NETWORK_IDLE_TIMEOUT_MS` | 0 / 8 000 ms |
| page size cap | `FMD_BROWSER_MAX_PAGE_BYTES` | 8 MiB |
| block images/media/fonts | `FMD_BROWSER_BLOCK_ASSETS` | 1 (on) |
| browser fetches per domain | `FMD_BROWSER_MAX_CONCURRENCY_PER_DOMAIN` | 1 |
| close deadline before SIGKILL | `FMD_BROWSER_HARD_KILL_GRACE_MS` | 3 000 ms |
| launch timeout | `FMD_BROWSER_LAUNCH_TIMEOUT_MS` | 15 000 ms |
| persistent profile dir | `FMD_BROWSER_PROFILE_DIR` | empty (fresh isolated context) |

Behaviour:

* `page.goto(..., wait_until="domcontentloaded", timeout=nav)`; network-idle is
  an opt-in *best effort* that logs and continues, so one dead analytics beacon
  cannot stall a fetch.
* Route interception aborts `image`, `media` and `font` when block-assets is on.
  `page.content()` over the cap raises `page_too_large` instead of buffering.
* One lock per registrable domain; two browser fetches for `a.ac.in` never run
  concurrently.
* Close runs in a daemon thread with a deadline. If it hangs, **new** Chromium
  children of this process (`/proc` scan, SIGKILL) are killed and the PIDs are
  logged (`browser zombies killed after navigation timeout`). No persistent
  profile is shared by default, so a crash cannot poison later runs.
* Failure strings map to the same taxonomy as HTTP (`classify_browser_error`):
  `ERR_NAME_NOT_RESOLVED` → `dns_failure` (no retry), `ERR_INTERNET_DISCONNECTED`
  → breaker, `ERR_CONNECTION_TIMED_OUT`/`Timeout … exceeded` → `connect_timeout`
  (retry), `Target crashed` → `browser_crash` (no retry), detached frame →
  one retry. Breaker trips propagate as `NetworkUnavailable`.

## 7. Playwright MCP (agent-driven browsing) — recommendations only

**Status: not configured in this repository, and not active.** There is no
`opencode.json*`, `.mcp.json` or `mcpServers` entry in the repo. The operator's
global OpenCode config lists other MCP servers, not Playwright. Therefore this
section is guidance a **human must apply**; nothing here is silently in force.

Agent browsing is a different path from the pipeline's patchright tier: it runs
outside the fetch ladder, has no robots integration and no per-domain bucket.
It must be bounded by the same rules.

| setting | recommended | why |
|---|---|---|
| navigation timeout | 20 s | matches `FMD_BROWSER_NAV_TIMEOUT_MS` |
| action timeout | 5 s | a click/selector on a dead page must not hang |
| retry on navigation failure | at most 1, only for timeout/5xx | DNS/offline must fail once |
| headless | yes (isolated) | no Xvfb, no window-manager hangs |
| block images/media/fonts | yes | same bandwidth reason |
| isolated context per task | yes | no shared cookies/cache; no zombie profile |
| max concurrent pages | **1** | one link, one headless browser |
| browser process cleanup | close context, then kill on timeout | same zombie rule |

If a human adds `npx @playwright/mcp`, pass at minimum
`--headless --isolated --block-service-workers` (or the current equivalent) and
keep one page. MCP servers live outside this repo, so these are not testable
here and are deliberately not represented as active.

## 8. Measured baseline (2026-11-18, this host, this link)

Probes used the new policy with short timeouts; no retries beyond the policy.

| probe | result | elapsed |
|---|---|---|
| page fetch `https://example.com/` | 200, 577 B | **0.40 s** |
| page fetch `https://nabh.co/` | 200, 298 846 B | **2.15 s** |
| page fetch `https://www.aicte-india.org/` | TLS handshake **timed out** (`connect_timeout`) | **4.33 s** (bounded) |
| DNS for `fmd-probe-does-not-exist.invalid` | `dns_failure`, one attempt | **0.11 s** |
| local llama.cpp `GET /v1/models` | 200, 716 B | **0.03–0.15 s** |
| local llama.cpp chat, `max_tokens=8` (cold) | **read_timeout** at 31 s (model load) | **31.05 s** |
| same chat, warm | 200, `content="ok"` | **1.92 s** |
| HF `HEAD …/config.json` | 200 | **0.66 s** |
| HF ranged `GET`, `bytes=0-255` | **206**, `Content-Range: bytes 0-255/3001` | **0.96 s** |

Interpretation: the link is alive but flaky (one TLS handshake exceeded 4 s and
would have burned 8 s under the old defaults); DNS fails in 0.1 s instead of
hanging. The local model endpoint answers instantly but a cold completion
exceeds a small `max_tokens` budget — **warm the server before a batch** (the
runbook's cold-load measurement is 44 s; a 4096-token profile budgets 286 s and
absorbs it, a 128-token profile does not). HF supports Range, so the resumable
path is real.

## 9. Verification and residual work

* Offline tests (all with `httpx.MockTransport`, injected sleepers, fake timers):
  `tests/test_fetch_network.py`, `tests/test_llm_network.py`,
  `tests/test_discover_network.py`, `apps/web/lib/__tests__/api-network.test.ts`.
  They cover DNS fail-fast, connect/read timeout, truncated body, 429 numeric
  and HTTP-date `Retry-After`, 5xx-then-success, breaker abort, Range-ignored
  resume, LLM mid-batch abort, single-slot local calls, browser classification
  and UI timeout/retry rules. No test opens a socket or launches a browser.
* Residual (outside this workstream's edit scope): `fmd/adapters/nabh.py` and
  `fmd/acquire/tiers.py` still construct `httpx.post` directly. Both should
  migrate to `fmd.net.policy.make_client`/`request_with_retries`; the local OCR
  call should also take the single-slot lock. Docling artifact prefetch is
  bounded by HF's env knobs and the existing offline guard, not by this module.
