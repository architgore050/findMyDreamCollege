# FMD-CLOUD-025-scrape-failure-logging · log every scraping failure · wave w14 · scrape-failures

~~~~
You are the one-shot cloud coding agent implementing scrape-failure logging (PR A).

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: f75bc11ad5316e92e23cd547d963462c80011105
Push to `v1/scrape-failures` immediately; commit + push after every logical step.
All output (code, comments, docs, commit messages, PR text, report) — English-only.

BRANCHING
1. `git switch -c v1/scrape-failures` from main at the base SHA; verify `git rev-parse HEAD`.
2. `git push origin v1/scrape-failures` before anything else.
3. Subagents (allowed) may branch off `v1/scrape-failures`; PRs into it. End: PR → main.

READ FIRST
- AGENTS.md (invariants; esp. #2 numbers, #4 append-only evidence)
- packages/pipeline/fmd/fetch/ladder.py (the choke point — read the whole file)
- packages/pipeline/fmd/fetch/types.py (FetchResult shape)
- packages/pipeline/fmd/fetch/robots.py (RobotsDecision shape)
- packages/pipeline/fmd/config/registry.py + config/sources.yaml (institution source entries with base_url/name)
- packages/pipeline/tests/test_fetch_ladder.py (FakeTier pattern)
- .gitignore (data/ exception style)

DESIGN — the log answers: "which good universities did the scraper lose out on, and why?"
Every scraping failure is appended to an append-only JSONL log; the `fmd failures` CLI
aggregates it BY INSTITUTION (the institution is the thing we lose — not a percentage).

1. `fmd/fetch/failure_log.py` (new module):
   - `FetchFailure` record: `ts` (ISO8601 UTC), `url`, `final_url` (opt), `status` (opt int),
     `error` (opt str), `kind` (str), `tier` (opt str), `robots_reason` (opt str),
     `method` (str, default "GET"), `elapsed_ms` (opt float), `institution` (opt str),
     `stage` (opt str).
   - `institution_hint(url) -> str | None` — longest-prefix match of the sources' `base_url`
     values (config registry) → returns the source `name`; cached; empty/None on no match.
   - `is_failure(result) -> bool` — error set OR status >= 400 OR result.quarantined.
     (200/304 are not failures; 404/410/400/401/500 etc. are failures.)
   - `failure_kind_for(result) -> str` — if `result.error`: first token before ":" (truncated);
     else "http_404" for 404, "http_410" for 410, "http_error" for other status >= 400,
     else "status_0".
   - `record_failure(failure) -> None` — best-effort append of one JSON line to
     `<failures_dir>/failures-YYYYMM.jsonl` (monthly files). Never raises
     (try/except everything; the fetch path must not break on logging). `failures_dir` =
     env `FMD_SCRAPE_FAILURES_DIR` or `<repo>/data/scrape-failures` (repo root resolved
     from __file__ like other code).
   - `read_failures(since: str | None = None) -> list[FetchFailure]` — read all month
     files sorted, filter ts >= since (ISO string compare).
   - `aggregate(failures, top_n: int = 10) -> dict` —
     `{"total": n, "by_institution": {name: {"n": c, "kinds": {kind: c}, "first_ts": s,
     "last_ts": s, "sample_urls": [...]}}, "unresolved": {"n": c, "kinds": {kind: c}},
     "top_institutions": [name, ...]}` — by_institution keys sorted by (-n, name), top_n
     truncated; "unresolved" = failures with institution None/empty.
2. `FetchLadder` (fmd/fetch/ladder.py):
   - `__init__` gains `stage: str | None = None` (record field; callers may pass it).
   - Wrap the public `fetch()`: rename the current body to `_fetch_impl(...)`; the new
     `fetch()` builds the request, calls `_fetch_impl`, then `_record_failure_if_any(result, url, start)`:
       if is_failure(result): record_failure(FetchFailure(ts=utcnow(), url=url,
         final_url=result.final_url, status=result.status, error=result.error,
         kind=failure_kind_for(result), tier=result.fetcher_tier,
         robots_reason=(result.robots.reason if result.robots else None),
         method=result.method or "GET", elapsed_ms=result.elapsed_ms,
         institution=institution_hint(url), stage=self.stage))
     (verify FetchResult.robots shape — robots.py — and FetchResult.final_url exists.)
   - This single wrapper covers ALL failure exits: transport exceptions, robots_disallowed,
     domain_quarantined, unsupported_replay, no_fetch_tiers, all_tiers_blocked, 404/410/4xx/5xx.
   - `Orchestrator` (fmd/orchestrate/orchestrator.py ~line 128): pass `FetchLadder(stage="crawl")`.
3. CLI — `fmd failures` subcommand (fmd/cli.py):
   - `_cmd_failures(args) -> int`: `failures = read_failures(since=args.since)`;
     `agg = aggregate(failures, top_n=args.top)`.
     `--json` → print json.dumps(agg, indent=2), return 0.
     Else human table:
       scrape failures: total=<n> since=<since or "all">
       <institution or (unresolved)>   <n>   kinds: <kind>:<c>, ...   sample: <url1>, <url2>
     sorted by (-n, institution).
   - Parser: add `fails_parser = subparsers.add_parser("failures", ...)` with
     `--since` (str, optional), `--top` (int, default 10), `--json` (action="store_true").
   - Dispatch: find the command dispatch in main(); add the `failures` branch.
4. `.gitignore`: add `!data/scrape-failures/` and `!data/scrape-failures/**`
   (append-only log, committed).
5. Tests (offline): `packages/pipeline/tests/test_fetch_failure_log.py` —
   - `test_record_and_read_roundtrip` (tmp dir via monkeypatched FMD_SCRAPE_FAILURES_DIR).
   - `test_is_failure_kind_matrix`: 200 → not failure; 304 → not failure; 404 result →
     is_failure, kind http_404; 410 → http_410; 500 → http_error; error "robots_disallowed"
     → kind robots_disallowed; error "dns_failure: ..." → kind dns_failure.
   - `test_institution_hint`: pick one real institution source with base_url from
     config/sources.yaml (agent chooses); assert hint == source name for a URL starting
     with base_url; assert None for a non-matching URL.
   - `test_aggregate_grouping`: synthetic records → by_institution counts/kinds; unresolved bucket.
   - Ladder (use the FakeTier pattern from test_fetch_ladder.py; verify constructor args the
     way existing tests do):
     - `test_ladder_logs_404_failure`: FakeTier returning status 404 (real institution URL) →
       after ladder.fetch: 1 record, kind http_404, institution non-None.
     - `test_ladder_logs_transport_failure`: FakeTier raising a generic Exception →
       record with status 0 and a kind (classify_exception).
     - `test_ladder_logs_robots_disallowed`: fake robots (allowed=False, reason=...) →
       record kind robots_disallowed.
     - `test_ladder_success_not_logged`: 200 → 0 records.
6. Docs: `docs/product-development-cycle/15-scrape-failures.md` —
   - What it is; storage; failure taxonomy; reading (`fmd failures [--json] --top N --since`);
     the "lost good universities" analysis: top institutions by failure count, kinds
     (robots_disallowed = real block; 404/410 = gone; transport = flaky; all_tiers_blocked =
     site fighting us); join with the DB: do we have rows for those institutions/programmes?
     (psql query example). Note: institution hint = longest prefix match on the source
     registry; unmatched URLs are "unresolved".
   - AGENTS.md: add one line under "Read in this order": `- docs/product-development-cycle/15-scrape-failures.md — reading the scraping-failure log (what we lose to blocked/failed fetches).`
7. Gates: `uv sync --all-extras; uv run ruff check .; uv run pytest -q | tail -2 (report counts); uv run python -m fmd.cli eval --golden ../../data/golden (provenance 1.0)`; smoke: `uv run python -m fmd.cli failures --json`.
8. Report (chat ≤ 400 words): what was built, test names + counts, gates, PR URL, top 3
   honest gaps (e.g., single-exit wrapper: failures during tier retries are not logged
   individually; 404 is logged as failure — operator judgment; institution hint is
   prefix-based).
~~~~
