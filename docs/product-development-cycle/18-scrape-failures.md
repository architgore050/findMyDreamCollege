# 18 — Scrape-failure log (what we lose to blocked/failed fetches)

Status: implemented 2026-11-23 (FMD-CLOUD-025, PR A). Scope: the fetch ladder's
single logging exit, the append-only JSONL log, and the `fmd failures` report.

The crawl measures what we *got*. Nothing measured what we *lost* — the good
university whose site robots-blocks us, the affiliation page that 404s, the
council portal that DNS-fails on hostel wifi. This log answers exactly one
question: **which institutions did the scraper lose out on, and why?** The
unit of loss is the institution, never a percentage: a site that blocks us
blocks everything we wanted from it.

## 1. What it is

Every failed fetch is appended as one JSON line to an append-only log.
`FetchLadder.fetch()` is the single logging exit (`_fetch_impl` stays
untouched below it), so every failure outcome is captured exactly once:

- transport exceptions mapped to the shared taxonomy (`13-network-robustness.md`),
- `robots_disallowed` refusals (invariant 9 — we honour robots),
- `domain_quarantined` politeness quarantine,
- `unsupported_replay` refusals,
- 404/410/4xx/5xx answers,
- `all_tiers_blocked` (every tier got a block signal),
- stage-level aborts (`NetworkUnavailable`, `TierUnavailable`) are logged, then
  re-raised unchanged.

Successes (200/304) are never logged. Logging is best-effort: `record_failure`
swallows every error, so the fetch path cannot break on logging (invariant 8:
append-only evidence — records are never mutated or deleted).

## 2. Storage

| knob | default |
|---|---|
| directory | `<repo>/data/scrape-failures` (override: `FMD_SCRAPE_FAILURES_DIR`) |
| files | `failures-YYYYMM.jsonl`, one per month, appended |
| committed | yes (`.gitignore` `!data/scrape-failures/**`) — the log is evidence |

One record (`fmd/fetch/failure_log.py:FetchFailure`): `ts` (ISO8601 UTC),
`url`, `final_url`, `status` (0 = no HTTP answer), `error`, `kind`, `tier`,
`robots_reason`, `method`, `elapsed_ms`, `institution`, `stage`. The
`Orchestrator` passes `stage="crawl"`; other callers may tag their own stage.

`institution` comes from `institution_hint(url)`: the **longest prefix match**
of the URL against every `base_url` in `config/sources.yaml` → that source's
`name`. A URL no source claims is `unresolved` — the log keeps the URL so the
gap stays visible instead of inventing an attribution.

## 3. Failure taxonomy (`kind`)

| kind | meaning | how to read it |
|---|---|---|
| `robots_disallowed` | explicit robots `Disallow` | a real policy block; the institution is off-limits until a human agreement |
| `http_404` / `http_410` | page gone | the link rotted; rediscover the page or drop the source entry |
| `http_error` | any other >= 400 | one bad endpoint or a moved path |
| `dns_failure`, `connect_timeout`, `connection_error`, `tls_error`, … | transport faults (full table: `13-network-robustness.md`) | flaky link or dead host; retry on a better connection |
| `all_tiers_blocked` | 403/429/503 from every tier | the site is fighting us; do not hammer it |
| `domain_quarantined` | politeness quarantine after timeouts | we caused the outage window; wait |
| `status_0`, `NetworkUnavailable`, `TierUnavailable` | no HTTP answer / stage abort | run-level problems, not site-level ones |

## 4. Reading the log

```
fmd failures                     # table, top 10 institutions
fmd failures --top 25            # more institutions
fmd failures --since 2026-11-01  # ISO8601 UTC cutoff (plain string compare on ts)
fmd failures --json              # machine-readable aggregate (same shape as below)
```

Table rows are sorted by failure count, then name; `unresolved` rows carry no
sample URLs because attribution failed, not because the fetch succeeded.

```json
{
  "total": 4,
  "by_institution": {
    "NIRF rankings (institution-level proxy only)": {
      "n": 2,
      "kinds": {"http_404": 1, "robots_disallowed": 1},
      "first_ts": "...", "last_ts": "...",
      "sample_urls": ["https://www.nirfindia.org/..."]
    }
  },
  "unresolved": {"n": 1, "kinds": {"dns_failure": 1}},
  "top_institutions": ["NIRF rankings (institution-level proxy only)"]
}
```

## 5. The "lost good universities" analysis

Read the report top-down:

1. **Top institutions by failure count** — each row is a university/source we
   repeatedly failed to fetch. High counts on one institution are a campaign,
   not noise: assign a human to unblock it (fix the URL, ask for permission,
   or find a mirror source in the registry).
2. **Kinds per institution** say what kind of unblocking:
   - `robots_disallowed` = a real block → only a human/policy fix helps;
   - `http_404`/`http_410` = the page is gone → rediscover or re-seed;
   - transport kinds = flaky network or dead host → retry later, check the host;
   - `all_tiers_blocked` = the site is fighting us → respect it (invariant 9)
     and look for an alternative source.
3. **`unresolved`** URLs are crawl-frontier discoveries outside the registry;
   promote the good ones into `config/sources.yaml` so future losses are
   attributed.

## 6. Join with the database

The log says we lost fetches; the DB says whether we lost *data we needed*.
Check coverage for a failing institution before spending effort on it:

```sql
-- Have we ever snapshotted anything from the failing host, and how stale is it?
SELECT d.domain, count(*) AS documents, max(s.fetched_at) AS last_snapshot
FROM documents d LEFT JOIN snapshots s ON s.document_id = d.id
WHERE d.domain = 'www.nirfindia.org'        -- host from a sample_url
GROUP BY d.domain;

-- Do we already hold programme rows for the failing institution?
SELECT e.canonical_name AS institution, count(p.id) AS programme_rows
FROM entities e LEFT JOIN programmes p ON p.institution_id = e.id
WHERE e.kind = 'institution'
  AND e.canonical_name ILIKE '%national assessment%'  -- paste `fmd failures` name
GROUP BY e.canonical_name;
```

An institution with zero programme rows **and** a wall of failures is the
highest-value next target: it is exactly the university our shortlist is
missing. An institution with rich rows and one stale 404 can wait.
