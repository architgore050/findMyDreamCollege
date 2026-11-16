# 09 — Local runbook (Docker)

The pipeline image installs the full `all` extra (crawl + extract + entity
resolution), so every stage runs inside the container. There is no host
virtualenv step and no "base image only" fallback.

Prerequisites: Docker Engine 24+ with Compose v2, ~8 GB free disk for images
(pipeline image is ~3.2 GB; web ~350 MB), and a network connection for the
initial image/dependency pull.

## 0. Configure (optional)

```bash
cp .env.example .env      # optional: every value has a working default
```

Compose works with **no** `.env` file. Credentials in `.env.example` are local
development defaults, not secrets; `.env` is gitignored.

## 1. Build

```bash
make build               # docker compose -f infra/docker-compose.yml build
```

Expected: two images, `findmydreamcollege/pipeline:local` and
`findmydreamcollege-web:latest`. First build takes several minutes (CPU torch,
docling, sentence-transformers); later source-only builds reuse the dependency
layer.

## 2. Start Postgres

```bash
docker compose -f infra/docker-compose.yml up -d db
docker compose -f infra/docker-compose.yml ps
```

Expected: `db ... Up (healthy)`, port `5433` on the host by default
(`${POSTGRES_PORT}`).

## 3. Migrate

```bash
make migrate
```

First run:

```
entrypoint: waiting for database (timeout 60s)
entrypoint: database is accepting connections
entrypoint: applying migrations
applied migrations: 0001_init, 0002_er_audit_and_replay, 0003_document_plans_and_pipeline_runs
entrypoint: exec: python -m fmd.cli migrate
database is up to date; no migrations applied
```

The entrypoint already migrates before executing the command, so the command
itself reports "up to date". That double run is intentional: it proves
idempotence (the migration runner takes a Postgres advisory lock, so api and
worker may migrate concurrently).

## 4. Seed the registry

```bash
make seed
```

Expected:

```
seeded registry: countries=13, sources=30, source_exclusions=4, exams=8
```

Counts come from `config/*.yaml`; they are not asserted in code. To have the
api entrypoint seed on boot instead, set `FMD_AUTO_SEED=1` (compose default is
`0` because `make seed` is the explicit step).

## 5. Plan a crawl without writing anything

```bash
make crawl ARGS="--dry-run --limit 5"
```

Expected: one JSON line with `"dry_run": true`, `"counts": {"planned": 5}`,
`"ok": true` and five `planned_urls`.

## 6. Run a real crawl (respects robots.txt and per-domain budgets)

```bash
make crawl ARGS="--source ugc_university_list --limit 3"
```

Expected (abridged):

```json
{"counts":{"deduped":2,"enqueued":3,"fetched":3,"plan.abstained":3,"stored":1},
 "dry_run":false,"failures":[],"ok":true,...}
```

`stored` is the number of new snapshots written under the `snapshots` volume;
`plan.abstained` with `family_undetected` is normal for listing pages whose
document family is genuinely ambiguous. `ok` is false only when an outcome is a
recorded failure (non-2xx, DNS error, circuit break), never for an honest
abstention.

## 7. API and web

```bash
docker compose -f infra/docker-compose.yml up -d api web
curl -s http://localhost:8000/api/health
```

Expected: the api container reaches `(healthy)` (its healthcheck parses
`/api/health` and requires `"status": "ok"`), and health returns:

```json
{"status":"ok","version":"0.1.0","database":"up","config":{...}}
```

Web: <http://localhost:3000>. `web` and `worker` wait for
`api: condition: service_healthy`, so they only start after migrations and the
config registry load have succeeded.

Discovery profile (optional SearXNG lead search, never evidence):

```bash
make discovery
```

Stop everything, keeping the database and snapshots:

```bash
make down
```

## 8. What runs where

| service | command | notes |
|---|---|---|
| `api` | `uvicorn fmd.api:app` | entrypoint waits for db + migrates; healthchecked |
| `worker` | `python -m fmd.cli worker` | same image; starts after api is healthy |
| one-shot | `docker compose ... run --rm api python -m fmd.cli <cmd>` | `make migrate/seed/crawl/extract/score` |
| `web` | Next standalone server | `NEXT_PUBLIC_API_URL` inlined at build time |

## Common failures and what they mean

### `TierUnavailable` for every source

The container was built from base dependencies only, so the crawl extras
(`curl_cffi`, `patchright`, `scrapy`, `protego`, `tldextract`) are missing and
the fetch ladder refuses to invent a tier. This image installs them from the
`all` extra. Verify and rebuild:

```bash
docker compose -f infra/docker-compose.yml run --rm --entrypoint python api -c \
  "import curl_cffi, patchright, scrapy, protego, tldextract, duckdb; print('extras ok')"
docker compose -f infra/docker-compose.yml build --no-cache api
```

Note: the patchright **browser binary** is not baked into the image (it would
add its own download step). If a site needs the JS tier, run
`patchright install chromium` inside the container or add it to the image;
httpx/curl_cffi tiers cover most registry sites.

### Robots abstention

`robots_disallowed` (or an `abstained` frontier state) is a *successful refusal*:
the site's `robots.txt` forbids the path. It is not a fetch bug and must not be
worked around (AGENTS invariant 9). Changing the path, waiting for the crawl
window, or using an officially published bulk file are the only acceptable
responses.

### `database is not ready` / `relation ... does not exist`

- Entrypoint timeout: Postgres is still starting. `docker compose ... up -d db`
  and watch `docker compose ... ps`; raise `FMD_DB_WAIT_SECONDS` on a slow disk.
- `relation "..." does not exist`: migrations were not applied, or the checkout
  is older than the code. Run `make migrate` and retry.
- Migration **checksum mismatch**: `db/migrations/*.sql` changed after it was
  applied to this volume. Migrations are forward-only; add a new numbered file
  instead of editing one. On a throwaway local volume, `make reset` rebuilds
  from scratch.
- Compose mounts `../db` read-only, so the migration files always match the
  checkout. Without that mount a stale baked copy can report "up to date" while
  columns added by the latest migration are missing.

### `Temporary failure in name resolution`

Docker copied the host's `resolv.conf`; hosts using systemd-resolved expose
`127.0.0.53`, which is unreachable from inside a container. Compose sets
`dns: 1.1.1.1, 8.8.8.8` for api/worker by default; override with
`FMD_DNS_1`/`FMD_DNS_2` if your network requires internal resolvers.

### api is `(unhealthy)` but responds on the port

The healthcheck requires `"status": "ok"`, which means the database is
reachable **and** the config registry loaded. A `"status": "degraded"` payload
names the failing half (`database`, `config_error`).
