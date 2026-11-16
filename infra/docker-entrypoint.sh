#!/bin/sh
# Deterministic container startup for the api, worker and one-shot CLI runs.
#
#   1. wait until Postgres accepts connections (so `docker compose run` does not
#      race the db healthcheck when the caller skips depends_on);
#   2. apply migrations — idempotent, advisory-locked in fmd.db.migrations, so
#      api and worker may both run it concurrently;
#   3. optionally seed the config registry (FMD_AUTO_SEED=1, the compose default
#      for api only);
#   4. exec the requested command so signals reach uvicorn/the worker.
#
# Escape hatches: FMD_SKIP_MIGRATE=1 and FMD_AUTO_SEED=0.
set -eu

DATABASE_URL="${DATABASE_URL:-postgresql://fmd:fmd@db:5432/fmd}"
DB_WAIT_SECONDS="${FMD_DB_WAIT_SECONDS:-60}"

log() { printf '%s entrypoint: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

log "waiting for database (timeout ${DB_WAIT_SECONDS}s)"
if [ "${FMD_SKIP_DB_WAIT:-0}" = "1" ]; then
    log "database wait skipped (FMD_SKIP_DB_WAIT=1)"
else
python - "$DB_WAIT_SECONDS" <<'PY'
import os
import sys
import time

import psycopg

url = os.environ.get("DATABASE_URL", "")
deadline = time.monotonic() + float(sys.argv[1])
last = None
while time.monotonic() < deadline:
    try:
        with psycopg.connect(url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        print("entrypoint: database is accepting connections", file=sys.stderr)
        raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready yet"
        last = exc
        time.sleep(1)
print(f"entrypoint: database not ready after {sys.argv[1]}s: {last}", file=sys.stderr)
raise SystemExit(1)
PY
fi

if [ "${FMD_SKIP_MIGRATE:-0}" != "1" ]; then
    log "applying migrations"
    python -m fmd.cli migrate
fi

if [ "${FMD_AUTO_SEED:-0}" = "1" ]; then
    log "seeding config registry"
    python -m fmd.cli seed
fi

log "exec: $*"
exec "$@"
