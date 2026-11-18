#!/usr/bin/env bash
# scripts/acceptance.sh — end-to-end acceptance path (wave w1, e2e-acceptance).
#
# Strings the isolated components together and fails loudly on any seam they
# expose. Offline-safe and idempotent: it never writes product data (the only
# POSTs carry a synthetic profile that the API is contractually forbidden to
# persist), every absent dependency yields SKIP with a reason, and only a real
# contract violation produces FAIL — a SKIP is never a pass.
#
#   scripts/acceptance.sh                 # full run against whatever is up
#   API_BASE_URL=http://127.0.0.1:8000 scripts/acceptance.sh
#   DATABASE_URL=postgresql://... scripts/acceptance.sh
#   scripts/acceptance.sh --json-out reports/acceptance.json
#
# Sections (each check prints PASS / FAIL / SKIP; the table at the end is the
# verdict; exit is non-zero iff at least one check FAILED):
#   1. Preflight          DB reachable, migrations applied, registry seeded,
#                         model reachable (fmd llm doctor --live), serve_local probe
#   2. API contract sweep GET health/sources/programmes/programmes/{id},
#                         POST programmes/{id}/cost (synthetic profile,
#                         in-process only), POST programmes/{id}/filters,
#                         POST rankings (minimal profile), GET compare,
#                         POST exams, POST exclusions
#   3. Web gates          pnpm lint, pnpm build, pnpm test, smoke-render of
#                         /, /shortlist, /programmes/<id>, /compare, /exams,
#                         /exclusions, /sources
#   4. Privacy            the synthetic profile sentinel must appear in no
#                         response body, no API log line and no DB row
#
# Assertions live in packages/pipeline/fmd/acceptance.py so they are unit
# tested (tests/test_api_acceptance.py) without a live stack; this script only
# orchestrates and records.

set -uo pipefail

# ----------------------------------------------------------------- resolve repo

SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_SOURCE")/.." && pwd)"
PIPELINE_DIR="$REPO_ROOT/packages/pipeline"
WEB_DIR="$REPO_ROOT/apps/web"

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
WEB_BASE_URL="${WEB_BASE_URL:-http://127.0.0.1:3000}"
API_PORT="${API_PORT:-8765}"
WEB_PORT="${WEB_PORT:-3100}"
WEB_BUILD=0   # set to 1 when this run built the web app itself

JSON_OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --json-out)
      [ $# -ge 2 ] || { echo "--json-out requires a path" >&2; exit 2; }
      JSON_OUT="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,30p' "$SCRIPT_SOURCE" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fmd-acceptance.XXXXXX")"
API_PID=""
WEB_PID=""
API_STARTED_BY_US=0
WEB_STARTED_BY_US=0

cleanup() {
  [ "$WEB_STARTED_BY_US" -eq 1 ] && [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null
  [ "$API_STARTED_BY_US" -eq 1 ] && [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

# ----------------------------------------------------------------- tool discovery

UV_BIN=""
if command -v uv >/dev/null 2>&1; then UV_BIN="uv"; fi
PNPM_BIN=""
if command -v pnpm >/dev/null 2>&1; then PNPM_BIN="pnpm"; fi

# Python that can run fmd.acceptance (stdlib-only) and, when the pipeline env
# is synced, fmd.db (psycopg) for the DB preflight/privacy probes.
PY_RUNNER=()
PY_KIND=""
if [ -n "$UV_BIN" ]; then
  PY_RUNNER=("$UV_BIN" run --project "$PIPELINE_DIR" python)
  PY_KIND="uv"
elif [ -x "$PIPELINE_DIR/.venv/bin/python" ]; then
  PY_RUNNER=("$PIPELINE_DIR/.venv/bin/python")
  PY_KIND="venv"
elif command -v python3 >/dev/null 2>&1; then
  PY_RUNNER=(python3)
  PY_KIND="system"
fi

PSYCHODS_AVAILABLE=0
if [ "${PY_KIND}" != "" ]; then
  if "${PY_RUNNER[@]}" -c "import psycopg" >/dev/null 2>&1; then
    PSYCHODS_AVAILABLE=1
  fi
fi

run_check() {
  # run_check <check-name> [extra args...]  — JSON body on stdin
  local check="$1"; shift
  if [ "${PY_KIND}" = "system" ]; then
    PYTHONPATH="$PIPELINE_DIR" "${PY_RUNNER[@]}" -m fmd.acceptance "$check" "$@"
  else
    "${PY_RUNNER[@]}" -m fmd.acceptance "$check" "$@"
  fi
}

# ----------------------------------------------------------------- recording

CHECK_NAMES=()
CHECK_RESULTS=()
CHECK_DETAILS=()
FAIL_COUNT=0
SKIP_COUNT=0
PASS_COUNT=0

record() {
  CHECK_NAMES+=("$1"); CHECK_RESULTS+=("$2"); CHECK_DETAILS+=("$3")
  case "$2" in
    PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
    FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)); echo "FAIL: $1 — $3" >&2 ;;
    SKIP) SKIP_COUNT=$((SKIP_COUNT + 1)); echo "SKIP: $1 — $3" >&2 ;;
  esac
}

note() { echo "acceptance: $*"; }

# ----------------------------------------------------------------- http helpers

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$1" 2>/dev/null; }

fetch_to() {
  # fetch_to <url> <outfile> — echoes the HTTP status, body goes to outfile
  curl -s -o "$2" -w '%{http_code}' --max-time 15 "$1" 2>/dev/null
}

post_to() {
  # post_to <url> <outfile> <bodyfile> — echoes the HTTP status
  curl -s -o "$2" -w '%{http_code}' --max-time 20 -H 'content-type: application/json' \
    --data-binary "@$3" "$1" 2>/dev/null
}

wait_for() {
  # wait_for <url> <seconds> — 0 once the URL answers with any HTTP status
  local url="$1" deadline=$((SECONDS + $2))
  while [ $SECONDS -lt $deadline ]; do
    [ "$(http_code "$url")" != "000" ] && return 0
    sleep 1
  done
  return 1
}

json_field() {
  # json_field <json-file> <python-expr over d> — empty string when absent
  "${PY_RUNNER[@]}" - "$1" "$2" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        d = json.load(fh)
except Exception:
    sys.exit(0)
expr = sys.argv[2]
try:
    value = eval(expr, {"d": d})  # noqa: S307 — fixed expressions from this script
    if value is not None:
        print(value)
except Exception:
    pass
PYEOF
}

# ----------------------------------------------------------------- db helpers

db_query() {
  # db_query <sql> — prints the first column of the first row, "" when down
  [ "$PSYCHODS_AVAILABLE" -eq 1 ] || return 1
  "${PY_RUNNER[@]}" - "$1" <<'PYEOF'
import os, sys
import psycopg
url = os.environ.get("DATABASE_URL", "postgresql://fmd:fmd@localhost:5433/fmd")
try:
    with psycopg.connect(url, connect_timeout=3) as conn:
        row = conn.execute(sys.argv[1]).fetchone()
        if row is not None and row[0] is not None:
            print(row[0])
except Exception:
    pass
PYEOF
}

# =================================================================== 1. PREFLIGHT

note "section 1: preflight"

DB_UP=0
if [ "$PSYCHODS_AVAILABLE" -ne 1 ]; then
  record "preflight/db-reachable" "SKIP" \
    "no python environment with psycopg (install uv or run: uv sync --all-extras)"
else
  if [ "$(db_query "SELECT 1")" = "1" ]; then
    DB_UP=1
    record "preflight/db-reachable" "PASS" "$(db_query "SELECT current_database()") via DATABASE_URL (LIVE)"
  else
    record "preflight/db-reachable" "SKIP" \
      "no database at DATABASE_URL (start it: docker compose -f infra/docker-compose.yml up -d db)"
  fi
fi

if [ "$DB_UP" -eq 1 ]; then
  applied="$(db_query "SELECT count(*) FROM schema_migrations")"
  expected="$(ls "$REPO_ROOT"/db/migrations/*.sql 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$applied" = "$expected" ] && [ -n "$applied" ]; then
    record "preflight/migrations" "PASS" "$applied of $expected migrations applied (LIVE)"
  else
    record "preflight/migrations" "FAIL" \
      "schema_migrations has ${applied:-0} rows but db/migrations holds $expected files — run: fmd migrate"
  fi
else
  record "preflight/migrations" "SKIP" "database not reachable"
fi

if [ "$DB_UP" -eq 1 ]; then
  sources_n="$(db_query "SELECT count(*) FROM sources")"
  countries_n="$(db_query "SELECT count(*) FROM countries")"
  if [ -n "$sources_n" ] && [ "$sources_n" -gt 0 ] && [ -n "$countries_n" ] && [ "$countries_n" -gt 0 ]; then
    record "preflight/registry-seeded" "PASS" "sources=$sources_n countries=$countries_n (LIVE)"
  else
    record "preflight/registry-seeded" "FAIL" \
      "registry tables are empty — run: fmd migrate && fmd seed"
  fi
else
  record "preflight/registry-seeded" "SKIP" "database not reachable"
fi

if [ -n "$UV_BIN" ]; then
  if (cd "$REPO_ROOT" && "$UV_BIN" run --project "$PIPELINE_DIR" python -m fmd.cli llm doctor --live) \
      >"$WORK_DIR/llm-doctor.txt" 2>&1; then
    record "preflight/model-reachable" "PASS" "fmd llm doctor --live: all required profiles usable (LIVE)"
  else
    record "preflight/model-reachable" "SKIP" \
      "fmd llm doctor --live reported required profiles unusable (see $WORK_DIR/llm-doctor.txt; a local model server is optional for the API/web path)"
  fi
else
  record "preflight/model-reachable" "SKIP" "uv not installed; cannot run fmd llm doctor --live"
fi

if [ -x "$REPO_ROOT/scripts/serve_local.sh" ]; then
  if (cd "$REPO_ROOT" && scripts/serve_local.sh probe) >"$WORK_DIR/serve-probe.txt" 2>&1; then
    record "preflight/serve-local-probe" "PASS" "scripts/serve_local.sh probe: text, no-thought-leak and guided-JSON assertions hold (LIVE)"
  else
    record "preflight/serve-local-probe" "SKIP" \
      "serve_local probe failed — GGUF weights absent or llama-server down (optional for the API/web path; see $WORK_DIR/serve-probe.txt)"
  fi
else
  record "preflight/serve-local-probe" "SKIP" "scripts/serve_local.sh not found"
fi

# =================================================================== 2. API SWEEP

note "section 2: API contract sweep"

API_UP=0
health_code="$(http_code "$API_BASE_URL/api/health")"
if [ "$health_code" = "200" ]; then
  API_UP=1
  record "api/server" "PASS" "API answering at $API_BASE_URL (operator-provided)"
elif [ -n "$UV_BIN" ]; then
  note "no API at $API_BASE_URL — starting uvicorn on port $API_PORT"
  API_LOG="$WORK_DIR/uvicorn.log"
  (cd "$REPO_ROOT" && "$UV_BIN" run --project "$PIPELINE_DIR" \
    uvicorn fmd.api:app --host 127.0.0.1 --port "$API_PORT") >"$API_LOG" 2>&1 &
  API_PID=$!
  API_STARTED_BY_US=1
  API_BASE_URL="http://127.0.0.1:$API_PORT"
  if wait_for "$API_BASE_URL/api/health" 30; then
    API_UP=1
    record "api/server" "PASS" "API started for the sweep on port $API_PORT (log: $API_LOG)"
  else
    record "api/server" "SKIP" \
      "uvicorn did not come up in 30s — run it manually: uv run --project packages/pipeline uvicorn fmd.api:app --port 8000 (log: $API_LOG)"
  fi
else
  record "api/server" "SKIP" \
    "no API at $API_BASE_URL and no uv to start one — run: uv run --project packages/pipeline uvicorn fmd.api:app --port 8000"
fi

PROGRAMME_ID=""
SECOND_PROGRAMME_ID=""
if [ "$API_UP" -eq 1 ]; then
  status="$(fetch_to "$API_BASE_URL/api/programmes?limit=10" "$WORK_DIR/programmes.json")"
  if [ "$status" = "200" ]; then
    PROGRAMME_ID="$(json_field "$WORK_DIR/programmes.json" "d['items'][0]['id'] if d.get('items') else None")"
    SECOND_PROGRAMME_ID="$(json_field "$WORK_DIR/programmes.json" "d['items'][1]['id'] if len(d.get('items', [])) > 1 else None")"
  fi
fi

api_get() {
  # api_get <label> <path> <check...> — envelope + invariant assertions on a GET
  local label="$1" path="$2"; shift 2
  local body="$WORK_DIR/api-${label}.json"
  local status
  status="$(fetch_to "$API_BASE_URL$path" "$body")"
  if [ "$status" = "000" ]; then
    record "api/$label" "SKIP" "API not reachable"
    return
  fi
  if [ "$status" = "503" ]; then
    record "api/$label" "SKIP" "API answered 503 database_unavailable — start Postgres, run fmd migrate && fmd seed"
    return
  fi
  if [ "$status" != "200" ]; then
    record "api/$label" "FAIL" "HTTP $status (expected 200): $(head -c 200 "$body" 2>/dev/null)"
    return
  fi
  local problems=0
  if ! run_check envelope "$body" >"$WORK_DIR/${label}-envelope.txt" 2>&1; then
    record "api/$label" "FAIL" "envelope: $(head -3 "$WORK_DIR/${label}-envelope.txt" | tr '\n' ' ')"
    problems=1
  fi
  local check_name
  for check_name in "$@"; do
    if ! run_check "$check_name" "$body" >"$WORK_DIR/${label}-${check_name}.txt" 2>&1; then
      record "api/$label" "FAIL" "$check_name: $(head -3 "$WORK_DIR/${label}-${check_name}.txt" | tr '\n' ' ')"
      problems=1
    fi
  done
  [ "$problems" -eq 0 ] && record "api/$label" "PASS" "GET $path → 200, envelope + $(echo "$*" | tr ' ' ',') clean"
}

api_post() {
  # api_post <label> <path> <body-file> <check...> — same for POSTs
  local label="$1" path="$2" body_file="$3"; shift 3
  local body="$WORK_DIR/api-${label}.json"
  local status
  status="$(post_to "$API_BASE_URL$path" "$body" "$body_file")"
  if [ "$status" = "000" ]; then
    record "api/$label" "SKIP" "API not reachable"
    return
  fi
  if [ "$status" = "503" ]; then
    record "api/$label" "SKIP" "API answered 503 database_unavailable — start Postgres, run fmd migrate && fmd seed"
    return
  fi
  if [ "$status" != "200" ]; then
    record "api/$label" "FAIL" "HTTP $status (expected 200): $(head -c 200 "$body" 2>/dev/null)"
    return
  fi
  local problems=0
  if ! run_check envelope "$body" >"$WORK_DIR/${label}-envelope.txt" 2>&1; then
    record "api/$label" "FAIL" "envelope: $(head -3 "$WORK_DIR/${label}-envelope.txt" | tr '\n' ' ')"
    problems=1
  fi
  local check_name
  for check_name in "$@"; do
    if ! run_check "$check_name" "$body" >"$WORK_DIR/${label}-${check_name}.txt" 2>&1; then
      record "api/$label" "FAIL" "$check_name: $(head -3 "$WORK_DIR/${label}-${check_name}.txt" | tr '\n' ' ')"
      problems=1
    fi
  done
  [ "$problems" -eq 0 ] && record "api/$label" "PASS" "POST $path → 200, envelope + $(echo "$*" | tr ' ' ',') clean"
}

if [ "$API_UP" -eq 1 ]; then
  api_get "health" "/api/health"
  api_get "sources" "/api/sources" unknown
  api_get "programmes" "/api/programmes?limit=10" unknown
  if [ -n "$PROGRAMME_ID" ]; then
    api_get "programme-detail" "/api/programmes/$PROGRAMME_ID" unknown money
  else
    record "api/programme-detail" "SKIP" \
      "no programmes in the database (crawl+inventory pending) — run: fmd migrate && fmd seed, then make crawl"
  fi
else
  for label in health sources programmes programme-detail; do
    record "api/$label" "SKIP" "API not reachable"
  done
fi

# Synthetic profile — built here, sent in POST bodies only. The sentinel is a
# per-run string that must never reappear in a response body, log or DB row.
SENTINEL="fmd-acceptance-sentinel-${RANDOM}-${SECONDS}"
MINIMAL_PROFILE_FILE="$WORK_DIR/profile-minimal.json"
SYNTH_PROFILE_FILE="$WORK_DIR/profile-synthetic.json"
cat >"$MINIMAL_PROFILE_FILE" <<EOF
{"profile": {"nationality": "IN", "target_intake_year": 2027, "target_field": "mlt"}}
EOF
cat >"$SYNTH_PROFILE_FILE" <<EOF
{
  "profile": {
    "nationality": "IN",
    "target_intake_year": 2027,
    "target_field": "mlt",
    "travel_for_exams": "no",
    "regions": ["$SENTINEL"],
    "acceptance_marker": "$SENTINEL",
    "financing": {
      "loan_required": false,
      "collateral_available": false,
      "annual_family_income": 600000
    }
  }
}
EOF

if [ "$API_UP" -eq 1 ] && [ -n "$PROGRAMME_ID" ]; then
  api_post "programme-cost" "/api/programmes/$PROGRAMME_ID/cost" "$SYNTH_PROFILE_FILE" \
    unknown money cost-profile
  if ! run_check cost-profile "$WORK_DIR/api-programme-cost.json" \
      --loan-required false --family-income 600000 \
      >"$WORK_DIR/cost-profile-args.txt" 2>&1; then
    record "api/programme-cost" "FAIL" \
      "profile ignored: $(head -3 "$WORK_DIR/cost-profile-args.txt" | tr '\n' ' ')"
  fi
  api_post "programme-filters" "/api/programmes/$PROGRAMME_ID/filters" "$SYNTH_PROFILE_FILE" filters
else
  record "api/programme-cost" "SKIP" \
    "needs the API + at least one programme row (synthetic profile is POSTed in-process only)"
  record "api/programme-filters" "SKIP" \
    "needs the API + at least one programme row"
fi

if [ "$API_UP" -eq 1 ]; then
  api_post "rankings" "/api/rankings" "$MINIMAL_PROFILE_FILE" unknown money filters
  api_post "exams" "/api/exams" "$SYNTH_PROFILE_FILE" unknown
  api_post "exclusions" "/api/exclusions" "$SYNTH_PROFILE_FILE" filters
  if [ -n "$PROGRAMME_ID" ] && [ -n "$SECOND_PROGRAMME_ID" ]; then
    api_get "compare" "/api/compare?ids=$PROGRAMME_ID,$SECOND_PROGRAMME_ID" unknown money
  else
    record "api/compare" "SKIP" "needs at least two programme rows (found ${PROGRAMME_ID:-0})"
  fi
else
  for label in rankings exams exclusions compare; do
    record "api/$label" "SKIP" "API not reachable"
  done
fi

# =================================================================== 3. WEB GATES

note "section 3: web gates"

if [ -z "$PNPM_BIN" ]; then
  record "web/pnpm" "SKIP" "pnpm not installed — run: corepack enable pnpm"
else
  if [ ! -d "$WEB_DIR/node_modules" ]; then
    note "installing web dependencies (pnpm install --frozen-lockfile)"
    (cd "$WEB_DIR" && "$PNPM_BIN" install --frozen-lockfile) >"$WORK_DIR/pnpm-install.log" 2>&1 \
      || record "web/pnpm-install" "FAIL" "pnpm install --frozen-lockfile failed (see $WORK_DIR/pnpm-install.log)"
  fi
  if (cd "$WEB_DIR" && "$PNPM_BIN" lint) >"$WORK_DIR/pnpm-lint.log" 2>&1; then
    record "web/lint" "PASS" "pnpm lint"
  else
    record "web/lint" "FAIL" "pnpm lint failed (see $WORK_DIR/pnpm-lint.log)"
  fi
  # NEXT_PUBLIC_API_URL is inlined into the client bundle at build time.
  if (cd "$WEB_DIR" && NEXT_PUBLIC_API_URL="$API_BASE_URL" "$PNPM_BIN" build) \
      >"$WORK_DIR/pnpm-build.log" 2>&1; then
    WEB_BUILD=1
    record "web/build" "PASS" "pnpm build"
  else
    record "web/build" "FAIL" "pnpm build failed (see $WORK_DIR/pnpm-build.log)"
  fi
  if (cd "$WEB_DIR" && "$PNPM_BIN" test) >"$WORK_DIR/pnpm-test.log" 2>&1; then
    record "web/test" "PASS" "pnpm test (vitest)"
  else
    record "web/test" "FAIL" "pnpm test failed (see $WORK_DIR/pnpm-test.log)"
  fi

  WEB_UP=0
  if [ "$WEB_BUILD" -eq 1 ]; then
    note "starting built web app on port $WEB_PORT (API: $API_BASE_URL)"
    (cd "$WEB_DIR" && NEXT_PUBLIC_API_URL="$API_BASE_URL" "$PNPM_BIN" start --port "$WEB_PORT") \
      >"$WORK_DIR/next-start.log" 2>&1 &
    WEB_PID=$!
    WEB_STARTED_BY_US=1
    WEB_BASE_URL="http://127.0.0.1:$WEB_PORT"
    if wait_for "$WEB_BASE_URL/" 60; then
      WEB_UP=1
      record "web/server" "PASS" "next start answering on port $WEB_PORT"
    else
      record "web/server" "SKIP" "next start did not answer in 60s (see $WORK_DIR/next-start.log)"
    fi
  else
    record "web/server" "SKIP" "web app not built (build failed or skipped)"
  fi

  if [ "$WEB_UP" -eq 1 ]; then
    routes=("/" "/shortlist" "/compare" "/exams" "/exclusions" "/sources")
    if [ -n "$PROGRAMME_ID" ]; then routes+=("/programmes/$PROGRAMME_ID"); fi
    for route in "${routes[@]}"; do
      slug="$(echo "$route" | sed 's|/|-|g;s|^-||;s|-$||')"
      [ -z "$slug" ] && slug="root"
      page="$WORK_DIR/page-${slug}.html"
      status="$(fetch_to "$WEB_BASE_URL$route" "$page")"
      if [ "$status" != "200" ]; then
        record "web/render $route" "FAIL" "HTTP $status (expected 200)"
        continue
      fi
      if ! run_check html "$page" >"$WORK_DIR/html-${slug}.txt" 2>&1; then
        record "web/render $route" "FAIL" \
          "artifact in rendered HTML: $(head -3 "$WORK_DIR/html-${slug}.txt" | tr '\n' ' ')"
        continue
      fi
      record "web/render $route" "PASS" "200, no raw NaN/undefined, no bare composite decimal"
    done
  else
    for route in "/" "/shortlist" "/programmes/<id>" "/compare" "/exams" "/exclusions" "/sources"; do
      record "web/render $route" "SKIP" "web server not running"
    done
  fi
fi

# =================================================================== 4. PRIVACY

note "section 4: privacy (profile never persisted)"

LEAKS=0
for body in "$WORK_DIR"/api-*.json "$WORK_DIR"/page-*.html; do
  [ -f "$body" ] || continue
  if ! run_check privacy --sentinel "$SENTINEL" "$body" >"$WORK_DIR/privacy-$(basename "$body").txt" 2>&1; then
    record "privacy/response-bodies" "FAIL" \
      "sentinel leaked into $(basename "$body")"
    LEAKS=1
  fi
done
if [ "$LEAKS" -eq 0 ] && compgen -G "$WORK_DIR/api-*.json" >/dev/null; then
  record "privacy/response-bodies" "PASS" \
    "sentinel absent from every captured API response and rendered page"
elif [ "$LEAKS" -eq 0 ]; then
  record "privacy/response-bodies" "SKIP" "no API responses captured (API down)"
fi

if [ "$API_STARTED_BY_US" -eq 1 ] && [ -f "$WORK_DIR/uvicorn.log" ]; then
  if grep -q "$SENTINEL" "$WORK_DIR/uvicorn.log"; then
    record "privacy/api-logs" "FAIL" "sentinel found in the API server log — profile data is being logged"
  else
    record "privacy/api-logs" "PASS" "sentinel absent from API server log"
  fi
else
  record "privacy/api-logs" "SKIP" "API log only available when this script started the server"
fi

if [ "$DB_UP" -eq 1 ] && [ "$PSYCHODS_AVAILABLE" -eq 1 ] && compgen -G "$WORK_DIR/api-*.json" >/dev/null; then
  if "${PY_RUNNER[@]}" - "$SENTINEL" <<'PYEOF'
import os, sys
import psycopg
sentinel = sys.argv[1]
url = os.environ.get("DATABASE_URL", "postgresql://fmd:fmd@localhost:5433/fmd")
with psycopg.connect(url, connect_timeout=3) as conn:
    tables = [
        row[0]
        for row in conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'
            """
        ).fetchall()
    ]
    for table in tables:
        hit = conn.execute(
            f'SELECT count(*) FROM "{table}" WHERE "{table}"::text LIKE %s',
            (f"%{sentinel}%",),
        ).fetchone()
        if int(hit[0]) > 0:
            print(f"sentinel persisted in table {table}")
            sys.exit(1)
PYEOF
  then
    record "privacy/db-rows" "PASS" "sentinel absent from every table (read-only scan)"
  else
    record "privacy/db-rows" "FAIL" "sentinel persisted in the database — profile data is being written"
  fi
else
  record "privacy/db-rows" "SKIP" "needs DB + API sweep outputs"
fi

# =================================================================== verdict

echo
echo "========================== ACCEPTANCE =========================="
printf '%-38s %-6s %s\n' "CHECK" "RESULT" "DETAIL"
printf '%s\n' "----------------------------------------------------------------"
for i in "${!CHECK_NAMES[@]}"; do
  printf '%-38s %-6s %s\n' "${CHECK_NAMES[$i]}" "${CHECK_RESULTS[$i]}" "${CHECK_DETAILS[$i]}"
done
printf '%s\n' "----------------------------------------------------------------"
printf '%-38s %-6s %s\n' "TOTAL" \
  "$PASS_COUNT passed" "$FAIL_COUNT failed, $SKIP_COUNT skipped"

if [ -n "$JSON_OUT" ]; then
  names_nl="$(printf '%s\n' "${CHECK_NAMES[@]}")"
  results_nl="$(printf '%s\n' "${CHECK_RESULTS[@]}")"
  details_nl="$(printf '%s\n' "${CHECK_DETAILS[@]}")"
  FMD_ACCEPTANCE_NAMES="$names_nl" \
  FMD_ACCEPTANCE_RESULTS="$results_nl" \
  FMD_ACCEPTANCE_DETAILS="$details_nl" \
  "${PY_RUNNER[@]}" - "$JSON_OUT" "$PASS_COUNT" "$FAIL_COUNT" "$SKIP_COUNT" "$SENTINEL" <<'PYEOF'
import json, os, sys
from datetime import datetime, timezone

path = sys.argv[1]
names = [n for n in os.environ.get("FMD_ACCEPTANCE_NAMES", "").split("\n") if n]
results = [r for r in os.environ.get("FMD_ACCEPTANCE_RESULTS", "").split("\n") if r]
details = [d for d in os.environ.get("FMD_ACCEPTANCE_DETAILS", "").split("\n")]
report = {
    "format": "fmd-acceptance-v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "sentinel": sys.argv[5],
    "totals": {
        "passed": int(sys.argv[2]),
        "failed": int(sys.argv[3]),
        "skipped": int(sys.argv[4]),
    },
    "checks": [
        {"name": name, "result": result, "detail": detail}
        for name, result, detail in zip(names, results, details)
    ],
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print(f"acceptance: report written to {path}")
PYEOF
fi

echo
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "VERDICT: FAIL — $FAIL_COUNT check(s) violated the contract."
  exit 1
fi
echo "VERDICT: PASS — no contract violations; $SKIP_COUNT dependency absent (each SKIP is a known gap, never a pass)."
exit 0
