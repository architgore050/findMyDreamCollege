#!/usr/bin/env bash
# Firecrawl Local lifecycle: start | stop | probe (v1/free-web-stack).
#
# FREE / SELF-HOSTED ONLY. Firecrawl Local is one Rust binary serving the
# Firecrawl v2 API on 127.0.0.1:3002 — no DB, no queue, no container. This
# script ALWAYS starts it with FIRECRAWL_CLOUD=off so nothing (search,
# blocked-page fallback) ever goes to Firecrawl Cloud; the Python client
# additionally fails closed on any cloud-sourced answer.
#
# Usage:
#   scripts/firecrawl_local.sh start   # install check + serve + health wait
#   scripts/firecrawl_local.sh probe   # GET /health on the local port
#   scripts/firecrawl_local.sh stop    # kill ONLY the PID this script started
#
# Install (never done automatically here; needs Rust/cargo):
#   cargo install --locked --version "${FIRECRAWL_LOCAL_VERSION:-0.1.1}" firecrawl-local
# 0.1.1 is the tested version (real-server matrix 2026-11-27: scrape/map/
# crawl/batch/PDF/DOCX all passed); override FIRECRAWL_LOCAL_VERSION deliberately.
# Config via env: FIRECRAWL_LOCAL_URL (default http://127.0.0.1:3002)

set -euo pipefail

STATE_DIR="${FIRECRAWL_LOCAL_STATE_DIR:-/tmp/opencode/firecrawl-local}"
PID_FILE="${STATE_DIR}/firecrawl-local.pid"
LOG_FILE="${STATE_DIR}/firecrawl-local.log"
LOCAL_URL="${FIRECRAWL_LOCAL_URL:-http://127.0.0.1:3002}"
HEALTH_PATH="/health"   # pinned: docs.firecrawl.dev/quickstarts/firecrawl-local
START_TIMEOUT_S="${FIRECRAWL_LOCAL_START_TIMEOUT_S:-30}"
# Tested version (2026-11-27 local real-server matrix). Env-overridable so an
# upgrade is a deliberate act, and the opt-in integration test
# (FMD_FIRECRAWL_INTEGRATION=1) fails loudly on contract drift.
FIRECRAWL_LOCAL_VERSION="${FIRECRAWL_LOCAL_VERSION:-0.1.1}"

no_cloud_banner() {
  echo "no-cloud guarantee: FIRECRAWL_CLOUD=off — nothing goes to Firecrawl Cloud;"
  echo "the client additionally fails closed on any firecrawl-cloud / blocked answer."
}

ensure_state_dir() {
  mkdir -p "${STATE_DIR}"
}

require_binary() {
  if ! command -v firecrawl-local >/dev/null 2>&1; then
    echo "firecrawl-local is not on PATH."
    echo "install it (never auto-installed): cargo install --locked --version \"${FIRECRAWL_LOCAL_VERSION}\" firecrawl-local"
    exit 1
  fi
}

is_running() {
  local pid
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

cmd_start() {
  require_binary
  ensure_state_dir
  if is_running; then
    echo "firecrawl-local already running (pid $(cat "${PID_FILE}"))"
    no_cloud_banner
    exit 0
  fi
  # FIRECRAWL_CLOUD=off is the whole point; unrecognized values are also "off"
  # upstream, but we pin the exact literal for auditability.
  echo "starting firecrawl-local serve on ${LOCAL_URL} (log: ${LOG_FILE})"
  nohup env FIRECRAWL_CLOUD=off firecrawl-local serve > "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  local waited=0
  while ! cmd_probe_quiet; do
    if ! is_running; then
      echo "firecrawl-local exited during startup; last log lines:" >&2
      tail -n 20 "${LOG_FILE}" >&2 || true
      rm -f "${PID_FILE}"
      exit 1
    fi
    if (( waited >= START_TIMEOUT_S )); then
      echo "health endpoint did not answer within ${START_TIMEOUT_S}s (pid kept: $(cat "${PID_FILE}"))" >&2
      exit 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "firecrawl-local is up: $(cat "${PID_FILE}")"
  no_cloud_banner
}

cmd_probe_quiet() {
  # Pinned health route from the Firecrawl Local quickstart. curl only; no
  # Python deps, so the probe works before the venv exists.
  curl -fsS --max-time 3 "${LOCAL_URL}${HEALTH_PATH}" >/dev/null 2>&1
}

cmd_probe() {
  if cmd_probe_quiet; then
    echo "firecrawl-local healthy: ${LOCAL_URL}${HEALTH_PATH}"
    no_cloud_banner
    return 0
  fi
  echo "firecrawl-local NOT reachable at ${LOCAL_URL}${HEALTH_PATH}"
  echo "start it with: scripts/firecrawl_local.sh start"
  return 1
}

cmd_stop() {
  if ! is_running; then
    echo "firecrawl-local is not running (no live pid file: ${PID_FILE})"
    exit 0
  fi
  # Kill ONLY the PID this script started — never a process-wide scan.
  local pid
  pid="$(cat "${PID_FILE}")"
  kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "pid ${pid} did not exit; sending SIGKILL"
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "firecrawl-local stopped (pid ${pid})"
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  probe) cmd_probe ;;
  *)
    echo "usage: $0 start|stop|probe" >&2
    exit 2
    ;;
esac
