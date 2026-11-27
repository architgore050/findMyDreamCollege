#!/usr/bin/env bash
# Visible long-command runner.
#
# Usage:
#   scripts/run_logged.sh /tmp/opencode/<task>/run.log <command> [args...]
#
# Records START + exact command + PID, a ps HEARTBEAT every 30s (override with
# RUN_LOGGED_HEARTBEAT_S), and END + exit code. A sibling <log>.status file is
# machine-readable. Command stdout/stderr goes to the log; this wrapper prints
# only START/END so harness output stays quiet.
set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <log-path> <command> [args...]" >&2
  exit 2
fi

LOG=$1
shift
printf -v CMD_DISPLAY '%q ' "$@"
HEARTBEAT_S=${RUN_LOGGED_HEARTBEAT_S:-30}
case "$HEARTBEAT_S" in
  ''|*[!0-9]*) echo "RUN_LOGGED_HEARTBEAT_S must be a positive integer" >&2; exit 2 ;;
esac
if [ "$HEARTBEAT_S" -lt 1 ]; then
  echo "RUN_LOGGED_HEARTBEAT_S must be >= 1" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")"
STATUS="${LOG}.status"
now() { date -u +%FT%TZ; }

{
  echo "[$(now)] START $CMD_DISPLAY"
  echo "[$(now)] WRAPPER_PID=$$"
} >>"$LOG"
printf 'RUNNING %s\n' "$CMD_DISPLAY" >"$STATUS"
echo "START $CMD_DISPLAY"
echo "log: $LOG"

"$@" >>"$LOG" 2>&1 &
CHILD_PID=$!
{
  echo "[$(now)] CHILD_PID=$CHILD_PID"
  echo "[$(now)] CMD $CMD_DISPLAY"
} >>"$LOG"

cleanup() {
  if kill -0 "$CHILD_PID" 2>/dev/null; then
    echo "[$(now)] WRAPPER_INTERRUPTED sending TERM to $CHILD_PID" >>"$LOG"
    kill -TERM "$CHILD_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

while kill -0 "$CHILD_PID" 2>/dev/null; do
  sleep "$HEARTBEAT_S"
  if kill -0 "$CHILD_PID" 2>/dev/null; then
    {
      echo "[$(now)] HEARTBEAT"
      ps -o pid,ppid,etime,%cpu,%mem,stat,cmd -p "$CHILD_PID" | sed 's/^/  /'
    } >>"$LOG"
  fi
done

wait "$CHILD_PID"
RC=$?
{
  echo "[$(now)] END rc=$RC $CMD_DISPLAY"
} >>"$LOG"
printf 'DONE rc=%s\n' "$RC" >"$STATUS"
echo "END rc=$RC"
echo "log: $LOG"
exit "$RC"
