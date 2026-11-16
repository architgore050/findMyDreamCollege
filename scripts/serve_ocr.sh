#!/usr/bin/env bash
# Serve the OCR stage (T2) that `fmd.acquire.tiers` calls at $FMD_T2_URL/extract.
#
#   scripts/serve_ocr.sh start    # docker compose --profile ocr up -d paddleocr
#   scripts/serve_ocr.sh build    # build the image only
#   scripts/serve_ocr.sh status   # container + /health
#   scripts/serve_ocr.sh stop     # stop and remove the container
#   scripts/serve_ocr.sh logs     # follow container logs
#
# The service is optional by design. When it is down or unhealthy, T2 raises
# TierUnavailable: the document is routed to the human queue and NO claims are
# produced from it (config/acquire.yaml, fmd/acquire/tiers.py::_fallback).
#
# Environment (all optional):
#   FMD_T2_URL   URL the pipeline calls (default http://localhost:8110)
#   FMD_T2_PORT  published port on the host (default 8110)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(docker compose -f "$REPO_ROOT/infra/docker-compose.yml" --profile ocr)
SERVICE=paddleocr
URL="${FMD_T2_URL:-http://localhost:${FMD_T2_PORT:-8110}}"

case "${1:-status}" in
  build)  "${COMPOSE[@]}" build "$SERVICE" ;;
  start)
    "${COMPOSE[@]}" up -d "$SERVICE"
    echo "waiting for $URL/health (first start downloads ~2 GB of weights)"
    for _ in $(seq 1 120); do
      if curl -fsS "$URL/health" 2>/dev/null | grep -q '"status": *"ok"'; then
        echo "ok: $URL/health"
        exit 0
      fi
      sleep 5
    done
    echo "not healthy after 10 min — see: $0 logs" >&2
    "${COMPOSE[@]}" ps "$SERVICE" || true
    exit 1
    ;;
  status)
    "${COMPOSE[@]}" ps "$SERVICE" || true
    curl -sS "$URL/health" || echo "(no response from $URL/health)"
    echo
    ;;
  stop)
    "${COMPOSE[@]}" stop "$SERVICE"
    "${COMPOSE[@]}" rm -f "$SERVICE"
    ;;
  logs)
    "${COMPOSE[@]}" logs --tail="${2:-100}" -f "$SERVICE"
    ;;
  *) echo "usage: $0 {build|start|status|stop|logs}" >&2; exit 2 ;;
esac
