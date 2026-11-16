#!/usr/bin/env bash
# Serve the OCR stage (T2 / T2b) that `fmd.acquire.tiers` calls at
# $FMD_T2_URL/extract.
#
#   scripts/serve_ocr.sh vision-install  # create the vLLM venv if it is missing
#   scripts/serve_ocr.sh vision-start    # serve Qwen3-VL-8B on 0.0.0.0:8001
#   scripts/serve_ocr.sh vision-status   # pid + /v1/models + VRAM
#   scripts/serve_ocr.sh vision-stop
#
#   scripts/serve_ocr.sh build           # docker compose --profile ocr build ocr
#   scripts/serve_ocr.sh start           # up -d the OCR adapter (no polling)
#   scripts/serve_ocr.sh status          # container state + one /health curl
#   scripts/serve_ocr.sh logs
#   scripts/serve_ocr.sh stop            # stop + remove the adapter
#
#   scripts/serve_ocr.sh paddle-start    # T2b alternative engine (CPU on sm_120)
#
# Two processes on purpose. The weights live in vLLM on the host (CUDA wheels
# with sm_120 kernels); the container is a thin HTTP adapter plus pdfium. Neither
# is a systemd unit: both are developer processes.
#
# The service is optional by design. When it is down or unhealthy, /health says
# `unavailable` with a reason and /extract answers 503, so T2 raises
# TierUnavailable: the document goes to the human queue and NO claims are
# produced from it (config/acquire.yaml, fmd/acquire/tiers.py::_fallback).
#
# Model ids and the served name come from config/acquire.yaml (backends.T2
# .service). No model path is hardcoded here.
#
# Environment (all optional):
#   FMD_MODEL_DIR       host weight root              (default /home/archit-gore/models)
#   FMD_VLLM_OCR_MODEL  override the local dir to serve
#   FMD_OCR_SERVED_NAME override the served name (must match the config)
#   FMD_OCR_VLLM_PORT   vLLM port                     (default 8001)
#   FMD_VLLM_GPU_MEM_UTIL                             (default 0.85)
#   FMD_OCR_MAX_MODEL_LEN context                     (default 8192)
#   FMD_T2_URL          URL the pipeline calls        (default http://localhost:8110)
#   FMD_T2_PORT         published adapter port        (default 8110)
#
# Weights are host data under $FMD_MODEL_DIR, never a Hub fetch at serve time and
# never a docker volume. Prefetch once with
#   scripts/fetch_model.sh Qwen/Qwen3-VL-8B-Instruct-FP8

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/config/acquire.yaml"
COMPOSE=(docker compose -f "$REPO_ROOT/infra/docker-compose.yml")
SERVICE=ocr
PADDLE_SERVICE=ocr-paddle
URL="${FMD_T2_URL:-http://localhost:${FMD_T2_PORT:-8110}}"
VENV="${FMD_VLLM_VENV:-$HOME/.venvs/fmd-vllm}"
PORT="${FMD_OCR_VLLM_PORT:-8001}"
RUN_DIR="${FMD_RUN_DIR:-/tmp/opencode/fmd-serve}"
PID_FILE="$RUN_DIR/vllm-ocr.pid"
LOG_FILE="$RUN_DIR/vllm-ocr.log"

mkdir -p "$RUN_DIR"

# Read one key from backends.<tier>.service without a YAML parser: the registry
# is indented consistently and every key is unique inside its block.
config_value() {
  local tier="$1" key="$2"
  sed -n "/^  $tier:/,/^  T[0-9A-Za-z]*:/p" "$CONFIG" \
    | sed -n "s/^ *$key: *//p" | head -1
}

MODEL_DIR="${FMD_MODEL_DIR:-/home/archit-gore/models}"
HUB_DIR="$(config_value T2 hub_id)"
if [ -z "${FMD_VLLM_OCR_MODEL:-}" ]; then
  case "$HUB_DIR" in
    /*) MODEL="$HUB_DIR" ;;                 # explicit absolute path in config
    "") MODEL="" ;;
    *)  MODEL="$MODEL_DIR/$HUB_DIR" ;;      # directory name under the host root
  esac
else
  MODEL="$FMD_VLLM_OCR_MODEL"
fi
SERVED="${FMD_OCR_SERVED_NAME:-$(config_value T2 served_name)}"
GPU_UTIL="${FMD_VLLM_GPU_MEM_UTIL:-0.85}"
MAX_LEN="${FMD_OCR_MAX_MODEL_LEN:-8192}"

cmd_vision_install() {
  if [ ! -x "$VENV/bin/vllm" ]; then
    FMD_VLLM_VENV="$VENV" "$REPO_ROOT/scripts/serve_local.sh" install
  fi
  "$VENV/bin/python" -c \
    "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
}

cmd_vision_start() {
  [ -n "$MODEL" ] || { echo "no hub_id in $CONFIG backends.T2.service" >&2; exit 1; }
  [ -x "$VENV/bin/vllm" ] || { echo "no vLLM in $VENV — run: $0 vision-install" >&2; exit 1; }
  if [ ! -d "$MODEL" ]; then
    echo "no vision weights at $MODEL" >&2
    echo "prefetch once: scripts/fetch_model.sh Qwen/Qwen3-VL-8B-Instruct-FP8" >&2
    exit 1
  fi
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running: pid $(cat "$PID_FILE")"; exit 0
  fi
  echo "serving $MODEL as '$SERVED' on 0.0.0.0:$PORT (ctx $MAX_LEN)"
  echo "log: $LOG_FILE"
  setsid nohup "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "pid $(cat "$PID_FILE") written to $PID_FILE"
  echo "wait for: curl -s localhost:$PORT/v1/models"
}

cmd_vision_status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running: pid $(cat "$PID_FILE")"
  else
    echo "not running (no live pid in $PID_FILE)"
  fi
  curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 400 || echo "(no /v1/models yet)"
  echo
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
}

cmd_vision_stop() {
  if [ ! -f "$PID_FILE" ]; then echo "no pid file"; return 0; fi
  local pid; pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "stopped"
}

case "${1:-status}" in
  vision-install) cmd_vision_install ;;
  vision-start)   cmd_vision_start ;;
  vision-status)  cmd_vision_status ;;
  vision-stop)    cmd_vision_stop ;;
  vision-logs)    tail -n "${2:-80}" "$LOG_FILE" ;;
  build)  "${COMPOSE[@]}" --profile ocr build "$SERVICE" ;;
  start)
    "${COMPOSE[@]}" --profile ocr up -d "$SERVICE"
    # One check, no polling: until vLLM answers, /health is `unavailable` with a
    # reason and that is the correct state, not something to wait out silently.
    curl -sS "$URL/health" || echo "(no response from $URL/health)"
    echo
    ;;
  status)
    "${COMPOSE[@]}" --profile ocr ps "$SERVICE" || true
    curl -sS "$URL/health" || echo "(no response from $URL/health)"
    echo
    ;;
  stop)
    "${COMPOSE[@]}" --profile ocr stop "$SERVICE"
    "${COMPOSE[@]}" --profile ocr rm -f "$SERVICE"
    ;;
  logs)   "${COMPOSE[@]}" --profile ocr logs --tail="${2:-100}" -f "$SERVICE" ;;
  paddle-start)
    "${COMPOSE[@]}" --profile ocr-paddle up -d "$PADDLE_SERVICE"
    curl -sS "http://localhost:${FMD_T2B_PORT:-8111}/health" || true
    echo
    ;;
  paddle-stop)
    "${COMPOSE[@]}" --profile ocr-paddle stop "$PADDLE_SERVICE"
    "${COMPOSE[@]}" --profile ocr-paddle rm -f "$PADDLE_SERVICE"
    ;;
  *)
    echo "usage: $0 {vision-install|vision-start|vision-status|vision-stop|vision-logs|build|start|status|stop|logs|paddle-start|paddle-stop}" >&2
    exit 2
    ;;
esac
