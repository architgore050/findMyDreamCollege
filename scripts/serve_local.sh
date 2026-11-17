#!/usr/bin/env bash
# Serve the bulk-extraction text model (Qwen3.5-9B int4) for the fmd pipeline.
#
# Not in the repo venv on purpose: vLLM pins torch/triton versions that would
# fight `packages/pipeline`'s pip set. The venv lives at $FMD_VLLM_VENV
# (default ~/.venvs/fmd-vllm) and is created by `serve_local.sh install`.
#
#   scripts/serve_local.sh install   # create the venv and install vLLM
#   scripts/serve_local.sh start     # background serve on 0.0.0.0:8010
#   scripts/serve_local.sh status    # pid + /v1/models + nvidia-smi memory
#   scripts/serve_local.sh stop      # SIGTERM the process group
#   scripts/serve_local.sh logs      # tail the log
#   scripts/serve_local.sh fg        # foreground (Ctrl-C to stop)
#
# Environment (all optional):
#   FMD_MODEL_DIR   host weight root          (default /home/archit-gore/models)
#   FMD_VLLM_VENV   venv path                 (default ~/.venvs/fmd-vllm)
#   FMD_VLLM_MODEL  local dir or hub repo     (default $FMD_MODEL_DIR/Qwen3.5-9B-AWQ)
#   FMD_SERVED_NAME OpenAI model id           (default Qwen3.5-9B-AWQ)
#   FMD_VLLM_PORT   port                      (default 8010)
#   FMD_MAX_MODEL_LEN context                 (default 32768)
#   FMD_VLLM_GPU_MEM_UTIL                     (default 0.90)
#
# Weights are host data: they live under FMD_MODEL_DIR and are never downloaded
# by an image build or a docker volume. Passing the local directory (not the Hub
# id) is what guarantees a restart cannot trigger a re-download.

set -euo pipefail

MODEL_DIR="${FMD_MODEL_DIR:-/home/archit-gore/models}"
VENV="${FMD_VLLM_VENV:-$HOME/.venvs/fmd-vllm}"
MODEL="${FMD_VLLM_MODEL:-$MODEL_DIR/Qwen3.5-9B-AWQ}"
SERVED="${FMD_SERVED_NAME:-Qwen3.5-9B-AWQ}"
PORT="${FMD_VLLM_PORT:-8010}"
MAX_LEN="${FMD_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${FMD_VLLM_GPU_MEM_UTIL:-0.90}"
RUN_DIR="${FMD_RUN_DIR:-/tmp/opencode/fmd-serve}"
PID_FILE="$RUN_DIR/vllm.pid"
LOG_FILE="$RUN_DIR/vllm.log"

mkdir -p "$RUN_DIR"

vllm() { "$VENV/bin/vllm" "$@"; }

cmd_install() {
  command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
  uv venv "$VENV" --python 3.12
  VIRTUAL_ENV="$VENV" uv pip install vllm
  "$VENV/bin/python" -c "import torch, vllm; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  "$VENV/bin/python" -c "import torch; print('device cap:', torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))"
}

cmd_serve() {
  [ -x "$VENV/bin/vllm" ] || { echo "no vLLM in $VENV — run: $0 install" >&2; exit 1; }
  # A missing local path would make vLLM fall back to a network fetch. Refuse
  # instead: weights are host data, and this script never downloads.
  if [ ! -d "$MODEL" ]; then
    echo "no weights at $MODEL" >&2
    echo "prefetch once: scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ" >&2
    echo "or point FMD_VLLM_MODEL at an existing directory" >&2
    exit 1
  fi
  # Thinking is disabled in the served chat template, not in the client: the
  # router does not forward `thinking: false` yet (config/models.yaml comment).
  # `--default-chat-template-kwargs` is the vLLM 0.33 spelling; `--chat-template-kwargs`
  # no longer exists, and passing it aborts startup.
  exec "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --trust-remote-code
}

cmd_start() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running: pid $(cat "$PID_FILE")"; exit 0
  fi
  [ -d "$MODEL" ] || {
    echo "no weights at $MODEL — prefetch once: scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ" >&2
    exit 1
  }
  echo "serving $MODEL as '$SERVED' on 0.0.0.0:$PORT (ctx $MAX_LEN)"
  echo "log: $LOG_FILE"
  setsid nohup bash -c "$(printf '%q ' "$0" fg)" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "pid $(cat "$PID_FILE") written to $PID_FILE"
  echo "wait for: curl -s localhost:$PORT/v1/models"
}

cmd_status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running: pid $(cat "$PID_FILE")"
  else
    echo "not running (no live pid in $PID_FILE)"
  fi
  curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 400 || echo "(no /v1/models yet)"
  echo
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
}

cmd_stop() {
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
  install) cmd_install ;;
  start)   cmd_start ;;
  fg)      cmd_serve ;;
  status)  cmd_status ;;
  stop)    cmd_stop ;;
  logs)    tail -n "${2:-80}" "$LOG_FILE" ;;
  *) echo "usage: $0 {install|start|fg|status|stop|logs}" >&2; exit 2 ;;
esac
