#!/usr/bin/env bash
# Serve the bulk-extraction text model with llama.cpp `llama-server` (GGUF).
#
# Decision of record: docs/product-development-cycle/10-inference-and-model-selection.md
# ("Decision update — 2026-11-17 (locked: Gemma 4 26B-A4B / UD-IQ4_XS)") and
# 11-inference-runbook.md §2.7.
#
# Weights : unsloth/gemma-4-26B-A4B-it-GGUF, file
#           gemma-4-26B-A4B-it-UD-IQ4_XS.gguf (13,597,177,568 B = 12.66 GiB)
#           — the *raw* (non-QAT) model, chosen deliberately.
# Server  : llama.cpp, not vLLM. Gemma 4 MoE GGUFs have no vLLM GGUF path on
#           this host and llama.cpp is the only server that applies the chat
#           template and the constrained decoder the router needs.
# KV cache: 8-bit (q8_0/q8_0) + flash attention. Measured cost table lives in
#           docs/.../11-inference-runbook.md §2.7; the default below is the
#           measured safe context with >=1 GiB VRAM headroom.
#
#   scripts/serve_local.sh install  # fetch the llama-server binary if absent
#   scripts/serve_local.sh start    # background serve on 127.0.0.1:8010
#   scripts/serve_local.sh status   # pid + /health + /v1/models + GPU memory
#   scripts/serve_local.sh probe    # 3 assertions: text, no thought leak, guided JSON (+nullable)
#   scripts/serve_local.sh stop     # SIGTERM the process group
#   scripts/serve_local.sh logs     # tail the log
#   scripts/serve_local.sh fg       # foreground (Ctrl-C to stop)
#
# Environment (all optional):
#   FMD_MODEL_DIR     host weight root     (default ~/models)
#   FMD_VLLM_MODEL    .gguf file OR dir     (default
#                     $FMD_MODEL_DIR/gemma-4-26B-A4B-it-UD-IQ4_XS)
#                     Name kept for compatibility with .env; selects a GGUF.
#   FMD_SERVED_NAME   OpenAI model id       (default gemma-4-26b-a4b-gguf)
#   FMD_LLAMA_SERVER  llama-server binary   (default ~/.local/llama.cpp/bin/llama-server)
#   FMD_VLLM_PORT     port                  (default 8010)
#   FMD_LLAMA_HOST    bind address          (default 127.0.0.1)
#   FMD_MAX_MODEL_LEN context               (default: measured safe value)
#   FMD_KV_TYPE       KV cache type          (default q8_0; set f16 for parity tests)
#   FMD_SPEC          MTP speculative decode (default 0 = off; see docs §MTP)
#   FMD_DRAFT         MTP drafter .gguf      (only read when FMD_SPEC=1)
#   FMD_NGL           GPU layers             (default 999)
#
# Weights are host data: they live under FMD_MODEL_DIR and are never downloaded
# by an image build or a docker volume. This script refuses to start without
# them, so a restart can never trigger a multi-GB fetch.

set -euo pipefail

MODEL_DIR="${FMD_MODEL_DIR:-${MODEL_DIR:-$HOME/models}}"
MODEL_SPEC="${FMD_VLLM_MODEL:-$MODEL_DIR/gemma-4-26B-A4B-it-UD-IQ4_XS}"
SERVED="${FMD_SERVED_NAME:-gemma-4-26b-a4b-gguf}"
PORT="${FMD_VLLM_PORT:-8010}"
BIND="${FMD_LLAMA_HOST:-127.0.0.1}"
# Context default: 65536 (runbook §2.7). It clears the >=1 GiB VRAM headroom
# rule at the measured ~0.5 GiB CUDA-context+compute overhead, but the 64k row
# of that table is still a projection — run the sweep in §2.7 before a long
# extraction run, and drop to 32768 if it fails. The 26B's 12.66 GiB of weights,
# not the KV cache, is what bounds context here.
CTX="${FMD_MAX_MODEL_LEN:-65536}"
KV="${FMD_KV_TYPE:-q8_0}"
NGL="${FMD_NGL:-999}"
SPEC="${FMD_SPEC:-0}"
DRAFT="${FMD_DRAFT:-$MODEL_DIR/gemma-4-26B-A4B-it-UD-IQ4_XS/mtp-gemma-4-26B-A4B-it.gguf}"
RUN_DIR="${FMD_RUN_DIR:-/tmp/opencode/fmd-serve}"
PID_FILE="$RUN_DIR/llama-server.pid"
LOG_FILE="$RUN_DIR/llama-server.log"
BASE_URL="http://127.0.0.1:$PORT"

mkdir -p "$RUN_DIR"

# Resolve a .gguf from a file or directory. In the weights directory the mmproj
# (vision projector) and mtp-*.gguf (multi-token-prediction head) files are
# ballast for this profile: -np 1 text extraction neither needs nor wants them.
resolve_model() {
  local spec="$1"
  if [ -f "$spec" ]; then echo "$spec"; return 0; fi
  if [ -d "$spec" ]; then
    local f
    f=$(find "$spec" -maxdepth 1 -name '*.gguf' \
          ! -name 'mmproj*' ! -name 'mtp-*' ! -name '*MTP*' \
          -printf '%s %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$f" ] && { echo "$f"; return 0; }
  fi
  return 1
}

MODEL="$(resolve_model "$MODEL_SPEC" || true)"

llama_bin() {
  local b="${FMD_LLAMA_SERVER:-$HOME/.local/llama.cpp/bin/llama-server}"
  if [ -x "$b" ]; then echo "$b"; return 0; fi
  if command -v llama-server >/dev/null 2>&1; then command -v llama-server; return 0; fi
  return 1
}

die_no_weights() {
  echo "no GGUF weights at: $MODEL_SPEC" >&2
  echo "fetch once (detached; resumable; HF_INCLUDE picks the single quant):" >&2
  echo "  HF_REPO_ID=unsloth/gemma-4-26B-A4B-it-GGUF HF_REPO_SUBDIR= HF_REVISION=main \\" >&2
  echo "    HF_INCLUDE='gemma-4-26B-A4B-it-UD-IQ4_XS.gguf' \\" >&2
  echo "    HF_OUTPUT_DIR=\"$MODEL_DIR/gemma-4-26B-A4B-it-UD-IQ4_XS\" \\" >&2
  echo "    ~/.venvs/hf-downloader/bin/python -u scripts/hf_download.py" >&2
  echo "then point FMD_VLLM_MODEL at the .gguf (or its directory)." >&2
}

require_weights() {
  [ -n "${MODEL:-}" ] && [ -s "$MODEL" ] || { die_no_weights; exit 1; }
}

cmd_install() {
  "$(dirname "$0")/install_llama_cpp.sh" "${1:-install}"
}

serve_args() {
  require_weights
  local args=(
    -m "$MODEL"
    --alias "$SERVED"
    --host "$BIND" --port "$PORT"
    -ngl "$NGL"
    -c "$CTX"
    -fa on
    -ctk "$KV" -ctv "$KV"
    -b 2048 -ub 512
    -np 1
    --jinja
    -rea off
    --chat-template-kwargs '{"enable_thinking":false}'
  )
  # Do NOT pass --no-mmap: it no longer exists (b12468 renamed it --load-mode),
  # and forcing a 12.66 GiB read into RAM is wrong on this host anyway — model
  # loading is mmap by default, which lets the page cache evict, and -ngl 999
  # still puts every weight on the GPU.
  # MTP / speculative decoding. Default OFF: it is only enabled after the
  # greedy-equality check in the runbook passes for the exact drafter file.
  if [ "$SPEC" = "1" ]; then
    args+=( --model-draft "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 4 )
  fi
  printf '%s\n' "${args[@]}"
}

cmd_serve() {
  local bin; bin="$(llama_bin)" || {
    echo "llama-server not found — run: $0 install" >&2; exit 1; }
  local -a args
  mapfile -t args < <(serve_args)
  # --jinja: the Gemma 4 GGUF chat template must be applied by the server.
  # -rea off: llama.cpp would otherwise route the JSON answer into
  #           `reasoning_content` and leave `content` empty — the router reads
  #           `content` only (fmd/llm/providers.py), so this is mandatory.
  exec "$bin" "${args[@]}"
}

cmd_start() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running: pid $(cat "$PID_FILE")"; exit 0
  fi
  llama_bin >/dev/null || { echo "llama-server not found — run: $0 install" >&2; exit 1; }
  require_weights
  echo "serving $(basename "$MODEL") as '$SERVED' on $BIND:$PORT"
  echo "  ctx $CTX, kv $KV, ngl $NGL, spec $SPEC"
  echo "log: $LOG_FILE"
  setsid nohup bash -c "$(printf '%q ' "$0" fg)" >>"$LOG_FILE" 2>&1 < /dev/null &
  echo $! >"$PID_FILE"
  echo "pid $(cat "$PID_FILE") written to $PID_FILE"
  echo "wait for: curl -fsS $BASE_URL/health"
}

cmd_status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running: pid $(cat "$PID_FILE")"
  else
    echo "not running (no live pid in $PID_FILE)"
  fi
  echo "model: ${MODEL:-<unresolved: $MODEL_SPEC>}"
  echo "ctx:   $CTX (kv $KV, ngl $NGL, spec $SPEC)"
  echo -n "health: "; curl -fsS "$BASE_URL/health" 2>/dev/null || echo "(no /health)"
  echo
  echo -n "/v1/models: "
  curl -fsS "$BASE_URL/v1/models" 2>/dev/null | head -c 400 || echo "(no /v1/models)"
  echo
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
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

# Probe: run only after the weights are complete and the server is up.
# Mirrors the request the router actually sends (fmd/llm/providers.py):
#   POST /v1/chat/completions with `temperature` and, when guided_json is on,
#   response_format = {"type":"json_schema","json_schema":{name,schema,strict}}.
# Nullable properties matter: the router's schemas use ["string","null"] and
# llama.cpp compiles those to `alternative-0 | null`, which has its own failure
# modes — so the probe asserts a nullable field is *emitted as null*.
cmd_probe() {
  require_weights
  curl -fsS "$BASE_URL/health" >/dev/null 2>&1 || {
    echo "server not answering at $BASE_URL/health — run: $0 start" >&2; exit 1; }
  python3 - "$BASE_URL" "$SERVED" <<'PY'
import json, sys, urllib.error, urllib.request

base, served = sys.argv[1], sys.argv[2]
failed = 0

def post(payload, timeout=300):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def fail(name, msg):
    global failed
    failed += 1
    print(f"FAIL {name}: {msg}")

# (a) a plain chat completion returns text at all
content = ""
try:
    body = post({
        "model": served,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "temperature": 0,
        "max_tokens": 64,
    })
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    if content.strip():
        print(f"PASS a/text: {content.strip()[:60]!r}")
    else:
        rc = (msg.get("reasoning_content") or "")[:120]
        fail("a/text", f"empty content; reasoning_content={rc!r} (missing -rea off?)")
except Exception as exc:  # noqa: BLE001
    fail("a/text", f"request failed: {exc}")
    print("\nprobe aborted: server unusable")
    sys.exit(1)

# (b) no raw thought markers leaked into content
leaks = [m for m in ("<|channel>", "<|channel>thought", "<|think|>", "<|thought|>", "thought\n")
         if m in content]
if leaks:
    fail("b/no-thought-leak", f"content contains {leaks}")
else:
    print("PASS b/no-thought-leak: no channel/thought markers in content")

# (c) guided response_format with a small flat NULLABLE schema (the router's shape).
# Two calls: one where every optional field is present, one where the model must
# emit null, so the `alternative-0 | null` branch is exercised and not just compiled.
schema = {
    "type": "object",
    "properties": {
        "institute_name": {"type": ["string", "null"]},
        "seats": {"type": ["integer", "null"]},
        "is_recognised": {"type": ["boolean", "null"]},
    },
    "required": ["institute_name", "seats", "is_recognised"],
    "additionalProperties": False,
}
types = {"institute_name": (str, type(None)), "seats": (int, type(None)),
         "is_recognised": (bool, type(None))}

def guided(label, system, user, require_null=()):
    try:
        body = post({
            "model": served,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "InstituteFact", "schema": schema, "strict": True},
            },
        })
        raw = body["choices"][0]["message"].get("content") or ""
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("not a JSON object")
        errs = []
        if set(obj) != set(schema["properties"]):
            errs.append(f"keys={sorted(obj)}")
        for k, ok in types.items():
            if k in obj and not isinstance(obj[k], ok):
                errs.append(f"{k}={obj[k]!r} wrong type")
        if isinstance(obj.get("seats"), bool):
            errs.append("seats is a bool")
        for k in require_null:
            if obj.get(k, "missing") is not None:
                errs.append(f"{k} should be null, got {obj.get(k)!r}")
        if errs:
            fail(label, "; ".join(errs) + f" raw={raw[:200]!r}")
        else:
            print(f"PASS {label}: {json.dumps(obj, ensure_ascii=False)[:120]}")
    except urllib.error.HTTPError as exc:
        fail(label, f"HTTP {exc.code}: {exc.read().decode()[:300]}")
    except Exception as exc:  # noqa: BLE001
        fail(label, f"schema-invalid or unparsable: {exc}")

guided(
    "c/guided-json",
    "Extract the facts as JSON. Use null when a field is absent.",
    "Government Medical College, Nagpur reports 120 MBBS seats for 2026-27; "
    "NMC recognition is listed as valid.",
)
guided(
    "d/guided-json-nullable",
    "Extract the facts as JSON. Use null when a field is absent.",
    "A college brochure was uploaded. It contains no seat count and no recognition status.",
    require_null=("seats", "is_recognised"),
)

print("\nPROBE " + ("FAILED" if failed else "OK"))
sys.exit(1 if failed else 0)
PY
}

case "${1:-status}" in
  install) cmd_install "${2:-install}" ;;
  start)   cmd_start ;;
  fg)      cmd_serve ;;
  status)  cmd_status ;;
  stop)    cmd_stop ;;
  logs)    tail -n "${2:-80}" "$LOG_FILE" ;;
  probe)   cmd_probe ;;
  *) echo "usage: $0 {install|start|fg|status|probe|stop|logs}" >&2; exit 2 ;;
esac
