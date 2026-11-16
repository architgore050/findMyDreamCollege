#!/usr/bin/env bash
# Prefetch a model ONCE into the host weight root. Nothing else downloads.
#
#   scripts/fetch_model.sh <hub-repo> [target-name]
#   scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ
#   scripts/fetch_model.sh Qwen/Qwen3-VL-8B-Instruct-FP8
#
# Why this exists: weights are host data under $FMD_MODEL_DIR (default
# /home/archit-gore/models), mounted read-only into containers. Image builds and
# docker volumes never hold weights, so rebuilding or pruning cannot force a
# re-download. This script is the only place a download is allowed, and it is
# run by a human on the host, never from a Dockerfile.
#
# Behaviour:
#   * compares every file against the Hub-reported size and resumes only what is
#     missing or short (`curl -C -`), so an interrupted fetch continues instead
#     of restarting 10 GB;
#   * a complete directory is left untouched (no re-download);
#   * writes a `.source` marker naming the repo it came from.
#
# Environment:
#   FMD_MODEL_DIR  host weight root (default /home/archit-gore/models)
#   HF_TOKEN       only needed for gated repos; never printed
#   HF_ENDPOINT    mirror (default https://huggingface.co)
#
# Deliberately curl + stdlib json, not `huggingface-cli`: the runtime venv must
# not grow a dependency, and per-file resume is the useful part.

set -euo pipefail

HUB="${1:-}"
NAME="${2:-}"
MODEL_DIR="${FMD_MODEL_DIR:-/home/archit-gore/models}"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

if [ -z "$HUB" ]; then
  echo "usage: $0 <hub-repo> [target-name]" >&2
  echo "example: $0 Qwen/Qwen3-VL-8B-Instruct-FP8" >&2
  exit 2
fi
[ -n "$NAME" ] || NAME="${HUB##*/}"
DEST="$MODEL_DIR/$NAME"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
mkdir -p "$DEST"

# Hugging Face /api/models lists every file with its byte size. Parsing it here
# means no filename list is maintained by hand and completeness is checkable.
python3 - "$ENDPOINT" "$HUB" >"$DEST/.files.tsv" <<'PY'
import json, sys, urllib.request

endpoint, hub = sys.argv[1], sys.argv[2]
req = urllib.request.Request(f"{endpoint}/api/models/{hub}")
token = __import__("os").environ.get("HF_TOKEN")
if token:
    req.add_header("Authorization", f"Bearer {token}")
with urllib.request.urlopen(req, timeout=60) as r:
    data = json.load(r)
for sib in data.get("siblings", []):
    name = sib.get("rfilename")
    size = sib.get("size")
    if name and isinstance(size, int):
        print(f"{size}\t{name}")
PY

if [ ! -s "$DEST/.files.tsv" ]; then
  echo "could not list $HUB (offline or gated)." >&2
  echo "Pass the file list instead, one '<bytes> <path>' per line on stdin." >&2
  echo "  $0 $HUB < files.tsv" >&2
  exit 1
fi

missing=0
while IFS=$'\t' read -r size path; do
  out="$DEST/$path"
  have=0
  [ -f "$out" ] && have="$(stat -c %s "$out")"
  if [ "$have" -lt "$size" ]; then
    missing=$((missing + 1))
    mkdir -p "$(dirname "$out")"
    echo "fetch $path ($((size / 1048576)) MiB, have $((have / 1048576)) MiB)"
    # -C - resumes an existing partial file; a fresh file starts at 0.
    curl -fsSL --retry 50 --retry-all-errors --retry-delay 2 -C - \
      -o "$out" "$ENDPOINT/$HUB/resolve/main/$path"
    after="$(stat -c %s "$out")"
    if [ "$after" -lt "$size" ]; then
      echo "short: $path is $after of $size bytes — re-run this script to resume" >&2
      exit 1
    fi
  else
    echo "have  $path"
  fi
done <"$DEST/.files.tsv"

printf '%s\n' "$HUB" >"$DEST/.source"
rm -f "$DEST/.files.tsv"
echo
echo "done: $DEST ($(du -sh "$DEST" | cut -f1))"
echo "served locally — no network fetch at serve time:"
echo "  text:   FMD_VLLM_MODEL=$DEST scripts/serve_local.sh start"
echo "  vision: FMD_VLLM_OCR_MODEL=$DEST scripts/serve_ocr.sh vision-start"
