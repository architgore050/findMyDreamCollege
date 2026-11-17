#!/usr/bin/env bash
# Offline test for scripts/fetch_model.sh.
#
# Serves a fake Hub (metadata + index + two safetensors shards) from localhost,
# interrupts a slow transfer to prove resume, truncates and bit-flips shards to
# prove verify/repair, and asserts the documented exit codes. No internet.
#
#   bash tests/test_fetch_model.sh
#
# Exit 0 = all assertions passed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/fetch_model.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/fmd-fetch-test.XXXXXX")"
WWW="$WORK/www"
LOG="$WORK/server.log"
MODELS="$WORK/models"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_eq() {
  [ "$1" = "$2" ] || fail "$3 (expected '$2', got '$1')"
}

assert_contains() {
  case "$1" in
    *"$2"*) ;;
    *) fail "$3 (missing '$2')" ;;
  esac
}

[ -x "$SCRIPT" ] || fail "$SCRIPT is not executable"

# ---- fixture -------------------------------------------------------------

mkdir -p "$WWW"
python3 - "$WWW" <<'PY'
import hashlib
import json
import os
import sys

root = sys.argv[1]
repo = os.path.join(root, "Org", "Repo", "resolve", "main")
os.makedirs(repo, exist_ok=True)


def safetensors(path, payload_len, seed):
    payload = bytes((seed + i) % 251 for i in range(payload_len))
    header = {
        "__metadata__": {"format": "pt"},
        "weight": {"dtype": "U8", "shape": [payload_len], "data_offsets": [0, payload_len]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    blob = len(raw).to_bytes(8, "little") + raw + payload
    with open(path, "wb") as fh:
        fh.write(blob)
    return hashlib.sha256(blob).hexdigest(), len(blob)


shards = [
    ("model-00001-of-00002.safetensors", 3 * 1024 * 1024, 7),
    ("model-00002-of-00002.safetensors", 2 * 1024 * 1024, 11),
]
weight_map = {}
siblings = []
total = 0
for i, (name, size, seed) in enumerate(shards, start=1):
    sha, actual = safetensors(os.path.join(repo, name), size, seed)
    weight_map[f"w{i}"] = name
    siblings.append({"rfilename": name, "size": actual, "lfs": {"sha256": sha}})
    total += size

index = {
    "metadata": {"total_size": total},
    "weight_map": weight_map,
}
index_path = os.path.join(repo, "model.safetensors.index.json")
with open(index_path, "w", encoding="utf-8") as fh:
    json.dump(index, fh)
siblings.append({"rfilename": "model.safetensors.index.json", "size": os.path.getsize(index_path)})

config = b'{\n  "model_type": "qwen3",\n  "architectures": ["Qwen3ForCausalLM"]\n}\n'
config_path = os.path.join(repo, "config.json")
with open(config_path, "wb") as fh:
    fh.write(config)
siblings.append({"rfilename": "config.json", "size": os.path.getsize(config_path)})

api = os.path.join(root, "api", "models", "Org", "Repo", "revision")
os.makedirs(api, exist_ok=True)
with open(os.path.join(api, "main"), "w", encoding="utf-8") as fh:
    json.dump({"id": "Org/Repo", "siblings": siblings}, fh)

# The tree listing is what fetch_model.sh treats as authoritative: a file that
# is not here does not exist in the repo and is therefore optional.
tree = []
for sibling in siblings:
    entry = {"type": "file", "path": sibling["rfilename"], "size": sibling["size"]}
    if "lfs" in sibling:
        entry["lfs"] = {"oid": sibling["lfs"]["sha256"], "size": sibling["size"]}
    tree.append(entry)
tree_dir = os.path.join(root, "api", "models", "Org", "Repo", "tree")
os.makedirs(tree_dir, exist_ok=True)
with open(os.path.join(tree_dir, "main"), "w", encoding="utf-8") as fh:
    json.dump(tree, fh)

# A directory entry must be ignored by the normaliser, not treated as a file.
with open(os.path.join(tree_dir, "main"), "r+", encoding="utf-8") as fh:
    entries = json.load(fh)
    entries.append({"type": "directory", "path": "subdir", "size": 0})
    fh.seek(0)
    json.dump(entries, fh)
    fh.truncate()

# Raw LFS pointer endpoints are only needed when metadata lacks sha256; provide
# them so the fallback path is exercisable, not required.
raw = os.path.join(root, "Org", "Repo", "raw", "main")
os.makedirs(raw, exist_ok=True)
for sibling in siblings:
    if not sibling["rfilename"].endswith(".safetensors"):
        continue
    with open(os.path.join(raw, sibling["rfilename"]), "w", encoding="utf-8") as fh:
        fh.write(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{sibling['lfs']['sha256']}\n"
            f"size {sibling['size']}\n"
        )
PY

# ---- fixture server ------------------------------------------------------

cat >"$WORK/server.py" <<'PY'
import http.server
import os
import sys
import time

root, port, logpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self._serve(head=True)

    def do_GET(self):
        self._serve(head=False)

    def _serve(self, head):
        rel = self.path.split("?", 1)[0].lstrip("/")
        target = os.path.normpath(os.path.join(root, rel))
        if not target.startswith(root + os.sep) or not os.path.isfile(target):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        with open(target, "rb") as fh:
            data = fh.read()
        size = len(data)
        rng = self.headers.get("Range")
        with open(logpath, "a", encoding="utf-8") as fh:
            fh.write(f"{rel}\t{rng or '-'}\n")

        norange = os.path.exists(os.path.join(root, ".norange"))
        start, end, status = 0, size - 1, 200
        if rng and rng.startswith("bytes=") and not norange:
            spec = rng[6:].split(",")[0]
            first, _, last = spec.partition("-")
            try:
                if first == "":
                    start = max(0, size - int(last))
                else:
                    start = int(first)
                    if last:
                        end = min(size - 1, int(last))
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206

        body = data[start:end + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return

        slow = os.path.exists(os.path.join(root, ".slow")) and rel.endswith(".safetensors")
        drop = os.path.exists(os.path.join(root, ".drop")) and rel.endswith(".safetensors")
        limit = 65536 if drop else len(body)
        sent = 0
        try:
            while sent < len(body) and sent < limit:
                chunk = body[sent:sent + 16384]
                self.wfile.write(chunk)
                self.wfile.flush()
                sent += len(chunk)
                if slow:
                    time.sleep(0.025)
        except (BrokenPipeError, ConnectionResetError):
            pass
        if drop:
            self.close_connection = True


server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
server.serve_forever()
PY

PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
BASE="http://127.0.0.1:$PORT"
python3 "$WORK/server.py" "$WWW" "$PORT" "$LOG" &
SERVER_PID=$!

for _ in $(seq 1 50); do
  if python3 -c "import socket,sys; s=socket.create_connection(('127.0.0.1',$PORT),0.2); s.close()" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
kill -0 "$SERVER_PID" 2>/dev/null || fail "fixture server did not start"

COMMON=(HF_ENDPOINT="$BASE" FMD_MODEL_DIR="$MODELS" FMD_CURL_RETRY=2
  FMD_FETCH_ATTEMPTS=4 FMD_BACKOFF_BASE=1 FMD_BACKOFF_MAX=2
  FMD_CONNECT_TIMEOUT=5 FMD_SPEED_TIME=10)

fetch() { env "${COMMON[@]}" "$SCRIPT" "$@"; }
fetch_rc() { set +e; fetch "$@" >"$WORK/out.log" 2>&1; RC=$?; set -e; }

run_fetch() {
  local expected=$1
  shift
  fetch_rc "$@" || true
  if [ "$RC" -ne "$expected" ]; then
    cat "$WORK/out.log" >&2
    fail "fetch $* exited $RC, expected $expected"
  fi
}

SHARD1="model-00001-of-00002.safetensors"
SHARD2="model-00002-of-00002.safetensors"
DIR="$MODELS/Repo"
PART1="$DIR/$SHARD1.part"

# ---- 1. interrupted transfer, then resume -------------------------------

touch "$WWW/.slow"
set -m
env "${COMMON[@]}" "$SCRIPT" Org/Repo >"$WORK/fetch1.log" 2>&1 &
P1=$!
set +m

for _ in $(seq 1 200); do
  if [ -f "$PART1" ] && [ "$(stat -c %s "$PART1" 2>/dev/null || echo 0)" -gt 131072 ]; then
    break
  fi
  sleep 0.05
done
[ -f "$PART1" ] || { cat "$WORK/fetch1.log" >&2; fail "no .part appeared before interrupt"; }
PARTIAL="$(stat -c %s "$PART1")"
[ "$PARTIAL" -gt 0 ] || fail "partial transfer is empty"
kill -TERM -- "-$P1" 2>/dev/null || kill -TERM "$P1" 2>/dev/null || true
wait "$P1" 2>/dev/null || true
rm -f "$WWW/.slow"

[ -f "$PART1" ] || fail "partial file vanished after interrupt"
PARTIAL="$(stat -c %s "$PART1")"
[ "$PARTIAL" -lt $((3 * 1024 * 1024 + 4096)) ] || fail "interrupted download already complete"

run_fetch 0 Org/Repo
[ ! -f "$PART1" ] || fail "completed shard still has a .part file"
assert_eq "$(cat "$DIR/.source")" "Org/Repo" ".source marker"

grep -F "$SHARD1" "$LOG" | grep -Fq "bytes=$PARTIAL-" ||
  fail "resume did not send Range from byte $PARTIAL"

grep -Fq '"integrity": "sha256"' "$DIR/.fmd-manifest.json" ||
  fail "manifest did not record sha256 integrity"
[ ! -f "$DIR/.fmd-files.tsv" ] || fail "temporary file list was left behind"

fetch_rc verify "$DIR"
assert_eq "$RC" 0 "verify on a complete directory"
assert_eq "$(grep -c '^PASS ' "$WORK/out.log")" "4" "verify PASS count"
grep -Fq "tensors=1" "$WORK/out.log" || fail "verify did not report tensor counts"

fetch_rc verify "$DIR"
[ "$RC" -eq 0 ] || fail "second verify should still pass"

# ---- 2. truncation is caught, repair fetches only the bad shard ---------

BEFORE1="$(grep -cF "$SHARD1" "$LOG")"
python3 - "$DIR/$SHARD2" <<'PY'
import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)
with open(path, "r+b") as fh:
    fh.truncate(size - 4096)
PY

fetch_rc verify "$DIR"
assert_eq "$RC" 2 "verify must fail on a truncated shard"
grep -Fq "FAIL $SHARD2: expected" "$WORK/out.log" ||
  fail "verify did not name the truncated shard with expected bytes"

fetch_rc repair "$DIR"
assert_eq "$RC" 0 "repair of a truncated shard"
assert_eq "$(grep -cF "$SHARD1" "$LOG")" "$BEFORE1" "repair must not re-fetch a valid shard"

# ---- 3. same-size corruption is caught by sha256 ------------------------

python3 - "$DIR/$SHARD2" <<'PY'
import os
import sys

path = sys.argv[1]
with open(path, "r+b") as fh:
    fh.seek(-1, os.SEEK_END)
    byte = fh.read(1)
    fh.seek(-1, os.SEEK_END)
    fh.write(bytes([byte[0] ^ 0xFF]))
PY

fetch_rc verify "$DIR"
assert_eq "$RC" 2 "verify must fail on a same-size bit flip"
grep -Fq "sha256 mismatch" "$WORK/out.log" || fail "bit flip was not blamed on sha256"

fetch_rc repair "$DIR"
assert_eq "$RC" 0 "repair of a bit-flipped shard"

# ---- 3b. repair an old directory with no manifest and no .source --------

rm -f "$DIR/.fmd-manifest.json" "$DIR/.source"
python3 - "$DIR/$SHARD2" <<'PY'
import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)
with open(path, "r+b") as fh:
    fh.truncate(size - 2048)
PY

set +e
env FMD_MODEL_REPO=Org/Repo "${COMMON[@]}" "$SCRIPT" repair "$DIR" >"$WORK/out.log" 2>&1
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  cat "$WORK/out.log" >&2
  fail "repair without a manifest or .source exited $RC"
fi
[ -s "$DIR/.fmd-manifest.json" ] || fail "repair did not rebuild the manifest"
assert_eq "$(cat "$DIR/.source")" "Org/Repo" "repair restored .source"

fetch_rc verify "$DIR"
assert_eq "$RC" 0 "verify after manifest-less repair"

# ---- 4. server that ignores Range (full-body restart) -------------------

MODELS_ALT="$WORK/models-norange"
mkdir -p "$MODELS_ALT/Repo"
head -c 1000 "$DIR/$SHARD1" >"$MODELS_ALT/Repo/$SHARD1.part"
touch "$WWW/.norange"
set +e
env HF_ENDPOINT="$BASE" FMD_MODEL_DIR="$MODELS_ALT" FMD_CURL_RETRY=1 \
  FMD_FETCH_ATTEMPTS=3 FMD_BACKOFF_BASE=1 FMD_BACKOFF_MAX=1 \
  "$SCRIPT" Org/Repo >"$WORK/norange.log" 2>&1
RC=$?
set -e
rm -f "$WWW/.norange"
assert_eq "$RC" 0 "fetch against a server that ignores Range"
cmp -s "$MODELS_ALT/Repo/$SHARD1" "$DIR/$SHARD1" ||
  fail "Range-ignoring server produced a corrupt shard"
[ ! -f "$MODELS_ALT/Repo/$SHARD1.part" ] || fail "leftover .part after full-body restart"

# ---- 5. attempt cap: a dying server fails loudly, names the shard --------

touch "$WWW/.drop"
set +e
env HF_ENDPOINT="$BASE" FMD_MODEL_DIR="$WORK/models-drop" FMD_CURL_RETRY=0 \
  FMD_FETCH_ATTEMPTS=1 FMD_BACKOFF_BASE=1 FMD_BACKOFF_MAX=1 FMD_SPEED_TIME=10 \
  "$SCRIPT" Org/Repo >"$WORK/drop.log" 2>&1
RC=$?
set -e
rm -f "$WWW/.drop"
assert_eq "$RC" 2 "attempt cap must exit 2"
grep -Fq "FAIL $SHARD1" "$WORK/drop.log" || fail "attempt cap did not name the failed shard"
grep -Fq "exhausted 1 attempts" "$WORK/drop.log" || fail "attempt cap message missing"

# ---- 6. exit 3 when the disk cannot hold the remaining bytes ------------

set +e
env HF_ENDPOINT="$BASE" FMD_MODEL_DIR="$WORK/models-disk" FMD_CURL_RETRY=1 \
  FMD_FETCH_ATTEMPTS=1 FMD_DISK_RESERVE_BYTES=999999999999999 \
  "$SCRIPT" Org/Repo >"$WORK/disk.log" 2>&1
RC=$?
set -e
assert_eq "$RC" 3 "no-disk-space exit code"
grep -Fq "disk: refusing to start" "$WORK/disk.log" || fail "disk message missing"

# ---- 7. exit 4 when metadata and index are unavailable ------------------

set +e
env HF_ENDPOINT="http://127.0.0.1:1" FMD_MODEL_DIR="$WORK/models-meta" \
  FMD_CURL_RETRY=0 FMD_FETCH_ATTEMPTS=1 FMD_CONNECT_TIMEOUT=2 \
  "$SCRIPT" Org/Repo >"$WORK/meta.log" 2>&1
RC=$?
set -e
assert_eq "$RC" 4 "metadata-unavailable exit code"

# ---- 7b. files absent from the Hub listing are optional, never failures --
# The operator's real failure: a stale manifest named video_preprocessor_config
# .json and vocab.json, the repo no longer listed them, and every run reported
# "missing". A file absent from the tree listing must not fail verification, must
# not be fetched, and must never be quarantined.

python3 - "$DIR" <<'PY'
import json
import os
import sys

directory = sys.argv[1]
path = os.path.join(directory, ".fmd-manifest.json")
with open(path, encoding="utf-8") as fh:
    manifest = json.load(fh)
for item in manifest["files"]:
    item.pop("required", None)          # simulate a pre-optional manifest
manifest["files"] += [
    {"path": "video_preprocessor_config.json", "size": 385, "sha256": None, "kind": "aux"},
    {"path": "vocab.json", "size": 6722759, "sha256": None, "kind": "aux"},
]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh)
PY

# An operator's local copy of an unlisted file is innocent: leave it alone.
printf 'local bytes that must survive\n' >"$DIR/vocab.json"

fetch_rc verify "$DIR"
assert_eq "$RC" 0 "verify must ignore files the Hub does not list"
grep -Fq "OPTIONAL vocab.json" "$WORK/out.log" ||
  fail "absent-but-local file was not reported optional"
grep -Fq "OPTIONAL video_preprocessor_config.json" "$WORK/out.log" ||
  fail "absent file was not reported optional"
grep -Fq "FAIL " "$WORK/out.log" &&
  fail "an absent optional file produced a FAIL line"
assert_eq "$(cat "$DIR/vocab.json")" "local bytes that must survive" \
  "verify must not touch an unlisted local file"

# fetch refreshes a pre-optional manifest and demotes the stale entries.
run_fetch 0 Org/Repo
grep -Fq '"required": false' "$DIR/.fmd-manifest.json" ||
  fail "stale manifest entries were not demoted to optional"
assert_eq "$(cat "$DIR/vocab.json")" "local bytes that must survive" \
  "fetch must not touch an unlisted local file"
[ ! -d "$DIR/.bad" ] || {
  find "$DIR/.bad" -type f | grep -Fq 'vocab.json' &&
    fail "unlisted local file was quarantined"
}

fetch_rc repair "$DIR"
assert_eq "$RC" 0 "repair with optional files present"
[ -f "$DIR/vocab.json" ] || fail "repair removed an unlisted local file"
grep -Fq "nothing to repair" "$WORK/out.log" ||
  fail "repair refetched healthy files after an optional-file change"

# ---- 7c. corrupt files are quarantined with a reason, never deleted ------

python3 - "$DIR/$SHARD2" <<'PY'
import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)
with open(path, "r+b") as fh:
    fh.truncate(size - 1024)
PY

fetch_rc repair "$DIR"
assert_eq "$RC" 0 "repair of a truncated shard (quarantine path)"
BAD="$(find "$DIR/.bad" -type f -name "$SHARD2" | head -n 1)"
[ -n "$BAD" ] || fail "corrupt shard was not moved to .bad/"
[ -s "$BAD.reason" ] || fail "quarantined shard has no .reason file"
grep -Fq "expected" "$BAD.reason" || fail ".reason does not say why the shard was bad"
[ ! -f "$DIR/$SHARD2.part" ] || fail "quarantined shard left a .part behind"
find "$DIR/.bad" -maxdepth 1 -mindepth 1 -type d | grep -Eq '/([0-9]{8}T[0-9]{6}Z)$' ||
  fail ".bad/ subdirectory is not a UTC timestamp"

# ---- 7d. speed probe measures throughput and recommends, offline-safe ----

fetch_rc speed Org/Repo
assert_eq "$RC" 0 "speed probe against the fixture"
grep -Fq "fastest:" "$WORK/out.log" || fail "speed did not print a recommendation"
grep -Fq "curl -4 single stream" "$WORK/out.log" ||
  fail "speed did not report the single-stream baseline"
grep -Fq "MiB/s" "$WORK/out.log" || fail "speed did not report MiB/s"
grep -Fq "ETA for" "$WORK/out.log" || fail "speed did not report an ETA"
grep -Eq 'use:' "$WORK/out.log" || fail "speed did not print the follow-up command"

# ---- 7e. multi-connection path (aria2c stand-in) and token fallback ------

ARIA_BIN="$WORK/bin"
ARIA_LOG="$WORK/aria2.urls"
mkdir -p "$ARIA_BIN"
cat >"$ARIA_BIN/aria2c" <<'SH'
#!/usr/bin/env bash
# Stand-in for aria2c: enough of the CLI to prove the dispatch and resume wiring.
dir=. out= url=
while [ "$#" -gt 0 ]; do
  case "$1" in
    -d) dir=$2; shift 2 ;;
    -o) out=$2; shift 2 ;;
    -x | -s | -k | --timeout | --connect-timeout | --lowest-speed-limit | \
      --summary-interval | --max-tries | --retry-wait) shift 2 ;;
    -*) shift ;;
    *) url=$1; shift ;;
  esac
done
printf '%s\n' "$url" >>"${ARIA_LOG:-/dev/null}"
mkdir -p "$dir"
exec curl -sS -L -C - --fail -o "$dir/$out" "$url"
SH
chmod +x "$ARIA_BIN/aria2c"

set +e
env HF_ENDPOINT="$BASE" FMD_MODEL_DIR="$WORK/models-aria" FMD_DOWNLOADER=aria2 \
  FMD_CURL_RETRY=1 FMD_FETCH_ATTEMPTS=2 PATH="$ARIA_BIN:$PATH" ARIA_LOG="$ARIA_LOG" \
  "$SCRIPT" Org/Repo >"$WORK/aria.log" 2>&1
RC=$?
set -e
assert_eq "$RC" 0 "fetch through the aria2 backend"
[ -s "$ARIA_LOG" ] || fail "aria2 backend was not used when FMD_DOWNLOADER=aria2"
grep -Fq "aria2c -x" "$WORK/aria.log" || fail "aria2 selection was not announced"
cmp -s "$WORK/models-aria/Repo/$SHARD1" "$DIR/$SHARD1" ||
  fail "aria2 path produced a corrupt shard"

# aria2 has no --header=@file: a token must force the curl path, never argv.
: >"$ARIA_LOG"
set +e
env HF_ENDPOINT="$BASE" FMD_MODEL_DIR="$WORK/models-aria-token" FMD_DOWNLOADER=auto \
  HF_TOKEN=secret-token FMD_CURL_RETRY=1 FMD_FETCH_ATTEMPTS=2 \
  PATH="$ARIA_BIN:$PATH" ARIA_LOG="$ARIA_LOG" \
  "$SCRIPT" Org/Repo >"$WORK/aria-token.log" 2>&1
RC=$?
set -e
assert_eq "$RC" 0 "fetch with a token and aria2 available"
[ ! -s "$ARIA_LOG" ] || fail "aria2 was used with a token; the token would leak via ps"
grep -Fq "secret-token" "$WORK/aria-token.log" && fail "token printed in output"

# ---- 7f. wall-clock budget stops work and exits 2 ------------------------

touch "$WWW/.slow"
set +e
env HF_ENDPOINT="$BASE" FMD_MODEL_DIR="$WORK/models-budget" FMD_CURL_RETRY=0 \
  FMD_FETCH_ATTEMPTS=1 FMD_WALL_BUDGET=1 FMD_BACKOFF_BASE=1 FMD_BACKOFF_MAX=1 \
  "$SCRIPT" Org/Repo >"$WORK/budget.log" 2>&1
RC=$?
set -e
rm -f "$WWW/.slow"
assert_eq "$RC" 2 "wall-clock budget exit code"
grep -Fq "budget: wall-clock budget" "$WORK/budget.log" ||
  fail "budget expiry was not announced"
grep -Fq "summary:" "$WORK/budget.log" || fail "no run summary after a budget stop"

# ---- 8. offline verify without a manifest (header-only) ------------------

rm -f "$DIR/.fmd-manifest.json"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

fetch_rc verify "$DIR"
assert_eq "$RC" 0 "offline verify without a manifest"
grep -Fq "without .fmd-manifest.json" "$WORK/out.log" 2>/dev/null ||
  grep -Fq "no .fmd-manifest.json" "$WORK/out.log" || fail "missing manifest was not announced"

python3 - "$DIR/$SHARD1" <<'PY'
import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)
with open(path, "r+b") as fh:
    fh.truncate(size - 4096)
PY
fetch_rc verify "$DIR"
assert_eq "$RC" 2 "offline verify must catch a truncated shard from its header"
grep -Fq "payload size mismatch" "$WORK/out.log" ||
  fail "header check did not report the truncated payload"

printf 'OK: fetch_model.sh offline tests passed\n'
