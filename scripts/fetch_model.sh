#!/usr/bin/env bash
# Prefetch a model ONCE into the host weight root, resumably, shard by shard.
#
#   scripts/fetch_model.sh <hub-repo> [target-name]
#   scripts/fetch_model.sh verify <dir>     re-check an existing directory
#   scripts/fetch_model.sh repair <dir>     move invalid files aside, re-fetch
#   scripts/fetch_model.sh speed [repo|dir] measure CDN throughput, recommend
#
# Examples:
#   scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ
#   scripts/fetch_model.sh repair /home/archit-gore/models/Qwen3.5-9B-AWQ
#   scripts/fetch_model.sh speed QuantTrio/Qwen3.5-9B-AWQ
#
# Why this exists: weights are host data under MODEL_DIR (default
# /home/archit-gore/models), mounted read-only into containers. Image builds and
# docker volumes never hold weights, so rebuilding or pruning cannot force a
# re-download. This script is the only place a download is allowed, and it is
# run by a human on the host, never from a Dockerfile.
#
# Behaviour:
#   * the Hub file listing (`/api/models/<repo>/tree/<rev>?recursive=true`) is
#     the authority: a file that is not listed is optional, is never fetched,
#     never fails verification and is never quarantined. Only a listed file can
#     be required, and only a listed file whose size/hash disagrees is corrupt;
#   * reads `model.safetensors.index.json` and fetches each shard independently;
#   * `curl -C -` (or `aria2c -c -x8`, see FMD_DOWNLOADER) per file into
#     `<file>.part`; the part file is only renamed into place after size +
#     safetensors header (+ sha256 when known) pass;
#   * bad files are moved to `<dir>/.bad/<utc-timestamp>/<path>` with a
#     `<path>.reason` file, never deleted, so a failure can be diagnosed later;
#   * survives dead transfers: per-attempt stall detection, exponential backoff
#     capped at 5 min (zero-progress attempts back off harder), a per-file
#     attempt cap, an optional global wall-clock budget, and curl's own retries;
#   * detects servers that ignore `Range` (a 200 while resuming) and restarts
#     that file from zero instead of appending a second copy;
#   * refuses to start when the remaining bytes do not fit the filesystem;
#   * verifies LFS sha256 when the Hub exposes it, else records
#     `sha256: unavailable` in `.fmd-manifest.json`;
#   * prints a run summary: bytes gained, average speed, per-file status.
#
# Environment:
#   FMD_MODEL_DIR            host weight root (default $MODEL_DIR, else
#                            /home/archit-gore/models)
#   MODEL_DIR                fallback for the same value (see .env.example)
#   FMD_MODEL_REVISION       revision/branch (default main)
#   HF_ENDPOINT              mirror, e.g. https://hf-mirror.com (default
#                            https://huggingface.co)
#   HF_TOKEN                 optional, for gated repos; never printed
#   FMD_DOWNLOADER           auto|curl|aria2 (default auto: aria2c if installed,
#                            else single-stream curl)
#   FMD_ARIA2_CONNECTIONS    aria2c -x/-s per file (default 8)
#   FMD_WALL_BUDGET          stop starting new work after N seconds (default 0=off)
#   FMD_FETCH_ATTEMPTS       outer attempts per file (default 10)
#   FMD_CURL_RETRY           curl internal retries per attempt (default 20)
#   FMD_CONNECT_TIMEOUT      seconds (default 30)
#   FMD_SPEED_LIMIT          bytes/sec below which a transfer is dead (default 10240)
#   FMD_SPEED_TIME           seconds below that speed before abort (default 60)
#   FMD_BACKOFF_BASE         first backoff in seconds (default 5)
#   FMD_BACKOFF_MAX          backoff cap in seconds (default 300)
#   FMD_DISK_RESERVE_BYTES   headroom to keep free (default 536870912)
#   FMD_VERIFY_SHA256        set 0 to skip hashing (default 1)
#   FMD_MODEL_REPO           repo id for `repair` when `.source` is absent
#   FMD_MIRROR               alternate endpoint probed by `speed`
#                            (default https://hf-mirror.com)
#   FMD_PROBE_BYTES          bytes per `speed` probe (default 8388608)
#   FMD_PROBE_TIME           seconds per `speed` probe (default 20)
#   FMD_SLOW_THRESHOLD       bytes/sec below which `speed` says "sneakernet"
#                            (default 1048576 = 1 MiB/s)
#
# Exit codes:
#   0  all files verified
#   2  some files failed (names printed)
#   3  not enough disk space
#   4  metadata / index unavailable
#   64 usage error
#
# Deliberately curl + stdlib json, not `huggingface-cli`: the runtime venv must
# not grow a dependency, and per-file resume is the useful part. aria2 is used
# only when it is already on PATH; nothing is installed by this script.

set -euo pipefail

EXIT_OK=0
EXIT_PARTIAL=2
EXIT_DISK=3
EXIT_META=4
EXIT_USAGE=64

MODEL_DIR="${FMD_MODEL_DIR:-${MODEL_DIR:-/home/archit-gore/models}}"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
ENDPOINT="${ENDPOINT%/}"
REVISION="${FMD_MODEL_REVISION:-main}"
TOKEN="${HF_TOKEN:-}"
ATTEMPTS="${FMD_FETCH_ATTEMPTS:-10}"
CURL_RETRY="${FMD_CURL_RETRY:-20}"
CONNECT_TIMEOUT="${FMD_CONNECT_TIMEOUT:-30}"
SPEED_LIMIT="${FMD_SPEED_LIMIT:-10240}"
SPEED_TIME="${FMD_SPEED_TIME:-60}"
BACKOFF_BASE="${FMD_BACKOFF_BASE:-5}"
BACKOFF_MAX="${FMD_BACKOFF_MAX:-300}"
DISK_RESERVE="${FMD_DISK_RESERVE_BYTES:-536870912}"
VERIFY_SHA="${FMD_VERIFY_SHA256:-1}"
DOWNLOADER="${FMD_DOWNLOADER:-auto}"
ARIA_CONN="${FMD_ARIA2_CONNECTIONS:-8}"
WALL_BUDGET="${FMD_WALL_BUDGET:-0}"
MIRROR="${FMD_MIRROR:-https://hf-mirror.com}"
MIRROR="${MIRROR%/}"
PROBE_BYTES="${FMD_PROBE_BYTES:-8388608}"
PROBE_TIME="${FMD_PROBE_TIME:-20}"
SLOW_THRESHOLD="${FMD_SLOW_THRESHOLD:-1048576}"

HUB=""
NAME=""
DIR=""
RUN_START="$(date +%s)"
RUN_GAIN=0
RUN_QUARANTINED=0

# Per-file run accounting (bash 4 arrays; fetch_one runs in this shell).
RUN_PATHS=()
RUN_STATUS=()
RUN_BYTES=()
RUN_SECS=()

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/fmd-fetch.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT

usage() {
  cat >&2 <<'EOF'
usage:
  scripts/fetch_model.sh <hub-repo> [target-name]   fetch/resume into MODEL_DIR
  scripts/fetch_model.sh verify <dir>               re-check a downloaded dir
  scripts/fetch_model.sh repair <dir>               quarantine bad, re-fetch
  scripts/fetch_model.sh speed [repo|dir]           measure throughput, recommend

environment: FMD_MODEL_DIR, FMD_MODEL_REVISION, HF_ENDPOINT, HF_TOKEN,
             FMD_DOWNLOADER, FMD_ARIA2_CONNECTIONS, FMD_WALL_BUDGET,
             FMD_FETCH_ATTEMPTS, FMD_CURL_RETRY, FMD_CONNECT_TIMEOUT,
             FMD_SPEED_LIMIT, FMD_SPEED_TIME, FMD_BACKOFF_BASE,
             FMD_BACKOFF_MAX, FMD_DISK_RESERVE_BYTES, FMD_VERIFY_SHA256,
             FMD_MODEL_REPO, FMD_MIRROR, FMD_PROBE_BYTES, FMD_PROBE_TIME,
             FMD_SLOW_THRESHOLD
EOF
}

log() { printf '%s\n' "$*"; }

human_bytes() {
  awk -v b="${1:-0}" 'BEGIN {
    split("B KiB MiB GiB TiB", u, " ");
    i = 1;
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    if (i == 1) printf "%d %s", b, u[i]; else printf "%.1f %s", b, u[i];
  }'
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$1" | cut -d' ' -f1
  else
    shasum -a 256 -- "$1" | cut -d' ' -f1
  fi
}

# Bytes/sec -> "1.2 MiB/s". Kept in awk so no bc dependency.
human_rate() {
  awk -v b="${1:-0}" 'BEGIN {
    split("B/s KiB/s MiB/s GiB/s", u, " ");
    i = 1;
    while (b >= 1024 && i < 4) { b /= 1024; i++ }
    printf "%.2f %s", b, u[i];
  }'
}

budget_left() {
  [ "$WALL_BUDGET" -gt 0 ] || return 0
  local used=$(( $(date +%s) - RUN_START ))
  [ "$used" -lt "$WALL_BUDGET" ]
}

run_record() { # path status bytes seconds
  RUN_PATHS+=("$1")
  RUN_STATUS+=("$2")
  RUN_BYTES+=("${3:-0}")
  RUN_SECS+=("${4:-0}")
  RUN_GAIN=$(( RUN_GAIN + ${3:-0} ))
}

# Never destroy evidence. A bad file moves to <dir>/.bad/<utc>/<path> and gets
# a sibling .reason file, so a later diagnosis does not depend on scrollback.
quarantine() { # path reason
  local path=$1 reason=$2 rel dest ts
  [ -e "$path" ] || return 0
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  rel="${path#"$DIR"/}"
  dest="$DIR/.bad/$ts/$rel"
  mkdir -p "$(dirname "$dest")"
  if mv -f "$path" "$dest" 2>/dev/null; then
    printf '%s\n' "$reason" >"$dest.reason"
    log "bad    $rel -> .bad/$ts/ ($reason)"
    RUN_QUARANTINED=$(( RUN_QUARANTINED + 1 ))
  else
    printf 'warning: could not move %s aside (%s); leaving it in place\n' \
      "$rel" "$reason" >&2
    mv -f "$path" "$path.part" 2>/dev/null || true
  fi
  return 0
}

# Parse the safetensors framing and prove the declared payload is complete.
# Prints "header=<n> tensors=<k>" on success; prints a reason and exits 1.
st_check() {
  python3 - "$1" <<'PY'
import json
import os
import sys

path = sys.argv[1]
try:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            print(f"truncated: only {len(raw)} bytes, need an 8-byte header length")
            sys.exit(1)
        n = int.from_bytes(raw, "little")
        if n <= 0 or n > 100_000_000:
            print(f"implausible header length {n}")
            sys.exit(1)
        if 8 + n > size:
            print(f"header length {n} exceeds file size {size}")
            sys.exit(1)
        header = fh.read(n)
except OSError as exc:
    print(f"unreadable: {exc}")
    sys.exit(1)

try:
    meta = json.loads(header)
except Exception as exc:  # noqa: BLE001 - report any parse failure
    print(f"header is not JSON: {exc}")
    sys.exit(1)
if not isinstance(meta, dict):
    print("header is not a JSON object")
    sys.exit(1)

end = 8 + n
tensors = 0
for name, tensor in meta.items():
    if name == "__metadata__":
        continue
    offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
    if not (
        isinstance(offsets, list)
        and len(offsets) == 2
        and all(isinstance(x, int) for x in offsets)
        and 0 <= offsets[0] <= offsets[1]
    ):
        print(f"tensor {name!r} has no valid data_offsets")
        sys.exit(1)
    tensors += 1
    end = max(end, 8 + n + offsets[1])

if tensors == 0:
    print("no tensors in header")
    sys.exit(1)
if end != size:
    print(f"payload size mismatch: header declares {end} bytes, file is {size}")
    sys.exit(1)

print(f"header={n} tensors={tensors}")
PY
}

normalize_hub_json() {
  # Accepts either the tree listing (a bare JSON array, entries with type/size/
  # lfs.oid) or the legacy `revision?blobs=true` object with `siblings`. Both are
  # reduced to "size<TAB>sha256<TAB>path"; a non-file entry (directory) is
  # dropped, because only files can be downloaded or verified.
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)

if isinstance(data, list):
    entries = data
elif isinstance(data, dict):
    entries = data.get("siblings") or []
else:
    entries = []

seen = set()
for entry in entries:
    if not isinstance(entry, dict):
        continue
    if entry.get("type") not in (None, "file"):
        continue
    path = entry.get("rfilename") or entry.get("path") or ""
    path = path.strip("/")
    if not path or path.startswith(".") or path in seen:
        continue
    seen.add(path)
    size = entry.get("size")
    if not isinstance(size, int) or size < 0:
        size = 0
    lfs = entry.get("lfs") or {}
    sha = lfs.get("sha256") or lfs.get("oid") or "-"
    print(f"{size}\t{sha}\t{path}")
PY
}

write_manifest() {
  # Files that were required by an older manifest but are no longer listed by
  # the Hub are demoted to optional and kept in the manifest as `optional`:
  # the repo moved on, the local bytes are innocent, and a file absent from the
  # listing must never fail verification or be quarantined.
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import datetime
import json
import os
import sys

directory, tsv, repo, revision, endpoint = sys.argv[1:6]

manifest_path = os.path.join(directory, ".fmd-manifest.json")
previous = {}
if os.path.exists(manifest_path):
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            previous = {
                item["path"]: item
                for item in (json.load(fh).get("files") or [])
                if isinstance(item.get("path"), str)
            }
    except Exception:  # noqa: BLE001 - a broken old manifest is not fatal
        previous = {}

shards = set()
index_path = os.path.join(directory, "model.safetensors.index.json")
if os.path.exists(index_path):
    try:
        with open(index_path, encoding="utf-8") as fh:
            weight_map = json.load(fh).get("weight_map") or {}
        shards = {v for v in weight_map.values() if isinstance(v, str)}
    except Exception:  # noqa: BLE001 - a broken index is not fatal here
        shards = set()

files = []
seen = set()
with open(tsv, encoding="utf-8") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        size, sha, path = parts
        if path in seen:
            continue
        seen.add(path)
        if path.endswith("model.safetensors.index.json"):
            kind = "index"
        elif path in shards or (not shards and path.endswith(".safetensors")):
            kind = "shard"
        else:
            kind = "aux"
        files.append({
            "path": path,
            "size": int(size or 0),
            "sha256": None if sha in ("", "-") else sha,
            "kind": kind,
            "required": True,
        })

demoted = 0
for path, item in previous.items():
    if path in seen:
        continue
    if path in shards or (not shards and path.endswith(".safetensors")):
        # The index still names this shard (or there is no index to consult and
        # it looks like a weight file), so it is not optional. Missing it is a
        # real failure, not manifest drift.
        files.append({**item, "required": True, "kind": "shard"})
        continue
    demoted += 1
    files.append({**item, "required": False, "kind": "optional"})

manifest = {
    "repo": repo,
    "revision": revision,
    "endpoint": endpoint,
    "generated_at": datetime.datetime.now(datetime.timezone.utc)
    .replace(microsecond=0)
    .isoformat(),
    "integrity": "sha256"
    if any(item.get("sha256") for item in files)
    else "size+header only (sha256 unavailable)",
    "files": files,
}

tmp = manifest_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp, manifest_path)
if demoted:
    print(f"manifest: {demoted} file(s) no longer listed by the Hub marked optional",
          file=sys.stderr)
PY
}

manifest_read() {
  # size, sha256, kind, required, path. `required` defaults to true so that
  # manifests written before optional files existed keep their old meaning.
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    manifest = json.load(fh)
for item in manifest.get("files") or []:
    sha = item.get("sha256") or "-"
    required = "1" if item.get("required", True) else "0"
    print(f'{item.get("size", 0)}\t{sha}\t{item.get("kind", "aux")}\t'
          f'{required}\t{item["path"]}')
PY
}

manifest_field() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    value = json.load(fh).get(sys.argv[2], "")
print(value or "")
PY
}

index_shards() {
  python3 - "$1" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        weight_map = json.load(fh).get("weight_map") or {}
except Exception:  # noqa: BLE001
    sys.exit(1)

seen = set()
for path in weight_map.values():
    if isinstance(path, str) and path not in seen:
        seen.add(path)
        print(path)
PY
}

# ---- curl plumbing -------------------------------------------------------

CURL=(curl -sS
  --connect-timeout "$CONNECT_TIMEOUT"
  --retry "$CURL_RETRY"
  --retry-delay 5
  --retry-all-errors
  --speed-limit "$SPEED_LIMIT"
  --speed-time "$SPEED_TIME")

# Passing the token on the command line would expose it in `ps`. curl reads
# `-H @file`, so the secret stays in a 0600 file.
CURL_AUTH=()
if [ -n "$TOKEN" ]; then
  HDR="$TMPD/auth.hdr"
  ( umask 077; printf 'Authorization: Bearer %s\n' "$TOKEN" >"$HDR" )
  CURL_AUTH=(-H "@$HDR")
fi

remote_size() {
  "${CURL[@]}" -L "${CURL_AUTH[@]}" -sSI --max-time 60 "$1" 2>/dev/null |
    awk 'tolower($1) == "content-length:" { v = $2 } END { gsub("\r", "", v); if (v == "") v = 0; print v + 0 }'
}

# When the Hub API does not expose LFS sha256 values, the cheap pointer file at
# /raw/<rev>/<path> does. Anything larger than a pointer is not downloaded here.
enrich_sha_from_pointers() {
  local dir=$1 repo=$2 rev=$3 tmp="$TMPD/sha.tsv"
  local size sha path body oid
  : >"$tmp"
  while IFS=$'\t' read -r size sha path; do
    if [ "$sha" = "-" ] && [[ "$path" == *.safetensors ]]; then
      body="$("${CURL[@]}" -L "${CURL_AUTH[@]}" --fail --max-time 30 \
        --max-filesize 8192 "$ENDPOINT/$repo/raw/$rev/$path" 2>/dev/null || true)"
      if [[ "$body" == "version https://git-lfs.github.com/spec/v1"* ]]; then
        oid="$(printf '%s\n' "$body" | awk -F'sha256:' '/^oid sha256:/ { print $2; exit }')"
        [ -n "$oid" ] && sha="$oid"
      fi
    fi
    printf '%s\t%s\t%s\n' "$size" "$sha" "$path" >>"$tmp"
  done <"$dir/.fmd-files.tsv"
  mv -f "$tmp" "$dir/.fmd-files.tsv"
}

fetch_listing() {
  local dir=$1 repo=$2 rev=$3 api idx size path
  # The tree endpoint is the authority on what the repo actually contains: a
  # file that is not here does not exist and must not be demanded locally.
  for api in \
    "$ENDPOINT/api/models/$repo/tree/$rev?recursive=true" \
    "$ENDPOINT/api/models/$repo/revision/$rev?blobs=true"; do
    if "${CURL[@]}" -L "${CURL_AUTH[@]}" --fail -H 'Accept: application/json' \
      --max-time 120 -o "$TMPD/hub.json" "$api" 2>>"$TMPD/curl.err"; then
      normalize_hub_json "$TMPD/hub.json" >"$dir/.fmd-files.tsv"
      if [ -s "$dir/.fmd-files.tsv" ]; then
        enrich_sha_from_pointers "$dir" "$repo" "$rev"
        return 0
      fi
    fi
  done

  # Fallback: the index alone still names every shard. Sizes come from HEAD.
  # Only reachable when the listing API is down; anything not named here stays
  # absent and is therefore optional, never a verification failure.
  printf '%s\n' "listing API unavailable; falling back to model.safetensors.index.json" >&2
  idx="$dir/model.safetensors.index.json"
  if "${CURL[@]}" -L "${CURL_AUTH[@]}" --fail --max-time 120 \
    -o "$idx.tmp" "$ENDPOINT/$repo/resolve/$rev/model.safetensors.index.json" \
    2>>"$TMPD/curl.err"; then
    mv -f "$idx.tmp" "$idx"
    : >"$dir/.fmd-files.tsv"
    while IFS= read -r path; do
      [ -n "$path" ] || continue
      size="$(remote_size "$ENDPOINT/$repo/resolve/$rev/$path")"
      printf '%s\t-\t%s\n' "${size:-0}" "$path" >>"$dir/.fmd-files.tsv"
    done < <(index_shards "$idx" || true)
    size="$(stat -c %s "$idx")"
    printf '%s\t-\t%s\n' "$size" "model.safetensors.index.json" >>"$dir/.fmd-files.tsv"
    if [ -s "$dir/.fmd-files.tsv" ]; then
      enrich_sha_from_pointers "$dir" "$repo" "$rev"
      return 0
    fi
  fi
  return 1
}

ensure_manifest() {
  local dir=$1 repo=$2 rev=$3 mrepo mrev mreq
  if [ -s "$dir/.fmd-manifest.json" ]; then
    mrepo="$(manifest_field "$dir/.fmd-manifest.json" repo)"
    mrev="$(manifest_field "$dir/.fmd-manifest.json" revision)"
    # Manifests written before optional-file classification carry no `required`
    # field; refresh once so absent files stop being reported as failures.
    mreq="$(grep -c '"required"' "$dir/.fmd-manifest.json" 2>/dev/null || true)"
    if [ "$mrepo" = "$repo" ] && [ "$mrev" = "$rev" ] && [ "${mreq:-0}" -gt 0 ]; then
      return 0
    fi
  fi
  log "metadata: $repo@$rev"
  fetch_listing "$dir" "$repo" "$rev" || return 1
  write_manifest "$dir" "$dir/.fmd-files.tsv" "$repo" "$rev" "$ENDPOINT"
}

ensure_index() {
  local dir=$1 path="model.safetensors.index.json" line size sha
  [ -s "$dir/$path" ] && return 0
  line="$(awk -F'\t' -v p="$path" '$3 == p { print; exit }' "$dir/.fmd-files.tsv" 2>/dev/null || true)"
  [ -n "$line" ] || return 1
  size="$(printf '%s' "$line" | cut -f1)"
  sha="$(printf '%s' "$line" | cut -f2)"
  log "index: fetching $path"
  fetch_one "$path" "$size" "$sha"
}

build_queue() {
  python3 - "$1" <<'PY'
import json
import os
import sys

directory = sys.argv[1]
with open(os.path.join(directory, ".fmd-manifest.json"), encoding="utf-8") as fh:
    manifest = json.load(fh)
files = {item["path"]: item for item in manifest.get("files") or []}

order = []
index_path = os.path.join(directory, "model.safetensors.index.json")
if os.path.exists(index_path):
    try:
        with open(index_path, encoding="utf-8") as fh:
            weight_map = json.load(fh).get("weight_map") or {}
        for path in weight_map.values():
            if path in files and path not in order:
                order.append(path)
    except Exception:  # noqa: BLE001
        pass
if not order:
    order = [path for path in files
             if files[path].get("required", True) and path.endswith(".safetensors")]
order += [path for path in files
          if files[path].get("required", True) and path not in order]

for path in order:
    item = files[path]
    print(f'{item.get("size", 0)}\t{item.get("sha256") or "-"}\t{path}')
PY
}

check_disk() {
  local dir=$1 queue=$2 avail need
  avail="$(df -Pk "$dir" 2>/dev/null | awk 'NR == 2 { print $4 * 1024 }')"
  [ -n "$avail" ] || return 0
  need="$(printf '%s\n' "$queue" | python3 -c '
import os
import sys

directory = sys.argv[1]
total = 0
for line in sys.stdin:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        continue
    size = int(parts[0] or 0)
    if size <= 0:
        continue
    path = parts[2]
    have = 0
    part = os.path.join(directory, path + ".part")
    out = os.path.join(directory, path)
    if os.path.exists(part):
        have = os.path.getsize(part)
    elif os.path.exists(out):
        have = os.path.getsize(out)
    if have < size:
        total += size - have
print(total)
' "$dir")"
  if [ "$need" -gt 0 ] && [ "$avail" -lt "$((need + DISK_RESERVE))" ]; then
    cat >&2 <<EOF
disk: refusing to start — $dir has $(human_bytes "$avail") free but this
      fetch still needs $(human_bytes "$need") plus $(human_bytes "$DISK_RESERVE") reserve.
      Free space, move MODEL_DIR, or lower FMD_DISK_RESERVE_BYTES.
EOF
    return 1
  fi
  return 0
}

# ---- verification --------------------------------------------------------

verify_entry() { # dir size sha path required
  local dir=$1 size=$2 sha=$3 path=$4 required="${5:-1}"
  local file="$dir/$path" actual hdr="" got=""
  if [ ! -f "$file" ]; then
    if [ "$required" = "1" ]; then
      printf 'FAIL %s: missing\n' "$path"
      return 1
    fi
    # Not listed by the Hub (or demoted by manifest drift): absence is correct.
    printf 'OPTIONAL %s: not listed upstream, absent locally\n' "$path"
    return 0
  fi
  if [ "$required" != "1" ]; then
    printf 'OPTIONAL %s: not listed upstream, present locally (left untouched)\n' "$path"
    return 0
  fi
  actual="$(stat -c %s "$file")"
  if [ "$size" -gt 0 ] && [ "$actual" -ne "$size" ]; then
    printf 'FAIL %s: expected %s bytes, found %s\n' "$path" "$size" "$actual"
    return 1
  fi
  if [[ "$path" == *.safetensors ]]; then
    if ! hdr="$(st_check "$file")"; then
      printf 'FAIL %s: safetensors header: %s\n' "$path" "$hdr"
      return 1
    fi
  fi
  if [ "$sha" != "-" ] && [ -n "$sha" ] && [ "$VERIFY_SHA" = "1" ]; then
    got="$(sha256_of "$file")"
    if [ "$got" != "$sha" ]; then
      printf 'FAIL %s: sha256 mismatch (expected %s…, got %s…)\n' \
        "$path" "${sha:0:12}" "${got:0:12}"
      return 1
    fi
  fi
  if [ -n "$hdr" ]; then
    printf 'PASS %s (%s bytes, %s)\n' "$path" "$actual" "$hdr"
  else
    printf 'PASS %s (%s bytes)\n' "$path" "$actual"
  fi
  return 0
}

verify_list() {
  local dir=$1 list=$2 size sha kind required path ok=0 bad=0 optional=0
  local -a failures=()
  while IFS=$'\t' read -r size sha kind required path; do
    [ -n "$path" ] || continue
    if verify_entry "$dir" "$size" "$sha" "$path" "${required:-1}"; then
      if [ "${required:-1}" = "1" ]; then
        ok=$((ok + 1))
      else
        optional=$((optional + 1))
      fi
    else
      bad=$((bad + 1))
      failures+=("$path")
    fi
  done <"$list"
  if [ "$optional" -gt 0 ]; then
    log "$ok file(s) verified, $optional optional, $bad failed"
  else
    log "$ok file(s) verified, $bad failed"
  fi
  if [ "$bad" -gt 0 ]; then
    printf 'failed: %s\n' "${failures[*]}" >&2
    return 1
  fi
  return 0
}

build_offline_list() {
  local dir=$1 count=0 f path
  if [ -s "$dir/model.safetensors.index.json" ]; then
    while IFS= read -r path; do
      [ -n "$path" ] || continue
      printf '0\t-\tshard\t1\t%s\n' "$path"
      count=$((count + 1))
    done < <(index_shards "$dir/model.safetensors.index.json" || true)
  fi
  if [ "$count" -eq 0 ]; then
    shopt -s nullglob
    for f in "$dir"/*.safetensors; do
      printf '0\t-\tshard\t1\t%s\n' "$(basename "$f")"
      count=$((count + 1))
    done
    shopt -u nullglob
  fi
  [ "$count" -gt 0 ]
}

# ---- download ------------------------------------------------------------

verify_downloaded() {
  local file=$1 path=$2 expected=$3 sha=$4 actual out got
  actual="$(stat -c %s "$file" 2>/dev/null || echo 0)"
  if [ "$expected" -gt 0 ] && [ "$actual" -ne "$expected" ]; then
    printf 'size mismatch: expected %s, got %s' "$expected" "$actual" >&2
    return 1
  fi
  if [[ "$path" == *.safetensors ]]; then
    if ! out="$(st_check "$file")"; then
      printf 'safetensors header invalid: %s' "$out" >&2
      return 1
    fi
  fi
  if [ "$sha" != "-" ] && [ -n "$sha" ] && [ "$VERIFY_SHA" = "1" ]; then
    got="$(sha256_of "$file")"
    if [ "$got" != "$sha" ]; then
      printf 'sha256 mismatch: expected %s…, got %s…' "${sha:0:16}" "${got:0:16}" >&2
      return 1
    fi
  fi
  return 0
}

ATTEMPT_MAX="${FMD_ATTEMPT_MAX:-1800}"

have_of() { # path -> bytes on disk (.part preferred), 0 when absent
  local out="$DIR/$1" part="$DIR/$1.part" size=0
  if [ -f "$part" ]; then
    size="$(stat -c %s "$part" 2>/dev/null || echo 0)"
  elif [ -f "$out" ]; then
    size="$(stat -c %s "$out" 2>/dev/null || echo 0)"
  fi
  printf '%s' "$size"
}

aria2_usable() {
  [ "$DOWNLOADER" != "curl" ] || return 1
  command -v aria2c >/dev/null 2>&1 || return 1
  # aria2 has no `--header=@file`, so a token would sit in argv where `ps` can
  # read it. Gated repos always use the curl path.
  [ -z "$TOKEN" ] || return 1
  return 0
}

fetch_one() {
  # Dispatcher: measures per-file progress for the run summary, then hands off
  # to aria2 (multi-connection) or curl (single stream).
  local path=$1 expected=$2 sha=$3 before after t0 t1 rc=0 gain elapsed
  before="$(have_of "$path")"
  t0="$(date +%s)"
  if aria2_usable; then
    fetch_one_aria2 "$path" "$expected" "$sha" || rc=$?
  else
    fetch_one_curl "$path" "$expected" "$sha" || rc=$?
  fi
  t1="$(date +%s)"
  elapsed=$((t1 - t0))
  [ "$elapsed" -gt 0 ] || elapsed=1
  after="$(have_of "$path")"
  gain=$((after - before))
  [ "$gain" -ge 0 ] || gain=0
  if [ "$rc" -eq 0 ]; then
    run_record "$path" "ok" "$gain" "$elapsed"
  else
    run_record "$path" "FAIL" "$gain" "$elapsed"
  fi
  return "$rc"
}

fetch_one_aria2() {
  local path=$1 expected=$2 sha=$3
  local out="$DIR/$path" part="$DIR/$path.part" url
  local attempt=1 start end got rc apid waited killed delay reason
  mkdir -p "$(dirname "$out")"
  url="$ENDPOINT/$HUB/resolve/$REVISION/$path"
  log "fetch  $path ($(human_bytes "$expected"), aria2c -x$ARIA_CONN, have $(human_bytes "$(have_of "$path")"))"

  while [ "$attempt" -le "$ATTEMPTS" ]; do
    if ! budget_left; then
      printf 'FAIL %s: wall-clock budget (%ss) reached before this file\n' \
        "$path" "$WALL_BUDGET" >&2
      return 1
    fi
    start="$(stat -c %s "$part" 2>/dev/null || echo 0)"
    rc=0
    aria2c -c -x "$ARIA_CONN" -s "$ARIA_CONN" -k 1M \
      --max-tries=0 --retry-wait=5 \
      --lowest-speed-limit=10K --timeout=60 \
      --connect-timeout="$CONNECT_TIMEOUT" \
      --file-allocation=none --auto-file-renaming=false --allow-overwrite=true \
      --summary-interval=30 --console-log-level=warn \
      -d "$(dirname "$part")" -o "$(basename "$part")" "$url" \
      >"$TMPD/aria2.log" 2>&1 &
    apid=$!
    waited=0
    killed=0
    while kill -0 "$apid" 2>/dev/null; do
      sleep 5
      waited=$((waited + 5))
      if [ "$waited" -ge "$ATTEMPT_MAX" ] || ! budget_left; then
        kill "$apid" 2>/dev/null || true
        killed=1
        break
      fi
    done
    wait "$apid" 2>/dev/null || rc=$?
    end="$(stat -c %s "$part" 2>/dev/null || echo 0)"
    got=$((end - start))
    [ "$got" -ge 0 ] || got=0
    [ "$waited" -gt 0 ] || waited=1
    printf '  %-44s attempt %d/%d  +%s  %s on disk  %s  aria2 rc=%s%s\n' \
      "$path" "$attempt" "$ATTEMPTS" "$(human_bytes "$got")" \
      "$(human_bytes "$end")" "$(human_rate "$((got / waited))")" \
      "$rc" "$([ "$killed" = 1 ] && printf ' (watchdog)')"

    if [ "$expected" -gt 0 ] && [ "$end" -eq "$expected" ]; then
      if reason="$(verify_downloaded "$part" "$path" "$expected" "$sha" 2>&1)"; then
        mv -f "$part" "$out"
        log "ok     $path ($(human_bytes "$end"))"
        return 0
      fi
      quarantine "$part" "verification failed after aria2: $reason"
    elif [ "$expected" -gt 0 ] && [ "$end" -gt "$expected" ]; then
      quarantine "$part" "oversize: $end bytes on disk, Hub lists $expected"
      log "  $path: restarting after quarantine"
    elif [ -s "$TMPD/aria2.log" ] && tail -n 40 "$TMPD/aria2.log" | grep -qiE 'error|failed'; then
      printf '  %s: aria2: %s\n' "$path" \
        "$(tail -n 3 "$TMPD/aria2.log" | tr '\n' ' ')"
    fi

    attempt=$((attempt + 1))
    if [ "$attempt" -le "$ATTEMPTS" ]; then
      delay=$((BACKOFF_BASE * (1 << (attempt - 2))))
      [ "$got" -gt 0 ] || delay=$((delay < 30 ? 30 : delay))
      [ "$delay" -le "$BACKOFF_MAX" ] || delay="$BACKOFF_MAX"
      log "  $path: retrying in ${delay}s (attempt $attempt/$ATTEMPTS)"
      sleep "$delay"
    fi
  done

  end="$(stat -c %s "$part" 2>/dev/null || echo 0)"
  printf 'FAIL %s: exhausted %d aria2 attempts (expected %s bytes, have %s)\n' \
    "$path" "$ATTEMPTS" "$expected" "$end" >&2
  [ -s "$TMPD/aria2.log" ] && printf '  last aria2 line: %s\n' \
    "$(tail -n 1 "$TMPD/aria2.log")" >&2
  return 1
}

fetch_one_curl() {
  local path=$1 expected=$2 sha=$3
  local out="$DIR/$path" part="$DIR/$path.part"
  local have=0 psz=0 url reason attempt=1 start end got rc code t0 t1 elapsed
  local delay size_label
  mkdir -p "$(dirname "$out")"
  url="$ENDPOINT/$HUB/resolve/$REVISION/$path"
  size_label="$(human_bytes "$expected")"
  [ "$expected" -gt 0 ] || size_label="unknown size"

  if [ -f "$out" ]; then
    have="$(stat -c %s "$out")"
    if [ "$expected" -gt 0 ] && [ "$have" -eq "$expected" ]; then
      if verify_downloaded "$out" "$path" "$expected" "$sha" 2>/dev/null; then
        log "have   $path ($(human_bytes "$have"))"
        return 0
      fi
      quarantine "$out" "full-size file failed verification (expected sha256/header does not match)"
      log "  $path: refetching"
    elif [ "$expected" -gt 0 ] && [ "$have" -gt "$expected" ]; then
      quarantine "$out" "oversize: $have bytes on disk, Hub lists $expected"
    elif [ ! -f "$part" ]; then
      # Preserve bytes left by an older/easier-to-interrupt run.
      mv -f "$out" "$part"
    else
      mv -f "$out" "$out.duplicate.$(date -u +%Y%m%dT%H%M%SZ)"
    fi
  fi

  if [ -f "$part" ]; then
    psz="$(stat -c %s "$part")"
    if [ "$expected" -gt 0 ] && [ "$psz" -eq "$expected" ]; then
      if verify_downloaded "$part" "$path" "$expected" "$sha"; then
        mv -f "$part" "$out"
        log "have   $path ($(human_bytes "$psz"))"
        return 0
      fi
      quarantine "$part" "partial reached full size but failed verification"
      log "  $path: restarting it"
    elif [ "$expected" -gt 0 ] && [ "$psz" -gt "$expected" ]; then
      quarantine "$part" "oversize partial: $psz bytes, Hub lists $expected"
    fi
  fi

  log "fetch  $path ($size_label, have $(human_bytes "$(stat -c %s "$part" 2>/dev/null || echo 0)"))"

  while [ "$attempt" -le "$ATTEMPTS" ]; do
    if ! budget_left; then
      printf 'FAIL %s: wall-clock budget (%ss) reached before this file\n' \
        "$path" "$WALL_BUDGET" >&2
      return 1
    fi
    start="$(stat -c %s "$part" 2>/dev/null || echo 0)"
    t0="$(date +%s)"
    rc=0
    code="$("${CURL[@]}" -L -C - --fail "${CURL_AUTH[@]}" \
      -o "$part" -w '%{http_code}' "$url" 2>>"$TMPD/curl.err")" || rc=$?
    t1="$(date +%s)"
    elapsed=$((t1 - t0))
    [ "$elapsed" -gt 0 ] || elapsed=1
    end="$(stat -c %s "$part" 2>/dev/null || echo 0)"
    got=$((end - start))
    [ "$got" -ge 0 ] || got=0
    [ -n "$code" ] || code="-"

    printf '  %-44s attempt %d/%d  +%s  %s on disk  %s  http=%s\n' \
      "$path" "$attempt" "$ATTEMPTS" "$(human_bytes "$got")" \
      "$(human_bytes "$end")" "$(human_rate "$((got / elapsed))")" "$code"

    # 401/403/404 will never succeed on retry (bad token, gated repo, wrong
    # path). Fail now instead of backing off for hours.
    if [ "$code" = "401" ] || [ "$code" = "403" ] || [ "$code" = "404" ]; then
      printf 'FAIL %s: http %s (not retryable; check repo/revision/HF_TOKEN)\n' \
        "$path" "$code" >&2
      return 1
    fi

    # A 200 to a ranged request means the server ignored `Range` and sent the
    # whole body; curl may have appended it. Never trust Content-Length here.
    if [ "$code" = "416" ] && [ "$expected" -eq 0 ] && [ -s "$part" ]; then
      # Unknown total size and the server says our offset is at EOF: verify.
      if reason="$(verify_downloaded "$part" "$path" "$expected" "$sha" 2>&1)"; then
        mv -f "$part" "$out"
        log "ok     $path ($(human_bytes "$end"))"
        return 0
      fi
      quarantine "$part" "http 416 at EOF but verification failed: $reason"
    elif [ "$start" -gt 0 ] && [ "$code" = "200" ] && [ "$rc" -ne 0 ]; then
      log "  $path: server ignored Range (http 200) — restarting from zero"
      rm -f "$part"
      rc=1
      end=0
    elif [ "$rc" -eq 33 ]; then
      log "  $path: server rejected resume — restarting from zero"
      rm -f "$part"
      end=0
    fi

    if { [ "$expected" -gt 0 ] && [ "$end" -eq "$expected" ]; } ||
      { [ "$expected" -eq 0 ] && [ "$rc" -eq 0 ]; }; then
      if reason="$(verify_downloaded "$part" "$path" "$expected" "$sha" 2>&1)"; then
        mv -f "$part" "$out"
        log "ok     $path ($(human_bytes "$end"))"
        return 0
      fi
      log "  $path: verification failed — $reason"
      quarantine "$part" "downloaded bytes failed verification: $reason"
    elif [ "$expected" -gt 0 ] && [ "$end" -gt "$expected" ]; then
      quarantine "$part" "oversize partial: $end bytes, Hub lists $expected"
    fi

    attempt=$((attempt + 1))
    if [ "$attempt" -le "$ATTEMPTS" ]; then
      delay=$((BACKOFF_BASE * (1 << (attempt - 2))))
      # A zero-progress attempt means the link is dead, not slow: back off
      # harder than the sequence alone would (never retry in 5 s on a stall).
      [ "$got" -gt 0 ] || delay=$((delay < 30 ? 30 : delay))
      [ "$delay" -le "$BACKOFF_MAX" ] || delay="$BACKOFF_MAX"
      log "  $path: retrying in ${delay}s (attempt $attempt/$ATTEMPTS)"
      sleep "$delay"
    fi
  done

  end="$(stat -c %s "$part" 2>/dev/null || echo 0)"
  printf 'FAIL %s: exhausted %d attempts (expected %s bytes, have %s)\n' \
    "$path" "$ATTEMPTS" "$expected" "$end" >&2
  if [ -s "$TMPD/curl.err" ]; then
    printf '  last curl error: %s\n' "$(tail -n 1 "$TMPD/curl.err")" >&2
  fi
  # Partial bytes stay put: they are resumable, and the summary names the file.
  return 1
}

# ---- speed probe ---------------------------------------------------------

probe_bytes_done() { # label url args... -> sets PROBE_RATE, PROBE_NOTE
  local label=$1 url=$2
  shift 2
  local speed
  speed="$(curl -sS -L -r "0-$((PROBE_BYTES - 1))" --max-time "$PROBE_TIME" \
    -o /dev/null -w '%{speed_download}' "$@" "$url" 2>"$TMPD/probe.err" || true)"
  case "$speed" in
    '' | *[!0-9.]*) speed=0 ;;
  esac
  # curl can print exponential notation; force an integer rather than letting
  # bash arithmetic choke on `5.2e+23`.
  PROBE_RATE="$(awk -v s="$speed" 'BEGIN { printf "%d", s + 0 }')"
  [ -n "$PROBE_RATE" ] || PROBE_RATE=0
  if [ "$PROBE_RATE" -le 0 ]; then
    PROBE_NOTE="$(tail -n 1 "$TMPD/probe.err" | cut -c1-60)"
    [ -n "$PROBE_NOTE" ] || PROBE_NOTE="no data"
  else
    PROBE_NOTE=""
  fi
}

probe_parallel() { # label url connections args... -> PROBE_RATE
  local label=$1 url=$2 n=$3
  shift 3
  local chunk=$((PROBE_BYTES / n)) i s e dir start end total elapsed psz
  [ "$chunk" -gt 65536 ] || chunk=65536
  dir="$TMPD/par.$$.$RANDOM"
  mkdir -p "$dir"
  start="$(date +%s%N)"
  for i in $(seq 0 $((n - 1))); do
    s=$((i * chunk))
    e=$((s + chunk - 1))
    # Measure real bytes on disk: curl's %{size_download} reports a sentinel on
    # some failed ranged requests, which produced a nonsense "GiB/s" reading.
    curl -sS -L -r "$s-$e" --max-time "$PROBE_TIME" -o "$dir/$i.bin" \
      "$@" "$url" >/dev/null 2>&1 &
  done
  wait
  end="$(date +%s%N)"
  total=0
  for i in $(seq 0 $((n - 1))); do
    psz="$(stat -c %s "$dir/$i.bin" 2>/dev/null || echo 0)"
    total=$((total + psz))
  done
  elapsed=$(( (end - start) / 1000000000 ))
  [ "$elapsed" -gt 0 ] || elapsed=1
  PROBE_RATE=$((total / elapsed))
  PROBE_NOTE="$n streams, $(human_bytes "$total") in ${elapsed}s"
  rm -rf "$dir"
}

cmd_speed() {
  local target="${1:-}" repo="" dir="" probe="" total=0 tsv="$TMPD/speed.tsv"
  local url best_label="" best_rate=0 rate
  if [ -n "$target" ] && [ -d "$target" ]; then
    dir="$target"
    repo="$(manifest_field "$dir/.fmd-manifest.json" repo 2>/dev/null || true)"
    [ -n "$repo" ] || repo="$(head -n 1 "$dir/.source" 2>/dev/null || true)"
  else
    repo="$target"
  fi
  [ -n "$repo" ] || repo="${FMD_MODEL_REPO:-}"
  if [ -z "$repo" ]; then
    usage
    exit "$EXIT_USAGE"
  fi

  log "probe:  $repo@$REVISION via $ENDPOINT"
  log "budget: $((PROBE_BYTES / 1048576)) MiB per method, ${PROBE_TIME}s cap each"
  log ""

  # A probe needs a real payload: pick the largest file the repo lists.
  if ! "${CURL[@]}" -L "${CURL_AUTH[@]}" --fail -H 'Accept: application/json' \
    --max-time 60 -o "$TMPD/speed.json" \
    "$ENDPOINT/api/models/$repo/tree/$REVISION?recursive=true" 2>/dev/null; then
    printf 'speed: could not read the file listing from %s\n' "$ENDPOINT" >&2
    exit "$EXIT_META"
  fi
  normalize_hub_json "$TMPD/speed.json" >"$tsv"
  read -r total probe < <(python3 - "$tsv" <<'PY'
import sys

total = 0
best = (0, "")
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        size, _sha, path = parts
        size = int(size or 0)
        total += size
        if path.endswith(".safetensors") and size > best[0]:
            best = (size, path)
print(total, best[1] or "")
PY
)
  if [ -z "$probe" ]; then
    printf 'speed: no .safetensors in the listing; nothing to probe\n' >&2
    exit "$EXIT_META"
  fi
  log "payload: $probe"
  log "repo total: $(human_bytes "$total")"
  log ""

  url="$ENDPOINT/$repo/resolve/$REVISION/$probe"
  printf '%-26s %12s  %s\n' "method" "throughput" "note"

  probe_bytes_done primary "$url" -4
  printf '%-26s %12s  %s\n' "curl -4 single stream" "$(human_rate "$PROBE_RATE")" "${PROBE_NOTE:-}"
  rate="$PROBE_RATE"
  [ "$rate" -le "$best_rate" ] || { best_rate="$rate"; best_label="curl -4 (IPv4, single stream)"; }

  probe_bytes_done primary6 "$url" -6
  if [ "$PROBE_RATE" -gt 0 ]; then
    printf '%-26s %12s  %s\n' "curl -6 single stream" "$(human_rate "$PROBE_RATE")" ""
    [ "$PROBE_RATE" -le "$best_rate" ] ||
      { best_rate="$PROBE_RATE"; best_label="curl -6 (IPv6, single stream)"; }
  else
    printf '%-26s %12s  %s\n' "curl -6 single stream" "n/a" "no IPv6 route"
  fi

  local conns="${FMD_PROBE_CONNECTIONS:-4}"
  probe_parallel primary "$url" "$conns" -4
  printf '%-26s %12s  %s\n' "curl -4 x$conns parallel" "$(human_rate "$PROBE_RATE")" "$PROBE_NOTE"
  [ "$PROBE_RATE" -le "$best_rate" ] ||
    { best_rate="$PROBE_RATE"; best_label="curl -4 x$conns parallel (FMD_DOWNLOADER=aria2)"; }

  if [ -n "$MIRROR" ] && [ "$MIRROR" != "$ENDPOINT" ]; then
    probe_bytes_done mirror "$MIRROR/$repo/resolve/$REVISION/$probe" -4
    printf '%-26s %12s  %s\n' "mirror $MIRROR" "$(human_rate "$PROBE_RATE")" "${PROBE_NOTE:-}"
    if [ "$PROBE_RATE" -gt "$best_rate" ]; then
      best_rate="$PROBE_RATE"
      best_label="mirror $MIRROR (HF_ENDPOINT=$MIRROR)"
    fi
    probe_parallel mirror "$MIRROR/$repo/resolve/$REVISION/$probe" "$conns" -4
    printf '%-26s %12s  %s\n' "mirror x$conns parallel" "$(human_rate "$PROBE_RATE")" "$PROBE_NOTE"
    if [ "$PROBE_RATE" -gt "$best_rate" ]; then
      best_rate="$PROBE_RATE"
      best_label="mirror x$conns (HF_ENDPOINT=$MIRROR + aria2)"
    fi
  fi

  if command -v aria2c >/dev/null 2>&1; then
    printf '%-26s %12s  %s\n' "aria2c" "available" "-x$ARIA_CONN -s$ARIA_CONN -k1M"
  else
    printf '%-26s %12s  %s\n' "aria2c" "absent" "install for -x8, or use curl x$conns"
  fi

  log ""
  log "fastest: $best_label at $(human_rate "$best_rate")"
  if [ "$best_rate" -gt 0 ]; then
    log "ETA for $(human_bytes "$total") at this rate: $(printf '%dh%02dm' $(((total / best_rate) / 3600)) $((((total / best_rate) % 3600) / 60)))"
  fi
  log ""
  case "$best_label" in
    *mirror*) log "use: HF_ENDPOINT=$MIRROR scripts/fetch_model.sh $repo" ;;
    *aria2*) log "use: FMD_DOWNLOADER=aria2 scripts/fetch_model.sh $repo" ;;
    *) log "use: scripts/fetch_model.sh $repo" ;;
  esac

  if [ "$best_rate" -lt "$SLOW_THRESHOLD" ]; then
    log ""
    log "verdict: below $(human_rate "$SLOW_THRESHOLD") — this link cannot finish $(human_bytes "$total") in a night."
    log "         fetch elsewhere, then copy and verify on this host:"
    log "           scp -r user@fasthost:~/models/$(basename "$repo") \"\$MODEL_DIR/\""
    log "           scripts/fetch_model.sh verify \"\$MODEL_DIR/$(basename "$repo")\""
  else
    log ""
    log "verdict: workable overnight; run with FMD_WALL_BUDGET=$((12 * 3600)) and re-run to resume."
  fi
  exit "$EXIT_OK"
}

# ---- commands ------------------------------------------------------------

check_downloader() {
  if [ "$DOWNLOADER" = "aria2" ] && ! command -v aria2c >/dev/null 2>&1; then
    printf 'FMD_DOWNLOADER=aria2 but aria2c is not on PATH; install it or use FMD_DOWNLOADER=curl\n' >&2
    exit "$EXIT_USAGE"
  fi
  if [ "$DOWNLOADER" = "auto" ] && aria2_usable; then
    log "downloader: aria2c -x$ARIA_CONN (multi-connection, resume)"
  else
    log "downloader: curl (single stream)"
  fi
}

print_summary() {
  local elapsed=$(( $(date +%s) - RUN_START )) i rate secs
  [ "$elapsed" -gt 0 ] || elapsed=1
  rate=$((RUN_GAIN / elapsed))
  log ""
  log "summary: +$(human_bytes "$RUN_GAIN") in $(printf '%dm%02ds' $((elapsed / 60)) $((elapsed % 60))) (avg $(human_rate "$rate")), ${#RUN_PATHS[@]} file(s) touched, $RUN_QUARANTINED quarantined"
  for i in "${!RUN_PATHS[@]}"; do
    secs="${RUN_SECS[$i]}"
    [ "$secs" -gt 0 ] 2>/dev/null || secs=1
    printf '  %-6s %-46s +%-10s %s\n' \
      "${RUN_STATUS[$i]}" "${RUN_PATHS[$i]}" \
      "$(human_bytes "${RUN_BYTES[$i]}")" \
      "$(human_rate "$(( ${RUN_BYTES[$i]} / secs ))")"
  done
}

cmd_fetch() {
  HUB="${1:-}"
  NAME="${2:-}"
  if [ -z "$HUB" ]; then
    usage
    exit "$EXIT_USAGE"
  fi
  check_downloader
  [ -n "$NAME" ] || NAME="${HUB##*/}"
  DIR="$MODEL_DIR/$NAME"
  mkdir -p "$DIR"

  log "repo:   $HUB@$REVISION"
  log "target: $DIR"

  if ! ensure_manifest "$DIR" "$HUB" "$REVISION"; then
    printf 'metadata unavailable for %s (checked %s and resolve/index).\n' \
      "$HUB" "$ENDPOINT" >&2
    exit "$EXIT_META"
  fi
  printf '%s\n' "$HUB" >"$DIR/.source"

  if [ -f "$DIR/.fmd-files.tsv" ]; then
    if ensure_index "$DIR"; then
      write_manifest "$DIR" "$DIR/.fmd-files.tsv" "$HUB" "$REVISION" "$ENDPOINT"
    else
      printf '%s\n' "index missing; falling back to metadata order" >&2
    fi
    rm -f "$DIR/.fmd-files.tsv"
  fi

  local queue failed=0 skipped=0
  local -a failed_names=()
  if ! queue="$(build_queue "$DIR")"; then
    printf 'could not build a file list from %s/.fmd-manifest.json\n' "$DIR" >&2
    exit "$EXIT_META"
  fi

  check_disk "$DIR" "$queue" || exit "$EXIT_DISK"

  while IFS=$'\t' read -r size sha path; do
    [ -n "$path" ] || continue
    if ! budget_left; then
      printf 'budget: wall-clock budget (%ss) reached; %s not started\n' \
        "$WALL_BUDGET" "$path" >&2
      failed=1
      skipped=$((skipped + 1))
      failed_names+=("$path(budget)")
      continue
    fi
    if ! fetch_one "$path" "$size" "$sha"; then
      failed=1
      failed_names+=("$path")
    fi
  done <<<"$queue"

  rm -f "$DIR/.fmd-files.tsv"

  print_summary
  if [ "$failed" -ne 0 ]; then
    printf 'failed shards: %s\n' "${failed_names[*]}" >&2
  fi
  if verify_list "$DIR" <(manifest_read "$DIR/.fmd-manifest.json"); then
    log "done: $DIR ($(du -sh "$DIR" 2>/dev/null | cut -f1))"
    log "served locally — no network fetch at serve time:"
    log "  text:   FMD_VLLM_MODEL=$DIR scripts/serve_local.sh start"
    exit "$EXIT_OK"
  fi
  printf 'some files failed verification in %s; re-run this command to resume.\n' "$DIR" >&2
  exit "$EXIT_PARTIAL"
}

cmd_verify() {
  local dir="${1:-}" list="$TMPD/verify.tsv"
  if [ -z "$dir" ]; then
    usage
    exit "$EXIT_USAGE"
  fi
  if [ ! -d "$dir" ]; then
    printf 'no such directory: %s\n' "$dir" >&2
    exit "$EXIT_PARTIAL"
  fi
  DIR="$dir"
  log "verify: $dir"

  if [ -s "$dir/.fmd-manifest.json" ]; then
    # A manifest written before optional-file classification cannot tell a
    # "missing" file from a "not in this repo" file. Refresh it once if the Hub
    # is reachable; if it is not, fall back to the old required-by-default read.
    if ! grep -q '"required"' "$dir/.fmd-manifest.json" 2>/dev/null; then
      local mrepo mrev
      mrepo="$(manifest_field "$dir/.fmd-manifest.json" repo)"
      mrev="$(manifest_field "$dir/.fmd-manifest.json" revision)"
      if [ -n "$mrepo" ] && fetch_listing "$dir" "$mrepo" "${mrev:-$REVISION}" 2>/dev/null; then
        write_manifest "$dir" "$dir/.fmd-files.tsv" "$mrepo" "${mrev:-$REVISION}" "$ENDPOINT"
        rm -f "$dir/.fmd-files.tsv"
      else
        printf '%s\n' "note: pre-optional manifest and no Hub access; checking every listed file" >&2
      fi
    fi
    manifest_read "$dir/.fmd-manifest.json" >"$list"
  elif ! build_offline_list "$dir" >"$list"; then
    printf 'no manifest, no model.safetensors.index.json and no *.safetensors in %s\n' "$dir" >&2
    exit "$EXIT_META"
  else
    printf '%s\n' "note: no .fmd-manifest.json — checking safetensors headers only" >&2
  fi

  if [ ! -s "$list" ]; then
    printf 'nothing to verify in %s\n' "$dir" >&2
    exit "$EXIT_META"
  fi

  if verify_list "$dir" "$list"; then
    exit "$EXIT_OK"
  fi
  exit "$EXIT_PARTIAL"
}

cmd_repair() {
  local dir="${1:-}" manifest repo="" rev="" list="$TMPD/repair.tsv"
  local queue="" size sha kind path out
  local -a repair_queue=()
  if [ -z "$dir" ]; then
    usage
    exit "$EXIT_USAGE"
  fi
  if [ ! -d "$dir" ]; then
    printf 'no such directory: %s\n' "$dir" >&2
    exit "$EXIT_PARTIAL"
  fi
  DIR="$dir"
  check_downloader
  manifest="$dir/.fmd-manifest.json"

  if [ -s "$manifest" ]; then
    repo="$(manifest_field "$manifest" repo)"
    rev="$(manifest_field "$manifest" revision)"
  fi

  if [ -z "$repo" ]; then
    repo="${FMD_MODEL_REPO:-}"
    if [ -z "$repo" ] && [ -s "$dir/.source" ]; then
      repo="$(head -n 1 "$dir/.source")"
    fi
    rev="$REVISION"
    if [ -z "$repo" ]; then
      printf '%s\n' "cannot repair: no .fmd-manifest.json and no .source; set FMD_MODEL_REPO=<hub-repo>" >&2
      exit "$EXIT_META"
    fi
    log "manifest missing; rebuilding it from $repo@$rev"
    if ! fetch_listing "$dir" "$repo" "$rev"; then
      printf 'metadata unavailable for %s\n' "$repo" >&2
      exit "$EXIT_META"
    fi
    write_manifest "$dir" "$dir/.fmd-files.tsv" "$repo" "$rev" "$ENDPOINT"
  fi

  if [ -z "$repo" ]; then
    printf 'manifest in %s does not name a repo\n' "$dir" >&2
    exit "$EXIT_META"
  fi

  HUB="$repo"
  REVISION="$rev"
  # Always re-read the live listing during a repair: the whole point is to learn
  # which failures were real and which files the repo simply does not ship.
  if fetch_listing "$dir" "$repo" "$rev" 2>/dev/null; then
    ensure_index "$dir" || true
    write_manifest "$dir" "$dir/.fmd-files.tsv" "$repo" "$rev" "$ENDPOINT"
    rm -f "$dir/.fmd-files.tsv"
  else
    printf '%s\n' "metadata refresh unavailable; repairing from the existing manifest" >&2
    if [ -f "$dir/.fmd-files.tsv" ]; then
      ensure_index "$dir" || true
      write_manifest "$dir" "$dir/.fmd-files.tsv" "$repo" "$rev" "$ENDPOINT"
      rm -f "$dir/.fmd-files.tsv"
    fi
  fi
  manifest_read "$manifest" >"$list"

  log "repair: $dir ($repo@$rev)"
  while IFS=$'\t' read -r size sha kind required path; do
    [ -n "$path" ] || continue
    [ "${required:-1}" = "1" ] || continue
    if ! out="$(verify_entry "$dir" "$size" "$sha" "$path" "${required:-1}" 2>&1)"; then
      printf '%s\n' "$out"
      if [ -f "$dir/$path" ]; then
        quarantine "$dir/$path" "repair: failed verification ($(printf '%s' "$out" | head -n 1))"
      fi
      # A short .part is resumable, not evidence of corruption: keep it unless
      # it is at or past the declared size, where the bytes cannot be trusted.
      if [ -f "$dir/$path.part" ] && [ "$size" -gt 0 ] &&
        [ "$(stat -c %s "$dir/$path.part" 2>/dev/null || echo 0)" -ge "$size" ]; then
        quarantine "$dir/$path.part" "repair: partial at full size failed verification"
      fi
      repair_queue+=("$size"$'\t'"$sha"$'\t'"$path")
    fi
  done <"$list"

  if [ "${#repair_queue[@]}" -eq 0 ]; then
    log "nothing to repair: all files verified"
    exit "$EXIT_OK"
  fi

  queue="$(printf '%s\n' "${repair_queue[@]}")"
  check_disk "$dir" "$queue" || exit "$EXIT_DISK"

  local failed=0
  local -a failed_names=()
  while IFS=$'\t' read -r size sha path; do
    [ -n "$path" ] || continue
    if ! budget_left; then
      printf 'budget: wall-clock budget (%ss) reached; %s not started\n' \
        "$WALL_BUDGET" "$path" >&2
      failed=1
      failed_names+=("$path(budget)")
      continue
    fi
    if ! fetch_one "$path" "$size" "$sha"; then
      failed=1
      failed_names+=("$path")
    fi
  done <<<"$queue"

  printf '%s\n' "$repo" >"$dir/.source"
  print_summary
  if [ "$failed" -ne 0 ]; then
    printf 'failed shards: %s\n' "${failed_names[*]}" >&2
  fi
  if verify_list "$dir" <(manifest_read "$manifest"); then
    exit "$EXIT_OK"
  fi
  exit "$EXIT_PARTIAL"
}

case "${1:-}" in
  verify) cmd_verify "${2:-}" ;;
  repair) cmd_repair "${2:-}" ;;
  speed | probe) cmd_speed "${2:-}" ;;
  fetch) cmd_fetch "${2:-}" "${3:-}" ;;
  -h | --help | help) usage; exit "$EXIT_OK" ;;
  *) cmd_fetch "${1:-}" "${2:-}" ;;
esac
