#!/usr/bin/env bash
# Prefetch a model ONCE into the host weight root, resumably, shard by shard.
#
#   scripts/fetch_model.sh <hub-repo> [target-name]
#   scripts/fetch_model.sh verify <dir>     re-check an existing directory
#   scripts/fetch_model.sh repair <dir>     delete invalid files, re-fetch them
#
# Examples:
#   scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ
#   scripts/fetch_model.sh repair /home/archit-gore/models/Qwen3.5-9B-AWQ
#
# Why this exists: weights are host data under MODEL_DIR (default
# /home/archit-gore/models), mounted read-only into containers. Image builds and
# docker volumes never hold weights, so rebuilding or pruning cannot force a
# re-download. This script is the only place a download is allowed, and it is
# run by a human on the host, never from a Dockerfile.
#
# Behaviour:
#   * reads `model.safetensors.index.json` and fetches each shard independently;
#   * `curl -C -` per file into `<file>.part`; the part file is only renamed
#     into place after size + safetensors header (+ sha256 when known) pass;
#   * survives dead transfers: per-attempt stall detection, exponential backoff,
#     a per-file attempt cap, and curl's own retry loop;
#   * detects servers that ignore `Range` (a 200 while resuming) and restarts
#     that file from zero instead of appending a second copy;
#   * refuses to start when the remaining bytes do not fit the filesystem;
#   * verifies LFS sha256 when the Hub exposes it, else records
#     `sha256: unavailable` in `.fmd-manifest.json`.
#
# Environment:
#   FMD_MODEL_DIR            host weight root (default $MODEL_DIR, else
#                            /home/archit-gore/models)
#   MODEL_DIR                fallback for the same value (see .env.example)
#   FMD_MODEL_REVISION       revision/branch (default main)
#   HF_ENDPOINT              mirror, e.g. https://hf-mirror.com (default
#                            https://huggingface.co)
#   HF_TOKEN                 optional, for gated repos; never printed
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
#
# Exit codes:
#   0  all files verified
#   2  some files failed (names printed)
#   3  not enough disk space
#   4  metadata / index unavailable
#   64 usage error
#
# Deliberately curl + stdlib json, not `huggingface-cli`: the runtime venv must
# not grow a dependency, and per-file resume is the useful part.

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

HUB=""
NAME=""
DIR=""

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/fmd-fetch.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT

usage() {
  cat >&2 <<'EOF'
usage:
  scripts/fetch_model.sh <hub-repo> [target-name]   fetch/resume into MODEL_DIR
  scripts/fetch_model.sh verify <dir>               re-check a downloaded dir
  scripts/fetch_model.sh repair <dir>               delete bad files, re-fetch

environment: FMD_MODEL_DIR, FMD_MODEL_REVISION, HF_ENDPOINT, HF_TOKEN,
             FMD_FETCH_ATTEMPTS, FMD_CURL_RETRY, FMD_CONNECT_TIMEOUT,
             FMD_SPEED_LIMIT, FMD_SPEED_TIME, FMD_DISK_RESERVE_BYTES,
             FMD_VERIFY_SHA256, FMD_MODEL_REPO
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
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)

seen = set()
for sibling in data.get("siblings") or []:
    path = sibling.get("rfilename") or ""
    if not path or path.endswith("/") or path.startswith(".") or path in seen:
        continue
    seen.add(path)
    size = sibling.get("size")
    if not isinstance(size, int) or size < 0:
        size = 0
    lfs = sibling.get("lfs") or {}
    sha = lfs.get("sha256") or "-"
    print(f"{size}\t{sha}\t{path}")
PY
}

write_manifest() {
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import datetime
import json
import os
import sys

directory, tsv, repo, revision, endpoint = sys.argv[1:6]

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
        })

manifest = {
    "repo": repo,
    "revision": revision,
    "endpoint": endpoint,
    "generated_at": datetime.datetime.now(datetime.timezone.utc)
    .replace(microsecond=0)
    .isoformat(),
    "integrity": "sha256"
    if any(item["sha256"] for item in files)
    else "size+header only (sha256 unavailable)",
    "files": files,
}

tmp = os.path.join(directory, ".fmd-manifest.json.tmp")
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp, os.path.join(directory, ".fmd-manifest.json"))
PY
}

manifest_read() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    manifest = json.load(fh)
for item in manifest.get("files") or []:
    sha = item.get("sha256") or "-"
    print(f'{item.get("size", 0)}\t{sha}\t{item.get("kind", "aux")}\t{item["path"]}')
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
  api="$ENDPOINT/api/models/$repo/revision/$rev?blobs=true"
  if "${CURL[@]}" -L "${CURL_AUTH[@]}" --fail -H 'Accept: application/json' \
    --max-time 120 -o "$TMPD/hub.json" "$api" 2>>"$TMPD/curl.err"; then
    normalize_hub_json "$TMPD/hub.json" >"$dir/.fmd-files.tsv"
    if [ -s "$dir/.fmd-files.tsv" ]; then
      enrich_sha_from_pointers "$dir" "$repo" "$rev"
      return 0
    fi
  fi

  # Fallback: the index alone still names every shard. Sizes come from HEAD.
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
  local dir=$1 repo=$2 rev=$3 mrepo mrev
  if [ -s "$dir/.fmd-manifest.json" ]; then
    mrepo="$(manifest_field "$dir/.fmd-manifest.json" repo)"
    mrev="$(manifest_field "$dir/.fmd-manifest.json" revision)"
    if [ "$mrepo" = "$repo" ] && [ "$mrev" = "$rev" ]; then
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
    order = [path for path in files if path.endswith(".safetensors")]
order += [path for path in files if path not in order]

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

verify_entry() {
  local dir=$1 size=$2 sha=$3 path=$4
  local file="$dir/$path" actual hdr="" got=""
  if [ ! -f "$file" ]; then
    printf 'FAIL %s: missing\n' "$path"
    return 1
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
  local dir=$1 list=$2 size sha kind path ok=0 bad=0
  local -a failures=()
  while IFS=$'\t' read -r size sha kind path; do
    [ -n "$path" ] || continue
    if verify_entry "$dir" "$size" "$sha" "$path"; then
      ok=$((ok + 1))
    else
      bad=$((bad + 1))
      failures+=("$path")
    fi
  done <"$list"
  log "$ok file(s) verified, $bad failed"
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
      printf '0\t-\tshard\t%s\n' "$path"
      count=$((count + 1))
    done < <(index_shards "$dir/model.safetensors.index.json" || true)
  fi
  if [ "$count" -eq 0 ]; then
    shopt -s nullglob
    for f in "$dir"/*.safetensors; do
      printf '0\t-\tshard\t%s\n' "$(basename "$f")"
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

fetch_one() {
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
      log "corrupt $path at full size — refetching"
      rm -f "$out"
    elif [ "$expected" -gt 0 ] && [ "$have" -gt "$expected" ]; then
      log "oversize $path ($have > $expected) — discarding"
      rm -f "$out"
    elif [ ! -f "$part" ]; then
      # Preserve bytes left by an older/easier-to-interrupt run.
      mv -f "$out" "$part"
    else
      rm -f "$out"
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
      log "corrupt $path partial at full size — restarting it"
      rm -f "$part"
    elif [ "$expected" -gt 0 ] && [ "$psz" -gt "$expected" ]; then
      log "oversize $path partial ($psz > $expected) — discarding"
      rm -f "$part"
    fi
  fi

  log "fetch  $path ($size_label, have $(human_bytes "$(stat -c %s "$part" 2>/dev/null || echo 0)"))"

  while [ "$attempt" -le "$ATTEMPTS" ]; do
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

    printf '  %-44s attempt %d/%d  +%s  %s on disk  %s/s  http=%s\n' \
      "$path" "$attempt" "$ATTEMPTS" "$(human_bytes "$got")" \
      "$(human_bytes "$end")" "$(human_bytes "$((got / elapsed))")" "$code"

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
      rm -f "$part"
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
      log "  $path: verification failed — $reason; discarding and retrying"
      rm -f "$part"
    elif [ "$expected" -gt 0 ] && [ "$end" -gt "$expected" ]; then
      log "  $path: $end > expected $expected bytes — discarding partial"
      rm -f "$part"
    elif [ "$rc" -ne 0 ] && [ "$expected" -gt 0 ] && [ "$end" -gt 0 ] && [ "$end" -ge "$expected" ]; then
      rm -f "$part"
    fi

    attempt=$((attempt + 1))
    if [ "$attempt" -le "$ATTEMPTS" ]; then
      delay=$((BACKOFF_BASE * (1 << (attempt - 2))))
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
  return 1
}

# ---- commands ------------------------------------------------------------

cmd_fetch() {
  HUB="${1:-}"
  NAME="${2:-}"
  if [ -z "$HUB" ]; then
    usage
    exit "$EXIT_USAGE"
  fi
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

  if ensure_index "$DIR"; then
    write_manifest "$DIR" "$DIR/.fmd-files.tsv" "$HUB" "$REVISION" "$ENDPOINT"
  else
    printf '%s\n' "index missing; falling back to metadata order" >&2
  fi

  local queue failed=0
  local -a failed_names=()
  if ! queue="$(build_queue "$DIR")"; then
    printf 'could not build a file list from %s/.fmd-manifest.json\n' "$DIR" >&2
    exit "$EXIT_META"
  fi

  check_disk "$DIR" "$queue" || exit "$EXIT_DISK"

  while IFS=$'\t' read -r size sha path; do
    [ -n "$path" ] || continue
    if ! fetch_one "$path" "$size" "$sha"; then
      failed=1
      failed_names+=("$path")
    fi
  done <<<"$queue"

  rm -f "$DIR/.fmd-files.tsv"

  log ""
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
  if [ -f "$dir/.fmd-files.tsv" ]; then
    ensure_index "$dir" || true
    write_manifest "$dir" "$dir/.fmd-files.tsv" "$repo" "$rev" "$ENDPOINT"
    rm -f "$dir/.fmd-files.tsv"
  fi
  manifest_read "$manifest" >"$list"

  log "repair: $dir ($repo@$rev)"
  while IFS=$'\t' read -r size sha kind path; do
    [ -n "$path" ] || continue
    if ! out="$(verify_entry "$dir" "$size" "$sha" "$path" 2>&1)"; then
      printf '%s\n' "$out"
      log "repair: removing invalid $path"
      rm -f "$dir/$path" "$dir/$path.part"
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
    if ! fetch_one "$path" "$size" "$sha"; then
      failed=1
      failed_names+=("$path")
    fi
  done <<<"$queue"

  printf '%s\n' "$repo" >"$dir/.source"
  log ""
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
  fetch) cmd_fetch "${2:-}" "${3:-}" ;;
  -h | --help | help) usage; exit "$EXIT_OK" ;;
  *) cmd_fetch "${1:-}" "${2:-}" ;;
esac
