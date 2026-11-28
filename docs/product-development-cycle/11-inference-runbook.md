# 11 — Inference runbook (local text model + OCR stage)

**Status:** engine facts measured on this host 2026-11-17 (§2.7, llama.cpp +
Gemma 4 26B-A4B). §2.1–§2.6 describe the superseded vLLM/9B path and are kept
as history. **Owner:** WS-SERVE.
**Companions:** `10-inference-and-model-selection.md` (why these models),
`config/models.yaml` (what the router calls), `config/acquire.yaml` (the OCR
stage), `09-local-runbook.md` (the rest of the stack).

**Temporary API extractor (2026-11-16):** while the local 9B shards are
truncated, `.env` sets `FMD_EXTRACT_PROVIDER=api` and
`FMD_EXTRACT_MODEL=deepseek-v4.1-flash-free`; the local pair stays commented.
The API key's free quota is account-wide, so a `429 FreeUsageLimitError` (with
`retry-after` in seconds) means every `-free` model is exhausted until reset —
it is not an extraction-quality failure. Revert to the local pair once the
weights pass the integrity check in §2.2.

Three processes, deliberately separate — and never two GPU-resident models at
once (§1 budget rule):

| Process | What it is | Port | Who calls it |
|---|---|---|---|
| `scripts/serve_local.sh` | vLLM, OpenAI-compatible `/v1`, text model | 8010 | `fmd.llm.router` (`bulk_extract`, `verify`, `entity_adjudication`) |
| `scripts/serve_ocr.sh vision-start` | vLLM, OpenAI-compatible `/v1`, **Qwen3-VL-8B** | 8001 | `infra/ocr/server.py` (the `qwen3vl` backend) |
| `scripts/serve_ocr.sh start` | OCR adapter behind `POST /extract` | 8110 | `fmd.acquire.tiers` tier T2 |

OCR is **not** a router profile. `ocr_vlm` was retired from `config/models.yaml`
on 2026-11-16: a block-extraction service is not an OpenAI chat endpoint, it
returns `{blocks:[{block_bbox, block_order, block_content}]}` rather than a JSON
schema completion, and keeping it as a profile made `fmd llm doctor` advertise a
capability the pipeline never used.

## 1. Host facts (measured)

```
GPU      NVIDIA GeForce RTX 5080 Laptop, 16303 MiB, sm_120 (12, 0), driver 580 / CUDA 13.0
Python   3.12.3 · uv 0.12.22 · Docker 29.1.3 · Compose 2.40
venv     ~/.venvs/fmd-vllm  →  vllm 0.33.0, torch 2.14.0+cu130, xgrammar 0.2.8
```

`torch.cuda.get_arch_list()` on that venv:

```
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

**sm_120 is supported by this stack** — no source build, no nightly index, no
patch. SGLang has no sm_120 wheels; vLLM ≥ 0.19 does, and 0.33 resolves to a
cu130 torch automatically.

Budget rule: one GPU process at a time. A 32k-context 9B int4 run reserves
enough KV that an 8B vision model will not co-reside in 16 GB, so T2b
(Qwen3-VL) is an alternative to the text model, not an addition.

### 1.1 What runs where, and where weights live

| Thing | Runs | Host disk | In a container? |
|---|---|---|---|
| vLLM text (:8010) | host process | `$MODEL_DIR/Qwen3.5-9B-AWQ` | no |
| vLLM vision (:8001) | host process | `$MODEL_DIR/Qwen3-VL-8B-Instruct-FP8` | no |
| OCR adapter (:8110) | container | mounts `$MODEL_DIR` **read-only** at `/models` | client only |
| `api` / `worker` | container | no weights needed | client only |

`MODEL_DIR` defaults to `/home/archit-gore/models`. Layout:

```
/home/archit-gore/models/
├── Qwen3.5-9B-AWQ/             text: bulk_extract, verify, entity_adjudication
├── Qwen3.5-2B-AWQ/             smoke-test text model (optional)
├── Qwen3-VL-8B-Instruct-FP8/   vision: acquire T2 (optional; not yet downloaded)
└── paddlex/                    T2b Paddle cache (only directory written by a container)
```

Rules that make this safe:

* **No image downloads weights.** `infra/Dockerfile.ocr` installs Python + pdfium
  and copies `server.py`; there is no `snapshot_download`. Rebuilding is cheap.
* **No named volume holds weights.** `paddlecache` was removed 2026-11-16; the
  Paddle cache is the host bind `$MODEL_DIR/paddlex`.
* **Serving is given an absolute directory**, never a Hub id, so vLLM cannot fall
  back to a network fetch. `scripts/serve_local.sh` / `serve_ocr.sh` refuse to
  start when the directory is absent and name the prefetch command.
* Downloads happen only in `scripts/hf_download.py` (preferred, §2.2) or
  `scripts/fetch_model.sh` (fallback), run by a human on the host.

**Container → host connectivity.** Linux Docker has no `host.docker.internal`;
every client service declares
`extra_hosts: ["host.docker.internal:host-gateway"]` (`api`, `worker`, `ocr`,
`ocr-paddle`). The URL has one name per consumer: `FMD_LOCAL_LLM_URL` for the
router (compose default `http://host.docker.internal:8010/v1`, host default
`http://localhost:8010/v1`) and `FMD_OCR_VLLM_URL` for the adapter. Port scheme:
**app API 8000** (published by `api`), **text vLLM 8010**, **vision vLLM 8001**,
**OCR adapter 8110**. A text model on :8000 would collide with the published API
port and is unreachable from any container; that collision was hit on
2026-11-16 and is why the default moved. Verify from inside the container, not
from the host shell:

```bash
docker compose -f infra/docker-compose.yml exec api \
  python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:8010/v1/models',timeout=5).status)"
# expect: 200
```

**`hard_pdf` choice (one line):** it is the *hosted fallback* (`api` provider,
model = `FMD_API_MODEL`); local vision is the acquire T2 stage on :8001, not a
router profile, so pairing provider `local` with an API model id was the
2026-11-16 defect and is gone.

## 2. Text model — copy-paste

### 2.1 Install (once, ~60–90 min on a slow link; 8.9 GB of packages)

```bash
scripts/serve_local.sh install
```

Creates `~/.venvs/fmd-vllm` and installs vLLM there. **Separate venv on
purpose**: vLLM pins torch/triton; installing it into `packages/pipeline/.venv`
would move the pipeline's torch and break `make test` for unrelated reasons.

Verify the arch list, this is the sm_120 gate:

```bash
~/.venvs/fmd-vllm/bin/python -c \
 "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
# expect: 2.14.0+cu130 13.0 [... 'sm_100', 'sm_120']
```

### 2.2 Weights

There is **no `Qwen/Qwen3.5-9B-AWQ` repo on the Hub** (checked 2026-11-16). The
closest genuine AWQ conversion is `QuantTrio/Qwen3.5-9B-AWQ` (quant_method awq,
4-bit, group 128), served under the alias `Qwen3.5-9B-AWQ` so
`config/models.yaml` stays stable.

**Preferred downloader.** `scripts/hf_download.py` — a verbatim copy of the
operator's `~/AI/hf_downloader.py` (read-only original), whose only changes are
the four constants below, which now come from the environment. It is a single
process, plain `requests`, no xet/aria2 dependency; one file at a time,
HTTP `Range` resume, 20 attempts per file with backoff 5 s → 120 s.

```bash
cd ~/projects/findMyDreamCollege
setsid nohup env \
  HF_REPO_ID=QuantTrio/Qwen3.5-9B-AWQ \
  HF_REPO_SUBDIR= \
  HF_REVISION=main \
  HF_OUTPUT_DIR="$HOME/models/Qwen3.5-9B-AWQ" \
  ~/.venvs/hf-downloader/bin/python -u scripts/hf_download.py \
  </dev/null >/tmp/opencode/hf-dl.log 2>&1 &
tail -f /tmp/opencode/hf-dl.log
```

| variable | meaning | original default (kept) |
|---|---|---|
| `HF_REPO_ID` | Hub repo | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` |
| `HF_REPO_SUBDIR` | subfolder to mirror; empty ⇒ repo root | `IQ3_XXS` |
| `HF_REVISION` | branch/tag/sha | `main` |
| `HF_OUTPUT_DIR` | destination directory | `/media/archit/Windows/AI/Models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` |

It needs `huggingface_hub` for the file listing (`uv venv ~/.venvs/hf-downloader
&& uv pip install huggingface_hub requests`). `HF_TOKEN` is read from the
environment if present and never printed; with no token and no tty it proceeds
anonymously, which is correct for `QuantTrio/Qwen3.5-9B-AWQ` (public).

What it resumes: files are written under their real names (no `.part`), and
every retry re-`stat`s the file and asks for `Range: bytes=<size>-`; a server
that answers `200` to a `Range` request causes a restart from byte 0 rather
than an append. Re-running the same command is the resume: complete files
(byte size == Hub size) are skipped, short files continue, oversized files are
deleted and re-fetched. It never writes a `.done` or manifest file — completion
is size equality only, and `VERIFY_SHA256 = False` by default.

Verify afterwards, and trust `fetch_model.sh` for the authoritative check:

```bash
scripts/fetch_model.sh verify ~/models/Qwen3.5-9B-AWQ     # exit 0 before serving
```

**Fallback downloader.** `scripts/fetch_model.sh` remains supported for
everything else (mirrors, `speed`, `repair`, gated repos, byte-range `curlx4`).
Use it when the preferred downloader cannot be used:

```bash
scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ
ls -l "$MODEL_DIR/Qwen3.5-9B-AWQ"       # expect 5 safetensors + configs
```

The Hub file listing — `/api/models/<repo>/tree/<rev>?recursive=true` — is the
authority on what the repo contains. Three states follow from it:

| state | meaning | effect |
|---|---|---|
| **required** | listed with a size (LFS `oid` when it is a weight file) | fetched, size + safetensors header + sha256 verified |
| **optional** | not listed, or listed by an older manifest but since removed upstream | never fetched, never fails verification, never quarantined |
| **corrupt** | listed, but local size/hash disagrees | quarantined, then re-fetched |

`video_preprocessor_config.json` and `vocab.json` are **only** required if the
current tree lists them — for `QuantTrio/Qwen3.5-9B-AWQ@main` it does, but a
stale manifest that names a file the repo no longer ships is demoted to
`optional` on the next run instead of failing forever. Files that are simply
absent from the listing are reported as `OPTIONAL <name>: not listed upstream`
and exit 0.

Each shard downloads into `<shard>.part`; the part file is renamed only after
the size matches `.fmd-manifest.json`, the safetensors header parses (8-byte
JSON length + tensor count + declared payload end), and, when the Hub exposes
LFS metadata, the sha256 matches. Completion is never inferred from
`Content-Length`. Disk space is checked against the remaining bytes first
(exit 3 rather than a full disk); `FMD_DISK_RESERVE_BYTES` keeps headroom.

**After a disconnect, run the same command again.** Bytes in `.part` are kept
and `curl -C -` (or `aria2c -c`) resumes at that offset. Each attempt prints
progress (bytes this session, total on disk, rate, HTTP status). A transfer
below 10 KiB/s for 60 s is killed; a zero-progress attempt backs off at least
30 s, then 5 s → 300 s exponential, capped at 5 min. `FMD_FETCH_ATTEMPTS`
(default 10) caps attempts per file. `FMD_WALL_BUDGET=43200` stops *starting*
new files after 12 h and still prints the summary. If a server ignores `Range`
(a 200 while resuming), the file is restarted from zero instead of appended
twice.

At the end of every fetch/repair:

```
summary: +3.1 GiB in 2h14m (avg 402.11 KiB/s), 6 file(s) touched, 0 quarantined
  ok     model-00001-of-00005.safetensors   +2.8 GiB  404.10 KiB/s
  FAIL   model-00003-of-00005.safetensors   +291 MiB  38.20 KiB/s
```

**Bad files are never deleted.** `repair` moves them to
`<dir>/.bad/<utc-timestamp>/<path>` and writes `<path>.reason` next to the
moved file, so a failed night can be diagnosed afterwards:

```bash
find "$MODEL_DIR/Qwen3.5-9B-AWQ/.bad" -name '*.reason' -exec sh -c 'echo "$1:"; cat "$1"' _ {} \;
```

A short `.part` is evidence of a disconnect, not corruption, so repair keeps it
for resume; only a `.part` at or past the declared size is quarantined.

**Verify an existing directory** (offline; no Hub needed):

```bash
scripts/fetch_model.sh verify "$MODEL_DIR/Qwen3.5-9B-AWQ"
# PASS model-00001-of-00005.safetensors (… bytes, header=… tensors=…)
# OPTIONAL vocab.json: not listed upstream, absent locally
```

Exit codes: `0` all files pass, `2` some failed (names printed), `3` disk
space, `4` metadata/index unavailable, `64` usage error.

**Repair a directory.** `repair` refreshes the listing first (so absent files
stop being reported as missing), quarantines only what actually fails, and
re-fetches exactly those files; valid shards are never re-downloaded:

```bash
# directories fetched by the old script have no manifest; name the repo:
FMD_MODEL_REPO=QuantTrio/Qwen3.5-9B-AWQ \
  scripts/fetch_model.sh repair "$MODEL_DIR/Qwen3.5-9B-AWQ"
scripts/fetch_model.sh verify "$MODEL_DIR/Qwen3.5-9B-AWQ"
```

**Expected sizes.** `.fmd-manifest.json` records the exact byte size of every
file on first contact; `verify` is the authority. Reference points:
`QuantTrio/Qwen3.5-9B-AWQ@main` is **11.5 GiB (12.2 GB)** across 5 safetensors
shards (2.8 / 2.8 / 2.8 / 2.8 / 0.40 GiB) plus ~20 MB of configs and
tokenizer; Qwen3.5-2B-AWQ ≈3.2 GB, Qwen3-VL-8B-Instruct-FP8 ≈10.6 GB (two
shards). A shard shorter than the manifest says is truncated — repair it; do
not serve it.

**Diagnosing a slow link (2.3 KiB/s and friends).** Retries do not fix
throughput; measure it. `speed` probes the largest file in the repo with a
bounded range request and reports MiB/s per method, then recommends one:

```bash
scripts/fetch_model.sh speed QuantTrio/Qwen3.5-9B-AWQ
# method                       throughput  note
# curl -4 single stream      225.08 KiB/s
# curl -6 single stream               n/a  no IPv6 route
# curl -4 x4 parallel        341.33 KiB/s  4 streams, 4.0 MiB in 12s
# mirror https://hf-mirror.com  84.11 KiB/s
# mirror x4 parallel         182.51 KiB/s  4 streams, 3.6 MiB in 20s
# fastest: curl -4 x4 parallel (FMD_DOWNLOADER=aria2) at 341.33 KiB/s
```

It measures IPv4 vs IPv6, single stream vs `FMD_PROBE_CONNECTIONS` parallel
streams, and the alternate endpoint `FMD_MIRROR` (default
`https://hf-mirror.com`). Each probe is capped by `FMD_PROBE_BYTES` (8 MiB) and
`FMD_PROBE_TIME` (20 s), so the whole diagnosis costs a few MiB. Below
`FMD_SLOW_THRESHOLD` (1 MiB/s) it prints the sneakernet path instead of a
command that would take days.

**Switch mirror or add connections.**

```bash
# alternate endpoint, no other change (token, resume and verification follow):
HF_ENDPOINT=https://hf-mirror.com scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ

# multi-connection: needs aria2c on PATH; auto-detected, falls back to curl
FMD_DOWNLOADER=aria2 FMD_ARIA2_CONNECTIONS=8 \
  scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ
```

`FMD_DOWNLOADER=auto` (default) uses `aria2c -c -x8 -s8 -k1M --max-tries=0
--retry-wait=5 --lowest-speed-limit=10K --timeout=60 --connect-timeout=30`
when it is installed and no `HF_TOKEN` is set, else single-stream curl. aria2
is deliberately skipped for gated repos: it has no `--header=@file`, so a token
would be visible in `ps`. Parallel streams only help when the cap is
per-connection; `speed` tells you whether that is the case here.

**When the host throttles.** `HF_ENDPOINT`, `HTTPS_PROXY` and IPv4-vs-IPv6 all
work without other flags. Lower `FMD_SPEED_LIMIT` only when the link is
genuinely slow but alive; raise `FMD_SPEED_TIME` if a 60 s stall probe is too
aggressive. Gated repos need `HF_TOKEN` (read from a 0600 file by curl; never
printed or placed on the command line).

**Sneakernet.** If `speed` stays under 1 MiB/s, downloading 11.5 GiB here is
not an overnight job (at 2.3 KiB/s it is ~61 days; at 300 KiB/s ~11 h of
perfect link). Fetch on a faster host and copy:

```bash
# on the fast host
hf download QuantTrio/Qwen3.5-9B-AWQ --local-dir ~/models/Qwen3.5-9B-AWQ
tar -C ~/models -czf qwen9b.tgz Qwen3.5-9B-AWQ    # ~11.5 GiB
# on this host
rsync -avP --partial user@fasthost:~/models/Qwen3.5-9B-AWQ/ "$MODEL_DIR/Qwen3.5-9B-AWQ/"
scripts/fetch_model.sh repair "$MODEL_DIR/Qwen3.5-9B-AWQ"   # fills any gap
scripts/fetch_model.sh verify "$MODEL_DIR/Qwen3.5-9B-AWQ"   # exit 0 before serving
```

The `hf` CLI in `packages/pipeline/.venv` (xet, chunked, resumable) is a good
choice on the fast host: it measured about the same as 4 parallel curl streams
here, and it cannot report resumable byte offsets the way `.part` files do.

Do **not** point vLLM at the Hub id for day-to-day serving: a Hub id can
silently fetch, a local directory cannot.

A 3.2 GB genuine-AWQ alternative for smoke tests is `QuantTrio/Qwen3.5-2B-AWQ`
(same architecture, same vendor) — useful while the 9B is still downloading.

### 2.3 Serve

```bash
scripts/serve_local.sh start        # background, pid file, log file
scripts/serve_local.sh fg           # foreground (Ctrl-C)
```

Exact command the script runs:

```bash
~/.venvs/fmd-vllm/bin/vllm serve "${FMD_VLLM_MODEL:-QuantTrio/Qwen3.5-9B-AWQ}" \
  --served-model-name "${FMD_SERVED_NAME:-Qwen3.5-9B-AWQ}" \
  --host 0.0.0.0 --port "${FMD_VLLM_PORT:-8010}" \
  --max-model-len "${FMD_MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${FMD_VLLM_GPU_MEM_UTIL:-0.90}" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --trust-remote-code
```

Flags that do **not** exist in 0.33 and abort startup if passed:
`--chat-template-kwargs` (renamed `--default-chat-template-kwargs`) and
`--guided-json-backend` (structured outputs use `StructuredOutputsConfig`,
`backend='auto'` → xgrammar). `--enable-auto-tool-choice` needs a
`--tool-call-parser`; the router does not use tool calling.

### 2.4 Health checks

```bash
curl -s localhost:8010/v1/models | python3 -m json.tool | head
# expect: {"object":"list","data":[{"id":"Qwen3.5-9B-AWQ", ...}]}

curl -s localhost:8010/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen3.5-9B-AWQ","messages":[{"role":"user","content":"Say OK."}],"max_tokens":32}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
# expect: "OK"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Then the router-level check inside the pipeline:

```bash
cd packages/pipeline
FMD_EXTRACT_PROVIDER=local FMD_LOCAL_LLM_URL=http://localhost:8010/v1 \
  .venv/bin/python -m fmd.cli llm doctor --live
# expect: bulk_extract usable=true (live), no blockers; keys are never printed
```

Guided-JSON check (this is unverified claim #3 in doc 10):

```bash
curl -s localhost:8010/v1/chat/completions -H 'content-type: application/json' -d '{
 "model":"Qwen3.5-9B-AWQ","temperature":0,"max_tokens":128,
 "messages":[{"role":"user","content":"Return the JSON object {\"ok\": true}."}],
 "response_format":{"type":"json_schema","json_schema":{"name":"Probe","strict":true,
   "schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}}}' \
| python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
# expect: {"ok": true} — a schema violation means xgrammar is not constraining
```

### 2.5 Stop

```bash
scripts/serve_local.sh stop    # SIGTERM to the process group, SIGKILL after 30 s
```

The pid file is `/tmp/opencode/fmd-serve/vllm.pid`, the log
`/tmp/opencode/fmd-serve/vllm.log`, so `pkill -f "vllm serve"` is never needed.
Nothing is installed as a systemd unit or a compose service: the model is a
developer process.

### 2.6 Switch local ↔ cloud

No YAML edit; the router reads env at load.

```bash
# local (default)
FMD_EXTRACT_PROVIDER=local
FMD_LOCAL_LLM_URL=http://host.docker.internal:8010/v1
FMD_EXTRACT_MODEL=Qwen3.5-9B-AWQ

# hosted OpenAI-compatible
FMD_EXTRACT_PROVIDER=api
FMD_API_LLM_URL=https://api.openai.com/v1
FMD_API_LLM_KEY=...
FMD_API_MODEL=<hosted id>          # shared fallback for api-provider profiles

# offline
FMD_EXTRACT_PROVIDER=mock
```

`FMD_EXTRACT_MODEL` must equal `--served-model-name`; a mismatch is a `404 model
not found`, not a schema error. `hard_pdf` reads `FMD_API_MODEL` directly
(`FMD_HARD_PDF_MODEL` was removed 2026-11-16).

## 2.7 Text model of record — Gemma 4 26B-A4B via llama.cpp (2026-11-17)

§2.1–§2.6 above describe the vLLM/9B path and are kept as history. **The
current text model is a GGUF served by `llama-server`.** Everything below was
measured on this host on 2026-11-17 unless marked **[modelled]**.

### A. Asset — exactly one file

HF tree listing of `unsloth/gemma-4-26B-A4B-it-GGUF` (root, `?recursive=true`):

| File | Bytes |
|---|---|
| `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` ← **the one we serve** | 13,597,177,568 (12.66 GiB) |
| `gemma-4-26B-A4B-it-UD-IQ4_NL.gguf` | 13,613,037,280 |
| `gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf` | 12,907,280,096 |
| `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | 17,010,980,576 (does not fit) |
| `gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf` | 21,217,769,184 (does not fit) |
| `mtp-gemma-4-26B-A4B-it.gguf` (Q8_0 drafter) | 461,766,816 |
| `mmproj-F16.gguf` (vision projector) | 1,193,058,784 |

**No sidecar files are needed for text.** Tokenizer, chat template and
hyper-parameters are inside the GGUF (`--jinja` reads the embedded template).
Skip `mmproj-*` (vision) and `MTP/*` (only for speculative decoding).
A partial download keeps its bytes under
`$MODEL_DIR/gemma-4-26B-A4B-it-UD-IQ4_XS/` and resumes; the tree has ~65 GB of
quants, so **always filter**:

```bash
HF_REPO_ID=unsloth/gemma-4-26B-A4B-it-GGUF HF_REPO_SUBDIR= HF_REVISION=main \
HF_INCLUDE='gemma-4-26B-A4B-it-UD-IQ4_XS.gguf' \
HF_OUTPUT_DIR="$HOME/models/gemma-4-26B-A4B-it-UD-IQ4_XS" \
  ~/.venvs/hf-downloader/bin/python -u scripts/hf_download.py
```

`HF_INCLUDE` is a comma/space separated list of case-insensitive globs applied
after the `HF_REPO_SUBDIR` prefix is stripped; empty (default) reproduces the
old "everything in the subdir" behaviour.

### B. Engine — source build, not the prebuilt tarball

The release asset `llama-b12468-bin-ubuntu-cuda-12.8-x64.tar.gz` links
`libcudart.so.12`; this host only has the CUDA **13** runtime. Rather than pull
the extra 594 MB cudart asset, `scripts/install_llama_cpp.sh` builds from source
against the installed toolkit:

```bash
scripts/install_llama_cpp.sh            # ~6 min, 12 jobs, arch sm_120
scripts/install_llama_cpp.sh --verify
```

Result **[measured]**:

```
version: 1.0.0-dev (build 1, commit 140a279)     # tag b12468, 2026-11-17
built with GNU 13.3.0 for Linux x86_64
Available devices:
  CUDA0: NVIDIA GeForce RTX 5080 Laptop GPU (15839 MiB, 15623 MiB free)
```

Configured with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`. The meson
header prints the supported FA KV types:
`f16-f16; q4_0-q4_0; q8_0-q8_0; bf16-bf16; f8_e4m3-f8_e4m3` — so **q8_0 KV with
flash attention is a first-class combination**, not a fallback.

Note the card reports **15839 MiB / 15623 MiB free** to llama.cpp, ~460 MiB less
than `nvidia-smi`'s 16303 MiB (desktop compositor). Budget against 15.26 GiB,
not 16.

### C. KV budget — the numbers that decide context

Hyper-parameters read straight from the GGUF header (not estimated):

```
gemma4.block_count                  30
gemma4.attention.sliding_window     1024
sliding_window_pattern              True x5, False,  (5 full-attention layers: 5,11,17,23,29)
full layers : head_count_kv 2, key_length 512, value_length 512
SWA  layers : head_count_kv 8, key_length 256, value_length 256
```

So per token: **10.62 KiB** of q8_0 KV on the 5 full layers, and the 25 SWA
layers saturate at a fixed **106 MiB** (window 1024, upstream SWA trimming on by
default — do not pass `--swa-full`). That is why long context is unusually cheap
on this model:

| ctx | KV q8_0 | KV f16 | weights+KV (q8) | left of 15.26 GiB | headroom at ~0.5 GiB overhead |
|---|---|---|---|---|---|
| 16k | 0.27 GiB | 0.51 GiB | 12.93 GiB | 2.33 GiB | 1.83 |
| 24k | 0.35 GiB | 0.66 GiB | 13.02 GiB | 2.24 GiB | 1.74 |
| 32k | 0.44 GiB | 0.82 GiB | 13.10 GiB | 2.16 GiB | 1.66 |
| 40k | 0.52 GiB | 0.98 GiB | 13.18 GiB | 2.08 GiB | 1.58 |
| 48k | 0.60 GiB | 1.13 GiB | 13.27 GiB | 1.99 GiB | 1.49 |
| **64k** | **0.77 GiB** | 1.45 GiB | **13.43 GiB** | **1.83 GiB** | **1.33 (default)** |
| 96k | 1.10 GiB | 2.07 GiB | 13.76 GiB | 1.50 GiB | 1.00 — at the limit |
| 128k | 1.43 GiB | 2.70 GiB | 14.10 GiB | 1.16 GiB | 0.66 — aggressive |
| 256k (native) | 2.76 GiB | 5.20 GiB | 15.42 GiB | **−0.16 GiB** | impossible |

`overhead` = CUDA context + compute buffers. **Half-measured, not yet
swept.** Anchor **[measured]**: `llama-server -ngl 999` with a trivial GGUF
moves `nvidia-smi` from 14 MiB to 228 MiB, i.e. **~0.21 GiB is CUDA
runtime + context** on this driver; the rest (est. 0.2–0.4 GiB) is the gemma4
MoE compute buffer at `-ub 512`. That is why the older 1.1 GiB figure in §1 is
pessimistic for llama.cpp — it was a vLLM-era number.

So **65536 is the default**, now measured rather than projected (§2.8): 15,150 MiB
used, 1,153 MiB free at `-b 4096 -ub 2048`. The sweeping also showed that the
`-ub` value, not the KV cache, is what sets the ceiling: 32k → 14,682 MiB, 96k →
15,684 MiB (619 MiB free, fails the rule), 128k → does not allocate its compute
buffers at all. Raising context above 64k therefore buys nothing that §2.8 can
see: decode is flat across 32k/64k/96k, while pages are 2k–12k tokens.

**The sweep below has since been run (§2.8.2, 2026-11-18) and it overturned part of
this table:** 128k does not "load with 0.66 GiB headroom", it fails to allocate
its compute buffers once `-ub` is 2048, and 96k leaves 619 MiB, not 1.00 GiB.
Use §2.8.2 for the decision; this table is kept as the projection it was. The
older one-shot sweep harness (superseded by `/tmp/opencode/tune/sweep_launch.sh`,
which restarts per config and reads `timings`):

```bash
/tmp/opencode/kv_budget_sweep.sh     # waits for the .gguf, then steps 16k..128k
cat /tmp/opencode/kv-sweep/results.txt
```

Each step starts `llama-server` at that `-c`, waits for `/health`, records
`nvidia-smi memory.used` and the `timings` block from a real completion, then
stops it. A step that fails to load *or* to answer is recorded as a failure —
"loads" alone is not a pass. Re-run after any driver, toolkit or weight change,
and raise `FMD_MAX_MODEL_LEN` only to a measured row that still clears 1 GiB.

### D. 8-bit KV cache — what to expect

`q8_0` blocks are 32 values + one fp16 scale (34 B/32 = 1.0625 B/element), so the
cache is half of f16 with a ~0.06-bit-per-element quantisation of the
activations. It does not change the graph, the sampling or the constrained
decoder, so the only risk is numeric. Parity check (greedy, `temperature 0`,
same prompt, `-c 8192`): the sweep harness writes both outputs to
`/tmp/opencode/kv-sweep/parity.txt`. **Record the verdict here once it exists.**
Until then: q8_0 is enabled because it is what makes 32k free on a 16 GB card,
and because the failure mode is a wording difference, not an unparsable schema.

### E. MTP / speculative decoding — **rejected: no drafter to test (2026-11-18)**

llama.cpp has first-class Gemma 4 MTP: `--spec-type draft-mtp`,
`--model-draft <mtp file>`, arch `gemma4-assistant`, `--spec-draft-n-max N`
(merged PR #23398 on 2026-06-07; build b12468 is 2026-11-17, and
`src/models/gemma4-assistant.cpp` is present in the tree). The drafter shares the
target's KV cache, so it does not need a second context.

**The parity gate could not be run, because `mtp-gemma-4-26B-A4B-it-Q8_0.gguf`
is not on this host.** The weights directory holds only the target GGUF; the
whole filesystem has exactly one `mtp *.gguf`,
`~/AI/Models/Strata-data/mtp/mtp-q2_0.gguf`, and it is a `qwen4exp-mtp` head for
`Qwen/Qwen3.8-Flash-Next`. Pointing llama-server at it next to the Gemma target:

```
E llama_model_load: error loading model: unknown model architecture: 'qwen4exp-mtp'
E common_speculative_init_result: failed to load draft model
E srv    load_model: failed to load draft model
```

That is not "parity failed", it is "there is no pair to compare": zero of the
three prompts (short factual, 6k extraction, 12k) could be run with MTP, so there
is no output pair and no differing excerpt to quote. **Verdict: off.** No token
is spent on a capability claim either way — speculative decoding has never
produced a token on this host.

Two measured reasons not to chase it:

1. **VRAM.** +461 MB would take the chosen 64k config from 15,150 to ~15,611 MiB
   (≈692 MiB free), which fails the ≥1 GiB rule on its own. Buying that back
   (`-ub 1024`) costs 3% prefill (§2.8.1).
2. **The workload is not decode-starved in a way a drafter fixes.** Decode is
   ~101 tok/s and flat from 32k to 96k; the measured cost of a document is its
   prefill (§2.8), which MTP does not touch.

The gate stays as written — byte-identical greedy output on all three prompts
**and** >20% decode gain — and it runs only after the `gemma4-assistant` drafter
exists. `scripts/serve_local.sh` keeps the `FMD_SPEC` plumbing but the default is
`0`:

```bash
FMD_SPEC=1 scripts/serve_local.sh start   # only after both gates pass
```

Do not enable it for a throughput number alone.

### F. Process facts and fallback

```
pid   /tmp/opencode/fmd-serve/llama-server.pid
log   /tmp/opencode/fmd-serve/llama-server.log
verbs scripts/serve_local.sh {install|start|status|logs|probe|stop|fg}
```

`probe` is the gate before flipping `FMD_EXTRACT_PROVIDER` to `local`; it asserts
(a) non-empty `content`, (b) no `<|channel>thought` leakage, (c) a guided
`json_schema` object that is schema-valid **and** (d) a nullable field actually
emitted as `null`.

**Fallback ladder, in order, when something OOMs or misbehaves under load:**

1. Lower context: `FMD_MAX_MODEL_LEN=16384 scripts/serve_local.sh start`. This is
   almost always the fix; it frees 0.17 GiB at 16k and removes the long-prompt
   compute-buffer spike.
2. Free the GPU: no other GPU process. One GPU tenant at a time (§1 budget rule);
   the OCR stage and the text model never co-reside.
3. Retry with f16 KV (`FMD_KV_TYPE=f16`) only to *diagnose* — it costs 0.38 GiB
   more at 32k, so it makes OOM worse, never better.
4. Last resort: the 12B pilot GGUF. It is 6.72 GiB, so it leaves ~8 GiB for
   context, at a real capability cost. This is a capability regression, not a
   tuning step — say so when you take it.

`FMD_EXTRACT_PROVIDER` stays `api` until `probe` is green on real weights and the
download has finished. Flip:

```bash
FMD_EXTRACT_PROVIDER=local
FMD_EXTRACT_MODEL=gemma-4-26b-a4b-gguf     # must equal the server's --alias
FMD_LOCAL_LLM_URL=http://localhost:8010/v1
```

---

## 2.8 Throughput sweep — measured 2026-11-18 (same host, same weights)

Method: `llama-server` b12468, greedy (`temperature 0`, `top_k 1`, seed 42),
`cache_prompt=false` so every run pays full prefill, prompts fixed to file
(short factual = 100 tok, extraction-style = 6,001 tok, long = 11,401 tok), and
numbers read from llama.cpp's own `timings` block. Temperature 0 and a fixed
prompt are the only controls that matter for a kernel benchmark; `PYTHONHASHSEED`
is irrelevant because nothing here is Python-side. Harness lives in
`/tmp/opencode/tune/` (throwaway); each row is the mean of 2–6 restarts, min–max
shown where it matters.

### 2.8.1 Prefill `-b`/`-ub` at 64k, single slot, 11,401-token prompt

| `-b`/`-ub` | prefill tok/s (samples) | decode tok/s | VRAM MiB | free MiB |
|---|---|---|---|---|
| 2048 / 512 (old) | **3,843** (3,765–4,018) | 103.5–108.1 | 14,492 | 1,811 |
| 4096 / 1024 | **4,508** (4,399–4,690) | 102.0–102.6 | 14,690 | 1,613 |
| **4096 / 2048 (chosen)** | **4,784** (4,642–5,011) | 100.5–100.8 | 15,150 | 1,153 |
| 8192 / 2048 | 4,803 (4,666–4,939) | — | 15,150 | 1,153 |

+24% prefill over the old pair, decode flat within noise. 8192/2048 adds nothing
over 4096/2048, so the logical batch is not the limiter. 4096/2048 is the largest
ubatch that still clears 1 GiB free at 64k on this card; that is the whole reason
it wins over 8192/2048 (identical) and over 4096/1024 (3% slower prefill).

### 2.8.2 Context, with the winning flags

| ctx | VRAM MiB | free MiB | prefill tok/s | decode tok/s | serves? |
|---|---|---|---|---|---|
| 32768 | 14,682 | 1,621 | 4,633 | 97.3 | yes |
| **65536 (chosen)** | **15,150** | **1,153** | **4,621** | **98.4** | yes |
| 98304 | 15,684 | 619 | 4,648 | 97.8 | yes, fails headroom |
| 131072 | — | — | — | — | **no**: `failed to allocate compute pp buffers` |

Decode and prefill are flat from 32k to 96k, so **nothing above 64k buys
anything real**: documents are 2k–12k tokens, i.e. 64k already holds a whole page
five times over, and the only measured effect of a bigger context is less VRAM
headroom. 64k stays the default; 96k is now a known-bad row (not just
"aggressive"), and 128k is a hard failure at `-ub 2048` rather than the old
"0.66 GiB headroom" projection.

### 2.8.3 `-np 2` — two concurrent 6k documents

| slots | wall for 2×6k | aggregate tok/s | per-request decode | VRAM MiB | free MiB |
|---|---|---|---|---|---|
| `-np 1` | 7.72–7.78 s | 65.8–66.4 | 101.4–101.8 | 15,150 | 1,153 |
| `-np 2` (`-c 65536`, 32k/slot) | 6.16–6.23 s | 82.2–83.1 | 63.9–74.3 | 15,340 | 963 |
| `-np 2 --kv-unified-per-slot 32768` | 6.13–6.22 s | 82.3–83.6 | 63.5–75.1 | 15,340 | 963 |
| `-np 2` + `-ub 1024` | 6.12–6.14 s | 83.4–83.6 | 76.4–78.2 | 14,838 | 1,465 |

Batch throughput does improve: **+25% aggregate** for two simultaneous documents,
at the cost of halving each stream's decode. Per-slot context is *not* too small
— 32,768 tokens per slot is still 2.7× the largest page plus its 4k completion.
`--kv-unified-per-slot` exists in b12468 but is a no-op when `-c` already sizes
the shared pool; it only matters if you want `-c` omitted.

**Default stays `-np 1`**, because the router issues one request at a time
(`fmd/llm` has no thread pool), two slots drop 64k → 32k per request, and
`-np 2` at `-ub 2048` leaves 963 MiB, under the rule. If a batch driver ever
exists, the recipe is `-np 2 -c 65536 -ub 1024` (1,465 MiB free, +25%).
Two GPUs' worth of extraction can also be had the boring way: two servers, one
document each — but that needs a second card, not a flag.

### 2.8.4 Thermals — 10 continuous minutes at the winning config

Continuous 11.4k-prefill + 256-token decode, telemetry every 20 s (30 samples):

| metric | idle | sustained |
|---|---|---|
| temp | 41–54 °C | **65 → 87 °C**, settles 83–87 |
| board power | 5–25 W | **112–133 W** (cap 115 W; boost excursions) |
| SM clock | 1,492 MHz | **1,500–2,197 MHz** of 3,090 max |
| utilisation | 0% | 82–100% |

The card is **power-limited, not thermally throttled**: temperature plateaus
below 90 °C and the clock tracks the power budget (it climbs back into the
2,000s whenever draw dips), with no drop to a lower base clock. Cost of a hot
card: decode 101 → 93.4 tok/s (−8%) and short-prompt decode 117.5 → 108.9 after
the 10-minute run; prefill is unchanged (4,621 vs 4,689). Long extraction runs
should be budgeted at the hot number, not the cold one.

### 2.8.5 Config of record

```bash
-c 65536 -b 4096 -ub 2048 -np 1 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 --jinja -rea off
```

4,784 tok/s prefill (11.4k prompt; 5,021 re-measured on the restarted server),
~101 tok/s decode cold / 93 hot, 15,150 MiB, 1,153 MiB free.
`scripts/serve_local.sh` and `config/models.yaml` carry these values, and
`scripts/serve_local.sh probe` is green on it (a/text, b/no-thought-leak,
c/guided-json, d/guided-json-nullable). `uv run pytest -q` in
`packages/pipeline` is 719 passed / 1 failed, and that failure is not this
change: `test_adapters_registry` still asserts
`max_pages_per_domain == 200` while the uncommitted bulk-ramp edit in
`config/sources.yaml` sets 20.

---

## 3. OCR stage (T2: RapidOCR/PP-OCR default; Qwen3-VL + PaddleOCR-VL retained)
### 3.1 Why this engine (amended 2026-11-19, measured)

T2's default backend is now **RapidOCR — the RUNNER, not the model** — an
onnxruntime wrapper around PaddleOCR PP-OCR detection + recognition graphs,
in-process, no HTTP service. Purpose-built detection+recognition is orders of
magnitude cheaper per page than a VLM; a VLM over a 200-page scan is exactly
what stalled the corpus (D18). The preference is **PROVISIONAL UNTIL MEASURED**:
`scripts/ocr_validate.py --ground-truth` scores both engines over the same
scans, and the decision rule (§3.8) — not this file — is what a measurement
changes. Table structure stays Docling/TableFormer (T1); it is not OCR's job.

| Backend | Where it runs | Role |
|---|---|---|
| `rapidocr` (default) | in-process (pipeline venv) | T2 engine of record, CPU default |
| `qwen3vl` | `/extract` service on `FMD_T2_URL` | Devanagari fallback + selectable backend; never default, never removed, never silent |
| `paddleocr` | `/extract` service, CPU-only on sm_120 | alternate full pipeline (T2b) |

Device policy is **measured, not assumed**: with llama-server resident
(14.4 GiB) only ~688 MiB VRAM is free; onnxruntime CUDA needs ~1.5–2 GiB and
OOMs there. So `device: cpu` is the default and measures **2.33 s/page
(25.8 pages/min) — 200 pp ≈ 8 min**. `device: cuda` is opt-in for batch runs
with the LLM server stopped; if free VRAM is below `min_free_vram_gib` (2.0),
the CUDA path fails with a recorded `cuda_unavailable_vram` reason and the run
continues on CPU — a device preference never crashes a run.

### 3.1b Models are pinned (no URL guessing)

`config/acquire.yaml:backends.T2.models` pins exact ids, from the runner's own
manifest (ModelScope `RapidAI/RapidOCR@v3.10.0`):

| File | Role | sha256 |
|---|---|---|
| `PP-OCRv6_det_small.onnx` | det (bundled with the runner wheel) | `090f04ab…1ff94f` |
| `devanagari_PP-OCRv5_rec_mobile.onnx` | rec — Devanagari is a **first-class LangRec value** (`LangRec.DEVANAGARI`) | `d6f0a906…52c91e` |
| `ppocrv5_devanagari_dict.txt` | rec dictionary | manifest-checked |

One-time prefetch (sha256-verified by the runner's manifest, re-verified by us):

```bash
# Runner from the package extra (don't install into a hand-made env):
uv sync --extra ocr                                    # rapidocr, onnxruntime, pypdfium2
# Development/CI installs every workstream extra, OCR included:
uv sync --all-extras
uv run python scripts/ocr_validate.py --prefetch      # → $FMD_RAPIDOCR_MODELS (default data/rapidocr-models)
```

Verify the install is real (packages, pinned models, execution provider) with:

```bash
uv run fmd llm doctor
# ocr  engine=rapidocr  device=cpu  models=present  provider=cpu  status=ok
```

`models=present` means the configured det/rec files (and the rec dictionary)
exist under `$FMD_RAPIDOCR_MODELS` — detection is by files on disk, never by a
successful import. Any gap prints `status=unavailable:<reason>` instead of a
green line: `rapidocr_missing`, `onnxruntime_missing`, `models_missing`,
`cuda_unavailable_no_provider`, `cuda_unavailable_vram`. The base install never
pulls the OCR stack; a package that was synced without the extra reports
`unavailable:rapidocr_missing`.

`backends.T2.models.artifacts_root` defaults to `data/rapidocr-models` (the
prefetch target) and a relative root resolves against the repository checkout,
not the process cwd. `FMD_RAPIDOCR_MODELS` still overrides it. This is the same
value `fmd llm doctor` and the T2 runner read, so a green doctor line means the
tier can resolve the same files.

T2/T2b are OCR *acquisition* stages and return text; claim extraction routes
them to `bulk_extract` (the local text model). They are never sent to
`hard_pdf`, which is the optional hosted vision profile and degrades with
`vision_unavailable` when unconfigured. A scan whose T1/Docling pass returns
empty text escalates to T2, and a T2 failure on a sparse scan parks it in T3
(`ocr_unavailable_sparse_scan`) instead of publishing page-number "text".

Missing artifacts are a named failure (`models_missing: … prefetch with uv run
python scripts/ocr_validate.py --prefetch`), never a download at serve time.
A missing runner is `engine_not_installed: … uv pip install "rapidocr>=3.10"
"onnxruntime>=1.32"` at the backend (the package equivalent is
`uv sync --extra ocr`).

### 3.1c Devanagari capability — proven, not assumed

The configured rec model id is checked against the runner's model manifest,
and a mock Devanagari page is asserted to route to the pinned model
(`tests/test_acquire_rapidocr.py`). The real-GPU check, after prefetch:

```bash
uv run pytest packages/pipeline/tests/test_acquire_rapidocr.py -q   # manifest check runs (not skipped) when rapidocr is installed
uv run python scripts/ocr_validate.py --ground-truth --engine rapidocr   # CER/chrF++ on doc 61 pages 3/5/7
```

Known, measured defects handled in the pipeline (not hidden): boxes under
`min_box_confidence` (0.7) are dropped **and counted** per page;
virama/conjunct sequences are error-prone (measured: शक्ति→शक्ि, विश्व→विश्ष) so
pages with conjunct clusters at low confidence get a per-page `conjunct_risk`
flag; Devanagari digits (०१२३) are normalised to ASCII when detected, logged,
and counted — untested in our corpus, so never assumed correct.

### 3.1d VRAM expectation (cuda opt-in)

| Config | Free VRAM needed | What happens below the floor |
|---|---|---|
| `device: cpu` (default) | none | always runs; 2.33 s/page measured |
| `device: cuda`, floor `min_free_vram_gib: 2.0` | ~1.5–2 GiB for the det+rec graphs | recorded `cuda_unavailable_vram` → CPU, run continues |
| llama-server resident | ~688 MiB free measured | cuda is refused by name — do not force it |

`FMD_OCR_DEVICE=cuda` (or acquire.yaml) opts in for a batch run with the LLM
server stopped; `FMD_OCR_DEVICE=cpu` returns to the safe default. Licences:
RapidOCR **Apache-2.0**, PP-OCRv5/v6 models **Apache-2.0** (ModelScope
RapidAI/RapidOCR), PaddleOCR-VL **Apache-2.0**, Qwen3-VL **Apache-2.0**.
Surya/Marker stay rejected (OpenRAIL-M, $5 M revenue cap).

### 3.2 Weights (once; ~10.6 GB on the default)

| Repo | Files on disk | Fits 16 GB? |
|---|---:|---|
| `Qwen/Qwen3-VL-8B-Instruct-FP8` (default) | 10.61 GB (5.36 + 5.23 shards) | yes, with an 8k context |
| `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit` (int4) | 7.24 GB (5.00 + 2.23) | yes, more KV headroom |
| `Qwen/Qwen3-VL-8B-Instruct` (bf16) | 17.55 GB (4 shards) | **no** — do not start here |

Sizes were read from the Hub API on 2026-11-16, not estimated. Weights are host
data under `$FMD_MODEL_DIR` (default `/home/archit-gore/models`), never a docker
volume and never a download at serve time. One command fetches them, resumably:

```bash
scripts/fetch_model.sh Qwen/Qwen3-VL-8B-Instruct-FP8   # ~10.6 GB, once
ls -l "$FMD_MODEL_DIR/Qwen3-VL-8B-Instruct-FP8"       # expect 2 safetensors + configs
```

`config/acquire.yaml:backends.T2.service.hub_id` holds the **directory name**
under that root (`Qwen3-VL-8B-Instruct-FP8`), which is what
`scripts/serve_ocr.sh vision-start` serves. Override with
`FMD_VLLM_OCR_MODEL=/abs/path` for a one-off; do not edit the script. For the
int4 alternative fetch `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit` into the same root
and override `FMD_VLLM_OCR_MODEL` (vLLM detects the bitsandbytes quantisation).

### 3.3 Bring-up (copy-paste, no polling)

```bash
scripts/serve_ocr.sh vision-install   # only if ~/.venvs/fmd-vllm is missing
scripts/serve_ocr.sh vision-start     # Qwen3-VL on 0.0.0.0:8001, pid/log in /tmp/opencode/fmd-serve
curl -s localhost:8001/v1/models | python3 -m json.tool | head   # expect the served id
scripts/serve_ocr.sh build            # compose --profile ocr build ocr (thin image, no CUDA)
scripts/serve_ocr.sh start            # up -d + one /health, no wait loop
curl -s localhost:8110/health
```

`/health` is `ok` only when the served name at 8001 equals
`backends.T2.service.served_name`. Before the weights load it answers 200 with
`{"status":"unavailable","reason":"vision_unavailable: vLLM is not answering ..."}`
— that is the correct state, not something to wait out silently.

How to stop:

```bash
scripts/serve_ocr.sh stop             # adapter container
scripts/serve_ocr.sh vision-stop      # SIGTERM the vLLM process group
```

### 3.4 One real page

```bash
python infra/ocr/probe.py infra/ocr/fixtures/scan_sample.png
```

`infra/ocr/probe.py` posts the file to `POST /extract` (`--url` defaults to
`$FMD_T2_URL`, else `http://localhost:8110`) and pretty-prints the block list; a
PDF path works the same way. It prints the service's error body verbatim and
exits non-zero, so an unavailable engine cannot look like a clean empty page.

Then the pipeline-level check on a scanned PDF:

```bash
FMD_T2_URL=http://localhost:8110 uv run python -m fmd.cli extract --limit 2
```

### 3.5 Contract (must match `fmd/acquire/tiers.py::_ocr_doc`)

`POST $FMD_T2_URL/extract` with `{"pdf_base64": "..."}` (or `"image_base64"`
for a single page) →

```json
{"tool":"fmd-ocr","engine":"qwen3vl","tool_version":null,"model":"Qwen3-VL-8B-Instruct",
 "pages":[{"page":1,"width":600,"height":800,"bbox":[0,0,600,800],"blank":false}],
 "blocks":[{"page":1,"block_order":0,"block_bbox":[48,150,560,180],
            "block_content":"Total fee: INR 50000 per year"}]}
```

* Boxes are pixels in the **analysed** raster: pages are rendered at
  `FMD_OCR_DPI` (200) and downscaled when the long side exceeds `max_side_px`
  (2400), and `pages[].width/height` is that same raster, so a block box always
  resolves inside its page box.
* `block_order` is assigned by the service after sorting top-to-bottom then
  left-to-right; it is monotonic per page.
* A box the model returns that cannot be trusted is reported as
  `block_bbox: null` rather than invented. T2 is deliberately outside
  `bbox.gate_tiers`, so a null box cannot silently become a span downstream.
* `blank: true` marks a page with no ink; that page legitimately has no blocks.

### 3.6 When it cannot run (and why that is not a silent success)

| `/health` reason | Meaning |
|---|---|
| `config_unreadable` | `config/acquire.yaml` missing or malformed at `FMD_OCR_CONFIG` / `/config/acquire.yaml` |
| `backend_unknown` / `model_unset` | `backends.<tier>.service.backend` or `served_name` absent |
| `vision_unavailable` | vLLM is not answering at `FMD_OCR_VLLM_URL` — start it, or pull the weights |
| `model_not_served` | a model is up but not the configured one (e.g. the 9B text model on 8001) |
| `engine_not_installed` / `engine_start_failed` | PaddleOCR-VL import/CUDA failure (CPU is the fix on sm_120) |

`POST /extract` answers **503 with the same `vision_unavailable: ...` reason**
when the engine is down, and **503 `empty_result: ...`** when a non-blank page
produced no blocks. It never returns 200 with an empty block list for a page that
had ink. Upstream, T2 raises `TierUnavailable`:

* image-only page → "T0 also produced no text, so this document cannot be
  extracted on this host" → **no claims**, document queued for review;
* page with usable T0 text → T0 fallback with a `T2 unavailable: ...` note.

The in-process `rapidocr` backend (the T2 default) raises the same classes with
its own stable codes: `models_missing: …` (names the prefetch command),
`engine_not_installed: …` (names the install command),
`no_text_layer: …` (unreadable image-only page → review), and
`acquisition_failed:acquisition_budget_exceeded` (D18: page/time budget hit —
no partial claims, the document is parked, the run continues with the next
document). `cuda_unavailable_vram` / `cuda_unavailable_no_provider` are
recorded reasons, not crashes: the backend continues on CPU.

**Measured limitation (2026-11-16).** `tiers._ocr_tier` formats an
`httpx.HTTPStatusError`, which carries only status and URL, so the note reads
`T2 unavailable: ... 503 Service Unavailable` and the `vision_unavailable` label
stays in the response body and in `/health`. Putting the label into the note needs
a one-line change in `_ocr_tier` (read the JSON `detail` when `status >= 400`); it
is an acquire-workstream change, not a service change.

### 3.7 T2b: PaddleOCR-VL, the retained alternative engine

PaddleOCR-VL is not deleted; it is selected by config, not by editing code:

```bash
scripts/serve_ocr.sh paddle-start     # compose --profile ocr-paddle, port 8111
curl -s localhost:8111/health         # engine: paddleocr, device: cpu
export FMD_T2B_URL=http://localhost:8111   # the tier's env var (acquire.yaml)
```

Compose service `ocr-paddle` builds `infra/ocr/requirements.txt`
(paddlepaddle-gpu 2.6.2, paddleocr 3.7.0) on a CUDA 12.9 base but runs with
`FMD_OCR_DEVICE=cpu`, because Paddle's GPU kernels stop at sm_90. Expect **seconds
per page**, not milliseconds. It is a second opinion for pages where Qwen3-VL
fails the Indic self-check, not the bulk path. Selecting it for T2 instead is
`FMD_OCR_BACKEND=paddleocr` on the `ocr` service.

### 3.8 Measurement, decision rule, and what shipping requires

**Shipping requires OCR enabled and measured.** An engine that has never been
scored against the corpus is a preference, not a decision. The gate is:
`uv run python scripts/ocr_validate.py --ground-truth` green (≥ 3 documents
with ground truth, engines reachable, baseline printed) **and** the D18
quality gates recorded (CER/chrF++ on real Hindi-header scans, block-granularity
offset resolvability). Until then D18 stays "must ship, not shipped".

Small batches first, always:

```bash
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd \
  uv run python scripts/ocr_validate.py --ground-truth --engine all --limit 3
```

* ground truth is doc 61 (gsv.ac.in Hindi annual report, pages 3/5/7) — its
  **real Unicode text layer**, never an invented transcript;
* reports CER, chrF++ and pages/min per engine, per document and aggregate,
  with the 2026-11-19 baseline (CER 10.9 %, 25.8 pages/min) printed for
  regression visibility;
* refuses > 10 documents without `--i-know`; supports `--engine` and `--limit`;
* engines that cannot run are **recorded** in the report, never silently skipped.

**Committed decision rule** (the numbers decide, not the patch):

1. PP-OCRv5 within ~5 chrF++ of Qwen3-VL on Devanagari → keep PP-OCRv5
   everywhere (`T2.backend: rapidocr` stays).
2. PP-OCRv5 loses by more → route conjunct-heavy / Devanagari pages to
   Qwen3-VL (`FMD_T2_BACKEND=qwen3vl` per page via the fallback path or the
   backend matrix), Latin stays on PP-OCRv5, per-page engine recorded in
   `engine_meta.pages[].engine` and the claim qualifier.

Either way the corpus stays reproducible: one engine per page, one transcript,
engine + engine_version + model_ids on every scanned claim.

## 4. Common failures

| Symptom | Meaning | Action |
|---|---|---|
| `no kernel image is available for execution on the device` | wheel built for ≤ sm_90 (Paddle, or an old torch) | text model: use vLLM ≥ 0.19 (resolves cu130 torch, sm_120 present). Paddle OCR: no Blackwell kernel — serve T2 with Qwen3-VL (§3.1) and run Paddle as T2b on CPU |
| `CUDA out of memory` at startup | `--gpu-memory-utilization 0.90` + desktop compositor on 16 GB | drop to `0.80`, then lower `FMD_MAX_MODEL_LEN` to 16384; record the change, 32k is the cost model's assumption |
| `xgrammar` grammar/architecture error | constrained decoding does not cover the Qwen3.5 hybrid architecture | set `providers.local.supports_guided_json: false`; the router then degrades to JSON-instruct + schema validation. Report the degraded schema-valid rate separately — it is a different candidate |
| repetition loop in raw `curl` output | decoding, not acquisition | confirm `enable_thinking: false` in `--default-chat-template-kwargs`, temperature 0, no repetition penalty. The pipeline's own guard (`fmd/acquire/guards.py`) routes loops to T3 |
| `404 model not found` | `FMD_EXTRACT_MODEL` ≠ `--served-model-name` | align both |
| `/health` says `model_not_served` | the vision port serves something else (usually the 9B text model) | start `scripts/serve_ocr.sh vision-start`; the vLLM on 8010 must not be reused for OCR |
| `T2 was selected but is unavailable` | OCR service down, `FMD_T2_URL` unset, or weights absent | expected degradation: `/health` names the reason (`vision_unavailable`, `model_not_served`, …); do not fabricate OCR text |
| `/extract` → 503 `empty_result` | a non-blank page produced zero blocks | the page is unreadable at these settings; route it to the human queue, do not lower the bar to "empty string is fine" |
| container gets `404` from `FMD_LOCAL_LLM_URL` or `Connection refused` on :8000 | the host :8000 is the published `api` port, not vLLM | serve the text model on :8010 (`FMD_VLLM_PORT=8010`) and point `FMD_LOCAL_LLM_URL` there |
| `SafetensorError: incomplete metadata` at model load | a shard is a truncated download | `scripts/fetch_model.sh <repo>` resumes it by size; never `-o` without `-C -` |

**Known guard false positive (found 2026-11-16).** Docling (T1) emits
section-level blocks; `acquire.yaml:anomaly.block_length_multiplier = 3.0`
therefore flags legitimate pages as `block_length_spike`. On the RUHS 11-page
fee PDF the page medians were 169/32/18 chars against maxima of 558/435/718, so
the document went to T3 and no claims were emitted. This is an acquisition
threshold question (mixed block granularity), not a model failure; it was
reported, not tuned away.

## 5. Measured on this host (2026-11-16)

| Item | Value | How measured |
|---|---|---|
| vLLM / torch / CUDA | 0.33.0 / 2.14.0+cu130 / 13.0 | `vllm.__version__`, `torch.version.cuda` |
| sm_120 in `torch.cuda.get_arch_list()` | yes | one-liner in §2.1 |
| Device | RTX 5080 Laptop, `(12, 0)`, 16303 MiB | `torch.cuda.get_device_capability`, `nvidia-smi` |
| Engine startup with 0.85 util | ~40 s to API server banner, weights load 4.5 s **after** download | `vllm.log` |
| Idle VRAM with model + 32k KV reserved | ~11.8 GiB peak during warmup (0.85 util) | `nvidia-smi` |
| Decode tokens/s, 9B int4, 32k ctx | **unverified** — weights still downloading | — |
| Guided JSON on Qwen3.5 + xgrammar | **unverified** — needs the served model | §2.4 probe |
| Pages/hour | **unverified** — depends on the two rows above | `fmd extract` wall clock |
| vLLM supports Qwen3-VL | yes — `Qwen3VLForConditionalGeneration` in `ModelRegistry.get_supported_archs()` | 2026-11-16, `~/.venvs/fmd-vllm` |
| Qwen3-VL-8B weights on this host | **absent** (only the text models are cached) | `ls ~/models` |
| OCR service with no weights | `/health` → `unavailable` + `vision_unavailable: ...`; `/extract` → 503 with the same reason | `uvicorn server:app` on :8199, `curl /health`, then the fixture through `/extract` |
| OCR blocks/pages through Qwen3-VL | **unverified** — needs the 10.6 GB download | §3.4 over the committed fixture |
| PaddleOCR-VL on GPU here | not attempted; wheels ≤ sm_90, CPU is the supported path | §3.7 |

Do not copy the doc-10 §6 cost model's "8 pages/min" into a report until the
unverified rows above are filled; they are the same unverified assumption. The
OCR rows have their own dependency: the 10.6 GB Qwen3-VL download.

## 6. Run this (copy-paste, in order)

The GPU has exactly one resident model: `llama-server`. Docling (T1 layout +
TableFormer) runs on **CPU** by default (`config/acquire.yaml:backends.T1.device:
cpu`). That is deliberate, not a fallback: at 64k the text model holds ~14 GiB
of the 15.5 GiB card, so a second CUDA consumer cannot fit. A Docling CUDA OOM
is now `acquisition_failed:docling_cuda_oom`: the document is marked
`needs_review`, the run summary is `ok:false`, and the CLI exits non-zero. It
never reports a clean run with zero claims.

**Measured 2026-11-20 (RTX 5080 Laptop 16303 MiB, 30 GB RAM; four documents,
7–434 pages).** The "cannot fit" claim above is now quantified:

| LLM config | VRAM used/free | decode tok/s (short / after 8.8k) | Docling CPU s/page | Docling CUDA s/page |
|---|---|---|---|---|
| 64k q8_0 KV on GPU (production) | 15052 / 1251 MiB | 130.8 / 106.8 | 1.0–3.8 | **OOM** (needs ~1.0–1.3 GiB; 589 MiB free after context) |
| 64k `--no-kv-offload` (KV in RAM) | 14130 / 2173 MiB | 66.2 / **18.0** | unchanged | 0.16–0.87 |
| 32k q8_0 KV on GPU | 14584 / 1719 MiB | 125.7 / 105.0 | unchanged | not safe (~1.06 GiB headroom) |
| 64k q4_0 KV on GPU | 14582 / 1721 MiB | 126.1 / 109.7 | unchanged | not safe; 4-bit KV quality unvalidated |

- `--no-kv-offload` frees only **~0.9 GiB** (the KV itself), which is exactly
  Docling's requirement; the pair fits with 23 MiB to spare. The price is
  **−49% decode short, −83% after an 8.8k prompt** (18 tok/s), and −91% when
  Docling is concurrently on CUDA. Not a default.
- **CPU Docling is nearly free from the LLM's point of view**: 64k KV on GPU +
  Docling on CPU ran concurrently at 47.3 pages/min with LLM decode −6%. Host
  RSS is **2.2 GiB (small docs) / 4.1 GiB (161 p) / 5.2 GiB (434 p)**.
- Docling on CUDA is **2–6× faster per page** (0.16 vs 1.31 s/page on a 161-page
  report). The only way to have it is to **stop `llama-server` for the
  acquisition batch**, then restart for extraction — or accept the KV-offload
  decode penalty. There is no dynamic "load KV when needed" in llama.cpp; KV is
  resident for the server's lifetime.
- 32k or q4_0 KV do not free enough for CUDA Docling either (~1.0–1.1 GiB), so
  shortening the context buys nothing for coexistence.

```bash
# 0) once per boot: host text model (weights already on host; no download)
scripts/serve_local.sh start
scripts/serve_local.sh probe                          # text, no leak, guided JSON, null
curl -s localhost:8010/v1/models                      # expect gemma-4-26b-a4b-gguf

# 1) stack (api waits for db); rebuilds never touch weights
docker compose -f infra/docker-compose.yml up -d db api worker
curl -s localhost:8000/api/health                     # expect status ok

# 2) router health from INSIDE the container (proves host-gateway works)
docker compose -f infra/docker-compose.yml exec api \
  python -m fmd.cli llm doctor --live

# 3) the ordered evidence pass: acquire -> extract -> verify -> score
#    (automatic phases per §6.2: acquire/extract time-share the one GPU by
#    stopping/restarting the model server themselves — no second CUDA consumer)
docker compose -f infra/docker-compose.yml exec api \
  python -m fmd.cli extract --limit 3
docker compose -f infra/docker-compose.yml exec api \
  python -m fmd.cli verify --limit 50
docker compose -f infra/docker-compose.yml exec api python -m fmd.cli score

# 4) stop (weights and DB survive; nothing is re-downloaded)
docker compose -f infra/docker-compose.yml stop worker api
scripts/serve_local.sh stop
```

**Host-run workaround** (snapshots copied out of the Docker volume):

```bash
export FMD_SNAPSHOT_DIR="$HOME/fmd-snap2"
export FMD_LOCAL_LLM_URL=http://127.0.0.1:8010/v1   # compose uses host.docker.internal
export FMD_EXTRACT_PROVIDER=local FMD_EXTRACT_MODEL=gemma-4-26b-a4b-gguf
export FMD_VERIFY_PROVIDER=local  FMD_VERIFY_MODEL=gemma-4-26b-a4b-gguf
uv run fmd extract --limit 3
```

`FMD_DOCLING_DEVICE` overrides `backends.T1.device` per shell (`cpu|auto|cuda`).
Use `cuda` only when the text server is stopped or the card is otherwise free;
when it is not, the preflight fails with
`acquisition_failed:insufficient_gpu_memory` before any weights are loaded.

Long steps (extraction over many documents, a first crawl) are the human's to
run: hand them the command, do not add a wait loop. `scripts/serve_ocr.sh
vision-start` is only needed for scanned pages and needs the optional 10.6 GB
vision fetch first (§3.2); without it T2 degrades to the human queue and that is
the honest state.
### 6.1 Batched acquire (W10, 2026-11-21) — stop → acquire on CUDA → extract

The fused loop idled the GPU during every conversion: Docling ran on CPU
(1.0–3.8 s/page) while the 26B held its 64k KV cache, and re-running
extraction re-converted every document. Two changes fix both halves:

* **Persistent T1 parse cache** (`document_parses`, migration 0011 + the
  content-addressed object store). Key = (snapshot sha256, tier=`T1`,
  policy digest, payload schema version). **The Docling device is not part of
  the key** — a CUDA parse is reusable by a CPU run and vice versa. A policy
  or schema bump invalidates; corrupt/truncated/mismatched entries reconvert
  with a recorded reason and never crash. Cache hit ⇒ zero Docling calls.
* **T2 parse cache (2026-11-21, wave v1-w3)** — same store, same table,
  tier-scoped. In-process OCR (RapidOCR/PP-OCR) results are cached under a key
  that additionally pins an **engine fingerprint**: `{engine, engine_version,
  model_ids(det/rec/rec_keys), model_source, model_sha256}`. An engine-version
  bump or a model-id change invalidates (miss ⇒ reconvert ⇒ new row); the
  **device is excluded** like every tier. The T2 policy digest covers the T2
  block + the RapidOCR settings that shape output (raster, budgets,
  confidence thresholds). **HTTP engines (qwen3vl/paddleocr) are NOT cached**:
  their engine version is only knowable by asking a service, so a model swap
  behind the same URL could not be detected — those results are refused with
  `t2_engine_fingerprint_unavailable`, never cached under an unpinned
  identity. A cached T2 parse implies these exact bytes are a scan the pinned
  engine read completely under the current policy: `fmd acquire`/`extract`
  skip the OCR run outright (T1 cache consulted first, T2 second).
* **Batched orchestrator** (`fmd extract --batched`): per cycle — ensure
  server stopped → acquire one page-batch on CUDA (zero LLM calls) → ensure
  server started and `/health` ready (poll with timeout) → extract that batch
  from cache → repeat. Sizing is **by pages, evaluated only at document
  boundaries** (`batch.pages`, default 100): the document that crosses the
  limit always finishes in full in the same cycle and no further document
  starts; a known-size document that would cross is deferred whole; a 434-page
  document runs alone. No document is ever truncated, dropped, or split across
  cycles, and nothing is cached or extracted while incomplete.

Measured constants behind the defaults (operator host, 2026-11-20): CUDA
conversion 0.166–0.9 s/page vs CPU 1.0–3.8 s/page; ~20–25 s per server
restart (`scripts/serve_local.sh` stop + start + `/health`); **break-even
~20–25 pages/cycle** — below that the restart costs more than the CUDA
speedup saves, above it batching wins. The 100-page default leaves ~4–5×
margin. A ~1,000-page backlog therefore plans to ~10–12 cycles (~10 cycles at
100 pages/cycle, more if large documents straddle boundaries).

```bash
# the batched pass (operator, GPU host): parse cache + page-batched cycles
scripts/batch_extract.sh --dry-run --limit 20            # plan only, writes nothing
scripts/batch_extract.sh --limit 20                      # real run; server stopped at the end
scripts/batch_extract.sh --limit 20 --keep-server        # leave the server running
scripts/batch_extract.sh --limit 20 --device cpu         # CPU acquisition instead of CUDA

# split by hand when you want to control the GPU window yourself:
python -m fmd.cli acquire --device cuda --pages 300      # convert + cache, zero LLM calls
scripts/serve_local.sh start
python -m fmd.cli extract --from-cache --limit 50        # zero Docling on this path
scripts/serve_local.sh stop

# Degraded mode: FMD_SERVER_CONTROL=off (or no server_script) falls back to
# CPU acquisition + inline extraction and records server_control=degraded in
# the summary — the run still happens, honestly labelled.
FMD_SERVER_CONTROL=off python -m fmd.cli extract --batched --limit 20
```

Exit codes: the run is non-zero only when nothing was extractable (or an
acquisition/device failure was recorded). Every exit path stops the model
server (try/finally) unless `--keep-server`; Ctrl-C between cycles loses only
in-flight work — persisted parses and claims stay, and the next run resumes
(skips documents with a cached parse and a completed extraction). Known gaps:
T2/OCR output is not cached (acquisition of scans re-runs OCR every time); a
Docling library upgrade is recorded in the payload but does not invalidate
stored parses; the GPU path is validated only through fakes in CI — the
operator runs the real pass.

### 6.2 Automatic resource phases + compute metrics (FMD-CLOUD-030, 2026-11-28)

The batched window above was trapped inside `fmd extract --batched`, and a
standalone `fmd acquire` still defaulted to CPU Docling (the 50-college E2E
run spent ~2 h on CPU acquisition — 21 then 2 parses per ~100 min — while the
26B sat on the GPU, and its logs only said "still running"). The server
lifecycle is now a reusable phase manager, and the phases are **automatic**:

* **`fmd/orchestrate/gpu_phases.py` (`ResourcePhaseManager`)** — the
  `ScriptServerController`/stop/start/readiness machinery extracted from
  `BatchedRunner` (which now uses it per cycle; no second implementation).
  `acquire_phase(device)` = ensure server stopped → pin
  `FMD_DOCLING_DEVICE`+`FMD_OCR_DEVICE` → yield → finally restart + poll
  `/health` ready (try/finally on every exit path). The VRAM verdict is taken
  **after** the stop (with a resident 26B the card always looks starved);
  a card still below `min_free_vram_gib` after the stop degrades that window
  to CPU with the reason recorded instead of OOM-ing Docling.
* **`fmd acquire --phased`** — default on (`acquire.yaml:batch.phased_acquire`):
  the conversion pass runs inside an automatic stop-convert-restart window
  (still **zero LLM calls**), and the summary records
  `phased/phase_control/phase_reason`. Without server control
  (`FMD_SERVER_CONTROL=off`, no script) or on a starved card it degrades to
  CPU with the reason in the summary — a run never fails purely for missing
  GPU control. `--device auto` picks CUDA when a GPU with enough free VRAM is
  visible, else CPU. **No manual server stop/start:**
  `fmd acquire --limit 500` alone phases the whole backlog; a phased acquire
  leaves the model server running and `/health`-ready, so the next
  `fmd extract` (or worker tick) consumes the parses immediately. A
  per-document progress line (`[acquire] doc=… pages=… device=… seconds=…
  cache_hit=…`) flushes to stderr, so logs say what happened, not only
  "still running".
* **The worker phases itself** — `acquire.yaml:batch.phased_extract` (default
  on): when a server controller exists and the router is a real provider (not
  replay fixtures), the worker's extract tick runs the batched phased route
  (`stop → acquire page-batch on CUDA → start → extract`) with no manual care.
  `FMD_PHASED_EXTRACT=off` overrides per shell.
* **Compute metrics** (`fmd/obs/metrics.py`) — append-only JSONL under
  `data/compute-metrics/<run_id>.jsonl` (gitignored except a README; no user
  data: `input_ref` is the content-hash reference). Per stage/phase/task:
  device, model/provider, duration, documents/pages/cache hits, tokens and
  tok/s, OCR engine, and a 5 s `ResourceSampler` (nvidia-smi util/mem/power +
  ps cpu%/rss around each phase; absent tools record null, never crash).
  Emission points: every router call (live + cache hits), every acquisition
  conversion/cache hit, batched cycles, phase windows, embedding providers.
  Read it back with **`fmd metrics summary [--run-id <id>] [--json]`** —
  p50/p95 duration and tok/s by stage+device, total GPU vs CPU time,
  cache-hit rates, and one ranked bottleneck line. `scripts/run_logged.sh`
  heartbeats now include a `GPU util%,mem_used_mib,mem_total_mib,power_w`
  line when nvidia-smi exists (START/END/.status contract unchanged).

```bash
# the whole evidence pass, automatically phased — no manual server control:
uv run python -m fmd.cli acquire --limit 500            # stops server, CUDA Docling, restarts ready
uv run python -m fmd.cli extract --batched --limit 500  # per cycle: stop -> acquire -> start -> extract
uv run python -m fmd.cli metrics summary --run-id <id>  # "what was the bottleneck?"
```

Still true: `FMD_SERVER_CONTROL=off` is the operator's switch for a strictly
CPU run, `keep_server` is the only exit path that leaves the server up during
a batched run, and the GPU path is validated only through fakes in CI — the
operator runs the real pass (LQ-4).
## 7. Offline mode (no outbound network)

Set three variables in the shell (or `.env`); `.env.example` carries the same
block:

```bash
export FMD_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
# Optional but required for full T1 (Docling layout + tables):
export FMD_DOCLING_ARTIFACTS="$HOME/.cache/docling-models"
```

`FMD_OFFLINE=1` is the project switch. `HF_HUB_OFFLINE=1` alone also triggers
the acquisition guard, so a container that only sets the Hugging Face flag still
fails fast rather than stalling. `fmd.acquire.tiers` pins
`HF_HUB_OFFLINE`/`HF_HUB_DISABLE_TELEMETRY` before importing Docling.

### 7.1 What works, what blocks

| Works offline | Why |
|---|---|
| `uv run pytest -q`, `uv run ruff check .` | no network by design; model provider is `mock` |
| psql / DB inspection, planner on stored text, scoring over stored evidence | local Postgres only |
| T0 (pdftext) on born-digital PDFs | pdfium ships in the wheel |
| T1 Docling **text** path (native parser) | no model weights; text + spans + bboxes |
| T1 Docling **layout + tables** | only if `FMD_DOCLING_ARTIFACTS` is populated |
| T2/T2b OCR | only if the vLLM/OCR service is already up locally |

| Blocks offline | Failure / reason |
|---|---|
| T1 Docling with no artifact cache | `artifacts_unavailable_offline` in **<1 s**, routed to T3 (human queue), never a timeout and never a silent T0 substitute |
| Docling artifact prefetch | `docling-tools models download` needs the network |
| Model downloads (`scripts/fetch_model.sh`, vLLM from a Hub id) | must be done online, once; serving is then local |
| Hosted router profiles (`api`, `anthropic`, `gemini`) | HTTP failure; use `FMD_EXTRACT_PROVIDER=mock` or a local vLLM |
| A first crawl / new snapshots | crawling is outbound; acquisition on **cached** snapshots is fine |

### 7.2 Exact commands

```bash
cd packages/pipeline
export FMD_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

# 1) Offline-safe checks (must both pass with no network)
uv run ruff check .
uv run pytest -q

# 2) Full T1, only if artifacts were prefetched while online:
#    docling-tools models download -o "$FMD_DOCLING_ARTIFACTS" layout tableformer
uv run python -m fmd.cli extract --limit 1

# 3) Offline T1 text-only proof (no layout/table weights required):
uv run python - <<'PY'
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.native_pdf_pipeline import NativePdfPipeline
from fmd.acquire.tiers import t1_extract
conv = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_cls=NativePdfPipeline)})
doc = t1_extract(Path("doc.pdf").read_bytes(), converter=conv)
print(doc.tier, len(doc.pages), len(doc.text), len(doc.tables))
PY

# 4) Prove the live router is local, not hosted (`mock` is the no-server option)
FMD_EXTRACT_PROVIDER=mock uv run python -m fmd.cli llm doctor
```

Measured 2026-11-17 on this host: `t1_extract` with `FMD_OFFLINE=1` and no
cache raised `artifacts_unavailable_offline` in **0.11 s**; the same call with
the default converter under `HF_HUB_OFFLINE=1` took **7.6 s** to surface the
Hugging Face cache error. The guard exists so the first number is the normal
offline behaviour.

**CPU T1 baseline (measured 2026-11-17):** the 11-page RUHS Ph.D. research-centre
guidelines/fee PDF (19,036 characters, 7 detected tables) ran T1 with
`device=cpu` in **47.9 s = 0.23 pages/s**, while `llama-server` stayed resident
at 64k. That is the intended split. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is set automatically only on the CUDA path; it cannot help CPU and is not a
substitute for stopping the other GPU consumer.

## 8. Embeddings — bge-m3 on CPU (2026-11-18)

Decision: **doc 10 §12**. The `embed` profile used to point at `local`
(:8010) — the 26B chat server, which does not serve `/v1/embeddings` — so ER
name matching and insight diversity silently ran on lexical heuristics. They
now run on **bge-m3 in-process on CPU** (`FMD_EMBED_PROVIDER=cpu`, the
default), costing **0 MiB VRAM** and leaving the one-GPU-tenant rule intact.
An OpenAI-compatible `POST {FMD_EMBED_URL}/embeddings` backend
(`FMD_EMBED_PROVIDER=embed_openai`, provider entry `embed_openai`) and the
offline `mock` backend are env switches, never code changes.

### 8.1 Cost and footprint

| Item | Value | Class |
|---|---|---|
| VRAM for embeddings | **0 MiB** (in-process CPU; the 26B keeps its 1,153 MiB headroom) | structural, not measured |
| System RAM, bge-m3 fp32 | ~2.2 GB of 32 GB | estimate [U] |
| System RAM, bge-m3 int8 (ONNX, if later needed) | ~1.1 GB | estimate [U] |
| Weights on disk (`~/.cache/huggingface`, bge-m3) | ~2.3 GB fp16 | estimate [U] |
| Throughput | **unverified until the operator runs §8.4** | — |

### 8.2 Install (once, online)

```bash
cd packages/pipeline
uv sync --all-extras          # sentence-transformers ships with the `er` extra
# prefetch the weights once so offline mode (§7) can use them:
uv run python -c "from sentence_transformers import SentenceTransformer; \
  SentenceTransformer('BAAI/bge-m3', device='cpu')"
```

Offline mode (`FMD_OFFLINE=1` / `HF_HUB_OFFLINE=1`) then loads from the local
cache; a missing cache is a **loud** `embeddings_unavailable` naming exactly
that, never a network retry loop.

### 8.3 Start / stop / env

Nothing to start or stop: the embedder lives in the pipeline process, lazily,
on first use. Selection:

```bash
# default (no env needed)
FMD_EMBED_PROVIDER=cpu          FMD_EMBED_MODEL=BAAI/bge-m3
# tuning (process env, read by fmd/llm/embeddings.py)
FMD_EMBED_DEVICE=cpu            # the only supported device for this backend
FMD_EMBED_BATCH=32              # texts per encode() batch
FMD_EMBED_DIM=1024              # must match the model; mismatch is rejected
# second llama.cpp instance (bge-m3 GGUF, --embedding, port 8011) or hosted:
FMD_EMBED_PROVIDER=embed_openai FMD_EMBED_URL=http://localhost:8011/v1
# CI / offline tests:
FMD_EMBED_PROVIDER=mock
```

**Never point the embed profile at `local`** (:8010): that process serves the
gemma-4-26B chat GGUF and cannot embed. `models.yaml` says so at the profile.
A `cpu` provider entry exists in `models.yaml` only because the registry
loader cross-checks profile→provider; it is not an HTTP endpoint, and the
doctor prints `in-process (cpu)` rather than a URL for it.

### 8.4 Verify (copy-paste)

```bash
cd packages/pipeline
# 1) doctor: the embed row must say provider=cpu, model=BAAI/bge-m3, no blockers
uv run python -m fmd.cli llm doctor --live
# expect: embed row usable, REACHABLE ok (cpu probe = import check, no weights
# loaded, no download), and `degrades to embeddings_unavailable` only if the
# `er` extra is missing.

# 2) one real embedding call (the §8.1 throughput row stops being unverified):
uv run python -m fmd.llm.embed_bench --probe
# expect JSON: {"backend":"cpu", "kind":"sentence_transformers", "dim":1024,
#   "vector_dim":1024, "ok":true, ...}; exit code 0.
# A dimension_mismatch or embeddings_unavailable here is a config error; fix it,
# do not lower the dimension.

# 3) measurement (VRAM before/after, texts/s, p50/p95, recommendation):
uv run python -m fmd.llm.embed_bench
```

`embed_bench` (full run) prints — and this is the **expected output shape**, with
every number to be filled by the operator's own run; none are pre-filled and
none may be quoted as measurements until they are:

```json
{
  "backend": "cpu",
  "vram_before": {"used_mib": "<nvidia-smi>", "total_mib": "<nvidia-smi>"},
  "warmup_s": "<first call, includes model load>",
  "rows": [
    {"kind": "names|sentences", "texts": "1|32|256|1024", "wall_s": "<measured>",
     "texts_per_s": "<measured>", "p50_ms": "<measured>", "p95_ms": "<measured>",
     "tokens_in": "<measured or \"[not reported]\">"}
  ],
  "vram_after": "<same keys as vram_before>",
  "vram_delta_mib": "<expected ~0 for the CPU path; a nonzero delta is a finding>",
  "recommendation": {
    "verdict": "cpu_backend_fit_for_purpose | cpu_backend_misses_budget",
    "checks": [
      {"check": "er_names_within_budget",      "measured_s": "<measured>", "budget_s": 60.0},
      {"check": "insight_sentences_within_budget", "measured_s": "<measured>", "budget_s": 10.0}
    ]
  }
}
```

Paste the filled JSON into §5 of this runbook; until then the §8.1 throughput
row stays `unverified`. Exit code 1 means the measured batch missed a budget:
tune `FMD_EMBED_BATCH` / torch threads before considering a backend switch
(doc 10 §12.3).

### 8.5 What `embeddings_unavailable` looks like (all surfaces, none silent)

| Surface | What appears |
|---|---|
| `fmd llm doctor` | embed row `usable=false`, blockers name the exact env var (`FMD_EMBED_*`), and `degrades to embeddings_unavailable` — the run still exits 0 because the profile is optional |
| ER audit (`er_audit.features` / resolution notes) | `features.embedding: null`; note `embedding_unavailable: <cause>` (the ER note prefix is the legacy singular token; the canonical `embeddings_unavailable` mode is carried on the scorer) |
| Insight sets (`insight_sets.coverage_json`) | `embedding_mode: "embeddings_unavailable"` + `embedding_note: <cause>`; `semantic_duplicates: []` — selection ran lexically |
| Scorer object (`fmd.er.similarity.EmbeddingScorer`) | `mode == "embeddings_unavailable"`, `reason` names the variable to fix; a backend that fails mid-run deactivates once, loudly, instead of retrying per pair |

The lexical fallback itself is the documented heuristic and is unchanged:
`token_jaccard + char_ratio` blend, acronym scoring, context matches with
state/city/pincode conflict penalties (`fmd/er/similarity.py`). Insight
selection without embeddings is predicate-string diversity only — recorded as
such, so a reviewer can tell which evidence produced a shortlist.

### 8.6 Common failures

| Symptom | Meaning | Action |
|---|---|---|
| doctor: `sentence-transformers is not installed` | `er` extra missing from the sync | `uv sync --all-extras`; verify with `uv run python -c "import sentence_transformers"` |
| `--probe` → `embeddings_unavailable ... weights could not be loaded` | weights never prefetched, or offline mode set without a cache | run the §8.2 prefetch once while online |
| `dimension_mismatch: ... 768, expected 1024` | `FMD_EMBED_MODEL` or `FMD_EMBED_DIM` changed without the other | align `profiles.embed.dim` with the served model; vectors are never truncated |
| embed row `embed_openai` + `FMD_EMBED_URL is empty` | URL unset | set it (`llama-server --embedding --pooling cls --port 8011` or a hosted endpoint); it is deliberately not defaulted |
| `vram_delta_mib` far from 0 on the CPU path | something loaded CUDA (torch picked GPU, or another process moved) | check `nvidia-smi`; `FMD_EMBED_DEVICE=cpu` is load-bearing |
| hosted endpoint 401/5xx mid-run | key/quota/server | the run records `embeddings_unavailable` with the reason and continues lexically; fix and re-run |

## Measured: KV budget and context ceiling (2026-11-17, RTX 5080 Laptop 16303 MiB)

Sweep at `-ngl 999 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja -rea off`, weights
`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` (12.66 GiB, sha256 verified `babd1e38…`).
Each step confirmed the server **served** a completion, not merely loaded.

| ctx | result | VRAM used (MiB) | free (MiB) | decode tok/s |
|---|---|---|---|---|
| 16k | ok | 13,708 | 2,595 | 84.5 |
| 24k | ok | 13,818 | 2,485 | 104.5 |
| 32k | ok | 13,942 | 2,361 | 106.8 |
| 48k | ok | 14,192 | 2,111 | 106.4 |
| **64k** | **ok** | **14,442** | **1,861** | **106.3** |
| 96k | ok | 14,942 | 1,361 | 103.0 |
| 128k | ok | 15,442 | 861 | 106.3 |

**Operating point: 65536.** 128k technically loads but leaves 861 MiB, which is not
enough for compute spikes plus a concurrent API/worker process. 96k is viable
(1.36 GiB free) if a job genuinely needs it; 64k is the default.

Probe (all PASS): text completion; no `<|channel>thought` leakage in `content`;
guided `json_schema` returns a schema-valid object; **nullable fields correctly
return `null`** (`{"institute_name": null, "seats": null, "is_recognised": null}`),
confirming GBNF neither forces nor forbids null.

Notes: first cold load took 44 s; subsequent loads 4–5 s (page cache). Idle GPU
after unload: 14 MiB, ~19 W.

## Bulk ramp: first bounded pass (2026-11-17)

The model is proven; the bottleneck is evidence volume. This is the first
bounded ramp from an almost-empty DB, not a scale run. Every cap below is
registry data and belongs in the run ledger with the numbers it produced.

### Caps (exact keys)

`config/sources.yaml` → `sources[id=institution_sites].params`:

| key | value | meaning |
|---|---|---|
| `max_pages_per_domain` | `20` | navigation pages fetched per institution domain |
| `max_documents_per_domain` | `3` | evidence PDFs kept per domain |
| `targeting.max_domains_per_run` | `50` | inventory-discovered domains planned in one ingest |

The per-run document cap is the CLI argument, not config: `fmd crawl --limit 300`.
Politeness stays at the registry/env defaults (`FMD_DEFAULT_RPS=1` per
registrable domain, robots honoured). Raising any cap requires a fresh review of
the previous run's `deduped`/`failed`/`abstained` counts.

### Order of operations (host venv)

```bash
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd
export FMD_SNAPSHOT_DIR=$HOME/fmd-snap2
cd packages/pipeline

# 1. inventory: fetch the UGC payloads, then entities/domains
uv run python -m fmd.cli crawl --source ugc_university_list
uv run python -c 'from fmd.config import load_registry; from fmd.db import Database; \
from fmd.orchestrate import Orchestrator; d=Database(None); \
print(Orchestrator(registry=load_registry(None), database=d).ingest().as_json()); d.close()'

# 2. evidence crawl (detached; one summary at the end)
setsid nohup env ... uv run python -m fmd.cli crawl --source institution_sites \
  --limit 300 > /tmp/opencode/ramp/crawl.log 2>&1 &

# 3. extract + inline verify, then an explicit verify pass for leftovers
FMD_EXTRACT_PROVIDER=local FMD_EXTRACT_MODEL=gemma-4-26b-a4b-gguf \
  uv run python -m fmd.cli extract --limit 50
FMD_VERIFY_PROVIDER=local FMD_VERIFY_MODEL=gemma-4-26b-a4b-gguf \
  uv run python -m fmd.cli verify --limit 500
```

### What the first pass produced

Inventory: **1,247 entities**, **1,204 institution domains** (1,153 pending,
50 queued, 1 planned) from 2,028 UGC records. The UGC payload has 1,304 rows for
`unitypeID=0`, 560 for `3`, 164 for `4`; `unitypeID=1,2` answered empty
(`yield_score=0`), so 1,200+ is the real ceiling of this source shape.

Evidence crawl (50 domains, 300-document cap, 38 min): 103 URLs planned,
94 fetched, 21 stored, **73 deduped** (simhash; see §Dedupe audit below —
the 2026-11-18 audit found these were *false positives*, not boilerplate),
20 robots abstentions, 15 network failures (13 connect timeouts, 1 read
timeout, 1 DNS). 47 documents linked to a family: `accreditation` 27,
`placement_report` 10, `curriculum` 9, `seat_matrix` 1. 27 PDFs have snapshots.

Extraction: 15 extractable snapshots → **7 documents extracted, 7 claims, 3
rejected claims** (`ambiguous_span` ×2, `unparseable_unit` ×1), 4 document
abstentions (3 `unresolved_provider` for `hard_pdf` because the `api` profile
has no base URL, 1 `TierUnavailable` for a scanned PDF with no T2 endpoint).
All 7 new claims verified `supported`; 0 unverified. Every claim carries
`qualifier_json.page_no` and a glyph `bbox`, and its quote resolves in
`extractions.text`.

Scoring: **no programmes existed at the time of the first pass** (0 rows), so
composite bands were not meaningful. Programme materialisation landed on
2026-11-18 (§Programme materialisation); the predicate-alias fix later the same
day made components scoreable — see that section for the recomputed bands.

### Log commands

```bash
tail -f /tmp/opencode/ramp/ingest.log       # ingest summary (single JSON line)
tail -f /tmp/opencode/ramp/crawl.log        # tier warnings, then run summary
tail -f /tmp/opencode/ramp/extract.log      # per-document outcomes + summary
tail -f /tmp/opencode/ramp/verify.log       # verify pass summary
tail -f /tmp/opencode/ramp/score.log        # score summary
```

Run summaries are also persisted in `pipeline_runs` (query by `run_id` printed
at the end of each log) and mirrored into `docs/` only when a report is asked
for; the logs above are the run-summary files for this pass.

## Programme materialisation (2026-11-18)

Scoring selects from `programmes`. Before this stage that table had 0 rows, so a
12-claim evidence base produced an empty shortlist by construction.

### Contract

`fmd/ingest/programmes.py` (pure) + `ProgrammeStage` in
`fmd/orchestrate/ingest.py` (persistence). One row per
`(institution, ER-normalised programme name)`.

* Input is **only** `claims.verification_status='supported'`, unsuperseded,
  `method <> 'modelled'`, from a `documents.role='evidence'` document whose
  entity is an institution. Navigation/inventory documents are refused in SQL.
* Every set attribute records its claim id in `programmes.field_sources`,
  keyed by the **predicate** that proved it
  (`{"seats.total": "5", "seats.programme_name": "4"}`). An attribute with no
  claim stays `NULL`; no seats/duration is ever defaulted.
* Names go through `fmd.er.normalise.normalise_name` plus the registry alias
  table `config/sources.yaml → institution_sites.params.programme_mapping`.
  `B.Sc. MLT` and `Bachelor of Medical Laboratory Technology (BMLT)` collapse to
  `bsc mlt`; MLT and MLS stay separate keys and are flagged
  `needs_review` (`possible_duplicate_programme_name:…`) rather than merged.
  Multiple names at one institution get one row each; unqualified attributes
  are left `NULL` with `attribute_programme_ambiguous`.
* Re-running is idempotent: `ON CONFLICT (institution_id, normalised_name)` and
  the row is replaced with the current draft (including NULLs).

Invocation (W8 onwards, no script needed; the crawl path runs the same code
via `ProgrammeStage`):

```bash
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd
cd packages/pipeline
uv run python -m fmd.cli programmes materialise
uv run python -m fmd.cli score
```

(2026-11-20, W8: materialisation is also available outside the crawl path as
`fmd programmes materialise|list|inspect` — see §12 below. The write loop
lives in `fmd/ingest/materialise.py`; `ProgrammeStage.run` delegates to it.)

### Measured on the 12-claim DB

2 programmes created from 12 supported claims; 3 entities rejected
(`no_programme_name_claim`: entities 1, 704, 1027 — accreditation/placement
facts without a programme name).

| programme | institution | name | seats | internship | duration | field_sources |
|---|---|---|---|---|---|---|
| 1 | 2 TN Dr MGR MU | B.Sc. Medical Laboratory Technology | 20 | — | — | seats.programme_name=4, seats.total=5 |
| 2 | 10 Adesh University | Bachelor of Medical Laboratory Sciences | — | 1 mo | — | curriculum.programme_name=10, curriculum.internship_months=12 |

Scores before the alias fix: 2 written, **0 with any scoreable sub-score**, composite
`UNKNOWN` [0,100], `evidence_tier=insufficient`. Two independent blockers were
recorded here:

1. **Predicate vocabulary mismatch** (the main one): extraction emits
   namespaced predicates (`seats.total`, `accreditation.naac_grade`,
   `curriculum.internship_months`) while `fmd/score/sub_scores.py` looked up
   bare names (`seats`, `naac_grade`, `internship_hours`). No claim was visible
   to any component.
2. **`min_claims_for_score=2`**: even after aliasing, entity 2 contributes one
   `seats` claim and entity 10 one `internship_*` claim, so a sub-score would
   still be UNKNOWN at this n. `duration` came from `curriculum.semesters=8`
   only; converting semesters to months is a derivation that this stage
   deliberately does not perform, so `duration_months` is NULL.

### Predicate alias fix and recomputation (2026-11-18, WS-SCORE-ALIAS)

`config/predicates.yaml` now maps every extraction predicate to a component,
unit kind and denominator/period rule; `fmd/score/sub_scores.py` builds its
component specs from that registry. `min_claims_for_score` is 1 with interval
widening, and a subset composite is `partial` with `components_used` /
`components_unknown` (see brief §11). Re-running `fmd score --force` on the same
12-claim DB produced:

| programme | component (n) | component value [interval] | composite band | tier | components used / unknown |
|---|---|---|---|---|---|
| 1 seats=20 | `admission_realism` (1) | 65.97 [58.59, 73.34] | 65.97 [5.86, 97.33] partial | thin | 1 / 6 |
| 2 internship=1 mo | `clinical_training` (1) | 8.33 [0.00, 38.05] | 8.33 [0.00, 90.71] partial | thin | 1 / 6 |

Every other component is `UNKNOWN` [0,100]: the remaining claims are
evidence-only (`seats.intake_year`, `seats.programme_name`,
`curriculum.programme_name`, `curriculum.semesters`) and do not score.
`unknown_predicates` is empty; the 1-month internship scores through the
explicit `internship_months` transform, not an hours conversion. Semesters are
still not converted to months.

Ranking at n=2 is **not defensible**: the two partial composites rest on
disjoint components (seats vs internship months), so `compare()` returns
`not_comparable` and no ordering is published. The numbers are evidence, not a
shortlist.

## Wave-2 fee/seat/placement predicate table (2026-11-18, WS-FEE-SEAT-SCHEMA)

The measured yield was accreditation/curriculum heavy and fee/seat poor: of 32
claims, `fee.tuition` and placement salary were effectively absent, so the
affordability/DSR25 half of the product had no evidence to score. Wave 2 adds
the missing fields to the `fee_structure` / `seat_matrix` / `placement_report`
families, the guards that keep them honest, and their registry rows. The
predicate table below is a historical summary of wave 2 (the seat rows were
re-split by §Seat predicate split and verifier descriptions, and
`config/predicates.yaml` remains the machine-readable source of truth, now
version `2026-11-18.3`, every row carrying a one-sentence `description`). No
weight value changed: every new predicate is evidence-only (`component: null`)
because scoring them would require new `weights.yaml` transforms, which is a
scoring-workstream change, not an extraction one.

| Predicate | Component | unit_kind | Period / denominator | Emitted by | Notes |
|---|---|---|---|---|---|
| `fee.mess` | null | money | period required | fee_structure v4 | mess/food charge; evidence-only |
| `fee.other` | null | money | period required | fee_structure v4 | transport/internet/sports/misc; distinct from `fee.mandatory` |
| `fee.period` | null | text | — | fee_structure v4 | document-level period (`year`\|`semester`\|`month`); supplies the period a money span omits |
| `seats.intake` | admission_realism | count | — | seat_matrix v5 | sanctioned intake for THIS programme at THIS institution (one row of the matrix; the Stanley case) |
| `seats.total` | admission_realism | count | — | seat_matrix v5 | aggregate across programmes/institutions; only when the document itself states a total row |
| `seats.admitted` | null | count | — | seat_matrix v4 | filled count; guarded ≤ sanctioned intake |
| `seats.duration_months` | null | months | — | seat_matrix v4 | consumed by programme materialisation; never derived from semesters |
| `placement.salary_period` | null | text | — | placement_report v3 | document-level salary period; supplies the period a salary span omits |
| `placement.recruiters_count` | null | count | — | placement_report v3 | distinct recruiters; gaming-resistance evidence |

Alternate spellings are declared **aliases**, not extra emitting fields — the
golden placement fixture pins the canonical names
(`placement.eligible_count`, `placement.placed_count`, `salary.median`,
`salary.highest_package`) and the golden seat-matrix entry pins `seats.intake`
(renamed from `seats.total` by the seat-semantics split: the Stanley quote is
one programme's row in one college, so the aggregate name was the bug), so
re-emitting the same fact under a second name would double-count claims and
break the eval. The registry accepts both spellings and resolves them to one
row: `seats → seats.intake`, `seats.sanctioned → seats.intake` (both legacy
spellings meant "the sanctioned seat count for one programme", a row-scoped
value — pointing them at the aggregate was the alias bug this wave fixed),
`placement.eligible → placement.eligible_count`, `placement.placed →
placement.placed_count`, `placement.median_salary → salary.median`,
`placement.highest_salary → salary.highest_package`.

**Guards added with the fields** (all in `fmd/extract/validate.py`, all
reject — never coerce):

* **Money requires currency + period on the claim.** A money field whose span
  prints no period takes it from the sibling claim named by the new
  `FieldSpec.period_source` (`fee.period`, `placement.salary_period`); the
  claim records that provenance as `qualifier.period_source`. A money field
  that ends up with no period anywhere is rejected `missing_period` — the old
  silent `per_year` default on fee fields is gone. Currency behaviour is
  unchanged: printed in the span, or inherited from the `fee.currency`
  sibling for the fee family.
* **`admitted_exceeds_sanctioned_intake`** — `seats.admitted` > `seats.total`
  rejects the admitted claim (same shape as `placed_exceeds_eligible`).
* Every existing guard still fires: denominator on rates, placed ≤ eligible,
  rate ≤ 100% and consistent with printed counts, quotas ≤ sanctioned intake,
  uncertain column header, out-of-band values.

**Row-context discipline is unchanged** and extends to the new columns: a
value in a sanctioned/admitted column resolves only inside the table block its
`context` names, and `qualifier.row` carries `{row_label, column_header,
value_cell, table_page}`, so 18 identical programme rows stay distinguishable.

**Prompts:** `fee_structure@v4` and `seat_matrix@v4` (both were at v3) and
`placement_report@v3` (was at v2) — each family keeps all previous versions,
and v1–v3/v1–v2 remain replayable. The recorded replay fixture for the
placement pipeline tests was re-recorded under the v3 prompt hash; the golden
placement entry still replays its committed v2-era `llm_fixture` directly, so
golden provenance stays 1.0.

**Programme materialisation** consumes `seats.intake` (the per-programme
sanctioned intake, with its declared `seats.sanctioned` alias spelling) and
`seats.duration_months` when present (`DEFAULT_PROGRAMME_MAPPING` and the
`config/sources.yaml → institution_sites.params.programme_mapping` block list
them in that order, so canonical claims win when both exist).
`seats.total` is deliberately **not** consumed: an aggregate across
programmes/institutions must never be written into one programme's `seats`
column — a document that states only a total leaves `programmes.seats` NULL
(`UNKNOWN` is a value). `seats.admitted` is a filled-count demand signal and is
never treated as intake. There is still no derivation:
`curriculum.semesters` is never converted to months, and a
`seats.duration_months` claim whose unit is not months is ignored. Every
consumed claim id lands in `programmes.field_sources` under its own predicate.

## Verifier row-context pass (2026-11-18, W2 verifier-row-context)

**The bug this fixes was measured, not hypothetical.** The wave-2 row-scoped
extraction contract (§Wave-2 fee/seat/placement predicate table) made the
extractor emit quotes that wrap the value in its own row's wording. On the real
Tamil Nadu AHS UG seat matrix, `seats.total` = 20 for the Stanley block carried
the quote `"3. B.Sc. Medical Laboratory Technology\n20"` with
`qualifier.context_quote = "Govt. Stanley Medical College & Hospital, Chennai -
600 001"` (`disambiguation: context_row`, row "3. B.Sc. Medical Laboratory
Technology", column "Intake"). The verifier was still `verify@v1`, whose prompt
assumes a quote is only the value: it judged that quote `wrong_field` **and the
repair pass returned the same verdict**, so a correct claim was discarded. That
is a false rejection — lost recall, not a caught error — and it hit exactly the
row-scoped claims wave 2 exists to produce.

### What verify@v2 changes

`config/models.yaml → profiles.verify.prompt_version: v2`; `verify/v1.md` and
`verify_repair/v1.md` are kept and still load (the verify family now has the
same completeness assertion as the extract families: contiguous v1..vN, every
file real and non-empty, latest = v2). The v2 prompt documents three things the
v1 prompt got wrong by omission:

| Contract | What the prompt now says |
|---|---|
| Quote shapes | A quote may be a bare value ("20 seats"), a label+value pair inside one table row ("3. B.Sc. Medical Laboratory Technology 20", label and number possibly on separate lines), or a value plus unit/period ("₹30,000 per annum"). A row label inside the quote is not a different field. |
| Row-scoped association | With `context`/`row_label`/`column_header` present, the thing verified is the association: does the value this quote wraps belong to this row and this column? A genuinely wrong row/column stays `wrong_field` (repair included); a correct association is never `wrong_field` merely because the quote repeats the row label. |
| Expected type gates | `count` / `money+period` / `percentage+denominator` / `grade` / `year` — a money claim whose window prints neither currency nor period is `absent`; a count next to a label is normal. |

### Plumbing (all in `fmd/verify/`, nothing in the extract path)

* **Row context travels in the payload.** `ClaimView` now carries `context`,
  `row_label`, `column_header` (parsed from `qualifier.context_quote` and
  `qualifier.row`), plus the qualifier's `context_char_start/end` and
  `row_char_start/end` spans. The field object the model sees is
  `{key, field, quote, expected, context?, row_label?, column_header?}` — the
  claimed value and the row's printed `value_cell` are still never sent.
* **Expected type is registry data, not claim data.** The `Verifier` takes the
  predicate registry (`config/predicates.yaml`) via the new `predicates=`
  argument, wired from `fmd verify`, `fmd eval --verify-fixtures`, the
  orchestrator's inline verifier and the offline case runner. `money` →
  `money+period`; `percent`/`ratio` with `requires_denominator` →
  `percentage+denominator`; `count`/`hours` → `count`; unknown kinds pass
  through verbatim; an unknown predicate arrives as `null`.
* **Window sizing covers the row.** The window is seeded from the union of the
  quote span and the row-context spans, so the verifier sees the row it is
  asked to judge; a block wider than `window_chars` falls back to the
  quote-centred window (the context still travels as payload text). Bundling
  still shares only identical windows, so the same quote asserted against two
  different rows is judged with each row's own context.
* **Verdict semantics unchanged otherwise.** Same four verdicts, `absent`
  still rejects immediately, `contradicted`/`wrong_field` still get exactly one
  single-field repair call, the verifier still never emits a value, and
  decisions now record `expected_type` and `row_context` in
  `verification_notes` next to `prompt_version: verify@v2` for review.

### Fixtures (offline, no network, recorded replay only)

* `fabrication_v1` was re-recorded under the v2 prompt hashes (same scripted
  verdicts as before, so its metrics are unchanged: 7/7 caught, 0 false
  accepts, 0 correct rejected, 2 bundles + 1 repair). The old v1-hash files
  stay for replay of historical runs, matching the wave-2 convention.
* New case `tests/fixtures/verify/row_context_v2/` is the anti-regression that
  matters. One page modelled on the real Stanley page, four claims:
  `s01` the real Stanley claim (label+value quote, context_row) → `supported`;
  `s02` the **counterfactual** — the same quote asserted against the
  Chengalpattu block, whose MLT row is "1." not "3." → `wrong_field`, repair
  re-issued `wrong_field`, final `rejected`
  (`verifier_wrong_field_after_repair`); `s03` a money claim quoting a bare
  amount with no currency or period anywhere → `absent`, no repair; `s04` a
  bare-value quote ("20 seats") with no row context → `supported` exactly as
  under v1. Metrics: 2/2 caught, 0 false accepts, 0 correct rejected,
  verified_rate 1.0.

```bash
cd packages/pipeline
uv run pytest tests/test_verify.py -q          # incl. the counterfactual tests
uv run python -m fmd.cli eval --golden ../../data/golden \
  --verify-fixtures tests/fixtures/verify/row_context_v2
# expect: verification: verified_rate 1.0 fabrication_caught 2/2 correct_rejected 0
```

**Known limits, stated plainly:** the committed fixtures prove the plumbing
(call → forced verdict → selective repair → persistence → metrics) against the
verdicts the v2 prompt contract requires; they do not prove a live model
generalises. The next operator pass over the real Stanley PDF with
`FMD_VERIFY_PROVIDER=local` should confirm `s01`-shaped claims come back
`supported` on hardware, and that `s02`-shaped claims still do not.

## Dedupe audit (2026-11-18)

The 19:22 crawl dropped 73 of 94 fetched documents as `near_duplicate_simhash`.
Audit of those 73 documents:

* **73/73 are PDFs**; 0 HTML/JSON.
* **33 distinct domains**; **71 distinct filename stems** — only one stem
  appears twice (a genuine same-name republication). 72/73 are therefore
  *distinctly named documents*: annual reports by year, English vs Hindi
  editions, prospectuses, certificates, circulars.
* Mechanism, measured from the object store: the crawler's `FetchResult.text`
  is raw body bytes decoded as UTF-8, not extracted PDF text. 64-bit simhash
  over decoded PDF container bytes is dominated by `%PDF`/object-stream/font
  boilerplate. Recomputing full-body simhash over 12 stored PDFs gives 31/66
  pairs at Hamming ≤3 (the configured threshold); a 64 KB-prefix pass over all
  122 stored PDFs gives 1,905/7,381 pairs ≤3. Unrelated PDFs collide by
  construction, so raising/lowering `simhash_threshold` cannot fix it.

Verdict: **false positives, material (≥72/73, 98.6%)**. The durable fix landed in
`fmd/fetch` + `fmd/orchestrate/crawl.py` on 2026-11-18: a PDF fingerprint is now
computed over a bounded **extracted-text** sample (T0 pdftext, first
`text_max_pages` pages, `text_max_chars` characters), never over container
bytes. `exact` remains the fallback when no text layer can be read. The config
temporary mitigation (`mode: exact`) was reverted to `mode: simhash` once the
tests and the recovery run below proved the new basis. See
§Dedupe basis and evidence recovery.

Re-run the audit offline against a run summary that carries
`metadata.dedupe` (URL, compared_url, similarity, hamming_distance, mode,
basis, text_sha256):

```python
from fmd.ingest.dedupe_audit import audit
print(audit(entries).as_dict())   # boilerplate vs false_positive, by_source
```

## Dedupe basis and evidence recovery (2026-11-18)

### What a decision is computed over

| `basis` | computed from | used when |
|---|---|---|
| `raw_bytes` | `sha256` of the fetched bytes | exact mode, and the fail-safe for a PDF with no readable text layer |
| `extracted_text` | `sha256` + 64-bit simhash of normalised **document text** | PDFs (T0 pdftext sample) and HTML/plain bodies |
| `identity` | row identities from configured fields | structured/paginated listings (`identity` mode) |

Every decision is persisted on the `documents` row (migration 0007) —
`text_sha256`, `dedupe_mode`, `dedupe_basis`, `dedupe_similarity`,
`dedupe_reason`, `dedupe_compared_document_id`, `dedupe_simhash` — because a
dropped document has no snapshot row. A stored snapshot also carries
`text_sha256` + `dedupe_basis`. A later decision replaces the previous one
wholesale, so a document that was dropped and later stored cannot keep showing
the old reason.

Order of decisions: same URL + same bytes (`unchanged_content`) → same bytes
under another URL (`duplicate_sha256`) → same extracted text under different
bytes (`duplicate_text_sha256`) → text simhash ≤ `simhash_threshold`
(`near_duplicate_simhash`, **non-PDF only**) → row identities
(`near_duplicate_identity`).

**PDFs are never dropped on approximate similarity.** Two editions of one
template differ in a handful of tokens (a year, a seat count). The first
recovery run dropped AdaMAS `PLACEMENT-INTERNSHIP-POLICY-2021-22` and
`…-2023-24` at Hamming distance 0 even though their extracted text differed.
The rule now is: a PDF is dropped only when its full normalised text digest
matches. The simhash is still computed and stored, so a review can see the near
miss. When extraction fails (scan, encrypted/broken PDF, missing extract extra)
the comparison degrades to `sha256` and the document is stored.

Registry keys (`config/sources.yaml → institution_sites.params.dedupe`):

```yaml
dedupe:
  mode: simhash
  simhash_threshold: 3
  text_max_pages: 3     # bounded read: first N pages
  text_max_chars: 20000 # bounded read: first N characters
```

### Recovery command

Re-crawling a dropped document is not `crawl` again: the frontier keeps
`done`, and the stored validator makes the retry a `304`. A revalidation is only
useful once a snapshot exists; for a document that was never snapshotted the
conditional headers must be cleared first (the replay shape persists them by
design, so they survive clearing `etag`/`last_modified`). Bounded recovery, same
caps:

```bash
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd
export FMD_SNAPSHOT_DIR=$HOME/fmd-snap2
cd packages/pipeline

# 1. pick the affected URLs: fetched 200/304, PDF, no snapshot row
psql "$DATABASE_URL" <<'SQL'
UPDATE documents d
   SET etag = NULL, last_modified = NULL,
       request_headers = COALESCE(d.request_headers,'{}'::jsonb)
                         - 'If-Modified-Since' - 'If-None-Match'
 WHERE d.source_id = 'institution_sites'
   AND d.url ~* '\.pdf'
   AND d.http_status IN (200, 304)
   AND NOT EXISTS (SELECT 1 FROM snapshots s WHERE s.document_id = d.id);

UPDATE crawl_frontier f
   SET state = 'queued', attempts = 0, next_allowed_at = now(), updated_at = now()
  FROM documents d
 WHERE f.url = d.url
   AND d.source_id = 'institution_sites'
   AND d.url ~* '\.pdf'
   AND d.http_status IN (200, 304)
   AND NOT EXISTS (SELECT 1 FROM snapshots s WHERE s.document_id = d.id);
SQL

# 2. drain only (no re-planning, so no new domains/documents enter the run)
uv run python - <<'PY'
import json
from fmd.config import load_registry
from fmd.db import Database
from fmd.orchestrate import Orchestrator
from fmd.orchestrate.crawl import Crawler
from fmd.orchestrate.ledger import RunSummary, SqlLedger, utcnow
db = Database(); reg = load_registry(None); o = Orchestrator(registry=reg, database=db)
src = reg.sources.by_id("institution_sites")
c = Crawler(frontier=o.frontier, fetcher=o.fetcher, store=o._store(),
            documents=o.documents, snapshots=o.snapshots,
            adapter_factory=o.adapter_factory,
            denylisted_domains=reg.sources.denied_domains(), sources=[src])
s = RunSummary(run_id="dedupe-recovery-0007", stage="crawl", started_at=utcnow())
c.drain(summary=s, limit=300); s.finish(); SqlLedger(db).record(s)
print(json.dumps(s.counts, sort_keys=True))
print(json.dumps(s.metadata.get("dedupe", []), indent=2))
db.close()
PY
```

Inspect the result without trusting the counters:

```sql
SELECT dedupe_basis, dedupe_reason, count(*) FROM documents GROUP BY 1,2 ORDER BY 3 DESC;
SELECT count(*) FROM snapshots;  -- must rise by the stored count
```

### Measured recovery (2026-11-18)

98 affected PDF URLs across 34 institution domains (all `http_status` 200,
no snapshot). Four bounded runs (`dedupe-recovery-0007`, `…b`, `…c`, `…d`, no
new planning) fetched **100**, stored **96**, deduped **3**. Of the 3 drops, 1
was a genuine byte-identical circular under two URLs and 2 were the AdaMAS
edition pair (false positives; fixed above, re-run and stored).

| metric | before | after |
|---|---|---|
| documents | 147 | 147 (same inventory) |
| snapshots | 39 | **135** (+96) |
| institution-site snapshots | 27 | **123** |
| evidence PDFs recovered | — | **96** |
| extractions | 16 | 22 |
| claims | 12 | **14** (1 `supported`, 1 `rejected:unresolved_span`) |
| claims verified | 12 | 13 |

The only 3 PDFs still without a snapshot are accounted for: 1 `404`
(`sau.int`), 1 genuine byte-identical duplicate of a stored circular
(`sindhucu.ac.in`, two URLs, same bytes), and 1 robots/timeout abstention
(`annamacharyauniversity.edu.in`, still queued). The other 9 documents without
a snapshot are not evidence (7 inventory, 2 HTML with `family_undetected`).
Extraction is bounded and resumable; the API `deepseek-v4.1-flash-free` profile
returned `429 FreeUsageLimitError` during the recovery extract, so the targeted
pass used the local `gemma-4-26b-a4b-gguf` profile. The two new claims come from
recovered NIRF reports (`cus.ac.in`, `aus.ac.in`).

## Planning pass (`fmd plan`) — replanning recovered evidence (2026-11-18, W4 planning-pass)

### The measured problem

After the dedupe recovery above (96 snapshots stored by a drain built without a
planner), the next harvest reported:

```json
{"counts": {"abstained": 4, "not_applicable": 14, "unplanned": 84, "claims": 0}, "ok": false}
```

`unplanned: 84` = evidence documents with `plan_state IS NULL` (never planned),
counted by `documents.count_unplanned()` during extract. Extraction never
selects them — `SnapshotsRepo.list_extractable` requires
`extraction_family IS NOT NULL AND entity_id IS NOT NULL` — so the run produced
zero claims. Planning existed only inside crawl/extract; documents that arrived
through snapshot recovery, backfills or re-crawls were never planned.

### The command

```bash
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd
cd packages/pipeline
uv run fmd plan [--limit N] [--force] [--dry-run]
```

Resumable (oldest document id first, `--limit` bounds a run) and idempotent (a
decision identical to the stored row is not written again — a second run makes
no changes). `--dry-run` computes every decision and the counters and writes
nothing.

**Candidate query** (`DocumentsRepo.list_plan_candidates`): every document that
should be extractable but is not —

| Queue | Rows |
|---|---|
| default | `plan_state IS NULL` (never planned) plus `plan_state = 'planned'` (Phase A deferrals, `plan_reason='defer_for_text'`) |
| `--force` | adds *decided* rows (`linked` / `needs_review` / `abstained` / `not_applicable`) whose `plan_reason` is planner-shaped but not stamped at the current planner version |

Non-evidence roles (inventory / navigation) with no decision are planned to
`not_applicable` — an answer, never an abstention. Decisions written by human
review (prose reasons such as `resolved by human review (link_entity)`) are
**never** replanned at any version: a manual decision outranks the planner.

### What each document gets

Precedence is the planner's own (reused, not reimplemented):
**registry-declared family > Phase B text evidence > Phase A URL evidence**.
Phase B reads the stored snapshot's born-digital text layer via T0 only — no
OCR, no model, no network, bounded to `plan.yaml → phase_b.text_sample_chars`.

* evidence with a snapshot and readable text → planned at Phase B;
* evidence with **no snapshot, or a snapshot with no T0 text layer** (a scan) →
  **`awaiting_text`**: the row is persisted exactly like a Phase A deferral
  (`plan_state='planned'`, `plan_reason='defer_for_text'`, family/entity
  preserved), so the existing acquisition pipeline owns it and the next pass —
  or the extract-time `PlanStage`, once a snapshot exists — resolves it. The
  pass never links a document from its URL alone and never acquires text;
* evidence whose T0 read fails (corrupt bytes, missing extraction extra) →
  counter `acquisition_failed`, row untouched, retried next run;
* evidence with stored reference hints (discovery family / inventory entity) and
  no text → re-affirmed at Phase A exactly as crawl would.

Planned decisions are stamped with the planner version
(`config/plan.yaml → version`, e.g. `plan_reason='plan_v1'`; abstentions carry
`<code>|plan_v1`) — the per-document version record, since `documents` has no
version column and migrations are out of scope. `--force` re-decides stale
versions only (unstamped crawl-era rows and older-version stamps); rows stamped
at the current version are skipped (`plan.skipped_current`).

### Counters (run summary and `pipeline_runs.counts_json`)

| Counter | Meaning |
|---|---|
| `plan.candidates` | documents examined |
| `planned` | decisions this run (the breakdown below sums to it) |
| `linked` | family + entity resolved; extraction can read them |
| `needs_review` | ambiguous entity/family; review queue |
| `still_abstained` | abstained; breakdown in `metadata.reasons` and `plan.reason.<code>` |
| `skipped_not_applicable` | inventory/navigation role |
| `awaiting_text` | no snapshot or no T0 text layer |
| `acquisition_failed` | T0 read raised; row untouched |
| `unchanged` | decision identical to the stored row (idempotency proof) |
| `plan.skipped_current` | `--force` skips: current-version stamp or human decision |

`pipeline_runs.stage` is a closed set (`crawl/extract/score/worker/ingest`) and
this wave must not migrate, so the run row is recorded under `stage='crawl'` —
planning is the crawl stage's responsibility and its counters are already
plan-shaped there. Identify a pass row by `run_id LIKE 'plan-%'` (status
`succeeded` / `dry_run` / `failed`):

```sql
SELECT run_id, started_at, status, counts_json
FROM pipeline_runs WHERE run_id LIKE 'plan-%' ORDER BY started_at DESC;
```

### Running order

Slot the pass between acquisition and extraction in the §6 ordered pass:
`fmd plan` → `fmd extract` → `fmd verify` → `fmd score`. After a recovery drain,
`fmd plan --dry-run` first: `plan.candidates` is the remaining planning debt and
must fall to `0` (deferrals aside) after a real run.

### Proof (offline; no network, no model)

`tests/test_orchestrate_plan_pass.py` covers: unplanned→planned; already-planned
skipped; `defer_for_text` resolved by Phase B; evidence without snapshot →
`awaiting_text`; role=inventory/navigation → `not_applicable`; idempotent second
run writes nothing; `--dry-run` writes nothing; counters accurate; `--force`
replans stale versions only; and `fmd extract` consumes the newly planned
documents against the recorded provider — no change to `fmd/orchestrate/extract.py`.
`tests/test_plan_db.py` pins the candidate SQL against Postgres. Measured CLI
smoke (userspace Postgres 16.2, committed registry): a recovery-style document
(url `…/fmd-2024-final.pdf`, stored snapshot, no plan) came out `linked`,
`plan_reason='plan_v1'`, `extraction_family='placement_report'`, entity linked,
with a `plan.candidates: 1, planned: 1, linked: 1` row in `pipeline_runs`.

## Family coverage from document shapes (2026-11-19, W5 family-coverage)

### The problem, measured

The measured crawl stood at 135 snapshots with **83 `abstained /
family_undetected`** — predominantly annual reports, prospectuses and NIRF
submissions, i.e. documents that *do* contain accreditation, placement and fee
evidence. Only 47 documents were `linked` (4 more linked-but-unextracted): the
pipeline was starved by classification, not by crawling. The planner's text
vocabulary could not score these documents' shapes, so the richest corpus was
discarded before extraction.

### The signal channels (`fmd/plan/rules.py`)

Per family, all channels contribute their weight independently and every fired
channel is recorded in the decision's `signals` (weights in
`SignalWeights`, thresholds in `config/plan.yaml`):

| Channel | Evidence | Weight |
|---|---|---|
| `url` / `anchor` / `extension` | Phase A as before | 0.35 / 0.45 / 0.20 |
| `text` | first-page token matches (T0 sample, bounded) | 0.30 |
| `heading` | heading-shaped lines ("Annual Report 2023-24", "5. PLACEMENTS") | 0.35 |
| `title` | the cover line that names a family (scan ≤ 5 headings) | 0.35 |
| `recurring` | a token occurring ≥ `phase_b.recurring_min` (3) times | 0.20 |
| `structure` | filename-shape hint (`fmd/plan/roles.py`), planner-only tier | 0.30 |

**Cover identity is decisive.** The title line — the first heading-shaped cover
line that names a family — decides the document outright
(`signals['title_decisive']=1`): an annual report that *says* "Annual Report"
is accreditation even when its body carries fee tables. A prospectus is the one
cover that does not name a single family, so it resolves through its own shape
chain instead: prospectus identity in the cover → fee-table rows
(≥ `phase_b.table_min_rows`) → `fee_structure`, else intake-table rows →
`seat_matrix`, else `curriculum`. The chain records
`prospectus_fee_table_rows` / `prospectus_intake_table_rows`; the pseudo-family
never escapes `detect_family`.

### The intended classification (pinned by `tests/test_plan_family_coverage.py`)

| Document type | Family |
|---|---|
| Annual report | `accreditation`, + `signals['secondary_hint']` (`placement_report`) when placement sections fire |
| Prospectus / admission brochure | `fee_structure` (fee tables) → `seat_matrix` (intake tables) → `curriculum` |
| NIRF submission / AQAR / SSR / IQAC report | `accreditation` |
| Fee circular / fee structure | `fee_structure` |
| Seat matrix / intake statement | `seat_matrix` |

### Precision guards (each with a fixture and a test)

These must never classify; they stay `family_undetected`:

staff circulars, tender notices, event/news pages, convocation notices,
examination schedules (a *registration* notification still classifies
`exam_notification`), and the known administrative case
"Fee for recognition … Research Centre" — whose fee veto now cancels **every**
text-derived channel (`text`, `heading`, `recurring`, `title`), not just body
tokens. Administrative filename shapes (tender, convocation, admit card, date
sheet, timetable, office order, news/event, press release) veto all structure
hints. Ambiguity is a refusal, never a coin flip: an ambiguous top-2 within
`margin` (0.10) returns `ambiguous_family`, and a filename hinting two families
(`prospectus-fee-structure-2025.pdf`) records
`structure_hints_ambiguous` and abstains.

### The structure-hint tier (`fmd/plan/roles.py`)

`family_hints_from_structure()` maps the file name's shape
(`Annual-Report-2024.pdf` → accreditation, `Prospectus-2025.pdf` → curriculum,
`Fee-Circular-2025.pdf` → fee_structure, `Seat-Matrix-2025.xlsx` → none —
structured payloads are inventory by default) to family hints. Structure only:
no text is read, no institution is named. The planner records the hints on
every decision (`signals['structure_hints']`) and uses them only when registry,
text and URL token evidence are all silent: at Phase A a single hint converts a
below-floor abstention into `defer_for_text` (one bounded T0 read is worth it);
at Phase B a single hint is the last fallback (`signals['fallback']='structure_hint'`).
Precedence is unchanged: **registry-declared family > Phase B text evidence >
Phase A URL evidence > structure hints**, and a declared family is never
reconsidered by any of the new signals.

### Planner version 2 — replanning the abstained set

`config/plan.yaml → version` is now `"2"`. The 83 rows were stamped
`family_undetected|plan_v1`, so one pass re-decides them:

```bash
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd
cd packages/pipeline
uv run fmd plan --dry-run   # plan.candidates is the remaining debt
uv run fmd plan             # abstentions/stale stamps are replanned; human decisions untouched
uv run fmd plan --force     # if any stale crawl-era rows remain
```

`--force` replans planner-shaped reasons at older stamps only — rows stamped at
the current version and human-review prose are skipped (`plan.skipped_current`).
Expected shape of the run after replanning: `plan.reason.family_undetected`
falls, `plan.family.accreditation` and the prospectus-resolved
`fee_structure`/`seat_matrix`/`curriculum` counts rise, and the negative corpus
(staff/tender/news/convocation/schedule) stays in `plan.reason.family_undetected`
by design — those documents are not evidence.

### Tests (all offline; DB tests need a live `DATABASE_URL`)

`tests/test_plan.py` (channels, veto, determinism corpus-wide, ambiguity),
`tests/test_plan_family_coverage.py` (the matrix above at planner level,
precedence, hint tiers), `tests/test_plan_phases.py` (Phase A deferral →
Phase B decision per representative shape), `tests/test_plan_roles.py`
(filename-shape hints and their vetoes). Golden eval unchanged: provenance 1.0.

## `hard_pdf` abstention semantics (2026-11-18)

The first pass recorded `unresolved_provider` ×3. That code means "the
environment named a provider that is not declared under `providers:`" — a
misconfiguration, and the wrong label for an optional tier. Verified against
the current registry (offline, `ModelRouter.call("hard_pdf", …)`):

| env | reason |
|---|---|
| no `FMD_API_LLM_URL` / `KEY` / `MODEL` | `vision_unavailable` |
| URL+key set, `FMD_API_MODEL` unset | `vision_unavailable` |
| model+URL+key set, unknown `FMD_HARD_PDF_PROVIDER` | `unresolved_provider` |

Fix: **point `hard_pdf` at an explicitly optional profile**, which the profile
already is (`required: false`, `degrade_reason: vision_unavailable`); the
observed rows predate the profile-blocker path. No routing change was needed.
If a deployment has no hosted vision model, leave `FMD_API_MODEL` unset and the
tier degrades per document with `vision_unavailable`; text PDFs still extract.
A non-empty `FMD_HARD_PDF_PROVIDER` must name a provider declared under
`providers:` (`api`, `gemini`, …); a typo is the only way to see
`unresolved_provider` again.

## Review console (2026-11-18, WS-G)

The review console is the correction path for every machine decision: a wrong
auto-accept, a bad entity merge suggestion, an ambiguous document plan. It is
the last 501 surface shipped — the endpoints below replace the
`not_implemented` stubs in `fmd/api.py`, and `/review` in the web app renders
the queues and submits decisions.

### Endpoints and auth

All review endpoints are admin-token gated (no user accounts, brief §2). The
token comes from `FMD_ADMIN_TOKEN` and is supplied per request as
`X-Admin-Token: <token>` or `Authorization: Bearer <token>`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/review/queue?kind=claim\|alias\|document&limit&offset` | One paginated adjudication queue |
| POST | `/api/review/claims/{id}` | `{action: accept\|reject\|correct, corrected_quote?, note?}` |
| POST | `/api/review/aliases/{id}` | `{action: confirm\|reject, note?}` |
| POST | `/api/review/documents/{id}` | `{action: link_entity\|mark_not_applicable, entity_id?, note?}` |
| GET | `/api/review/audit?kind=...&subject_id=...` | Decision trail for one queue item |

Gate semantics, and why they differ from `/api/scoring/rerun`: a missing or
wrong token answers **403 `forbidden`** on the review surface — there is no
login flow to re-attempt, so a failed token is a final permission verdict. The
scoring-rerun gate keeps its historical 401. With `FMD_ADMIN_TOKEN` unset the
review surface answers 503 `admin_disabled` (feature off, not forbidden). The
gate resolves **before** the database dependency, so an unauthorised caller
never makes the API probe Postgres; a valid token with no database answers 503
`database_unavailable`.

### Queue semantics

* **claims** — everything with `verification_status <> 'supported'` and
  `superseded_by IS NULL`: machine-rejected rows stay visible (a verifier can
  be wrong; the console is the only correction path), still-unverified rows
  queue, and anything a human accepted leaves the queue. Ordered **confidence
  ASC** (lowest first), then id; backed by the partial index
  `claims_review_confidence_idx` (migration 0008). Each item carries the
  provenance chain (`url`, `fetched_at`, `sha256`, `snapshot_id`,
  `fetcher_tier`) and the page/bbox recorded in `qualifier_json` by the
  extractor.
* **aliases** — `entity_aliases.review_status='pending'` (migration 0008
  state machine `pending → confirmed | rejected`), oldest first, with the
  entity's canonical name joined for context.
* **documents** — `plan_state='needs_review'` (the planner's abstention),
  least-confident plan first, then most recently seen.

### Claim actions

* **accept** → `verification_status='supported'`, `verified_by='human'`.
  Repeating an already-applied action returns `already_done: true` and writes
  **no** second audit row (idempotent double-submit).
* **reject** → `verification_status='rejected'`, `verified_by='human'`.
  Rejecting a claim that is currently `supported` **requires `note`** (422
  `missing_note` otherwise): overturning the verifier silently is exactly the
  failure mode the console exists to prevent. The verifier's original notes
  are preserved under `verification_notes.previous_verification`.
* **correct** → requires `corrected_quote` **and** `note`. The corrected quote
  must occur verbatim in the extraction's text (422
  `unresolvable_corrected_quote` otherwise — invariant 3 applies to
  replacements too). `ClaimsRepo.correct_claim` performs one transaction:
  INSERT the replacement claim (same predicate/value/entity, corrected quote +
  resolved span, confidence 1.0, born `supported` with `verified_by='human'`)
  and point the old row's `superseded_by` at it. The original claim's content
  columns are **never** mutated. Acting on a superseded claim is 409
  `already_superseded`.

### Alias actions and the ER lock

Confirming an alias sets `review_status='confirmed'`,
`confirmed_by='human'`, `confirmed_at=now()`. For a `manual` alias this is
what arms entity resolution's manual authority (`fmd/er/types.py`):
a mention that exactly matches the confirmed alias resolves to that entity
unattended (`decided_by='manual'`, note `manual_alias_authoritative`), and any
score- or model-based auto-accept **onto** that entity is declined to
adjudication (`manual_canonical_lock`) — the pair stays with a human.
`test_review_api.py::test_alias_confirm_blocks_auto_merge` proves this chain
end-to-end: queue → confirm via the API → candidate universe built exactly the
way the planner builds it (`AliasesRepo.list_for_entity`) → `resolve_with_audit`.

Rejecting an alias sets `review_status='rejected'`; the ER-facing read
(`AliasesRepo.list_for_entity`) filters rejected rows, so a refused suggestion
stops boosting similarity immediately. A decision is terminal: flipping it is
409 `decision_conflict`, raised before any audit row is written. Producers
(`er/aliases.py` mining, inventory ingest) still create `pending` rows; nothing
writes `confirmed` except this console.

### Document actions

`link_entity` (requires `entity_id` of an existing entity) sets
`plan_state='linked'`, `entity_id`, `plan_confidence=1.0` and a
`plan_reason` naming the human decision; the planner's own fields
(`extraction_family`, role) survive. `mark_not_applicable` takes the document
out of the plan. Both are idempotent on their own target state and 409 on a
flip (`linked → not_applicable` or vice versa).

### Audit trail

Every applied action appends one `review_audit` row (migration 0008) with
`kind`, `subject_id`, `action`, `previous_state` (the review-relevant fields
before the action), `details_json` (note / corrected quote / target entity),
`decided_by='human'` (CHECK-enforced) and `created_at`. The table has no
foreign keys on `subject_id` by design: the trail survives subject removal and
one table serves three subject kinds. There is no UPDATE and no DELETE — a
mistaken action is corrected by a further action. The web console renders the
trail under each item after every action.

### Web console

`/review` (apps/web) is an admin surface: it asks for the token once and keeps
it in React state only — never localStorage, never a cookie, sent only as
`X-Admin-Token` to the API. It renders the three queues with predicate, quote,
span, page/bbox, source URL, fetch tier ("evidence tier"), verification status
and the deciding claim; accept/reject/correct with the reject-a-supported-note
warning; confirm/reject on aliases (with the ER-lock warning on `manual`);
link/not-applicable on documents. A 403 renders as a permission failure, never
a retry-with-credentials prompt. `components/review-console.test.tsx` renders
the queue offline against fixture data — no API required.

### Run this (copy-paste)

```bash
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd
export FMD_ADMIN_TOKEN=change-me
cd packages/pipeline && uv run pytest -q tests/test_review_api.py   # 22 contract tests
make migrate                                                        # applies 0008_review_actions
curl -s -H "X-Admin-Token: $FMD_ADMIN_TOKEN" \
  "http://localhost:8000/api/review/queue?kind=claim&limit=5" | python3 -m json.tool | head -40
curl -s -X POST -H "X-Admin-Token: $FMD_ADMIN_TOKEN" -H 'content-type: application/json' \
  -d '{"action":"accept"}' http://localhost:8000/api/review/claims/5 | python3 -m json.tool
```

Without `DATABASE_URL` the DB-backed review tests skip loudly (same contract
as `test_db.py`); the gate tests (403/503) run everywhere.

### Known limitations

* The console has no user accounts and no per-person identity: the audit log
  records `decided_by='human'` for every action, not who. A deployment that
  needs named reviewers must add an identity layer above the shared token
  (out of v1 scope, brief §2).
* The corrected-quote check locates the quote in the stored extraction text;
  it cannot detect that the *correction itself* is true. Review remains a
  human judgement — the console only guarantees the span stays resolvable.

## Seat predicate split and verifier descriptions (2026-11-18, W2 seat-semantics)

**Two measured problems, one real run** (`gemma-4-26b-a4b-gguf`, verify@v2, the
Tamil Nadu AHS UG seat matrix):

1. **Semantic mismatch.** The Stanley claim was stored as `seats.total = 20`
   with the quote `3. B.Sc. Medical Laboratory Technology\n20`, row context
   (`context_row` → the Stanley block, column "Intake"), `expected_type=count` —
   and the verifier still returned `wrong_field` after repair. It was arguably
   right: `"total"` reads as an aggregate, while the quote is one programme's
   row in one college. A verifier asked to confirm "this is the total" will
   reject a row value. The predicate name was the bug, not the verdict.
2. **Ambiguous predicates.** The verifier received the predicate NAME and its
   TYPE but not its DEFINITION, so any predicate whose plain-English reading
   differs from the intended reading gets mis-judged — this would recur beyond
   seats.

### What changed

**The seat vocabulary is split** (`config/predicates.yaml`, version
`2026-11-18.3`):

| Predicate | Scope | Meaning |
|---|---|---|
| `seats.intake` | per-row | sanctioned intake for THIS programme at THIS institution — one row of a seat matrix under its intake/sanctioned column (the Stanley case) |
| `seats.total` | aggregate | an aggregate the document itself states as a total (a printed "Total" row, an institution-wide figure); never a single programme row's value |
| `seats.admitted` | per-row | the filled/admitted count for THIS programme row; unchanged, never intake |
| `seats.duration_months` | per-row | programme duration printed on the matrix row; unchanged |

Both scored predicates map to `admission_realism` with the same `seats`
transform as before — no weight value changed, evidence tiers unchanged. The
extraction schema (`SeatMatrix`) renamed `total_seats` → `intake`
(`seats.intake`) and added `aggregate_seats` (`seats.total`), and
`seat_matrix@v5` documents the split with a worked Total-row example (v1–v4
kept and still replayable). `validate.py`'s guards bound quota/admitted counts
by `seats.intake` first, falling back to a `seats.total` aggregate as the
upper bound when no intake row was extracted — the guards did not get weaker.

**Alias fix.** `seats.sanctioned` (and the bare legacy `seats` spelling) now
alias `seats.intake`, not `seats.total`: both spellings meant "the sanctioned
seat count for one programme", which is a row-scoped value.

**Every registry row carries a one-sentence `description`** stating exactly
what the predicate means and its scope (per-row vs aggregate, per-year vs
total). 96 rows, all non-empty; the config loader and `PredicateRegistry`
reject a missing or blank description at load, so a new predicate cannot ship
undefined. The descriptions are the authority the verifier uses.

**verify@v3 passes the definition.** The field payload is now
`{key, field, quote, expected, description?, context?, row_label?,
column_header?}`; `description` is resolved from the registry (aliases resolve
to their canonical row's description; unknown predicates arrive `null`). The
v3 prompt adds a "judge against the description, not the name" contract: when
the description says per-programme and the quote is one table row, the
row-scoped quote is CORRECT; when the description says aggregate and the quote
wraps a single row's value, the scope itself is wrong and that stays
`wrong_field`. `verify_repair@v3` mirrors it. `config/models.yaml` pins
`prompt_version: v3`; v1/v2 are kept and still load.

**Fixtures.** The committed replay fixtures were re-recorded under the v3
prompt hashes (same verdict semantics as the v2 era). `row_context_v2`'s s01
and s04 are now `seats.intake` (the real Stanley predicate after the split);
s02 stays the wrong-row counterfactual. The new `seat_semantics_v3` case is
the discrimination this wave exists for, offline: the same quote "3. B.Sc.
Medical Laboratory Technology\n20" is `supported` as `seats.intake` (t01) and
`wrong_field` after repair as `seats.total` (t02); the document's own "Total
195" row verifies as `seats.total` (t03); a second row-scoped intake
(Chengalpattu) is supported (t04); money-without-period is still `absent` with
no repair (t05). The wrong-row rule from verify@v2 is untouched.

**Golden row renamed.** The Stanley entry's expected predicate is now
`seats.intake` — same value (20), same quote, same context/row block, same
committed page-1 fixture, provenance still 1.0. No loader migration was
needed: golden `expected` keys are the predicate strings themselves, and the
loader re-checks the quote/context exactly as before (the rename is covered by
`test_the_real_seat_matrix_row_is_verified_against_one_named_row`).

**Programme materialisation** reads `seats.intake` (then its
`seats.sanctioned` alias spelling) for the sanctioned-seats field, with every
consumed claim id in `programmes.field_sources`. `seats.total` is deliberately
not consumed — an aggregate across programmes/institutions must never be
written into one programme's `seats` column — and `seats.admitted` is never
treated as intake. A document that states only an aggregate leaves
`programmes.seats` NULL: `UNKNOWN` is a value, and a wrong attribution is
worse than none.

**Scoring** is unchanged: `seats.intake` feeds `admission_realism` through the
same `seats` transform `seats.total` used; having both vocabulary rows just
means an entity can be scored from either (or both, averaged as components)
without any weight edit.

### Runbook table with definitions

The predicate table above now carries the split rows; the authoritative
per-predicate definitions (one sentence each, scope included) live in
`config/predicates.yaml:predicates[<name>].description` — read those for any
predicate before judging a claim against it, in code or by hand.

### Verification results

```bash
cd packages/pipeline && uv sync --all-extras
uv run ruff check .                                          # clean
uv run pytest -q                                             # 978 passed (no DB)
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q   # 1039 passed
uv run python -m fmd.cli eval --golden ../../data/golden     # provenance 1.0, gate PASS
```

`test_predicates.py` pins the registry side (descriptions non-empty, seat
aliases, dangling references rejected); `test_verify.py` pins the v3 payload
and the seat-semantics discrimination; `test_extract_schemas.py` pins v5 and
the intake/aggregate spec pair.

## 8. Exam registration windows — exam_notification@v3 (2026-11-18)

Programme application windows were tracked; the registration windows of the
EXAMS an applicant must sit were not, and an unknown window was
indistinguishable from an open one. Spec `17-exam-registration-requirements.md`
makes them a v1 hard-filter requirement. This section is the operator-facing
summary of what now runs; the acceptance tests pin it.

### Extraction (`exam_notification@v3`)

`fmd/extract/prompts/exam_notification/v3.md` replaces v2 as the latest
version (v1/v2 kept and still replayable). The schema
(`fmd/extract/schemas/exam_notification.py`) adds three nullable fields —
`scope` (`exam.scope`), `state` (`exam.state`) and `sitting_dates`
(`exam.sitting_dates`) — and keeps every v1/v2 field, so old payloads still
validate. New predicates carry registry rows in `config/predicates.yaml` with
one-sentence descriptions, like every other row.

The prompt contract adds the two rules the product lives or dies by:

1. **Registration close is not a sitting date.** `registration_closes` is the
   last date to apply/register; `sitting_dates` is when the exam is held. A
   document that prints only a sitting date leaves `registration_closes` null.
   The filter uses registration close only; sitting dates are informational.
2. **Printed precision is preserved.** A month-only print ("March 2027") is
   quoted verbatim — never completed into a day. A printed time of day is
   quoted together with its date ("up to 5:00 pm on 12/03/2027").

### Parsing (`fmd/extract/validate.py`, date branch only)

`_date_derived_value` is the deterministic parser behind `unit_kind: date`:

- **Full date** → `precision: day`, value `YYYY-MM-DD`, printed token stored in
  the claim qualifier as `date_text`.
- **Full date + printed time** → value is the UTC instant
  (`2027-03-12T11:30:00+00:00` for "5:00 pm on 12/03/2027"), qualifier carries
  `precision: day`, `date_text` and `tz: Asia/Kolkata`. IST has no DST, so a
  same-day clock time never shifts the printed day.
- **Month-only** → value `YYYY-MM`, `precision: month`, `date_text` verbatim.
  No day is fabricated, anywhere. `Session 2026-27` and other non-month words
  are rejected, not guessed.
- Full-date patterns always run before the month pattern, so `12 March 2027`
  never degrades to month precision.

### Filter (`fmd/filters/impl.py: exam_registration_open`)

Unchanged invariants: binary cutoffs, `unknown` is a value, the TTL rule
(`weights.yaml filter_policy.stale_exam_window_ttl_days`) still turns a stale
window into `unknown` — render "verify", never "open". On top of that:

- **Supersession.** A claim whose `superseded_by` is set (or that another live
  claim retires via a `supersedes` qualifier) never decides. A reopened or
  extended window flips the verdict back to `pass` on the next run; both rows
  are kept (append-only).
- **Cycles coexist.** Cycle-tagged windows (`qualifier.cycle_year`) are decided
  per profile target intake year; a 2026 late-round window says nothing about a
  2027 intake. No target year → the latest observed tagged window decides and
  its cycle is reported in `detail.exams[].cycle_year`.
- **Month precision.** Pass before the month starts, `unknown` inside it
  (not stale), fail after it ends. Day-level questions inside the month are
  unanswerable from a month print.
- **Day boundary.** A date-only close is honoured through the whole IST day
  (closes 12/03/2027 is still open at 10:00Z on the 12th, closed at 18:31Z).
  A naive printed time is read as IST; an aware value (UTC from the validator)
  is compared as an instant.
- **Exclusion code.** A fail carries `detail.exclusion_reason =
  exam_registration_closed` beside the deciding claim id and `observed_at`,
  which the shortlist and exclusions surface.

### API and web

`POST /api/exams` keeps its flat `items` contract and adds: `scope`,
`cycle_year` (the cycle that decides for the supplied profile), `closes`,
`closes_text` (printed text), `precision`, `sitting_dates`, per-exam `cycles`
(each with opens/closes/sittings/`observed_at`/staleness/its own status and
`source_claim_id`), `programme_ids` (programmes whose extracted
`exam_required` claims need this exam), and a top-level `next_deadlines`
sorted ascending by the act-before instant (month precision sorts by the
month's first IST instant — the earliest the window may close). A
month-precision window never publishes a fabricated day: `closes` and
`deadline` are null and `deadline_month` carries `2027-03`.

`apps/web` (`/exams`) renders the cycles column, a "next actionable deadlines"
panel for the profile, and a month-precision chip; unverified still never
renders as "open". `apps/web/lib/types.ts` gained additive exam types only.

### Tests (all offline unless a live DB is supplied)

```bash
cd packages/pipeline && uv run ruff check .
uv run pytest tests/test_extract_exam_dates.py tests/test_extract_exam_notification.py
uv run pytest tests/test_filters.py tests/test_exams_api.py
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden   # provenance 1.0
```

`tests/fixtures/exam/` carries two notification texts (exact dates with a
printed deadline time; month-precision with "to be notified" sitting). The
month-precision, supersession, cycle and no-day-shift behaviours are pinned in
`test_filters.py`; the /api/exams contract in `test_exams_api.py`.

## 9. Context chunking — long documents (2026-11-18, W4)

### The problem, measured

Annual reports are the documents that carry accreditation, fee and placement
evidence, and they are exactly the ones that could not be extracted: the whole
canonical text went into one `user_message` and the server answered 400 with

```
request (195492 tokens) exceeds the available context size (65536)
request (75572 tokens)  exceeds the available context size (65536)
request (71621 tokens)  exceeds the available context size (65536)
```

which the provider layer classified as `http_error` — a network shape. A
sizing fault looked like an outage, and a 200-page report could never produce a
claim.

### Sizing rule

`config/models.yaml:chunking` drives everything; there are no constants in
`fmd/extract/chunking.py`:

```
usable_tokens = context_tokens - output_tokens - prompt_headroom_tokens
window_chars  = usable_tokens * chars_per_token          # body a call may carry
overlap_chars = overlap_tokens  * chars_per_token        # default ~1k tokens ~ 1 page
```

`context_tokens` prefers the `chunking` block, falls back to the routed
profile's own `context_tokens` (`bulk_extract`: 65536, the served `-c`),
`output_tokens` to the profile's `max_tokens` (4096). A document whose canonical
text fits `window_chars` takes the single-pass path exactly as before; a longer
one is split by `fmd/extract/chunking.plan_windows`:

* windows are **verbatim slices** of the canonical text, never rewrites;
* cuts snap back to page boundaries (`\n\n`), then line boundaries, then the
  raw budget cut — a table row is never split from its label;
* every window after the first starts `overlap_chars` before the previous cut,
  so a quote straddling a cut is whole in at least one of the two windows;
* the plan is deterministic, which is what resume depends on.

At the production numbers: `65536 - 4096 - 2048 = 59392` usable tokens →
**237,568 chars per window** (≈15 pages), overlap 4,096 chars. A 200-page
annual report (~470k chars) becomes 3 calls.

### Spans stay absolute

The model quotes from the window; `validate_document` resolves the quote
against the window text exactly as in the single-pass path, and
`chunking.to_absolute_offsets` then shifts `char_start`/`char_end` (plus the
qualifier's `context_char_*`/`row_char_*` spans) by the window's absolute
start. Every stored span is **global** in the original canonical text, so
bbox/page projection (`fmd/acquire/canonical.py`) and the verifier's source
windows work unchanged. One extraction row per run carries the full
`extractions.text`, per-window outcomes and the chunk summary
(`structure.chunking`).

### Dedupe across overlapping windows

The same fact recovered from two windows becomes ONE claim
(`chunking.dedupe_claims`):

* primary key `(predicate, char_start, char_end)` — same span, same window
  overlap;
* secondary key `(predicate, normalised value, row context)` — same value
  (NFKC, grouping commas folded, `120` == `1,200` as a value shape is NOT
  assumed: values are compared normalised) attributed to the same table row
  (`table_index, row_index`), row label, or the model's `context` quote.

Claims with no row context only merge on the identical span. Overlap duplicates
neither inflate claim counts nor leak past a resume: the run's dedupe keys are
stored per window (`claims_key`) and a retried window's claims are filtered
against everything a prior run already stored.

### Failure isolation, resume, cap

| Code | Meaning | Where |
|---|---|---|
| `context_exceeded` | input cannot fit the served context: pre-flight (chunking disabled), or the provider 400 text matched an overflow signature | document outcome (abstention) |
| `chunk_failed:<n>` | window *n* (1-based) failed; underlying provider reason is recorded beside it (`reason: timeout`, `reason: context_exceeded`, …) | per-window row in `structure.chunking.windows`, and aggregated on the run outcome (`reason: chunk_failed:<failed-count>`) |
| `chunk_budget_exhausted` | the per-document window cap stopped the plan with text uncovered; `pages_covered` vs `pages_total` reported | run outcome when `chunking.truncated` |

* One window failing never stops the others: the rest still run and the
  document records partial success with per-window outcomes.
* A rerun (`--force`, or the next run that re-selects the snapshot) **resumes**:
  windows the prior run marked `ok`/`completed_prior_run` are skipped (matched
  by absolute span, never by index) and only failed windows are re-called;
  claims already stored are excluded from the new pass by their dedupe keys.
* `max_windows_per_document` (24) bounds one run's cost: a 400-page report
  (~27 windows) stops at the cap and reports `chunk_budget_exhausted` with the
  covered page count. Raise the cap in config and rerun to extract the tail
  without re-extracting the windows already done.
* When **every** window attempted in a pass fails, nothing is persisted and the
  document is recorded as an abstention `chunk_failed:<n>` — never an empty
  success.

`http_error` can no longer mask a sizing fault: the pipeline pre-flight raises
`ContextExceeded` (reason `context_exceeded`) before wasting a call, and a
provider 400 whose text matches `chunking.OVERFLOW_SIGNATURES` (llama.cpp's
"exceeds the available context size", vLLM/OpenAI/Anthropic/Gemini phrasings)
is reclassified the same way.

### Known limits

* Cross-field guards (`placed ≤ eligible`, rate-vs-counts) apply **within a
  window**; fields split across windows are still verified per claim by the
  verifier, but a guard that needs two siblings in one call can no longer see
  them. Values stay honest; the guard coverage is narrower.
* A money field whose `period_source` sibling is printed outside the window is
  rejected (`missing_period`) rather than defaulted — the sibling must be
  visible for the claim to exist. Raise `overlap_tokens` or let the document's
  period lines repeat per page.
* Scanned pages needing OCR are unchanged: chunking operates on canonical text
  the tiers already produced; an image-only page that T2 cannot decode still
  routes to T3 before any window is planned.

### Tests (all offline unless a live DB is supplied)

```bash
cd packages/pipeline && uv run ruff check .
uv run pytest tests/test_extract_chunking.py tests/test_extract_pipeline.py
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden   # provenance 1.0
```

`tests/fixtures/extract/chunk/annual_report.txt` is a 60-page synthetic report
with evidence markers on fixed pages (12, 20, 34, 45, 52). The window-sizing
arithmetic, boundary cuts, cap, dedupe keys, resume and the honest failure
codes are pinned in `test_extract_chunking.py` (21 cases) and the pipeline
behaviour — split, absolute offsets, overlap dedupe, one-window failure,
all-fail persistence, cap, single-pass call count 1, `context_exceeded` never
`http_error` — in `test_extract_pipeline.py`.

## 10. Span resolution scope — the W7 ladder (2026-11-19, W7 span-scope)

### The problem, measured

A 17-document extraction run produced 5 claims and 14 rejections, every one
`ambiguous_span`. Each rejected quote was correct in substance; it failed
because the resolver demanded **document-wide uniqueness**. In a 100-page
annual report the same words recur by design — a running header on every page,
"A" as a grade, "2024" as a year, a programme name once per section — so a
correct value was discarded for repeating. That is a resolver design flaw, not
a document defect.

### The ladder (narrowest qualifying scope first)

`resolve_quote` / `validate_document` (in `fmd/extract/validate.py`) now take
`pages` (the document's page spans) and `sections` (printed heading blocks
from `fmd/acquire/canonical.py`). When a quote is ambiguous document-wide and
the model supplied **no** row context, uniqueness is re-attempted on
progressively narrower instances, and the rung that resolves is recorded on
the claim qualifier as `resolution_scope`:

| Rung | Qualifies when | Recorded as |
|---|---|---|
| a | exactly one occurrence document-wide | `document` |
| b | exactly one occurrence inside the claim's row context (table block, else the ±`CONTEXT_WINDOW_CHARS` window) | `row` |
| c | exactly one **page** holds exactly one occurrence | `page` + `scope_page_no` |
| d | no page qualifies, but exactly one **section** holds exactly one occurrence | `section` + `section_heading` |
| e | nothing qualifies | `ambiguous_span` → UNKNOWN |

Hard rules that make the ladder safe:

* **Never first-match.** A scope instance qualifies only when it holds exactly
  ONE occurrence of the quote. Two qualifying instances of the same width
  cancel out — two pages or two sections with one occurrence each leave the
  claim UNKNOWN. There are no positional tie-breaks and no "closest to top".
* **Table rows still win.** The row rung (b) runs before page/section, so a
  row-scoped quote resolves at `row` even when a page-level unique match
  exists elsewhere.
* **A failed context is never widened past.** The ladder widens only from
  *absence* of context. When the model supplied a context and it did not
  resolve (a table block that does not contain the quote, or a window with no
  adjacent occurrence), the claim stays UNKNOWN — widening past a stated
  location would attribute an occurrence the model did not point at.
* **Provenance stays absolute.** `char_start`/`char_end` remain offsets into
  the canonical document text whatever rung resolved the span; scope affects
  uniqueness only.
* **Sections are conservative.** `fmd/acquire/canonical.looks_like_heading`
  accepts an ALL-CAPS line of ≥2 words, a colon-terminated label without
  sentence punctuation, or an annexure-family lead word. Title-case prose,
  single all-caps cells ("TOTAL", "INR") and numbered row labels are refused.
  Occurrences outside every heading block are invisible to the section rung —
  attributing them would be a guess. A missed heading costs yield, never
  correctness; the section rung is the last rung before UNKNOWN.

The page number is recorded for every resolved span when page geometry is
supplied (`scope_page_no`), and the qualifying instance's span is recorded as
`scope_char_start`/`scope_char_end` for page- and section-scoped claims.
`fmd/verify` reads these keys: `ClaimView` carries `resolution_scope`,
`scope_page` and `scope_span`; the field payload sent to the verifier includes
`resolution_scope` and `page`, and the verifier window widens to the scope
instance when it fits the window budget. Verdict semantics are unchanged — a
page-scoped quote is judged as a resolved location, not a suspicious bare
quote. The verify prompt is unchanged (verify@v3): the scope travels as data
in the payload, so no recorded replay fixture is invalidated.

### Chunked windows

The single-pass path passes the document's page map and sections as-is. The
chunked path passes **window-relative** bounds: page spans are clipped to the
window and shifted by its start; a section is kept only when its printed
heading lies fully inside the window (a heading the model never saw cannot
scope a quote) and is clipped at the window end otherwise. The shifted
`scope_char_start`/`scope_char_end` move with the claim when
`to_absolute_offsets` re-anchors spans.

### Dedupe rule (chunking + resume)

The dedupe identity (`fmd/extract/chunking.claim_dedupe_keys`) is:

* primary key: `(predicate, char_start, char_end)` — the same span recovered
  from two overlapping windows;
* location key: `(predicate, normalised value, location)` where location is
  the table row identity (or row label, or the model's context quote) —
  **or, new in W7**, the section identity `(section, page, heading)` when the
  claim carries `resolution_scope: section`.

So: the same predicate/value within the same **row or section** is one claim
(an overlapping window re-quoted the same fact), while the same predicate/value
on **different pages is not one claim** — a per-page repetition (a running
header, a fee line printed per page) is genuine evidence, and the page number
is part of the key for that reason.

### Known limits

* If the model supplies a context that fails, the ladder does not rescue the
  claim; precision is worth more than the yield of a guess.
* A quote whose occurrences all sit on pages holding ≥2 occurrences, with no
  unique heading block, stays UNKNOWN. Rescuing it would need model-supplied
  page anchors, which are unreliable and out of scope.
* Sections come from a text heuristic on tier blocks. Scanned documents whose
  OCR merges headings into body blocks derive fewer or no sections; the page
  rung is unaffected (page geometry is exact).
* Extract prompts are unchanged (v4/v5 stay current): the model supplies
  nothing new — pages and sections are document-derived, so no
  `prompt_version` bump, and already-extracted snapshots remain valid.

### Tests (all offline unless a live DB is supplied)

```bash
cd packages/pipeline && uv run ruff check .
uv run pytest tests/test_extract_validate.py tests/test_extract_context.py
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden   # provenance 1.0
```

`tests/fixtures/extract/annual_report_scope.txt` is a 10-page synthetic
annual report with a running header, three repeated audit phrases, one
ALL-CAPS heading pair and evidence on pages 2/5/6/9/10; the real-shape
regression uses the existing 60-page
`tests/fixtures/extract/chunk/annual_report.txt` (the running header repeats
60×, the audit sentence 180×, "96" on pages 20 and 34). The ladder rungs, the
failed-context hard stop, the row-beats-page precedence, absolute offsets,
section derivation, window-relative bounds, the dedupe rule and the verifier
plumbing are pinned in `test_extract_validate.py` and
`test_extract_context.py`.

## 11. Document section tree — the W8 outline (2026-11-19, W8 section-structure)

### The problem, measured

W7 made the span resolver scope-aware, but the "sections" it saw were a flat
text heuristic (`looks_like_heading`: ALL-CAPS lines, colon labels, annexure
lead words) with no levels, no hierarchy and no structure for anything
downstream. The operator proposal this wave implements: a parsed PDF has a
natural outline, and keeping it gives real section boundaries for scope, a
breadcrumb ("4. Fees › 4.2 Hostel") for the model, section-aligned chunk
boundaries, and stable structure for the planner and the review UI.

### Extraction (Docling first, T0/T2 heuristic fallback)

`fmd/acquire/outline.py` builds one tree per document, inside
`build_canonical` (same blocks, same offsets, zero extra passes):

* **Docling labels preferred.** T1 records the item label on every block
  (`section_header`, `title`); `page_header`/`footer`/`footnote` are page
  furniture and never enter the tree. When at least one usable label exists,
  the tree is Docling-only and the text heuristics do not run — one extractor
  per tree, recorded on every node and on the extraction structure
  (`outline.extractor`: `docling` | `heuristic`).
* **Heuristic fallback** (T0/T2, or T1 with no usable labels): numbered
  headings (`1.`, `4.2`, `(iii)`, `c.`, Devanagari `१.`, `१.२`), short
  ALL-CAPS lines, colon-terminated labels and annexure lead words — the W7
  signals plus numbering and Devanagari. Every heuristic carries a
  false-positive guard, each pinned by a test: a numbered *sentence*
  ("1. The institute was established …") fails the sentence-connector veto,
  the terminal-punctuation veto and the length caps; a numbered *table row*
  ("1. Tuition fee 85000") fails the amount veto (3-digit numbers, currency
  tokens); "B.Sc. MLT" fails the interior-dot veto; one lone numbered line is
  refused outright (chapters come in sequences; `outline.numbered_min_sequence`,
  default 2). A Devanagari sentence (danda) is refused; a single Devanagari
  word is refused (a cell, not a title). All thresholds live in
  `config/acquire.yaml:outline`; none in Python.
* **Levels are normalised 1..N with no gaps.** Numbered headings open
  ancestors on a stack keyed on numbering depth (1 → "4.", 2 → "4.2"); an
  unnumbered heading is a sibling of the previous heading; the stack
  guarantees `level <= previous + 1`, so "1 → 1.1.1" normalises to
  "1 → 2". Ranges are classic outline ranges: heading start → next heading at
  the same or shallower level, so children nest inside parents and siblings
  never overlap. Each node carries `node_key` (stable ordinal), `level`,
  `title`, `char_start`/`char_end`, `page_no`, `bbox`, `source`, `parent_key`.

### Persistence and lookup

`db/migrations/0009_document_sections.sql` adds `document_sections`
(`document_id`, `parent_id`, `node_key`, `level`, `title`, char range,
heading char range, `page_no`, `bbox_json`, `extractor`,
`extractor_version`), idempotent (`CREATE TABLE IF NOT EXISTS`), forward-only,
no edits to 0001–0008. Sections are **derived structure, never evidence**:
`SectionsRepo.replace_for_document` writes the tree wholesale in one
transaction, so a re-extraction cannot leave a stale tree. A no-headings
document writes nothing (it never clears a tree a prior snapshot produced).

The lookup API is O(log n)-ish, never a linear scan per claim:
`fmd.acquire.outline.SectionIndex.locate(start, end)` bisects the sorted
heading starts (the deepest containing node is the last one starting at or
before the span) and walks the precomputed parent chain — O(depth) — to return
the narrowest containing node plus its root→leaf breadcrumb. In SQL,
`SectionsRepo.breadcrumb(document_id, start, end)` is one indexed query
(`document_id, char_start, level DESC` btree) with a recursive parent walk.

### Resolver: real section scope

`validate_document`/`resolve_quote` take the tree bounds when the document has
an outline (`outline_to_sections`: every node with its level and breadcrumb);
the flat W7 heuristic bounds remain the fallback and are recorded as
`section_source: heuristic` on section-scoped claims (outline bounds record
`outline`). The ladder is unchanged — document → row → page → section →
UNKNOWN — and the W7 hard rules hold: never first-match, a section qualifies
only with exactly ONE occurrence, two qualifying sections cancel out. What
changes is attribution: each occurrence is attributed to its **narrowest**
containing node, so nested sections never double-count one occurrence, a
chapter's range never swallows its subsections, and the qualifier records the
breadcrumb (`section_path: ["4. Fees", "4.2 Hostel"]`) beside the leaf
heading (`section_heading`). Occurrences outside every section (before the
first heading, in a no-headings document) stay invisible to the rung — a
missed heading costs yield, never correctness. Provenance stays absolute:
`char_start`/`char_end` are canonical offsets whatever rung resolved.

### Prompts: breadcrumbs as metadata, never text

Each window's user message carries its outline after the delimited document
text (`DOCUMENT SECTIONS (metadata only — NOT document text; never quote
it)`): the single pass lists the full outline with page numbers; a chunked
window gets `window breadcrumb: 4. Fees > 4.2 Hostel` plus the headings fully
inside it. Validation resolves quotes against the window body only, so a
heading string that exists only in the metadata cannot resolve as a claim
quote (pinned by test). The system prompts are unchanged (v4/v5 stay current):
the outline travels as data, so no `prompt_version` bump and recorded replays
stay valid.

### Chunking: cuts prefer section boundaries

`plan_windows` takes `section_starts`; a cut whose look-back zone holds a
heading start lands exactly on it, so a window opens where a section opens and
its metadata carries one coherent breadcrumb. A section larger than the budget
splits inside it (page/line fallback) and every window covering it reports the
same breadcrumb. Short documents stay single-pass; the dedupe identity is
unchanged (the W7 section key `(predicate, value, page, heading)` already
identifies the fact; the breadcrumb adds no identity the heading does not
carry).

### Planner

`score_families`/`detect_family` accept `outline_headings`: the tree titles
join the heading channel and a tree cover line that names a family decides the
cover (`cover_title`). With no tree (or no family-naming title) the behaviour
is byte-identical to W5; precedence is unchanged (registry > text > filename
hints).

### Known limits

* One extractor per tree: when Docling emits labels but misses some headings,
  the heuristic does not fill the gaps (mixing sources risks inconsistent
  levels). A T1 document whose labels are unreliable yields a shallower tree;
  yield cost, not correctness.
* A "4.2" whose "4." chapter is never printed normalises to level 1 (no-gap
  rule): attribution stays exact, breadcrumbs are shallower than the visual
  outline.
* Numbered list rows without amounts or sentence words ("1. Medical
  Laboratory Technology") can still become sections when they form a
  sequence; ranges stay correct, so the cost is cosmetic breadcrumbs, not
  mis-attribution.
* `document_sections` is keyed by document; two snapshots of the same URL
  share it and the latest non-empty tree wins.

### Tests (all offline unless a live DB is supplied)

```bash
cd packages/pipeline && uv run ruff check .
uv run pytest tests/test_acquire_outline.py tests/test_extract_validate.py tests/test_extract_sections_metadata.py tests/test_extract_chunking.py tests/test_plan_family_coverage.py
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden   # provenance 1.0
```

`tests/test_acquire_outline.py` pins the detection guards, level
normalisation, Docling-label preference (page furniture excluded), the
mid-subsection breadcrumb lookup and the no-headings diagnostic;
`test_extract_validate.py` pins the section rung on tree bounds
(`section_path`, `section_source`, two-section cancellation, narrowest
containment); `test_extract_sections_metadata.py` pins the metadata block and
the unquotable-heading rule; `test_extract_chunking.py` pins heading-aligned
cuts; `test_plan_family_coverage.py` pins the planner seam; `test_db.py` pins
the 0009 round trip, breadcrumb SQL and wholesale replacement.

## 12. On-demand programme materialisation — `fmd programmes` (2026-11-20, W8)

### The problem, measured

Materialisation used to run only inside the crawl/ingest path
(`ProgrammeStage`), so claims verified *after* a crawl — or re-extracted after
a prompt/scope fix — never became programme rows. The corpus held 61 verified
claims (28 accreditation, 14 seats, 7 fee, 7 curriculum, 5 placement) while
`programmes` had 2 rows, both `partial` at best: the tool could not rank
anything despite holding seat data.

### What changed

`fmd/orchestrate/ingest.py::ProgrammeStage.run` no longer owns the write loop;
it delegates to `materialise_stage_run()` in the new
`fmd/ingest/materialise.py`. The same module exposes the on-demand entry
points, so the crawl path and the CLI share one implementation and cannot
drift:

* `materialise_programmes(programmes, entities, mapping, *, dry_run, force)` —
  the reusable core; works against any repo exposing
  `materialisation_claims()` / `upsert()` (real repo or test fake);
* `materialise_from_database(database, registry, *, dry_run, force)` — one
  call against a live database, no crawl, no model;
* `evidence_completeness(claims, mapping, predicates)` — the list view's
  completeness metric.

New CLI (no crawl, no extraction, no model):

```bash
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd
cd packages/pipeline
uv run python -m fmd.cli programmes materialise            # write changed rows only
uv run python -m fmd.cli programmes materialise --dry-run  # print the plan, write nothing
uv run python -m fmd.cli programmes materialise --force    # rewrite even unchanged rows
uv run python -m fmd.cli programmes list                   # seats + claim id, completeness, review reasons
uv run python -m fmd.cli programmes inspect <id>           # every field, its claim id, quote, page
```

### Semantics (unchanged contract, now explicit)

* Input is still only `claims` with `verification_status='supported'`,
  `superseded_by IS NULL`, `method <> 'modelled'` from a
  `documents.role='evidence'` document whose entity is an institution
  (`materialisation_claims()` filters in SQL; the pure mapper re-checks).
* One row per `(institution, ER-normalised programme name)`; every set
  attribute records its claim id in `field_sources` (keyed by predicate);
  missing attributes stay `NULL`; nothing is invented and nothing is derived
  (semesters are still never converted to months).
* **Idempotent and resumable**: a row whose draft matches the stored row
  (values *and* per-field claim ids) is not rewritten at all — the rerun
  reports `unchanged` and issues no `UPDATE`. A row whose proving claim
  changed is rewritten and the plan reports the change per field
  (`seats (claim 2 -> 9)`). `--force` rewrites unchanged rows too (the
  crawl-path stage keeps its historical always-rewrite semantics via
  `materialise_stage_run`, so a vanished claim cannot linger there).
* `--dry-run` builds the full plan — creates, per-field changes, refused
  aggregates, rejections — and writes nothing.

### The aggregate-vs-row rule (`seats.total` guard)

`config/predicates.yaml` splits the seat vocabulary: `seats.intake` is the
per-programme row value; `seats.total` is an aggregate the document itself
states (a printed Total row or institution-wide figure). Materialisation now
reports the refusal explicitly instead of skipping silently: every eligible
`seats.total` claim is refused with reason
`seats_total_aggregate_not_programme_intake` (claim id, institution, value in
the plan/JSON and on the stage summary's `programme_aggregates_refused`), and
the pure layer returns the same refusal in its rejection list. The claim
itself stays in `claims` — append-only evidence is never lost — and the
scorer may still read the aggregate at institution scope. A document that
states only an aggregate leaves `seats` NULL (`UNKNOWN` is a value).

### Evidence completeness (the `list` view)

`fmd programmes list` prints two completeness numbers per row, both computed,
never hardcoded:

* `fields k/6` — how many of the six materialisation fields (programme name,
  credential level, sanctioned intake, aggregate, duration months, internship
  months) have at least one eligible claim behind them. `credential_level` is
  classified from the name claim, so it is backed exactly when the name is;
  the aggregate counts as *evidence present* even though it is never stored
  on a row.
* `components k/N` — how many scoring components (the sub-scores of
  `config/weights.yaml`, mapped through `config/predicates.yaml`, N=7 today)
  are fed by at least one eligible claim of the programme's institution.

Note: the W8 brief says "6 scoring components"; the registry actually defines
seven (`outcomes`, `affordability_roi`, `clinical_training`,
`academic_quality`, `student_life_safety`, `admission_realism`,
`financing_feasibility`). The view reports backed/total from the registry, so
the denominator follows the config rather than a hardcoded 6.

### Inspect (the reviewer's audit path)

`fmd programmes inspect <id>` prints every row field with the claim id that
proved it (`credential_level` cites the name claim with a *derived from* note;
`seats_total_aggregate` prints the never-stored rule), then each
`field_sources` entry joined to claim + provenance: quote, extraction
`page_range`, URL, fetch timestamp, sha256 and char span
(`ProgrammesRepo.evidence_for_programme`, one query). Chain:
`programme row → field → claim → extraction → snapshot → document URL`.

### Schema

No migration was needed. `normalised_name`, `field_sources`, `needs_review`,
`review_reasons` and `source_family` (migration 0006) already carry
everything; refused aggregates stay in `claims` and the run output, not in a
new column. (The W8 brief allowed "migration 0009" — 0009 was already taken
by `0009_document_sections.sql` (W8 section-structure), and no column proved
necessary.)

### Measured on the local scratch DB (12-claim seed, 2 institutions)

* `materialise`: 2 created; `seats.total=195` refused with reason;
  `B.Sc. MLT` programme carries `seats=20` (claim id cited).
* Rerun: `wrote=0 (create=0 update=0 unchanged=2)`, no duplicate rows.
* `--dry-run --force`: plan reports 2 would-be rewrites, writes nothing.
* `list`: `fields 4/6 | components 1/7 (admission_realism)` for the
  seat-matrix-only institution; `fields 3/6 | components 3/7
  (academic_quality, clinical_training, outcomes)` for the
  curriculum+accreditation+fee+placement institution.
* `inspect 1`: every field with claim id, quote `B.Sc. MLT | 20`, `page=1-2`,
  URL + sha256.

### Known limits

* Aggregate seat totals are refused, not stored: an institution-level total
  column can be added later (a new migration) if a use case demands it; the
  evidence remains queryable in `claims` meanwhile.
* Completeness reads only claims already filtered by
  `materialisation_claims()`; a claim attached to a programme-named
  *sub-entity* (no such rows exist yet) would not count.
* `inspect` shows the extraction's `page_range` (window precision); claim
  level → printed page mapping stays coarse until per-span page attribution
  lands (same limitation as the review console).

### Tests (all offline unless a live DB is supplied)

```bash
cd packages/pipeline && uv run ruff check .
uv run pytest tests/test_programmes_materialise.py tests/test_ingest_programmes.py
DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd uv run pytest -q
uv run python -m fmd.cli eval --golden ../../data/golden   # provenance 1.0
```

`tests/test_programmes_materialise.py` pins the reusable function offline
(fake repos): creation with field sources, skip-unchanged idempotency, forced
rewrites, per-field change reports (old claim id → new), NULL-when-no-claim,
dry-run plans, the aggregate guard (fixture
`tests/fixtures/programmes/seat_matrix_row_plus_total.json`:
`seats.intake=20` row lands, `seats.total=195` refused), alias-confirmed
grouping (B.Sc. MLT ≡ BMLT → one row) vs unconfirmed MLT/MLS (two rows,
flagged), and the completeness metric;
`tests/test_programmes_db.py` pins the same against Postgres (rerun, dry run,
superseded-claim update, `get_by_key`, `evidence_for_programme`);
`tests/test_programmes_cli.py` pins the CLI surfaces end to end.


## 13. Three-tier programme scoping — `config/programmes.yaml` (2026-11-20, W9)

### The problem, measured

After materialisation the operator DB held 7 programme rows, 3 of which are not
this product: an `MBBS` row (Adesh University), a `B.Sc (Hons) Agriculture`
row (Rani Lakshmi Bai Central Agricultural University), and a duplicate
`"3. B.Sc. Medical Laboratory Technology"` row (TN Dr. M.G.R. Medical
University) that existed only because a seat-matrix row number became part of
the name. Two defects: no target-field scoping, and no name normalisation.

### What changed

* **`config/programmes.yaml` (the tenth registry file)** — country-keyed
  taxonomy: `countries.IN` carries the include/exclude regexes and the alias
  table; AU/NZ/US/CA/GB/IE/PH/AE/SA/DE ship as declared **empty stubs** (no
  invented patterns); `default` is the empty fallback and any name decided
  under it is flagged `country_taxonomy_missing`. `definition` is the
  conceptual target-field statement (diagnostic laboratory science: clinical
  biochemistry, haematology, microbiology, histopathology, blood bank,
  immunology) — the authority the semantic tier judges against; its version is
  part of the semantic cache key. Precedence: **manual > exclude > include >
  semantic > uncertain**.
* **Normalisation (`fmd/programmes/normalise.py`)** — leading row numbering
  (`"3. "`, `"3) "`, `"(3) "`, `"#3 "`, `"Sl. 3. "`) and bullets are stripped
  before ER normalisation, in `normalise_programme_name` itself, so the
  grouping key and every rule match see the same text. A bare leading number
  without punctuation is deliberately *not* furniture (`"3-Year B.Sc."`
  survives). `"3. B.Sc. MLT"` and `"B.Sc. MLT"` now produce one row.
* **Tier 1 (`fmd/programmes/taxonomy.py`)** — fast deterministic regex rules;
  exclude beats include. MBBS/BDS/BAMS/BHMS/BUMS/BSMS/nursing/GNM/ANM/BPT/
  physiotherapy/pharm(acy)/agricultur(e/al) are excluded for IN; B.Sc./BMLT/
  BMLS/MLS/MLT/DMLT/medical-or-clinical-laboratory names are included.
  Undecided names are *not* guessed — they fall through.
* **Tier 2 (`fmd/programmes/classifier.py`)** — semantic equivalence through
  the router purpose `programme_equivalence` (models.yaml profile, default
  local Gemma, optional: degrades to `uncertain`). Prompt is versioned
  (`fmd/programmes/prompts/equivalence/v1.md`, prompt_version `v1`); responses
  are guided JSON `yes|no|uncertain` + confidence + rationale + evidence.
  Conservative by construction: no evidence and no field signal ⇒ `uncertain`
  without a call; a `yes` citing nothing is downgraded to `uncertain`; any
  provider failure degrades to `uncertain` → `needs_review`, never ranked.
  Batched (bundle_size from config), cached by sha256 over (normalised name,
  institution type, definition version) — one call per new name, not per row —
  persisted in `programme_scope_decisions` with model, prompt version and cache
  flag per decision.
* **Tier 3 (`fmd/programmes/service.py`)** — `fmd programmes decide <id>
  --equivalent|--not-equivalent --note` writes a `manual` decision
  (`decided_by='human'`) under the same cache key; Tier 1 consumes it before
  rules and the classifier from the next run, so the system accumulates
  knowledge instead of re-asking. `fmd programmes requalify [--dry-run]`
  re-scopes every stored row under the current taxonomy + decisions: folds the
  row-numbering duplicates (`merge_into` re-points `cost_items`, deletes only
  the derived duplicate row — claims are never touched), rekeys stale
  pre-W9 grouping keys, marks MBBS/Agriculture out, applies manual verdicts,
  and is idempotent (unchanged rows are not rewritten).
* **Persistence (migration `0010_programme_scoping.sql`)** — `programmes`
  gains `in_scope`, `scope_reason`, `scope_tier` (rule|semantic|manual),
  `scope_evidence`, `scope_model`, `scope_prompt_version`, `scope_cached`,
  `scope_decided_at`; `programme_scope_decisions` is the durable cache +
  learning-loop store (unique per cache key, never deleted).
* **Scoring** — `fmd score` skips `in_scope=false` rows before any scoring
  work and reports the skipped count (`out_of_scope` in the outcome and the
  run summary); `fmd/score` additionally refuses an out-of-scope programme
  directly (`OutOfScopeProgramme`), so no caller can rank an MBBS. Out-of-scope
  rows stay in the database for audit (`fmd programmes list --all`,
  `inspect` shows tier + evidence).

### Operating it

```bash
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd
cd packages/pipeline
uv run python -m fmd.cli programmes requalify --dry-run   # the plan, nothing written
uv run python -m fmd.cli programmes requalify             # merge + rescope + manual verdicts
uv run python -m fmd.cli programmes decide 8 --not-equivalent --note "MBBS is not MLT"
uv run python -m fmd.cli programmes list --all            # audit view incl. out-of-scope rows
uv run python -m fmd.cli programmes inspect <id>          # tier that decided + the evidence
```

Expected on the operator DB (7 rows before): the `"3. B.Sc."` duplicate folds
into row 1, MBBS and Agriculture flip to `in_scope=false` with
`rule_exclude:*` reasons, and `fmd score` reports `out_of_scope=2` (or `3`
before the merge) while the ranked set holds only laboratory-technology rows.
Names no rule decides and no evidence backs stay `uncertain` →
`needs_review` — that is the designed conservative outcome, not a failure.

`--dry-run` is offline by contract: it never constructs the model router, so
it succeeds with `FMD_LOCAL_LLM_URL` pointing at a dead port. Rules and manual
decisions still decide as usual; names that would reach Tier 2 are reported as
`would_classify` (and counted in `would_defer`, since no decision is written)
alongside `would_admit`, `would_reject` and `would_defer`. Every real semantic
decision persists its provenance — the configured profile's model id
(`scope_model`, the reproducible string; the provider-resolved id is not
exposed by the router yet — known gap) alongside the prompt
version (`v1`) and definition version in the cache key, and `scope_cached`
(false for a fresh call, true for a cache hit or manual replay).
