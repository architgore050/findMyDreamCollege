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

So **65536 is the provisional default**: it clears the ≥1 GiB headroom rule at
every overhead between 0.45 and 0.83 GiB, and 96k is the likely true ceiling.
Both are projections until the sweep below runs — the sweep is the deliverable,
this table is the expectation it has to confirm or overturn.

**To measure the ceiling (one command, replaces the guess):**

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

### E. MTP / speculative decoding — supported, **off**

llama.cpp has first-class Gemma 4 MTP: `--spec-type draft-mtp`,
`--model-draft <mtp file>`, arch `gemma4-assistant`, `--spec-draft-n-max N`
(merged PR #23398 on 2026-06-07; build b12468 is 2026-11-17, and
`src/models/gemma4-assistant.cpp` is present in the tree). The drafter shares the
target's KV cache, so it does not need a second context, and the upstream README
claims 159 → 257 tok/s at 0.72 acceptance on a Q4_K_M target on a B200.

**It is off by default anyway**, for two reasons that are ours and not upstream's:

1. **Correctness is unverified on this pair.** Speculative decoding is supposed
   to be output-identical under greedy sampling, but that guarantee is worth
   exactly as much as the equality test. We have not run it, so we do not claim it.
2. **VRAM is the scarce resource, not speed.** The drafter is +461 MB resident on
   a card where 64k of context already costs headroom. A 26B-A4B at ~4B active
   is not decode-bound the way a dense model is.

Enable only after both of these pass on the loaded pair:

```bash
# 1. parity: identical greedy bytes with and without the drafter, fixed prompt
# 2. speed: >20% decode tok/s on a 512-token generation, twice, same prompt
FMD_SPEC=1 scripts/serve_local.sh start   # only then
```

If either fails, the verdict is "off" and stays off. Do not enable it for a
throughput number alone.

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

## 3. OCR stage (T2 Qwen3-VL / T2b PaddleOCR-VL)
### 3.1 Why this engine

Qwen3-VL-8B-Instruct is the T2 engine because it is the only local option with a
published real-scan Devanagari number (**chrF++ 75.2**, arXiv:2606.29213, doc 10
§4) *and* because vLLM, not Paddle, has kernels for this GPU:

| Fact | Value | How measured (2026-11-16) |
|---|---|---|
| `sm_120` in `torch.cuda.get_arch_list()` | yes | one-liner in §2.1 on `~/.venvs/fmd-vllm` |
| vLLM supports the Qwen3-VL architecture | yes — `Qwen3VLForConditionalGeneration` | `ModelRegistry.get_supported_archs()` |
| PaddlePaddle GPU wheels | stop at `sm_90` | wheel metadata; runbook §4 row |

Licences: Qwen3-VL **Apache-2.0**, PaddleOCR-VL **Apache-2.0**. Surya/Marker stay
rejected (OpenRAIL-M, $5 M revenue cap).

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
#    (acquire = CPU Docling; never start a second GPU model here)
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
