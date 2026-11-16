# 11 — Inference runbook (local text model + OCR stage)

**Status:** engine facts measured on this host 2026-11-16; end-to-end numbers
depend on a weight download that was still in flight when this was written, and
are flagged where unverified. **Owner:** WS-SERVE.
**Companions:** `10-inference-and-model-selection.md` (why these models),
`config/models.yaml` (what the router calls), `config/acquire.yaml` (the OCR
stage), `09-local-runbook.md` (the rest of the stack).

Three processes, deliberately separate — and never two GPU-resident models at
once (§1 budget rule):

| Process | What it is | Port | Who calls it |
|---|---|---|---|
| `scripts/serve_local.sh` | vLLM, OpenAI-compatible `/v1`, text model | 8000 | `fmd.llm.router` (`bulk_extract`, `verify`, `entity_adjudication`) |
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
| vLLM text (:8000) | host process | `$MODEL_DIR/Qwen3.5-9B-AWQ` | no |
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
* `scripts/fetch_model.sh` is the only downloader, run by a human on the host.

**Container → host connectivity.** Linux Docker has no `host.docker.internal`;
every client service declares
`extra_hosts: ["host.docker.internal:host-gateway"]` (`api`, `worker`, `ocr`,
`ocr-paddle`). The URL has one name per consumer: `FMD_LOCAL_LLM_URL` for the
router (compose default `http://host.docker.internal:8000/v1`, host default
`http://localhost:8000/v1`) and `FMD_OCR_VLLM_URL` for the adapter. Verify from
inside the container, not from the host shell:

```bash
docker compose -f infra/docker-compose.yml exec api \
  python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:8000/v1/models',timeout=5).status)"
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

```bash
# one command, resumable, into the host weight root (never into an image):
scripts/fetch_model.sh QuantTrio/Qwen3.5-9B-AWQ
ls -l "$FMD_MODEL_DIR/Qwen3.5-9B-AWQ"   # expect 5 safetensors + configs
```

`fetch_model.sh` compares each file against the Hub-reported size and resumes
short ones with `curl -C -`; a complete directory is left alone. Do **not** point
vLLM at the Hub id for day-to-day serving: a Hub id can silently fetch, a local
directory cannot.

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
  --host 0.0.0.0 --port "${FMD_VLLM_PORT:-8000}" \
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
curl -s localhost:8000/v1/models | python3 -m json.tool | head
# expect: {"object":"list","data":[{"id":"Qwen3.5-9B-AWQ", ...}]}

curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen3.5-9B-AWQ","messages":[{"role":"user","content":"Say OK."}],"max_tokens":32}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
# expect: "OK"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Then the router-level check inside the pipeline:

```bash
cd packages/pipeline
FMD_EXTRACT_PROVIDER=local FMD_LOCAL_LLM_URL=http://localhost:8000/v1 \
  .venv/bin/python -m fmd.cli llm doctor --live
# expect: bulk_extract usable=true (live), no blockers; keys are never printed
```

Guided-JSON check (this is unverified claim #3 in doc 10):

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
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
FMD_LOCAL_LLM_URL=http://host.docker.internal:8000/v1
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
| `/health` says `model_not_served` | the vision port serves something else (usually the 9B text model) | start `scripts/serve_ocr.sh vision-start`; the vLLM on 8000 must not be reused for OCR |
| `T2 was selected but is unavailable` | OCR service down, `FMD_T2_URL` unset, or weights absent | expected degradation: `/health` names the reason (`vision_unavailable`, `model_not_served`, …); do not fabricate OCR text |
| `/extract` → 503 `empty_result` | a non-blank page produced zero blocks | the page is unreadable at these settings; route it to the human queue, do not lower the bar to "empty string is fine" |

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

```bash
# 0) once per boot: host text model (weights already on host; no download)
scripts/serve_local.sh start
curl -s localhost:8000/v1/models                      # expect id Qwen3.5-9B-AWQ

# 1) stack (api waits for db); rebuilds never touch weights
docker compose -f infra/docker-compose.yml up -d db api worker
curl -s localhost:8000/api/health                     # expect status ok

# 2) router health from INSIDE the container (proves host-gateway works)
docker compose -f infra/docker-compose.yml exec api \
  python -m fmd.cli llm doctor --live

# 3) one bounded real extraction + verification
docker compose -f infra/docker-compose.yml exec api \
  python -m fmd.cli extract --limit 3
docker compose -f infra/docker-compose.yml exec api \
  python -m fmd.cli verify --limit 50

# 4) score (versioned; needs the eval diff per AGENTS.md)
docker compose -f infra/docker-compose.yml exec api python -m fmd.cli score

# 5) stop (weights and DB survive; nothing is re-downloaded)
docker compose -f infra/docker-compose.yml stop worker api
scripts/serve_local.sh stop
```

Long steps (extraction over many documents, a first crawl) are the human's to
run: hand them the command, do not add a wait loop. `scripts/serve_ocr.sh
vision-start` is only needed for scanned pages and needs the optional 10.6 GB
vision fetch first (§3.2); without it T2 degrades to the human queue and that is
the honest state.
