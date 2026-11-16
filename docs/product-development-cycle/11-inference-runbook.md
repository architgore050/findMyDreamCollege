# 11 — Inference runbook (local text model + OCR stage)

**Status:** engine facts measured on this host 2026-11-16; end-to-end numbers
depend on a weight download that was still in flight when this was written, and
are flagged where unverified. **Owner:** WS-SERVE.
**Companions:** `10-inference-and-model-selection.md` (why these models),
`config/models.yaml` (what the router calls), `config/acquire.yaml` (the OCR
stage), `09-local-runbook.md` (the rest of the stack).

Two processes, deliberately separate:

| Process | What it is | Port | Who calls it |
|---|---|---|---|
| `scripts/serve_local.sh` | vLLM, OpenAI-compatible `/v1` | 8000 | `fmd.llm.router` (`bulk_extract`, `verify`, `entity_adjudication`) |
| `scripts/serve_ocr.sh` | PaddleOCR-VL behind `POST /extract` | 8110 | `fmd.acquire.tiers` tier T2 |

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
# let vLLM fetch it, or pre-fetch to a local dir when the link is flaky:
mkdir -p ~/models/Qwen3.5-9B-AWQ && cd ~/models/Qwen3.5-9B-AWQ
for i in 1 2 3 4 5; do
  curl -sSL -C - --retry 50 --retry-all-errors \
    -o model-0000$i-of-00005.safetensors \
    "https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ/resolve/main/model-0000$i-of-00005.safetensors" &
done; wait
# plus config.json tokenizer.json tokenizer_config.json preprocessor_config.json
# merges.txt chat_template.jinja generation_config.json model.safetensors.index.json
```

Shards are 12.4 GB total (the repo keeps vision/linear-attention blocks
unquantized). Then point the server at the directory:

```bash
FMD_VLLM_MODEL=~/models/Qwen3.5-9B-AWQ scripts/serve_local.sh start
```

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

## 3. OCR stage (T2 / T2b)

### 3.1 Bring-up

```bash
scripts/serve_ocr.sh build     # docker compose --profile ocr build paddleocr
scripts/serve_ocr.sh start     # up -d, waits up to 10 min for /health
scripts/serve_ocr.sh status
scripts/serve_ocr.sh logs
scripts/serve_ocr.sh stop      # stop + rm (weights stay in the paddlecache volume)
```

Compose service `paddleocr` (profile `ocr`, so `make up` never starts it):
`infra/Dockerfile.ocr` (CUDA 12.9 runtime base) + `infra/ocr/server.py` +
`infra/ocr/requirements.txt` (paddlepaddle-gpu 2.6.2, paddleocr 3.7.0,
pypdfium2, fastapi). First start downloads ~2 GB of weights into
`paddlecache:/root/.paddlex`.

### 3.2 Contract (must match `fmd/acquire/tiers.py::_ocr_doc`)

`POST $FMD_T2_URL/extract` with `{"pdf_base64": "..."}` (or `"image_base64"`
for a single page) →

```json
{"tool":"paddleocr-vl","tool_version":"3.7.0","model":"PaddleOCR-VL-1.6",
 "pages":[{"page":1,"width":1654,"height":2339,"bbox":[0,0,1654,2339]}],
 "blocks":[{"page":1,"block_order":0,"block_bbox":[88,120,1560,210],
            "block_content":"..."}]}
```

Boxes are page-image pixels at `FMD_OCR_DPI` (default 200); `pages[].width/height`
is the same image size, so a block box always resolves inside its page box.

### 3.3 When it cannot run

Unset `FMD_T2_URL` (the compose default inside the cluster is
`http://paddleocr:8110`). Then T2 raises `TierUnavailable`:

* image-only page → "T0 also produced no text, so this document cannot be
  extracted on this host" → **no claims**, document queued for review;
* page with usable T0 text → T0 fallback with a `T2 unavailable: ...` note.

Neither path invents text. **The Paddle/Blackwell question was not settled on
this host**: the image build did not complete within the session (multi-GB image
pull on a ~1 MB/s link). PaddlePaddle's GPU wheels top out at sm_90, so
`no kernel image is available for execution on the device` is the expected
failure and must be recorded as such — not worked around by pretending OCR ran.

## 4. Common failures

| Symptom | Meaning | Action |
|---|---|---|
| `no kernel image is available for execution on the device` | wheel built for ≤ sm_90 (Paddle, or an old torch) | text model: use vLLM ≥ 0.19 (resolves cu130 torch, sm_120 present). OCR: no Paddle kernel for Blackwell; keep T2 unset and queue pages |
| `CUDA out of memory` at startup | `--gpu-memory-utilization 0.90` + desktop compositor on 16 GB | drop to `0.80`, then lower `FMD_MAX_MODEL_LEN` to 16384; record the change, 32k is the cost model's assumption |
| `xgrammar` grammar/architecture error | constrained decoding does not cover the Qwen3.5 hybrid architecture | set `providers.local.supports_guided_json: false`; the router then degrades to JSON-instruct + schema validation. Report the degraded schema-valid rate separately — it is a different candidate |
| repetition loop in raw `curl` output | decoding, not acquisition | confirm `enable_thinking: false` in `--default-chat-template-kwargs`, temperature 0, no repetition penalty. The pipeline's own guard (`fmd/acquire/guards.py`) routes loops to T3 |
| `404 model not found` | `FMD_EXTRACT_MODEL` ≠ `--served-model-name` | align both |
| `T2 was selected but is unavailable` | OCR service down or `FMD_T2_URL` unset | expected degradation; do not fabricate OCR text |

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

Do not copy the doc-10 §6 cost model's "8 pages/min" into a report until these
three rows are filled; they are the same unverified assumption.

## 6. What to run next (in order)

1. `scripts/serve_local.sh install` (if the venv does not exist) and the §2.1
   arch-list check.
2. Finish the weights (`§2.2`), `scripts/serve_local.sh start`.
3. `curl /v1/models` + the 32-token chat completion + the guided-JSON probe.
4. `cd packages/pipeline && .venv/bin/python -m fmd.cli llm doctor --live`.
5. `make migrate` (DB up), then `make extract ARGS="--limit 5"`.
6. `python -m fmd.cli verify --limit 50` (second pass over the stored claims),
   then re-read `claims` for verified/rejected/unverified.
7. Only then start the OCR service (§3.1) and re-run the scanned documents.
