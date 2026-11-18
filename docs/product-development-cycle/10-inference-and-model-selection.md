# 10 — Inference and model selection (provisional)

**Status:** provisional decision, pending the 20-document bake-off.
**Owner:** WS-MODELDOC. **Evidence cut-off:** 2026-11-16.
**Consumers:** `config/models.yaml`, `.env.example`, `scripts/bakeoff.py`, the extraction and adjudication workstreams.

> This document records the inference research, not a finished procurement. Every default below is a *hypothesis to be tested on our documents*. No provider price here is a quote; no benchmark here was run on our corpus. Numbers that came from a vendor card, a page-parsing benchmark, or a third-party price aggregator are marked explicitly.
>
> *Provenance:* assembled by WS-MODELDOC from the research brief and public sources re-checked on 2026-11-16. Where a figure could not be confirmed against a first-party page, it carries a **[U]** or **[V]** marker rather than being presented as fact.

---

## 0. Decision update — 2026-11-17 (locked: Gemma 4 26B-A4B / UD-IQ4_XS / llama.cpp)

**This section supersedes §1 for the bulk-extraction role.** The operator locked
the model on 2026-11-17; it is not open for re-litigation by the bake-off. The
rest of this document remains the evidence trail for *why the engine and the
allocation shape changed*, and for the OCR/adjudication roles it still governs.

| Item | Decision | Note |
|---|---|---|
| Weights | `unsloth/gemma-4-26B-A4B-it-GGUF`, **raw (non-QAT)** | Deliberate: raw GGUF beat QAT variants on retained capability in the operator's own runs. This is operator experience, not a published benchmark — recorded as **[U-use]**, an accepted risk, not a measurement. |
| Quant | **`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`** — 13,597,177,568 B (12.66 GiB) | Largest quant that fits 16 GB with room for a 64k KV cache. |
| Engine | **llama.cpp `llama-server`** (GGUF), not vLLM | vLLM has no GGUF path; the chat template, the constrained decoder (`response_format: json_schema` → GBNF) and the MoE kernels all come from llama.cpp. |
| KV cache | **q8_0/q8_0 with flash attention** (`-fa on -ctk q8_0 -ctv q8_0`) | Halves KV bytes vs f16 at the cost of a per-block scale; parity checked at 8k (runbook §2.7). |
| Alias | `gemma-4-26b-a4b-gguf` | Served id, pinned in `config/models.yaml` and `.env`. |
| Context | **65536** provisional default; budget table + sweep in runbook §2.7 | Must stay ≤ the server's `-c`. The sweep is the authority; 32768 is the safe fallback. |
| MTP / speculative decoding | **Supported but OFF by default** | llama.cpp implements it (`--spec-type draft-mtp`, arch `gemma4-assistant`, merged PR #23398 on 2026-06-07; the pinned build b12468 is from 2026-11-17). Enabled only when greedy-output parity and a real speed win are both demonstrated. |

Three consequences that are easy to get wrong:

1. **Architecture, not parameter count, is what made this fit.** 26B total but
   ~4B active per token (128 experts, 8 used). The weights are 12.66 GiB — the
   size class an 8-bit KV cache can still live beside on a 16 GB card. A *dense*
   26B at this quant would not leave room for a useful context.
2. **Sliding-window layout keeps long context cheap.** 25 of 30 layers are
   sliding-window (window 1024, 8 KV heads, head dim 256); only 5 are full
   attention (2 KV heads, head dim 512). With upstream's SWA trimming
   (`--swa-full` false, the default) the 25 SWA layers' KV is *constant* in
   context length, so the marginal cost of context is paid by 5 layers only.
   The measured/derived table is in runbook §2.7.
3. **The GGUF repo root holds ~65 GB of quants.** Fetching it naively pulls
   everything, so the downloader gained an `HF_INCLUDE` glob filter
   (`scripts/hf_download.py`). Default empty = old behaviour.

**[U] Open item, not a conclusion:** the raw-vs-QAT advantage is operator
experience. If the bake-off later shows the QAT variant within noise on
span-exact, the 6.7 GiB saved is worth revisiting — for now the locked decision
stands and the fallback is a *smaller context or the 12B pilot*, never a silent
quant swap.

---

## 1. Decision in one table

| Role | Provisional default | Why (document/extraction evidence) | Evidence class | Confidence |
|---|---|---|---|---|
| Bulk text extraction | **Qwen3.5-9B-AWQ**, int4, 32k context, thinking off | Best capability-per-GB candidate that fits 16 GB and supports guided JSON; 262k native context leaves room for long fee/seat documents. **No field-extraction benchmark exists for it yet** — this is why the bake-off exists. | vendor card + chat benchmarks, not extraction | Low until bake-off |
| OCR / scanned pages | **Qwen3-VL-8B-Instruct** (FP8) | On real printed Devanagari scans it reaches chrF++ 75.2, the best open model tested and ahead of GPT-5.5 (58.5); it fits a 16 GB card. It is *not* the overall leader — Gemini 2.5 Flash is 86.3 — so it is the local/offline default, not the accuracy ceiling. | OCR benchmark, word/short-phrase scans | Medium |
| Adjudication (conflicting claims) | **Gemini-class mid-tier**, example `gemini-2.5-flash` | The second pass only sees conflicts and low-confidence claims; a hosted mid-tier buys cross-family disagreement at low volume. Must beat local 9B on our bake-off to keep the default. | price + absence of field evidence | Low until bake-off |
| NLP rubric (qualitative evidence) | **Gemini-class mid-tier**, example `gemini-2.5-flash` | Rubric sentences need long-context reading and quote-grounded output; same cost envelope as adjudication. | inference, not benchmarked | Low until bake-off |
| Entity / pairwise adjudication | **Qwen3.5-9B-AWQ**, local | Pairwise name matching is the cheapest task in the set; keeping it local avoids egress and per-call cost. | inference | Medium |
| Embeddings | **BAAI/bge-m3** (1024-d) | Unchanged from the current registry. Multilingual, 1024 dimensions, already integrated; no new evidence justified a switch. | existing implementation | Medium |

**Not chosen for bulk extraction:** GPT-5.5, Claude Opus 4.7, Gemini 2.5 Flash. They lead the Devanagari OCR table, but that is an OCR result, not a field-extraction result, and their per-page cost is ~5–20× the local candidate. They remain bake-off candidates for adjudication, not bulk extraction.

**Not chosen for OCR:** the specialised OCR-VLMs (DeepSeek-OCR, Unlimited-OCR). On the same real-scan benchmark they collapse to chrF++ 10.4 and 24.7, with catastrophic rates of 89% and 46.7%. A parsing benchmark winner is not an Indic-scan winner.

---

## 2. How to read the evidence

Not all numbers in this document are equal. Every claim is tagged with one of these classes:

| Tag | Meaning | May decide a default? |
|---|---|---|
| **F** | Field/extraction evidence: measured on spans/claims, ideally our schema | Yes |
| **P** | Page/region parsing: OCRBench, DocVQA, OmniDocBench, CER/chrF on transcriptions | Only as a screen; not a field-extraction result |
| **V** | Vendor/promotional card, leaderboard aggregation, or third-party price page | Never alone |
| **D** | Chat/reasoning benchmark (MMLU-Pro, GPQA, AIME, LiveCodeBench) | **No.** Deliberately excluded from the decision |
| **U** | Unverified, modelled, or assumption-dependent | Must be checked before committing |

Two traps this document is written to avoid:

1. **The English-OCR trap.** A model that wins English document benchmarks can lose badly on Devanagari. The benchmark in §4 is the clearest evidence: GPT-5.5 drops to chrF++ 58.5, tying classical EasyOCR, while the open Qwen3-VL-8B reaches 75.2.
2. **The parser trap.** OCRBench and DocVQA measure whether text/layout was read, not whether a *specific field* was extracted with the correct value, unit, and source span. A high OCRBench score does not clear the span-exact gate.

---

## 3. Key evidence

### 3.1 Bulk extraction candidate: Qwen3.5-9B

| Fact | Value | Source / date | Tag |
|---|---|---|---|
| Release | Feb–Mar 2026; third-party pages disagree (2026-02-23, 2026-03-02, 2026-03-10) | Hugging Face model card; DataLearner; BenchmarkList | V |
| Parameters / licence | 9.7B dense, Apache-2.0 | Hugging Face card | V |
| Native context | 262,144 tokens | Hugging Face card | V |
| int4 footprint | ~5.9 GB weights; ~7.7 GB at 32k context | aimodel.directory estimate from `config.json` | V/U |
| MMLU-Pro | 82.5 | vendor card | V/D |
| IFEval | 91.5 | vendor card | V/D |
| OCRBench v2 | 64.1 | BenchmarkList aggregation | V/P |
| Hosted price | $0.10 input / $0.15 output per Mtok (DeepInfra); $0.17/$0.25 (Together) | OpenRouter / aggregators, checked 2026-11-09 | V/U |

**Why it is the provisional bulk default:** at 9.7B it fits the 16 GB hardware budget at int4 with a 32k working context; it is open-weight, so the corpus never leaves the host; and it is the only candidate in this size class with a current guided-JSON path. **What is missing:** a measurement of span-exact field extraction on our schema. MMLU-Pro and IFEval do not predict that. The bake-off in §7 is the deciding evidence.

> **Promotional-number flag.** MMLU-Pro 82.5, IFEval 91.5, and OCRBench v2 64.1 come from the model card and aggregator pages, not from an independent field-extraction study. They are recorded only to show the candidate is not obviously underpowered. They must not be quoted as extraction accuracy.

### 3.2 OCR candidate: Qwen3-VL-8B

| Fact | Value | Source / date | Tag |
|---|---|---|---|
| Technical report | Qwen3-VL series | arXiv:2511.21631, 2025-11-26 | P |
| DocVQA (Instruct) | 96.1 | vendor card, 2025-10-15 | V/P |
| OCRBench | 896 / 1000 | vendor card / leaderboard, 2026 | V/P |
| Real printed Devanagari, chrF++ | **75.2** (median CER 0.0, catastrophic 3.3%) | arXiv:2606.29213, 2026-06-28 | P |
| Hardware | FP8 ~8–9 GB; 4-bit ~4–5 GB | deployment summaries, 2026 | V/U |

DocVQA and OCRBench are **page-parsing** measures. A 96.1 DocVQA score means the model can answer questions about a page image; it does not mean it returns the fee, the seat count, and the internship rule with an exact span. The Devanagari benchmark is closer to our use case, but it is still word/short-phrase transcription, not field extraction.

### 3.3 Adjudication / NLP: hosted mid-tier

Gemini 2.5 Flash is the documented example, not a commitment:

| Fact | Value | Source / date | Tag |
|---|---|---|---|
| Standard price | $0.30 input / $2.50 output per Mtok | Google Gemini pricing, checked 2026-10-30; re-checked 2026-11-16 | V (list price) |
| Cached input | $0.03 per Mtok | same | V |
| Batch price | $0.15 input / $1.25 output per Mtok | same | V |
| Status | Scheduled to reroute to `gemini-3.5-flash-lite` on 2026-12-01 | Google AI Studio deprecations page, 2026-11-16 | V |
| Devanagari chrF++ | 86.3 (real scans) | arXiv:2606.29213, 2026-06-28 | P |

A mid-tier hosted model is proposed for the **second pass only**: conflict resolution and rubric sentences, not bulk extraction. The price is a list price, not a negotiated or guaranteed quote, and the model id will change in December 2026. Keep the provider/model pair env-overridable; do not hardcode `gemini-2.5-flash` anywhere.

### 3.4 Embeddings: bge-m3

No new evidence justified a change. bge-m3 remains the 1024-dimensional multilingual default already wired into `config/models.yaml`. This is an **unchanged baseline**, not a fresh benchmark result. If retrieval quality becomes a bottleneck, compare bge-m3 against a current multilingual embedding on our own claim/alias pairs; do not switch on a leaderboard alone.

**Serving decision 2026-11-18:** see §12 — bge-m3 runs on CPU, in-process; the :8010 chat server is not an embedder.

---

## 4. The Devanagari finding (why it drives the OCR decision)

**Finding: frontier OCR ≠ Indic OCR.** The published benchmark "Can OCR-VLMs Read Devanagari?" (arXiv:2606.29213, 2026-06-28) evaluated ten systems on 100 rendered FLORES-Hindi sentences under four degradations, plus 300 real printed Devanagari scans. Its results:

| System | Family | Real-scan chrF++ ↑ | Median CER ↓ | Catastrophic (CER > 50%) ↓ |
|---|---|---:|---:|---:|
| Gemini 2.5 Flash | frontier | **86.3** | 0.0% | 0.7% |
| Claude Opus 4.7 | frontier | 82.2 | 0.0% | 0.7% |
| Mistral OCR | frontier | 77.6 | 0.0% | 8.0% |
| Qwen3-VL-8B | open | **75.2** | 0.0% | 3.3% |
| GPT-5.5 | frontier | 58.5 | 12.5% | 7.7% |
| EasyOCR | classical | 58.3 | 16.7% | 26.3% |
| Qwen2.5-VL-3B | open | 45.4 | 20.3% | 14.7% |
| olmOCR-7B | open | 40.5 | 20.0% | 12.0% |
| Unlimited-OCR | specialised | 24.7 | 50.0% | 46.7% |
| DeepSeek-OCR | specialised | 10.4 | 100.0% | 89.0% |

*All numbers in this table are transcription metrics (**P**), not field-extraction metrics (**F**). They screen OCR quality; they do not prove that a fee or seat count will be extracted correctly.*

Two further findings matter more than the ranking:

1. **Clean renders hide the gap.** On clean synthetic text all ten systems sit in a narrow chrF++ 91–98 band. A synthetic-only evaluation would have picked almost anything. On real scans the field spreads over 76 chrF++ points. **Our bake-off must include real scans, not rendered text.**
2. **The English ranking does not transfer.** GPT-5.5, strong on English document OCR, falls to 58.5 — level with classical EasyOCR. olmOCR-7B, the model behind olmOCR-Bench, falls to 40.5. Conversely Qwen3-VL-8B reaches 75.2 on a single 24 GB GPU.

**Why this drives the OCR decision.** The OCR default is chosen for *our* corpus (Indian college PDFs, many with Devanagari names, Hindi instructions, and mixed-script tables), not for an English benchmark. Qwen3-VL-8B is the best open model that fits 16 GB and is close enough to the hosted leaders to be the local default. Gemini remains the accuracy escalation path for pages where the local model is uncertain, not the bulk OCR engine: at 86.3 vs 75.2 it is better, but it costs per page and sends document images to a third party.

**Caveats on this benchmark (do not overstate it):**
- The real set is 300 **word/short-phrase** scans from historical Sanskrit-typeset material, not full-page Hindi documents. The authors state this explicitly and note it disadvantages page-oriented models.
- Closed-model numbers reflect API versions at eval time and will drift.
- CER is code-point, not grapheme-cluster, level; Devanagari graphemes span multiple code points.
- It measures transcription, not field extraction. A 75.2 chrF++ model can still miss the one fee field that matters.

---

## 5. The span-fidelity finding (why extraction is quote-first)

**Finding: models are unreliable at generating character offsets.** They are good at identifying the relevant text and bad at returning the exact `start`/`end` indices. This is not a prompt-tuning problem; it is a counting problem.

Evidence:

| Finding | Source / date | Tag |
|---|---|---|
| "initial LLM offset error rate of roughly 20–25%"; strict exact-substring filtering reduced the final annotation error to under 2% | MedExACT 2026 @ ACL, 2026-06-12 | F |
| "vanilla LLMs do not return verified character offsets out of the box"; LangExtract post-processes with sequence alignment to recover spans | Google LangExtract maintainer, issue #259, 2025-10-04 | F |
| LMKG asks the model for a verbatim quote and a deterministic resolver assigns the span; when no match exists, the span stays empty rather than being guessed | LMKG, AKBC 2026, 2026-10-05 | F |
| Inline citation/span accuracy for top models still hovers around 85–95%, i.e. 1 in 10–20 is misleading | ZeroEntropy citation-extraction note, 2026 | V/U |

**Design consequence.** Extraction is **quote-first**:

1. the model emits a verbatim `quote`, a `confidence`, and `reason_if_null`; it does **not** emit `value`, `unit`, `char_start`, or `char_end`;
2. `fmd.extract.validate` resolves the quote exactly, then within a fuzzy window at ≥0.95 similarity;
3. a resolved span rewrites/logs a near-miss and lowers confidence; an unresolved quote is rejected, never published with a guessed offset;
4. the value and unit are derived deterministically from the resolved span.

This is the contract already documented in `fmd/extract/schemas/base.py`; **WS-SPAN is implementing the resolver/validator side.** The bake-off therefore scores `span_ok` only when the predicted quote resolves at the predicted span *and* equals the golden quote. A candidate that returns plausible values with guessed offsets fails the span-exact gate even if its value accuracy looks good.

**Implication for model choice.** A model that is weaker at prose but reliable at verbatim copying can beat a stronger chat model here. That is one reason the bulk decision is not taken from MMLU-Pro: the winning behaviour is "copy the line exactly", not "reason about the line".

---

## 6. Two-pass economics per 1,000 pages

**Currency:** USD. **Price date:** 2026-11-16. **Status:** arithmetic on stated assumptions, not a provider quote. Prices are list prices from Google's pricing page and third-party aggregators; local electricity and throughput are modelled. Marked **[M]** where modelled and **[U]** where the underlying price is unverified.

### 6.1 Assumptions

| Assumption | Value | Basis |
|---|---:|---|
| Pages in the run | 1,000 | defined for this calculation |
| Pass-1 input tokens / page | 1,200 | prompt + extracted text; **[M]** |
| Pass-1 output tokens / page | 500 | quote-first JSON; **[M]** |
| Pass-2 sample | 20% of pages (200) | conflicts + low confidence; **[M]** |
| Pass-2 input / output per sampled page | 1,200 / 300 | **[M]** |
| OCR share of pages | 15% (150) | corpus assumption; **[M]** |
| OCR input / output per scanned page | 2,000 / 800 | vision tokens; **[M]** |
| Local GPU power | 200 W | 16 GB class card; **[M]** |
| Electricity | $0.15 / kWh | **[M]** |
| Local throughput | 8 pages/min | int4 9B, one request at a time; **[M]** |
| Rented 16 GB GPU | $0.40 / hour | market assumption; **[M/U]** |

### 6.2 Pass 1 — bulk extraction

| Route | Arithmetic | Cost / 1,000 pages |
|---|---|---:|
| Local Qwen3.5-9B int4, owned GPU | (1,000 ÷ 8) min = 125 min = 2.08 h; 2.08 h × 0.2 kW = 0.417 kWh; 0.417 × $0.15 | **$0.06 [M]** |
| Local Qwen3.5-9B int4, rented GPU | 2.08 h × $0.40 | **$0.83 [M/U]** |
| Hosted Qwen3.5-9B, DeepInfra list | 1.2M input × $0.10 = $0.120; 0.5M output × $0.15 = $0.075 | **$0.195 [U]** |
| Hosted Qwen3.5-9B, Together list | 1.2M × $0.17 = $0.204; 0.5M × $0.25 = $0.125 | **$0.329 [U]** |

**Honest reading:** local is only cheaper if the GPU is already owned and would otherwise be idle. At rented-GPU prices, hosted bulk is ~4× cheaper than the local route ($0.195 vs $0.83). The 16 GB hardware budget therefore buys privacy and offline operation, not necessarily lower cost.

### 6.3 Pass 2 — adjudication (hosted mid-tier)

| Scope | Arithmetic | Cost / 1,000 pages |
|---|---|---:|
| 20% sample (200 pages), Gemini 2.5 Flash standard | 0.24M input × $0.30 = $0.072; 0.06M output × $2.50 = $0.150 | **$0.222 [U]** |
| 100% of pages, standard | 1.2M × $0.30 = $0.360; 0.3M × $2.50 = $0.750 | **$1.110 [U]** |
| 20% sample, batch price | 0.24M × $0.15 = $0.036; 0.06M × $1.25 = $0.075 | **$0.111 [U]** |

### 6.4 OCR pass

| Route | Arithmetic | Cost / 1,000 pages |
|---|---|---:|
| Local Qwen3-VL-8B, 15% scanned | ~0.25 GPU-h at 200 W ≈ 0.05 kWh × $0.15 | **$0.008 [M]** |
| Hosted Qwen3.5-9B (proxy price), 15% scanned | 0.30M input × $0.10 = $0.030; 0.12M output × $0.15 = $0.018 | **$0.048 [U]** |

### 6.5 Totals

| Configuration | Arithmetic | Cost / 1,000 pages |
|---|---|---:|
| Local bulk + local OCR + 20% hosted adjudication | $0.06 + $0.008 + $0.222 | **≈ $0.29 [M/U]** |
| Hosted bulk + local OCR + 20% hosted adjudication | $0.195 + $0.008 + $0.222 | **≈ $0.43 [U]** |
| Hosted bulk + local OCR + 100% hosted adjudication | $0.195 + $0.008 + $1.110 | **≈ $1.31 [U]** |
| Rented local bulk + local OCR + 20% adjudication | $0.83 + $0.008 + $0.222 | **≈ $1.06 [M/U]** |
| Hosted bulk + hosted OCR + 20% adjudication | $0.195 + $0.048 + $0.222 | **≈ $0.47 [U]** |

**Two-pass economics conclusion.** The second pass is affordable *because it is sampled*. Running the hosted mid-tier on every page costs more than the entire local first pass; sampling 20% keeps the total near $0.30–0.47 per 1,000 pages. This is a cost argument for the sampling policy, not proof that the second pass improves accuracy — the bake-off must measure that.

**Price caveats.** Google's list price may change; `gemini-2.5-flash` reroutes to `gemini-3.5-flash-lite` on 2026-12-01; third-party Qwen3.5-9B prices vary by 2× across hosts; none of these prices is a contractual quote. Re-verify before committing.

---

## 7. The 20-document bake-off protocol

The bake-off is the decision mechanism. Until it runs, the defaults in §1 are provisional.

### 7.1 Document selection (20 documents)

| Stratum | Count | Selection criterion |
|---|---:|---|
| Born-digital text PDFs | 6 | Clean embedded text; fee/seat/placement sections; 2–30 pages |
| Scanned or mixed-script | 6 | At least 3 with Devanagari; include one low-DPI and one skewed scan |
| Table-heavy | 4 | Fee matrices, seat matrices, placement tables; merged cells and footnotes |
| Adversarial | 4 | Repeated numbers, multi-column reading order, absent fee, internship counted as employment, a rate without a denominator |

Stratify by state, ownership, and **data thinness** (deliberately over-sample thin documents). Freeze the 20 files and their SHA-256 before any candidate runs. Hold back 5 documents from prompt iteration; run the final comparison on all 20, but report the held-out subset separately so prompt tuning cannot leak.

### 7.2 Ground truth

- Same YAML schema as `data/golden` (`GoldenEntry` / `GoldenValue`).
- Two annotators per document, then adjudication; record disagreements.
- At least 8 verified fields per document, ≥160 verified rows in total. Unverified rows are allowed and counted, but **never scored**.
- A row is verified only if the quote is verbatim in the document text or a cited source is recorded. **The harness refuses to run with zero verified rows**; it does not invent ground truth.
- For real PDFs, use `source_url` + `retrieved_at` rather than `fixture` (the golden loader only verifies quotes against UTF-8 text). The bake-off still checks the quote against the extracted text at scoring time.

### 7.3 Prompts and decoding

- Use the production family prompt for each document's `family`; pin `prompt_version` and keep it identical across candidates.
- `temperature: 0`; guided JSON on; thinking off for bulk extraction.
- One retry is allowed **only** on a schema violation, and the retry is counted. Do not silently repair a candidate's output.
- No candidate-specific prompt tuning. If a candidate needs a different prompt to work, that is a finding, not a free pass.
- Log the prompt hash and input hash for every call; the bake-off is reproducible.

### 7.4 Metrics

| Metric | Definition | Source |
|---|---|---|
| Span-exact precision | TP / (TP + FP); TP requires value, normalised unit, and a quote that resolves at the reported span **and** equals the golden quote | `fmd/eval/metrics.py` `summarise_fields` |
| Fuzzy precision | value + unit correct, span ignored, over all predictions | derived from `FieldResult` |
| Catastrophic rate | predictions with a value that differs from golden / all predictions | derived from `FieldResult` |
| Null precision | correct abstentions / all abstentions | `abstention_correctness` |
| Null recall | correct abstentions / golden abstain rows | derived from `expectation == "abstain"` |
| Schema-valid rate | successful guided-JSON validations / attempts | harness |
| Cost / page | observed token cost / pages, only when caller supplies prices and the provider returns usage | harness + router sink |
| p50 / p95 latency | per-document wall time for the model call | harness |

### 7.5 Hard gates

A candidate is **eliminated** if any of these fails:

| Gate | Threshold | Rationale |
|---|---:|---|
| Span-exact precision | **≥ 98%** | A published number without a resolving span is a product defect (brief §19) |
| Catastrophic rate | **≤ 0.5%** | A confidently wrong number is worse than an abstention |
| Null precision | **≥ 95%** | The model must not invent values the document does not contain |

Schema-valid rate and null recall are reported but are not hard gates in v1. A candidate with schema-valid rate < 100% will usually fail span-exact anyway because missing fields become false negatives.

### 7.6 Decision rule

1. Eliminate every candidate that fails a hard gate.
2. Among survivors, rank by **cost per page**, then **p95 latency**; break ties by span-exact precision.
3. If the top two survivors are within **2 percentage points** of span-exact across 20 documents, declare no winner and keep the incumbent default. Twenty documents cannot resolve a smaller difference.
4. Report Wilson 95% confidence intervals for span-exact and null precision. Do not quote a point estimate without its interval.
5. If no candidate passes, keep the current registry defaults and expand the document set; do not lower the gates to manufacture a winner.
6. Publish the full JSON report and the per-field table with the decision. A model choice without its evidence is not auditable.

### 7.7 Running the harness

From `packages/pipeline` (the uv project):

```bash
uv run python scripts/bakeoff.py \
  --pdfs /data/bakeoff/docs \
  --ground-truth /data/bakeoff/golden \
  --root /data/bakeoff \
  --candidates-file /data/bakeoff/candidates.yaml \
  --json-out /data/bakeoff/report.json
```

`--root` is the base for relative `fixture:` paths; it must be the directory that
contains `docs/` and `golden/` for an external bake-off. Use `--mock
--mock-fixtures <dir>` to replay recorded responses offline. The harness exits
non-zero when every candidate fails a hard gate; `--allow-no-pass` overrides that
for exploratory runs. Candidate prices are caller-supplied metadata, never
invented by the harness; omit them and the `$/page` column reads `n/a`.

---

## 8. Provisional defaults and overrides

| Profile | Default | Env override | Note |
|---|---|---|---|
| `bulk_extract` | `local` / `Qwen3.5-9B-AWQ`, 32k, int4, thinking off | `FMD_EXTRACT_PROVIDER`, `FMD_EXTRACT_MODEL` | Guided JSON on |
| `hard_pdf` | `api` / unset (escalation only) | `FMD_HARD_PDF_PROVIDER`, `FMD_HARD_PDF_MODEL` | Optional; degrades per document |
| `ocr_vlm` | **retired 2026-11-16** — OCR is an acquisition stage, not a router profile | `FMD_T2_URL` (T2 PaddleOCR-VL), `FMD_T2B_URL` (T2b Qwen3-VL) in `config/acquire.yaml` | Optional; when unset the tier raises `TierUnavailable` and the page goes to the human queue |
| `adjudicate` | `gemini` / `gemini-2.5-flash` | `FMD_ADJUDICATE_PROVIDER`, `FMD_ADJUDICATE_MODEL` | Documented example; reroute 2026-12-01 |
| `entity_adjudication` | `local` / `Qwen3.5-9B-AWQ` | `FMD_ENTITY_ADJUDICATE_PROVIDER`, `FMD_ENTITY_ADJUDICATE_MODEL` | Pairwise only |
| `nlp_rubric` | `gemini` / `gemini-2.5-flash` | `FMD_NLP_PROVIDER`, `FMD_NLP_MODEL` | Quote required |
| `embed` | `cpu` (in-process) / `BAAI/bge-m3` | `FMD_EMBED_PROVIDER`, `FMD_EMBED_MODEL` | 1024-d; §12 (2026-11-18): CPU default, 0 VRAM; `embed_openai` = POST /v1/embeddings |

**Guided JSON / architecture caveat.** `guided_json: true` is kept, but xgrammar (or the served backend's constrained decoder) must support the Qwen3.5 architecture in the pinned vLLM/SGLang build. If it does not, the router degrades to JSON-instruct plus schema validation; the bake-off must then report the degraded path's schema-valid rate, because it is not the same candidate. Verify the xgrammar/backend version before recording results.

**Thinking-disable caveat.** `thinking: false` is recorded in `config/models.yaml` as intent, but `fmd/llm/providers.py` does not yet forward it. Enforce it in the served chat template (for example a chat-template kwarg) until the provider adapter carries it. If the model thinks by default, token counts, latency, and cost in §6 are wrong.

**Model-id caveat (2026-11-16).** No `Qwen/Qwen3.5-9B-AWQ` repository existed on the Hub; the local default is a community int4 AWQ conversion served under the alias `Qwen3.5-9B-AWQ`. The alias is what `config/models.yaml` pins, so the id is stable, but the weights are not the vendor's. Record which conversion the bake-off used.

---

## 9. Claims to verify before committing

These are the research's open claims, kept as claims — not as facts. Each must be checked on our corpus or against a first-party source before the defaults become a commitment.

1. **"Qwen3.5-9B at int4 and 32k context fits in 16 GB and keeps extraction agreement within 1 pp of bf16."** — Check: serve it, run the bake-off, compare against a bf16/FP8 run on the same 20 documents.
2. **"Thinking can be disabled for Qwen3.5-9B in the pinned vLLM/SGLang build."** — Check: inspect the chat template and verify token counts with thinking on vs off.
3. **"xgrammar supports the Qwen3.5 architecture for guided JSON."** — Check: run one guided call on the pinned backend; confirm `guided_json` is not silently degraded.
4. **"Qwen3-VL-8B-Instruct is the best open Devanagari OCR we can run in 16 GB."** — Check: our scan stratum, not the 300-word benchmark; include Gemini as an accuracy reference.
5. **"The Devanagari real-scan collapse reproduces on our documents."** — Check: compare clean vs scanned pages in the bake-off; the published benchmark used word/short-phrase scans.
6. **"32k context covers ≥95% of our PDFs without chunking."** — Check: measure page/token distribution on the corpus; add chunking if the tail is long.
7. **"Gemini 2.5 Flash is callable and priced at $0.30/$2.50 through the bake-off; the 2026-12-01 reroute does not change behaviour."** — Check: first-party pricing page and a live call; pin the model id.
8. **"Hosted mid-tier adjudication beats local 9B on span-exact by at least 1 pp at 20% sampling."** — Check: bake-off pass 2; if not, keep adjudication local and save the money.
9. **"LLM offset error is ~20–25% under our prompts."** — Check: instrument a sample; this is why the design is quote-first, but the number itself is from another task.
10. **"bge-m3 remains the best multilingual embedding per GB for our retrieval."** — Check: retrieval/alias evaluation on our pairs; this is an unchanged baseline, not a fresh result.
11. **"Local electricity is ~$0.06 per 1,000 pages and a rented GPU is ~$0.83."** — Check: measure wall-clock and power; these are modelled, not metered.
12. **"Aggregator prices ($0.10/$0.15 for Qwen3.5-9B) match first-party pages."** — Check: DeepInfra/SiliconFlow/Together pages directly; aggregator prices moved within days during this research.
13. **"Twenty documents and ≥160 verified rows are enough to separate candidates by 2 pp."** — Check: power calculation on the observed variance; if not, expand the set before declaring a winner.

---

## 10. References

1. Qwen3.5-9B model card and benchmark table — Hugging Face / ModelScope, accessed 2026-11-16. Vendor card; release date disagrees across third-party pages.
2. Qwen3-VL Technical Report — arXiv:2511.21631, 2025-11-26.
3. Qwen3-VL-4B/8B release note and OCRBench/DocVQA figures — DataLearner, 2025-10-15. Vendor-sourced.
4. "Can OCR-VLMs Read Devanagari? A Stress-Test Benchmark and Post-Correction Study" — arXiv:2606.29213, 2026-06-28; benchmark repository `Aditya-PS-05/devanagari-ocr-benchmark`.
5. "Team Aurum at MedExACT 2026@ACL" — ACL Anthology 2026.bionlp-2.29, 2026-06-12. LLM offset error 20–25%.
6. Google LangExtract issue #259, maintainer reply 2025-10-04. "Vanilla LLMs do not return verified character offsets."
7. "LMKG: a flexible method for ontology-aware Knowledge Graph Extraction from text" — ACL Anthology 2026.akbc-1.23, 2026-10-05. Quote + deterministic resolver.
8. Gemini Developer API pricing — `ai.google.dev/gemini-api/docs/pricing`, checked 2026-10-30 and 2026-11-16. List price, not a quote.
9. Gemini API deprecations — `ai.google.dev/gemini-api/docs/deprecations`, checked 2026-11-16. `gemini-2.5-flash` reroutes 2026-12-01.
10. Qwen3.5-9B hosted price pages — OpenRouter model page; DeepInfra/SiliconFlow/Together listings via aggregators, checked 2026-11-07 to 2026-11-09. Third-party; unverified against first-party pages.

---

## 11. What would change this decision

- The bake-off fails every candidate on span-exact: keep the current registry defaults and expand the document set; do not relax the gate.
- Qwen3.5-9B cannot reliably disable thinking or cannot run guided JSON on the pinned backend: fall back to the previous `Qwen3-14B-AWQ` for bulk, or a hosted mid-tier, and re-run.
- The Devanagari stratum shows Qwen3-VL-8B below the null/span gates: move OCR to a hosted Gemini-class model for scanned pages, accepting the egress and per-page cost.
- Hosted adjudication does not beat local 9B by 1 pp: make adjudication local and reduce the two-pass cost to near zero.
- A first-party price or model id changes before the bake-off: update the candidates file, not this document's arithmetic; record the new date and source.

**Until the bake-off runs, treat every default in this document as a hypothesis with a named test.**

---

## 12. Decision note — 2026-11-18 (embed provider: bge-m3 on CPU; the :8010 chat server is not an embedder)

**This note supersedes §3.4 and the §8 `embed` row for serving only.** The
embed model itself is unchanged: bge-m3, 1024-d, multilingual (§3.4 stands for
*which* model; this note decides *where it runs*). The defect this fixes: the
`embed` profile pointed at `local` (`http://localhost:8010/v1`), which is the
llama.cpp `llama-server` hosting `gemma-4-26b-a4b-gguf` — a chat MoE whose GGUF
is not an embedding model and which does not serve `POST /v1/embeddings`. ER
name matching and insight diversity therefore fell back to the lexical
heuristics *silently*.

### 12.1 The three options, against the host's free VRAM

Anchors: the config of record (64k ctx, `-b 4096 -ub 2048`, §2.8 of the
runbook) measures **15,150 MiB used / 1,153 MiB free**; the runbook's
headroom rule is ≥ 1 GiB free; one GPU tenant at a time (runbook §1);
CUDA context alone is **~0.21 GiB** (runbook §2.7-B anchor). bge-m3 is 568M
params; footprints below are **estimates [U]**, not measurements: fp16
~2.2 GB, int8 ~1.1 GB, q8_0 GGUF ~0.6 GB.

| Option | VRAM cost (est. [U] + measured context) | Verdict |
|---|---|---|
| (a) bge-m3 on GPU alongside the 26B | fp16 ~2.2 GB or int8 ~1.1 GB, or q8 GGUF ~0.6 GB; **+ ~0.21 GiB second CUDA context + activation buffers (est. 0.1–0.4 GiB)** ⇒ ≥ 0.9–1.3 GiB total against **1,153 MiB free** | **Rejected on arithmetic.** Even the smallest footprint consumes the entire measured headroom and breaks the ≥1 GiB rule; any variance OOMs the *required* text profile to serve an optional one. The same-server variant (`--embedding` on :8010) is not a fallback either: the served 26B chat GGUF has no embedding head, so "alongside" always means a second GPU process — forbidden by the one-tenant rule. |
| (b) bge-m3 on CPU, in-process | **0 MiB VRAM**; ~2.2 GB fp32 / ~1.1 GB int8 system RAM (est. [U]) of 32 GB | **Chosen.** Workload is small (ER names ≤ 32 tokens; insight sentences; hundreds per run, not millions), the Core Ultra 9 HX covers a 568M encoder for that scale [U — `python -m fmd.llm.embed_bench` is the named test], the sentence-transformers path is the reference pooling/normalisation (no GGUF conversion drift), and it is deterministic per fixed input/build. Same posture the runbook already takes for Docling T1 (CPU deliberately, runbook §6). |
| (c) second small GGUF, own llama.cpp instance | GPU: same arithmetic as (a) — rejected; CPU: 0 MiB VRAM but a second process, port, `--pooling cls`/`-ub` config surface and GGUF conversion between us and the reference vectors | **Rejected as redundant.** On GPU it inherits (a)'s failure; on CPU it buys nothing over (b) and adds exactly the kind of second server this runbook documents as a failure domain. Revisit only if a measured ER batch misses the embed budget and the CPU path cannot be tuned (`FMD_EMBED_BATCH`, threads). |

**Decision: (b).** `FMD_EMBED_PROVIDER=cpu` (default), in-process
sentence-transformers, weights prefetched once by the operator. The decision is
reversible without code changes: `FMD_EMBED_PROVIDER=embed_openai` points the
same profile at any OpenAI-compatible `POST /v1/embeddings` endpoint (llama.cpp
`--embedding` on :8011, or hosted), and `embed_bench` re-runs as the acceptance
test either way. Cost accounting: v1 forbids paid APIs; the default route keeps
that true at zero egress.

### 12.2 Failure semantics (no silent degradation)

* Backend/model absent ⇒ `embeddings_unavailable` is **recorded** (router
  doctor row, ER audit note, `insight_sets.coverage_json.embedding_mode`) and
  the documented lexical features (token Jaccard + char ratio + acronym +
  context) keep running with the mode surfaced. The fallback is a *reported
  state*, not a silent downgrade.
* Dimension mismatch ⇒ **rejected** (`dimension_mismatch`), never truncated;
  count mismatches and out-of-order batches are contract errors too.
* Named test for the throughput claim: `uv run python -m fmd.llm.embed_bench`
  (runbook §8 documents expected output shape; the numbers are the operator's,
  not this document's — none are quoted here and none were invented).

### 12.3 What would change this decision

* `embed_bench` shows the CPU path missing the ER/insight latency budgets on
  real corpora: first tune `FMD_EMBED_BATCH`/threads, then try an int8 ONNX
  export, and only then put bge-m3 GGUF behind `embed_openai` on :8011 —
  accepting that the 26B must drop to 32k context to make VRAM room.
* The text model leaves the card or drops below 64k by policy: re-run the
  §12.1 arithmetic; fp16 GPU co-residency becomes plausible above ~2.5 GiB
  free and is then worth measuring.
* A hosted embeddings endpoint with bge-m3-compatible quality enters the
  no-paid-API boundary: it is an env switch (`embed_openai`), not a fork.
