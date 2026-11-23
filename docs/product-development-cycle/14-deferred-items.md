# Deferred items — revisit after classes / stable network

**Trigger:** when the operator next asks "what's left for v1", read this file aloud (concisely) *before* proposing new work.

Items here are deliberately **not** being worked on now. They are blocked by GPU time, sustained network, or human review — not by design uncertainty.

---

## A. Needs GPU hours (do when the laptop is free)

| ID | Item | Why it waits | Est. |
|---|---|---|---|
| D1 | **Re-harvest the backlog** — ~86 unextracted snapshots | Long GPU run; current claims won't be invalidated, so no urgency | 30–40 min GPU |
| D2 | **Golden set → ~50 verified rows** | Needs a human to confirm values against sources; agents can only draft | 2–4 h human |
| D3 | **20-document bake-off** against the gates (span-exact ≥98%, catastrophic ≤0.5%, null precision ≥95%, recall ≥85%) | Needs GPU + the expanded golden set | 1 day |
| D4 | **OCR / T2 tier** for scanned PDFs (~25% of corpus) | Needs weights + GPU; currently routes to T3 | 0.5–1 day |

## A2. New requirement (added 2026-11-18) — exam registration windows

| ID | Item | Why it matters | Blocked by |
|---|---|---|---|
| D17 | **Exam registration windows** — per exam, per cycle, registration open/close vs sitting date, with observed_at + TTL and supersession; closed ⇒ binary exclusion | A programme can be open while its required exam is closed; the applicant is then ineligible regardless of merit. Spec: `17-exam-registration-requirements.md` | Network for live notifications; **coding can proceed now** (schema/prompts/tests offline) |

## A3. OCR / vision — must ship, currently dev-disabled (operator decision 2026-11-19)

**Product requirement:** the tool must NOT ship without document vision. Image-only PDFs,
table images, charts and scanned annexures are a large share of Indian institutional
disclosures; without vision they route to a human queue and the evidence is simply missing.

**Current state (dev):** Docling's *built-in* OCR (`RapidOCR`) is disabled by default via a
hotfix, because it is unbounded (no page cap, no time budget) and stalled a run for 25+
minutes. That is a **development-mode** setting, not the product policy.

**D18 status (W7 patch, 2026-11-19):** the bounded T2 tier is IMPLEMENTED — default backend
`rapidocr` (PP-OCR det+rec over the RapidOCR runner, in-process), `device: cpu` default
(measured 2.33 s/page ≈ 25.8 pages/min; cuda is opt-in with a recorded
`cuda_unavailable_vram` fallback), pinned PP-OCRv6-det + PP-OCRv5-devanagari models,
page/time budgets with `acquisition_budget_exceeded` abort semantics, per-page confidence
floor + `conjunct_risk` + digit-normalisation accounting, engine metadata on every OCR claim,
and Qwen3-VL retained as Devanagari fallback + selectable backend. **Not yet measured**: the
`scripts/ocr_validate.py --ground-truth` run on real scans (doc 61 pages 3/5/7 seed) and the
D18 quality gates — the engine preference is provisional until that run (runbook §3.8).

**Policy to implement (D18):**
1. **Primary OCR: PaddleOCR PP-OCRv5 server det+rec models, executed through RapidOCR on
   onnxruntime-gpu.** Rationale: purpose-built detector+recogniser, permissive licence, GPU
   batching, and it is the same model family RapidOCR wraps — so "RapidOCR vs PaddleOCR" was a
   false choice: RapidOCR is the *runner*, PP-OCRv5 is the *model*. Table structure is NOT its
   job; Docling/TableFormer keeps that. **Qwen3-VL-8B stays wired as a measured fallback** for
   pages whose Devanagari CER/chrF++ fails the gate, and PaddleOCR-VL remains an alternate
   full-pipeline engine. One engine per page, one transcript, engine+version+model ids in the
   claim qualifier.

   **Measured (2026-11-19, local):** PP-OCRv5 Devanagari page CER **10.9%** on a dense Hindi page,
   25.8 pages/min on CPU (2.33 s/page), 200 pp ≈ 8 min. GPU is **not** usable while the 26B is
   resident (688 MiB free vs ~1.5–2 GiB needed) — CPU is the default; GPU only in batch mode with
   the LLM stopped. Known defects: virama/conjunct errors (`शक्ति→शक्ि`), watermark boxes with
   confidence <0.7, Devanagari digits untested. Qwen3-VL remains the fallback for conjunct-heavy
   pages; the mixed-engine rule stays.
2. **Never remove visibility.** Every scanned page records `bbox_granularity: block` plus the
   OCR engine + version in the claim qualifier, so a reviewer can always see what was read and
   by what. Aborted OCR is a visible `acquisition_budget_exceeded`, never a silent zero.
3. **Bounded batches.** OCR validation runs on small batches (≤10 documents) first, with
   `max_pages` + `max_seconds` per document and a stage budget; measure pages/min before any
   bulk run. Re-enable only when the measured rate is acceptable.
4. **Quality gate before shipping:** CER/chrF++ on real Hindi-header scans, table TEDS on
   merged-cell seat/fee tables, and offset-resolvability 100% at block granularity. Evaluation
   set from our own corpus, not synthetic pages.
5. **Dev/prod switch is explicit**, not ambient: `T1.do_ocr` tri-state + `FMD_T2_URL` presence,
   reported in `fmd llm doctor` and the run summary ("vision: on/off, engine, pages OCR'd").

## B. Needs stable network

| ID | Item | Why it waits | Est. |
|---|---|---|---|
| D5 | **Container repair** — `api` image crashes on `ModuleNotFoundError: yaml`; we run from the host venv | Rebuild pulls packages; needs a stable link | 1–2 h |
| D6 | **Crawl scale-up** — 50 domains → 300–500, deeper per domain | Sustained bandwidth over hours | 1–3 days |
| D7 | **Deployment** — VM, object storage for snapshots, read-only public DB/API, methodology page | Not a v1 blocker; needs hosted infra | 2–3 days |

## B2. Post-v1: cross-country credential equivalence (not in v1)

The v1 programme taxonomy is **India-first and rule-based**. Outside India, the same degree is named
differently — and universities sometimes invent a one-off name that appears in no alias list:

- US/Canada: Medical Laboratory Science / Clinical Laboratory Science
- UK/Ireland: Biomedical Science (with IBMS accreditation), Healthcare Science
- Australia/NZ: Laboratory Medicine, Medical Laboratory Science, Biomedical Science
- Philippines: BS Medical Technology / Medical Technology
- Gulf: Medical Laboratory Sciences, Clinical Laboratory Technology
- Germany/Nordics: often language-gated (MTLA, bioanalytiker), not English-taught

**Exact alias matching is insufficient by design.** A future agent must implement a three-tier
equivalence decision, not a longer list:

1. **Rules** — curated include/exclude per country in `config/programmes.yaml` (v1 already stubs the
   `IN` block; other countries land as empty stubs, never as invented patterns).
2. **Semantic equivalence classifier** — for names that match no rule: given the programme name, the
   institution, department, and any curriculum/regulator claims, decide
   `equivalent: yes | no | uncertain` with a **citation** (which claims/evidence) and a rationale.
   Reference definition is conceptual ("diagnostic laboratory science: clinical biochemistry,
   haematology, microbiology, histopathology, blood bank"), **not** a name list. Must be conservative:
   `uncertain` → `needs_review`, excluded from ranking; never auto-admitted.
3. **Learning loop** — a human `needs_review` decision becomes a `manual` alias that the rule tier
   consumes next run, so the classifier is asked less over time and the system accumulates knowledge
   instead of re-deciding.

Guardrails for whoever builds this: no invented equivalences, every admission cites evidence, the
classifier's model/prompt version is recorded per programme, and a false positive (admitting MBBS or
Nursing as MLT) is worse than leaving it `uncertain`.

**Sources to consult before expanding:** country regulators and accreditation bodies (NAACLS/ASCP US,
CSMLS Canada, IBMS/HCPC UK, AIMS Australia, PCQ NZ, PRC Philippines, DHA/SCFHS Gulf), plus national
qualification frameworks. Do not infer equivalence from a name alone.

## C. Needs a human decision (not engineering)

| ID | Item | Why it waits |
|---|---|---|
| D8 | **Gemma 4 licence review** — Hub metadata says Apache-2.0; `license_link` points at Gemma terms. Contradictory | Must be resolved before any public launch |
| D9 | **Playwright MCP settings** applied on the operator's machine (concurrency 1, nav/action timeouts, asset blocking) | Lives outside the repo; documented only |
| D10 | **Country scope** — Tier B (PH/MY/AE/ZA) after India depth holds | Product decision, ~2 weeks per country |
| D11 | **Publish policy** — public DB/API, takedown process, corrections channel | Governance decision |

## D. Known code debt (small, unblocked, not v1-blocking)

| ID | Item |
|---|---|
| D12 | Two bare `httpx.post` call sites (`fmd/adapters/nabh.py`, `fmd/acquire/tiers.py`) bypass the shared network policy |
| D13 | Review console UI still 501 (schema + audit rows exist) |
| D14 | `hard_pdf` optional-tier semantics verified; vision path unbuilt |
| D15 | Scheduled crawling from registry cadence |
| D16 | Embedding provider — see WS-EMBED decision (CPU bge-m3 vs GPU headroom) |

## E. Future — per-profile prompt-injection pass (LLM)

**Why it waits:** at current scale the admission contract already requires cited quotes and
span resolution, so the injection surface is unmeasured and coding would be optimization, not
necessity. Decide from data first (the scrape-failure log and the prompt-effectiveness stream).

| ID | Item | Why it waits | Est. |
|---|---|---|---|
| D18 | **Per-admission LLM injection pass** — one LLM pass per admission/plan item, screening the
  admission input (classification evidence + context sample) for injected instructions
  (verdict: clean / injected_instruction / malformed_quote); quarantine or reject on finding.
  Also screens plan prompts for instructive-looking quoted context. | Adds one LLM call per admission (cost/latency); measure via the quoting contract first | Small (guard module + guard prompts + tests) |

---

## Status snapshot at deferral (2026-11-18)

- Inventory: 1,247 institutions · 1,204 domains
- Evidence: 135 snapshots · 49 extractions · 32 claims (29 verified `supported`)
- Programmes: 2 · scores partial, **not rankable**
- Tests: 763 pipeline · 51 web · provenance coverage 1.0 (16 verified golden rows)
- Model: Gemma-4-26B-A4B UD-IQ4_XS, llama.cpp, 64k ctx, 4,784 tok/s prefill / ~95 tok/s decode
