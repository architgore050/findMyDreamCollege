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

## B. Needs stable network

| ID | Item | Why it waits | Est. |
|---|---|---|---|
| D5 | **Container repair** — `api` image crashes on `ModuleNotFoundError: yaml`; we run from the host venv | Rebuild pulls packages; needs a stable link | 1–2 h |
| D6 | **Crawl scale-up** — 50 domains → 300–500, deeper per domain | Sustained bandwidth over hours | 1–3 days |
| D7 | **Deployment** — VM, object storage for snapshots, read-only public DB/API, methodology page | Not a v1 blocker; needs hosted infra | 2–3 days |

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

---

## Status snapshot at deferral (2026-11-18)

- Inventory: 1,247 institutions · 1,204 domains
- Evidence: 135 snapshots · 49 extractions · 32 claims (29 verified `supported`)
- Programmes: 2 · scores partial, **not rankable**
- Tests: 763 pipeline · 51 web · provenance coverage 1.0 (16 verified golden rows)
- Model: Gemma-4-26B-A4B UD-IQ4_XS, llama.cpp, 64k ctx, 4,784 tok/s prefill / ~95 tok/s decode
