# FMD-CLOUD-002 · embed-provider · wave w1

You are a one-shot cloud coding agent for project `fmd` (findMyDreamCollege).

**Repo:** https://github.com/architgore050/findMyDreamCollege.git
**Base SHA:** `ff7d13e2192d22ae25f666f69f03b73a1d9f6063`
**Branch:** `feature/fmd-w1-embed-provider-ff7d13e-<UTC>-01`
Do not push, open PRs, merge, or use secrets.

## Read first
- `AGENTS.md`
- `docs/product-development-cycle/04-phase-3-implementation-brief.md` §15 (model router)
- `docs/product-development-cycle/10-inference-and-model-selection.md`
- `docs/product-development-cycle/11-inference-runbook.md`
- `packages/pipeline/config/models.yaml` (`embed` profile — note the path is `config/models.yaml`)
- `packages/pipeline/fmd/llm/{router,providers}.py`, `fmd/er/similarity.py`, `fmd/score/insights.py`

## Problem
The `embed` profile points at `http://localhost:8010/v1` — the llama.cpp server hosting `gemma-4-26b-a4b-gguf`, which does **not** serve embeddings. Entity-resolution name matching and insight diversity silently fall back to weak lexical heuristics. It must become real, or fail loudly.

## Host facts (the operator's machine — you cannot measure it)
- GPU: 16 GB VRAM; **~14.4 GiB already committed** to the 26B model at 64k context ⇒ ~1.2–1.9 GiB free.
- CPU: Intel Core Ultra 9 HX; 32 GB system RAM.
- `bge-m3`: 568M params ⇒ ~2.2 GB fp16 / ~1.1 GB int8 / ~0.6 GB q8 GGUF (estimates — state them as such).

## Scope — files you own
`config/models.yaml` (`embed` profile only), `fmd/llm/**` (embedding path only), `fmd/er/similarity.py`, `fmd/score/insights.py` (embedding usage only), `tests/test_llm*.py`, `tests/test_er_similarity.py`, `docs/product-development-cycle/11-inference-runbook.md`, `docs/product-development-cycle/10-inference-and-model-selection.md` (one dated decision note).

**Forbidden:** `fmd/fetch/**`, `fmd/extract/**`, `fmd/verify/**`, `fmd/acquire/**`, `fmd/db/**`, `db/**`, `apps/**`, `infra/**`, `.env`, `scripts/**`.

## Acceptance criteria
1. **Decision documented, not assumed.** Compare (a) bge-m3 on GPU alongside the 26B, (b) bge-m3 on CPU, (c) a second small GGUF served by its own llama.cpp instance. Reason from the free-VRAM figure above and CPU class. **CPU is an acceptable and possibly correct answer** — argue it honestly; do not force GPU.
2. **No silent degradation.** If the embedding backend or model is absent, record `embeddings_unavailable` and use the documented lexical fallback, with the mode surfaced in output. Tests must prove the fallback is visible.
3. **Provider contract:** OpenAI-compatible `POST /v1/embeddings` (works for llama.cpp and hosted), plus a local `sentence-transformers` CPU backend. Selected by config/env, not code branches.
4. **Validation:** dimension mismatch rejected; batching supported; deterministic ordering; empty input handled.
5. **A measurement script** the operator can run locally (outputs VRAM before/after, tokens/s, and a recommendation) — you cannot run it, so document expected outputs rather than inventing numbers.
6. Docs: exact start command for the chosen backend, RAM/VRAM cost, and verification (`llm doctor --live` plus one embedding call).

## Constraints
- No weight downloads in tests; no live model calls; no network in tests.
- No secrets, binaries, `tmp/`, `runs/`, `.venv`, `node_modules`.
- Scoped diff. If a forbidden file is needed, STOP and record it in `known_gaps`.

## Run (must pass)
```
cd packages/pipeline
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## Output — three files at workspace root (UTC)
```
fmd-patch-w1-embed-provider-ff7d13e-<UTC>-01.patch
fmd-patch-w1-embed-provider-ff7d13e-<UTC>-01.json
fmd-patch-w1-embed-provider-ff7d13e-<UTC>-01.md
```
Patch: `git diff --binary --full-index --find-renames ff7d13e2192d22ae25f666f69f03b73a1d9f6063...HEAD > <patch>`
Manifest `format: fmd-patch-v1`, wave `w1`, workstream `embed-provider`, evals `not-run`, known_gaps.

Final reply (max 220 words): the decision with numbers, fallback semantics, output paths, known gaps.
