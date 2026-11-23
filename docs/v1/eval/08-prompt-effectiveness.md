# 08 — Prompt-effectiveness stream (product top-3 vs data baselines)

**Wave:** FMD-CLOUD-026 / prompt-effectiveness
**Status: implemented and wired as an offline product stream.** The canary has run on the
committed fixture + committed truth tables; no real-corpus run exists yet (that is LQ-2's
product, see "A few runs" below).
**Companions:** `01-eval-designs.md` §1.2 (proxy streams this builds on), `07-verdict-live-runbook.md`,
`docs/v1/LOCAL-QUEUE.md` (LQ-2, LQ-7).

## 1. What it measures

`fmd eval --eval-kind product --stream prompt-effectiveness` asks one question mechanically:
**is the prompt-driven pipeline producing a top-3 that the committed data baselines agree with?**

The product (a shortlist) is produced by the prompt-driven pipeline — admissions extracted via
the LLM, then plan ordering. The data-baseline strategies (`nirf-proxy`, `naac-proxy`) are pure
data: committed truth tables with verified provenance (`data/truth/nirf/nirf_2025_medical.yaml`,
`data/truth/naac/naac_grades.yaml`). The stream ranks the SAME shortlist both ways:

- **product top-3** — the shortlist's own order (`shortlist.ordered_keys()[:3]`), i.e. what the
  prompt-driven product put on top;
- **baseline top-3s** — each truth table's reference order restricted to the institutions the
  shortlist names (`run_nirf_proxy(... )["reference_order"][:3]`, same for NAAC).

The per-profile case reports `product_top3`, `nirf_proxy_top3`, `naac_proxy_top3`,
`overlap_nirf_top3`, `overlap_naac_top3` (`top_k_overlap` at k=3), and UNKNOWN tracking for the
product's #1 (`product_top_in_nirf`, `product_top_in_naac`, `product_top_unknown`).

If the two top-3s agree, the prompt-driven product is aligned with the data baseline (prompt
"working"). If not, the prompt/product needs work — and the comparison tells us by how much.
Institutions with no truth entry stay UNKNOWN (AGENTS.md invariant 5): they can never match a
baseline top-k and are counted in `product_top_unknown`, never coerced into a match.

## 2. How to run

```
cd packages/pipeline
uv run python -m fmd.cli eval --eval-kind product --stream prompt-effectiveness \
  --report-out reports/eval-prompt-effectiveness-<ts>.json
```

- Default input: the committed canary shortlist
  `packages/pipeline/tests/fixtures/eval_product/prompt_effectiveness_fixture.json` — its top two
  rows are the NIRF 2025 Medical truth table's top two institutions in truth order (AIMS rank 1,
  PGIMER rank 2) and its third row is deliberately in neither truth table, so the UNKNOWN path is
  exercised offline, deterministically, with no network and no model.
- To use a real shortlist: export the plan rows as a shortlist JSON (same schema as
  `tests/fixtures/eval_product/shortlist_fixture.json`: `profile_id`, `rows[{rank,
  institution_name, programme, state, ...}]`) and pass `--fixture-path <that file>`.
- `--truth-dir` overrides the truth table directory (default: the repo's `data/truth`).
- `--dry-run` renders the one-screen summary without writing a report.

## 3. Interpretation

The report aggregate carries:

| Field | Meaning |
| --- | --- |
| `mean_overlap_nirf` / `mean_overlap_naac` | mean size of `product_top3 ∩ baseline_top3` across profiles (0–3). Low vs NIRF = the prompt's ordering disagrees with the committed institutional ranking on the institutions both can see. NAAC is empty today, so `mean_overlap_naac` is structurally 0 until that table is ingested. |
| `n_profiles_overlap_ge2` | profiles where `max(overlap_nirf_top3, overlap_naac_top3) >= 2` — the product top-3 shares at least 2 institutions with at least one baseline top-3. |
| `prompting_working` | the fixed rule: `n_profiles_overlap_ge2 / n_profiles >= 0.8`. The rule and the 0.8 threshold are fixed (this doc is the record of them); they are not tunable per run. |
| `product_top_unknown_frac` | fraction of profiles whose #1 has no truth entry anywhere. High values mean the stream is measuring nothing for those profiles — fix coverage (more truth), not the threshold. |
| `n_profiles_top_in_nirf` / `n_profiles_top_in_naac` | how often the product's #1 exists in each truth table at all. |

Read `prompting_working` with the UNKNOWN fraction: a pass built on unknown tops is vacuous, and
the aggregate shows it instead of hiding it.

## 4. A few runs

This is a stream you re-run at fixed points; the **aggregate shifts** tell you whether prompting
moved:

1. **After LQ-2** (real crawl + extraction + scoring pass): export the real plan shortlist(s) as
   JSON, run the stream with `--fixture-path`. This is the first non-canary baseline; record the
   aggregate next to the report.
2. **After admissions changes (LQ-7's 26B classification pass feeding admissions)**: re-run on the
   same exported shortlists. If `mean_overlap_nirf` / `n_profiles_overlap_ge2` move, the
   admissions prompting moved the product — in either direction. A drop is a finding, not a
   failure of the stream.
3. **After product changes** (ordering, weights, filters): re-run; unexpected drift here is the
   signal that a change moved the top of the shortlist.

Keep one report per checkpoint under `reports/` and compare aggregates across reports; do not
re-baseline silently.

## 5. Honest limits

- **Overlap is not correctness.** The baselines rank only the institutions the shortlist names
  (institution-level, NIRF publishes no paramedical/MLT category, NAAC grades are institutional
  and 5-year-cyclic). A perfect overlap means "consistent with the committed data", not "the
  objectively best colleges for this profile".
- **NAAC is empty** (`truth_version: 0`, naac.gov.in unreachable at ingestion time), so the NAAC
  baseline contributes 0 overlap until a real fetch succeeds.
- **One shortlist per invocation** (the CLI composes one profile per run). Multi-profile
  aggregation arrives with real exported shortlists; the aggregate rule already handles N
  profiles.
- **The canary is constructed**, not a product run: it proves the harness wiring and the rule, and
  pins the UNKNOWN path offline.
