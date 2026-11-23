# FMD-CLOUD-026-prompt-effectiveness · product vs data baseline · wave w13 · prompt-effectiveness

~~~~
You are the one-shot cloud coding agent building the "prompt-effectiveness" eval stream.

REPO: https://github.com/architgore050/findMyDreamCollege.git
BASE SHA: f75bc11ad5316e92e23cd547d963462c80011105
Push to `v1/prompt-effectiveness` immediately; commit + push after every logical step.
All output (code, comments, docs, commit messages, PR text, report) — English-only.

BRANCHING
1. `git switch -c v1/prompt-effectiveness` from main at the base SHA; verify `git rev-parse HEAD`.
2. `git push origin v1/prompt-effectiveness` before anything else.
3. Subagents (allowed) may branch off it. End: PR → main.

READ FIRST
- AGENTS.md (invariants; esp. #3 UNKNOWN is a value, #2 numbers are computed by harness)
- docs/v1/eval/01-eval-designs.md, 02-findings-viable.md, 00-research-summary.md (context)
- packages/pipeline/fmd/eval/product.py (read in full — reuse run_nirf_proxy, run_naac_proxy,
  run_proxy_stream, load_shortlist, load_truth_for_stream, top_k_overlap, Shortlist)
- packages/pipeline/fmd/eval/truth.py (NirfTruthFile/NaacTruthFile.by_key/ranked_keys)
- fmd/cli.py `_cmd_eval_product` (stream dispatch + parser)
- packages/pipeline/tests/test_eval_product_cli.py + test_eval_product.py (style)
- data/truth/nirf/nirf_2025_medical.yaml (pick real institution names that exist — the
  stream's hand-computed expectations in the tests must match the truth files)

DESIGN — "judge whether prompting is working": the product (a shortlist) is produced by
the prompt-driven pipeline (admissions via LLM + plan ordering). The data-baseline
strategies (nirf-proxy, naac-proxy) are pure data (committed truth tables). So:
**prompt-effectiveness measures the overlap between the PRODUCT's own top-3 and the
data-baseline top-3, on the SAME shortlist.** If they agree, the prompt-driven product
is aligned with the data baseline (prompt "working"). If not, the prompt/product needs
work — and the comparison tells us by how much.

1. `fmd/eval/product.py` — new function `run_prompt_effectiveness(shortlist, nirf_truth,
   naac_truth) -> dict`:
   - `product_order = shortlist.ordered_keys()`; `product_top3 = product_order[:3]`.
   - `nirf_case = run_nirf_proxy(shortlist, nirf_truth)`; `naac_case = run_naac_proxy(shortlist, naac_truth)`.
   - `nirf_proxy_top3 = nirf_case["reference_order"][:3]`; `naac_proxy_top3 = naac_case["reference_order"][:3]`.
   - `overlap_nirf_top3 = top_k_overlap(product_top3, nirf_proxy_top3, 3)`;
     `overlap_naac_top3 = top_k_overlap(product_top3, naac_proxy_top3, 3)`
     (`top_k_overlap(a, b, k) = |set(a[:k]) ∩ set(b[:k])|` — read truth.py to confirm).
   - Per-profile: `product_top = product_top3[0]`; `product_top_in_nirf = product_top in
     nirf_case`-style via `nirf_truth.by_key()` (use `nirf_truth.by_key().get(product_top)`;
     product_top may be None if shortlist has no rows — Shortlist requires ≥1 row, so ok).
     `product_top_unknown = (product_top is None) or (product_top not in nirf_by_key
     and product_top not in naac_by_key)`.
   - Cases (one per profile): `{"profile_id", "shortlist_rows", "product_top3",
     "overlap_nirf_top3", "overlap_naac_top3", "nirf_proxy_top3", "naac_proxy_top3",
     "product_top_in_nirf", "product_top_in_naac", "product_top_unknown"}`.
   - Aggregate: `{"n_profiles", "mean_overlap_nirf", "mean_overlap_naac",
     "n_profiles_overlap_ge2", "prompting_working", "product_top_unknown_frac",
     "n_profiles_top_in_nirf", "n_profiles_top_in_naac"}` where
     `prompting_working = n_profiles_overlap_ge2 / n_profiles >= 0.8` and
     `n_profiles_overlap_ge2 = #(max(overlap_nirf, overlap_naac) >= 2)`.
     (Rule is fixed; doc 07-equivalent is in docs/v1/eval/08.)
2. `fmd/cli.py`:
   - `_cmd_eval_product`: add `"prompt-effectiveness"` to the stream branch:
     ```
     elif stream == "prompt-effectiveness":
         shortlist = load_shortlist(_fixture_default("prompt_effectiveness_fixture.json"))
         truth_dir = Path(args.truth_dir) if args.truth_dir else repo_root / "data" / "truth"
         nirf_truth = load_truth_for_stream("nirf-proxy", truth_dir, root=repo_root)
         naac_truth = load_truth_for_stream("naac-proxy", truth_dir, root=repo_root)
         cases = [run_prompt_effectiveness(shortlist, nirf_truth, naac_truth)]
         report = build_product_report("prompt-effectiveness", cases)
         # reuse the existing write/render path used by other product streams
     ```
     (Read the current branch of `_cmd_eval_product` to wire exactly — lines 1277-1360 — the
     agent integrates with the existing code structure.)
   - Parser `--stream` choices: add `"prompt-effectiveness"` + help text line.
3. Fixtures:
   - `packages/pipeline/tests/fixtures/eval_product/prompt_effectiveness_fixture.json`:
     a committed canary shortlist. AGENT: read `data/truth/nirf/nirf_2025_medical.yaml` NOW
     (or via load) and pick real institution names that exist in the NIRF truth
     (e.g. `All India Institute of Medical Sciences, Delhi` and
     `Post Graduate Institute of Medical Education and Research` — verify their
     `institution_key`s via `normalise_institution_key` and the truth `by_key()`), put the
     first 2 rows as those (order: the truth's top: AIMS rank 1, PGIMER rank 2) and a
     third row a name that is in NEITHER truth (UNKNOWN path). All rows:
     `{"rank": 1|2|3, "institution_name": <name>, "programme": "B.Sc. Medical Laboratory
     Technology", "state": <state>}` — mirror the shape of shortlist_fixture.json.
   - Tests (offline) `packages/pipeline/tests/test_eval_prompt_effectiveness.py`:
     - `test_run_prompt_effectiveness_expected_overlaps`: load the committed fixture +
       truth (NIRF + NAAC via `load_truth_for_stream`); run `run_prompt_effectiveness`;
       assert `product_top3 == [key(AIMS), key(PGIMER), key(CityName)]`;
       `overlap_nirf_top3 == 2` (hand-computed: product top-3 ∩ NIRF reference top-3 =
       {AIMS, PGIMER}); `overlap_naac_top3 == 0` (NAAC table empty in the committed data);
       `product_top_in_nirf is True`, `product_top_in_naac is False`;
       aggregate `n_profiles == 1`, `mean_overlap_nirf == 2`, `n_profiles_overlap_ge2 == 1`,
       `prompting_working is True`.
       (Agent verifies the NIRF reference order: NIRF truth has 50 entries ranked by rank;
       the two shortlist institutions must both be in it, and no third row can be in it
       unless real — keep the 3rd row a non-truth name.)
     - `test_product_top_unknown_when_not_in_truth`: synthetic shortlist (1 row, name not
       in any truth) → `product_top_unknown is True`, `overlap_nirf_top3 == 0`,
       `prompting_working is False` (n_profiles_overlap_ge2 = 0).
     - `test_run_is_deterministic`: run twice, compare aggregates.
     - `test_cli_runs_stream`: `main(["eval","--eval-kind","product","--stream",
       "prompt-effectiveness","--report-out", str(tmp_path/"r.json")]) == 0` and the report
       has the stream name + aggregate; `main([...,"--dry-run"]) == 0`.
4. Docs: `docs/v1/eval/08-prompt-effectiveness.md`:
   - What it measures; why it judges prompting (product top produced by prompt pipeline vs
     data baselines); default input (committed canary fixture); how to run
     (`fmd eval --eval-kind product --stream prompt-effectiveness --report-out reports/eval-prompt-effectiveness-<ts>.json`);
   - Interpretation: mean_overlap_nirf/naac, n_profiles_overlap_ge2, prompting_working;
     the rule (max overlap ≥ 2 in ≥ 80% of profiles) and threshold;
   - "A few runs": re-run after LQ-2 (real corpus), after admissions changes (LQ-7),
     after product changes — the aggregate shifts tell you if prompting moved;
   - To use a real shortlist: export the plan rows as a shortlist JSON (same schema as
     tests/fixtures/eval_product/shortlist_fixture.json) and pass `--fixture-path`.
5. Gates: `uv sync --all-extras; uv run ruff check .; uv run pytest -q | tail -2 (report
   counts); uv run python -m fmd.cli eval --golden ../../data/golden (provenance 1.0)`.
6. Report (chat ≤ 400 words): what was built, test names + counts, gates, PR URL, top 3
   honest gaps.
~~~~
