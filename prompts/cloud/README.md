# Cloud-agent prompt pack — naming, hand-off, and intake

## Who does what

- **Cloud agents (elsewhere):** all coding, research, docs, tests, review. You run these prompts; I never launch them.
- **Local agents (here):** only patch intake, applying patches, running local gates, local-only tasks (DB queries, pipeline runs, GPU work).

## Naming scheme

| Thing | Pattern | Example |
|---|---|---|
| Prompt ID | `FMD-CLOUD-<seq>-<workstream>` | `FMD-CLOUD-001-crawler-targeting` |
| Wave | `w1`, `w2`, … | `w1` |
| Branch (agent) | `feature/fmd-<wave>-<workstream>-<base7>-<utc>-<seq>` | `feature/fmd-w1-crawler-targeting-ff7d13e-20261118T2100-01` |
| Patch set | `fmd-patch-<wave>-<workstream>-<base7>-<UTC>-<seq>.{patch,json,md}` | `fmd-patch-w1-crawler-targeting-ff7d13e-20261118T2100-01.patch` |
| Review patch | `fmd-patch-<wave>-<workstream>-review-<prhead7>-<UTC>-<seq>-r01.{patch,json,md}` | — |

`<base7>` = first 7 chars of the recorded base SHA. Currently **`ff7d13e`** (full: `ff7d13e2192d22ae25f666f69f03b73a1d9f6063`).

## Hand-off

1. Copy one prompt file to a cloud agent (full text, no edits — it is self-contained).
2. The agent clones `https://github.com/architgore050/findMyDreamCollege.git`, checks out the base SHA, works on its branch, and writes the three patch files at its workspace root.
3. Copy the three files into `~/Downloads` (or anywhere you like). Any extra artifact
   MUST be named `<patch-stem>-<what>.json/.md` — never `findings.json`, because a second
   agent's file would collide and only download order would distinguish them.
4. Tell me: **"FMD-CLOUD-00N finished"**.
5. I run one local intake agent for the batch, which:
   - finds the files by the patch stem (searches `~/Downloads`, `~/Downloads/*/`, and `tmp/patches/incoming/`);
   - validates the manifest (`format: fmd-patch-v1`, matching base SHA, no secrets/binaries/`tmp/`);
   - creates a fresh worktree at main, `git apply --check` then `git apply --index`;
   - commits with `Patch-Id:` / `Base-Sha:` trailers;
   - runs the manifest tests plus repo gates: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run python -m fmd.cli eval --golden data/golden`, `cd apps/web && pnpm lint && pnpm build && pnpm test`;
   - pushes a branch and reports PASS/FAIL.
6. I do **not** merge red. A conflicting patch goes back to you for a rebase prompt (new prompt ID suffix `-r02`).

## Rules I enforce

- One prompt = one workstream = one patch set.
- Base SHA is frozen per wave; a stale patch gets `git apply --3way`, and if that conflicts, it is rejected.
- No secret, `.env`, `.venv`, `node_modules`, `tmp/`, `runs/`, or downloaded binary in a patch.
- Docs updated in the same patch as the code.
- Max 3 prompt packs in flight at once.

## Current pack

| Prompt ID | Workstream | Depends on |
|---|---|---|
| `FMD-CLOUD-001-crawler-targeting` | Evidence targeting: seat/fee/prospectus prioritisation | — |
| `FMD-CLOUD-002-embed-provider` | Real embeddings + explicit unavailability | — |
| `FMD-CLOUD-003-e2e-acceptance` | `scripts/acceptance.sh` + integration fixes | 001, 002 (soft) |
