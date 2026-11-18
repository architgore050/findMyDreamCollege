# Patch workflow (cloud-agent protocol, adapted)

Token policy: **cloud agents do all coding, research, docs, tests and reviews.** The orchestrator plans, writes self-contained one-shot prompts, runs patch intake, applies/tests, opens PRs, requests review, and merges only when green. The orchestrator does not write production code.

Max **3 concurrent workstream agents**. Everything else queues.

## Units

- One agent = one workstream slice = one patch set = one PR.
- Base is always a recorded, immovable commit (`BASE_SHA`). Never a moving branch.
- Agents work in an isolated **git worktree** of the repo at `BASE_SHA`; never in the main tree.

## Patch set naming

```
fmd-patch-<wave>-<workstream>-<base7>-<UTC>-<seq>.patch
fmd-patch-<wave>-<workstream>-<base7>-<UTC>-<seq>.json
fmd-patch-<wave>-<workstream>-<base7>-<UTC>-<seq>.md
```

- `<base7>` = first 7 chars of `BASE_SHA`
- `<seq>` = 01, 02, …
- `.patch` = `git diff --binary --full-index --find-renames <BASE_SHA>...HEAD`
- Review patches: `fmd-patch-<wave>-<workstream>-review-<prhead7>-<UTC>-<seq>-r01.patch`

**Auxiliary artifacts are always prefixed with the patch stem**, never generic names:
`<patch-stem>-findings.json`, `<patch-stem>-notes.md`, `<patch-stem>-screenshot.png`.
A file called `findings.json` is ambiguous the moment a second agent runs — download order
must never be the only thing distinguishing two agents' output.

## Manifest (`.json`)

```json
{
  "format": "fmd-patch-v1",
  "patch_id": "...",
  "wave": "w1",
  "workstream": "crawler-targeting",
  "base_sha": "<full sha>",
  "branch": "feature/<patch_id>",
  "commit_message": "...",
  "summary": "one paragraph",
  "files_changed": ["..."],
  "tests_run": [{"command": "...", "result": "pass", "notes": ""}],
  "evals": {"run_id": "...", "result": "pass|not-run", "notes": ""},
  "known_gaps": ["..."],
  "created_at": "<UTC ISO8601>",
  "agent": "<agent id/name>"
}
```

Review manifests add: `review_of`, `target_branch`, `review_verdict` (`approve|approve-with-fixes|request-changes`), and `base_sha` = PR head.

## Intake (local, one subagent per batch)

1. Copy the three files to `tmp/patches/incoming/<patch_id>/`.
2. Validate: manifest `format` and `base_sha`; patch id matches the filename; no `tmp/`, `runs/`, `evals_runs/`, `.env`, secrets, `.venv`, `node_modules`, binaries.
3. Create a worktree at current main → branch `feature/<patch_id>`.
4. `git apply --check` then `git apply --index`. If it fails, **stop** — request a rebase patch; never hand-edit feature code.
5. Commit with the manifest message plus `Patch-Id:` and `Base-Sha:` trailers.
6. Run the manifest's tests **plus** repo hygiene: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`, `uv run python -m fmd.cli eval --golden data/golden`, `cd apps/web && pnpm lint && pnpm build && pnpm test`.
7. Push, open PR. Do not merge.
8. Empty patch + `approve` verdict is valid: verify CI, merge.

## Review loop

- Review the PR head, not main.
- Severity scale: blocker / major / minor / nit / question / praise.
- Reviewer runs the PR's gates, fixes real issues **on the PR branch**, emits a review patch based on the PR head.
- Apply locally with `git apply --3way`; rerun gates; push; merge only when green.

## Merge gate (all must hold)

- Patch applies cleanly at the recorded base.
- `pytest`, `ruff`, `eval` provenance 1.0, `pnpm build`/`test` green.
- No user-specific data, no secrets, no schema edit to an applied migration.
- Docs updated in the same patch (`AGENTS.md`, runbook, status).
- Review verdict `approve` or `approve-with-fixes` with fixes applied.

## Status reporting to the operator

Every update names: workstream · base SHA · PR/merge state · blocker (if any). No vague "in progress".
