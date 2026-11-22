#!/usr/bin/env bash
# scripts/batch_extract.sh — thin wrapper over `fmd extract --batched` (W10).
#
# The batched orchestrator is the CLI; this wrapper only pins the working
# directory and forwards every argument, so an operator never has to remember
# that the pipeline lives in packages/pipeline:
#
#   scripts/batch_extract.sh                        # acquire.yaml batch defaults
#   scripts/batch_extract.sh --batch-pages 100      # pages per cycle
#   scripts/batch_extract.sh --limit 20             # cap the queue
#   scripts/batch_extract.sh --device cpu           # override the CUDA default
#   scripts/batch_extract.sh --dry-run              # plan only, writes nothing
#   scripts/batch_extract.sh --keep-server          # don't stop the server at the end
#   scripts/batch_extract.sh --from-cache           # extract only cached parses
#
# Per cycle (runbook §6): ensure server stopped -> acquire one page-batch on
# CUDA (zero LLM calls) -> ensure server started and /health ready -> extract
# that batch from cache -> repeat. The server is stopped on every exit path
# unless --keep-server. Sizing is evaluated only at document boundaries: the
# document that crosses --batch-pages always finishes; no document is ever
# truncated or split across cycles.
#
# Requires the snapshots/DB to be reachable (DATABASE_URL) and, for a real
# run, FMD_LOCAL_LLM_URL. A dry run needs neither a model server nor writes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/packages/pipeline"

exec uv run python -m fmd.cli extract --batched "$@"
