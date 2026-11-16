# `fmd` pipeline package

Foundation + engine package for findMyDreamCollege. Read
`docs/product-development-cycle/04-phase-3-implementation-brief.md` and
`packages/pipeline/CONTRACT.md` before changing an interface.

```
uv sync --all-extras      # supported dev/CI setup: every workstream extra
uv run pytest -q
uv run ruff check .
```

**Never** run `uv sync --extra crawl` (or any single extra): `uv sync --extra`
*replaces* the environment, so it uninstalls extraction/ER dependencies and the
suite silently skips those tests. Use `--all-extras`; the `all` extra is the
explicit composition of `crawl` + `extract` + `er`.

## Local pipeline, end to end (v1: India)

```bash
# 1. Database (pgvector, published on :5433)
docker compose -f infra/docker-compose.yml up -d db
export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd

# 2. Environment + schema + registry
cd packages/pipeline
uv sync --all-extras
uv run fmd migrate
uv run fmd seed

# 3. Model profile sanity (presence only; no secrets printed)
uv run fmd llm-ping --json

# 4. Crawl  -> snapshots            [STUB: CLI exits 2, WS-B]
uv run fmd crawl
# 5. Extract -> extractions/claims  [STUB: CLI exits 2, WS-D]
uv run fmd extract

# 6. Score claims -> scores + insights (implemented, idempotent)
uv run fmd score --dry-run         # offline summary, writes nothing
uv run fmd score --limit 500

# 7. Evaluation                     [STUB: CLI exits 2, WS-F]
uv run fmd eval
```

`make` wrappers for the same stages run inside the `api` container:
`make migrate`, `make seed`, `make crawl`, `make extract`, `make score`,
`make eval`. `make test` now syncs `--all-extras` before pytest.

### What is actually runnable today

| Stage | Status |
|---|---|
| `migrate`, `seed`, `llm-ping` | real |
| `score` (+ `--dry-run`, `--force`, `--limit`, `--profile`) | real; writes `scores` + `insight_sets` |
| source adapters (UGC/NAAC/NABH/state counselling) | parsing real; fetching only through the fetch layer, not the CLI |
| `crawl` / `worker` | **stub** — exits 2 (`WS-B`) |
| `extract` | pipeline real (`fmd.extract.pipeline.run_extract`), CLI verb **stub** (`WS-D`) |
| `eval` | **stub** — `NotImplementedError` (`WS-F`); there is no provenance-coverage report yet |

`fmd score` against a freshly migrated, seeded-but-empty database is a valid
smoke test: it prints `fmd score: would write 0 ... programmes 0`.

### Offline extraction (until the CLI verb lands)

```python
from fmd.extract.pipeline import ExtractTarget, run_extract, build_router

router = build_router(mock=True)          # replay fixtures; no network, no spend
results = run_extract(
    [ExtractTarget(snapshot_id=1, entity_id=1, family="fee_structure")],
    database=database, store=store, router=router,
)
```

Tests must never hit the network or a real model. Use the mock/replay provider
and committed fixtures under `tests/fixtures/`.

The base Docker image installs no ML extras (brief §21); heavy workstream
dependencies are opt-in through the extras above.
