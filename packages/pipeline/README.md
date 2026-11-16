# `fmd` pipeline package

Foundation package for findMyDreamCollege (workstream A). See
`docs/product-development-cycle/04-phase-3-implementation-brief.md`.

```
uv sync
uv run pytest -q
uv run ruff check .
```

The package is installed into the base Docker image with `uv pip install --system`
(no ML extras). Workstream modules (`fetch`, `frontier`, `discover`, `adapters`,
`extract`, `er`, `score`, `filters`, `cost`, `eval`) are explicit stubs in this
workstream and raise `NotImplementedError`.
