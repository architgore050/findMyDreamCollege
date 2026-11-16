"""Path and environment resolution shared by the CLI, API and tests.

The package runs in three layouts:

* repository checkout: ``packages/pipeline/fmd`` with ``config/`` and
  ``db/migrations`` two levels up;
* Docker image: ``/app/fmd`` with ``/config`` and ``/db/migrations``;
* tests: explicit paths injected by fixtures.

Resolution order is always: explicit argument -> ``FMD_*`` environment
variable -> nearby repository layout -> container default. Nothing about a
country, source or user lives here.
"""

from __future__ import annotations

import os
from pathlib import Path


class PathResolutionError(RuntimeError):
    """Raised when a required directory cannot be located."""


def _walk_up(start: Path, relative: str) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / relative).exists():
            return candidate / relative
    return None


def resolve_config_dir(explicit: str | Path | None = None) -> Path:
    """Locate ``config/`` containing ``countries.yaml``."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("FMD_CONFIG_DIR")
    if env:
        candidates.append(Path(env))
    found = _walk_up(Path(__file__).resolve(), "config/countries.yaml")
    if found is not None:
        candidates.append(found.parent)
    candidates.append(Path("/config"))

    for candidate in candidates:
        if (candidate / "countries.yaml").is_file():
            return candidate.resolve()
    searched = ", ".join(str(c) for c in candidates)
    raise PathResolutionError(
        "could not find the config registry (countries.yaml). "
        f"Searched: {searched}. Set FMD_CONFIG_DIR to the directory containing "
        "countries.yaml, sources.yaml, weights.yaml, models.yaml, exams.yaml and "
        "cost_rules.yaml."
    )


def resolve_migrations_dir(explicit: str | Path | None = None) -> Path:
    """Locate ``db/migrations`` containing numbered ``*.sql`` files."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("FMD_MIGRATIONS_DIR")
    if env:
        candidates.append(Path(env))
    found = _walk_up(Path(__file__).resolve(), "db/migrations")
    if found is not None:
        candidates.append(found)
    candidates.append(Path("/db/migrations"))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    searched = ", ".join(str(c) for c in candidates)
    raise PathResolutionError(
        "could not find database migrations. "
        f"Searched: {searched}. Set FMD_MIGRATIONS_DIR to the directory containing "
        "numbered *.sql migrations."
    )


def resolve_snapshot_dir(explicit: str | Path | None = None) -> Path:
    """Locate the content-addressed snapshot root.

    Unlike config/migrations this directory may not exist yet; callers create it.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("FMD_SNAPSHOT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    found = _walk_up(Path(__file__).resolve(), "snapshots")
    if found is not None:
        return found.resolve()
    return Path("/data/snapshots")
