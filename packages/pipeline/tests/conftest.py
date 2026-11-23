from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import psycopg
import pytest

from fmd.config import load_registry
from fmd.db import Database, run_migrations

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(autouse=True)
def _scrape_failures_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the scrape-failure log to a tmp dir for every test.

    The log's default directory is committed (`<repo>/data/scrape-failures`);
    without this fixture any ladder-driven test failure would append test noise
    to the evidence log. Tests that need a specific directory set
    ``FMD_SCRAPE_FAILURES_DIR`` themselves (monkeypatch overrides this fixture).
    """
    monkeypatch.setenv("FMD_SCRAPE_FAILURES_DIR", str(tmp_path / "scrape-failures"))


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A mutable copy of the committed registry (never edit config/ in tests)."""
    target = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, target)
    return target


@pytest.fixture(scope="session")
def registry():
    return load_registry(CONFIG_DIR, env={})


@pytest.fixture(scope="session")
def pg_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip(
            "DATABASE_URL is not set; DB repository tests skipped. "
            "Run: docker compose -f infra/docker-compose.yml up -d db && "
            "export DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd"
        )
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        pytest.skip(f"DATABASE_URL is set but Postgres is unreachable: {exc}")
    return url


@pytest.fixture(scope="session")
def pg_database(pg_url: str):
    schema = "test_" + uuid.uuid4().hex[:12]
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    database = Database(pg_url, search_path=schema, min_size=1, max_size=4)
    run_migrations(database)
    try:
        yield database
    finally:
        database.close()
        with psycopg.connect(pg_url, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
