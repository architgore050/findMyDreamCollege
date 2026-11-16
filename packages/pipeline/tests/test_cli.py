from __future__ import annotations

import json
import uuid
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest

from fmd.cli import main
from fmd.db import Database, run_migrations

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture
def empty_database(pg_url: str):
    """A migrated, row-free schema so "empty DB" cannot inherit sibling-test rows."""
    schema = "test_cli_" + uuid.uuid4().hex[:10]
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    database = Database(pg_url, search_path=schema, min_size=1, max_size=2)
    run_migrations(database)
    try:
        yield database
    finally:
        database.close()
        with psycopg.connect(pg_url, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_unimplemented_commands_exit_non_zero(capsys) -> None:
    expected = {
        "eval": "WS-F",
    }
    for command, workstream in expected.items():
        assert main([command]) == 2
        captured = capsys.readouterr()
        assert f"not implemented ({workstream}," in captured.err
        assert "see brief §" in captured.err


def _dsn_with_schema(database) -> str:
    """Point the CLI at the fixture's throw-away schema via a conninfo option."""
    option = quote(f"-c search_path={database.search_path}")
    separator = "&" if "?" in database.url else "?"
    return f"{database.url}{separator}options={option}"


def test_score_dry_run_on_empty_database(empty_database, capsys) -> None:
    # The pg_url fixture skips loudly when DATABASE_URL is unset or unreachable.
    argv = [
        "--database-url",
        _dsn_with_schema(empty_database),
        "--config-dir",
        str(CONFIG_DIR),
        "score",
        "--dry-run",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "fmd score: would write 0" in out
    assert "programmes 0" in out


def test_llm_ping_reports_resolution(capsys) -> None:
    assert main(["--config-dir", str(CONFIG_DIR), "llm-ping", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    profiles = {row["profile"]: row for row in payload["profiles"]}
    assert profiles["bulk_extract"]["resolved"] is True
    assert "key_present" in profiles["bulk_extract"]
    assert payload["budgets"]["hard_stop"] is True
    # No secret material may appear in the report.
    assert "api_key" not in json.dumps(payload)


def test_llm_ping_table_never_prints_key_values(capsys) -> None:
    assert main(["--config-dir", str(CONFIG_DIR), "llm-ping"]) == 0
    out = capsys.readouterr().out
    assert "profiles in" in out
    assert "key=" in out
    assert "hard_stop=True" in out
