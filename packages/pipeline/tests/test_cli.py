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


def test_llm_doctor_reports_resolution(capsys) -> None:
    assert main(["--config-dir", str(CONFIG_DIR), "llm", "doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    profiles = {row["profile"]: row for row in payload["profiles"]}
    assert profiles["bulk_extract"]["usable"] is True
    assert profiles["bulk_extract"]["model"] == "gemma-4-26b-a4b-gguf"
    assert "key_present" in profiles["bulk_extract"]
    assert payload["budgets"]["hard_stop"] is True
    assert payload["required_ok"] is True
    # Only the env-var name is published, never a key value.
    assert '"api_key":' not in json.dumps(payload)
    assert "secret" not in json.dumps(payload).lower()


def test_llm_doctor_table_never_prints_key_values(capsys) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("FMD_API_LLM_KEY", "sk-must-not-appear")
    try:
        assert main(["--config-dir", str(CONFIG_DIR), "llm", "doctor"]) == 0
    finally:
        monkeypatch.undo()
    out = capsys.readouterr().out
    assert "model profiles in" in out
    assert "BASE_URL" in out and "MODEL" in out
    assert "key_present" not in out
    assert "sk-must-not-appear" not in out
    assert "hard_stop=True" in out


def test_llm_doctor_exits_non_zero_when_required_profile_is_unusable(
    capsys, monkeypatch
) -> None:
    monkeypatch.setenv("FMD_EXTRACT_MODEL", "")
    assert main(["--config-dir", str(CONFIG_DIR), "llm", "doctor"]) == 1
    out = capsys.readouterr().out
    assert "FMD_EXTRACT_MODEL is empty" in out


def test_llm_ping_remains_a_doctor_alias(capsys) -> None:
    assert main(["--config-dir", str(CONFIG_DIR), "llm-ping", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profiles"]


def test_eval_with_verify_fixtures_surfaces_verification_metrics(capsys) -> None:
    repo_root = CONFIG_DIR.parent
    argv = [
        "--config-dir",
        str(CONFIG_DIR),
        "eval",
        "--golden",
        str(repo_root / "data" / "golden"),
        "--verify-fixtures",
        str(
            Path(__file__).parent
            / "fixtures"
            / "verify"
            / "fabrication_v1"
        ),
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "verification: verified_rate 1.0" in out
    assert "fabrication_caught 7/7" in out
    assert "correct_rejected 0" in out
    # The provenance gate is unchanged by the verification block.
    assert "gate provenance>=1.0: PASS" in out