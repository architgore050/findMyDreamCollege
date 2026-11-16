from __future__ import annotations

import json
from pathlib import Path

from fmd.cli import main

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def test_unimplemented_commands_exit_non_zero(capsys) -> None:
    expected = {
        "crawl": "WS-B",
        "extract": "WS-D",
        "score": "WS-F",
        "eval": "WS-F",
        "worker": "WS-B",
    }
    for command, workstream in expected.items():
        assert main([command]) == 2
        captured = capsys.readouterr()
        assert f"not implemented ({workstream}," in captured.err
        assert "see brief §" in captured.err


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
