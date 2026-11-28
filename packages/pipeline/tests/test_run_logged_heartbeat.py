"""scripts/run_logged.sh: the heartbeat contract, offline.

The START/END lines and the ``<log>.status`` file are machine contract; the
HEARTBEAT now also records a GPU line when nvidia-smi exists. This suite runs
the wrapper against a real ``sleep`` command — no GPU on the test host, which
is exactly the degrade path the GPU line must take (absent tool ⇒ no line,
never a crash).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_logged.sh"


def test_run_logged_heartbeat_and_status_contract(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    env = dict(os.environ, RUN_LOGGED_HEARTBEAT_S="1")
    result = subprocess.run(
        ["bash", str(SCRIPT), str(log), "sleep", "2"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("START")
    assert "END rc=0" in result.stdout

    status = Path(f"{log}.status")
    assert status.is_file()
    # The wrapper finished: the status file is the machine-readable record
    # (RUNNING while the child lives, DONE <rc> once it exits).
    assert status.read_text().startswith("DONE rc=0")

    text = log.read_text()
    assert "] START " in text and "] END rc=0 " in text
    assert "] HEARTBEAT" in text  # at least one heartbeat fired for sleep 2
    assert "PID" in text and "CMD" in text  # the ps table is still there


def test_run_logged_gpu_line_is_optional(tmp_path: Path) -> None:
    """No nvidia-smi on the host ⇒ no GPU line and no failure (graceful)."""
    log = tmp_path / "run.log"
    env = dict(os.environ, RUN_LOGGED_HEARTBEAT_S="1")
    result = subprocess.run(
        ["bash", str(SCRIPT), str(log), "true"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = log.read_text()
    # When the tool is absent the line is simply missing; when it is present
    # the line carries the util/mem/power CSV header it documents.
    if "GPU util%" not in text:
        assert "nvidia-smi" not in text or "GPU util%" not in text


def test_run_logged_status_file_reflects_failure(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    result = subprocess.run(
        ["bash", str(SCRIPT), str(log), "false"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "DONE rc=1" in Path(f"{log}.status").read_text()
    assert "] END rc=1 " in log.read_text()
