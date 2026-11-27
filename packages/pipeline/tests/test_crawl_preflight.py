"""Crawl preflight: the UA contact must be real before public crawling.

The shipped default `FMD_USER_AGENT` carries a placeholder contact; the
preflight warns loudly (stderr) so the operator sets a real address — it
never invents one (invariant 9), and it stays a warning so dry-runs and
tests remain usable.
"""

from __future__ import annotations

import pytest

from fmd.cli import _warn_placeholder_user_agent


def test_placeholder_contact_warns_loudly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FMD_USER_AGENT", "findMyDreamCollege/0.1 (+contact: set-me@example.com)")

    _warn_placeholder_user_agent()

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "set-me@example.com" in err
    assert "FMD_USER_AGENT" in err


def test_example_com_placeholder_warns_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FMD_USER_AGENT", "findMyDreamCollege/0.1 (+contact: me@example.com)")

    _warn_placeholder_user_agent()

    assert "WARNING" in capsys.readouterr().err


def test_real_contact_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(
        "FMD_USER_AGENT", "findMyDreamCollege/0.1 (+contact: ops@findmydreamcollege.org)"
    )

    _warn_placeholder_user_agent()

    assert capsys.readouterr().err == ""
