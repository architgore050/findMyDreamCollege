"""Crawl preflight: the effective UA contact must be real before public crawling.

The preflight warns loudly on stderr for a missing or placeholder contact. It
never invents an address (invariant 9), and remains a warning so dry-runs and
offline/local fixtures stay usable.
"""

from __future__ import annotations

import pytest

from fmd.cli import _warn_placeholder_user_agent

# Reserved test-only domain: this is a fixture, never an operator contact.
TEST_CONTACT = "crawler@example.invalid"


@pytest.fixture(autouse=True)
def clear_crawler_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FMD_CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("FMD_USER_AGENT", raising=False)


def test_missing_contact_warns_loudly(capsys: pytest.CaptureFixture[str]) -> None:
    _warn_placeholder_user_agent()

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "does not contain a contact address" in err
    assert "FMD_CONTACT_EMAIL=..." in err
    assert "FMD_USER_AGENT takes precedence over FMD_CONTACT_EMAIL" in err


@pytest.mark.parametrize(
    "placeholder",
    ["set-me@example.com", "me@example.com", "ops@example.org"],
)
def test_placeholder_contact_setting_warns_loudly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    placeholder: str,
) -> None:
    monkeypatch.setenv("FMD_CONTACT_EMAIL", placeholder)

    _warn_placeholder_user_agent()

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "placeholder contact" in err
    assert placeholder in err
    assert "FMD_CONTACT_EMAIL=..." in err


def test_explicit_placeholder_user_agent_warns_even_with_contact_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FMD_CONTACT_EMAIL", TEST_CONTACT)
    monkeypatch.setenv("FMD_USER_AGENT", "findMyDreamCollege/0.1 (+contact: set-me@example.com)")

    _warn_placeholder_user_agent()

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "set-me@example.com" in err
    assert "FMD_USER_AGENT" in err


def test_explicit_user_agent_without_contact_warns_even_with_contact_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FMD_CONTACT_EMAIL", TEST_CONTACT)
    monkeypatch.setenv("FMD_USER_AGENT", "custom-crawler/2.0")

    _warn_placeholder_user_agent()

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "does not contain a contact address" in err
    assert "FMD_USER_AGENT" in err


def test_configured_contact_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FMD_CONTACT_EMAIL", TEST_CONTACT)

    _warn_placeholder_user_agent()

    assert capsys.readouterr().err == ""


def test_explicit_user_agent_contact_takes_precedence_and_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FMD_CONTACT_EMAIL", "set-me@example.com")
    monkeypatch.setenv(
        "FMD_USER_AGENT", "findMyDreamCollege/0.1 (+contact: operator@example.invalid)"
    )

    _warn_placeholder_user_agent()

    assert capsys.readouterr().err == ""
