"""Runtime User-Agent construction from crawler contact settings."""

from __future__ import annotations

import pytest

from fmd import __version__
from fmd.settings import Settings

# Reserved test-only domain: this is a fixture, never an operator contact.
TEST_CONTACT = "crawler@example.invalid"


@pytest.fixture(autouse=True)
def clear_crawler_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FMD_CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("FMD_USER_AGENT", raising=False)


@pytest.mark.parametrize("contact", ["", "   "])
def test_default_user_agent_without_contact(monkeypatch: pytest.MonkeyPatch, contact: str) -> None:
    monkeypatch.setenv("FMD_CONTACT_EMAIL", contact)

    settings = Settings()

    assert settings.user_agent == f"findMyDreamCollege/{__version__}"


def test_default_user_agent_includes_trimmed_contact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMD_CONTACT_EMAIL", f"  {TEST_CONTACT}  ")

    settings = Settings()

    assert settings.user_agent == f"findMyDreamCollege/{__version__} (+contact: {TEST_CONTACT})"


@pytest.mark.parametrize("contact", ["", TEST_CONTACT])
def test_explicit_user_agent_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, contact: str
) -> None:
    override = "custom-crawler/2.0 (+contact: operator@example.invalid)"
    monkeypatch.setenv("FMD_CONTACT_EMAIL", contact)
    monkeypatch.setenv("FMD_USER_AGENT", override)

    settings = Settings()

    assert settings.user_agent == override
