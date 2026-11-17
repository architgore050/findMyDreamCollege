"""Hard-PDF tier semantics: an unconfigured optional tier degrades cleanly.

The extraction run recorded `unresolved_provider` for three documents. That
reason means *misconfiguration* (a provider name that is not declared), not
"this optional tier is unavailable". This test pins the distinction at the
registry/router boundary, offline.
"""

from __future__ import annotations

import os

import pytest

from fmd.config import load_registry
from fmd.extract.schemas.curriculum import Curriculum
from fmd.llm.providers import ProviderError
from fmd.llm.router import ModelRouter

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "config"))


def _models(env: dict[str, str]):
    return load_registry(CONFIG_DIR, env=env).models


def test_hard_pdf_is_declared_optional_with_a_degrade_reason() -> None:
    profile = _models({}).profiles["hard_pdf"]
    assert profile.required is False
    assert profile.degrade_reason == "vision_unavailable"
    assert profile.provider == "api"


def test_unconfigured_api_yields_vision_unavailable_not_unresolved_provider() -> None:
    models = _models({})  # no base URL, no key, no model
    router = ModelRouter(models)
    with pytest.raises(ProviderError) as excinfo:
        router.call("hard_pdf", Curriculum, "system", "user", "ref")
    assert excinfo.value.reason == "vision_unavailable"


def test_missing_model_only_also_degrades_cleanly() -> None:
    models = _models({"FMD_API_LLM_URL": "https://example.invalid/v1", "FMD_API_LLM_KEY": "k"})
    router = ModelRouter(models)
    with pytest.raises(ProviderError) as excinfo:
        router.call("hard_pdf", Curriculum, "system", "user", "ref")
    assert excinfo.value.reason == "vision_unavailable"
