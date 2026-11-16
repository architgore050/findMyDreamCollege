from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from fmd.config.models import ModelsConfig
from fmd.db.repositories import LLMCallLog
from fmd.llm import (
    BudgetExhausted,
    FixtureMissing,
    ModelRouter,
    ProfileUnresolved,
    ProviderError,
    prompt_hash,
)


class Answer(BaseModel):
    value: int
    quote: str | None = None


class CollectingSink:
    def __init__(self) -> None:
        self.calls: list[LLMCallLog] = []

    def record(self, call: LLMCallLog) -> None:
        self.calls.append(call)


def _models(
    fixtures_dir: Path,
    *,
    pricing: dict[str, float] | None = None,
    budgets: dict[str, Any] | None = None,
    fallbacks: list[dict[str, Any]] | None = None,
) -> ModelsConfig:
    profile: dict[str, Any] = {
        "provider": "mock",
        "model": "recorded-model",
        "temperature": 0,
        "guided_json": True,
        "max_tokens": 512,
    }
    if pricing is not None:
        profile["pricing"] = pricing
    return ModelsConfig.model_validate(
        {
            "providers": {
                "mock": {"kind": "replay", "fixtures_dir": str(fixtures_dir)},
            },
            "profiles": {"bulk_extract": profile},
            "budgets": budgets
            or {
                "daily_usd": 5,
                "per_run_usd": 2,
                "hard_stop": True,
                "cache": True,
            },
            "fallbacks": fallbacks or [],
        }
    )


def _write_fixture(fixtures_dir: Path, system: str, user: str, content: dict) -> str:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    digest = prompt_hash(system, user)
    (fixtures_dir / f"{digest}.json").write_text(
        json.dumps(
            {"content": content, "tokens_in": 120, "tokens_out": 30, "model": "recorded-model"}
        ),
        encoding="utf-8",
    )
    return digest


def test_mock_router_replays_fixture_and_audits(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    sink = CollectingSink()
    router = ModelRouter(_models(fixtures), sink=sink)
    system, user = "extract claims", "document text"
    _write_fixture(fixtures, system, user, {"value": 1800, "quote": "1,800 hours"})

    result = router.call(
        "bulk_extract", Answer, system, user, "sha256:document"
    )

    assert isinstance(result, Answer)
    assert result.value == 1800
    assert len(sink.calls) == 1
    call = sink.calls[0]
    assert call.provider == "mock"
    assert call.model == "recorded-model"
    assert call.cache_hit is False
    assert call.tokens_in == 120
    assert call.tokens_out == 30
    assert call.latency_ms is not None
    assert call.cost_usd == 0.0
    assert call.prompt_sha256 == prompt_hash(system, user)


def test_cache_hit_is_audited_and_avoids_second_provider_call(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    sink = CollectingSink()
    router = ModelRouter(_models(fixtures), sink=sink)
    system, user = "s", "u"
    _write_fixture(fixtures, system, user, {"value": 1})

    router.call("bulk_extract", Answer, system, user, "ref")
    router.call("bulk_extract", Answer, system, user, "ref")

    assert len(sink.calls) == 2
    assert [call.cache_hit for call in sink.calls] == [False, True]
    assert sink.calls[1].cost_usd == 0.0


def test_missing_fixture_error_is_actionable(tmp_path: Path) -> None:
    router = ModelRouter(_models(tmp_path / "fixtures"))
    with pytest.raises(FixtureMissing) as excinfo:
        router.call("bulk_extract", Answer, "s", "u", "ref")
    message = str(excinfo.value)
    assert prompt_hash("s", "u") in message
    assert "Expected a fixture at" in message


def test_unresolved_profile_fails_before_any_call(tmp_path: Path) -> None:
    models = _models(tmp_path / "fixtures")
    models.profiles["bulk_extract"].model = None
    router = ModelRouter(models)
    with pytest.raises(ProfileUnresolved):
        router.call("bulk_extract", Answer, "s", "u", "ref")


def test_hard_budget_stop(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    system, user = "s", "u"
    _write_fixture(fixtures, system, user, {"value": 1})
    models = _models(
        fixtures,
        pricing={"input_per_mtok": 1_000_000.0, "output_per_mtok": 0.0},
        budgets={"daily_usd": 5, "per_run_usd": 0.5, "hard_stop": True, "cache": True},
    )
    router = ModelRouter(models)
    router.call("bulk_extract", Answer, system, user, "first")
    with pytest.raises(BudgetExhausted) as excinfo:
        # Different input_ref so the cache cannot satisfy this call.
        router.call("bulk_extract", Answer, system, user, "second")
    assert "hard budget stop" in str(excinfo.value)


def test_fallback_chain_uses_next_provider(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    system, user = "s", "u"
    _write_fixture(fixtures, system, user, {"value": 7})
    models = ModelsConfig.model_validate(
        {
            "providers": {
                "api": {
                    "kind": "openai_compatible",
                    "base_url": "http://unused.invalid/v1",
                    "api_key": "x",
                },
                "mock": {"kind": "replay", "fixtures_dir": str(fixtures)},
            },
            "profiles": {
                "bulk_extract": {
                    "provider": "api",
                    "model": "remote-model",
                    "guided_json": True,
                }
            },
            "budgets": {
                "daily_usd": 5,
                "per_run_usd": 2,
                "hard_stop": True,
                "cache": False,
            },
            "fallbacks": [
                {"from": "api", "to": "mock", "on": ["connection_error"]},
            ],
        }
    )

    class FailingProvider:
        name = "api"
        kind = "openai_compatible"

        def complete(self, **kwargs: Any):
            raise ProviderError("connection_error", "simulated network refusal")

    from fmd.llm.providers import MockProvider

    mock = MockProvider("mock", models.providers["mock"])
    router = ModelRouter(models, providers={"api": FailingProvider(), "mock": mock})
    result = router.call("bulk_extract", Answer, system, user, "ref")
    assert result.value == 7


def test_ping_never_prints_secrets(tmp_path: Path) -> None:
    models = ModelsConfig.model_validate(
        {
            "providers": {
                "api": {
                    "kind": "openai_compatible",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "super-secret-key",
                },
                "mock": {"kind": "replay", "fixtures_dir": str(tmp_path)},
            },
            "profiles": {
                "hard_pdf": {"provider": "api", "model": "remote-model"},
                "bulk_extract": {"provider": "mock", "model": "recorded"},
            },
            "budgets": {
                "daily_usd": 1,
                "per_run_usd": 1,
                "hard_stop": True,
                "cache": True,
            },
        }
    )
    report = ModelRouter(models).ping()
    serialised = json.dumps(report)
    assert "super-secret-key" not in serialised
    by_profile = {row["profile"]: row for row in report}
    assert by_profile["hard_pdf"]["resolved"] is True
    assert by_profile["hard_pdf"]["key_present"] is True
    assert by_profile["bulk_extract"]["key_present"] is None
