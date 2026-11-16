from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from fmd.config import load_registry
from fmd.config.models import ModelsConfig, ProfileConfig, ProviderConfig
from fmd.db.repositories import LLMCallLog
from fmd.llm import (
    AnthropicProvider,
    BudgetExhausted,
    FixtureMissing,
    GeminiProvider,
    ModelRouter,
    OpenAICompatibleProvider,
    ProfileUnresolved,
    ProviderError,
    prompt_hash,
)
from fmd.llm.doctor import doctor_report

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


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

    result = router.call("bulk_extract", Answer, system, user, "sha256:document")

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


# --------------------------------------------------------------------- env switching


def test_every_profile_provider_and_model_are_env_switchable() -> None:
    env = {
        "FMD_EXTRACT_PROVIDER": "anthropic",
        "FMD_EXTRACT_MODEL": "claude-test",
        "FMD_HARD_PDF_PROVIDER": "gemini",
        # `hard_pdf` reads the shared FMD_API_MODEL: FMD_HARD_PDF_MODEL was
        # removed on 2026-11-16 (docs/.../10-...md §8).
        "FMD_API_MODEL": "gemini-test",
        "FMD_ADJUDICATE_PROVIDER": "api",
        "FMD_ADJUDICATE_MODEL": "gpt-test",
        "FMD_ENTITY_ADJUDICATE_MODEL": "entity-test",
        "FMD_NLP_MODEL": "nlp-test",
        "FMD_VERIFY_MODEL": "verify-test",
        "FMD_EMBED_MODEL": "embed-test",
        "FMD_EMBED_PROVIDER": "mock",
    }
    models = load_registry(CONFIG_DIR, env=env).models

    assert models.profiles["bulk_extract"].provider == "anthropic"
    assert models.profiles["bulk_extract"].model == "claude-test"
    assert models.profiles["hard_pdf"].provider == "gemini"
    assert models.profiles["hard_pdf"].model == "gemini-test"
    assert models.profiles["adjudicate"].model == "gpt-test"
    assert models.profiles["entity_adjudication"].model == "entity-test"
    assert models.profiles["nlp_rubric"].model == "nlp-test"
    assert models.profiles["verify"].model == "verify-test"
    assert models.profiles["embed"].provider == "mock"
    assert models.profiles["embed"].model == "embed-test"
    # OCR is an acquisition stage, not a router profile: `ocr_vlm` was retired on
    # 2026-11-16 and must not come back as a doctor-visible capability.
    assert set(models.profiles) == {
        "bulk_extract",
        "hard_pdf",
        "adjudicate",
        "entity_adjudication",
        "nlp_rubric",
        "verify",
        "embed",
    }
    assert "ocr_vlm" not in models.profiles


def _diagnostic_models() -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "providers": {
                "api": {
                    "kind": "openai_compatible",
                    "base_url": None,
                    "api_key": None,
                    "base_url_env": "FMD_API_LLM_URL",
                    "api_key_env": "FMD_API_LLM_KEY",
                    "requires_api_key": True,
                }
            },
            "profiles": {
                "bulk_extract": {
                    "provider": "api",
                    "model": None,
                    "model_env": "FMD_EXTRACT_MODEL",
                    "required": True,
                },
                "hard_pdf": {
                    "provider": "api",
                    "model": None,
                    "model_env": "FMD_API_MODEL",
                    "required": False,
                    "degrade_reason": "vision_unavailable",
                },
            },
            "budgets": {"daily_usd": 1, "per_run_usd": 1, "hard_stop": True, "cache": True},
        }
    )


def test_doctor_remediation_names_the_exact_missing_variables() -> None:
    report = doctor_report(_diagnostic_models())
    rows = {row["profile"]: row for row in report["profiles"]}

    bulk = "; ".join(rows["bulk_extract"]["remediation"])
    assert "FMD_EXTRACT_MODEL is empty" in bulk
    assert "FMD_API_LLM_URL is empty" in bulk
    assert "FMD_API_LLM_KEY is empty" in bulk
    assert "FMD_API_MODEL is empty" in "; ".join(rows["hard_pdf"]["remediation"])


def test_doctor_exit_semantics_follow_required_profiles() -> None:
    models = _diagnostic_models()
    assert doctor_report(models)["required_ok"] is False

    # Fix the required profile; the optional hard-PDF profile stays broken.
    models.providers["api"].base_url = "https://api.example/v1"
    models.providers["api"].api_key = "token"
    models.profiles["bulk_extract"].model = "text-model"
    report = doctor_report(models)
    assert report["required_ok"] is True
    assert report["profiles"][0]["usable"] is True
    by_profile = {row["profile"]: row for row in report["profiles"]}
    assert by_profile["hard_pdf"]["usable"] is False
    assert by_profile["hard_pdf"]["degrade_reason"] == "vision_unavailable"


def test_optional_profile_degrades_to_vision_unavailable(tmp_path: Path) -> None:
    models = _diagnostic_models()
    models.providers["api"].base_url = "https://api.example/v1"
    models.providers["api"].api_key = "token"
    models.profiles["bulk_extract"].model = "text-model"
    router = ModelRouter(models)
    with pytest.raises(ProviderError) as excinfo:
        router.call("hard_pdf", Answer, "s", "u", "ref")
    assert excinfo.value.reason == "vision_unavailable"
    assert "FMD_API_MODEL" in str(excinfo.value)
    assert "run continues without it" in str(excinfo.value)


def test_required_profile_failure_names_its_env_var() -> None:
    router = ModelRouter(_diagnostic_models())
    with pytest.raises(ProfileUnresolved) as excinfo:
        router.call("bulk_extract", Answer, "s", "u", "ref")
    message = str(excinfo.value)
    assert "FMD_EXTRACT_MODEL" in message
    assert "FMD_API_LLM_KEY" in message


# --------------------------------------------------------------------- provider shapes


def _anthropic_profile() -> ProfileConfig:
    return ProfileConfig(provider="anthropic", model="claude-test", guided_json=True)


def _gemini_profile() -> ProfileConfig:
    return ProfileConfig(provider="gemini", model="gemini-test", guided_json=True)


def test_anthropic_request_shape_uses_messages_and_a_tool() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [{"type": "tool_use", "name": "respond", "input": {"value": 42}}],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )

    provider = AnthropicProvider(
        "anthropic",
        ProviderConfig(
            kind="anthropic", base_url="https://anthropic.test/v1", api_key="secret-anthropic"
        ),
        transport=httpx.MockTransport(handler),
    )
    result = provider.complete(
        model="claude-test",
        system="system",
        user="user",
        schema=Answer,
        profile=_anthropic_profile(),
        prompt_sha256="x",
        purpose="bulk_extract",
    )

    assert captured["url"] == "https://anthropic.test/v1/messages"
    assert captured["headers"]["x-api-key"] == "secret-anthropic"
    payload = captured["payload"]
    assert payload["system"] == "system"
    assert payload["messages"] == [{"role": "user", "content": "user"}]
    assert payload["tools"][0]["input_schema"]["properties"]["value"]["type"] == "integer"
    assert payload["tool_choice"] == {"type": "tool", "name": "respond"}
    assert "secret-anthropic" not in json.dumps(payload)
    assert result.content == {"value": 42}
    assert (result.tokens_in, result.tokens_out) == (11, 7)


def test_gemini_request_shape_uses_generate_content_and_response_schema() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "modelVersion": "gemini-test",
                "candidates": [{"content": {"parts": [{"text": '{"value": 9}'}]}}],
                "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 4},
            },
        )

    provider = GeminiProvider(
        "gemini",
        ProviderConfig(
            kind="gemini", base_url="https://gemini.test/v1beta", api_key="secret-gemini"
        ),
        transport=httpx.MockTransport(handler),
    )
    result = provider.complete(
        model="gemini-test",
        system="system",
        user="user",
        schema=Answer,
        profile=_gemini_profile(),
        prompt_sha256="x",
        purpose="bulk_extract",
    )

    assert captured["url"] == "https://gemini.test/v1beta/models/gemini-test:generateContent"
    assert captured["headers"]["x-goog-api-key"] == "secret-gemini"
    assert "secret-gemini" not in captured["url"]
    payload = captured["payload"]
    assert payload["systemInstruction"]["parts"][0]["text"] == "system"
    generation = payload["generationConfig"]
    assert generation["responseMimeType"] == "application/json"
    assert generation["responseSchema"]["properties"]["value"]["type"] == "integer"
    assert "additionalProperties" not in json.dumps(generation)
    assert result.content == {"value": 9}
    assert (result.tokens_in, result.tokens_out) == (13, 4)


def test_guided_json_falls_back_to_json_instruct_and_validation() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "text-model",
                "choices": [{"message": {"content": '{"value": 5}'}}],
                "usage": {},
            },
        )

    models = ModelsConfig.model_validate(
        {
            "providers": {
                "local": {
                    "kind": "openai_compatible",
                    "base_url": "https://local.test/v1",
                    "api_key": "not-needed",
                    "supports_guided_json": False,
                }
            },
            "profiles": {
                "bulk_extract": {
                    "provider": "local",
                    "model": "text-model",
                    "guided_json": True,
                }
            },
            "budgets": {"daily_usd": 1, "per_run_usd": 1, "hard_stop": True, "cache": False},
        }
    )
    provider = OpenAICompatibleProvider(
        "local", models.providers["local"], transport=httpx.MockTransport(handler)
    )
    router = ModelRouter(models, providers={"local": provider})

    result = router.call("bulk_extract", Answer, "system", "user", "ref")

    assert result.value == 5
    payload = captured["payload"]
    assert "response_format" not in payload
    assert "JSON Schema" in payload["messages"][0]["content"]


# --------------------------------------------------------------------- secret hygiene


def test_doctor_never_prints_or_logs_secret_values(caplog) -> None:
    models = _diagnostic_models()
    models.providers["api"].base_url = "https://api.example/v1"
    models.providers["api"].api_key = "sk-do-not-print"
    models.profiles["bulk_extract"].model = "text-model"

    with caplog.at_level(logging.DEBUG):
        report = doctor_report(models, live=False)

    serialised = json.dumps(report)
    assert "sk-do-not-print" not in serialised
    assert "sk-do-not-print" not in caplog.text


def test_provider_http_errors_never_echo_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    provider = OpenAICompatibleProvider(
        "api",
        ProviderConfig(
            kind="openai_compatible", base_url="https://api.example/v1", api_key="sk-secret"
        ),
        transport=httpx.MockTransport(handler),
    )
    profile = ProfileConfig(provider="api", model="m", guided_json=True)
    with pytest.raises(ProviderError) as excinfo:
        provider.complete(
            model="m",
            system="s",
            user="u",
            schema=Answer,
            profile=profile,
            prompt_sha256="x",
            purpose="bulk_extract",
        )
    assert "sk-secret" not in str(excinfo.value)
