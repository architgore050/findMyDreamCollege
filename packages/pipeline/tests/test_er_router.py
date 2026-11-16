"""Model router: purpose-only routing_rules and the entity_adjudication profile."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from fmd.config.models import ModelsConfig
from fmd.llm import MockProvider, ModelRouter, prompt_hash


class Verdict(BaseModel):
    ok: bool


def _models(
    tmp_path: Path,
    *,
    profiles: dict[str, dict],
    routing_rules: list[dict] | None = None,
) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "providers": {"mock": {"kind": "replay", "fixtures_dir": str(tmp_path)}},
            "profiles": profiles,
            "budgets": {"daily_usd": 0, "per_run_usd": 0, "hard_stop": True, "cache": True},
            "routing_rules": routing_rules or [],
            "fallbacks": [],
        }
    )


def _profile(model: str) -> dict:
    return {
        "provider": "mock",
        "model": model,
        "temperature": 0,
        "guided_json": True,
        "max_tokens": 256,
    }


def _record(tmp_path: Path, system: str, user: str, model: str) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    digest = prompt_hash(system, user)
    payload = {"content": {"ok": True}, "model": model}
    (tmp_path / f"{digest}.json").write_text(json.dumps(payload), encoding="utf-8")


def _router(models: ModelsConfig) -> ModelRouter:
    return ModelRouter(models, providers={"mock": MockProvider("mock", models.providers["mock"])})


def test_purpose_only_routing_rule_is_honoured(tmp_path: Path) -> None:
    # No dedicated `entity_adjudication` profile: the rule must route the call
    # to `adjudicate` instead of raising ProfileUnresolved.
    models = _models(
        tmp_path,
        profiles={"adjudicate": _profile("adjudicator-model")},
        routing_rules=[{"if": {"purpose": "entity_adjudication"}, "use": "adjudicate"}],
    )
    router = _router(models)
    system, user = "resolve names", "pair list"
    _record(tmp_path, system, user, "adjudicator-model")

    result = router.call("entity_adjudication", Verdict, system, user, "pairs:1")
    assert isinstance(result, Verdict)
    assert result.ok is True
    assert router._profile("entity_adjudication").model == "adjudicator-model"


def test_dedicated_profile_wins_over_a_routing_rule(tmp_path: Path) -> None:
    models = _models(
        tmp_path,
        profiles={
            "entity_adjudication": _profile("dedicated-model"),
            "adjudicate": _profile("fallback-model"),
        },
        routing_rules=[{"if": {"purpose": "entity_adjudication"}, "use": "adjudicate"}],
    )
    router = _router(models)
    assert router._profile("entity_adjudication").model == "dedicated-model"


def test_document_conditional_rules_are_not_applied_to_bare_purposes(tmp_path: Path) -> None:
    models = _models(
        tmp_path,
        profiles={"adjudicate": _profile("adjudicator-model")},
        routing_rules=[
            {"if": {"purpose": "entity_adjudication", "doc_kind": "pdf_scan"}, "use": "adjudicate"}
        ],
    )
    router = _router(models)
    from fmd.llm import ProfileUnresolved

    try:
        router._profile("entity_adjudication")
    except ProfileUnresolved:
        pass
    else:  # pragma: no cover - guards the honesty of the skip
        raise AssertionError("a rule needing document context must not be guessed")


def test_registry_configures_entity_adjudication(registry) -> None:
    assert "entity_adjudication" in registry.models.profiles
    assert registry.models.profiles["entity_adjudication"].provider == "local"
    assert registry.er.adjudication.purpose == "entity_adjudication"
    # `adjudicate` stays configured as the fallback purpose.
    assert "adjudicate" in registry.models.profiles
    assert "adjudicate" in registry.er.adjudication.fallback_purposes
