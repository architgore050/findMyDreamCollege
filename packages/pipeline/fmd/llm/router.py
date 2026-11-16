"""Provider-agnostic model router (brief §15).

* profiles and budgets come from ``config/models.yaml``;
* ``${ENV_VAR}`` interpolation happens at registry load;
* every actual call is audited through an :class:`LLMCallSink` (the DB repo);
* responses are cached by ``(purpose, prompt_sha256, input_sha256, model)``;
* a hard budget stop raises instead of silently overspending;
* fallbacks follow ``config/models.yaml:fallbacks`` and are reason-aware.

Tests use the ``mock`` replay provider only. Nothing here makes a network call
unless the configured provider is ``openai_compatible``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from fmd.config import Registry, get_registry
from fmd.config.models import ModelsConfig, Pricing, ProfileConfig
from fmd.db.repositories import LLMCallLog
from fmd.llm.providers import (
    BudgetExhausted,
    FixtureMissing,
    LLMError,
    MockProvider,
    OpenAICompatibleProvider,
    ProfileUnresolved,
    Provider,
    ProviderError,
    ProviderResult,
    ReplayProvider,
)


class LLMCallSink(Protocol):
    def record(self, call: LLMCallLog) -> Any: ...


def prompt_hash(system: str, user: str) -> str:
    """Stable hash of the exact prompt text (system and user are both covered)."""
    hasher = hashlib.sha256()
    hasher.update(system.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(user.encode("utf-8"))
    return hasher.hexdigest()


def input_hash(input_ref: str) -> str:
    return hashlib.sha256(input_ref.encode("utf-8")).hexdigest()


def _pricing_cost(pricing: Pricing | None, tokens_in: int | None, tokens_out: int | None) -> float:
    if pricing is None:
        return 0.0
    cost = (tokens_in or 0) / 1_000_000 * pricing.input_per_mtok + (
        tokens_out or 0
    ) / 1_000_000 * pricing.output_per_mtok
    return round(cost, 6)


class ModelRouter:
    def __init__(
        self,
        models: ModelsConfig | None = None,
        *,
        registry: Registry | None = None,
        sink: LLMCallSink | None = None,
        daily_spent_usd: float = 0.0,
        providers: dict[str, Provider] | None = None,
    ) -> None:
        if models is None:
            models = (registry or get_registry()).models
        self.models = models
        self.sink = sink
        self.budgets = models.budgets
        self._run_cost = 0.0
        self._daily_cost = daily_spent_usd
        self._cache: dict[tuple[str, str, str, str], ProviderResult] = {}
        self._provider_instances: dict[str, Provider] = dict(providers or {})

    # ------------------------------------------------------------------ public API

    def call(
        self,
        purpose: str,
        schema: type[BaseModel],
        system: str,
        user: str,
        input_ref: str,
    ) -> BaseModel:
        """Call the configured profile and return a validated ``schema`` instance."""
        profile = self._profile(purpose)
        if not profile.model:
            raise ProfileUnresolved(
                f"model profile {purpose!r} is unresolved: its model name comes from an "
                "environment variable that is not set. Run `fmd llm-ping` or use the mock "
                "provider for offline work."
            )

        p_hash = prompt_hash(system, user)
        i_hash = input_hash(input_ref)
        cache_key = (purpose, p_hash, i_hash, profile.model or "")
        if self.budgets.cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            self._audit(
                purpose=purpose,
                prompt_sha256=p_hash,
                model=cached.model,
                provider=cached.provider,
                profile=profile,
                schema=schema,
                input_ref=input_ref,
                result=cached,
                latency_ms=0,
                cost=0.0,
                cache_hit=True,
            )
            return self._validate(schema, cached.content, purpose)

        provider_name = profile.provider
        tried: set[str] = set()
        last_error: LLMError | None = None

        while provider_name and provider_name not in tried:
            tried.add(provider_name)
            try:
                if self._budget_exhausted():
                    raise BudgetExhausted(self._budget_message(provider_name))
                provider = self._get_provider(provider_name)
                started = time.perf_counter()
                result = provider.complete(
                    model=profile.model,
                    system=system,
                    user=user,
                    schema=schema,
                    profile=profile,
                    prompt_sha256=p_hash,
                    purpose=purpose,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                validated = self._validate(schema, result.content, purpose)
                cost = _pricing_cost(profile.pricing, result.tokens_in, result.tokens_out)
                self._run_cost += cost
                self._daily_cost += cost
                self._cache[cache_key] = result
                self._audit(
                    purpose=purpose,
                    prompt_sha256=p_hash,
                    model=result.model,
                    provider=provider_name,
                    profile=profile,
                    schema=schema,
                    input_ref=input_ref,
                    result=result,
                    latency_ms=latency_ms,
                    cost=cost,
                    cache_hit=False,
                )
                return validated
            except (ProviderError, BudgetExhausted, FixtureMissing) as exc:
                last_error = exc
                reason = _reason(exc)
                fallback = self._fallback_for(provider_name, reason)
                if fallback is None:
                    raise
                provider_name = fallback

        if last_error is not None:
            raise last_error
        raise LLMError(f"no provider could satisfy profile {purpose!r}")

    def ping(self) -> list[dict[str, Any]]:
        """Report profile resolution without printing secrets."""
        report: list[dict[str, Any]] = []
        for name, profile in sorted(self.models.profiles.items()):
            provider = self.models.providers.get(profile.provider)
            kind = provider.kind if provider else "unknown"
            base_url_present = bool(provider and provider.base_url)
            fixtures = self._fixtures_dir(name, provider)
            resolved = bool(profile.model) and (
                (kind == "replay" and fixtures is not None)
                or (kind == "openai_compatible" and base_url_present)
            )
            key_present = (
                bool(provider and provider.api_key) if kind == "openai_compatible" else None
            )
            report.append(
                {
                    "profile": name,
                    "provider": profile.provider,
                    "kind": kind,
                    "model": profile.model,
                    "resolved": resolved,
                    "key_present": key_present,
                    "base_url": provider.base_url if provider else None,
                    "fixtures_dir": str(fixtures) if fixtures is not None else None,
                    "reason": None
                    if resolved
                    else "missing model/base_url/fixtures_dir environment value",
                }
            )
        return report

    # ------------------------------------------------------------------ internals

    def _profile(self, purpose: str) -> ProfileConfig:
        if purpose in self.models.profiles:
            return self.models.profiles[purpose]
        routed = self._routed_purpose(purpose)
        if routed is not None and routed in self.models.profiles:
            return self.models.profiles[routed]
        raise ProfileUnresolved(
            f"unknown model purpose {purpose!r}. Configured profiles: "
            + ", ".join(sorted(self.models.profiles))
        )

    def _routed_purpose(self, purpose: str) -> str | None:
        """Resolve a purpose-only ``routing_rules`` entry to a profile name.

        A rule is honoured only when its condition is exactly ``{purpose: X}``:
        rules that also depend on document properties (``doc_kind``,
        ``pages_gt``) cannot be evaluated from a purpose alone, so they are left
        for the code path that knows the document. A dedicated profile with the
        requested name always wins over a routing rule.
        """
        for rule in self.models.routing_rules:
            condition = rule.get("if")
            target = rule.get("use")
            if not isinstance(condition, dict) or not target:
                continue
            if condition.get("purpose") != purpose:
                continue
            if any(key != "purpose" for key in condition):
                continue
            return str(target)
        return None

    def _get_provider(self, name: str) -> Provider:
        if name in self._provider_instances:
            return self._provider_instances[name]
        config = self.models.providers.get(name)
        if config is None:
            raise ProviderError(
                "unresolved_provider", f"provider {name!r} is not declared in models.yaml"
            )
        if config.kind == "replay":
            instance: Provider = (
                MockProvider(name, config) if name == "mock" else ReplayProvider(name, config)
            )
        else:
            instance = OpenAICompatibleProvider(name, config)
        self._provider_instances[name] = instance
        return instance

    def _fallback_for(self, provider_name: str, reason: str) -> str | None:
        for rule in self.models.fallbacks:
            if rule.from_provider == provider_name and reason in rule.on:
                return rule.to_provider
        return None

    def _budget_exhausted(self) -> bool:
        if not self.budgets.hard_stop:
            return False
        if self.budgets.per_run_usd > 0 and self._run_cost >= self.budgets.per_run_usd:
            return True
        return self.budgets.daily_usd > 0 and self._daily_cost >= self.budgets.daily_usd

    def _budget_message(self, provider_name: str) -> str:
        return (
            f"hard budget stop before provider {provider_name!r}: run "
            f"${self._run_cost:.4f}/${self.budgets.per_run_usd:.2f}, day "
            f"${self._daily_cost:.4f}/${self.budgets.daily_usd:.2f}. No further calls will "
            "be made (config/models.yaml:budgets.hard_stop)."
        )

    def _validate(
        self, schema: type[BaseModel], content: dict[str, Any], purpose: str
    ) -> BaseModel:
        try:
            return schema.model_validate(content)
        except ValidationError as exc:
            raise ProviderError(
                "json_schema_violation",
                f"provider response for purpose {purpose!r} did not match {schema.__name__}: {exc}",
            ) from exc

    def _audit(
        self,
        *,
        purpose: str,
        prompt_sha256: str,
        model: str,
        provider: str,
        profile: ProfileConfig,
        schema: type,
        input_ref: str,
        result: ProviderResult,
        latency_ms: int,
        cost: float,
        cache_hit: bool,
    ) -> None:
        if self.sink is None:
            return
        self.sink.record(
            LLMCallLog(
                purpose=purpose,
                prompt_sha256=prompt_sha256,
                prompt_path=None,
                model=model,
                provider=provider,
                params={
                    "schema": schema.__name__,
                    "input_ref": input_ref,
                    "temperature": profile.temperature,
                    "guided_json": profile.guided_json,
                    "max_tokens": profile.max_tokens,
                },
                raw_response=result.raw,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=latency_ms,
                cost_usd=cost,
                cache_hit=cache_hit,
            )
        )

    def _fixtures_dir(self, profile_name: str, provider: Any) -> Path | None:
        if provider is None or not provider.fixtures_dir:
            return None
        from fmd.llm.providers import _resolve_fixtures_dir

        path = _resolve_fixtures_dir(provider.fixtures_dir)
        return path if path.is_dir() else None


def _reason(exc: LLMError) -> str:
    if isinstance(exc, ProviderError):
        return exc.reason
    if isinstance(exc, BudgetExhausted):
        return "budget_exhausted"
    if isinstance(exc, FixtureMissing):
        return "fixture_missing"
    return "unknown"
