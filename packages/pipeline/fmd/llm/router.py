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
from fmd.llm.doctor import doctor_report, profile_blockers, profile_remediation
from fmd.llm.providers import (
    BudgetExhausted,
    FixtureMissing,
    LLMError,
    NetworkOutage,
    ProfileUnresolved,
    Provider,
    ProviderError,
    ProviderResult,
    build_provider,
    json_instruction,
)
from fmd.net.policy import NetworkGuard
from fmd.obs.metrics import emit as metrics_emit

#: Consecutive transient failures after which the endpoint is treated as gone.
#: Two is too eager (one dropped packet), five wastes a minute on a dead link.
MAX_CONSECUTIVE_NETWORK_FAILURES = 3

#: Provider reasons that mean "the endpoint is unreachable", not "this
#: document is bad". A 5xx is host-specific and does not trip the run breaker.
OUTAGE_REASONS = frozenset(
    {
        "network_unavailable",
        "dns_failure",
        "timeout",
        "connect_timeout",
        "read_timeout",
        "write_timeout",
        "connection_error",
        "network_unreachable",
    }
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
        # One breaker per router (i.e. per run). When the local server vanishes
        # mid-batch, remaining items fail immediately instead of each waiting
        # out a full read timeout; the caller keeps the claims it already saved.
        self.guard = NetworkGuard(threshold=MAX_CONSECUTIVE_NETWORK_FAILURES)
        self._network_failures = 0
        self._outage: str | None = None

    # ------------------------------------------------------------------ network
    def network_state(self) -> dict[str, Any]:
        """Expose the outage state so a run summary can report it honestly."""
        return {
            "outage": self._outage,
            "consecutive_failures": self._network_failures,
            "guard": self.guard.state(),
        }

    def _check_outage(self) -> None:
        if self._outage is not None:
            raise NetworkOutage(self._outage)

    def _record_provider_failure(self, reason: str, exc: LLMError) -> None:
        if reason not in OUTAGE_REASONS:
            return
        self._network_failures += 1
        if self._network_failures < MAX_CONSECUTIVE_NETWORK_FAILURES:
            return
        self._outage = (
            f"endpoint stopped answering after {self._network_failures} consecutive "
            f"transient failures (last: {reason}). Aborting the remaining batch; "
            "documents already persisted stay persisted and the rest are left "
            "unverified rather than reported as extracted."
        )
        raise NetworkOutage(
            self._outage,
        ) from exc

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
        _provider_cfg, blockers = profile_blockers(self.models, profile)
        if blockers:
            message = profile_remediation(self.models, purpose)
            if profile.required:
                raise ProfileUnresolved(message)
            raise ProviderError(
                profile.degrade_reason or "profile_unavailable",
                message + ". This profile is optional; the run continues without it.",
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
            metrics_emit(
                "llm",
                purpose,
                device="cuda" if cached.provider == "local" else "cpu",
                model=cached.model,
                provider=cached.provider,
                duration_ms=0,
                tokens_in=cached.tokens_in,
                tokens_out=cached.tokens_out,
                cache_hit=True,
                input_ref=input_ref,
                status="ok",
            )
            return self._validate(schema, cached.content, purpose)

        provider_name = profile.provider
        tried: set[str] = set()
        last_error: LLMError | None = None
        self._check_outage()

        while provider_name and provider_name not in tried:
            tried.add(provider_name)
            try:
                if self._budget_exhausted():
                    raise BudgetExhausted(self._budget_message(provider_name))
                provider = self._get_provider(provider_name)
                effective_profile, effective_system = self._guided_profile(
                    provider, profile, schema, system
                )
                started = time.perf_counter()
                result = provider.complete(
                    model=profile.model,
                    system=effective_system,
                    user=user,
                    schema=schema,
                    profile=effective_profile,
                    prompt_sha256=p_hash,
                    purpose=purpose,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                validated = self._validate(schema, result.content, purpose)
                self._network_failures = 0
                _emit_llm_metric(
                    purpose,
                    model=result.model,
                    provider=provider_name,
                    latency_ms=latency_ms,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    input_ref=input_ref,
                )
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
            except NetworkOutage:
                raise
            except (ProviderError, BudgetExhausted, FixtureMissing) as exc:
                last_error = exc
                reason = _reason(exc)
                self._record_provider_failure(reason, exc)
                fallback = (
                    self._fallback_for(provider_name, reason) if profile.allow_fallback else None
                )
                if fallback is None:
                    raise
                provider_name = fallback

        if last_error is not None:
            raise last_error
        raise LLMError(f"no provider could satisfy profile {purpose!r}")

    def ping(self) -> list[dict[str, Any]]:
        """Report profile resolution without printing secrets.

        Kept for compatibility; ``fmd llm doctor`` is the richer surface.
        """
        report: list[dict[str, Any]] = []
        for row in self.doctor()["profiles"]:
            provider = self.models.providers.get(row["provider"])
            fixtures = self._fixtures_dir(row["profile"], provider)
            report.append(
                {
                    "profile": row["profile"],
                    "provider": row["provider"],
                    "kind": row["kind"],
                    "model": row["model"],
                    "resolved": row["usable"],
                    "key_present": None if row["kind"] == "replay" else row["key_present"],
                    "base_url": row["base_url"],
                    "fixtures_dir": str(fixtures) if fixtures is not None else None,
                    "reason": None if row["usable"] else "; ".join(row["blockers"]),
                    "remediation": row["remediation"],
                    "guided_json_expected": row["guided_json_expected"],
                    "required": row["required"],
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
        instance = build_provider(name, config, guard=self.guard)
        self._provider_instances[name] = instance
        return instance

    def _guided_profile(
        self,
        provider: Provider,
        profile: ProfileConfig,
        schema: type[BaseModel],
        system: str,
    ) -> tuple[ProfileConfig, str]:
        """Degrade guided JSON to JSON-instruct + validation when unsupported.

        Replay providers keep the original prompt so fixture hashes stay stable.
        """
        if provider.kind == "replay":
            return profile, system
        supports = bool(getattr(provider, "supports_guided_json", True)) and bool(
            getattr(getattr(provider, "config", None), "supports_guided_json", True)
        )
        if profile.guided_json and supports:
            return profile, system
        effective = profile.model_copy(update={"guided_json": False})
        return effective, system + "\n\n" + json_instruction(schema)

    def doctor(self, *, live: bool = False, transport: Any | None = None) -> dict[str, Any]:
        """Per-profile diagnostics; secrets are never included."""
        return doctor_report(self.models, live=live, transport=transport)

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


def _emit_llm_metric(
    purpose: str,
    *,
    model: str,
    provider: str,
    latency_ms: int,
    tokens_in: int | None,
    tokens_out: int | None,
    input_ref: str,
) -> None:
    """One compute-metrics record per live call (no-op when none is active).

    Throughput is derived from the provider's own usage numbers: decode tok/s
    needs the server's timing split, which the OpenAI-compatible payload does
    not expose, so ``total_tps`` is the honest (tokens / wall) figure and the
    prefill/decode fields stay None unless a provider reports them.
    """
    seconds = latency_ms / 1000.0
    total_tps: float | None = None
    if seconds > 0 and (tokens_in is not None or tokens_out is not None):
        total_tps = ((tokens_in or 0) + (tokens_out or 0)) / seconds
    metrics_emit(
        "llm",
        purpose,
        # The local 26B is GPU-resident (runbook §6); hosted/replay providers
        # bill no card of ours. This is what makes GPU-vs-CPU time honest.
        device="cuda" if provider == "local" else "cpu",
        model=model,
        provider=provider,
        duration_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tps=total_tps,
        input_ref=input_ref,
        status="ok",
    )


def _reason(exc: LLMError) -> str:
    if isinstance(exc, ProviderError):
        return exc.reason
    if isinstance(exc, BudgetExhausted):
        return "budget_exhausted"
    if isinstance(exc, FixtureMissing):
        return "fixture_missing"
    return "unknown"
