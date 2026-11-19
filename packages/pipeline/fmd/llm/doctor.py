"""``fmd llm doctor``: self-diagnosing model configuration (brief §15).

Every profile reports whether its provider, base URL, key and model resolved,
what the router expects for guided JSON, and — only with ``live=True`` — a cheap
reachability probe. Remediation lines name the exact environment variable, so an
operator can fix a run without reading YAML. Secret values are never included.
"""

from __future__ import annotations

from typing import Any

import httpx

from fmd.config.models import ModelsConfig, ProfileConfig, ProviderConfig
from fmd.llm.providers import Provider, ProviderError, build_provider


def _env_label(name: str | None) -> str:
    return name or "<unset environment variable>"


def profile_blockers(
    models: ModelsConfig, profile: ProfileConfig
) -> tuple[ProviderConfig | None, list[str]]:
    """Return the provider config and precise, human-readable blockers.

    An empty list means the profile can be called (modulo live reachability).
    """
    provider: ProviderConfig | None = models.providers.get(profile.provider)
    blockers: list[str] = []
    if provider is None:
        known = ", ".join(sorted(models.providers)) or "<none>"
        blockers.append(
            f"provider {profile.provider!r} is not declared in models.yaml; set "
            f"{_env_label(profile.provider_env)} to one of: {known}"
        )
        return None, blockers

    if not profile.model:
        example = f" (for example {profile.model_example})" if profile.model_example else ""
        blockers.append(
            f"{_env_label(profile.model_env)} is empty: set it to a model id accepted "
            f"by provider {profile.provider!r}{example}"
        )

    if provider.kind == "replay":
        if not provider.fixtures_dir:
            blockers.append(
                "provider has no fixtures_dir: set it in config/models.yaml for offline replay"
            )
        return provider, blockers

    if not provider.base_url:
        example = (
            f" (for example {provider.base_url_example})" if provider.base_url_example else ""
        )
        blockers.append(
            f"{_env_label(provider.base_url_env)} is empty: set it to your provider "
            f"base URL{example}"
        )

    if provider.requires_api_key and not provider.api_key:
        blockers.append(
            f"{_env_label(provider.api_key_env)} is empty: set it to your provider token"
        )
    return provider, blockers


def _reachability(
    provider: ProviderConfig | None,
    *,
    name: str,
    live: bool,
    transport: httpx.BaseTransport | None,
) -> tuple[bool | None, str | None]:
    if not live:
        return None, None
    if provider is None:
        return False, "provider is not declared"
    try:
        instance: Provider = build_provider(name, provider, transport=transport)
        return instance.probe()
    except ProviderError as exc:
        return False, str(exc)


def diagnose_profile(
    models: ModelsConfig,
    name: str,
    profile: ProfileConfig,
    *,
    live: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    if name == "embed":
        # The embed profile is not a chat-completions target: its blockers,
        # probe and mode come from the embedding backend factory, which knows
        # the difference between the in-process CPU backend and an HTTP
        # /v1/embeddings endpoint (the :8010 chat server is neither).
        from fmd.llm.embeddings import describe_embed_profile

        return describe_embed_profile(models, profile, live=live, transport=transport)
    provider, blockers = profile_blockers(models, profile)
    reachable, reach_note = _reachability(
        provider, name=profile.provider, live=live, transport=transport
    )
    guided_expected = bool(
        profile.guided_json
        and provider is not None
        and provider.supports_guided_json
        and provider.kind != "replay"
    )
    usable = not blockers and (reachable is not False if live else True)
    remediation = list(blockers)
    if live and reachable is False and reach_note:
        remediation.append(f"live probe failed: {reach_note}")

    return {
        "profile": name,
        "provider": profile.provider,
        "kind": provider.kind if provider else "unknown",
        "base_url": provider.base_url if provider else None,
        "base_url_resolved": bool(provider and provider.base_url),
        "base_url_env": provider.base_url_env if provider else None,
        "key_present": bool(provider.api_key) if provider else False,
        "key_required": provider.requires_api_key if provider else False,
        "api_key_env": provider.api_key_env if provider else None,
        "model": profile.model,
        "model_env": profile.model_env,
        "guided_json_expected": guided_expected,
        "guided_json_declared": profile.guided_json,
        "reachable": reachable,
        "reachability_note": reach_note,
        "required": profile.required,
        "degrade_reason": profile.degrade_reason,
        "usable": usable,
        "blockers": blockers,
        "remediation": remediation,
    }


def doctor_report(
    models: ModelsConfig,
    *,
    live: bool = False,
    transport: httpx.BaseTransport | None = None,
    ocr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profiles = [
        diagnose_profile(models, name, profile, live=live, transport=transport)
        for name, profile in sorted(models.profiles.items())
    ]
    required_ok = all(row["usable"] for row in profiles if row["required"])
    remediation = [
        f"{row['profile']}: {line}"
        for row in profiles
        for line in row["remediation"]
    ]
    return {
        "profiles": profiles,
        "required_ok": required_ok,
        "ok": required_ok,
        "live": live,
        "remediation": remediation,
        "budgets": models.budgets.model_dump(),
        "ocr": ocr,
    }


def profile_remediation(models: ModelsConfig, name: str) -> str:
    """One-line, exact-variable message for the router's failure path."""
    profile = models.profiles.get(name)
    if profile is None:
        return f"unknown profile {name!r}; configured: " + ", ".join(sorted(models.profiles))
    _provider, blockers = profile_blockers(models, profile)
    if not blockers:
        return f"profile {name!r} resolved; run `fmd llm doctor --live` to test reachability"
    return f"profile {name!r} is unusable: " + "; ".join(blockers)
