"""Load and validate the ``config/*.yaml`` registry.

Failures are collected across all files and raised as one :class:`ConfigError`
with file, path and a fix hint. Partial loads are never returned.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

import fmd.filters as filters_module
from fmd.adapters import KNOWN_ADAPTERS
from fmd.config.models import (
    DISCOVERY_PARAMS_REQUIRED,
    CostRulesConfig,
    CountriesConfig,
    ExamsConfig,
    ModelsConfig,
    SourcesConfig,
    WeightsConfig,
)
from fmd.filters import IMPLEMENTATIONS as FILTER_IMPLEMENTATIONS
from fmd.paths import resolve_config_dir

CONFIG_FILES: dict[str, str] = {
    "countries": "countries.yaml",
    "sources": "sources.yaml",
    "weights": "weights.yaml",
    "models": "models.yaml",
    "exams": "exams.yaml",
    "cost_rules": "cost_rules.yaml",
}

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """One or more registry files are missing, malformed or inconsistent."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        joined = "\n".join(f"  - {error}" for error in errors)
        super().__init__("config registry is invalid. Fix the following and re-run:\n" + joined)


@dataclass(frozen=True)
class Registry:
    """Validated, typed view of every config file."""

    countries: CountriesConfig
    sources: SourcesConfig
    weights: WeightsConfig
    models: ModelsConfig
    exams: ExamsConfig
    cost_rules: CostRulesConfig
    config_dir: Path

    @property
    def scorer_version(self) -> str:
        return self.weights.scorer_version

    @property
    def weights_version(self) -> str:
        return self.weights.weights_version

    def enabled_country_codes(self) -> list[str]:
        return sorted(self.countries.enabled)

    def enabled_sources(self) -> list[Any]:
        return self.sources.enabled

    def exam_by_id(self, exam_id: str) -> Any:
        return next((e for e in self.exams.exams if e.id == exam_id), None)


def interpolate_env(value: Any, env: Mapping[str, str]) -> Any:
    """Resolve ``${VAR}`` / ``${VAR:-default}`` recursively.

    A whole-value placeholder whose variable is unset and has no default becomes
    ``None`` so the router can report an unresolved profile instead of crashing
    the whole registry. A missing variable embedded in a larger string is an
    error: silently mangling a URL is worse than stopping.
    """

    if isinstance(value, dict):
        return {k: interpolate_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v, env) for v in value]
    if not isinstance(value, str):
        return value

    whole = _ENV_PATTERN.fullmatch(value)
    if whole is not None:
        name, default = whole.group(1), whole.group(2)
        if name in env:
            return env[name]
        return default

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        raise KeyError(name)

    try:
        return _ENV_PATTERN.sub(replace, value)
    except KeyError as exc:
        raise ConfigError(
            [
                f"environment variable ${{{exc.args[0]}}} is required by a string value "
                f"({value!r}) and has no default"
            ]
        ) from exc


def _read_yaml(config_dir: Path, filename: str, errors: list[str]) -> dict[str, Any] | None:
    path = config_dir / filename
    if not path.is_file():
        errors.append(f"{filename}: file not found in {config_dir}")
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        errors.append(f"{filename}: invalid YAML: {exc}")
        return None
    if loaded is None:
        errors.append(f"{filename}: file is empty")
        return None
    if not isinstance(loaded, dict):
        errors.append(f"{filename}: top level must be a mapping")
        return None
    return loaded


def _format_validation_error(filename: str, exc: ValidationError) -> list[str]:
    formatted: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        formatted.append(f"{filename}: {location}: {error['msg']}")
    return formatted


def _validate_cross_references(registry: Registry, errors: list[str]) -> None:
    countries = registry.countries
    sources = registry.sources
    weights = registry.weights
    models = registry.models
    exams = registry.exams
    cost_rules = registry.cost_rules

    if not countries.enabled:
        errors.append(
            "countries.yaml: no country is enabled. Set `enabled: true` on at least one "
            "country and give it a capabilities block."
        )
    for code, country in countries.enabled.items():
        if country.capabilities is None:
            errors.append(
                f"countries.yaml: enabled country {code!r} has no `capabilities` block. "
                "Add the block (programme_inventory, regulator, recognition_chain, "
                "admissions_calendar, fee_disclosure, col_basket, licensure_gate, "
                "student_work_rights, outcome_data) or disable the country."
            )

    seen_source_ids: set[str] = set()
    for position, source in enumerate(sources.sources):
        where = f"sources.yaml: sources[{position}] ({source.id!r})"
        if source.id in seen_source_ids:
            errors.append(f"{where}: duplicate source id")
        seen_source_ids.add(source.id)
        if source.adapter is None:
            errors.append(
                f"{where}: `adapter` is null. Every source must name an implementation "
                "from fmd/adapters/__init__.py:KNOWN_ADAPTERS."
            )
        elif source.adapter not in KNOWN_ADAPTERS:
            errors.append(
                f"{where}: unknown adapter {source.adapter!r}. Known adapters: "
                + ", ".join(sorted(KNOWN_ADAPTERS))
            )
        if source.enabled and source.adapter is None:
            errors.append(f"{where}: enabled sources need an adapter implementation")
        if source.kind == "discovery":
            if source.params is None:
                errors.append(
                    f"{where}: discovery sources need a `params` block. Query templates, "
                    "domain suffixes and PDF patterns are registry data (invariant 7), "
                    "never Python constants."
                )
            else:
                for field_name in DISCOVERY_PARAMS_REQUIRED.get(source.adapter or "", ()):
                    if not getattr(source.params, field_name):
                        errors.append(
                            f"{where}: adapter {source.adapter!r} needs "
                            f"`params.{field_name}` to be a non-empty list/string."
                        )

    for exclusion in sources.exclusions:
        if not exclusion.reason.strip():
            errors.append(f"sources.yaml: exclusion {exclusion.id!r} needs a non-empty reason")

    total = sum(weights.weights.values())
    if any(weight < 0 for weight in weights.weights.values()):
        errors.append("weights.yaml: weights must be non-negative")
    if not weights.weights:
        errors.append("weights.yaml: `weights` must not be empty")
    elif abs(total - 100.0) > 1e-6:
        breakdown = ", ".join(f"{k}={v:g}" for k, v in weights.weights.items())
        errors.append(
            f"weights.yaml: sub-score weights must sum to 100, got {total:g} ({breakdown}). "
            "Adjust the weights; do not renormalise silently."
        )
    if not weights.weights_version.strip():
        errors.append("weights.yaml: weights_version must be non-empty")
    if not weights.scorer_version.strip():
        errors.append("weights.yaml: scorer_version must be non-empty")

    seen_filters: set[str] = set()
    for filter_id in weights.filters:
        if filter_id in seen_filters:
            errors.append(f"weights.yaml: duplicate filter {filter_id!r}")
        seen_filters.add(filter_id)
        implementation = FILTER_IMPLEMENTATIONS.get(filter_id)
        if implementation is None:
            errors.append(
                f"weights.yaml: filter {filter_id!r} has no implementation name. "
                "Add it to fmd/filters/__init__.py:IMPLEMENTATIONS (brief §12)."
            )
        elif not callable(getattr(filters_module, implementation, None)):
            errors.append(
                f"weights.yaml: filter {filter_id!r} names {implementation!r}, which does "
                "not exist in fmd.filters"
            )

    for profile_name, profile in models.profiles.items():
        if profile.provider not in models.providers:
            errors.append(
                f"models.yaml: profile {profile_name!r} references unknown provider "
                f"{profile.provider!r}. Known providers: " + ", ".join(sorted(models.providers))
            )
    for index, fallback in enumerate(models.fallbacks):
        if fallback.from_provider not in models.providers:
            errors.append(
                f"models.yaml: fallbacks[{index}].from references unknown provider "
                f"{fallback.from_provider!r}"
            )
        if fallback.to_provider not in models.providers:
            errors.append(
                f"models.yaml: fallbacks[{index}].to references unknown provider "
                f"{fallback.to_provider!r}"
            )
    for provider_name, provider in models.providers.items():
        if provider.kind == "replay" and not provider.fixtures_dir:
            errors.append(f"models.yaml: replay provider {provider_name!r} needs `fixtures_dir`")
        if provider.kind == "openai_compatible" and not provider.base_url:
            # base_url may be unresolved because an env var is unset; ping reports it.
            continue

    seen_exam_ids: set[str] = set()
    for position, exam in enumerate(exams.exams):
        where = f"exams.yaml: exams[{position}] ({exam.id!r})"
        if exam.id in seen_exam_ids:
            errors.append(f"{where}: duplicate exam id")
        seen_exam_ids.add(exam.id)
        if exam.country not in countries.countries:
            errors.append(f"{where}: country {exam.country!r} is not declared in countries.yaml")
        if exam.scope not in {"national", "state", "institution"}:
            errors.append(f"{where}: scope must be national|state|institution, got {exam.scope!r}")

    seen_categories: set[str] = set()
    for category in cost_rules.cost_categories:
        if category in seen_categories:
            errors.append(f"cost_rules.yaml: duplicate cost category {category!r}")
        seen_categories.add(category)
    for scenario_name, scenario in cost_rules.living_scenarios.items():
        if not scenario.components:
            errors.append(f"cost_rules.yaml: living scenario {scenario_name!r} has no components")
    for band_name, band in cost_rules.dsr25.bands.items():
        if not band:
            errors.append(f"cost_rules.yaml: DSR25 band {band_name!r} is empty")


def load_registry(
    config_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Registry:
    """Read, interpolate, parse and cross-validate all six config files."""
    directory = Path(config_dir) if config_dir is not None else resolve_config_dir()
    environment: Mapping[str, str] = os.environ if env is None else env

    errors: list[str] = []
    documents: dict[str, dict[str, Any]] = {}
    for key, filename in CONFIG_FILES.items():
        loaded = _read_yaml(directory, filename, errors)
        if loaded is None:
            continue
        try:
            documents[key] = interpolate_env(loaded, environment)
        except ConfigError as exc:
            errors.extend(f"{filename}: {message}" for message in exc.errors)

    if errors:
        raise ConfigError(errors)

    parsed: dict[str, Any] = {}
    model_for = {
        "countries": CountriesConfig,
        "sources": SourcesConfig,
        "weights": WeightsConfig,
        "models": ModelsConfig,
        "exams": ExamsConfig,
        "cost_rules": CostRulesConfig,
    }
    for key, model in model_for.items():
        filename = CONFIG_FILES[key]
        try:
            parsed[key] = model.model_validate(documents[key])
        except ValidationError as exc:
            errors.extend(_format_validation_error(filename, exc))

    if errors:
        raise ConfigError(errors)

    registry = Registry(
        countries=parsed["countries"],
        sources=parsed["sources"],
        weights=parsed["weights"],
        models=parsed["models"],
        exams=parsed["exams"],
        cost_rules=parsed["cost_rules"],
        config_dir=directory.resolve(),
    )
    _validate_cross_references(registry, errors)
    if errors:
        raise ConfigError(errors)
    return registry


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    """Process-wide registry singleton (validated once)."""
    return load_registry()


def reset_registry_cache() -> None:
    """Test helper: force the next :func:`get_registry` to re-read config."""
    get_registry.cache_clear()
