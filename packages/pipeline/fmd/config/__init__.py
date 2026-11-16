"""Typed registry loading for ``config/*.yaml``."""

from fmd.config.loader import (
    CONFIG_FILES,
    ConfigError,
    RecognitionOriginError,
    Registry,
    denied_domain,
    get_registry,
    host_of,
    interpolate_env,
    load_registry,
    reset_registry_cache,
    validate_recognition_origin,
)
from fmd.config.models import (
    CostRulesConfig,
    CountriesConfig,
    DenylistedDomain,
    ExamsConfig,
    ModelsConfig,
    SourcesConfig,
    WeightsConfig,
)

__all__ = [
    "CONFIG_FILES",
    "ConfigError",
    "CountriesConfig",
    "CostRulesConfig",
    "DenylistedDomain",
    "ExamsConfig",
    "ModelsConfig",
    "RecognitionOriginError",
    "Registry",
    "SourcesConfig",
    "WeightsConfig",
    "denied_domain",
    "get_registry",
    "host_of",
    "interpolate_env",
    "load_registry",
    "reset_registry_cache",
    "validate_recognition_origin",
]
