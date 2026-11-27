from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmd.config import ConfigError, load_registry
from fmd.config.loader import interpolate_env


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_valid_registry_loads(registry) -> None:
    assert "IN" in registry.countries.enabled
    assert registry.countries.enabled["IN"].capabilities is not None
    assert len(registry.sources.sources) >= 20
    assert all(source.adapter for source in registry.sources.sources)
    assert all(source.adapter for source in registry.sources.enabled)
    assert abs(sum(registry.weights.weights.values()) - 100.0) < 1e-9
    assert registry.weights.filters
    assert registry.models.profiles["bulk_extract"].provider == "local"
    # The live profile set. OCR is an acquisition stage, never a router profile;
    # `hard_pdf` is the hosted fallback and reads the shared FMD_API_MODEL;
    # `programme_equivalence` is the W9 Tier-2 classifier (config/programmes.yaml);
    # `verdict_live` is the w13 live-verdict research profile (docs/v1/eval/07).
    assert set(registry.models.profiles) == {
        "bulk_extract",
        "hard_pdf",
        "adjudicate",
        "entity_adjudication",
        "nlp_rubric",
        "verify",
        "programme_equivalence",
        "verdict_live",
        "embed",
    }
    assert registry.models.profiles["hard_pdf"].provider == "api"
    assert registry.models.profiles["hard_pdf"].model is None
    assert registry.models.profiles["hard_pdf"].model_env == "FMD_API_MODEL"
    assert registry.exams.exams
    assert registry.cost_rules.cost_categories
    assert registry.scorer_version and registry.weights_version


def test_profile_fallback_is_registry_driven(registry) -> None:
    # Existing profiles retain the historical fallback behaviour by default;
    # verdict_live explicitly opts out to protect invariant 10 and schema retries.
    assert registry.models.profiles["bulk_extract"].allow_fallback is True
    assert registry.models.profiles["verdict_live"].allow_fallback is False


def test_weights_must_sum_to_100(config_dir: Path) -> None:
    path = config_dir / "weights.yaml"
    payload = _read(path)
    payload["weights"]["outcomes"] = 31
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "must sum to 100" in message
    assert "weights.yaml" in message


def test_unknown_adapter_is_rejected(config_dir: Path) -> None:
    path = config_dir / "sources.yaml"
    payload = _read(path)
    payload["sources"][0]["adapter"] = "does_not_exist"
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "unknown adapter 'does_not_exist'" in message
    assert "Known adapters" in message


def test_null_adapter_is_rejected(config_dir: Path) -> None:
    path = config_dir / "sources.yaml"
    payload = _read(path)
    payload["sources"][0]["adapter"] = None
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    assert "`adapter` is null" in str(excinfo.value)


def _discovery_source(payload: dict, source_id: str) -> dict:
    return next(source for source in payload["sources"] if source["id"] == source_id)


def test_discovery_source_without_params_is_rejected(config_dir: Path) -> None:
    path = config_dir / "sources.yaml"
    payload = _read(path)
    _discovery_source(payload, "searxng").pop("params", None)
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "discovery sources need a `params` block" in message
    assert "searxng" in message


def test_discovery_source_missing_required_param_is_rejected(config_dir: Path) -> None:
    path = config_dir / "sources.yaml"
    payload = _read(path)
    _discovery_source(payload, "common_crawl_index")["params"]["pdf_url_patterns"] = []
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "params.pdf_url_patterns" in message
    assert "common_crawl" in message


def test_discovery_params_are_registry_data(registry) -> None:
    searxng = registry.sources.by_id("searxng")
    crawl = registry.sources.by_id("common_crawl_index")
    assert searxng is not None and searxng.params is not None
    assert crawl is not None and crawl.params is not None
    assert searxng.params.query_templates
    assert crawl.params.domain_suffixes
    assert crawl.params.pdf_url_patterns
    assert crawl.params.index_glob


def test_enabled_country_requires_capabilities(config_dir: Path) -> None:
    path = config_dir / "countries.yaml"
    payload = _read(path)
    payload["countries"]["PH"]["enabled"] = True
    payload["countries"]["PH"].pop("capabilities", None)
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "enabled country 'PH' has no `capabilities` block" in message


def test_unknown_filter_implementation_is_rejected(config_dir: Path) -> None:
    path = config_dir / "weights.yaml"
    payload = _read(path)
    payload["filters"].append("no_such_filter")
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    assert "has no implementation name" in str(excinfo.value)


def test_missing_config_file_is_actionable(config_dir: Path) -> None:
    (config_dir / "models.yaml").unlink()
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    assert "models.yaml: file not found" in str(excinfo.value)


# ------------------------------------------------- programmes.yaml strictness
#
# The tenth registry file decides whether a programme is admitted to the pool.
# A typo here must fail the load rather than silently admit everything.


def _programmes_payload(config_dir: Path) -> dict:
    return _read(config_dir / "programmes.yaml")


def test_programmes_empty_include_patterns_is_rejected(config_dir: Path) -> None:
    path = config_dir / "programmes.yaml"
    payload = _programmes_payload(config_dir)
    payload["countries"]["IN"]["include_patterns"] = []
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "programmes.yaml" in message
    assert "countries.IN.include_patterns" in message
    assert "must not be empty when declared" in message


def test_programmes_empty_exclude_patterns_is_rejected(config_dir: Path) -> None:
    path = config_dir / "programmes.yaml"
    payload = _programmes_payload(config_dir)
    payload["countries"]["IN"]["exclude_patterns"] = []
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "programmes.yaml" in message
    assert "countries.IN.exclude_patterns" in message
    assert "must not be empty when declared" in message


@pytest.mark.parametrize("pattern", [".*", "^.*$", "(?s).*"])
def test_programmes_catch_all_pattern_is_rejected(config_dir: Path, pattern: str) -> None:
    path = config_dir / "programmes.yaml"
    payload = _programmes_payload(config_dir)
    payload["countries"]["IN"]["include_patterns"].append(pattern)
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "programmes.yaml" in message
    assert "countries.IN.include_patterns" in message
    assert "catch-all pattern" in message
    assert "MBBS" in message


def test_programmes_unknown_top_level_key_is_rejected(config_dir: Path) -> None:
    path = config_dir / "programmes.yaml"
    payload = _programmes_payload(config_dir)
    payload["countriez"] = {}
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "programmes.yaml" in message
    assert "countriez" in message


def test_programmes_unknown_taxonomy_key_is_rejected(config_dir: Path) -> None:
    path = config_dir / "programmes.yaml"
    payload = _programmes_payload(config_dir)
    payload["countries"]["IN"]["include_patternz"] = [r"\bmlt\b"]
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "programmes.yaml" in message
    assert "include_patternz" in message


def test_programmes_missing_definition_is_rejected(config_dir: Path) -> None:
    path = config_dir / "programmes.yaml"
    payload = _programmes_payload(config_dir)
    payload.pop("definition")
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "programmes.yaml" in message
    assert "definition" in message


def test_env_interpolation_defaults_and_unresolved(monkeypatch) -> None:
    env = {"SET_VAR": "value"}
    assert interpolate_env("${SET_VAR}", env) == "value"
    assert interpolate_env("${MISSING:-fallback}", env) == "fallback"
    assert interpolate_env("prefix-${SET_VAR}", env) == "prefix-value"
    assert interpolate_env("${MISSING}", env) is None
    with pytest.raises(ConfigError):
        interpolate_env("prefix-${MISSING}", env)


def test_unresolved_api_profile_is_reported_not_crashing(registry) -> None:
    # config/models.yaml deliberately has empty API env vars in tests.
    profile = registry.models.profiles["hard_pdf"]
    assert profile.model is None
    assert registry.models.providers["api"].base_url is None
