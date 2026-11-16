"""ER config surface: registry loading, cross-validation, recognition origin."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmd.config import ConfigError, RecognitionOriginError, load_registry
from fmd.config.loader import validate_recognition_origin


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_er_yaml_loads_through_the_registry(registry) -> None:
    er = registry.er
    assert er.version == "er-v1"
    assert 0.0 <= er.thresholds.low < er.thresholds.high <= 1.0
    assert er.bands.order == ("reject", "adjudicate", "accept")
    assert er.adjudication.bundle_size == 20
    assert er.similarity.alias_type_weights["manual"] == 1.0
    # Honest placeholder: no calibrated m/u priors ship in v1.
    assert er.priors.method == "not_calibrated"
    assert er.priors.m == {} and er.priors.u == {}
    assert registry.er.adjudication.purpose in registry.models.profiles


def test_unordered_bands_are_rejected(config_dir: Path) -> None:
    path = config_dir / "er.yaml"
    payload = _read(path)
    payload["bands"]["order"] = ["accept", "adjudicate", "reject"]
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    message = str(excinfo.value)
    assert "bands.order" in message
    assert "er.yaml" in message


def test_negative_alias_weight_is_rejected(config_dir: Path) -> None:
    path = config_dir / "er.yaml"
    payload = _read(path)
    payload["alias_weights"]["acronym"] = -0.5
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    assert "alias_type_weights['acronym'] must be >= 0" in str(excinfo.value)


def test_negative_similarity_weight_is_rejected(config_dir: Path) -> None:
    path = config_dir / "er.yaml"
    payload = _read(path)
    payload["similarity"] = {"token_weight": -0.1, "char_weight": 0.35, "context_weight": 0.10}
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    assert "must be >= 0" in str(excinfo.value)


def test_insane_thresholds_are_rejected(config_dir: Path) -> None:
    path = config_dir / "er.yaml"
    payload = _read(path)
    payload["thresholds"] = {"high": 0.30, "low": 0.70}
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    assert "thresholds must satisfy" in str(excinfo.value)


def test_denylisted_domain_cannot_originate_recognition(registry) -> None:
    denied = registry.sources.denied_domains()
    assert denied, "sources.yaml must keep at least one denylisted domain for this guard"
    for domain in denied:
        for url in (f"https://{domain}/recognition", f"https://www.{domain}/council"):
            with pytest.raises(RecognitionOriginError):
                validate_recognition_origin(url, denied)
    # A statutory host is untouched by the same guard.
    validate_recognition_origin("https://ncahp.abdm.gov.in/list", denied)


def test_er_yaml_may_not_duplicate_the_denylist(config_dir: Path) -> None:
    path = config_dir / "er.yaml"
    payload = _read(path)
    payload["denylisted_domains"] = [{"domain": "example.org", "reason": "duplicate"}]
    _write(path, payload)
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    assert "denylisted_domains" in str(excinfo.value)
