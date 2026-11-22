"""W10 batched-acquire config surface.

The `batch`/`cache` blocks decide how a ~1,000-page backlog is cut, so a typo
must fail config load rather than silently change batching. These tests pin the
loader rejections (found untested at intake of FMD-CLOUD-020).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmd.acquire.parse_cache import PARSE_SCHEMA_VERSION
from fmd.config import ConfigError, load_registry


def _acquire(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "acquire.yaml").read_text(encoding="utf-8"))


def _write(config_dir: Path, payload: dict) -> None:
    (config_dir / "acquire.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _batch(config_dir: Path) -> dict:
    payload = _acquire(config_dir)
    return payload["batch"]


def _load_error(config_dir: Path) -> str:
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    return str(excinfo.value)


def test_batch_and_cache_blocks_are_loaded(config_dir: Path) -> None:
    registry = load_registry(config_dir, env={})
    assert registry.acquire.batch is not None
    assert registry.acquire.batch.pages == 100
    assert registry.acquire.batch.device == "cuda"
    assert registry.acquire.batch.keep_server is False
    assert registry.acquire.cache is not None
    assert registry.acquire.cache.schema_version == PARSE_SCHEMA_VERSION


@pytest.mark.parametrize("pages", [0, -1, -100])
def test_batch_pages_must_be_positive(config_dir: Path, pages: int) -> None:
    payload = _acquire(config_dir)
    payload["batch"]["pages"] = pages
    _write(config_dir, payload)

    message = _load_error(config_dir)
    assert "batch.pages must be > 0" in message
    assert "acquire.yaml" in message


def test_batch_device_must_be_known(config_dir: Path) -> None:
    payload = _acquire(config_dir)
    payload["batch"]["device"] = "gpu"
    _write(config_dir, payload)

    message = _load_error(config_dir)
    assert "batch.device 'gpu' is outside the known set" in message
    assert "cpu, auto, cuda" in message


def test_batch_poll_timeout_must_be_positive(config_dir: Path) -> None:
    payload = _acquire(config_dir)
    payload["batch"]["poll_timeout_s"] = 0
    _write(config_dir, payload)

    assert "batch.poll_timeout_s must be > 0" in _load_error(config_dir)


def test_batch_server_script_rejects_empty_string(config_dir: Path) -> None:
    payload = _acquire(config_dir)
    payload["batch"]["server_script"] = ""
    _write(config_dir, payload)

    message = _load_error(config_dir)
    assert "batch.server_script must be a path or unset" in message


def test_batch_unknown_key_is_rejected(config_dir: Path) -> None:
    payload = _acquire(config_dir)
    payload["batch"]["batch_pages"] = 100  # plausible typo for pages
    _write(config_dir, payload)

    message = _load_error(config_dir)
    assert "batch.batch_pages" in message
    assert "Extra inputs" in message


def test_batch_server_script_may_be_null(config_dir: Path) -> None:
    payload = _acquire(config_dir)
    payload["batch"]["server_script"] = None
    _write(config_dir, payload)

    registry = load_registry(config_dir, env={})
    assert registry.acquire.batch is not None
    assert registry.acquire.batch.server_script is None


def test_cache_schema_version_must_be_positive(config_dir: Path) -> None:
    payload = _acquire(config_dir)
    payload["cache"]["schema_version"] = 0
    _write(config_dir, payload)

    assert "cache.schema_version must be > 0" in _load_error(config_dir)


def test_cache_schema_version_must_match_the_code(config_dir: Path) -> None:
    payload = _acquire(config_dir)
    payload["cache"]["schema_version"] = PARSE_SCHEMA_VERSION + 1
    _write(config_dir, payload)

    message = _load_error(config_dir)
    assert "must equal fmd.acquire.parse_cache.PARSE_SCHEMA_VERSION" in message
    assert str(PARSE_SCHEMA_VERSION) in message


def test_batch_blocks_are_optional(config_dir: Path) -> None:
    payload = _acquire(config_dir)
    payload.pop("batch")
    payload.pop("cache")
    _write(config_dir, payload)

    registry = load_registry(config_dir, env={})
    assert registry.acquire.batch is None
    assert registry.acquire.cache is None


def test_batch_helper_reads_the_committed_block(config_dir: Path) -> None:
    # Guards the helpers above: a missing block would make every rejection test
    # pass for the wrong reason.
    assert _batch(config_dir)["pages"] == 100
