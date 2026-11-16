"""Stage 2 tests: blocking, including absence-of-splink fallback."""

from __future__ import annotations

from typing import Literal

import pytest
from er_support import candidates, er_config, lexicon

from fmd.er.blocking import (
    CapabilityUnavailable,
    block,
    block_deterministic,
    block_duckdb,
    block_splink,
    duckdb_available,
    splink_available,
)
from fmd.er.config import BlockingConfig
from fmd.er.types import BlockingResult, EntityCandidate, Mention


def _cfg(backend: Literal["auto", "deterministic", "duckdb", "splink"]) -> BlockingConfig:
    return BlockingConfig(backend=backend)


def test_deterministic_blocking_finds_alias_and_state_matches() -> None:
    mention = Mention(text="KGMU", state="Uttar Pradesh")
    result = block_deterministic(
        [mention], candidates(), config=_cfg("deterministic"), lexicon=lexicon()
    )
    assert result.backend == "deterministic"
    blocked_ids = {candidates()[index].entity_id for index in result.pairs["0"]}
    assert 1 in blocked_ids
    assert result.keys["0"]


def test_pincode_and_soundex_key_path() -> None:
    candidate = EntityCandidate(
        entity_id=42,
        canonical_name="Alpha Medical College",
        state="Bihar",
        pincode="800001",
    )
    mention = Mention(text="Alfa Medical College", pincode="800001")
    result = block_deterministic(
        [mention], [candidate], config=_cfg("deterministic"), lexicon=lexicon()
    )
    assert result.pairs["0"] == [0]
    assert any(key.startswith("pincode_soundex:800001") for key in result.keys["0"])


def test_splink_absent_falls_back_with_a_reason() -> None:
    if splink_available():
        pytest.skip("splink installed in this environment")
    result = block(
        [Mention(text="Delhi University", state="Delhi")],
        candidates(),
        config=_cfg("splink"),
        lexicon=lexicon(),
    )
    assert isinstance(result, BlockingResult)
    assert result.backend == "deterministic"
    assert result.used_fallback is True
    assert "splink" in (result.reason or "")
    assert result.pairs["0"], "fallback must never return zero candidates"
    with pytest.raises(CapabilityUnavailable):
        block_splink([Mention(text="Delhi University")], candidates(), lexicon=lexicon())


def test_duckdb_absent_falls_back_with_a_reason() -> None:
    if duckdb_available():
        pytest.skip("duckdb installed in this environment")
    result = block(
        [Mention(text="Delhi University", state="Delhi")],
        candidates(),
        config=_cfg("duckdb"),
        lexicon=lexicon(),
    )
    assert result.backend == "deterministic"
    assert result.used_fallback is True
    with pytest.raises(CapabilityUnavailable):
        block_duckdb([Mention(text="Delhi University")], candidates(), lexicon=lexicon())


def test_duckdb_backend_matches_deterministic_pairs_when_installed() -> None:
    if not duckdb_available():
        pytest.skip("duckdb not installed")
    mentions_ = [Mention(text="KGMU", state="Uttar Pradesh"), Mention(text="Delhi University")]
    duck = block_duckdb(mentions_, candidates(), config=_cfg("duckdb"), lexicon=lexicon())
    deterministic = block_deterministic(
        mentions_, candidates(), config=_cfg("deterministic"), lexicon=lexicon()
    )
    assert duck.backend == "duckdb"
    assert duck.pairs == deterministic.pairs


def test_auto_backend_is_offline_and_never_empty() -> None:
    result = block(
        [Mention(text="Completely Unrelated Name", state=None)],
        candidates(),
        config=_cfg("auto"),
        lexicon=lexicon(),
    )
    assert result.pairs["0"], "global fallback keeps recall non-zero"
    assert result.empty_mentions
    assert result.used_fallback is True


def test_blocking_with_no_candidates_returns_empty_lists_not_a_crash() -> None:
    result = block_deterministic(
        [Mention(text="Delhi University")], [], config=_cfg("deterministic"), lexicon=lexicon()
    )
    assert result.pairs == {"0": []}
    assert result.empty_mentions == []


def test_blocking_is_deterministic() -> None:
    mentions = [Mention(text="Delhi University", state="Delhi")]
    first = block_deterministic(
        mentions, candidates(), config=_cfg("deterministic"), lexicon=lexicon()
    )
    second = block_deterministic(
        mentions, candidates(), config=_cfg("deterministic"), lexicon=lexicon()
    )
    assert first.model_dump() == second.model_dump()
    assert first.pairs["0"] == sorted(first.pairs["0"])


def test_er_config_defaults_to_a_testable_backend_when_overridden() -> None:
    config = er_config(blocking={"backend": "deterministic"})
    result = block(
        [Mention(text="Delhi University", state="Delhi")],
        candidates(),
        config=config.blocking,
        lexicon=config.lexicon,
    )
    assert result.backend == "deterministic"
