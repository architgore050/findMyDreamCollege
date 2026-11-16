"""Stage 2 — candidate blocking (brief §10).

Deterministic keys are the always-available path: ``(state, first token)`` and
``(pincode, soundex)`` plus weak token/soundex keys for recall. splink/DuckDB
pair generation sits behind a capability check and degrades to the deterministic
blocker with a logged reason — blocking never crashes and never silently returns
zero candidates for a mention.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from typing import Any

from fmd.er.config import BlockingConfig, ERConfig, Lexicon
from fmd.er.normalise import blocking_keys, normalise_name
from fmd.er.types import BlockingResult, EntityCandidate, Mention

LOGGER = logging.getLogger(__name__)

CAPABILITY_MODULES = {"splink": "splink", "duckdb": "duckdb"}


class CapabilityUnavailable(RuntimeError):
    """Raised when an optional ER backend is not installed."""

    def __init__(self, capability: str, detail: str = "") -> None:
        self.capability = capability
        message = f"capability {capability!r} is unavailable"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


def capability_available(name: str) -> bool:
    module = CAPABILITY_MODULES.get(name, name)
    return importlib.util.find_spec(module) is not None


def splink_available() -> bool:
    return capability_available("splink")


def duckdb_available() -> bool:
    return capability_available("duckdb")


def _name_keys(
    text: str,
    *,
    state: str | None,
    pincode: str | None,
    config: BlockingConfig,
    lexicon: Lexicon,
    transliterate: Any | None = None,
):
    normalised = normalise_name(text, lexicon, transliterate=transliterate)
    return blocking_keys(
        normalised, state=state, pincode=pincode, config=config, lexicon=lexicon
    ).keys


def _mention_keys(
    mention: Mention,
    *,
    config: BlockingConfig,
    lexicon: Lexicon,
    transliterate: Any | None = None,
) -> tuple[str, ...]:
    forms = [mention.text, *(alias.alias for alias in mention.aliases)]
    keys: list[str] = []
    for form in forms:
        keys.extend(
            _name_keys(
                form,
                state=mention.state,
                pincode=mention.pincode,
                config=config,
                lexicon=lexicon,
                transliterate=transliterate,
            )
        )
    return tuple(dict.fromkeys(keys))


def _candidate_keys(
    candidate: EntityCandidate,
    *,
    config: BlockingConfig,
    lexicon: Lexicon,
    transliterate: Any | None = None,
) -> tuple[str, ...]:
    forms = [candidate.canonical_name, *(alias.alias for alias in candidate.aliases)]
    keys: list[str] = []
    for form in forms:
        keys.extend(
            _name_keys(
                form,
                state=candidate.state,
                pincode=candidate.pincode,
                config=config,
                lexicon=lexicon,
                transliterate=transliterate,
            )
        )
    return tuple(dict.fromkeys(keys))


def _pairs_from_keys(
    mention_keys: Sequence[tuple[str, ...]],
    candidate_keys: Sequence[tuple[str, ...]],
) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for position, keys in enumerate(candidate_keys):
        for key in keys:
            index.setdefault(key, []).append(position)
    pairs: dict[str, list[int]] = {}
    for position, keys in enumerate(mention_keys):
        found: set[int] = set()
        for key in keys:
            found.update(index.get(key, ()))
        pairs[str(position)] = sorted(found)
    return pairs


def _global_fallback(
    pairs: dict[str, list[int]],
    candidate_count: int,
) -> list[int]:
    empty = [int(key) for key, value in pairs.items() if not value]
    if not empty:
        return []
    everyone = list(range(candidate_count))
    for position in empty:
        pairs[str(position)] = list(everyone)
    return empty


def block_deterministic(
    mentions: Sequence[Mention],
    candidates: Sequence[EntityCandidate],
    *,
    config: BlockingConfig | None = None,
    lexicon: Lexicon | None = None,
) -> BlockingResult:
    cfg = config or BlockingConfig()
    lex = lexicon or Lexicon.default()
    mention_keys = [_mention_keys(m, config=cfg, lexicon=lex) for m in mentions]
    candidate_keys = [_candidate_keys(c, config=cfg, lexicon=lex) for c in candidates]
    pairs = _pairs_from_keys(mention_keys, candidate_keys)
    empty_mentions: list[str] = []
    if cfg.fallback_all_candidates:
        positions = _global_fallback(pairs, len(candidates))
        empty_mentions = [mentions[int(pos)].text for pos in positions]
    return BlockingResult(
        backend="deterministic",
        pairs=pairs,
        keys={str(i): list(keys) for i, keys in enumerate(mention_keys)},
        reason="deterministic keys (state+token, pincode+soundex, token, soundex)",
        used_fallback=bool(empty_mentions),
        empty_mentions=empty_mentions,
    )


def _duckdb_pairs(
    mention_keys: Sequence[tuple[str, ...]],
    candidate_keys: Sequence[tuple[str, ...]],
    *,
    backend: str,
) -> tuple[dict[str, list[int]], str]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only with er extra
        raise CapabilityUnavailable("duckdb", str(exc)) from exc

    mention_rows = [
        (int(position), key) for position, keys in enumerate(mention_keys) for key in keys
    ]
    candidate_rows = [
        (int(position), key) for position, keys in enumerate(candidate_keys) for key in keys
    ]
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            "CREATE TABLE mention_keys (mention_id INTEGER, key VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE candidate_keys (candidate_id INTEGER, key VARCHAR)"
        )
        if mention_rows:
            connection.executemany("INSERT INTO mention_keys VALUES (?, ?)", mention_rows)
        if candidate_rows:
            connection.executemany("INSERT INTO candidate_keys VALUES (?, ?)", candidate_rows)
        rows = connection.execute(
            "SELECT DISTINCT m.mention_id, c.candidate_id FROM mention_keys m "
            "JOIN candidate_keys c USING (key) ORDER BY m.mention_id, c.candidate_id"
        ).fetchall()
    finally:
        connection.close()
    pairs: dict[str, list[int]] = {str(i): [] for i in range(len(mention_keys))}
    for mention_id, candidate_id in rows:
        pairs[str(int(mention_id))].append(int(candidate_id))
    return pairs, backend


def block_duckdb(
    mentions: Sequence[Mention],
    candidates: Sequence[EntityCandidate],
    *,
    config: BlockingConfig | None = None,
    lexicon: Lexicon | None = None,
) -> BlockingResult:
    """DuckDB-backed exact-key pair generation (capability-gated)."""
    if not duckdb_available():
        raise CapabilityUnavailable("duckdb", "install the `er` extra: uv sync --extra er")
    cfg = config or BlockingConfig()
    lex = lexicon or Lexicon.default()
    mention_keys = [_mention_keys(m, config=cfg, lexicon=lex) for m in mentions]
    candidate_keys = [_candidate_keys(c, config=cfg, lexicon=lex) for c in candidates]
    pairs, backend = _duckdb_pairs(mention_keys, candidate_keys, backend="duckdb")
    empty_mentions: list[str] = []
    if cfg.fallback_all_candidates:
        positions = _global_fallback(pairs, len(candidates))
        empty_mentions = [mentions[int(pos)].text for pos in positions]
    return BlockingResult(
        backend=backend,  # type: ignore[arg-type]
        pairs=pairs,
        keys={str(i): list(keys) for i, keys in enumerate(mention_keys)},
        reason="duckdb exact-key join",
        used_fallback=bool(empty_mentions),
        empty_mentions=empty_mentions,
    )


def block_splink(
    mentions: Sequence[Mention],
    candidates: Sequence[EntityCandidate],
    *,
    config: BlockingConfig | None = None,
    lexicon: Lexicon | None = None,
) -> BlockingResult:
    """splink-backed blocking (capability-gated).

    TODO(WS-E): splink settings (m/u priors, clerical matching) are not yet wired.
    Availability is checked, then the DuckDB pair generator that splink itself
    uses supplies candidate pairs; comparison/scoring stays in ``similarity``.
    """
    if not splink_available():
        raise CapabilityUnavailable("splink", "install the `er` extra: uv sync --extra er")
    result = block_duckdb(
        mentions, candidates, config=config, lexicon=lexicon
    )
    return result.model_copy(
        update={
            "backend": "splink",
            "reason": "splink available; DuckDB exact-key pairs (m/u model not yet wired)",
        }
    )


def block(
    mentions: Sequence[Mention],
    candidates: Sequence[EntityCandidate],
    *,
    config: BlockingConfig | None = None,
    lexicon: Lexicon | None = None,
    logger: logging.Logger | None = None,
) -> BlockingResult:
    """Run blocking, falling back deterministically and never raising."""
    cfg = config or BlockingConfig()
    log = logger or LOGGER
    lex = lexicon or Lexicon.default()
    if cfg.backend in {"splink", "duckdb"}:
        try:
            if cfg.backend == "splink":
                return block_splink(mentions, candidates, config=cfg, lexicon=lex)
            return block_duckdb(mentions, candidates, config=cfg, lexicon=lex)
        except CapabilityUnavailable as exc:
            log.warning("ER blocking: %s; falling back to deterministic keys", exc)
            fallback = block_deterministic(mentions, candidates, config=cfg, lexicon=lex)
            return fallback.model_copy(
                update={"reason": f"{exc}; fell back to deterministic blocking",
                        "used_fallback": True}
            )
    if cfg.backend == "auto":
        for name in ("splink", "duckdb"):
            if capability_available(name):
                try:
                    if name == "splink":
                        return block_splink(mentions, candidates, config=cfg, lexicon=lex)
                    return block_duckdb(mentions, candidates, config=cfg, lexicon=lex)
                except CapabilityUnavailable as exc:  # pragma: no cover - defensive
                    log.warning("ER blocking: %s; trying next backend", exc)
        log.info("ER blocking: neither splink nor duckdb installed; deterministic keys")
        result = block_deterministic(mentions, candidates, config=cfg, lexicon=lex)
        return result.model_copy(
            update={
                "reason": "splink/duckdb not installed; deterministic blocking",
                "used_fallback": True,
            }
        )
    return block_deterministic(mentions, candidates, config=cfg, lexicon=lex)


__all__ = [
    "CapabilityUnavailable",
    "block",
    "block_deterministic",
    "block_duckdb",
    "block_splink",
    "capability_available",
    "duckdb_available",
    "splink_available",
]
