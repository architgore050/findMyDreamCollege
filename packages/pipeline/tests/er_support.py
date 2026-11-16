"""Shared fixtures/helpers for the entity-resolution tests (offline only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fmd.er.config import ERConfig, Lexicon
from fmd.er.types import EntityCandidate, Mention

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "er"


def load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def lexicon() -> Lexicon:
    return Lexicon.from_mapping(load_json("lexicon.json"))


def er_config(**overrides: Any) -> ERConfig:
    payload: dict[str, Any] = {"lexicon": lexicon()}
    payload.update(overrides)
    return ERConfig.from_mapping(payload)


def candidates() -> list[EntityCandidate]:
    return [
        EntityCandidate.model_validate(item) for item in load_json("candidates.json")["candidates"]
    ]


def mentions() -> list[Mention]:
    return [Mention.model_validate(item) for item in load_json("mentions.json")["mentions"]]


def candidate_by_id(entity_id: int) -> EntityCandidate:
    return next(item for item in candidates() if item.entity_id == entity_id)
