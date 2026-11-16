"""Configuration and data lexicons for entity resolution.

Thresholds, weights, alias markers and abbreviation lexicons are injected here —
never baked into stage logic. A caller can build :class:`ERConfig` from a mapping
(e.g. a future ``config/er.yaml``) or use the documented defaults for offline
work. ``config/er.yaml`` is not part of this workstream's scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fmd.er.types import AliasType

# Generic English/institution vocabulary only. Nothing here identifies a real
# institution, state or acronym lexicon; those arrive from the caller/fixtures.
GENERIC_LEXICON_DATA: dict[str, Any] = {
    "abbreviations": {
        "instt": "institute",
        "inst": "institute",
        "instn": "institution",
        "univ": "university",
        "coll": "college",
        "govt": "government",
        "med": "medical",
        "tech": "technology",
        "sci": "science",
        "dept": "department",
    },
    "acronyms": {},
    "legal_phrases": (
        "university of",
        "college of",
        "institute of",
        "school of",
        "academy of",
        "department of",
        "university",
        "college",
        "institute",
        "institution",
        "school",
        "academy",
    ),
    "stopwords": ("the", "of", "and", "for", "at", "in", "a", "an", "to", "on"),
    "alias_markers": (
        {"marker": "also known as", "alias_type": "former_name", "confidence": 0.8},
        {"marker": "formerly known as", "alias_type": "former_name", "confidence": 0.85},
        {"marker": "formerly", "alias_type": "former_name", "confidence": 0.8},
        {"marker": "erstwhile", "alias_type": "former_name", "confidence": 0.75},
        {"marker": "previously called", "alias_type": "former_name", "confidence": 0.75},
        {"marker": "previously", "alias_type": "former_name", "confidence": 0.7},
    ),
}


class ErConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AliasMarker(ErConfigModel):
    marker: str
    alias_type: AliasType = "former_name"
    confidence: float = 0.7

    @model_validator(mode="after")
    def _marker_not_blank(self) -> AliasMarker:
        if not self.marker.strip():
            raise ValueError("alias marker must not be blank")
        return self


class Lexicon(ErConfigModel):
    """Language/vocabulary data. Every stage receives one; nothing is hardcoded."""

    abbreviations: dict[str, str] = Field(default_factory=dict)
    acronyms: dict[str, str] = Field(default_factory=dict)
    legal_phrases: tuple[str, ...] = ()
    stopwords: frozenset[str] = frozenset()
    alias_markers: tuple[AliasMarker, ...] = ()

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> Lexicon:
        payload = dict(GENERIC_LEXICON_DATA)
        payload.update(data or {})
        return cls.model_validate(payload)

    @classmethod
    def default(cls) -> Lexicon:
        return cls.from_mapping(None)


DEFAULT_LEXICON = Lexicon.default()


class Thresholds(ErConfigModel):
    high: float = 0.85
    low: float = 0.40

    @model_validator(mode="after")
    def _ordered(self) -> Thresholds:
        if not 0.0 <= self.low < self.high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
        return self


class BlockingConfig(ErConfigModel):
    backend: Literal["auto", "deterministic", "duckdb", "splink"] = "auto"
    use_state_token: bool = True
    use_pincode_soundex: bool = True
    use_name_soundex: bool = True
    min_token_length: int = 3
    soundex_length: int = 4
    allow_global_fallback: bool = True
    fallback_all_candidates: bool = True


class SimilarityConfig(ErConfigModel):
    token_weight: float = 0.55
    char_weight: float = 0.35
    context_weight: float = 0.10
    acronym_score: float = 0.97
    alias_type_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "manual": 1.0,
            "former_name": 1.0,
            "transliteration": 0.95,
            "local_name": 0.95,
            "acronym": 0.97,
            "misspelling": 0.90,
        }
    )
    state_conflict_penalty: float = 0.45
    city_conflict_penalty: float = 0.80
    pincode_conflict_penalty: float = 0.35
    use_embeddings: bool = True
    embedding_model: str = "BAAI/bge-m3"
    embedding_weight: float = 1.0

    @model_validator(mode="after")
    def _weights_sum(self) -> SimilarityConfig:
        total = self.token_weight + self.char_weight + self.context_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("token_weight + char_weight + context_weight must equal 1")
        missing = {"manual", "former_name", "transliteration", "local_name", "acronym"}
        missing -= set(self.alias_type_weights)
        if missing:
            raise ValueError(f"alias_type_weights missing: {sorted(missing)}")
        return self


class AdjudicationConfig(ErConfigModel):
    enabled: bool = True
    purpose: str = "entity_adjudication"
    fallback_purposes: tuple[str, ...] = ("adjudicate",)
    bundle_size: int = 20
    cache: bool = True
    system_prompt: str = (
        "You resolve institution names for a sourced college inventory. "
        "Two records are 'same' only when they denote the same legal institution, "
        "not merely a similar name in the same region. Campus or affiliate names are "
        "'different' unless the record names the same institution. Return only the "
        "requested JSON."
    )
    instructions: str = (
        "For each pair decide strictly: same | different | unsure. "
        "`evidence` must quote the overlapping name tokens that support the verdict. "
        "Never merge records that differ in city/state unless one name is an explicit "
        "former name or acronym of the other."
    )


class AliasMiningConfig(ErConfigModel):
    header_lines: int = 3
    footer_lines: int = 3
    max_alias_tokens: int = 8
    max_aliases: int = 50
    bracket_acronym_confidence: float = 0.90
    caps_acronym_confidence: float = 0.60
    caps_near_name_window: int = 240
    local_name_confidence: float = 0.50
    transliteration_confidence: float = 0.65
    misspelling_similarity: float = 0.90


class ERConfig(ErConfigModel):
    """Everything tunable about entity resolution, in one injectable object."""

    version: str = "er-v1"
    lexicon: Lexicon = Field(default_factory=lambda: DEFAULT_LEXICON)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    blocking: BlockingConfig = Field(default_factory=BlockingConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    adjudication: AdjudicationConfig = Field(default_factory=AdjudicationConfig)
    alias_mining: AliasMiningConfig = Field(default_factory=AliasMiningConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ERConfig:
        payload = dict(data)
        if "lexicon" in payload:
            payload["lexicon"] = Lexicon.from_mapping(payload["lexicon"])
        return cls.model_validate(payload)

    @classmethod
    def load(cls, path: str | Path) -> ERConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: entity-resolution config must be a mapping")
        return cls.from_mapping(raw)


def load_lexicon(path: str | Path) -> Lexicon:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: lexicon must be a mapping")
    return Lexicon.from_mapping(raw)
