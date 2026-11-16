"""Pure data types shared by the entity-resolution stages (brief §10).

Everything here is a value object: the stages take these in and return these out.
No database, network, or model access lives in this module.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Band = Literal["accept", "reject", "adjudicate"]

# The five types the schema allows today (db/migrations/0001_init.sql).
SchemaAliasType = Literal[
    "acronym",
    "former_name",
    "transliteration",
    "misspelling",
    "local_name",
]

# ``manual`` is an ER-only marker: a human has confirmed this name. It is
# deliberately not persisted as an alias today (see the WS-E report/schema gap).
AliasType = Literal[
    "acronym",
    "former_name",
    "transliteration",
    "misspelling",
    "local_name",
    "manual",
]

Script = Literal["latin", "devanagari", "mixed", "other"]


class ErModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Alias(ErModel):
    """An alternate name for an entity, with its provenance and type."""

    alias: str
    alias_type: AliasType
    confidence: float = 1.0
    source_id: str | None = None


class Mention(ErModel):
    """A name observed in evidence (a document, header, sitemap entry, ...)."""

    text: str
    state: str | None = None
    city: str | None = None
    pincode: str | None = None
    source_id: str | None = None
    doc_id: str | None = None
    aliases: list[Alias] = Field(default_factory=list)


class EntityCandidate(ErModel):
    """A known entity that a mention may resolve to."""

    canonical_name: str
    entity_id: int | None = None
    kind: str = "institution"
    state: str | None = None
    city: str | None = None
    pincode: str | None = None
    aliases: list[Alias] = Field(default_factory=list)

    @property
    def manual_lock(self) -> bool:
        """True when a human has asserted this entity's name.

        A manually confirmed canonical name may never be changed by an automatic
        merge; only an exact manual-alias match may resolve here unattended.
        """
        return any(alias.alias_type == "manual" for alias in self.aliases)

    @property
    def manual_aliases(self) -> list[Alias]:
        return [alias for alias in self.aliases if alias.alias_type == "manual"]


class NormalisedName(ErModel):
    """Output of :func:`fmd.er.normalise.normalise_name`."""

    raw: str
    text: str
    tokens: tuple[str, ...]
    acronyms: tuple[str, ...]
    script: Script
    transliterated: bool = False
    notes: tuple[str, ...] = ()


class BlockingKeys(ErModel):
    """Deterministic blocking keys for one name occurrence."""

    keys: tuple[str, ...]
    first_token: str
    soundex: str
    state_key: str | None = None
    pincode_key: str | None = None


class BlockingResult(ErModel):
    """Candidate indices per mention index (mention_id -> candidate indices)."""

    backend: Literal["deterministic", "duckdb", "splink"]
    pairs: dict[str, list[int]]
    keys: dict[str, list[str]] = Field(default_factory=dict)
    reason: str | None = None
    used_fallback: bool = False
    empty_mentions: list[str] = Field(default_factory=list)


class SimilarityFeatures(ErModel):
    """The evidence behind a similarity score; all values are 0..1 unless noted."""

    token_jaccard: float
    char_ratio: float
    acronym_match: float
    embedding: float | None = None
    state_match: float | None = None
    city_match: float | None = None
    pincode_match: float | None = None
    state_conflict: float = 0.0
    city_conflict: float = 0.0
    pincode_conflict: float = 0.0


class ScoredCandidate(ErModel):
    candidate: EntityCandidate
    score: float
    band: Band
    features: SimilarityFeatures
    matched_alias: str | None = None
    matched_alias_type: AliasType | None = None
    similarity_method: str = "difflib"


class ModelVerdict(ErModel):
    """A model's forced-choice answer about one candidate pair."""

    pair_id: str
    verdict: Literal["same", "different", "unsure"]
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    cached: bool = False


class AuditRecord(ErModel):
    """Why a mention resolved the way it did; written per decision for review."""

    mention_text: str
    normalised: str
    blocking_keys: tuple[str, ...]
    blocking_backend: str
    blocking_reason: str | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    config_version: str = "er-v1"
    band: Band
    score: float
    decided_by: Literal["score", "model", "manual", "none"] = "score"
    model_verdict: ModelVerdict | None = None
    adjudication_pairs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    features: dict[str, Any] = Field(default_factory=dict)


class Resolution(ErModel):
    mention: Mention
    normalised: NormalisedName
    band: Band
    chosen: EntityCandidate | None = None
    score: float = 0.0
    candidates: list[ScoredCandidate] = Field(default_factory=list)
    audit: AuditRecord


class ResolutionBatch(ErModel):
    resolutions: list[Resolution]
    blocking: BlockingResult
    adjudication: dict[str, ModelVerdict] = Field(default_factory=dict)
