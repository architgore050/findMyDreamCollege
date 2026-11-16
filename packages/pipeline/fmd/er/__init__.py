"""Entity resolution — workstream E (brief §10).

Pipeline: normalise → block → compare/score → decide (bands + LLM adjudication)
→ alias mining → resolve with an audit record per decision.

All stages are pure: no database writes, no network (except the explicitly
injected model router), no GPU requirement.
"""

from fmd.er.aliases import AliasCandidate, mine_aliases, to_alias_creates
from fmd.er.blocking import (
    CapabilityUnavailable,
    block,
    block_deterministic,
    block_duckdb,
    block_splink,
    duckdb_available,
    splink_available,
)
from fmd.er.config import (
    DEFAULT_LEXICON,
    AdjudicationConfig,
    AliasMiningConfig,
    BandsConfig,
    BlockingConfig,
    ERConfig,
    Lexicon,
    PriorsConfig,
    SimilarityConfig,
    Thresholds,
)
from fmd.er.decide import (
    AdjudicationCache,
    AdjudicationOutcome,
    AdjudicationPair,
    adjudicate,
    band_for,
    pair_hash,
)
from fmd.er.normalise import blocking_key, normalise_name, soundex
from fmd.er.service import AuditSink, audit_decisions, resolve, resolve_with_audit
from fmd.er.similarity import (
    EmbeddingBackend,
    EmbeddingScorer,
    SentenceTransformerEmbedder,
    score_pair,
    score_pairs,
)
from fmd.er.types import (
    Alias,
    AuditDecision,
    AuditRecord,
    Band,
    BlockingResult,
    EntityCandidate,
    Mention,
    ModelVerdict,
    NormalisedName,
    Resolution,
    ResolutionBatch,
    ScoredCandidate,
)

__all__ = [
    "DEFAULT_LEXICON",
    "AdjudicationCache",
    "AdjudicationConfig",
    "AdjudicationOutcome",
    "AdjudicationPair",
    "Alias",
    "AliasCandidate",
    "AliasMiningConfig",
    "AuditDecision",
    "AuditRecord",
    "AuditSink",
    "Band",
    "BandsConfig",
    "BlockingConfig",
    "BlockingResult",
    "CapabilityUnavailable",
    "ERConfig",
    "EmbeddingBackend",
    "EmbeddingScorer",
    "EntityCandidate",
    "Lexicon",
    "Mention",
    "ModelVerdict",
    "NormalisedName",
    "PriorsConfig",
    "Resolution",
    "ResolutionBatch",
    "ScoredCandidate",
    "SentenceTransformerEmbedder",
    "SimilarityConfig",
    "Thresholds",
    "adjudicate",
    "audit_decisions",
    "band_for",
    "block",
    "block_deterministic",
    "block_duckdb",
    "block_splink",
    "blocking_key",
    "duckdb_available",
    "mine_aliases",
    "normalise_name",
    "pair_hash",
    "resolve",
    "resolve_with_audit",
    "score_pair",
    "score_pairs",
    "soundex",
    "splink_available",
    "to_alias_creates",
]
