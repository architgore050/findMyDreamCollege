"""Stage 3 — similarity scoring (brief §10).

Token/character similarity (rapidfuzz when installed, ``difflib`` otherwise),
alias-type-aware comparison and an optional bge-m3 embedding path behind a
capability check. The output is a heuristic, calibrated-ish probability plus the
features that produced it, so a reviewer can audit a merge.

This is not yet Fellegi–Sunter: there are no configured m/u priors (see report).
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Sequence
from typing import Protocol

from fmd.er.config import ERConfig, Lexicon, SimilarityConfig
from fmd.er.normalise import normalise_name, normalise_text
from fmd.er.types import (
    Alias,
    EntityCandidate,
    Mention,
    NormalisedName,
    ScoredCandidate,
    SimilarityFeatures,
)

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - installed only with the `er` extra
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except ImportError:  # pragma: no cover - standard offline path
    _rapidfuzz_fuzz = None


def similarity_backend() -> str:
    return "rapidfuzz" if _rapidfuzz_fuzz is not None else "difflib"


def token_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _difflib_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def char_ratio(a: str, b: str) -> float:
    """Character-level similarity in [0, 1]."""
    if not a or not b:
        return 0.0
    if _rapidfuzz_fuzz is not None:  # pragma: no cover - er extra
        return float(_rapidfuzz_fuzz.ratio(a, b)) / 100.0
    return _difflib_ratio(a, b)


def token_ratio(a: str, b: str) -> float:
    """Token-sort similarity in [0, 1] (rapidfuzz ``token_sort_ratio`` or difflib)."""
    if not a or not b:
        return 0.0
    left = " ".join(sorted(a.split()))
    right = " ".join(sorted(b.split()))
    if _rapidfuzz_fuzz is not None:  # pragma: no cover - er extra
        return float(_rapidfuzz_fuzz.token_sort_ratio(left, right)) / 100.0
    return _difflib_ratio(left, right)


class EmbeddingBackend(Protocol):
    name: str

    def similarity(self, left: str, right: str) -> float: ...


class SentenceTransformerEmbedder:
    """bge-m3 via sentence-transformers; lazy, CPU-friendly, embedding-cached."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._model = None
        self._vectors: dict[str, list[float]] = {}

    @staticmethod
    def available() -> bool:
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, text: str) -> list[float]:
        if text not in self._vectors:
            vector = self._load().encode([text], normalize_embeddings=True)[0]
            self._vectors[text] = [float(value) for value in vector]
        return self._vectors[text]

    def similarity(self, left: str, right: str) -> float:
        a, b = self._embed(left), self._embed(right)
        return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))


class EmbeddingScorer:
    """Capability-gated embedding similarity with an in-memory pair cache."""

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        *,
        model_name: str = "BAAI/bge-m3",
        cache: dict[str, float] | None = None,
    ) -> None:
        self.reason: str | None = None
        if backend is None:
            if SentenceTransformerEmbedder.available():
                backend = SentenceTransformerEmbedder(model_name)
            else:
                self.reason = "sentence-transformers not installed"
        self.backend = backend
        self._cache = cache if cache is not None else {}

    @property
    def available(self) -> bool:
        return self.backend is not None

    def similarity(self, left: str, right: str) -> float | None:
        if self.backend is None:
            return None
        digest = hashlib.sha256(
            f"{self.backend.name}\x00{left}\x00{right}".encode()
        ).hexdigest()
        if digest in self._cache:
            return self._cache[digest]
        try:
            score = max(0.0, min(1.0, float(self.backend.similarity(left, right))))
        except Exception as exc:  # noqa: BLE001 - degradation must never break ER
            self.reason = f"embedding backend failed: {exc}"
            LOGGER.warning("ER similarity: %s", self.reason)
            return None
        self._cache[digest] = score
        return score


def _candidate_forms(
    candidate: EntityCandidate,
    lexicon: Lexicon,
) -> list[tuple[Alias | None, NormalisedName]]:
    forms: list[tuple[Alias | None, NormalisedName]] = [
        (None, normalise_name(candidate.canonical_name, lexicon))
    ]
    forms.extend(
        (alias, normalise_name(alias.alias, lexicon)) for alias in candidate.aliases
    )
    return forms


def _acronym_match(
    mention: NormalisedName,
    form: NormalisedName,
) -> float:
    if len(form.tokens) == 1 and form.tokens[0] in mention.acronyms:
        return 1.0
    if len(mention.tokens) == 1 and mention.tokens[0] in form.acronyms:
        return 1.0
    if mention.text == form.text:
        return 0.0  # identical names are handled by token/char similarity
    if set(mention.acronyms) & set(form.acronyms):
        return 1.0
    return 0.0


def _context(
    mention: Mention,
    candidate: EntityCandidate,
) -> tuple[dict[str, float | None], dict[str, float]]:
    pairs = {
        "state": (mention.state, candidate.state),
        "city": (mention.city, candidate.city),
        "pincode": (mention.pincode, candidate.pincode),
    }
    matches: dict[str, float | None] = {}
    conflicts: dict[str, float] = {}
    for name, (left, right) in pairs.items():
        if not left or not right:
            matches[name] = None
            conflicts[name] = 0.0
            continue
        equal = normalise_text(str(left), Lexicon.default()) == normalise_text(
            str(right), Lexicon.default()
        )
        matches[name] = 1.0 if equal else 0.0
        conflicts[name] = 0.0 if equal else 1.0
    return matches, conflicts


def _context_value(matches: dict[str, float | None]) -> float:
    known = [value for value in matches.values() if value is not None]
    if not known:
        return 0.5  # no context evidence: neutral, never a penalty
    return sum(known) / len(known)


def score_pair(
    mention: Mention,
    mention_norm: NormalisedName,
    candidate: EntityCandidate,
    *,
    config: SimilarityConfig | None = None,
    lexicon: Lexicon | None = None,
    embedder: EmbeddingScorer | None = None,
) -> ScoredCandidate:
    """Score one mention/candidate pair and return the features used."""
    cfg = config or SimilarityConfig()
    lex = lexicon or Lexicon.default()
    forms = _candidate_forms(candidate, lex)

    best_alias: Alias | None = None
    best_form = forms[0][1]
    best_name_score = -1.0
    best_acronym = 0.0
    best_token = 0.0
    best_char = 0.0
    for alias, form in forms:
        tokens = token_jaccard(mention_norm.tokens, form.tokens)
        chars = char_ratio(mention_norm.text, form.text)
        name_score = cfg.token_weight * tokens + cfg.char_weight * chars
        acronym = _acronym_match(mention_norm, form)
        if acronym:
            name_score = max(name_score, cfg.acronym_score)
        alias_type = alias.alias_type if alias is not None else None
        if alias is not None:
            name_score *= cfg.alias_type_weights.get(alias_type or "", 1.0)
        if name_score > best_name_score:
            best_name_score = name_score
            best_alias = alias
            best_form = form
            best_acronym = acronym
            best_token = tokens
            best_char = chars

    matches, conflicts = _context(mention, candidate)
    embedding: float | None = None
    if embedder is not None and cfg.use_embeddings:
        embedding = embedder.similarity(mention_norm.text, best_form.text)
        if embedding is not None:
            best_name_score = max(best_name_score, embedding * cfg.embedding_weight)

    score = best_name_score + cfg.context_weight * _context_value(matches)
    if conflicts["state"]:
        score *= cfg.state_conflict_penalty
    if conflicts["city"]:
        score *= cfg.city_conflict_penalty
    if conflicts["pincode"]:
        score *= cfg.pincode_conflict_penalty
    score = max(0.0, min(1.0, round(score, 6)))

    features = SimilarityFeatures(
        token_jaccard=round(best_token, 6),
        char_ratio=round(best_char, 6),
        acronym_match=best_acronym,
        embedding=None if embedding is None else round(embedding, 6),
        state_match=matches["state"],
        city_match=matches["city"],
        pincode_match=matches["pincode"],
        state_conflict=conflicts["state"],
        city_conflict=conflicts["city"],
        pincode_conflict=conflicts["pincode"],
    )
    return ScoredCandidate(
        candidate=candidate,
        score=score,
        band="adjudicate",  # decided later, in decide.band_for
        features=features,
        matched_alias=best_alias.alias if best_alias is not None else None,
        matched_alias_type=best_alias.alias_type if best_alias is not None else None,
        similarity_method=similarity_backend(),
    )


def score_pairs(
    mention: Mention,
    mention_norm: NormalisedName,
    candidates: Sequence[EntityCandidate],
    *,
    config: ERConfig | None = None,
    embedder: EmbeddingScorer | None = None,
) -> list[ScoredCandidate]:
    cfg = config or ERConfig()
    scored = [
        score_pair(
            mention,
            mention_norm,
            candidate,
            config=cfg.similarity,
            lexicon=cfg.lexicon,
            embedder=embedder,
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (-item.score, _candidate_sort_key(item.candidate)))
    return scored


def _candidate_sort_key(candidate: EntityCandidate) -> tuple[int, int, str]:
    return (
        1 if candidate.entity_id is None else 0,
        candidate.entity_id if candidate.entity_id is not None else 0,
        candidate.canonical_name,
    )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity for callers that already hold vectors."""
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if norm == 0 else max(0.0, min(1.0, dot / norm))


__all__ = [
    "EmbeddingBackend",
    "EmbeddingScorer",
    "SentenceTransformerEmbedder",
    "char_ratio",
    "cosine",
    "score_pair",
    "score_pairs",
    "similarity_backend",
    "token_jaccard",
    "token_ratio",
]
