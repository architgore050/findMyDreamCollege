"""Stage 3 tests: similarity features, alias awareness, embeddings, cache."""

from __future__ import annotations

from er_support import candidate_by_id, er_config, lexicon

from fmd.er.config import Lexicon, SimilarityConfig
from fmd.er.normalise import normalise_name
from fmd.er.similarity import (
    EmbeddingScorer,
    SentenceTransformerEmbedder,
    char_ratio,
    score_pair,
    similarity_backend,
    token_jaccard,
)
from fmd.er.types import Alias, EntityCandidate, Mention
from fmd.llm.embeddings import MockEmbeddings

CFG = er_config(blocking={"backend": "deterministic"})


def _score(mention_text: str, candidate: EntityCandidate, *, mention_state=None, cfg=None):
    config = cfg or CFG
    mention = Mention(text=mention_text, state=mention_state)
    norm = normalise_name(mention.text, config.lexicon)
    return score_pair(mention, norm, candidate, config=config.similarity, lexicon=config.lexicon)


def test_difflib_is_the_offline_fallback() -> None:
    assert similarity_backend() in {"difflib", "rapidfuzz"}


def test_char_and_token_measures() -> None:
    assert token_jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert token_jaccard(["a", "b"], ["c", "d"]) == 0.0
    assert char_ratio("delhi", "delhi") == 1.0
    assert char_ratio("delhi", "delhii") > 0.9
    assert char_ratio("", "delhi") == 0.0


def test_exact_name_scores_accept_band() -> None:
    scored = _score(
        "King George's Medical University", candidate_by_id(1), mention_state="Uttar Pradesh"
    )
    assert scored.score >= CFG.thresholds.high
    assert scored.features.token_jaccard == 1.0
    assert scored.features.state_match == 1.0
    assert scored.matched_alias_type is None


def test_acronym_alias_matches_without_lexicon_expansion() -> None:
    lex = Lexicon(
        abbreviations=lexicon().abbreviations,
        acronyms={},  # no KGMU mapping: acronym matching must stand on its own
        legal_phrases=lexicon().legal_phrases,
        stopwords=lexicon().stopwords,
    )
    config = CFG.model_copy(update={"lexicon": lex})
    candidate = EntityCandidate(
        entity_id=11,
        canonical_name="King George's Medical University",
        state="Uttar Pradesh",
    )
    mention = Mention(text="KGMU", state="Uttar Pradesh")
    scored = score_pair(
        mention,
        normalise_name(mention.text, config.lexicon),
        candidate,
        config=config.similarity,
        lexicon=config.lexicon,
    )
    assert scored.features.acronym_match == 1.0
    assert scored.score >= config.thresholds.high


def test_state_conflict_penalises_identical_names_below_reject() -> None:
    candidate = EntityCandidate(
        entity_id=20,
        canonical_name="Government Medical College",
        state="Maharashtra",
        city="Nagpur",
        pincode="440010",
    )
    scored = _score(
        "Government Medical College",
        candidate,
        mention_state="Bihar",
    )
    assert scored.features.state_conflict == 1.0
    assert scored.score <= CFG.thresholds.low
    assert scored.score < CFG.thresholds.high


def test_alias_type_changes_the_score() -> None:
    former = EntityCandidate(
        entity_id=21,
        canonical_name="Delhi University",
        aliases=[Alias(alias="Delhi Vishwavidyalaya", alias_type="former_name")],
    )
    misspelt = EntityCandidate(
        entity_id=22,
        canonical_name="Delhi University",
        aliases=[Alias(alias="Delhi Vishwavidyalaya", alias_type="misspelling")],
    )
    former_score = _score("Delhi Vishwavidyalaya", former)
    misspelt_score = _score("Delhi Vishwavidyalaya", misspelt)
    assert former_score.matched_alias_type == "former_name"
    assert misspelt_score.matched_alias_type == "misspelling"
    assert former_score.score > misspelt_score.score


def test_manual_alias_is_weighted_as_authoritative() -> None:
    candidate = EntityCandidate(
        entity_id=23,
        canonical_name="Delhi University",
        aliases=[Alias(alias="Delhi Vishwavidyalaya", alias_type="manual")],
    )
    scored = _score("Delhi Vishwavidyalaya", candidate)
    assert scored.matched_alias_type == "manual"
    assert scored.score >= CFG.thresholds.high


class _FakeEmbedding:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def similarity(self, left: str, right: str) -> float:
        self.calls += 1
        return 0.99


def test_embedding_path_and_in_memory_cache() -> None:
    backend = _FakeEmbedding()
    embedder = EmbeddingScorer(backend=backend)
    candidate = EntityCandidate(entity_id=30, canonical_name="Beta Institute")
    first = score_pair(
        Mention(text="Alpha Academy"),
        normalise_name("Alpha Academy", CFG.lexicon),
        candidate,
        config=SimilarityConfig(),
        lexicon=CFG.lexicon,
        embedder=embedder,
    )
    second = score_pair(
        Mention(text="Alpha Academy"),
        normalise_name("Alpha Academy", CFG.lexicon),
        candidate,
        config=SimilarityConfig(),
        lexicon=CFG.lexicon,
        embedder=embedder,
    )
    assert first.features.embedding == 0.99
    assert first.score >= CFG.thresholds.high
    assert second.features.embedding == 0.99
    assert backend.calls == 1, "pair score must be cached"
    assert embedder.available is True


def test_embedding_unavailable_degrades_without_error(monkeypatch) -> None:
    monkeypatch.setattr(SentenceTransformerEmbedder, "available", staticmethod(lambda: False))
    embedder = EmbeddingScorer()
    assert embedder.available is False
    assert embedder.reason
    assert embedder.similarity("a", "b") is None


def test_scores_are_bounded_and_deterministic() -> None:
    candidate = candidate_by_id(6)
    first = _score("Rajendra Institute of Medical Technology", candidate, mention_state="Bihar")
    second = _score("Rajendra Institute of Medical Technology", candidate, mention_state="Bihar")
    assert 0.0 <= first.score <= 1.0
    assert first.model_dump() == second.model_dump()
    assert first.similarity_method == similarity_backend()


# ----------------------------------------------------------------- W1 embed-provider:
# the fallback must be visible. When the backend or model is absent the scorer
# records the canonical `embeddings_unavailable` mode with the exact variable to
# fix, keeps the lexical features, and leaves features.embedding empty.


class _BrokenBackend:
    """A backend that resolves but explodes on first use (bad URL, dead server)."""

    name = "broken"

    def similarity(self, left: str, right: str) -> float:
        raise RuntimeError("connection refused")


def test_unavailable_mode_names_the_exact_env_variable() -> None:
    from fmd.config.models import ModelsConfig
    from fmd.er.similarity import EmbeddingScorer

    models = ModelsConfig.model_validate(
        {
            "providers": {},
            "profiles": {
                "embed": {
                    "provider": "ghost",
                    "model": "bge-m3",
                    "dim": 1024,
                    "provider_env": "FMD_EMBED_PROVIDER",
                }
            },
            "budgets": {"daily_usd": 1, "per_run_usd": 1, "hard_stop": True, "cache": True},
        }
    )
    embedder = EmbeddingScorer.from_models(models)
    assert embedder.available is False
    assert embedder.mode == "embeddings_unavailable"
    assert "FMD_EMBED_PROVIDER" in embedder.reason


def test_unavailable_embedder_keeps_lexical_features_and_surfaces_mode() -> None:
    from fmd.config.models import ModelsConfig
    from fmd.er.similarity import EmbeddingScorer

    models = ModelsConfig.model_validate(
        {
            "providers": {},
            "profiles": {
                "embed": {
                    "provider": "ghost",
                    "model": "bge-m3",
                    "dim": 1024,
                    "provider_env": "FMD_EMBED_PROVIDER",
                }
            },
            "budgets": {"daily_usd": 1, "per_run_usd": 1, "hard_stop": True, "cache": True},
        }
    )
    embedder = EmbeddingScorer.from_models(models)
    scored = score_pair(
        Mention(text="Alpha Academy"),
        normalise_name("Alpha Academy", CFG.lexicon),
        EntityCandidate(entity_id=40, canonical_name="Alpha Academy"),
        config=SimilarityConfig(use_embeddings=True),
        lexicon=CFG.lexicon,
        embedder=embedder,
    )
    assert scored.features.embedding is None
    assert scored.features.token_jaccard == 1.0  # documented lexical features intact
    assert scored.similarity_method == similarity_backend()


def test_runtime_backend_failure_fails_fast_and_becomes_visible() -> None:
    from fmd.er.similarity import EmbeddingScorer

    embedder = EmbeddingScorer(backend=_BrokenBackend())
    assert embedder.mode == "embeddings"
    assert embedder.similarity("a", "b") is None
    assert embedder.available is False, "one failure deactivates the path for the run"
    assert embedder.mode == "embeddings_unavailable"
    assert "connection refused" in embedder.reason
    assert embedder.similarity("a", "b") is None  # no retry churn, still lexical


def test_router_adapter_serves_pairwise_similarity_with_a_vector_cache() -> None:
    from fmd.er.similarity import RouterEmbeddings

    class _CountingMock(MockEmbeddings):
        calls = 0

        def embed(self, texts):
            type(self).calls += 1
            return super().embed(texts)

    adapter = RouterEmbeddings(_CountingMock(dim=64))
    first = adapter.similarity("Alpha Academy", "Alpha Academy")
    assert first == 1.0
    adapter.similarity("Alpha Academy", "Beta Institute")
    adapter.similarity("Alpha Academy", "Beta Institute")  # cached pair
    assert _CountingMock.calls == 2, "one embed call per distinct text, not per pair"
