"""Embedding backends: config/env selection, HTTP contract, offline modes.

Offline only: HTTP tests use ``httpx.MockTransport``; the CPU backend is never
asked to load weights (its encoder is stubbed); the mock backend is pure
hashing. These tests prove the three acceptance behaviours of the W1 embed
wave:

1. backend selection is config/env only (``FMD_EMBED_PROVIDER``);
2. contract violations (dimension mismatch, count mismatch) are rejected, not
   truncated, and ordering/batching/empty-input behave deterministically;
3. an absent backend/model records ``embeddings_unavailable`` with the exact
   variable to fix — the lexical fallback is visible, not silent.

The insight-selection half proves the semantic-diversity gate consumes the
same embedder contract and records ``embedding_mode`` in the persisted
selection dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from fmd.config import load_registry
from fmd.config.models import ModelsConfig, ProviderConfig
from fmd.llm import ModelRouter
from fmd.llm.doctor import doctor_report
from fmd.llm.embeddings import (
    EmbeddingError,
    EmbeddingsUnavailable,
    MockEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformersEmbeddings,
    build_embedding_backend,
)
from fmd.llm.providers import ProviderError
from fmd.net.policy import RetryPolicy
from fmd.score.insights import Insight, select

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


# --------------------------------------------------------------------- selection


def _models(**profile_extra: Any) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "providers": {
                "embed_openai": {
                    "kind": "openai_compatible",
                    "base_url": "http://embed.test/v1",
                    "base_url_env": "FMD_EMBED_URL",
                    "api_key_env": "FMD_EMBED_KEY",
                    "requires_api_key": False,
                },
                "mock": {"kind": "replay", "fixtures_dir": "packages/pipeline/tests/fixtures/llm"},
            },
            "profiles": {
                "embed": {
                    "provider": "embed_openai",
                    "model": "bge-m3",
                    "dim": 1024,
                    "batch_size": "2",  # YAML interpolation yields strings
                    "model_env": "FMD_EMBED_MODEL",
                    "provider_env": "FMD_EMBED_PROVIDER",
                }
            },
            "budgets": {"daily_usd": 1, "per_run_usd": 1, "hard_stop": True, "cache": True},
        }
    )


def test_registry_defaults_to_the_cpu_backend_and_bge_m3(monkeypatch) -> None:
    for var in (
        "FMD_EMBED_PROVIDER",
        "FMD_EMBED_MODEL",
        "FMD_EMBED_DIM",
        "FMD_EMBED_BATCH",
        "FMD_EMBED_DEVICE",
    ):
        monkeypatch.delenv(var, raising=False)
    models = load_registry(CONFIG_DIR, env={}).models
    embed = models.profiles["embed"]
    assert embed.provider == "cpu"
    assert embed.model == "BAAI/bge-m3"
    assert embed.dim == 1024
    assert embed.degrade_reason == "embeddings_unavailable"
    assert embed.required is False
    backend = build_embedding_backend(models)
    assert isinstance(backend, SentenceTransformersEmbeddings)
    assert backend.device == "cpu"
    assert backend.model == "BAAI/bge-m3"
    assert backend.dim == 1024
    assert backend.batch_size == 32


def test_provider_is_switchable_by_env_without_code_changes(monkeypatch) -> None:
    # Provider/model selection flows through the registry's env interpolation;
    # tuning knobs (dim/batch/device) are read from the process environment.
    monkeypatch.setenv("FMD_EMBED_DIM", "64")
    env = {
        "FMD_EMBED_PROVIDER": "mock",
        "FMD_EMBED_MODEL": "mock-embedder",
    }
    models = load_registry(CONFIG_DIR, env=env)
    backend = build_embedding_backend(models.models)
    assert isinstance(backend, MockEmbeddings)
    assert backend.model == "mock-embedder"
    assert backend.dim == 64


def test_http_backend_selected_from_a_declared_provider() -> None:
    backend = build_embedding_backend(_models())
    assert isinstance(backend, OpenAIEmbeddings)
    assert backend.config.base_url == "http://embed.test/v1"
    assert backend.model == "bge-m3"
    assert backend.dim == 1024
    assert backend.batch_size == 2


def test_undeclared_provider_is_a_loud_unavailability() -> None:
    models = _models()
    models.profiles["embed"].provider = "does_not_exist"
    with pytest.raises(EmbeddingsUnavailable) as excinfo:
        build_embedding_backend(models)
    message = str(excinfo.value)
    assert excinfo.value.reason == "embeddings_unavailable"
    assert "does_not_exist" in message
    assert "FMD_EMBED_PROVIDER" in message


def test_non_openai_provider_kind_cannot_serve_embeddings() -> None:
    models = _models()
    models.providers["gemini"] = ProviderConfig(
        kind="gemini", base_url="https://gemini.test", api_key="k"
    )
    models.profiles["embed"].provider = "gemini"
    with pytest.raises(EmbeddingsUnavailable) as excinfo:
        build_embedding_backend(models)
    assert "cannot serve embeddings" in str(excinfo.value)


def test_empty_model_is_a_loud_unavailability() -> None:
    models = _models()
    models.profiles["embed"].model = None
    with pytest.raises(EmbeddingsUnavailable) as excinfo:
        build_embedding_backend(models)
    assert "FMD_EMBED_MODEL" in str(excinfo.value)


def test_http_provider_with_unset_url_names_the_variable() -> None:
    models = load_registry(CONFIG_DIR, env={}).models
    models.profiles["embed"].provider = "embed_openai"
    with pytest.raises(EmbeddingsUnavailable) as excinfo:
        build_embedding_backend(models)
    message = str(excinfo.value)
    assert "FMD_EMBED_URL" in message
    # The default URL must never be the :8010 chat server.
    assert "8010" not in message


# --------------------------------------------------------------------- mock backend


def test_mock_backend_is_deterministic_and_unit_norm() -> None:
    first = MockEmbeddings(dim=64).embed(["Alpha Academy", "Beta Institute"])
    second = MockEmbeddings(dim=64).embed(["Alpha Academy", "Beta Institute"])
    assert first.vectors == second.vectors
    for vector in first.vectors:
        assert len(vector) == 64
        assert abs(sum(value * value for value in vector) - 1.0) < 1e-9
    assert first.vectors[0] != first.vectors[1]
    assert MockEmbeddings(dim=64).embed([]).vectors == ()


def test_identical_text_scores_one_via_router_adapter() -> None:
    from fmd.er.similarity import RouterEmbeddings

    adapter = RouterEmbeddings(MockEmbeddings(dim=128))
    assert adapter.similarity("Government Medical College", "Government Medical College") == 1.0
    other = adapter.similarity("Government Medical College", "KGMU")
    assert 0.0 <= other < 1.0


# --------------------------------------------------------------------- HTTP contract


def _openai_backend(handler, *, dim: int = 4, batch_size: int = 2) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        "embed_openai",
        ProviderConfig(
            kind="openai_compatible",
            base_url="http://embed.test/v1",
            api_key="embed-secret-key",
        ),
        model="bge-m3",
        dim=dim,
        batch_size=batch_size,
        transport=httpx.MockTransport(handler),
        retry=RetryPolicy(max_attempts=1),
        sleeper=lambda seconds: None,
    )


def test_request_shape_headers_and_authorization() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 1.0, 0.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 9},
            },
        )

    backend = _openai_backend(handler)
    batch = backend.embed(["Alpha Academy", "Beta Institute"])
    assert captured["url"] == "http://embed.test/v1/embeddings"
    assert captured["headers"]["authorization"] == "Bearer embed-secret-key"
    assert captured["payload"] == {"model": "bge-m3", "input": ["Alpha Academy", "Beta Institute"]}
    assert batch.vectors == ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
    assert batch.tokens_in == 9
    assert batch.model == "bge-m3"


def test_response_rows_are_reordered_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                ]
            },
        )

    batch = _openai_backend(handler).embed(["Alpha Academy", "Beta Institute"])
    # Deterministic ordering: output i is input i, whatever the server sent.
    assert batch.vectors[0] == (1.0, 0.0, 0.0, 0.0)
    assert batch.vectors[1] == (0.0, 1.0, 0.0, 0.0)


def test_batching_splits_requests_and_preserves_order() -> None:
    requests: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload["input"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": offset, "embedding": [float(len(requests)), float(offset), 0.0, 0.0]}
                    for offset in range(len(payload["input"]))
                ]
            },
        )

    texts = ["a", "b", "c", "d", "e"]
    batch = _openai_backend(handler, batch_size=2).embed(texts)
    assert requests == [["a", "b"], ["c", "d"], ["e"]]
    assert len(batch.vectors) == 5
    assert batch.vectors[0][0] == 1.0  # request 1
    assert batch.vectors[2][0] == 2.0  # request 2
    assert batch.vectors[4][0] == 3.0  # request 3
    assert batch.vectors[3][1] == 1.0  # second row of request 2


def test_empty_input_makes_no_request() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"data": []})

    batch = _openai_backend(handler).embed([])
    assert batch.vectors == ()
    assert batch.tokens_in is None
    assert calls["n"] == 0, "empty input must not touch the network"


def test_dimension_mismatch_is_rejected_not_truncated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5, 0.5]}]})

    with pytest.raises(EmbeddingError) as excinfo:
        _openai_backend(handler, dim=4).embed(["Alpha Academy"])
    assert excinfo.value.reason == "dimension_mismatch"
    assert "expected 4" in str(excinfo.value)


def test_count_mismatch_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                    {"index": 0, "embedding": [0.0, 1.0, 0.0, 0.0]},
                ]
            },
        )

    with pytest.raises(EmbeddingError) as excinfo:
        _openai_backend(handler).embed(["a", "b"])
    assert excinfo.value.reason == "bad_provider_response"


def test_http_errors_do_not_echo_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    with pytest.raises(ProviderError) as excinfo:
        _openai_backend(handler).embed(["Alpha Academy"])
    assert excinfo.value.reason == "http_error"
    assert "embed-secret-key" not in str(excinfo.value)


# --------------------------------------------------------------------- CPU backend (no weights)


class _StubEncoder:
    """Stands in for a loaded SentenceTransformer; no weights, no network."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(
        self,
        texts,
        *,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        import numpy as np

        assert batch_size == 8, "profile batch_size must reach the encoder"
        return np.ones((len(texts), self.dim), dtype=float)


def test_cpu_backend_probe_never_loads_weights() -> None:
    backend = SentenceTransformersEmbeddings(model="BAAI/bge-m3", dim=1024, device="cpu")
    if SentenceTransformersEmbeddings.available():
        ok, note = backend.probe()
        assert ok is True
        assert "installed" in note
        assert backend._encoder is None, "probe must stay a metadata check"


def test_cpu_backend_validates_dimension_from_the_encoder(monkeypatch) -> None:
    backend = SentenceTransformersEmbeddings(dim=1024, batch_size=8)
    monkeypatch.setattr(backend, "_load", lambda: _StubEncoder(dim=768))
    with pytest.raises(EmbeddingError) as excinfo:
        backend.embed(["Alpha Academy", "Beta Institute"])
    assert excinfo.value.reason == "dimension_mismatch"
    assert "768" in str(excinfo.value)
    assert "1024" in str(excinfo.value)


def test_cpu_backend_returns_positional_vectors(monkeypatch) -> None:
    backend = SentenceTransformersEmbeddings(dim=1024, batch_size=8)
    monkeypatch.setattr(backend, "_load", lambda: _StubEncoder(dim=1024))
    batch = backend.embed(["Alpha Academy", "Beta Institute"])
    assert len(batch.vectors) == 2
    assert all(len(vector) == 1024 for vector in batch.vectors)
    assert batch.backend == SentenceTransformersEmbeddings.kind or batch.backend == backend.name
    assert backend.embed([]).vectors == ()


def test_cpu_backend_wraps_weight_load_failure_as_unavailability(monkeypatch) -> None:
    backend = SentenceTransformersEmbeddings(dim=1024)

    def boom() -> None:
        raise OSError("offline: weights not cached")

    monkeypatch.setattr(backend, "_load", boom)
    with pytest.raises(EmbeddingsUnavailable) as excinfo:
        backend.embed(["Alpha Academy"])
    message = str(excinfo.value)
    assert excinfo.value.reason == "embeddings_unavailable"
    assert "weights" in message
    assert "FMD_OFFLINE" in message


# --------------------------------------------------------------------- doctor surface


def test_doctor_reports_embed_mode_and_blockers() -> None:
    models = load_registry(CONFIG_DIR, env={}).models
    report = doctor_report(models)
    row = next(item for item in report["profiles"] if item["profile"] == "embed")
    assert row["provider"] == "cpu"
    assert row["mode"] == "embeddings"
    assert row["embed_backend"] == "cpu"
    assert row["degrade_reason"] == "embeddings_unavailable"
    assert row["required"] is False

    # A resolved-but-unreachable HTTP backend reports the blocker and the mode.
    models.profiles["embed"].provider = "embed_openai"
    report = doctor_report(models, live=False)
    row = next(item for item in report["profiles"] if item["profile"] == "embed")
    assert row["mode"] == "embeddings_unavailable"
    assert any("FMD_EMBED_URL" in blocker for blocker in row["blockers"])
    # Optional profile: it degrades, it does not fail the doctor.
    assert report["required_ok"] is True


def test_doctor_embed_row_is_ping_safe() -> None:
    models = load_registry(CONFIG_DIR, env={}).models
    report = ModelRouter(models).ping()
    row = next(item for item in report if item["profile"] == "embed")
    assert row["provider"] == "cpu"
    assert row["resolved"] is True
    assert json.dumps(report) != ""  # serialisable; secrets stay out by construction


# --------------------------------------------------------------------- insight selection


def _insight(id: str, predicate: str, units: set[str], text: str | None = None) -> Insight:
    return Insight(
        id=id,
        predicate=predicate,
        text=text or f"{predicate} insight",
        coverage=frozenset(units),
    )


class _Embedder:
    available = True

    def __init__(self, fn) -> None:
        self._fn = fn
        self.calls: list[tuple[str, str]] = []

    def similarity(self, left: str, right: str) -> float | None:
        self.calls.append((left, right))
        return self._fn(left, right)


def test_select_without_embedder_records_lexical_mode() -> None:
    selection = select([_insight("a", "outcomes", {"x"})], max_insights=2)
    assert selection.embedding_mode == "embeddings_unavailable"
    assert selection.embedding_note and "lexical" in selection.embedding_note
    assert selection.semantic_duplicates == ()
    payload = selection.as_dict()
    assert payload["embedding_mode"] == "embeddings_unavailable"
    assert payload["semantic_duplicates"] == []


def test_select_with_unavailable_embedder_surfaces_the_reason() -> None:
    from fmd.er.similarity import EmbeddingScorer

    embedder = EmbeddingScorer(
        reason="FMD_EMBED_PROVIDER names provider 'ghost', which is not declared under providers:"
    )
    assert embedder.mode == "embeddings_unavailable"
    baseline = select([_insight("a", "outcomes", {"x"}), _insight("b", "naac", {"y"})])
    gated = select(
        [_insight("a", "outcomes", {"x"}), _insight("b", "naac", {"y"})],
        embedder=embedder,
    )
    assert [item.id for item in gated.selected] == [item.id for item in baseline.selected]
    assert gated.embedding_mode == "embeddings_unavailable"
    assert gated.embedding_note and "FMD_EMBED_PROVIDER" in gated.embedding_note


def test_semantic_duplicate_is_suppressed_and_recorded() -> None:
    text = "Median package 4.2 LPA for 2024 graduates."
    candidates = [
        _insight("a1", "placement_rate", {"u1"}, text=text),
        _insight("a2", "placement_median", {"u2"}, text=text),
    ]
    # Without embeddings, both distinct predicates survive the lexical gate.
    lexical = select(candidates, max_insights=2)
    assert [item.id for item in lexical.selected] == ["a1", "a2"]
    assert lexical.embedding_mode == "embeddings_unavailable"

    embedder = _Embedder(lambda left, right: 1.0 if left == right else 0.0)
    gated = select(candidates, max_insights=2, embedder=embedder)
    assert [item.id for item in gated.selected] == ["a1"]
    assert gated.semantic_duplicates == ("a2",)
    assert gated.embedding_mode == "embeddings"
    assert gated.embedding_note is None
    assert gated.stopped_reason == "semantic_duplicates_suppressed"
    assert gated.as_dict()["semantic_duplicates"] == ["a2"]


def test_below_threshold_text_is_not_suppressed() -> None:
    embedder = _Embedder(lambda left, right: 0.4)
    selection = select(
        [
            _insight("a1", "placement_rate", {"u1"}, text="Median package 4.2 LPA."),
            _insight("a2", "naac_grade", {"u2"}, text="NAAC grade A since 2019."),
        ],
        max_insights=2,
        embedder=embedder,
    )
    assert [item.id for item in selection.selected] == ["a1", "a2"]
    assert selection.semantic_duplicates == ()
    assert selection.embedding_mode == "embeddings"


def test_embedder_error_fails_open_and_stays_visible() -> None:
    embedder = _Embedder(lambda left, right: None)
    selection = select(
        [_insight("a", "outcomes", {"x"}), _insight("b", "naac", {"y"})],
        max_insights=2,
        embedder=embedder,
    )
    assert [item.id for item in selection.selected] == ["a", "b"]
    assert selection.embedding_mode == "embeddings"


def test_mandatory_insights_are_never_semantically_suppressed() -> None:
    text = "Recognition status: recognised"
    mandatory = _insight("recognition_status", "recognition_status", {"r"}, text=text)
    candidates = [_insight("copy", "recognition_mirror", {"x"}, text=text)]
    embedder = _Embedder(lambda left, right: 1.0)
    selection = select(candidates, mandatory=[mandatory], max_insights=2, embedder=embedder)
    assert [item.id for item in selection.mandatory] == ["recognition_status"]
    assert selection.semantic_duplicates == ("copy",)
