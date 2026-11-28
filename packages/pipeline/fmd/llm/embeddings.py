"""Embedding backends for the ``embed`` profile (brief §15; W1 embed-provider).

The 26B llama.cpp server on :8010 is a *chat* model: its GGUF is not an
embedding model and it does not serve ``POST /v1/embeddings``. The embed path
therefore never defaults to it. Two backends are provided and selected by
config/env — never by code branches at the call sites:

* ``cpu`` (default) — bge-m3 in-process via sentence-transformers, on CPU.
  Zero VRAM: the text model keeps its measured 1,153 MiB headroom at 64k
  (runbook §2.8) and the one-GPU-tenant rule (runbook §1) stays intact.
* any declared ``openai_compatible`` provider — ``POST {base_url}/embeddings``
  in the OpenAI shape, which llama.cpp (``llama-server --embedding``) and
  hosted embedding APIs both implement. ``config/models.yaml`` declares the
  embed-only provider ``embed_openai`` for this; chat profiles must not use it.

Everything here fails loudly: when the configured backend or model is absent,
:func:`build_embedding_backend` raises :class:`EmbeddingsUnavailable` (reason
``embeddings_unavailable``) naming the exact env variable to fix. Callers
record that reason and take the documented lexical fallback — they never fall
back silently. Contract validation (dimension mismatch, count mismatch,
ordering) raises :class:`EmbeddingError` instead of truncating.

No weights are downloaded and no model is loaded at import time, at probe
time, or in tests; the CPU backend loads weights lazily on first ``embed()``.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fmd.config.models import ModelsConfig, ProfileConfig, ProviderConfig
from fmd.llm.providers import (
    LLMError,
    _endpoint_slot,
    _post_json,
    _probe_get,
)
from fmd.net.policy import NetworkGuard, RetryPolicy, Timeouts
from fmd.obs.metrics import emit as metrics_emit

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBatch",
    "EmbeddingError",
    "EmbeddingsUnavailable",
    "MockEmbeddings",
    "OpenAIEmbeddings",
    "SentenceTransformersEmbeddings",
    "build_embedding_backend",
    "describe_embed_profile",
]

#: Reason token recorded (never swallowed) when the embedding backend or model
#: is absent. This is the string the runbook and doctor surface.
UNAVAILABLE_REASON = "embeddings_unavailable"

#: In-process backend selector names (the ``embed`` profile's ``provider``).
#: Anything else must name a provider declared under ``providers:``.
CPU_BACKEND_NAMES = frozenset({"cpu", "sentence_transformers", "st"})
MOCK_BACKEND_NAMES = frozenset({"mock"})

_DEFAULT_DIM = 1024
_DEFAULT_BATCH = 32


class EmbeddingsUnavailable(LLMError):
    """The configured embedding backend or model is absent.

    ``reason`` is always ``embeddings_unavailable``; the message names the exact
    env variable(s) to set. Optional-tier semantics: callers record the reason,
    take the documented lexical fallback, and surface the mode — they do not
    silently proceed as if embeddings ran.
    """

    reason = UNAVAILABLE_REASON


class EmbeddingError(LLMError):
    """Contract violation from an embedding backend.

    ``reason`` is ``dimension_mismatch`` / ``bad_provider_response`` / an HTTP
    fault reason. A dimension mismatch is *rejected*, never truncated.
    """

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class EmbeddingBatch:
    """Vectors for one ``embed()`` call, positionally aligned with the input."""

    vectors: tuple[tuple[float, ...], ...]
    model: str
    backend: str
    dim: int
    tokens_in: int | None = None


class EmbeddingBackend:
    """Base class: ``embed(texts)`` returns one vector per input, in order."""

    name: str = "base"
    kind: str = "base"
    dim: int = _DEFAULT_DIM

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise NotImplementedError

    def probe(self) -> tuple[bool, str]:
        raise NotImplementedError


def _validate_batch(
    *,
    vectors: Sequence[Sequence[float]],
    texts: Sequence[str],
    dim: int,
    backend: str,
    model: str,
) -> EmbeddingBatch:
    """Shared contract checks: count, per-vector dimension, ordering is positional."""
    if len(vectors) != len(texts):
        raise EmbeddingError(
            "bad_provider_response",
            f"{backend}: expected {len(texts)} vectors for {len(texts)} inputs, got {len(vectors)}",
        )
    out: list[tuple[float, ...]] = []
    for index, vector in enumerate(vectors):
        values = tuple(float(value) for value in vector)
        if len(values) != dim:
            raise EmbeddingError(
                "dimension_mismatch",
                f"{backend}: vector {index} for model {model!r} has dimension "
                f"{len(values)}, expected {dim}. Set config/models.yaml "
                f"profiles.embed.dim to match the model, or use a model with "
                f"the configured dimension — vectors are never truncated.",
            )
        out.append(values)
    return EmbeddingBatch(vectors=tuple(out), model=model, backend=backend, dim=dim)


class OpenAIEmbeddings(EmbeddingBackend):
    """OpenAI-compatible ``POST {base_url}/embeddings``.

    Works for llama.cpp (``llama-server --embedding --pooling cls``) and for
    hosted embedding APIs. Responses are re-sorted by the returned ``index``,
    so deterministic ordering does not depend on server behaviour. Requests to
    local endpoints are single-flight (same slot discipline as chat), because
    the host serves embeddings beside — never inside — the 26B tenant.
    """

    kind = "openai_compatible"

    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        *,
        model: str,
        dim: int = _DEFAULT_DIM,
        batch_size: int = _DEFAULT_BATCH,
        transport: Any | None = None,
        guard: NetworkGuard | None = None,
        retry: RetryPolicy | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not config.base_url:
            raise EmbeddingsUnavailable(
                f"{config.base_url_env or 'the provider base URL'} is empty: set it "
                f"to an OpenAI-compatible embeddings endpoint (for example "
                f"{config.base_url_example or 'http://localhost:8011/v1'})"
            )
        self.name = name
        self.config = config
        self.model = model
        self.dim = dim
        self.batch_size = max(1, int(batch_size))
        self.transport = transport
        self.guard = guard
        self.retry = retry
        self._sleeper = sleeper

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _post_chunk(self, chunk: list[str]) -> tuple[list[list[float]], int | None]:
        url = self.config.base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.model, "input": chunk}
        body = _post_json(
            url=url,
            headers=self._headers(),
            payload=payload,
            name=self.name,
            timeouts=Timeouts(connect=4.0, read=120.0, write=60.0, pool=4.0),
            retry=self.retry,
            transport=self.transport,
            guard=self.guard,
            sleeper=self._sleeper,
        )
        data = body.get("data")
        if not isinstance(data, list):
            raise EmbeddingError(
                "bad_provider_response",
                f"{self.name}: embeddings response has no `data` list",
            )
        rows: list[tuple[int, list[float]]] = []
        for item in data:
            if not isinstance(item, dict) or "embedding" not in item:
                raise EmbeddingError(
                    "bad_provider_response",
                    f"{self.name}: embeddings response rows must carry `embedding`",
                )
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(chunk):
                raise EmbeddingError(
                    "bad_provider_response",
                    f"{self.name}: embeddings response row has invalid index "
                    f"{index!r} for a batch of {len(chunk)}",
                )
            rows.append((index, item["embedding"]))
        if len({index for index, _ in rows}) != len(rows) or len(rows) != len(chunk):
            raise EmbeddingError(
                "bad_provider_response",
                f"{self.name}: expected {len(chunk)} unique embedding rows, got {len(rows)}",
            )
        rows.sort(key=lambda row: row[0])  # deterministic: input order, not server order
        usage = body.get("usage", {})
        tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        return [vector for _, vector in rows], tokens

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(vectors=(), model=self.model, backend=self.name, dim=self.dim)
        started = time.perf_counter()
        vectors: list[list[float]] = []
        tokens_in = 0
        has_tokens = False
        url = self.config.base_url.rstrip("/") + "/embeddings"
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            with _endpoint_slot(url):
                chunk_vectors, chunk_tokens = self._post_chunk(chunk)
            vectors.extend(chunk_vectors)
            if chunk_tokens is not None:
                has_tokens = True
                tokens_in += int(chunk_tokens)
        metrics_emit(
            "embeddings",
            self.name,
            device="cpu",
            model=self.model,
            provider=self.kind,
            documents=len(texts),
            batch_size=self.batch_size,
            duration_ms=int((time.perf_counter() - started) * 1000),
            tokens_in=tokens_in if has_tokens else None,
            status="ok",
        )
        batch = _validate_batch(
            vectors=vectors,
            texts=texts,
            dim=self.dim,
            backend=self.name,
            model=self.model,
        )
        return EmbeddingBatch(
            vectors=batch.vectors,
            model=batch.model,
            backend=batch.backend,
            dim=batch.dim,
            tokens_in=tokens_in if has_tokens else None,
        )

    def probe(self) -> tuple[bool, str]:
        return _probe_get(
            url=self.config.base_url.rstrip("/") + "/models",
            headers=self._headers(),
            name=self.name,
            transport=self.transport,
        )


class SentenceTransformersEmbeddings(EmbeddingBackend):
    """bge-m3 in-process on CPU via sentence-transformers (the default backend).

    Weights load lazily on the first ``embed()`` call and never in tests or
    probes. ``probe()`` only checks that the package is importable, so a doctor
    run stays a metadata check, not a multi-gigabyte download.
    """

    kind = "sentence_transformers"
    name = "cpu"

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-m3",
        dim: int = _DEFAULT_DIM,
        device: str = "cpu",
        batch_size: int = _DEFAULT_BATCH,
    ) -> None:
        self.model = model
        self.dim = dim
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._encoder: Any | None = None

    @staticmethod
    def available() -> bool:
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None

    def _offline(self) -> bool:
        return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("FMD_OFFLINE") == "1"

    def _load(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"device": self.device}
            # Honour offline mode without assuming the kwarg exists in the
            # installed sentence-transformers version.
            if (
                self._offline()
                and "local_files_only" in inspect.signature(SentenceTransformer.__init__).parameters
            ):
                kwargs["local_files_only"] = True
            self._encoder = SentenceTransformer(self.model, **kwargs)
        return self._encoder

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(vectors=(), model=self.model, backend=self.name, dim=self.dim)
        try:
            encoder = self._load()
        except ImportError as exc:
            raise EmbeddingsUnavailable(
                "sentence-transformers is not installed: run `uv sync --all-extras` "
                "in packages/pipeline, or set FMD_EMBED_PROVIDER to a declared "
                "openai_compatible embeddings provider"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - weight-load failure is a loud absence
            raise EmbeddingsUnavailable(
                f"embedding model {self.model!r} could not be loaded on "
                f"{self.device}: {exc}. In offline mode (FMD_OFFLINE=1 / "
                "HF_HUB_OFFLINE=1) the weights must already be cached locally; "
                "prefetch them once while online (runbook §8)."
            ) from exc
        import numpy as np

        started = time.perf_counter()
        array = encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        metrics_emit(
            "embeddings",
            self.name,
            device=self.device,
            model=self.model,
            provider=self.kind,
            documents=len(texts),
            batch_size=self.batch_size,
            duration_ms=int((time.perf_counter() - started) * 1000),
            status="ok",
        )
        vectors: list[list[float]] = [row.tolist() for row in np.asarray(array, dtype=float)]
        return _validate_batch(
            vectors=vectors,
            texts=texts,
            dim=self.dim,
            backend=self.name,
            model=self.model,
        )

    def probe(self) -> tuple[bool, str]:
        if self.available():
            return True, f"sentence-transformers installed (device={self.device})"
        return False, "sentence-transformers not installed: run `uv sync --all-extras`"


class MockEmbeddings(EmbeddingBackend):
    """Deterministic offline backend for tests/CI — the ``mock`` provider.

    Vectors are derived from SHA-256 digests of ``(model, text)``; no network,
    no weights, no server. Identical text always yields the identical
    (unit-norm) vector across processes and runs.
    """

    kind = "replay"
    name = "mock"

    def __init__(self, *, model: str = "mock-embedder", dim: int = _DEFAULT_DIM) -> None:
        self.model = model
        self.dim = dim

    @staticmethod
    def _vector(text: str, dim: int) -> tuple[float, ...]:
        values: list[float] = []
        counter = 0
        while len(values) < dim:
            digest = hashlib.sha256(f"{text}\x00{counter}".encode()).digest()
            for offset in range(0, len(digest), 2):
                if len(values) >= dim:
                    break
                sample = int.from_bytes(digest[offset : offset + 2], "big")
                values.append(sample / 65535.0 - 0.5)
            counter += 1
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return tuple(value / norm for value in values)

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return _validate_batch(
            vectors=[self._vector(text, self.dim) for text in texts],
            texts=texts,
            dim=self.dim,
            backend=self.name,
            model=self.model,
        )

    def probe(self) -> tuple[bool, str]:
        return True, "deterministic mock backend (offline)"


def _profile_int(profile: ProfileConfig, key: str, default: int, env_name: str) -> int:
    value = getattr(profile, key, None)
    env_value = os.environ.get(env_name)
    raw = env_value if env_value is not None else value
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise EmbeddingsUnavailable(
            f"{env_name if env_value is not None else 'profiles.embed.' + key} must be "
            f"an integer, got {raw!r}"
        ) from exc


def build_embedding_backend(
    models: ModelsConfig,
    *,
    profile_name: str = "embed",
) -> EmbeddingBackend:
    """Resolve the ``embed`` profile to a backend, by config/env only.

    Raises :class:`EmbeddingsUnavailable` (never falls back silently) with a
    remediation message naming the exact env variable when the profile, the
    provider, the model or the URL is absent/unusable.
    """
    profile = models.profiles.get(profile_name)
    if profile is None:
        raise EmbeddingsUnavailable(
            f"profile {profile_name!r} is not declared in config/models.yaml"
        )
    if not profile.model:
        raise EmbeddingsUnavailable(
            f"{profile.model_env or 'the embed model variable'} is empty: set it to "
            f"an embedding model id (for example {profile.model_example or 'BAAI/bge-m3'})"
        )
    dim = _profile_int(profile, "dim", _DEFAULT_DIM, "FMD_EMBED_DIM")
    batch_size = _profile_int(profile, "batch_size", _DEFAULT_BATCH, "FMD_EMBED_BATCH")
    model = profile.model
    provider_name = profile.provider

    if provider_name in CPU_BACKEND_NAMES:
        device = os.environ.get("FMD_EMBED_DEVICE", "cpu")
        return SentenceTransformersEmbeddings(
            model=model, dim=dim, device=device, batch_size=batch_size
        )
    if provider_name in MOCK_BACKEND_NAMES:
        return MockEmbeddings(model=model, dim=dim)

    config = models.providers.get(provider_name)
    if config is None:
        known = ", ".join(sorted(models.providers)) or "<none>"
        raise EmbeddingsUnavailable(
            f"{profile.provider_env or 'the embed provider variable'} names provider "
            f"{provider_name!r}, which is not declared under providers: in "
            f"config/models.yaml. Declared providers: {known}. Use 'cpu' (default, "
            "in-process bge-m3) or 'embed_openai' (POST /v1/embeddings)."
        )
    if config.kind != "openai_compatible":
        raise EmbeddingsUnavailable(
            f"provider {provider_name!r} has kind {config.kind!r}, which cannot serve "
            "embeddings. Embeddings need an openai_compatible provider "
            "(POST /v1/embeddings) or the 'cpu' backend."
        )
    return OpenAIEmbeddings(
        provider_name,
        config,
        model=model,
        dim=dim,
        batch_size=batch_size,
    )


def describe_embed_profile(
    models: ModelsConfig,
    profile: ProfileConfig,
    *,
    live: bool = False,
    transport: Any | None = None,
) -> dict[str, Any]:
    """Diagnostics for the ``embed`` profile; same key shape as doctor rows.

    A live probe for the CPU backend checks the package import only — it never
    loads weights and never touches the network. For an HTTP backend it is the
    usual cheap reachability GET.
    """
    blockers: list[str] = []
    remediation: list[str] = []
    backend: EmbeddingBackend | None = None
    try:
        backend = build_embedding_backend(models, profile_name="embed")
    except EmbeddingsUnavailable as exc:
        blockers.append(str(exc))
        remediation.append(str(exc))
    except EmbeddingError as exc:  # defensive: construction does not validate shapes
        blockers.append(f"{exc.reason}: {exc}")
        remediation.append(f"{exc.reason}: {exc}")

    reachable: bool | None = None
    reach_note: str | None = None
    if live:
        if backend is None:
            reachable, reach_note = False, "backend not resolved"
        else:
            reachable, reach_note = backend.probe()
            if not reachable and reach_note:
                remediation.append(f"live probe failed: {reach_note}")

    provider_cfg = models.providers.get(profile.provider)
    mode = UNAVAILABLE_REASON if backend is None else "embeddings"
    in_process = backend is not None and backend.kind in {"sentence_transformers", "replay"}
    usable = not blockers and (reachable is not False if live else True)
    # The in-process backends have no endpoint: the `cpu` provider entry in
    # models.yaml is a registry placemarker, not a URL. Report a display string
    # (never None — the doctor table formats base_url) and never name
    # FMD_EMBED_DEVICE_URL as a missing variable, it is not one.
    if in_process:
        base_url = f"in-process ({getattr(backend, 'device', 'cpu')})"[:34]
        base_url_env = None
        key_present = False
        key_required = False
        api_key_env = None
        resolved = True
    else:
        base_url = provider_cfg.base_url if provider_cfg else None
        base_url_env = provider_cfg.base_url_env if provider_cfg else None
        key_present = bool(provider_cfg.api_key) if provider_cfg else False
        key_required = bool(provider_cfg.requires_api_key) if provider_cfg else False
        api_key_env = provider_cfg.api_key_env if provider_cfg else None
        resolved = bool(provider_cfg and provider_cfg.base_url)
    return {
        "profile": "embed",
        "provider": profile.provider,
        "kind": backend.kind if backend is not None else "unknown",
        "base_url": base_url,
        "base_url_resolved": resolved,
        "base_url_env": base_url_env,
        "key_present": key_present,
        "key_required": key_required,
        "api_key_env": api_key_env,
        "model": profile.model,
        "model_env": profile.model_env,
        "guided_json_expected": False,
        "guided_json_declared": False,
        "reachable": reachable,
        "reachability_note": reach_note,
        "required": profile.required,
        "degrade_reason": profile.degrade_reason or UNAVAILABLE_REASON,
        "usable": usable,
        "blockers": blockers,
        "remediation": remediation,
        "embed_backend": backend.name if backend is not None else None,
        "embed_dim": getattr(profile, "dim", _DEFAULT_DIM),
        "mode": mode,
    }
