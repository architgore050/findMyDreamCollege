"""LLM network behaviour: scaled timeouts, retries, single-slot, outage abort.

Offline only: every HTTP test uses ``httpx.MockTransport`` and an injected
sleeper. The mid-batch test uses an in-process provider that fails after one
success, proving prior results survive and the rest fail fast instead of each
waiting out a read timeout.
"""

from __future__ import annotations

import threading
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from fmd.config.models import ModelsConfig, ProfileConfig, ProviderConfig
from fmd.llm import (
    ModelRouter,
    NetworkOutage,
    OpenAICompatibleProvider,
    ProviderError,
)
from fmd.llm.providers import ProviderResult
from fmd.net.policy import RetryPolicy, llm_timeouts


class Answer(BaseModel):
    value: int


class Sleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _profile(max_tokens: int = 256) -> ProfileConfig:
    return ProfileConfig(provider="local", model="m", guided_json=True, max_tokens=max_tokens)


def _provider(handler, *, base_url: str = "https://llm.test/v1", **kwargs):
    return OpenAICompatibleProvider(
        "local",
        ProviderConfig(kind="openai_compatible", base_url=base_url),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _complete(provider, *, max_tokens: int = 256) -> ProviderResult:
    return provider.complete(
        model="m",
        system="s",
        user="u",
        schema=Answer,
        profile=_profile(max_tokens),
        prompt_sha256="x",
        purpose="bulk_extract",
    )


def test_completion_timeout_scales_with_max_tokens() -> None:
    small = llm_timeouts(128)
    large = llm_timeouts(4096)
    huge = llm_timeouts(1_000_000)

    assert small.read < large.read < huge.read
    assert small.read == 38.0  # 30 s floor + 128/16
    assert large.read == 286.0  # 30 + 4096/16
    assert huge.read == 600.0  # hard cap: no five-minute-plus call
    assert small.connect == 4.0


def test_429_retry_after_is_honoured_then_succeeds() -> None:
    sleeper = Sleeper()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "1.5"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"value": 7}'}}], "model": "m"},
        )

    provider = _provider(handler, retry=RetryPolicy(max_attempts=2), sleeper=sleeper)
    result = _complete(provider)

    assert result.content == {"value": 7}
    assert calls["n"] == 2
    assert sleeper.delays == [1.5]


def test_5xx_is_retried_and_then_reported() -> None:
    sleeper = Sleeper()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    provider = _provider(handler, retry=RetryPolicy(max_attempts=3), sleeper=sleeper)
    with pytest.raises(ProviderError) as excinfo:
        _complete(provider)

    assert excinfo.value.reason == "server_error"
    assert calls["n"] == 3
    assert len(sleeper.delays) == 2


def test_dns_failure_is_not_retried_and_does_not_echo_the_key() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError(
            "Name or service not known",
            request=request,
        )

    provider = OpenAICompatibleProvider(
        "api",
        ProviderConfig(
            kind="openai_compatible", base_url="https://llm.test/v1", api_key="sk-secret"
        ),
        transport=httpx.MockTransport(handler),
        retry=RetryPolicy(max_attempts=3),
        sleeper=Sleeper(),
    )
    with pytest.raises(ProviderError) as excinfo:
        _complete(provider)

    assert excinfo.value.reason == "dns_failure"
    assert calls["n"] == 1
    assert "sk-secret" not in str(excinfo.value)


def test_local_endpoint_requests_are_single_slot() -> None:
    active = {"now": 0, "max": 0}
    lock = threading.Lock()
    serial = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        with serial:
            pass
        # Give a second thread time to overlap if the slot were not exclusive.
        threading.Event().wait(0.05)
        with lock:
            active["now"] -= 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"value": 1}'}}], "model": "m"},
        )

    provider = _provider(handler, base_url="http://localhost:8010/v1")
    errors: list[BaseException] = []

    def run() -> None:
        try:
            _complete(provider, max_tokens=64)
        except BaseException as exc:  # pragma: no cover - failure diagnostic
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert active["max"] == 1, "local llama.cpp endpoint must be single-slot"


class FlakyProvider:
    """One good answer, then the endpoint disappears."""

    name = "local"
    kind = "openai_compatible"
    supports_guided_json = True

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **kwargs: Any) -> ProviderResult:
        self.calls += 1
        if self.calls == 1:
            return ProviderResult(
                content={"value": 1}, raw='{"value": 1}', model="m", provider="local"
            )
        raise ProviderError("timeout", "local: request timed out")

    def probe(self) -> tuple[bool, str]:
        return True, "ok"


def _router(provider: FlakyProvider) -> ModelRouter:
    models = ModelsConfig.model_validate(
        {
            "providers": {
                "local": {
                    "kind": "openai_compatible",
                    "base_url": "http://localhost:8010/v1",
                    "supports_guided_json": True,
                }
            },
            "profiles": {
                "bulk_extract": {
                    "provider": "local",
                    "model": "m",
                    "guided_json": False,
                    "max_tokens": 256,
                }
            },
            "budgets": {
                "daily_usd": 5,
                "per_run_usd": 2,
                "hard_stop": True,
                "cache": False,
            },
            "fallbacks": [],
        }
    )
    return ModelRouter(models, providers={"local": provider})


def test_mid_batch_outage_preserves_prior_result_and_aborts() -> None:
    provider = FlakyProvider()
    router = _router(provider)

    first = router.call("bulk_extract", Answer, "s", "u1", "ref-1")
    assert first.value == 1
    assert provider.calls == 1

    # Three consecutive transient failures trip the breaker.
    for _ in range(3):
        with pytest.raises(ProviderError):
            router.call("bulk_extract", Answer, "s", "u2", "ref-2")

    assert provider.calls == 4
    state = router.network_state()
    assert state["outage"] is not None
    assert state["consecutive_failures"] == 3

    # The next item fails immediately without another network attempt.
    with pytest.raises(NetworkOutage) as excinfo:
        router.call("bulk_extract", Answer, "s", "u3", "ref-3")
    assert provider.calls == 4
    assert "unverified" in str(excinfo.value) or "Aborting" in str(excinfo.value)

    # The claim produced before the outage is untouched (still returned/valid).
    assert first.value == 1
