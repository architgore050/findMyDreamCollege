"""Provider abstraction and concrete providers (brief §15)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from fmd.config.models import ProfileConfig, ProviderConfig


class LLMError(RuntimeError):
    """Base class for router/provider failures."""


class ProviderError(LLMError):
    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class BudgetExhausted(LLMError):
    pass


class FixtureMissing(LLMError):
    pass


class ProfileUnresolved(LLMError):
    pass


@dataclass(frozen=True)
class ProviderResult:
    content: dict[str, Any]
    raw: str
    model: str
    provider: str
    tokens_in: int | None = None
    tokens_out: int | None = None


class Provider(Protocol):
    name: str
    kind: str

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type,
        profile: ProfileConfig,
        prompt_sha256: str,
        purpose: str,
    ) -> ProviderResult: ...


class OpenAICompatibleProvider:
    """vLLM/SGLang/OpenAI-compatible chat completions with guided JSON."""

    kind = "openai_compatible"

    def __init__(self, name: str, config: ProviderConfig, timeout: float = 60.0) -> None:
        if not config.base_url:
            raise ProviderError(
                "unresolved_provider",
                f"provider {name!r} has no base_url; set the environment variable its "
                "config references (see .env.example)",
            )
        self.name = name
        self.config = config
        self.timeout = timeout

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type,
        profile: ProfileConfig,
        prompt_sha256: str,
        purpose: str,
    ) -> ProviderResult:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
        }
        if profile.guided_json:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("timeout", f"{self.name}: request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "connection_error", f"{self.name}: connection failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise ProviderError(
                "http_error",
                f"{self.name}: HTTP {response.status_code}: {response.text[:500]}",
            )
        try:
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "bad_provider_response", f"{self.name}: malformed chat completion: {exc}"
            ) from exc
        content = _parse_json_object(raw, self.name)
        return ProviderResult(
            content=content,
            raw=raw,
            model=body.get("model", model),
            provider=self.name,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )


def _parse_json_object(raw: str, provider: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            "json_schema_violation",
            f"{provider}: response was not JSON ({exc}); first 300 chars: {raw[:300]!r}",
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderError("json_schema_violation", f"{provider}: response JSON must be an object")
    return parsed


def _resolve_fixtures_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Return the repo-relative interpretation so the error message is useful.
    return Path(__file__).resolve().parents[3] / path


class ReplayProvider:
    """Deterministic fixture replay; the offline/CI path (brief §15, §19).

    Fixtures are JSON files named ``<prompt_sha256>.json`` under ``fixtures_dir``
    (optionally under a ``<purpose>/`` subdirectory). Each contains the response
    object plus optional usage metadata::

        {"content": {...}, "tokens_in": 120, "tokens_out": 40}
    """

    kind = "replay"

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config
        if not config.fixtures_dir:
            raise ProviderError(
                "unresolved_provider", f"replay provider {name!r} has no fixtures_dir"
            )
        self.fixtures_dir = _resolve_fixtures_dir(config.fixtures_dir)

    def _candidate_paths(self, purpose: str, prompt_sha256: str) -> list[Path]:
        return [
            self.fixtures_dir / purpose / f"{prompt_sha256}.json",
            self.fixtures_dir / f"{prompt_sha256}.json",
        ]

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type,
        profile: ProfileConfig,
        prompt_sha256: str,
        purpose: str,
    ) -> ProviderResult:
        candidates = self._candidate_paths(purpose, prompt_sha256)
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            expected = " or ".join(str(c) for c in candidates)
            raise FixtureMissing(
                f"no recorded response for purpose={purpose!r} prompt_sha256={prompt_sha256}. "
                f"Expected a fixture at {expected}. Record one by running the pipeline with "
                f"the replay provider and saving the response object as "
                f'{{"content": {{...}}, "tokens_in": 0, "tokens_out": 0}}.'
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "bad_fixture", f"{self.name}: cannot read fixture {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or "content" not in payload:
            raise ProviderError(
                "bad_fixture",
                f"{self.name}: fixture {path} must be an object with a `content` key",
            )
        content = payload["content"]
        if not isinstance(content, dict):
            raise ProviderError(
                "json_schema_violation", f"{self.name}: fixture content must be an object"
            )
        return ProviderResult(
            content=content,
            raw=json.dumps(content, ensure_ascii=False, sort_keys=True),
            model=payload.get("model", model),
            provider=self.name,
            tokens_in=payload.get("tokens_in"),
            tokens_out=payload.get("tokens_out"),
        )


class MockProvider(ReplayProvider):
    """Offline `--mock` provider. Same replay semantics, never touches a network."""

    kind = "replay"
