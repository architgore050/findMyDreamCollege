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
    reason = "profile_unresolved"


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
    supports_guided_json: bool

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

    def probe(self) -> tuple[bool, str]: ...


def _post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    name: str,
    timeout: float,
    transport: httpx.BaseTransport | None,
) -> dict[str, Any]:
    """POST JSON and map transport/HTTP failures to provider reasons.

    Errors never echo headers or the request URL: a provider key must not leak
    into logs through an exception message.
    """
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise ProviderError("timeout", f"{name}: request timed out") from exc
    except httpx.HTTPError as exc:
        raise ProviderError("connection_error", f"{name}: connection failed: {exc}") from exc
    if response.status_code >= 400:
        raise ProviderError(
            "http_error",
            f"{name}: HTTP {response.status_code}: {response.text[:500]}",
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderError(
            "bad_provider_response", f"{name}: response was not JSON: {exc}"
        ) from exc
    if not isinstance(body, dict):
        raise ProviderError("bad_provider_response", f"{name}: response JSON must be an object")
    return body


def _probe_get(
    *, url: str, headers: dict[str, str], name: str, timeout: float,
    transport: httpx.BaseTransport | None,
) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.get(url, headers=headers)
    except httpx.TimeoutException:
        return False, "timed out"
    except httpx.HTTPError as exc:
        return False, f"connection failed: {exc}"
    if response.status_code == 401 or response.status_code == 403:
        return False, f"HTTP {response.status_code}: key rejected"
    if response.status_code >= 400:
        return False, f"HTTP {response.status_code}"
    return True, "ok"


def json_instruction(schema: type) -> str:
    """Prompt suffix for providers/profiles without native guided JSON.

    The router validates the parsed object against ``schema`` either way, so a
    provider that only does JSON-instruct degrades honestly instead of failing
    the whole run.
    """
    return (
        "Return exactly one JSON object and nothing else (no prose, no markdown "
        "fences). It must validate against this JSON Schema:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
    )


def _clean_gemini_schema(value: Any, root: dict[str, Any] | None = None) -> Any:
    """Inline ``$ref``\\ s and drop keywords Gemini's responseSchema rejects."""
    root = root if root is not None else value
    if isinstance(value, list):
        return [_clean_gemini_schema(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        target = (root.get("$defs") or {}).get(reference.rsplit("/", 1)[-1])
        if isinstance(target, dict):
            merged = {k: v for k, v in value.items() if k != "$ref"}
            merged.update(target)
            return _clean_gemini_schema(merged, root)
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"$defs", "$schema", "additionalProperties", "title", "default"}:
            continue
        cleaned[key] = _clean_gemini_schema(item, root)
    return cleaned


class OpenAICompatibleProvider:
    """vLLM/SGLang/OpenAI-compatible chat completions with guided JSON."""

    kind = "openai_compatible"
    supports_guided_json = True

    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        timeout: float = 60.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not config.base_url:
            raise ProviderError(
                "unresolved_provider",
                f"provider {name!r} has no base_url; set "
                f"{config.base_url_env or 'its base URL environment variable'} "
                f"(for example {config.base_url_example or 'https://api.openai.com/v1'})",
            )
        self.name = name
        self.config = config
        self.timeout = timeout
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

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
        body = _post_json(
            url=url,
            headers=self._headers(),
            payload=payload,
            name=self.name,
            timeout=self.timeout,
            transport=self.transport,
        )
        try:
            raw = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
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

    def probe(self) -> tuple[bool, str]:
        return _probe_get(
            url=self.config.base_url.rstrip("/") + "/models",
            headers=self._headers(),
            name=self.name,
            timeout=self.timeout,
            transport=self.transport,
        )


class AnthropicProvider:
    """Anthropic Messages API, guided JSON through a single ``respond`` tool."""

    kind = "anthropic"
    supports_guided_json = True

    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        timeout: float = 60.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not config.base_url:
            raise ProviderError(
                "unresolved_provider",
                f"provider {name!r} has no base_url; set "
                f"{config.base_url_env or 'its base URL environment variable'}",
            )
        self.name = name
        self.config = config
        self.timeout = timeout
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        return headers

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
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": profile.max_tokens,
            "temperature": profile.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if profile.guided_json:
            payload["tools"] = [
                {
                    "name": "respond",
                    "description": "Return the structured extraction result.",
                    "input_schema": schema.model_json_schema(),
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": "respond"}
        body = _post_json(
            url=self.config.base_url.rstrip("/") + "/messages",
            headers=self._headers(),
            payload=payload,
            name=self.name,
            timeout=self.timeout,
            transport=self.transport,
        )
        usage = body.get("usage", {})
        blocks = body.get("content")
        if not isinstance(blocks, list):
            raise ProviderError(
                "bad_provider_response", f"{self.name}: response has no content blocks"
            )
        tool_input: dict[str, Any] | None = None
        text_parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                tool_input = block["input"]
            elif block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        if tool_input is not None:
            content = tool_input
            raw = json.dumps(content, ensure_ascii=False)
        else:
            raw = "\n".join(text_parts)
            content = _parse_json_object(raw, self.name)
        return ProviderResult(
            content=content,
            raw=raw,
            model=body.get("model", model),
            provider=self.name,
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
        )

    def probe(self) -> tuple[bool, str]:
        return _probe_get(
            url=self.config.base_url.rstrip("/") + "/models",
            headers=self._headers(),
            name=self.name,
            timeout=self.timeout,
            transport=self.transport,
        )


class GeminiProvider:
    """Google Gemini ``generateContent`` with ``responseSchema`` guided JSON."""

    kind = "gemini"
    supports_guided_json = True

    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        timeout: float = 60.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not config.base_url:
            raise ProviderError(
                "unresolved_provider",
                f"provider {name!r} has no base_url; set "
                f"{config.base_url_env or 'its base URL environment variable'}",
            )
        self.name = name
        self.config = config
        self.timeout = timeout
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            # Header, never the query string: URLs end up in logs.
            headers["x-goog-api-key"] = self.config.api_key
        return headers

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
        url = (
            self.config.base_url.rstrip("/")
            + f"/models/{model}:generateContent"
        )
        generation: dict[str, Any] = {
            "temperature": profile.temperature,
            "maxOutputTokens": profile.max_tokens,
        }
        if profile.guided_json:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = _clean_gemini_schema(schema.model_json_schema())
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        body = _post_json(
            url=url,
            headers=self._headers(),
            payload=payload,
            name=self.name,
            timeout=self.timeout,
            transport=self.transport,
        )
        usage = body.get("usageMetadata", {})
        try:
            parts = body["candidates"][0]["content"]["parts"]
            raw = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "bad_provider_response", f"{self.name}: malformed generateContent response: {exc}"
            ) from exc
        content = _parse_json_object(raw, self.name)
        return ProviderResult(
            content=content,
            raw=raw,
            model=body.get("modelVersion", model),
            provider=self.name,
            tokens_in=usage.get("promptTokenCount"),
            tokens_out=usage.get("candidatesTokenCount"),
        )

    def probe(self) -> tuple[bool, str]:
        return _probe_get(
            url=self.config.base_url.rstrip("/") + "/models",
            headers=self._headers(),
            name=self.name,
            timeout=self.timeout,
            transport=self.transport,
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
    supports_guided_json = False

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

    def probe(self) -> tuple[bool, str]:
        if self.fixtures_dir.is_dir():
            return True, "fixtures present"
        return False, f"fixtures directory missing: {self.fixtures_dir}"


class MockProvider(ReplayProvider):
    """Offline `--mock` provider. Same replay semantics, never touches a network."""

    kind = "replay"


def build_provider(
    name: str,
    config: ProviderConfig,
    *,
    transport: httpx.BaseTransport | None = None,
) -> Provider:
    """Construct the adapter for a provider registry entry (``kind`` selects it)."""
    if config.kind == "replay":
        return MockProvider(name, config) if name == "mock" else ReplayProvider(name, config)
    if config.kind == "openai_compatible":
        return OpenAICompatibleProvider(name, config, transport=transport)
    if config.kind == "anthropic":
        return AnthropicProvider(name, config, transport=transport)
    if config.kind == "gemini":
        return GeminiProvider(name, config, transport=transport)
    raise ProviderError(
        "unresolved_provider",
        f"provider {name!r} has unknown kind {config.kind!r}; expected "
        "openai_compatible|anthropic|gemini|replay",
    )

