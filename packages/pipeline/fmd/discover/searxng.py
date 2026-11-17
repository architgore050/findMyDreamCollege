"""SearXNG JSON client (brief §7, §8).

Results are *leads only*. Query templates are registry data: they are read from
``config/sources.yaml`` (the source's optional ``config.query_templates``) or
supplied by the caller. This module never hardcodes a country, state or exam.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from fmd.discover.base import DocumentRef, dedupe_refs, params_from_source
from fmd.net.policy import (
    DEFAULT_RETRY,
    NetworkGuard,
    NetworkUnavailable,
    RetryPolicy,
    Timeouts,
    classify_exception,
    make_client,
    request_with_retries,
)

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_ID = "searxng"


class SearxngUnavailable(RuntimeError):
    """SearXNG is not reachable or returned a non-JSON body."""


def templates_from_source(source: Any) -> list[str]:
    """Read ``params.query_templates`` from a registry or seeded source entry.

    Missing templates are a registry gap, reported by the loader as a config
    error, never hardcoded here.
    """
    config = params_from_source(source)
    templates = config.get("query_templates")
    if templates is None and isinstance(source, Mapping):
        legacy = source.get("config")
        if isinstance(legacy, Mapping):
            templates = legacy.get("query_templates")
    return [str(item) for item in templates or []]


class SearxngClient:
    def __init__(
        self,
        base_url: str,
        *,
        query_templates: Sequence[str] = (),
        client: Any | None = None,
        source_id: str = DEFAULT_SOURCE_ID,
        language: str | None = None,
        time_range: str | None = None,
        results_per_query: int = 10,
        timeout: float = 20.0,
        guard: NetworkGuard | None = None,
        retry: RetryPolicy | None = None,
        sleeper=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.query_templates = list(query_templates)
        self._client = client
        self._owns_client = client is None
        self.source_id = source_id
        self.language = language
        self.time_range = time_range
        self.results_per_query = results_per_query
        self.timeout = timeout
        self.guard = guard
        self.retry = retry or DEFAULT_RETRY
        self._sleeper = sleeper

    @classmethod
    def from_source(cls, source: Any, **kwargs: Any) -> SearxngClient:
        """Build from a registry/seeded source entry; templates stay in config."""
        base_url = getattr(source, "base_url", None)
        source_id = getattr(source, "id", None)
        if isinstance(source, Mapping):
            base_url = base_url or source.get("base_url")
            source_id = source_id or source.get("id")
        if not base_url:
            raise ValueError("searxng source has no base_url in the registry")
        kwargs.setdefault("query_templates", templates_from_source(source))
        kwargs.setdefault("source_id", str(source_id or DEFAULT_SOURCE_ID))
        return cls(str(base_url), **kwargs)

    # ------------------------------------------------------------------- client
    def _get_client(self):
        if self._client is None:
            from fmd.settings import get_settings

            self._client = make_client(
                base_url=self.base_url,
                timeouts=Timeouts(connect=4.0, read=self.timeout, write=20.0, pool=4.0),
                headers={"User-Agent": get_settings().user_agent},
                follow_redirects=False,
            )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------- query
    def search(self, query: str) -> list[DocumentRef]:
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if self.language:
            params["language"] = self.language
        if self.time_range:
            params["time_range"] = self.time_range
        try:
            response = request_with_retries(
                self._get_client(),
                "GET",
                "/search",
                retry=self.retry,
                params=params,
                guard=self.guard,
                **({"sleep": self._sleeper} if self._sleeper is not None else {}),
            )
            response.raise_for_status()
            payload = response.json()
        except NetworkUnavailable:
            # Total outage: the stage must abort, not record one empty query
            # per template.
            raise
        except httpx.HTTPStatusError as exc:
            raise SearxngUnavailable(
                f"SearXNG query failed ({self.base_url}): HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            fault = classify_exception(exc)
            raise SearxngUnavailable(
                f"SearXNG query failed ({self.base_url}): {fault.kind}"
            ) from exc
        except ValueError as exc:
            raise SearxngUnavailable(f"SearXNG returned a non-JSON body ({self.base_url})") from exc
        except Exception as exc:
            raise SearxngUnavailable(f"SearXNG query failed ({self.base_url}): {exc}") from exc

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise SearxngUnavailable("SearXNG response has no `results` list")
        refs: list[DocumentRef] = []
        for result in results[: self.results_per_query]:
            if not isinstance(result, Mapping):
                continue
            url = result.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            refs.append(
                DocumentRef(
                    url=url,
                    source_id=self.source_id,
                    discovery_method="searxng",
                    query=query,
                    metadata={
                        "title": result.get("title"),
                        "engine": result.get("engine"),
                        "score": result.get("score"),
                    },
                )
            )
        return refs

    def render(self, template: str, values: Mapping[str, object]) -> str:
        try:
            return template.format(**values)
        except KeyError as exc:
            raise ValueError(
                f"query template {template!r} needs a value for {exc.args[0]!r}"
            ) from exc

    def discover(
        self,
        values: Mapping[str, object] | None = None,
        *,
        templates: Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        """Render every configured template and return deduped leads."""
        values = values or {}
        templates = list(templates if templates is not None else self.query_templates)
        if not templates:
            raise ValueError(
                "no SearXNG query templates configured. Add `config.query_templates` "
                "to the searxng source entry (registry data, not code)."
            )
        refs: list[DocumentRef] = []
        for template in templates:
            query = self.render(str(template), values)
            refs.extend(self.search(query))
        return dedupe_refs(refs)
