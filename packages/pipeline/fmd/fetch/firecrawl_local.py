"""Firecrawl Local tier: the free, self-hosted scrape/map backend (v1/free-web-stack).

`firecrawl-local <https://docs.firecrawl.dev/quickstarts/firecrawl-local>`_ is
a single Rust binary (``cargo install firecrawl-local``) that serves the
Firecrawl v2 API on ``127.0.0.1:3002`` — no database, no queue, no container.
Scrape/crawl/map/PDF parsing all run locally.

**The no-cloud invariant.** The upstream server sends *search* and
blocked-page fallbacks to Firecrawl Cloud by default. This client refuses any
answer that came from the cloud: if any provenance field (top level, ``data``
or ``metadata``) says ``source == "firecrawl-cloud"``, the result fails closed
with :class:`FirecrawlCloudHandoffForbidden` — a typed error, recorded in the
scrape-failure log, never silently accepted. A **local** block
(``blocked == true`` with a local source) is different: it is reported
honestly as ``target_blocked`` with the real status so the ladder can escalate
or quarantine — it is never treated as content, and never as cloud taint.
Operators must start the server with ``FIRECRAWL_CLOUD=off``
(``scripts/firecrawl_local.sh`` does exactly that); an unrecognized
``FIRECRAWL_CLOUD`` value also means "off" upstream.

httpx only — the ``firecrawl`` SDK is deliberately not used (it is cloud-first
and would blur the local-only guarantee).

Config (env, no secrets): ``FIRECRAWL_LOCAL_ENABLED`` (default ``0`` until the
operator starts the server), ``FIRECRAWL_LOCAL_URL`` (default
``http://127.0.0.1:3002``), ``FIRECRAWL_LOCAL_TIMEOUT_S`` (default ``60``),
``FIRECRAWL_LOCAL_MAX_AGE`` (default ``0`` = always fresh; caching is a local
decision, never a cloud lookup).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from fmd.fetch.ladder import TIER_FIRECRAWL_LOCAL
from fmd.fetch.types import (
    FetchRequest,
    FetchResult,
    TierUnavailable,
    UnsupportedReplay,
)
from fmd.net.policy import NetworkGuard

logger = logging.getLogger(__name__)

#: The only source string a local answer may never carry.
CLOUD_SOURCE = "firecrawl-cloud"

DEFAULT_LOCAL_URL = "http://127.0.0.1:3002"
DEFAULT_TIMEOUT_S = 60.0


# --------------------------------------------------------------------------- errors
class FirecrawlLocalError(RuntimeError):
    """Base class: every failure kind this client can report."""


class FirecrawlLocalDisabled(FirecrawlLocalError):
    """``FIRECRAWL_LOCAL_ENABLED`` is off (default until the operator opts in)."""


class FirecrawlLocalUnavailable(FirecrawlLocalError):
    """The local server is not reachable on its base URL."""


class FirecrawlCloudHandoffForbidden(FirecrawlLocalError):
    """FAIL CLOSED: the answer says it came from Firecrawl Cloud (or was
    blocked by the cloud fallback). It is never accepted."""


class FirecrawlBadResponse(FirecrawlLocalError):
    """The server answered but the payload is not a usable v2 shape."""


class FirecrawlHttpStatusError(FirecrawlLocalError):
    """The local server answered with an HTTP error status."""

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = int(status)
        super().__init__(f"firecrawl_local HTTP {self.status}" + (f": {detail}" if detail else ""))


class FirecrawlTransportError(FirecrawlLocalError):
    """httpx could not reach the local server at all."""


# --------------------------------------------------------------------------- types
@dataclass(frozen=True)
class MapLink:
    """One discovered URL from ``/v2/map`` — a lead, never evidence."""

    url: str
    title: str | None = None
    description: str | None = None


# --------------------------------------------------------------------------- client
def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _first_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _scan_provenance(payload: Any, *, _depth: int = 0) -> tuple[bool, bool]:
    """Scan a v2 answer for cloud taint and block flags, wherever they hide.

    Returns ``(cloud, blocked)``. The walk covers the top level, ``data``,
    ``metadata`` and nested known containers (dict/list values, small depth
    cap), so a top-level ``source`` or a flag inside ``data.metadata`` cannot
    slip past the guard (defense in depth over the documented v2 shapes).
    """
    if _depth > 4:
        return False, False
    if isinstance(payload, dict):
        cloud = False
        blocked = False
        source = payload.get("source")
        if isinstance(source, str) and source.strip().lower() == CLOUD_SOURCE:
            cloud = True
        flag = payload.get("blocked")
        if flag is True or (isinstance(flag, str) and flag.strip().lower() == "true"):
            blocked = True
        children = list(payload.values())
    elif isinstance(payload, list):
        cloud = False
        blocked = False
        children = list(payload)
    else:
        return False, False
    for child in children:
        sub_cloud, sub_blocked = _scan_provenance(child, _depth=_depth + 1)
        cloud = cloud or sub_cloud
        blocked = blocked or sub_blocked
    return cloud, blocked


class FirecrawlLocalClient:
    """httpx client for the locally served Firecrawl v2 API.

    Every method is strict about provenance: a response that looks like it
    came from Firecrawl Cloud raises :class:`FirecrawlCloudHandoffForbidden`
    instead of being returned, whatever the payload otherwise contains.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        enabled: bool | None = None,
        max_age: int | None = None,
        client: Any | None = None,
    ) -> None:
        raw_url = base_url or os.environ.get("FIRECRAWL_LOCAL_URL", "") or DEFAULT_LOCAL_URL
        self.base_url = raw_url.rstrip("/")
        try:
            raw_timeout = (
                timeout
                if timeout is not None
                else os.environ.get("FIRECRAWL_LOCAL_TIMEOUT_S", DEFAULT_TIMEOUT_S)
            )
            self.timeout = float(raw_timeout)
        except ValueError:
            self.timeout = DEFAULT_TIMEOUT_S
        self._enabled_override = enabled
        self.max_age = int(
            max_age if max_age is not None else os.environ.get("FIRECRAWL_LOCAL_MAX_AGE", 0)
        )
        self._client = client
        self._owns_client = client is None

    @classmethod
    def from_env(cls, **overrides: Any) -> FirecrawlLocalClient:
        return cls(**overrides)

    @property
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        return _env_flag("FIRECRAWL_LOCAL_ENABLED", "0")

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise FirecrawlLocalDisabled(
                "Firecrawl Local is disabled (FIRECRAWL_LOCAL_ENABLED=0); start it with "
                "`scripts/firecrawl_local.sh start` (FIRECRAWL_CLOUD=off)."
            )

    def _get_client(self):
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------- internals
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON; map transport/HTTP failures onto the typed taxonomy."""
        try:
            response = self._get_client().post(path, json=payload)
        except httpx.HTTPError as exc:
            raise FirecrawlTransportError(
                f"Firecrawl Local not reachable at {self.base_url}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise FirecrawlHttpStatusError(response.status_code, response.text[:200])
        try:
            body = response.json()
        except ValueError as exc:
            raise FirecrawlBadResponse(
                f"firecrawl_bad_response: non-JSON body from {path}"
            ) from exc
        if not isinstance(body, dict):
            raise FirecrawlBadResponse(f"firecrawl_bad_response: {path} response is not an object")
        return body

    # ------------------------------------------------------------------- v2 API
    @staticmethod
    def _final_url(metadata: dict[str, Any], requested: str) -> str:
        """The URL the fetch actually landed on.

        The real API (0.1.1, local matrix 2026-11-27) keeps the *requested*
        URL in ``metadata.sourceURL`` and puts the *final* one — after a 302 —
        in ``metadata.url``. Prefer the final URL, fall back to sourceURL and
        then to the requested URL for older/sparser shapes.
        """
        for key in ("url", "sourceURL"):
            value = metadata.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
        return requested

    @staticmethod
    def _body_and_type(data: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, str]:
        """The body to hand downstream and the content-type that describes IT.

        Local PDF/DOCX parsing arrives as parsed markdown while
        ``metadata.contentType`` still says ``application/pdf`` (real 0.1.1
        shape, 2026-11-27 finding #6) — claiming ``text/html`` for a markdown
        body poisons every downstream consumer. An HTML body is
        ``text/html``; a markdown body (parsed document or the markdown
        fallback) is ``text/markdown``. The declared ``contentType`` is used
        only when its base type matches the bytes actually returned (keeping
        a charset parameter); otherwise the derived type wins.
        """
        html = data.get("html")
        markdown = data.get("markdown")
        if isinstance(html, str) and html:
            body = html
            derived = "text/html"
        elif isinstance(markdown, str) and markdown:
            body = markdown
            derived = "text/markdown"
        else:  # defensive: v2 always returns strings here
            return "", "text/html"
        declared = str(metadata.get("contentType") or "").strip()
        if declared.lower().split(";", 1)[0].strip() == derived:
            return body, declared
        return body, derived

    def scrape(self, url: str) -> FetchResult:
        """One page via ``POST /v2/scrape`` (formats: html + markdown, fresh).

        Cloud-sourced or blocked answers fail closed with a typed error; a
        healthy local answer becomes an honest :class:`FetchResult`.
        """
        self._require_enabled()
        payload = self._post(
            "/v2/scrape",
            {"url": url, "formats": ["html", "markdown"], "maxAge": self.max_age},
        )
        success = payload.get("success")
        if success is not True and "data" not in payload:
            warning = str(payload.get("warning") or payload.get("error") or "unknown error")
            raise FirecrawlBadResponse(f"firecrawl_bad_response: scrape failed: {warning}")
        data = _first_dict(payload, "data")
        if not data:
            raise FirecrawlBadResponse("firecrawl_bad_response: scrape response has no data object")
        cloud, blocked = _scan_provenance(payload)
        if cloud:
            raise FirecrawlCloudHandoffForbidden(
                "cloud_handoff_forbidden: Firecrawl Local answer came from firecrawl-cloud "
                "(top level, data or metadata); refusing it (start the server with "
                "FIRECRAWL_CLOUD=off)"
            )
        metadata = _first_dict(data, "metadata")
        status_raw = metadata.get("statusCode")
        final_url = self._final_url(metadata, url)
        if blocked:
            # A LOCAL block (data.source=local, blocked=true — the real 0.1.1
            # shape for a forbidden target) is an honest target failure, not a
            # cloud hand-off: surface the real status so the ladder can
            # escalate or quarantine, and never return content.
            status = int(status_raw) if status_raw is not None else 403
            return FetchResult(
                url=url,
                final_url=final_url,
                status=status,
                headers={},
                body=b"",
                fetcher_tier=TIER_FIRECRAWL_LOCAL,
                error="target_blocked",
                method="GET",
            )
        if status_raw is None:
            raise FirecrawlBadResponse("firecrawl_bad_response: scrape metadata has no statusCode")
        body, content_type = self._body_and_type(data, metadata)
        return FetchResult(
            url=url,
            final_url=final_url,
            status=int(status_raw),
            headers={"content-type": content_type},
            body=body.encode("utf-8"),
            fetcher_tier=TIER_FIRECRAWL_LOCAL,
            method="GET",
        )

    def map(self, url: str, *, query: str | None = None, limit: int = 50) -> list[MapLink]:
        """Site URLs via ``POST /v2/map`` (local only; leads, never evidence)."""
        self._require_enabled()
        payload: dict[str, Any] = {"url": url, "limit": max(1, int(limit))}
        if query:
            payload["search"] = query
        body = self._post("/v2/map", payload)
        if body.get("success") is not True and "links" not in body:
            warning = str(body.get("warning") or body.get("error") or "unknown error")
            raise FirecrawlBadResponse(f"firecrawl_bad_response: map failed: {warning}")
        links = body.get("links")
        if links is None and isinstance(body.get("data"), dict):
            links = body["data"].get("links")
        if not isinstance(links, list):
            raise FirecrawlBadResponse("firecrawl_bad_response: map response has no links list")
        out: list[MapLink] = []
        for link in links:
            if not isinstance(link, dict):
                continue
            link_url = link.get("url")
            if not isinstance(link_url, str) or not link_url.startswith(("http://", "https://")):
                continue
            title = link.get("title")
            description = link.get("description")
            out.append(
                MapLink(
                    url=link_url,
                    title=title if isinstance(title, str) else None,
                    description=description if isinstance(description, str) else None,
                )
            )
        return out


# --------------------------------------------------------------------------- fetcher
class FirecrawlLocalFetcher:
    """Last local tier of the ladder, around :class:`FirecrawlLocalClient`.

    Disabled or unreachable -> :class:`TierUnavailable` (the ladder records the
    missing tier and escalates past it). Everything else becomes an honest
    :class:`FetchResult` so the scrape-failure log records exactly what was
    lost and why.
    """

    name = TIER_FIRECRAWL_LOCAL

    def __init__(
        self,
        client: FirecrawlLocalClient | None = None,
        *,
        guard: NetworkGuard | None = None,
    ) -> None:
        self._client = client
        self.guard = guard

    def _get_client(self) -> FirecrawlLocalClient:
        if self._client is None:
            self._client = FirecrawlLocalClient.from_env()
        return self._client

    def fetch(self, request: FetchRequest) -> FetchResult:
        method = str(request.method or "GET").upper()
        if method != "GET":
            # A scrape replays one URL; faking a POST body would fetch a page
            # the persisted request never produced. Refuse like every tier.
            raise UnsupportedReplay(
                f"{self.name} cannot faithfully replay {method} requests; "
                "v2 scrape only issues GET. Refusing rather than downgrading."
            )
        if self.guard is not None:
            self.guard.check()
        try:
            return self._get_client().scrape(request.url)
        except (FirecrawlLocalDisabled, FirecrawlLocalUnavailable, FirecrawlTransportError) as exc:
            raise TierUnavailable(f"{self.name} tier unavailable: {exc}") from exc
        except FirecrawlHttpStatusError as exc:
            # A real HTTP answer from the site behind the local renderer.
            return FetchResult(
                url=request.url,
                final_url=request.url,
                status=exc.status,
                fetcher_tier=self.name,
                error=str(exc),
                method=method,
            )
        except (FirecrawlCloudHandoffForbidden, FirecrawlBadResponse) as exc:
            # Fail closed: an unusable or cloud-tainted answer is recorded as a
            # failure, never passed downstream as content.
            return FetchResult(
                url=request.url,
                final_url=request.url,
                status=0,
                fetcher_tier=self.name,
                error=str(exc),
                method=method,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
