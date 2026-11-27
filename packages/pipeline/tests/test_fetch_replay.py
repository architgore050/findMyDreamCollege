"""Fetch-ladder replay tests: method/body/headers survive every tier.

Offline only: httpx uses ``MockTransport``, curl_cffi is replaced in
``sys.modules`` with a recording stand-in, and the MCP browser tier refuses a
non-GET before it ever talks to the server. This is the regression suite for the
GET-only ladder that served NABH page 1 for every POST page.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx
import pytest

from fmd.fetch.ladder import CurlCffiFetcher, FetchLadder, HttpxFetcher
from fmd.fetch.mcp_browser import InvisiblePlaywrightMCPFetcher
from fmd.fetch.politeness import PolitenessManager
from fmd.fetch.robots import RobotsCache
from fmd.fetch.types import FetchRequest, FetchResult, UnsupportedReplay
from fmd.frontier.state import InMemoryDomainStateStore

URL = "https://nabh.co/wp-admin/admin-ajax.php"


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds


def make_ladder(tiers, clock: FakeClock) -> FetchLadder:
    return FetchLadder(
        fetchers=tiers,
        robots=RobotsCache(fetch=lambda url: (404, ""), clock=clock),
        politeness=PolitenessManager(clock=clock, jitter=0.0, store=InMemoryDomainStateStore()),
        clock=clock,
        sleeper=lambda seconds: clock.advance(seconds),
    )


# --------------------------------------------------------------------- httpx tier
def test_get_is_unchanged_and_reports_its_method() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["content"] = request.content
        return httpx.Response(200, content=b"ok")

    clock = FakeClock()
    tier = HttpxFetcher(transport=httpx.MockTransport(handler))
    try:
        result = make_ladder([tier], clock).fetch(URL)
    finally:
        tier.close()

    assert result.status == 200
    assert result.method == "GET"
    assert seen == {"method": "GET", "content": b""}


def test_post_json_body_reaches_mock_with_method_and_content_type() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = request.content
        seen["content-type"] = request.headers.get("content-type")
        seen["accept"] = request.headers.get("accept")
        return httpx.Response(200, json={"page": 2})

    clock = FakeClock()
    tier = HttpxFetcher(transport=httpx.MockTransport(handler))
    request = FetchRequest(
        url=URL,
        method="POST",
        json_body={"action": "get_hospitals", "page": 2},
        headers={"Accept": "application/json"},
    )
    try:
        result = make_ladder([tier], clock).fetch(request)
    finally:
        tier.close()

    assert result.method == "POST"
    assert seen["method"] == "POST"
    assert json.loads(seen["body"]) == {"action": "get_hospitals", "page": 2}
    assert seen["content-type"] == "application/json"
    assert seen["accept"] == "application/json"


def test_mapping_body_is_form_encoded_like_the_persisted_shape() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["content-type"] = request.headers.get("content-type")
        seen["body"] = request.content
        return httpx.Response(200, content=b"{}")

    clock = FakeClock()
    tier = HttpxFetcher(transport=httpx.MockTransport(handler))
    body = {"action": "get_hospitals", "page": 2}
    try:
        make_ladder([tier], clock).fetch(FetchRequest(url=URL, method="POST", body=body))
    finally:
        tier.close()

    assert seen["content-type"] == "application/x-www-form-urlencoded"
    assert seen["body"] == urlencode(body).encode("utf-8")


def test_caller_headers_merge_with_conditionals_not_replace() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        for name in ("if-none-match", "if-modified-since", "referer", "x-trace", "content-type"):
            seen[name] = request.headers.get(name, "")
        return httpx.Response(304)

    clock = FakeClock()
    tier = HttpxFetcher(transport=httpx.MockTransport(handler))
    request = FetchRequest(
        url=URL,
        method="POST",
        body=b"opaque",
        etag='"stored"',
        last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
        referer="https://nabh.co/find-a-healthcare-organisation/",
        headers={
            "If-None-Match": '"stale-from-discovery"',
            "X-Trace": "abc123",
            "Content-Type": "application/octet-stream",
        },
    )
    try:
        result = make_ladder([tier], clock).fetch(request)
    finally:
        tier.close()

    assert result.status == 304
    assert seen["if-none-match"] == '"stored"'  # stored validator wins
    assert seen["if-modified-since"] == "Wed, 01 Jan 2026 00:00:00 GMT"
    assert seen["referer"] == "https://nabh.co/find-a-healthcare-organisation/"
    assert seen["x-trace"] == "abc123"  # caller header survives
    assert seen["content-type"] == "application/octet-stream"  # never overwritten


def test_post_304_preserves_method_and_body() -> None:
    seen: list[tuple[str, bytes, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.content, request.headers.get("if-none-match", "")))
        return httpx.Response(304)

    clock = FakeClock()
    tier = HttpxFetcher(transport=httpx.MockTransport(handler))
    request = FetchRequest(
        url=URL,
        method="POST",
        body={"action": "get_hospitals", "page": 3},
        etag='"v3"',
    )
    try:
        result = make_ladder([tier], clock).fetch(request)
    finally:
        tier.close()

    assert result.status == 304
    assert result.not_modified
    assert result.method == "POST"
    assert seen == [("POST", urlencode({"action": "get_hospitals", "page": 3}).encode(), '"v3"')]


# --------------------------------------------------------------- curl_cffi tier
class _FakeResponse:
    def __init__(self) -> None:
        self.url = URL
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self.content = b'{"page": 2}'


def test_curl_cffi_tier_replays_post(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_request(method, url, **kwargs):
        seen.update(
            {
                "method": method,
                "url": url,
                "data": kwargs.get("data"),
                "content-type": dict(kwargs.get("headers") or {}).get("Content-Type"),
            }
        )
        return _FakeResponse()

    fake_requests = types.SimpleNamespace(request=fake_request)
    fake_module = types.ModuleType("curl_cffi")
    fake_module.requests = fake_requests  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "curl_cffi", fake_module)

    clock = FakeClock()
    tier = CurlCffiFetcher()
    request = FetchRequest(
        url=URL,
        method="POST",
        json_body={"action": "get_hospitals", "page": 2},
    )
    result = make_ladder([tier], clock).fetch(request)

    assert result.method == "POST"
    assert seen["method"] == "POST"
    assert seen["url"] == URL
    assert json.loads(seen["data"]) == {"action": "get_hospitals", "page": 2}
    assert seen["content-type"] == "application/json"


# ----------------------------------------------------------------- mcp browser tier
def test_mcp_browser_refuses_post_instead_of_downgrading() -> None:
    tier = InvisiblePlaywrightMCPFetcher(enabled=True, session_factory=_no_session)
    with pytest.raises(UnsupportedReplay, match="cannot faithfully replay POST"):
        tier.fetch(FetchRequest(url=URL, method="POST", body=b"action=x&page=2"))


def _no_session() -> None:
    raise AssertionError("no session may be created for a non-GET request")


def test_ladder_records_unsupported_replay_instead_of_serving_page_one() -> None:
    clock = FakeClock()
    tier = InvisiblePlaywrightMCPFetcher(enabled=True, session_factory=_no_session)
    ladder = make_ladder([tier], clock)

    result = ladder.fetch(FetchRequest(url=URL, method="POST", body=b"action=x&page=2"))

    assert result.status == 0
    assert result.method == "POST"
    assert result.error is not None and result.error.startswith("unsupported_replay:")
    assert "invisible_playwright_mcp" in (result.error or "")


def test_result_method_is_exposed_for_orchestration() -> None:
    class Tier:
        name = "httpx"

        def fetch(self, request: FetchRequest) -> FetchResult:
            return FetchResult(
                url=request.url,
                final_url=request.url,
                status=200,
                body=b"ok",
                fetcher_tier=self.name,
            )

    clock = FakeClock()
    result = make_ladder([Tier()], clock).fetch(
        FetchRequest(url=URL, method="POST", json_body={"page": 2})
    )

    # The tier did not set it; the ladder exposes the effective method.
    assert result.method == "POST"
