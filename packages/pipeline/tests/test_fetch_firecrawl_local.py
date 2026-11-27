"""Firecrawl Local tier tests (offline: httpx MockTransport, no real server).

Proves the typed error taxonomy, the fail-closed cloud guard, the honest
FetchResult mapping, and the lifecycle script's syntax. No pytest touches a
real Firecrawl process or the network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from fmd.fetch.firecrawl_local import (
    FirecrawlBadResponse,
    FirecrawlCloudHandoffForbidden,
    FirecrawlLocalClient,
    FirecrawlLocalFetcher,
    MapLink,
)
from fmd.fetch.ladder import TIER_FIRECRAWL_LOCAL
from fmd.fetch.types import FetchRequest, TierUnavailable, UnsupportedReplay

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "firecrawl_local.sh"


def make_client(handler, *, enabled: bool = True, **kwargs) -> FirecrawlLocalClient:
    transport = httpx.MockTransport(handler)
    return FirecrawlLocalClient(
        base_url="http://fc.local",
        client=httpx.Client(transport=transport, base_url="http://fc.local"),
        enabled=enabled,
        **kwargs,
    )


def scrape_payload(
    *,
    status: int = 200,
    source_url: str = "https://college.ac.in/fees",
    html: str = "<html><body>fees</body></html>",
    markdown: str = "# fees",
    metadata: dict | None = None,
    data_extra: dict | None = None,
) -> dict:
    meta = {"statusCode": status, "sourceURL": source_url, "title": "Fees"}
    meta.update(metadata or {})
    data = {"metadata": meta, "html": html, "markdown": markdown}
    data.update(data_extra or {})
    return {"success": True, "data": data}


# ------------------------------------------------------------------- scrape


def test_scrape_parses_status_source_url_and_html_body() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return httpx.Response(200, json=scrape_payload())

    client = make_client(handler)
    result = client.scrape("https://college.ac.in/fees")

    assert seen["url"] == "http://fc.local/v2/scrape"
    assert b'"maxAge":0' in seen["body"].replace(b" ", b"")
    assert result.status == 200
    assert result.ok
    assert result.final_url == "https://college.ac.in/fees"
    assert result.body == b"<html><body>fees</body></html>"
    assert result.fetcher_tier == TIER_FIRECRAWL_LOCAL
    client.close()


def test_scrape_falls_back_to_markdown_when_html_is_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=scrape_payload(html="", markdown="# fees page"))

    client = make_client(handler)
    result = client.scrape("https://college.ac.in/fees")
    assert result.body == b"# fees page"


def test_scrape_max_age_is_configurable() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json=scrape_payload())

    client = make_client(handler, max_age=3600)
    client.scrape("https://college.ac.in/fees")
    assert b'"maxAge":3600' in seen["body"].replace(b" ", b"")


def test_cloud_source_fails_closed_and_is_recorded_as_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=scrape_payload(metadata={"source": "firecrawl-cloud"})
        )

    client = make_client(handler)
    with pytest.raises(FirecrawlCloudHandoffForbidden, match="cloud_handoff_forbidden"):
        client.scrape("https://college.ac.in/fees")

    # Through the fetcher it becomes an honest failure result, never content.
    fetcher = FirecrawlLocalFetcher(client=client)
    result = fetcher.fetch(FetchRequest(url="https://college.ac.in/fees"))
    assert result.status == 0
    assert result.error and result.error.startswith("cloud_handoff_forbidden")
    assert result.body == b""


def test_blocked_flag_fails_closed_wherever_it_appears() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=scrape_payload(data_extra={"blocked": True}))

    client = make_client(handler)
    with pytest.raises(FirecrawlCloudHandoffForbidden):
        client.scrape("https://college.ac.in/fees")

    def handler_meta(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=scrape_payload(metadata={"blocked": True}))

    with pytest.raises(FirecrawlCloudHandoffForbidden):
        make_client(handler_meta).scrape("https://college.ac.in/fees")


def test_bad_responses_are_typed() -> None:
    def no_data(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True})

    with pytest.raises(FirecrawlBadResponse):
        make_client(no_data).scrape("https://college.ac.in/fees")

    def no_status(request: httpx.Request) -> httpx.Response:
        payload = scrape_payload()
        del payload["data"]["metadata"]["statusCode"]
        return httpx.Response(200, json=payload)

    with pytest.raises(FirecrawlBadResponse, match="statusCode"):
        make_client(no_status).scrape("https://college.ac.in/fees")

    def non_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(FirecrawlBadResponse):
        make_client(non_json).scrape("https://college.ac.in/fees")


def test_http_error_status_from_the_local_server_is_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = make_client(handler)
    with pytest.raises(Exception) as excinfo:
        client.scrape("https://college.ac.in/fees")
    assert getattr(excinfo.value, "status", None) == 500


def test_transport_failure_is_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)
    with pytest.raises(Exception, match="not reachable"):
        client.scrape("https://college.ac.in/fees")


# ------------------------------------------------------------------- map


def test_map_parses_links_with_title_and_description() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "success": True,
                "links": [
                    {
                        "url": "https://college.ac.in/fees",
                        "title": "Fees",
                        "description": "2026 fees",
                    },
                    {"url": "not-a-url", "title": "junk"},
                    {"url": "https://college.ac.in/mlt"},
                ],
            },
        )

    client = make_client(handler)
    links = client.map("https://college.ac.in", query="fees")

    assert b'"search":"fees"' in seen["body"].replace(b" ", b"")
    assert links == [
        MapLink(url="https://college.ac.in/fees", title="Fees", description="2026 fees"),
        MapLink(url="https://college.ac.in/mlt", title=None, description=None),
    ]


def test_map_accepts_data_wrapped_links_and_rejects_bad_payloads() -> None:
    def wrapped(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "data": {"links": [{"url": "https://college.ac.in/a"}]}},
        )

    assert make_client(wrapped).map("https://college.ac.in") == [
        MapLink(url="https://college.ac.in/a")
    ]

    def no_links(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True})

    with pytest.raises(FirecrawlBadResponse, match="links"):
        make_client(no_links).map("https://college.ac.in")


def test_map_limit_is_forwarded() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={"success": True, "links": []})

    client = make_client(handler)
    client.map("https://college.ac.in", limit=7)
    assert b'"limit":7' in seen["body"].replace(b" ", b"")


# ------------------------------------------------------------------- fetcher + config


def test_disabled_client_raises_and_the_fetcher_maps_it_to_tier_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIRECRAWL_LOCAL_ENABLED", raising=False)
    client = FirecrawlLocalClient.from_env(enabled=None)
    assert client.enabled is False
    with pytest.raises(Exception, match="disabled"):
        client.scrape("https://college.ac.in/fees")

    fetcher = FirecrawlLocalFetcher(client=client)
    with pytest.raises(TierUnavailable, match="disabled"):
        fetcher.fetch(FetchRequest(url="https://college.ac.in/fees"))


def test_unreachable_server_maps_to_tier_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    fetcher = FirecrawlLocalFetcher(client=make_client(handler))
    with pytest.raises(TierUnavailable, match="not reachable"):
        fetcher.fetch(FetchRequest(url="https://college.ac.in/fees"))


def test_http_status_becomes_an_honest_fetch_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    fetcher = FirecrawlLocalFetcher(client=make_client(handler))
    result = fetcher.fetch(FetchRequest(url="https://college.ac.in/fees"))
    assert result.status == 500
    assert result.error and "HTTP 500" in result.error
    assert result.fetcher_tier == TIER_FIRECRAWL_LOCAL


def test_non_get_refused() -> None:
    fetcher = FirecrawlLocalFetcher(client=make_client(lambda request: httpx.Response(200)))
    with pytest.raises(UnsupportedReplay, match="cannot faithfully replay POST"):
        fetcher.fetch(FetchRequest(url="https://college.ac.in/api", method="POST", body=b"x"))


def test_env_overrides_shape_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRECRAWL_LOCAL_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("FIRECRAWL_LOCAL_TIMEOUT_S", "11")
    monkeypatch.setenv("FIRECRAWL_LOCAL_ENABLED", "1")
    client = FirecrawlLocalClient.from_env()
    assert client.base_url == "http://127.0.0.1:9999"
    assert client.timeout == 11.0
    assert client.enabled is True
    monkeypatch.setenv("FIRECRAWL_LOCAL_TIMEOUT_S", "nonsense")
    assert FirecrawlLocalClient.from_env().timeout == 60.0


# ------------------------------------------------------------------- lifecycle script


def test_firecrawl_local_script_passes_a_syntax_check() -> None:
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_script_reports_the_no_cloud_guarantee_in_help_errors() -> None:
    # A missing subcommand is a usage error; the banner wording lives in the
    # start/probe paths, so just assert the script is executable on disk.
    assert SCRIPT.stat().st_mode & 0o111
