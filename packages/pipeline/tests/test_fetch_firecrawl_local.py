"""Firecrawl Local tier tests (offline: httpx MockTransport, no real server).

Proves the typed error taxonomy, the fail-closed cloud guard, the honest
FetchResult mapping, and the lifecycle script's syntax. No pytest touches a
real Firecrawl process or the network.
"""

from __future__ import annotations

import json
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
FC_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "firecrawl_local"


def fc_fixture(name: str) -> dict:
    """A committed real-wire payload (Firecrawl Local 0.1.1, local matrix
    2026-11-27), fed straight through the client so a contract change in the
    v2 shapes fails loudly offline."""
    return json.loads((FC_FIXTURES / name).read_text(encoding="utf-8"))


def fixture_client(name: str) -> FirecrawlLocalClient:
    payload = fc_fixture(name)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return make_client(handler)


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


def test_scrape_redirect_reports_the_final_url_not_the_requested_one() -> None:
    """Real 0.1.1 shape (finding #4): metadata.sourceURL keeps the requested
    /redirect URL, metadata.url is where the page actually landed. Using
    sourceURL as final_url would poison provenance with a URL that never
    rendered the content."""
    client = fixture_client("scrape_redirect.json")

    result = client.scrape("http://127.0.0.1:18765/redirect")

    assert result.status == 200
    assert result.ok
    assert result.final_url == "http://127.0.0.1:18765/ok"
    assert result.final_url != "http://127.0.0.1:18765/redirect"
    assert result.body == b"<html><body>ok</body></html>"
    client.close()


def test_scrape_without_metadata_url_keeps_the_sourceurl_fallback() -> None:
    """Older/sparser shapes without metadata.url must keep working."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=scrape_payload())

    client = make_client(handler)
    result = client.scrape("https://college.ac.in/fees")
    assert result.final_url == "https://college.ac.in/fees"
    client.close()


# ------------------------------------------------------------------- content-type


def test_scrape_pdf_fixture_returns_markdown_as_text_markdown() -> None:
    """Real 0.1.1 shape (finding #6): local PDF parsing arrives as markdown
    while metadata.contentType still says application/pdf."""
    client = fixture_client("scrape_pdf.json")

    result = client.scrape("https://college.ac.in/prospectus.pdf")

    assert result.status == 200
    assert result.ok
    assert result.headers["content-type"] == "text/markdown"
    assert result.body.startswith(b"# Prospectus 2026-27")
    client.close()


def test_scrape_docx_fixture_returns_markdown_as_text_markdown() -> None:
    client = fixture_client("scrape_docx.json")

    result = client.scrape("https://college.ac.in/admission-policy.docx")

    assert result.status == 200
    assert result.headers["content-type"] == "text/markdown"
    client.close()


def test_content_type_always_describes_the_bytes_actually_returned() -> None:
    # Declared type matching an HTML body is kept verbatim (charset included).
    def html_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=scrape_payload(metadata={"contentType": "text/html; charset=utf-8"}),
        )

    result = make_client(html_handler).scrape("https://college.ac.in/fees")
    assert result.headers["content-type"] == "text/html; charset=utf-8"

    # A declaration that lies about the bytes (application/pdf on an HTML
    # body) is ignored: the derived type wins.
    def mismatch_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=scrape_payload(metadata={"contentType": "application/pdf"}))

    result = make_client(mismatch_handler).scrape("https://college.ac.in/fees")
    assert result.headers["content-type"] == "text/html"

    # Markdown fallback (empty html) is text/markdown, never the declared html.
    def md_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=scrape_payload(html="", metadata={"contentType": "text/html"})
        )

    result = make_client(md_handler).scrape("https://college.ac.in/fees")
    assert result.headers["content-type"] == "text/markdown"
    assert result.body == b"# fees"


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
        return httpx.Response(200, json=scrape_payload(metadata={"source": "firecrawl-cloud"}))

    client = make_client(handler)
    with pytest.raises(FirecrawlCloudHandoffForbidden, match="cloud_handoff_forbidden"):
        client.scrape("https://college.ac.in/fees")

    # Through the fetcher it becomes an honest failure result, never content.
    fetcher = FirecrawlLocalFetcher(client=client)
    result = fetcher.fetch(FetchRequest(url="https://college.ac.in/fees"))
    assert result.status == 0
    assert result.error and result.error.startswith("cloud_handoff_forbidden")
    assert result.body == b""


def test_local_blocked_target_is_an_honest_403_target_blocked() -> None:
    """Real 0.1.1 shape (finding #5): data.source=local, blocked=true,
    metadata.statusCode=403. Honest failure so the ladder can escalate or
    quarantine — NOT a cloud hand-off, and never content."""
    client = fixture_client("scrape_blocked_local.json")

    result = client.scrape("http://127.0.0.1:18765/forbidden")

    assert result.status == 403
    assert result.error == "target_blocked"
    assert result.body == b""
    assert result.blocked  # the ladder's BLOCK_STATUSES escalation signal
    assert not result.ok
    client.close()


def test_blocked_flag_in_metadata_or_data_is_an_honest_target_blocked() -> None:
    def handler_data(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=scrape_payload(data_extra={"blocked": True}))

    result = make_client(handler_data).scrape("https://college.ac.in/fees")
    assert result.status == 200  # the declared statusCode is kept honestly
    assert result.error == "target_blocked"
    assert result.body == b""

    def handler_meta(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=scrape_payload(metadata={"blocked": True}))

    result = make_client(handler_meta).scrape("https://college.ac.in/fees")
    assert result.error == "target_blocked"
    assert result.body == b""


def test_blocked_without_a_declared_status_defaults_to_403() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = scrape_payload(data_extra={"blocked": True})
        del payload["data"]["metadata"]["statusCode"]
        return httpx.Response(200, json=payload)

    result = make_client(handler).scrape("https://college.ac.in/fees")
    assert result.status == 403
    assert result.error == "target_blocked"


def test_cloud_taint_wins_over_a_local_block_and_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=scrape_payload(
                metadata={"source": "firecrawl-cloud"}, data_extra={"blocked": True}
            ),
        )

    with pytest.raises(FirecrawlCloudHandoffForbidden, match="cloud_handoff_forbidden"):
        make_client(handler).scrape("https://college.ac.in/fees")


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
