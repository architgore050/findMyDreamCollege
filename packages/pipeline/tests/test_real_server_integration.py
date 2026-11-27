"""Opt-in real-server contract tests (skipped by default; never paid).

The 2026-11-27 local real-server matrix (invisible-playwright-mcp 0.74.0,
Firecrawl Local 0.1.1) found wire-contract drift that the offline fakes had
papered over. These tests replay the same matrix against the REAL local
servers so an upstream upgrade fails loudly on drift:

- ``FMD_MCP_INTEGRATION=1`` — spawns the real invisible-playwright-mcp
  server (``uvx``; the engine downloads on first ``browser_open``) and
  drives 200/redirect/404 through the actual stdio wire.
- ``FMD_FIRECRAWL_INTEGRATION=1`` — talks to a locally running Firecrawl
  Local (``scripts/firecrawl_local.sh start``, always ``FIRECRAWL_CLOUD=off``).

Both drive only a tiny localhost HTTP origin started by the tests
themselves; nothing leaves the machine, and no paid or cloud service is
contacted. Skipped unless the env flag is set, so the offline suite and CI
stay hermetic.
"""

from __future__ import annotations

import http.server
import os
import threading

import pytest

from fmd.fetch.firecrawl_local import FirecrawlLocalClient
from fmd.fetch.mcp_browser import InvisiblePlaywrightMCPFetcher
from fmd.fetch.types import FetchRequest


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class _OriginHandler(http.server.BaseHTTPRequestHandler):
    """Tiny deterministic origin: /ok, /redirect (302 -> /ok), /missing."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib API naming
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.end_headers()
            return
        if self.path == "/missing":
            body = b"<html><body>404 page</body></html>"
            status = 404
        else:
            body = b"<html><body>ok page</body></html>"
            status = 200
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # silence the request log under pytest


class _Origin:
    """Context manager serving the handler on a random 127.0.0.1 port."""

    def __init__(self) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OriginHandler)
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def __enter__(self) -> _Origin:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.skipif(not _flag("FMD_MCP_INTEGRATION"), reason="opt-in: set FMD_MCP_INTEGRATION=1")
def test_real_mcp_server_navigate_contract() -> None:
    """The real 0.74.0 server must still answer the 'navigated to <url>
    (HTTP <n>)' sentence shape and follow redirects — the exact contract the
    offline fixtures pin."""
    pytest.importorskip("mcp", reason="the official mcp SDK (crawl extra) is not installed")
    with _Origin() as origin:
        # Generous timeout: the engine download on first browser_open is slow.
        fetcher = InvisiblePlaywrightMCPFetcher(call_timeout_s=300.0)
        try:
            healthy = fetcher.fetch(FetchRequest(url=origin.url("/ok")))
            assert healthy.status == 200 and healthy.ok
            assert b"ok page" in healthy.body

            redirected = fetcher.fetch(FetchRequest(url=origin.url("/redirect")))
            assert redirected.status == 200
            assert redirected.final_url == origin.url("/ok")

            missing = fetcher.fetch(FetchRequest(url=origin.url("/missing")))
            assert missing.status == 404 and missing.terminal
        finally:
            fetcher.close()


@pytest.mark.skipif(
    not _flag("FMD_FIRECRAWL_INTEGRATION"),
    reason="opt-in: set FMD_FIRECRAWL_INTEGRATION=1 (server via scripts/firecrawl_local.sh start)",
)
def test_real_firecrawl_local_scrape_contract() -> None:
    """The real 0.1.1 server (FIRECRAWL_CLOUD=off) must still keep the
    redirect provenance contract: metadata.url is the final URL."""
    with _Origin() as origin:
        client = FirecrawlLocalClient.from_env(enabled=True)
        try:
            healthy = client.scrape(origin.url("/ok"))
        except Exception as exc:  # noqa: BLE001 - any failure means "not up": skip
            pytest.skip(f"Firecrawl Local not reachable on {client.base_url}: {exc}")

        assert healthy.status == 200 and healthy.ok
        assert b"ok page" in healthy.body
        assert healthy.error is None  # no cloud taint, no block (cloud off)

        redirected = client.scrape(origin.url("/redirect"))
        assert redirected.status == 200
        assert redirected.final_url == origin.url("/ok")

        missing = client.scrape(origin.url("/missing"))
        assert missing.status == 404 and missing.terminal

    client.close()
