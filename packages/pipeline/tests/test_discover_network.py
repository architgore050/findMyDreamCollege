"""Discovery network policy: SearXNG retries, outage abort, bounded sitemaps.

Offline only: ``httpx.MockTransport`` for SearXNG and in-process fakes for
DuckDB. No test reaches the internet.
"""

from __future__ import annotations

import httpx
import pytest

from fmd.discover import common_crawl, sitemaps
from fmd.discover.searxng import SearxngClient, SearxngUnavailable
from fmd.net.policy import NetworkGuard, NetworkUnavailable, RetryPolicy, make_client


class Sleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _client(handler) -> httpx.Client:
    return make_client(
        base_url="https://searx.example",
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )


def test_searxng_retries_503_then_succeeds() -> None:
    sleeper = Sleeper()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "results": [{"url": "https://college.ac.in/placement.pdf", "title": "Placements"}]
            },
        )

    searx = SearxngClient(
        "https://searx.example",
        client=_client(handler),
        retry=RetryPolicy(max_attempts=2),
        sleeper=sleeper,
    )
    refs = searx.search("mlt placements")

    assert calls["n"] == 2
    assert len(sleeper.delays) == 1
    assert [ref.url for ref in refs] == ["https://college.ac.in/placement.pdf"]


def test_searxng_dns_failure_is_one_bounded_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("Name or service not known", request=request)

    searx = SearxngClient(
        "https://searx.example",
        client=_client(handler),
        retry=RetryPolicy(max_attempts=3),
        sleeper=Sleeper(),
    )
    with pytest.raises(SearxngUnavailable, match="dns_failure"):
        searx.search("q")
    assert calls["n"] == 1


def test_searxng_total_outage_aborts_instead_of_empty_results() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("Name or service not known", request=request)

    guard = NetworkGuard(threshold=2)
    searx = SearxngClient(
        "https://searx.example",
        client=_client(handler),
        retry=RetryPolicy(max_attempts=1),
        guard=guard,
        sleeper=Sleeper(),
    )
    with pytest.raises(SearxngUnavailable):
        searx.search("q")
    with pytest.raises(NetworkUnavailable):
        searx.search("q")
    with pytest.raises(NetworkUnavailable):
        searx.search("q")
    assert calls["n"] == 2


def test_sitemap_discovery_stops_after_consecutive_failures() -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        raise ConnectionError("network is down")

    discovery = sitemaps.SitemapDiscovery(fetch, max_consecutive_failures=2)
    refs = discovery.discover([f"https://college.ac.in/sitemap-{i}.xml" for i in range(5)])

    assert refs == []
    assert len(calls) == 2
    assert discovery.aborted_reason is not None
    assert discovery.aborted_reason.startswith("network_unavailable")


def test_sitemap_success_resets_the_failure_streak() -> None:
    calls = {"n": 0}

    def fetch(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("blip")
        return b"<urlset><url><loc>/a</loc></url></urlset>"

    discovery = sitemaps.SitemapDiscovery(fetch, max_consecutive_failures=2)
    refs = discovery.discover(["https://college.ac.in/s1.xml", "https://college.ac.in/s2.xml"])

    assert discovery.aborted_reason is None
    assert [ref.url for ref in refs] == ["https://college.ac.in/a"]


def test_common_crawl_sets_bounded_httpfs_timeouts(monkeypatch) -> None:
    statements: list[str] = []

    class Result:
        def fetchall(self):
            return []

    class Connection:
        def execute(self, sql: str):
            statements.append(sql)
            return Result()

        def close(self) -> None:
            pass

    class DuckDB:
        @staticmethod
        def connect():
            return Connection()

    monkeypatch.setattr(common_crawl, "_import_duckdb", lambda: DuckDB)
    index = common_crawl.CommonCrawlIndex(domains=("ac.in",), pdf_url_patterns=("%.pdf",))
    assert index.discover(limit=1) == []

    assert statements[0] == "INSTALL httpfs"
    assert statements[1] == "LOAD httpfs"
    assert f"SET http_timeout={common_crawl.DUCKDB_HTTP_TIMEOUT_S}" in statements
    assert f"SET http_retries={common_crawl.DUCKDB_HTTP_RETRIES}" in statements
    # The timeout is applied before the query that reads parquet over HTTP.
    select_at = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT"))
    assert statements.index("SET http_timeout=20") < select_at
