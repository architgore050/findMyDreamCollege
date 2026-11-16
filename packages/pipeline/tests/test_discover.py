from __future__ import annotations

import httpx
import pytest

from fmd.discover import common_crawl
from fmd.discover.base import DiscoveryUnavailable, DocumentRef, dedupe_refs
from fmd.discover.common_crawl import CommonCrawlIndex
from fmd.discover.searxng import SearxngClient, templates_from_source
from fmd.discover.sitemaps import SitemapDiscovery


def searx_client(results_payload: dict) -> tuple[SearxngClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=results_payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="http://searxng:8080", transport=transport)
    return (
        SearxngClient(
            "http://searxng:8080",
            query_templates=["{institution} placement report", "{institution} BSc MLT"],
            client=client,
        ),
        seen,
    )


def test_searxng_emits_leads_only_and_dedupes() -> None:
    payload = {
        "results": [
            {"url": "https://college.ac.in/placement.pdf", "title": "Placements"},
            {"url": "https://college.ac.in/placement.pdf", "title": "duplicate"},
            {"url": "https://other.ac.in/fees", "title": "Fees"},
            {"url": "not-a-url", "title": "bad"},
        ]
    }
    client, requests = searx_client(payload)

    refs = client.discover({"institution": "Example College"})

    assert [ref.url for ref in refs] == [
        "https://college.ac.in/placement.pdf",
        "https://other.ac.in/fees",
    ]
    assert all(isinstance(ref, DocumentRef) for ref in refs)
    assert all(ref.is_lead for ref in refs)
    assert all(ref.discovery_method == "searxng" for ref in refs)
    assert requests[0].url.params["format"] == "json"
    assert requests[0].url.params["q"] == "Example College placement report"


def test_searxng_requires_templates() -> None:
    client = SearxngClient("http://searxng:8080", query_templates=[])
    with pytest.raises(ValueError, match="query templates"):
        client.discover({"institution": "x"})


def test_templates_come_from_registry_config() -> None:
    source = {"params": {"query_templates": ["{institution} seat matrix"]}}
    assert templates_from_source(source) == ["{institution} seat matrix"]
    assert templates_from_source({"config": {"query_templates": ["legacy"]}}) == ["legacy"]
    assert templates_from_source({}) == []


def test_searxng_templates_come_from_the_committed_registry(registry) -> None:
    source = registry.sources.by_id("searxng")
    assert source is not None and source.params is not None
    templates = templates_from_source(source)
    assert templates
    assert all("{institution}" in template for template in templates)

    client = SearxngClient.from_source(source)
    assert client.query_templates == templates
    assert client.base_url == source.base_url


def test_common_crawl_query_maps_rows_to_refs() -> None:
    def runner(index_glob, domains, limit):
        assert domains == ("ac.in",)
        return [
            ("https://a.ac.in/report.pdf",),
            ("https://a.ac.in/report.pdf",),
            ("https://b.gov.in/seat-matrix.pdf",),
        ]

    index = CommonCrawlIndex(
        query_runner=runner,
        domains=["ac.in"],
        pdf_url_patterns=["%.pdf", "%.pdf?%"],
    )
    refs = index.discover(limit=10)

    assert [ref.url for ref in refs] == [
        "https://a.ac.in/report.pdf",
        "https://b.gov.in/seat-matrix.pdf",
    ]
    assert all(ref.discovery_method == "common_crawl_index" for ref in refs)


def test_common_crawl_is_built_entirely_from_registry_params(registry) -> None:
    source = registry.sources.by_id("common_crawl_index")
    assert source is not None and source.params is not None

    index = CommonCrawlIndex.from_source(source)
    assert index.domains == tuple(source.params.domain_suffixes)
    assert index.pdf_url_patterns == tuple(source.params.pdf_url_patterns)
    assert index.index_glob == source.params.index_glob
    assert index.source_id == "common_crawl_index"

    sql = index._sql(index.domains, 5)
    for suffix in source.params.domain_suffixes:
        assert f"url LIKE '%.{suffix}/%'" in sql
    for pattern in source.params.pdf_url_patterns:
        assert f"url LIKE '{pattern}'" in sql


def test_common_crawl_requires_registry_params() -> None:
    index = CommonCrawlIndex(query_runner=lambda *args: [])
    with pytest.raises(DiscoveryUnavailable, match="domain_suffixes"):
        index.discover()

    index = CommonCrawlIndex(query_runner=lambda *args: [], domains=["ac.in"])
    with pytest.raises(DiscoveryUnavailable, match="pdf_url_patterns"):
        index.discover()


def test_common_crawl_degrades_without_duckdb(monkeypatch) -> None:
    def missing():
        raise DiscoveryUnavailable("Common Crawl discovery needs duckdb")

    monkeypatch.setattr(common_crawl, "_import_duckdb", missing)
    index = CommonCrawlIndex(domains=["ac.in"], pdf_url_patterns=["%.pdf"])
    with pytest.raises(DiscoveryUnavailable, match="duckdb"):
        index.discover()


def test_sitemap_expands_index_and_filters_domain() -> None:
    pages = {
        "https://college.ac.in/sitemap.xml": """<?xml version="1.0"?>
            <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <sitemap><loc>https://college.ac.in/pages.xml</loc></sitemap>
            </sitemapindex>""",
        "https://college.ac.in/pages.xml": """<?xml version="1.0"?>
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://college.ac.in/about</loc></url>
              <url><loc>https://college.ac.in/placement.pdf</loc></url>
              <url><loc>https://evil.example/report.pdf</loc></url>
            </urlset>""",
    }
    discovery = SitemapDiscovery(pages.__getitem__)

    refs = discovery.discover(["https://college.ac.in/sitemap.xml"], domain="college.ac.in")

    assert [ref.url for ref in refs] == [
        "https://college.ac.in/about",
        "https://college.ac.in/placement.pdf",
    ]
    assert all(ref.discovery_method == "sitemap" for ref in refs)


def test_sitemap_is_bounded() -> None:
    body = (
        "<urlset>"
        + "".join(f"<url><loc>https://college.ac.in/p{index}</loc></url>" for index in range(50))
        + "</urlset>"
    )
    discovery = SitemapDiscovery(lambda url: body, max_urls=5)

    refs = discovery.discover(["https://college.ac.in/sitemap.xml"])
    assert len(refs) == 5


def test_dedupe_refs_is_stable() -> None:
    refs = [
        DocumentRef("https://a.ac.in/x", source_id="s"),
        DocumentRef("https://a.ac.in/x", source_id="s2"),
        DocumentRef("https://a.ac.in/y", source_id="s"),
    ]
    assert [ref.url for ref in dedupe_refs(refs)] == [
        "https://a.ac.in/x",
        "https://a.ac.in/y",
    ]


def test_document_ref_carries_conditional_get_and_priority_hints() -> None:
    ref = DocumentRef(
        "https://a.ac.in/x",
        source_id="s",
        etag='"v1"',
        last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
        priority=7,
    )

    from fmd.adapters import DocumentRef as AdapterDocumentRef

    assert isinstance(ref, AdapterDocumentRef)
    assert (ref.etag, ref.last_modified, ref.priority) == (
        '"v1"',
        "Wed, 01 Jan 2026 00:00:00 GMT",
        7,
    )
    # Existing constructors keep working: every new field is optional.
    assert DocumentRef("https://a.ac.in/y").priority == 0
