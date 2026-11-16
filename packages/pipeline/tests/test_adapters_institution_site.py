from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, failed_result, fixture_text, ok_result

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.institution_site import InstitutionSiteAdapter
from fmd.discover.base import DocumentRef

NIRF_LISTING = "institution_site_nirf_listing_snippet.html"
SEED = "https://mtct.ac.in/nirf/"


@pytest.fixture
def adapter(registry):
    return InstitutionSiteAdapter(registry.sources.by_id("institution_sites"))


def test_document_refs_are_scored_and_pdf_only(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NIRF_LISTING), base_url=SEED)
    urls = [ref.url for ref in refs]
    assert urls
    assert all(url.lower().endswith(".pdf") for url in urls)
    assert not any(url.endswith((".docx", ".xlsx")) for url in urls)
    assert not any("wp-login" in url for url in urls)
    assert any("NIRF-2026-College-Category-Data.pdf" in url for url in urls)
    # Highest-scoring link first, deterministically.
    scores = [ref.metadata["link_score"] for ref in refs]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(ref, DocumentRef) for ref in refs)
    assert all(ref.source_id == "institution_sites" for ref in refs)


def test_login_javascript_and_mailto_are_hard_skips(adapter) -> None:
    html = (
        '<a href="javascript:void(0)">Placement report</a>'
        '<a href="mailto:principal@example.ac.in">Placement</a>'
        '<a href="#placement">Placement</a>'
        '<a href="/wp-login.php?redirect=placement.pdf">Login</a>'
        '<a href="/files/placement-2025.pdf">Placement report</a>'
    )
    refs = adapter.parse_listing(html, base_url="https://example.ac.in/")
    assert [ref.url for ref in refs] == ["https://example.ac.in/files/placement-2025.pdf"]


def test_navigation_stays_on_the_registrable_domain(adapter) -> None:
    nav = adapter.parse_navigation(fixture_text(NIRF_LISTING), base_url=SEED)
    assert nav
    assert all(url.startswith("https://") for url in nav)
    assert not any("wp-login" in url for url in nav)
    assert not any(url.startswith("javascript:") for url in nav)
    # The fixture links an off-site social profile in spirit only; nothing here
    # may leave mtct.ac.in.
    assert all("mtct.ac.in" in url for url in nav)


def test_bounded_crawl_yields_documents_and_survives_404_pages(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder({SEED: ok_result(SEED, fixture_text(NIRF_LISTING))})
    refs = list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[SEED]))
    assert refs
    assert all(ref.metadata["found_on"] == SEED for ref in refs)
    # Depth is bounded: navigation links are attempted but never unbounded.
    assert len(ladder.calls) <= 1 + int(source.params.max_pages_per_domain)


def test_robots_disallow_abstains(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder(ok_result(SEED, fixture_text(NIRF_LISTING)), allowed=False)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[SEED]))
    assert excinfo.value.reason == "robots_disallow"


def test_empty_but_valid_page_is_not_a_failure(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder(ok_result(SEED, "<html><body><p>Nothing published yet</p></body></html>"))
    assert list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[SEED])) == []


def test_unreachable_domain_abstains(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder(failed_result(SEED, status=0, error="timeout"))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[SEED]))
    assert excinfo.value.reason == "unreachable"


def test_no_domain_abstains_without_fetching(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder(ok_result(SEED, fixture_text(NIRF_LISTING)))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(InstitutionSiteAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "no_domain"
    assert ladder.calls == []


def test_yield_stop_rule_bounds_a_dry_crawl(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    page = ok_result(SEED, fixture_text(NIRF_LISTING))
    ladder = FakeLadder({SEED: page})
    adapter = InstitutionSiteAdapter(source, ladder=ladder)
    refs = list(adapter.fetch(domains=[SEED]))
    assert refs
    # With max_depth=2 the crawl may visit nav pages, but the page budget is hard.
    assert len(ladder.calls) <= 1 + int(source.params.max_pages_per_domain)


def test_link_scoring_prefers_placement_and_aqar_over_news(adapter) -> None:
    html = (
        '<a href="/news/annual-report-2025.pdf">Annual report</a>'
        '<a href="/files/aqar-2024.pdf">AQAR 2024</a>'
        '<a href="/files/placement-report-2025.pdf">Placement Report</a>'
    )
    refs = adapter.parse_listing(html, base_url="https://example.ac.in/")

    urls = [ref.url for ref in refs]
    assert urls[-1].endswith("annual-report-2025.pdf")
    assert {url.rsplit("/", 1)[-1] for url in urls[:2]} == {
        "aqar-2024.pdf",
        "placement-report-2025.pdf",
    }
    assert refs[0].metadata["link_score"] >= refs[-1].metadata["link_score"]
    # The weights are registry data, not constants in this module.
    assert "link_weights" in adapter.source.params.model_dump(exclude_none=True)


def test_navigation_skips_news_events_and_gallery_pages(adapter) -> None:
    html = (
        '<a href="/about/gallery">Photo Gallery</a>'
        '<a href="/about/news">News</a>'
        '<a href="/about/events">Events</a>'
        '<a href="/naac/ssr">NAAC SSR</a>'
        '<a href="/downloads">Downloads</a>'
    )
    nav = adapter.parse_navigation(html, base_url="https://example.ac.in/")

    assert not any(token in url for url in nav for token in ("gallery", "news", "events"))
    assert nav[0].endswith("/naac/ssr")  # evidence-adjacent sections come first
