from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, failed_result, fixture_text, ok_result

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.health_university import HealthUniversityAdapter

RUHS_HTML = "health_university_ruhs_listing_snippet.html"
SITEMAP_XML = "health_university_tnmgrmu_sitemap_snippet.xml"
JS_SHELL = "js_shell_sample.html"
RUHS_URL = "https://ruhsraj.org/affiliated-colleges/"
SITEMAP_URL = "https://www.tnmgrmu.ac.in/wp-sitemap-posts-page-1.xml"


@pytest.fixture
def adapter(registry):
    return HealthUniversityAdapter(registry.sources.by_id("state_health_universities"))


def test_college_select_options_are_parsed(adapter) -> None:
    colleges = adapter.parse_colleges(fixture_text(RUHS_HTML))
    assert len(colleges) >= 30
    # A table-only parser would report zero: the real list is a <select>.
    assert all("college" in name.lower() or "institut" in name.lower() for name in colleges)
    assert len(set(colleges)) == len(colleges)


def test_html_listing_yields_the_page_and_its_pdfs(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(RUHS_HTML), base_url=RUHS_URL)
    page = refs[0]
    assert page.url == RUHS_URL
    assert page.metadata["kind"] == "affiliated_college_list"
    assert page.metadata["college_count"] >= 30
    documents = refs[1:]
    assert documents
    assert all(ref.url.lower().endswith(".pdf") for ref in documents)
    assert all(ref.metadata["kind"] == "affiliation_document" for ref in documents)


def test_sitemap_mode_finds_affiliation_pages(adapter) -> None:
    refs = adapter.parse_sitemap(fixture_text(SITEMAP_XML), sitemap_url=SITEMAP_URL)
    urls = [ref.url for ref in refs]
    assert "https://www.tnmgrmu.ac.in/affiliated-colleges-institutions/" in urls
    assert any("seat-matrix" in url for url in urls)
    assert all(ref.metadata["kind"] == "affiliation_page" for ref in refs)
    assert all(ref.source_id == "state_health_universities" for ref in refs)


def test_fetch_uses_the_configured_listing_page(registry) -> None:
    source = registry.sources.by_id("state_health_universities")
    ladder = FakeLadder(
        {
            RUHS_URL: ok_result(RUHS_URL, fixture_text(RUHS_HTML)),
            SITEMAP_URL: ok_result(SITEMAP_URL, fixture_text(SITEMAP_XML)),
        }
    )
    refs = list(HealthUniversityAdapter(source, ladder=ladder).fetch())
    assert refs
    assert RUHS_URL in ladder.calls
    assert SITEMAP_URL in ladder.calls
    assert all(ref.source_id == "state_health_universities" for ref in refs)
    assert all("skipped_sites" in ref.metadata for ref in refs)


def test_unverified_site_is_skipped_with_a_reason(registry) -> None:
    source = registry.sources.by_id("state_health_universities").model_copy(
        update={"seeds": ["https://wbuhs.ac.in"]}
    )
    ladder = FakeLadder(ok_result("https://wbuhs.ac.in/", "<html></html>"))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(HealthUniversityAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "verify_first_unverified"
    assert ladder.calls == []


def test_js_shell_is_detected_not_parsed(adapter) -> None:
    assert adapter.is_js_shell(fixture_text(JS_SHELL)) == "js_shell"
    assert adapter.parse_colleges(fixture_text(JS_SHELL)) == []


def test_js_shell_abstains_in_fetch(registry) -> None:
    source = registry.sources.by_id("state_health_universities").model_copy(
        update={"seeds": [RUHS_URL]}
    )
    ladder = FakeLadder(ok_result(RUHS_URL, fixture_text(JS_SHELL)))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(HealthUniversityAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "js_shell"


def test_robots_disallow_abstains(registry) -> None:
    source = registry.sources.by_id("state_health_universities")
    ladder = FakeLadder(ok_result(RUHS_URL, fixture_text(RUHS_HTML)), allowed=False)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(HealthUniversityAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "robots_disallow"


def test_empty_but_valid_listing_is_not_a_failure(registry) -> None:
    source = registry.sources.by_id("state_health_universities").model_copy(
        update={"seeds": [RUHS_URL]}
    )
    empty = (
        "<html><head><title>Affiliated Colleges</title></head><body>"
        "<h1>Affiliated Colleges</h1><p>No colleges are listed for this cycle.</p>"
        "</body></html>"
    )
    ladder = FakeLadder(ok_result(RUHS_URL, empty))
    refs = list(HealthUniversityAdapter(source, ladder=ladder).fetch())
    # The page exists and is snapshotted; it simply names nothing yet.
    assert [ref.metadata["college_count"] for ref in refs] == [0]


def test_http_error_abstains(registry) -> None:
    source = registry.sources.by_id("state_health_universities").model_copy(
        update={"seeds": [RUHS_URL]}
    )
    ladder = FakeLadder(failed_result(RUHS_URL, status=503, error="http_503"))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(HealthUniversityAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason in {"blocked", "http_503"}
