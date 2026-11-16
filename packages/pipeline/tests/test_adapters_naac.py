from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, failed_result, fixture_text, ok_result

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.naac import NaacAdapter

NAAC_HTML = "naac_accreditationresults_snippet.html"


@pytest.fixture
def adapter(registry):
    return NaacAdapter(registry.sources.by_id("naac_results"))


def test_listing_resolves_relative_pdf_anchors(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NAAC_HTML))
    assert len(refs) == 8
    assert all(
        ref.url.startswith("https://naac.gov.in/images/docs/AccreditationResults/") for ref in refs
    )
    assert any(ref.url.endswith("SC_258_Health_Sciences.pdf") for ref in refs)
    assert all(ref.source_id == "naac_results" for ref in refs)


def test_teacher_education_rows_are_not_misclassified(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NAAC_HTML))
    assert not any("Teacher_Education" in ref.url for ref in refs)


def test_parse_listing_empty_valid_page(adapter) -> None:
    assert adapter.parse_listing("<html><body><p>No results</p></body></html>") == []


def test_fetch_uses_registry_listing_url(registry) -> None:
    source = registry.sources.by_id("naac_results")
    url = "https://naac.gov.in/index.php/en/19-quick-links/62-accreditationresults"
    ladder = FakeLadder(ok_result(url, fixture_text(NAAC_HTML)))
    refs = list(NaacAdapter(source, ladder=ladder).fetch())
    assert len(refs) == 8
    assert ladder.calls == [url]


def test_http_error_abstains_instead_of_returning_empty(registry) -> None:
    source = registry.sources.by_id("naac_results")
    url = "https://naac.gov.in/index.php/en/19-quick-links/62-accreditationresults"
    ladder = FakeLadder(failed_result(url, status=503, error="http_503"))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(NaacAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "blocked"


def test_robots_disallow_abstains(registry) -> None:
    source = registry.sources.by_id("naac_results")
    url = "https://naac.gov.in/index.php/en/19-quick-links/62-accreditationresults"
    ladder = FakeLadder(ok_result(url, fixture_text(NAAC_HTML)), allowed=False)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(NaacAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "robots_disallow"
