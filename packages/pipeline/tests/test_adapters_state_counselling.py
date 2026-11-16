from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, failed_result, fixture_text, ok_result

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.state_counselling import StateCounsellingAdapter

NOTICES_HTML = "state_counselling_notices_snippet.html"
MH_LISTING = "https://cetcell.mahacet.org/notices/"


@pytest.fixture
def adapter(registry):
    return StateCounsellingAdapter(registry.sources.by_id("counselling_maharashtra"))


def test_listing_resolves_notice_pdfs(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NOTICES_HTML), base_url=MH_LISTING)
    urls = [ref.url for ref in refs]
    assert len(refs) == 3
    assert (
        "https://cetcell.mahacet.org/wp-content/uploads/2026/08/Seat-Matrix-Paramedical-2026.pdf"
    ) in urls
    assert all(url.startswith("https://cetcell.mahacet.org/") for url in urls)
    assert all(ref.source_id == "counselling_maharashtra" for ref in refs)


def test_undated_and_non_pdf_links_are_not_documents(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NOTICES_HTML), base_url=MH_LISTING)
    assert not any(ref.url.endswith("/notices/page/2/") for ref in refs)
    assert not any(ref.url.endswith(".html") for ref in refs)


def test_notice_date_comes_from_the_url_when_present(registry) -> None:
    source = registry.sources.by_id("counselling_maharashtra")
    html = '<a href="/wp-content/uploads/2026/08/2026-08-11_Seat_Matrix.pdf">Seat Matrix</a>'
    refs = StateCounsellingAdapter(source).parse_listing(html, base_url=MH_LISTING)
    assert refs[0].metadata["notice_date_token"] == "2026-08-11"


def test_empty_but_valid_listing_is_not_a_failure(adapter) -> None:
    assert adapter.parse_listing("<html><body>No notices</body></html>", base_url=MH_LISTING) == []


def test_fetch_uses_every_registry_listing_url(registry) -> None:
    source = registry.sources.by_id("counselling_maharashtra")
    ladder = FakeLadder(ok_result(MH_LISTING, fixture_text(NOTICES_HTML)))
    refs = list(StateCounsellingAdapter(source, ladder=ladder).fetch())
    assert len(refs) == 3
    assert ladder.calls == [MH_LISTING]


def test_http_error_abstains(registry) -> None:
    source = registry.sources.by_id("counselling_maharashtra")
    ladder = FakeLadder(failed_result(MH_LISTING, status=503, error="http_503"))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(StateCounsellingAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "blocked"


def test_robots_disallow_abstains(registry) -> None:
    source = registry.sources.by_id("counselling_maharashtra")
    ladder = FakeLadder(ok_result(MH_LISTING, fixture_text(NOTICES_HTML)), allowed=False)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(StateCounsellingAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "robots_disallow"


def test_missing_listing_url_abstains(registry) -> None:
    source = registry.sources.by_id("counselling_andhra_pradesh").model_copy(
        update={"verify_first": False}
    )
    with pytest.raises(SourceUnavailable) as excinfo:
        list(StateCounsellingAdapter(source).fetch())
    assert excinfo.value.reason == "no_listing_url"
