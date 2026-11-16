from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, failed_result, fixture_text, ok_result, with_params

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.nirf import NirfAdapter

MEDICAL_HTML = "nirf_2025_medical_snippet.html"
MEDICAL_URL = "https://www.nirfindia.org/Rankings/2025/MedicalRanking.html"


@pytest.fixture
def adapter(registry):
    return NirfAdapter(registry.sources.by_id("nirf_rankings"))


def test_ranking_page_and_data_sheets_are_refs(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(MEDICAL_HTML), base_url=MEDICAL_URL)
    assert refs[0].url == MEDICAL_URL
    assert refs[0].metadata["kind"] == "ranking_page"
    pdfs = [ref for ref in refs if ref.metadata["kind"] == "institute_data_sheet"]
    assert pdfs
    assert any(ref.metadata["institute_id"] == "IR-D-N-15" for ref in pdfs)
    assert all("nirfpdfcdn/2025/pdf/" in ref.url for ref in pdfs)


def test_every_ref_is_institution_level_only(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(MEDICAL_HTML), base_url=MEDICAL_URL)
    assert refs
    for ref in refs:
        assert ref.metadata["institution_level_only"] is True
        assert ref.metadata["programme_claims_allowed"] is False
        assert ref.metadata["ranking_year"] == "2025"
        assert ref.metadata["ranking_category"].lower() == "medical"


def test_adapter_declares_no_programme_scope(adapter) -> None:
    assert adapter.allowed_claim_scope == "institution"


def test_graph_images_are_not_documents(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(MEDICAL_HTML), base_url=MEDICAL_URL)
    assert not any(ref.url.endswith(".jpg") for ref in refs)


def test_fetch_marks_the_escalated_tier_requirement(registry) -> None:
    source = registry.sources.by_id("nirf_rankings")
    ladder = FakeLadder(ok_result(MEDICAL_URL, fixture_text(MEDICAL_HTML)))
    refs = list(NirfAdapter(source, ladder=ladder).fetch())
    assert refs
    assert all(ref.metadata["fetch_tier_required"] == "escalated" for ref in refs)
    # Every configured ranking page is attempted; unreadable ones do not abort.
    assert MEDICAL_URL in ladder.calls
    assert MEDICAL_URL in {ref.url for ref in refs}


def test_cloudfront_block_abstains_instead_of_empty(registry) -> None:
    source = registry.sources.by_id("nirf_rankings")
    blocked = failed_result(MEDICAL_URL, status=0, error="all_tiers_blocked")
    blocked.quarantined = True
    ladder = FakeLadder(blocked)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(NirfAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "blocked"


def test_robots_disallow_abstains(registry) -> None:
    source = registry.sources.by_id("nirf_rankings")
    ladder = FakeLadder(ok_result(MEDICAL_URL, fixture_text(MEDICAL_HTML)), allowed=False)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(NirfAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "robots_disallow"


def test_empty_but_valid_page_keeps_the_snapshot_ref(registry) -> None:
    source = with_params(
        registry.sources.by_id("nirf_rankings"),
        listing_urls=[MEDICAL_URL],
        pdf_href_pattern=r"(?i)nirfpdfcdn/.*\.pdf$",
    )
    ladder = FakeLadder(ok_result(MEDICAL_URL, "<html><body>Published soon</body></html>"))
    refs = list(NirfAdapter(source, ladder=ladder).fetch())
    assert [ref.url for ref in refs] == [MEDICAL_URL]
    assert refs[0].metadata["institution_level_only"] is True
