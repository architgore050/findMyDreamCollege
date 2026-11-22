from __future__ import annotations

from urllib.parse import urlparse

import pytest
from test_adapters_support import FakeLadder, failed_result, fixture_text, ok_result

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.ncahp import NcahpAdapter
from fmd.discover.base import DocumentRef

LISTING = "https://ncahp.abdm.gov.in/"
FIXTURE = "ncahp_regulator_listing.html"
EXPECTED_PDFS = [
    "https://ncahp.abdm.gov.in/ahp/Recognition/Rural-Allied-Healthcare-Institute.pdf",
    "https://ncahp.abdm.gov.in/notifications/Regulations/AHP-Registration-Regulations-2026.pdf",
]


@pytest.fixture
def adapter(registry):
    return NcahpAdapter(registry.sources.by_id("ncahp"))


def test_fetch_yields_pdf_refs_with_metadata(registry) -> None:
    source = registry.sources.by_id("ncahp")
    ladder = FakeLadder(ok_result(LISTING, fixture_text(FIXTURE)))
    refs = list(NcahpAdapter(source, ladder=ladder).fetch())
    assert refs
    assert ladder.calls == [LISTING]
    for ref in refs:
        assert isinstance(ref, DocumentRef)
        assert ref.url.lower().endswith(".pdf")
        assert urlparse(ref.url).hostname == "ncahp.abdm.gov.in"
        assert ref.source_id == "ncahp"
        assert ref.discovery_method == "ncahp"
        assert ref.metadata["kind"] == "recognition_document"
        assert ref.metadata["anchor_text"]
        assert ref.metadata["source_id"] == "ncahp"
        assert ref.metadata["source_notes"] == source.notes


def test_parse_listing_yields_pdf_refs(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(FIXTURE), base_url=LISTING)
    assert [ref.url for ref in refs] == EXPECTED_PDFS
    for ref in refs:
        assert ref.metadata["kind"] == "recognition_document"


def test_non_pdf_and_js_links_are_not_selected(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(FIXTURE), base_url=LISTING)
    urls = [ref.url for ref in refs]
    assert LISTING not in urls
    assert not any("guidelines" in url for url in urls)
    assert not any(url.startswith("javascript:") for url in urls)


def test_duplicate_links_dedupe_to_one_ref(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(FIXTURE), base_url=LISTING)
    urls = [ref.url for ref in refs]
    assert len(refs) == 2
    assert len(urls) == len(set(urls))


@pytest.mark.parametrize(
    ("result", "token"),
    [
        (failed_result(LISTING, status=0, error="unreachable"), "unreachable"),
        (failed_result(LISTING, status=404, error="http_404"), "http_404"),
    ],
)
def test_ladder_failure_abstains_with_stable_token(registry, result, token) -> None:
    source = registry.sources.by_id("ncahp")
    ladder = FakeLadder(result)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(NcahpAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == token


def test_abstain_source_abstains(registry) -> None:
    source = registry.sources.by_id("ncahp").model_copy(update={"abstain": True})
    ladder = FakeLadder(ok_result(LISTING, fixture_text(FIXTURE)))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(NcahpAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "abstained"
    assert ladder.calls == []


def test_source_plans_for_the_listing_url(adapter) -> None:
    ref = adapter.ref(LISTING)
    assert ref.source_id == "ncahp"
    assert ref.discovery_method == "ncahp"


def test_empty_listing_is_not_a_failure(registry) -> None:
    source = registry.sources.by_id("ncahp")
    ladder = FakeLadder(ok_result(LISTING, "<html><body>no notices published</body></html>"))
    assert list(NcahpAdapter(source, ladder=ladder).fetch()) == []
