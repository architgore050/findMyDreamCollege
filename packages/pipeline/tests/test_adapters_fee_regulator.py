from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, fixture_text, ok_result, with_params

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.fee_regulator import FeeRegulatorAdapter
from fmd.discover.base import DocumentRef

MH_FORM = "fee_regulator_mhafra_form.html"
ORDERS = "fee_regulator_order_listing_structural.html"
LISTING = "https://fee.example.gov.in/orders/"


@pytest.fixture
def adapter(registry):
    return FeeRegulatorAdapter(registry.sources.by_id("fee_regulators"))


def test_order_pdfs_become_document_refs_only(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(ORDERS), base_url=LISTING, portal_id="example")
    assert len(refs) == 3
    assert all(isinstance(ref, DocumentRef) for ref in refs)
    assert all(ref.url.lower().endswith(".pdf") for ref in refs)
    assert all(ref.metadata["kind"] == "fee_order" for ref in refs)
    assert all(ref.metadata["portal_id"] == "example" for ref in refs)
    # No amount, currency or number is read here; that is WS-D's job.
    assert all("amount" not in ref.metadata for ref in refs)


def test_non_order_links_are_not_selected(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(ORDERS), base_url=LISTING)
    assert not any("Public-Notice" in ref.url for ref in refs)


def test_scanned_flag_is_explicit_metadata(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(ORDERS), base_url=LISTING)
    scanned = {ref.metadata["scanned"] for ref in refs}
    assert scanned == {True}
    assert refs[0].metadata["scanned"] is True


def test_scan_hint_overrides_a_born_digital_claim(registry) -> None:
    source = with_params(registry.sources.by_id("fee_regulators"), assume_scanned=False)
    adapter = FeeRegulatorAdapter(source)
    refs = adapter.parse_listing(fixture_text(ORDERS), base_url=LISTING)
    assert next(ref for ref in refs if "Gazette" in ref.url).metadata["scanned"] is True
    assert next(ref for ref in refs if "Fee-Order" in ref.url).metadata["scanned"] is False


def test_form_portal_abstains_without_a_fetch(registry) -> None:
    source = registry.sources.by_id("fee_regulators")
    ladder = FakeLadder(ok_result(LISTING, fixture_text(MH_FORM)))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(FeeRegulatorAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "form_replay_unverified"
    assert ladder.calls == []


def test_form_page_is_recognised_as_a_shell(adapter) -> None:
    assert adapter.is_js_shell(fixture_text(MH_FORM)) is True


def test_verified_html_portal_extracts_orders(registry) -> None:
    source = with_params(
        registry.sources.by_id("fee_regulators"),
        portals=[{"id": "example", "listing_url": LISTING, "mode": "html", "verified": True}],
    )
    ladder = FakeLadder(ok_result(LISTING, fixture_text(ORDERS)))
    refs = list(FeeRegulatorAdapter(source, ladder=ladder).fetch())
    assert len(refs) == 3
    assert ladder.calls == [LISTING]


def test_empty_but_valid_order_listing_is_not_a_failure(registry) -> None:
    source = with_params(
        registry.sources.by_id("fee_regulators"),
        portals=[{"id": "example", "listing_url": LISTING, "mode": "html", "verified": True}],
    )
    ladder = FakeLadder(ok_result(LISTING, "<html><body>No orders yet</body></html>"))
    assert list(FeeRegulatorAdapter(source, ladder=ladder).fetch()) == []


def test_robots_disallow_abstains(registry) -> None:
    source = with_params(
        registry.sources.by_id("fee_regulators"),
        portals=[{"id": "example", "listing_url": LISTING, "mode": "html", "verified": True}],
    )
    ladder = FakeLadder(ok_result(LISTING, fixture_text(ORDERS)), allowed=False)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(FeeRegulatorAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "robots_disallow"


def test_no_portals_abstains(registry) -> None:
    source = registry.sources.by_id("fee_regulators").model_copy(
        update={"params": None}
    )
    with pytest.raises(SourceUnavailable) as excinfo:
        list(FeeRegulatorAdapter(source).fetch())
    assert excinfo.value.reason == "no_listing_url"
