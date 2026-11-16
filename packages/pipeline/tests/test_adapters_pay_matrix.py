from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, ok_result, with_params

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.pay_matrix import PayMatrixAdapter
from fmd.discover.base import DocumentRef


@pytest.fixture
def adapter(registry):
    return PayMatrixAdapter(registry.sources.by_id("seventh_cpc_pay_matrix"))


def test_fetch_emits_exactly_one_document_ref(adapter) -> None:
    refs = list(adapter.fetch())
    assert len(refs) == 1
    ref = refs[0]
    assert isinstance(ref, DocumentRef)
    assert ref.url == adapter.param("document_url")


def test_ref_is_marked_level_4_scanned_and_verify_first(adapter) -> None:
    ref = list(adapter.fetch())[0]
    assert ref.metadata["level"] == 4
    assert ref.metadata["scanned"] is True
    assert ref.metadata["verify_first"] is True
    assert ref.metadata["do_not_parse"] is True


def test_adapter_never_fetches_or_parses_the_pdf(adapter) -> None:
    ladder = FakeLadder(ok_result("https://example.invalid/never", ""))
    refs = list(PayMatrixAdapter(adapter.source, ladder=ladder).fetch())
    assert len(refs) == 1
    assert ladder.calls == []


def test_parse_listing_is_the_single_document(adapter) -> None:
    refs = adapter.parse_listing("")
    assert len(refs) == 1
    assert refs[0].metadata["scanned"] is True


def test_unverified_url_abstains(registry) -> None:
    source = with_params(registry.sources.by_id("seventh_cpc_pay_matrix"), verified=False)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(PayMatrixAdapter(source).fetch())
    assert excinfo.value.reason == "verify_first_unverified"


def test_missing_url_abstains(registry) -> None:
    source = with_params(
        registry.sources.by_id("seventh_cpc_pay_matrix"), verified=True, document_url=""
    ).model_copy(update={"base_url": None})
    with pytest.raises(SourceUnavailable) as excinfo:
        list(PayMatrixAdapter(source).fetch())
    assert excinfo.value.reason == "no_document_url"
