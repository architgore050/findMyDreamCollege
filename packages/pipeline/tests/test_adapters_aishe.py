from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, fixture_text, ok_result, with_params

from fmd.adapters.aishe import AisheAdapter
from fmd.adapters.base import SourceUnavailable

AISHE_HTML = "aishe_final_report_index_snippet.html"


@pytest.fixture
def adapter(registry):
    return AisheAdapter(registry.sources.by_id("aishe_reports"))


def test_static_xlsx_url_is_built_from_registry_template(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(AISHE_HTML))
    assert [ref.url for ref in refs] == [
        "https://he.nic.in/aishereport/assets/excel/2023-24.xlsx",
        "https://he.nic.in/aishereport/assets/excel/2022-23.xlsx",
    ]
    assert [ref.metadata["report_id"] for ref in refs] == ["2023-24", "2022-23"]


def test_empty_but_valid_index_is_empty_not_a_failure(adapter) -> None:
    assert adapter.parse_listing("<html><body>no reports</body></html>") == []


def test_fetch_abstains_until_verified(adapter) -> None:
    with pytest.raises(SourceUnavailable) as excinfo:
        list(adapter.fetch())
    assert excinfo.value.reason == "verify_first_unverified"


def test_registry_marks_aishe_disabled_and_unverified(registry) -> None:
    source = registry.sources.by_id("aishe_reports")
    assert source.enabled is False
    assert source.verify_first is True
    assert source.params.static_xlsx_url.endswith("/2023-24.xlsx")


def test_explicit_abstain_reason_wins(registry) -> None:
    source = registry.sources.by_id("aishe_reports").model_copy(update={"abstain": True})
    with pytest.raises(SourceUnavailable) as excinfo:
        list(AisheAdapter(source).fetch())
    assert excinfo.value.reason == "abstained"


def test_verified_source_fetches_the_index(registry) -> None:
    source = with_params(registry.sources.by_id("aishe_reports"), verified=True)
    url = "https://aishe.gov.in/aishe-final-report/"
    ladder = FakeLadder(ok_result(url, fixture_text(AISHE_HTML)))
    refs = list(AisheAdapter(source, ladder=ladder).fetch())
    assert len(refs) == 2
