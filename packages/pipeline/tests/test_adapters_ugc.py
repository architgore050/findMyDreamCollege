from __future__ import annotations

import json

import pytest
from test_adapters_support import FakeLadder, fixture_text, ok_result, with_params

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.ugc import UgcAdapter, UgcErrorShell

UGC_JSON = "ugc_university_central.json"
UGC_ERROR = "ugc_error_shell.html"


@pytest.fixture
def adapter(registry, ladder=None):
    source = registry.sources.by_id("ugc_university_list")
    return UgcAdapter(source, ladder=ladder)


def test_registry_params_are_the_only_source_of_urls(adapter) -> None:
    assert adapter.param("unitype_param") == "unitypeID"
    assert adapter.param("unitype_ids") == [0, 1, 2, 3, 4]
    assert adapter.pii_fields


def test_normalise_rows_extracts_whitelisted_fields(adapter) -> None:
    rows = adapter.normalise_rows(fixture_text(UGC_JSON))
    assert len(rows) == 57
    amu = next(row for row in rows if row["uni_name"] == "Aligarh Muslim University")
    assert amu["state"] == "Uttar Pradesh"  # trailing space trimmed
    assert amu["uni_type"] == "Central"
    assert amu["status"] == "2(f) & 12B"
    assert amu["AIshe_code"] is None  # unknown, not an invented value


def test_pii_never_leaks_into_a_ref_or_normalised_row(adapter) -> None:
    text = fixture_text(UGC_JSON)
    leaky = {"user_password", "user_username", "Nodal_Mob", "created_Ip", "VCPNO"}
    assert leaky.issubset(set(adapter.pii_fields))
    for row in adapter.normalise_rows(text):
        assert leaky.isdisjoint(row)
    refs = adapter.parse_listing(text)
    assert refs
    for ref in refs:
        assert leaky.isdisjoint(ref.metadata["ugc"])
        assert "user_password" not in json.dumps(ref.metadata)


def test_parse_listing_yields_website_seeds(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(UGC_JSON))
    urls = {ref.url for ref in refs}
    assert "http://www.amu.ac.in" in urls
    assert all(ref.source_id == "ugc_university_list" for ref in refs)


def test_error_shell_is_detected_not_treated_as_a_document(adapter) -> None:
    with pytest.raises(UgcErrorShell) as excinfo:
        adapter.parse_listing(fixture_text(UGC_ERROR))
    assert excinfo.value.reason == "ugc_error_shell"
    assert isinstance(excinfo.value, SourceUnavailable)


def test_fetch_aborts_on_the_http_200_error_shell(registry) -> None:
    source = with_params(registry.sources.by_id("ugc_university_list"), unitype_ids=[1])
    url = "https://www.ugc.gov.in/universitydetails/Getuniversity_details?unitypeID=1"
    ladder = FakeLadder(ok_result(url, fixture_text(UGC_ERROR)))
    with pytest.raises(UgcErrorShell):
        UgcAdapter(source, ladder=ladder).fetch()


def test_fetch_valid_json_yields_endpoint_refs(registry) -> None:
    source = with_params(registry.sources.by_id("ugc_university_list"), unitype_ids=[1])
    url = "https://www.ugc.gov.in/universitydetails/Getuniversity_details?unitypeID=1"
    ladder = FakeLadder(ok_result(url, fixture_text(UGC_JSON)))
    refs = list(UgcAdapter(source, ladder=ladder).fetch())
    assert [ref.url for ref in refs] == [url]
    assert refs[0].metadata["unitype_id"] == 1


def test_empty_but_valid_listing_is_not_a_failure(registry) -> None:
    source = with_params(registry.sources.by_id("ugc_university_list"), unitype_ids=[1])
    url = "https://www.ugc.gov.in/universitydetails/Getuniversity_details?unitypeID=1"
    body = json.dumps({"m": "Data filled.", "c": "success", "List": []})
    ladder = FakeLadder(ok_result(url, body, content_type="application/json"))
    assert list(UgcAdapter(source, ladder=ladder).fetch()) == []
    assert UgcAdapter(source).parse_listing(body) == []


def test_robots_disallow_abstains_loudly(registry) -> None:
    source = with_params(registry.sources.by_id("ugc_university_list"), unitype_ids=[1])
    url = "https://www.ugc.gov.in/universitydetails/Getuniversity_details?unitypeID=1"
    ladder = FakeLadder(
        ok_result(url, fixture_text(UGC_JSON)), allowed=False, reason="disallowed_by_robots"
    )
    with pytest.raises(SourceUnavailable) as excinfo:
        list(UgcAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "robots_disallow"
