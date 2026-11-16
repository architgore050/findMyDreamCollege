from __future__ import annotations

import json

import pytest
from test_adapters_support import fixture_text

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.nabh import NabhAdapter

NABH_JSON = "nabh_get_hospitals_delhi_p1.json"


@pytest.fixture
def adapter(registry):
    return NabhAdapter(registry.sources.by_id("nabh_hospitals"))


def test_fixture_has_the_three_documented_keys(adapter) -> None:
    payload = json.loads(fixture_text(NABH_JSON))
    assert {"html", "pagination", "mapData"} <= set(payload)
    assert "HCO Name" in payload["html"]
    assert "Acc. No." in payload["html"]


def test_parse_hospitals_extracts_whitelisted_fields(adapter) -> None:
    rows = adapter.parse_hospitals(fixture_text(NABH_JSON))
    assert len(rows) >= 3
    assert all(set(row) <= {"name", "address", "lat", "lng"} for row in rows)
    assert any("Hospital" in (row.get("name") or "") for row in rows)


def test_parse_listing_emits_bounded_post_page_refs(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NABH_JSON), state="Delhi")
    assert len(refs) == 50  # max_pages from the registry, not 866
    first = refs[0]
    assert first.url == "https://nabh.co/wp-admin/admin-ajax.php"
    assert first.metadata["request_method"] == "POST"
    assert first.metadata["request_body"] == {
        "action": "get_hospitals",
        "selectState": "Delhi",
        "page": 1,
    }
    assert refs[-1].metadata["request_body"]["page"] == 50


def test_empty_but_valid_directory_page(adapter) -> None:
    body = json.dumps({"html": "", "pagination": {"total_results": 0}, "mapData": []})
    assert adapter.parse_listing(body, state="Delhi") == []
    assert adapter.parse_hospitals(body) == []


def test_invalid_json_abstains(adapter) -> None:
    with pytest.raises(SourceUnavailable) as excinfo:
        adapter.parse_listing("<html>Request Rejected</html>")
    assert excinfo.value.reason == "invalid_json"


def test_parse_states_trims_leading_spaces(adapter) -> None:
    states = adapter.parse_states('[" Ahmedabad", "Andhra Pradesh", ""]')
    assert states == ["Ahmedabad", "Andhra Pradesh"]


def test_fetch_without_states_abstains(registry) -> None:
    source = registry.sources.by_id("nabh_hospitals")
    from test_adapters_support import FakeLadder, ok_result

    ladder = FakeLadder(ok_result("https://nabh.co/wp-admin/admin-ajax.php", "[]"))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(NabhAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "no_states"
