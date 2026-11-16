from __future__ import annotations

import json

import pytest
from test_adapters_support import FakeLadder, fixture_text, ok_result

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.nabh import NabhAdapter

#: Trimmed, redacted capture of the live page 1 (2026-11-16): three directory
#: rows, the real pagination totals (866 pages / 21,637 results) and three
#: mapData markers. Website/phone contact details and presigned URLs are removed.
NABH_JSON = "nabh_get_hospitals_page1.json"
AJAX = "https://nabh.co/wp-admin/admin-ajax.php"


@pytest.fixture
def adapter(registry):
    source = registry.sources.by_id("nabh_hospitals")
    ladder = FakeLadder(ok_result(AJAX, "[]"))
    return NabhAdapter(source, ladder=ladder)


def test_fixture_has_the_three_documented_keys(adapter) -> None:
    payload = json.loads(fixture_text(NABH_JSON))
    assert {"html", "pagination", "mapData"} <= set(payload)
    assert "HCO Name" in payload["html"]
    assert "Acc. No." in payload["html"]
    assert payload["pagination"]["total_results"] == 21637
    assert payload["pagination"]["total_pages"] == 866


def test_parse_hospitals_uses_html_rows_and_joins_coordinates(adapter) -> None:
    rows = adapter.parse_hospitals(fixture_text(NABH_JSON))
    assert len(rows) == 3
    allowed = {"name", "address", "acc_no", "certificate_url", "lat", "lng"}
    assert all(set(row) <= allowed for row in rows)
    first = rows[0]
    assert first["name"] == "SGK Hospital & Research Centre"
    assert "Jaipur" in first["address"]
    assert first["acc_no"] == "SHCO-2023-0928"
    assert first["certificate_url"].startswith("https://portal.nabh.co/")
    # mapData names are padded with tabs; the join must still find the marker.
    assert first["lat"] == pytest.approx(26.9488)
    assert first["lng"] == pytest.approx(75.7219)
    # Contact details are never projected onto a record.
    assert not any("phone" in row or "website" in row for row in rows)


def test_parse_hospitals_falls_back_to_markers_without_html(adapter) -> None:
    body = json.dumps(
        {
            "html": "",
            "pagination": {"total_results": 1, "total_pages": 1},
            "mapData": [{"name": "Ek Hospital", "address": "Pune", "lat": 18.5, "lng": 73.8}],
        }
    )
    rows = adapter.parse_hospitals(body)
    assert rows == [{"name": "Ek Hospital", "address": "Pune", "lat": 18.5, "lng": 73.8}]


def test_parse_listing_emits_bounded_post_page_refs(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NABH_JSON))
    assert len(refs) == 50  # max_pages from the registry, not 866
    first = refs[0]
    # The URL repeats the safe parameters so a method-blind GET is not HTTP 400.
    assert first.url.startswith(f"{AJAX}?")
    assert "action=get_hospitals" in first.url
    assert first.metadata["request_method"] == "POST"
    # page is read only from the POST body, never the query string.
    assert first.metadata["request_body"] == {"action": "get_hospitals", "page": 1}
    assert first.metadata["request_headers"]["Referer"].endswith("/find-a-healthcare-organisation/")
    assert refs[-1].metadata["request_body"][adapter.param("page_param")] == 50
    assert refs[-1].metadata["page"] == 50


def test_parse_listing_keeps_registry_state_when_supplied(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NABH_JSON), state="Delhi")
    assert refs[0].metadata["request_body"]["selectState"] == "Delhi"


def test_empty_but_valid_directory_page(adapter) -> None:
    body = json.dumps({"html": "", "pagination": {"total_results": 0}, "mapData": []})
    assert adapter.parse_listing(body) == []
    assert adapter.parse_hospitals(body) == []


def test_invalid_json_abstains_with_a_body_excerpt(adapter) -> None:
    with pytest.raises(SourceUnavailable) as excinfo:
        adapter.parse_listing("<html>Request Rejected</html>")
    assert excinfo.value.reason == "invalid_json"
    assert "Request Rejected" in (excinfo.value.detail or "")


def test_parse_states_trims_leading_spaces(adapter) -> None:
    states = adapter.parse_states('[" Ahmedabad", "Andhra Pradesh", ""]')
    assert states == ["Ahmedabad", "Andhra Pradesh"]


def test_fetch_without_configured_states_sweeps_once(adapter, monkeypatch) -> None:
    """The live endpoint ignores selectState, so no per-state sweep is issued."""
    posts: list[dict] = []

    def fake_post(url: str, body: dict) -> str:
        posts.append(body)
        return fixture_text(NABH_JSON)

    monkeypatch.setattr(adapter, "_post_json", fake_post)
    refs = list(adapter.fetch())
    assert posts == [{"action": "get_hospitals", "page": 1}]
    assert len(refs) == 50


def test_http_400_surfaces_the_body_loudly(adapter, monkeypatch) -> None:
    """A WordPress 400 (`0`) must never become an empty success."""

    class Response:
        status_code = 400
        text = "0"
        content = b"0"

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response())
    with pytest.raises(SourceUnavailable) as excinfo:
        adapter._post_json(AJAX, {"action": "get_hospitals"})
    assert excinfo.value.reason == "http_400"
    assert "0" in (excinfo.value.detail or "")


def test_http_400_detail_is_truncated(adapter, monkeypatch) -> None:
    body = "0" + ("A" * 5000)

    class Response:
        status_code = 400
        text = body
        content = body.encode()

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response())
    with pytest.raises(SourceUnavailable) as excinfo:
        adapter._post_json(AJAX, {"action": "get_hospitals"})
    detail = excinfo.value.detail or ""
    assert len(detail) <= 400
    assert detail.startswith("0")
