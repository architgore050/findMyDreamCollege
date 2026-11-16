from __future__ import annotations

import pytest
from test_adapters_support import FakeLadder, fixture_text, ok_result, with_params

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.state_council import StateCouncilAdapter

LISTING = "state_council_ncahp_statecouncil_snippet.html"
LISTING_URL = "https://ncahp.abdm.gov.in/StateCouncil"
DENYLISTED = ("statealliedandhealthcarecouncil.com", "sahstc.com")


@pytest.fixture
def adapter(registry):
    return StateCouncilAdapter(
        registry.sources.by_id("state_paramedical_councils"),
        denylisted_domains=registry.sources.denied_domains(),
    )


def test_notification_pdfs_are_extracted_with_state_and_date(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(LISTING), base_url=LISTING_URL)
    assert len(refs) >= 60
    assert all(ref.url.startswith("https://ncahp.abdm.gov.in/statecouncil/") for ref in refs)
    tokens = {ref.metadata.get("state_token") for ref in refs}
    assert {"kerala", "maharastra"} <= tokens
    dated = [ref for ref in refs if ref.metadata.get("notification_date")]
    assert dated
    assert all(len(ref.metadata["notification_date"]) == 10 for ref in dated)
    assert all(ref.metadata["authority"] == "ncahp_notification" for ref in refs)
    assert all(ref.metadata["untrusted_origin"] is False for ref in refs)


def test_rules_pdfs_are_classified_separately(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(LISTING), base_url=LISTING_URL)
    rules = [ref for ref in refs if ref.metadata["kind"] == "council_rules"]
    assert rules
    assert all("SAHC_Rules" in ref.url for ref in rules)


def test_a_denylisted_source_url_is_refused_before_any_fetch(registry) -> None:
    source = registry.sources.by_id("state_paramedical_councils").model_copy(
        update={"base_url": "https://www.sahstc.com/"}
    )
    adapter = StateCouncilAdapter(
        source, ladder=FakeLadder(ok_result(LISTING_URL, fixture_text(LISTING))),
        denylisted_domains=registry.sources.denied_domains(),
    )
    with pytest.raises(SourceUnavailable) as excinfo:
        list(adapter.fetch())
    assert excinfo.value.reason == "denylisted_domain"
    assert adapter.ladder.calls == []


def test_a_denylisted_listing_url_is_refused_before_any_fetch(registry) -> None:
    source = with_params(
        registry.sources.by_id("state_paramedical_councils"),
        listing_url="https://statealliedandhealthcarecouncil.com/notifications",
    )
    ladder = FakeLadder(ok_result(LISTING_URL, fixture_text(LISTING)))
    adapter = StateCouncilAdapter(
        source, ladder=ladder, denylisted_domains=registry.sources.denied_domains()
    )
    with pytest.raises(SourceUnavailable) as excinfo:
        list(adapter.fetch())
    assert excinfo.value.reason == "denylisted_domain"
    assert ladder.calls == []


def test_a_denylisted_href_is_never_yielded(adapter) -> None:
    html = (
        f'<a href="{LISTING_URL}">x</a>'
        '<a href="https://www.sahstc.com/council/Kerala.pdf">Kerala</a>'
        '<a href="/statecouncil/Kerala notification.pdf">28.09.2022</a>'
    )
    refs = adapter.parse_listing(html, base_url=LISTING_URL)
    assert [ref.url for ref in refs] == [
        "https://ncahp.abdm.gov.in/statecouncil/Kerala notification.pdf"
    ]


def test_non_ncahp_origin_is_marked_untrusted(registry) -> None:
    source = with_params(
        registry.sources.by_id("state_paramedical_councils"), require_ncahp_host=False
    )
    adapter = StateCouncilAdapter(
        source, denylisted_domains=registry.sources.denied_domains()
    )
    html = '<a href="https://example.com/statecouncil/Kerala.pdf">28.09.2022</a>'
    refs = adapter.parse_listing(html, base_url=LISTING_URL)
    assert refs[0].metadata["authority"] == "untrusted_origin"
    assert refs[0].metadata["untrusted_origin"] is True
    assert adapter.origin("https://kesahc.kerala.gov.in/") == "statutory_host"


def test_require_ncahp_host_drops_foreign_links(adapter) -> None:
    html = (
        '<a href="https://kesahc.kerala.gov.in/statecouncil/register.pdf">Register</a>'
        '<a href="/statecouncil/Kerala notification.pdf">28.09.2022</a>'
    )
    refs = adapter.parse_listing(html, base_url=LISTING_URL)
    assert all("ncahp.abdm.gov.in" in ref.url for ref in refs)


def test_council_site_is_only_admitted_with_an_ncahp_notification(registry) -> None:
    base = registry.sources.by_id("state_paramedical_councils")
    admitted = with_params(
        base,
        council_sites=[
            {
                "url": "https://kesahc.kerala.gov.in/",
                "state_token": "kerala",
                "source_url": "https://ncahp.abdm.gov.in/statecouncil/Kerala notification.pdf",
            }
        ],
    )
    ladder = FakeLadder(ok_result(LISTING_URL, fixture_text(LISTING)))
    adapter = StateCouncilAdapter(
        admitted, ladder=ladder, denylisted_domains=registry.sources.denied_domains()
    )
    refs = {ref.url: ref for ref in adapter.fetch()}
    assert refs["https://kesahc.kerala.gov.in/"].metadata["authority"] == "ncahp_admitted"
    assert refs["https://kesahc.kerala.gov.in/"].metadata["untrusted_origin"] is False

    unadmitted = with_params(
        base, council_sites=[{"url": "https://kesahc.kerala.gov.in/", "state_token": "kerala"}]
    )
    adapter = StateCouncilAdapter(
        unadmitted, ladder=ladder, denylisted_domains=registry.sources.denied_domains()
    )
    refs = {ref.url: ref for ref in adapter.fetch()}
    assert refs["https://kesahc.kerala.gov.in/"].metadata["authority"] == "statutory_host"
    assert refs["https://kesahc.kerala.gov.in/"].metadata["untrusted_origin"] is False


def test_robots_disallow_abstains(registry) -> None:
    adapter = StateCouncilAdapter(
        registry.sources.by_id("state_paramedical_councils"),
        ladder=FakeLadder(ok_result(LISTING_URL, fixture_text(LISTING)), allowed=False),
        denylisted_domains=registry.sources.denied_domains(),
    )
    with pytest.raises(SourceUnavailable) as excinfo:
        list(adapter.fetch())
    assert excinfo.value.reason == "robots_disallow"


def test_empty_but_valid_listing_is_not_a_failure(registry) -> None:
    adapter = StateCouncilAdapter(
        registry.sources.by_id("state_paramedical_councils"),
        ladder=FakeLadder(ok_result(LISTING_URL, "<html><body>No notifications</body></html>")),
        denylisted_domains=registry.sources.denied_domains(),
    )
    assert list(adapter.fetch()) == []
