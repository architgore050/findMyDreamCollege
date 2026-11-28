from __future__ import annotations

import logging

import pytest
from test_adapters_support import (
    FakeLadder,
    failed_result,
    fixture_text,
    ok_result,
    with_params,
)

from fmd.adapters.base import SourceUnavailable
from fmd.adapters.institution_site import InstitutionSiteAdapter
from fmd.discover.base import DocumentRef

NIRF_LISTING = "institution_site_nirf_listing_snippet.html"
EVIDENCE_LISTING = "institution_site_evidence_listing_snippet.html"
SEED = "https://mtct.ac.in/nirf/"
DEMO_SEED = "https://demoihs.ac.in/downloads"
#: The real observed negative case (RUHS, 2026-11-17): an administrative order
#: whose URL contains "Fee" in an unrelated sense.
ADMIN_FEE_URL = (
    "https://demoihs.ac.in/files/admin/Guidelines-Fee-for-recognition-of-"
    "Institution-as-Research-Centre-for-Ph.D-Course.pdf"
)


@pytest.fixture
def adapter(registry):
    return InstitutionSiteAdapter(registry.sources.by_id("institution_sites"))


def test_document_refs_are_scored_and_pdf_only(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(NIRF_LISTING), base_url=SEED)
    urls = [ref.url for ref in refs]
    assert urls
    assert all(url.lower().endswith(".pdf") for url in urls)
    assert not any(url.endswith((".docx", ".xlsx")) for url in urls)
    assert not any("wp-login" in url for url in urls)
    assert any("NIRF-2026-College-Category-Data.pdf" in url for url in urls)
    # Highest-scoring link first, deterministically.
    scores = [ref.metadata["link_score"] for ref in refs]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(ref, DocumentRef) for ref in refs)
    assert all(ref.source_id == "institution_sites" for ref in refs)


def test_login_javascript_and_mailto_are_hard_skips(adapter) -> None:
    html = (
        '<a href="javascript:void(0)">Placement report</a>'
        '<a href="mailto:principal@example.ac.in">Placement</a>'
        '<a href="#placement">Placement</a>'
        '<a href="/wp-login.php?redirect=placement.pdf">Login</a>'
        '<a href="/files/placement-2025.pdf">Placement report</a>'
    )
    refs = adapter.parse_listing(html, base_url="https://example.ac.in/")
    assert [ref.url for ref in refs] == ["https://example.ac.in/files/placement-2025.pdf"]


def test_navigation_stays_on_the_registrable_domain(adapter) -> None:
    nav = adapter.parse_navigation(fixture_text(NIRF_LISTING), base_url=SEED)
    assert nav
    assert all(url.startswith("https://") for url in nav)
    assert not any("wp-login" in url for url in nav)
    assert not any(url.startswith("javascript:") for url in nav)
    # The fixture links an off-site social profile in spirit only; nothing here
    # may leave mtct.ac.in.
    assert all("mtct.ac.in" in url for url in nav)


@pytest.mark.parametrize(
    "malformed_href",
    [
        "http://[::1/files/malformed-placement.pdf",
        "https://example.ac.in:invalid/files/malformed-placement.pdf",
        "https:///files/malformed-placement.pdf",
    ],
    ids=["invalid-ipv6", "invalid-port", "missing-host"],
)
def test_malformed_urls_do_not_abort_listing(
    adapter, malformed_href: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="fmd.adapters.institution_site")
    base_url = "https://example.ac.in/downloads"
    html = (
        '<a href="/files/placement-report-2025.pdf">Placement Report</a>'
        f'<a href="{malformed_href}">Malformed placement report</a>'
        '<a href="/naac/ssr">NAAC SSR</a>'
    )

    refs = adapter.parse_listing(html, base_url=base_url)
    nav = adapter.parse_navigation(html, base_url=base_url)

    assert [ref.url for ref in refs] == ["https://example.ac.in/files/placement-report-2025.pdf"]
    assert nav == ["https://example.ac.in/naac/ssr"]
    assert malformed_href in caplog.text


def test_bounded_crawl_yields_documents_and_survives_404_pages(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder({SEED: ok_result(SEED, fixture_text(NIRF_LISTING))})
    refs = list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[SEED]))
    assert refs
    assert all(ref.metadata["found_on"] == SEED for ref in refs)
    # Depth is bounded: navigation links are attempted but never unbounded.
    assert len(ladder.calls) <= 1 + int(source.params.max_pages_per_domain)


def test_robots_disallow_abstains(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder(ok_result(SEED, fixture_text(NIRF_LISTING)), allowed=False)
    with pytest.raises(SourceUnavailable) as excinfo:
        list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[SEED]))
    assert excinfo.value.reason == "robots_disallow"


def test_empty_but_valid_page_is_not_a_failure(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder(ok_result(SEED, "<html><body><p>Nothing published yet</p></body></html>"))
    assert list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[SEED])) == []


def test_unreachable_domain_abstains(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder(failed_result(SEED, status=0, error="timeout"))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[SEED]))
    assert excinfo.value.reason == "unreachable"


def test_no_domain_abstains_without_fetching(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    ladder = FakeLadder(ok_result(SEED, fixture_text(NIRF_LISTING)))
    with pytest.raises(SourceUnavailable) as excinfo:
        list(InstitutionSiteAdapter(source, ladder=ladder).fetch())
    assert excinfo.value.reason == "no_domain"
    assert ladder.calls == []


def test_yield_stop_rule_bounds_a_dry_crawl(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    page = ok_result(SEED, fixture_text(NIRF_LISTING))
    ladder = FakeLadder({SEED: page})
    adapter = InstitutionSiteAdapter(source, ladder=ladder)
    refs = list(adapter.fetch(domains=[SEED]))
    assert refs
    # With max_depth=2 the crawl may visit nav pages, but the page budget is hard.
    assert len(ladder.calls) <= 1 + int(source.params.max_pages_per_domain)


def test_link_scoring_prefers_placement_and_aqar_over_news(adapter) -> None:
    html = (
        '<a href="/news/annual-report-2025.pdf">Annual report</a>'
        '<a href="/files/aqar-2024.pdf">AQAR 2024</a>'
        '<a href="/files/placement-report-2025.pdf">Placement Report</a>'
    )
    refs = adapter.parse_listing(html, base_url="https://example.ac.in/")

    urls = [ref.url for ref in refs]
    assert urls[-1].endswith("annual-report-2025.pdf")
    assert {url.rsplit("/", 1)[-1] for url in urls[:2]} == {
        "aqar-2024.pdf",
        "placement-report-2025.pdf",
    }
    assert refs[0].metadata["link_score"] >= refs[-1].metadata["link_score"]
    # The weights are registry data, not constants in this module.
    assert "link_weights" in adapter.source.params.model_dump(exclude_none=True)


def test_navigation_skips_news_events_and_gallery_pages(adapter) -> None:
    html = (
        '<a href="/about/gallery">Photo Gallery</a>'
        '<a href="/about/news">News</a>'
        '<a href="/about/events">Events</a>'
        '<a href="/naac/ssr">NAAC SSR</a>'
        '<a href="/downloads">Downloads</a>'
    )
    nav = adapter.parse_navigation(html, base_url="https://example.ac.in/")

    assert not any(token in url for url in nav for token in ("gallery", "news", "events"))
    assert nav[0].endswith("/naac/ssr")  # evidence-adjacent sections come first


# ------------------------------------------------------------------- W1: evidence
# targeting (2026-11-18). The measured 50-domain run produced fee_structure 0 /
# seat_matrix 1: the per-domain document cap was spent on SSR/AQAR/NIRF mirrors.
# These tests pin the evidence-type boost table, the administrative-fee negative
# case, the PDF-first budget reserve and the family hints — all offline.


def _refs_by_filename(refs: list[DocumentRef]) -> dict[str, DocumentRef]:
    return {ref.url.rsplit("/", 1)[-1]: ref for ref in refs}


def test_evidence_type_scoring_ranks_seat_and_fee_above_news_and_galleries(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(EVIDENCE_LISTING), base_url=DEMO_SEED)

    assert refs, "the fixture must produce document refs"
    scores = [ref.metadata["link_score"] for ref in refs]
    assert scores == sorted(scores, reverse=True)
    names = _refs_by_filename(refs)
    # Seat and fee structure PDFs outrank everything else on the page.
    top = {ref.url.rsplit("/", 1)[-1] for ref in refs[:2]}
    assert top == {"Fee-Structure-2025-26.pdf", "Seat-Matrix-2025-26.pdf"}
    # News, gallery, tender and staff-circular documents rank last.
    bottom = [ref.url.rsplit("/", 1)[-1] for ref in refs[-4:]]
    assert set(bottom) == {
        "newsletter-january.pdf",
        "annual-day-photos.pdf",
        "tender-notice-laboratory.pdf",
        "staff-circular-duty-roster.pdf",
    }
    for name in (
        "newsletter-january.pdf",
        "annual-day-photos.pdf",
        "tender-notice-laboratory.pdf",
        "staff-circular-duty-roster.pdf",
    ):
        assert (
            names[name].metadata["link_score"] < names["NIRF-2026-Data.pdf"].metadata["link_score"]
        )


def test_admin_fee_order_is_not_a_fee_structure_candidate(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(EVIDENCE_LISTING), base_url=DEMO_SEED)
    admin = _refs_by_filename(refs)[
        "Guidelines-Fee-for-recognition-of-Institution-as-Research-Centre-for-Ph.D-Course.pdf"
    ]

    # Not classified: no family hint, no evidence-type candidacy.
    assert "family" not in admin.metadata
    assert not admin.metadata.get("evidence_types")
    # Not boosted: an untyped document, never ahead of a real fee structure.
    generic = _refs_by_filename(refs)["generic-application-form.pdf"]
    assert admin.metadata["link_score"] == generic.metadata["link_score"]
    assert admin.metadata["link_score"] < 40


def test_seat_and_fee_family_hints_are_discovered_from_anchor_and_filename(adapter) -> None:
    refs = adapter.parse_listing(fixture_text(EVIDENCE_LISTING), base_url=DEMO_SEED)
    names = _refs_by_filename(refs)

    assert names["Fee-Structure-2025-26.pdf"].metadata["family"] == "fee_structure"
    assert names["Seat-Matrix-2025-26.pdf"].metadata["family"] == "seat_matrix"
    # Types without a declared family boost the queue but never classify.
    assert "family" not in names["Admission-Brochure-2026.pdf"].metadata
    assert "family" not in names["NIRF-2026-Data.pdf"].metadata
    assert "family" not in names["Placement-Report-2024-25.pdf"].metadata


def test_registry_declared_family_suppresses_discovery_hints(registry) -> None:
    source = with_params(
        registry.sources.by_id("institution_sites"), extraction_family="placement_report"
    )
    refs = InstitutionSiteAdapter(source).parse_listing(
        fixture_text(EVIDENCE_LISTING), base_url=DEMO_SEED
    )

    assert refs
    assert all("family" not in ref.metadata for ref in refs)


def test_evidence_scoring_table_comes_from_config(registry) -> None:
    source = registry.sources.by_id("institution_sites")
    baseline = InstitutionSiteAdapter(source).parse_listing(
        fixture_text(EVIDENCE_LISTING), base_url=DEMO_SEED
    )
    fee_score = _refs_by_filename(baseline)["Fee-Structure-2025-26.pdf"].metadata["link_score"]
    assert fee_score >= 40  # the configured fee_structure boost is applied

    # Zeroing the fee weight in config drops the fee PDF back to the untyped
    # pack: the table is registry data, not a constant in the adapter.
    demoted = InstitutionSiteAdapter(
        with_params(source, evidence_type_weights={"fee_structure": 0})
    ).parse_listing(fixture_text(EVIDENCE_LISTING), base_url=DEMO_SEED)
    demoted_score = _refs_by_filename(demoted)["Fee-Structure-2025-26.pdf"].metadata["link_score"]
    assert demoted_score < fee_score

    # An added type is data too: a custom hostel fee vocabulary boosts its links.
    custom = InstitutionSiteAdapter(
        with_params(
            source,
            evidence_types={
                "hostel": {
                    "filename": ["(?i)hostel[-_ ]?tariff"],
                }
            },
            evidence_type_weights={"hostel": 60},
        )
    ).parse_listing(fixture_text(EVIDENCE_LISTING), base_url=DEMO_SEED)
    assert custom, "custom evidence types must not break parsing"


def test_default_evidence_table_applies_without_registry_params(registry) -> None:
    source = registry.sources.by_id("institution_sites").model_copy(
        update={"params": type(registry.sources.by_id("institution_sites").params)()}
    )
    html = '<a href="/files/fee/Fee-Structure-2026.pdf">Fee Structure</a>'
    refs = InstitutionSiteAdapter(source).parse_listing(html, base_url=DEMO_SEED)

    assert refs
    assert refs[0].metadata["link_score"] >= 40
    assert refs[0].metadata["family"] == "fee_structure"


def test_pdf_first_reserve_keeps_evidence_documents_within_the_cap(registry) -> None:
    source = with_params(
        registry.sources.by_id("institution_sites"),
        max_documents_per_domain=3,
        pdf_first_reserve_percent=50,
    )
    ladder = FakeLadder({DEMO_SEED: ok_result(DEMO_SEED, fixture_text(EVIDENCE_LISTING))})
    refs = list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[DEMO_SEED]))

    assert len(refs) == 3  # the document cap is respected
    # The reserved share (ceil(3 * 50%) = 2 slots) held only evidence-typed
    # documents, and the best evidence filled the whole cap here.
    assert all(ref.metadata.get("evidence_types") for ref in refs)
    assert {ref.url.rsplit("/", 1)[-1] for ref in refs} == {
        "Fee-Structure-2025-26.pdf",
        "Seat-Matrix-2025-26.pdf",
        "Admission-Brochure-2026.pdf",
    }


def test_pdf_first_reserve_caps_non_evidence_documents(registry) -> None:
    source = with_params(
        registry.sources.by_id("institution_sites"),
        max_documents_per_domain=3,
        pdf_first_reserve_percent=50,
    )
    html = "".join(f'<a href="/files/news/doc-{i}.pdf">News update {i}</a>' for i in range(5))
    ladder = FakeLadder({DEMO_SEED: ok_result(DEMO_SEED, html)})
    refs = list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[DEMO_SEED]))

    # ceil(3 * 50%) = 2 slots are reserved for evidence; news documents may only
    # fill the remaining one.
    assert len(refs) == 1
    assert all(not ref.metadata.get("evidence_types") for ref in refs)


def test_pdf_first_reserve_zero_disables_the_reserve(registry) -> None:
    source = with_params(
        registry.sources.by_id("institution_sites"),
        max_documents_per_domain=3,
        pdf_first_reserve_percent=0,
    )
    html = "".join(f'<a href="/files/news/doc-{i}.pdf">News update {i}</a>' for i in range(5))
    ladder = FakeLadder({DEMO_SEED: ok_result(DEMO_SEED, html)})
    refs = list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[DEMO_SEED]))

    assert len(refs) == 3  # plain top-N behaviour, no reserve


def test_pdf_first_reserve_never_exceeds_the_document_cap(registry) -> None:
    source = with_params(
        registry.sources.by_id("institution_sites"),
        max_documents_per_domain=2,
        pdf_first_reserve_percent=100,
    )
    ladder = FakeLadder({DEMO_SEED: ok_result(DEMO_SEED, fixture_text(EVIDENCE_LISTING))})
    refs = list(InstitutionSiteAdapter(source, ladder=ladder).fetch(domains=[DEMO_SEED]))

    assert len(refs) == 2
    assert all(ref.metadata.get("evidence_types") for ref in refs)


def test_navigation_prefers_evidence_sections(adapter) -> None:
    html = (
        '<a href="/about/">About Us</a>'
        '<a href="/admissions/fee-structure">Fee Structure</a>'
        '<a href="/admissions/seat-matrix">Seat Matrix</a>'
        '<a href="/gallery/">Photo Gallery</a>'
    )
    nav = adapter.parse_navigation(html, base_url=DEMO_SEED)

    assert not any("gallery" in url for url in nav)
    assert nav[0].endswith("/admissions/fee-structure")
    assert any(url.endswith("/admissions/seat-matrix") for url in nav[:2])
