"""Wave v1-w6: the four family lanes — fee, seat, placement, admission.

The discovery layer for these families is deliberately composed, not
duplicated: ``fee_regulator`` (state fee-commission orders), ``state_counsel
ling`` (seat matrices and admission notifications on counselling portals) and
``institution_site`` (fee/seat/prospectus/placement evidence types with
administrative-fee vetoes) cover the four lanes across 14 registry entries.
What PLAN #6 acceptance demanded and the suite lacked is here:

* every ``DocumentRef`` produced from a fixture listing resolves **verbatim**:
  the anchor text appears byte-for-byte in the listing HTML and the URL is
  absolute — the fetch-layer analogue of claim span resolution;
* the four families are discovered from one institution listing with the
  right priorities (fee/seat above placement, navigation never a document);
* family hints ride on the strong phrases only, and the administrative-fee
  veto keeps "Fee for recognition of Institution as Research Centre" out of
  the fee lane;
* ``state_counselling`` joins the cross-adapter contract with its committed
  fixture (seat matrix + admission notices are one portal listing apart).

No adapter parses a number into a claim (contract test still guards this);
the real crawl over these sources remains LOCAL-QUEUE LQ-2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_adapters_support import FakeLadder, fixture_text, ok_result

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"
EVIDENCE_HTML = "institution_site_evidence_listing_snippet.html"
NOTICES_HTML = "state_counselling_notices_snippet.html"
BASE = "https://demoihs.ac.in/admissions/"
MH_LISTING = "https://counselling.mh.gov.in/notices/"


def _site_adapter(registry, ladder=None):
    from fmd.adapters.institution_site import InstitutionSiteAdapter

    source = registry.sources.by_id("institution_sites")
    return InstitutionSiteAdapter(source, ladder=ladder)


def _counselling_adapter(registry):
    from fmd.adapters.state_counselling import StateCounsellingAdapter

    return StateCounsellingAdapter(registry.sources.by_id("counselling_maharashtra"))


# ---------------------------------------------------------------- span resolution


@pytest.mark.parametrize(
    "source_id,html,base",
    [
        ("institution_sites", EVIDENCE_HTML, BASE),
        ("counselling_maharashtra", NOTICES_HTML, MH_LISTING),
        ("counselling_karnataka", NOTICES_HTML, MH_LISTING),
        (
            "fee_regulators",
            "fee_regulator_order_listing_structural.html",
            "https://fee.example.gov.in/orders/",
        ),
    ],
    ids=["institution_site", "counselling_mh", "counselling_ka", "fee_regulator"],
)
def test_every_ref_resolves_verbatim_in_its_listing(registry, source_id, html, base) -> None:
    """Fetch-layer span resolution: anchor text and URL must come from the page.

    A DocumentRef whose anchor text is not in the listing HTML, or whose URL is
    not absolute, cannot be traced back to the page it was read from — the
    discovery-layer equivalent of a claim whose quote does not resolve.
    """
    text = fixture_text(html)
    # Anchor text is produced from parsed+entity-decoded HTML: the verbatim
    # check must run on the decoded text ("News & Events"), not the raw bytes
    # ("News &amp; Events").
    import html as html_module

    decoded = html_module.unescape(text)
    source = registry.sources.by_id(source_id)
    adapter_name = str(getattr(source, "adapter", "") or "")
    assert adapter_name, f"{source_id} has no adapter"
    from fmd.adapters import is_known_adapter

    assert is_known_adapter(adapter_name)
    adapters = {
        "institution_site": _site_adapter,
        "state_counselling": _counselling_adapter,
    }
    if adapter_name == "fee_regulator":
        from fmd.adapters.fee_regulator import FeeRegulatorAdapter

        adapter = FeeRegulatorAdapter(source)
    else:
        adapter = adapters[adapter_name](registry)
    refs = adapter.parse_listing(text, base_url=base)
    assert refs, f"{source_id}: fixture produced no documents"
    for ref in refs:
        assert ref.url.startswith("http"), ref.url
        anchor = str(ref.metadata.get("anchor_text", "") or "")
        if anchor:
            assert anchor in decoded, f"{source_id}: anchor {anchor!r} not verbatim in listing"


# ------------------------------------------------------------------ four lanes


def test_all_four_family_lanes_fire_from_one_institution_listing(registry) -> None:
    """Fee, seat, placement and admission documents are all discovered."""
    adapter = _site_adapter(registry)
    refs = adapter.parse_listing(fixture_text(EVIDENCE_HTML), base_url=BASE)
    families = {str(ref.metadata.get("family")) for ref in refs if ref.metadata.get("family")}
    assert {"fee_structure", "seat_matrix"} <= families
    urls = {ref.url for ref in refs}
    assert "https://demoihs.ac.in/files/placement/Placement-Report-2024-25.pdf" in urls
    assert "https://demoihs.ac.in/files/admissions/Admission-Brochure-2026.pdf" in urls
    assert "https://demoihs.ac.in/files/naac/AQAR-2024-25.pdf" in urls


def test_fee_and_seat_outrank_placement_and_navigation_is_never_a_document(
    registry,
) -> None:
    """The reserved budget ordering: fee/seat first, placement next, navigation
    pages (About/Gallery/News/Tenders) never appear as document refs."""
    adapter = _site_adapter(registry)
    refs = adapter.parse_listing(fixture_text(EVIDENCE_HTML), base_url=BASE)
    urls = [ref.url for ref in refs]
    fee_i = next(i for i, u in enumerate(urls) if "Fee-Structure" in u)
    seat_i = next(i for i, u in enumerate(urls) if "Seat-Matrix" in u)
    placement_i = next(i for i, u in enumerate(urls) if "Placement-Report" in u)
    assert fee_i < placement_i and seat_i < placement_i
    # Navigation *pages* are never document refs; a PDF that happens to live
    # under files/gallery/ is still a document (ranked last, never dropped).
    for url in urls:
        page_bases = (
            "https://demoihs.ac.in/gallery/",
            "https://demoihs.ac.in/news/",
            "https://demoihs.ac.in/tenders/",
            "https://demoihs.ac.in/about/",
        )
        assert not any(url == base or url.startswith(base) for base in page_bases), url


def test_administrative_fee_order_is_vetoed_from_the_fee_lane(registry) -> None:
    """The known administrative-fee case: carries "fee" vocabulary, is not a
    student fee structure — vetoed at the link level, no family hint."""
    admin_html = (
        "<html><body>"
        '<a href="https://demoihs.ac.in/files/_fee/Guidelines-Fee-for-recognition'
        '-of-Institution-as-Research-Centre.pdf">Fee for recognition of '
        "Institution as Research Centre</a>"
        '<a href="https://demoihs.ac.in/files/fee/Fee-Structure-2025-26.pdf">'
        "Fee Structure 2025-26</a>"
        "</body></html>"
    )
    adapter = _site_adapter(registry)
    refs = adapter.parse_listing(admin_html, base_url=BASE)
    by_url = {ref.url: ref for ref in refs}
    admin = next((ref for url, ref in by_url.items() if "Fee-for-recognition" in url), None)
    good = next((ref for url, ref in by_url.items() if "Fee-Structure-2025-26" in url), None)
    assert good is not None and good.metadata.get("family") == "fee_structure"
    assert admin is not None
    types = list(admin.metadata.get("evidence_types") or [])
    assert "fee_structure" not in types
    assert admin.metadata.get("family") is None


def test_family_hints_only_ride_on_strong_phrases(registry) -> None:
    """A bare "fees" mention must not claim the fee family; only the strong
    hint phrases ("fee structure", "seat matrix") carry a family to the
    planner — the planner still treats it as a reference hint."""
    weak_html = (
        "<html><body>"
        '<a href="https://demoihs.ac.in/files/misc/fees-note.pdf">Revised fees note</a>'
        "</body></html>"
    )
    adapter = _site_adapter(registry)
    refs = adapter.parse_listing(weak_html, base_url=BASE)
    assert refs, "the weak-fee document must still be discovered"
    ref = refs[0]
    assert ref.metadata.get("family") is None


# --------------------------------------------------- counselling joins the contract


def test_counselling_listing_yields_seat_and_admission_documents(registry) -> None:
    """The counselling portal listing is the seat-matrix/admission lane."""
    adapter = _counselling_adapter(registry)
    refs = adapter.parse_listing(fixture_text(NOTICES_HTML), base_url=MH_LISTING)
    assert refs
    assert all(ref.url.lower().endswith(".pdf") for ref in refs)
    # every ref is attributable to its registry source
    assert {ref.source_id for ref in refs} == {"counselling_maharashtra"}


def test_counselling_fetch_walks_every_registry_listing_url(registry) -> None:
    """All registry listing URLs are fetched; robots refusals abstain loudly."""
    adapter = _counselling_adapter(registry)
    ladder = FakeLadder(ok_result(MH_LISTING, fixture_text(NOTICES_HTML)))
    adapter.ladder = ladder
    refs = list(adapter.fetch())
    assert refs
    fetched = set(ladder.calls)  # FakeLadder records fetched URL strings
    # registry-driven: every configured listing URL was actually fetched
    assert set(adapter._listing_urls()) <= fetched
    assert fetched


def test_discovery_layer_stays_fetch_only(registry) -> None:
    """Guard the composition: the four lanes never grow an extraction path."""
    import ast

    for module in ("fee_regulator", "state_counselling", "institution_site"):
        path = Path(__file__).parent.parent / "fmd" / "adapters" / f"{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("fmd.extract", "fmd.db", "fmd.score")), (
                    f"{module} imports {node.module}"
                )
