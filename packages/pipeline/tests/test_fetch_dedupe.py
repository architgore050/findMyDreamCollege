"""Dedupe policy tests (brief §7): structured pages, identities and spans.

The real defect these tests lock down: a paginated JSON listing was treated as
boilerplate by text simhash, so distinct pages were dropped as
``near_duplicate_simhash`` and the run looked merely "sparse".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_orchestrate import Harness
from test_orchestrate_support import FakeAdapter, FakeFetcher, FakeSource, response

from fmd.discover.base import DocumentRef
from fmd.fetch.dedupe import (
    DedupeSettings,
    dedupe_settings,
    effective_mode,
    hamming,
    is_pdf_payload,
    pdf_text_sample,
    row_identities,
    simhash64,
    text_sha256,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dedupe"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _ref(url: str, *, source_id: str = "fixture") -> DocumentRef:
    return DocumentRef(url=url, source_id=source_id)


def _run(
    tmp_path: Path,
    registry: Any,
    bodies: dict[str, str | bytes],
    *,
    content_type: str,
    dedupe: dict[str, Any] | None = None,
    source_id: str = "fixture",
    frontend_limit: int = 100,
) -> Harness:
    source = FakeSource(id=source_id, params={"dedupe": dedupe} if dedupe else None)
    refs = [_ref(url, source_id=source_id) for url in bodies]
    fetcher = FakeFetcher(
        {
            url: response(url, body=body, headers={"content-type": content_type})
            for url, body in bodies.items()
        }
    )
    harness = Harness(
        tmp_path,
        registry,
        sources=[source],
        adapters={source_id: FakeAdapter(refs)},
        fetcher=fetcher,
        frontend_limit=frontend_limit,
    )
    harness.summary = harness.orchestrator.crawl(limit=10)
    return harness


def test_structured_pagination_with_disjoint_rows_stores_every_page(tmp_path, registry) -> None:
    first = "https://nabh.co/wp-admin/admin-ajax.php?action=get_hospitals&page=1"
    second = "https://nabh.co/wp-admin/admin-ajax.php?action=get_hospitals&page=2"
    bodies = {
        first: fixture_text("paginated_page_1.json"),
        second: fixture_text("paginated_page_2.json"),
    }
    # The old failure: these two pages are structurally near-identical.
    assert hamming(simhash64(bodies[first]), simhash64(bodies[second])) <= 3

    harness = _run(
        tmp_path,
        registry,
        bodies,
        content_type="application/json",
        dedupe={"mode": "identity", "identity_fields": ["acc_no", "name"]},
    )

    assert harness.summary.counts["fetched"] == 2
    assert harness.summary.counts["stored"] == 2
    assert harness.summary.counts.get("deduped", 0) == 0
    assert len(harness.snapshots.rows) == 2


def test_structured_payload_defaults_to_exact_never_simhash(tmp_path, registry) -> None:
    first = "https://example.test/page-1.json"
    second = "https://example.test/page-2.json"
    bodies = {
        first: fixture_text("paginated_page_1.json"),
        second: fixture_text("paginated_page_2.json"),
    }
    assert hamming(simhash64(bodies[first]), simhash64(bodies[second])) <= 3

    # No registry block: a JSON listing must not fall into text-simhash.
    harness = _run(tmp_path, registry, bodies, content_type="application/json")

    assert harness.summary.counts["stored"] == 2
    assert harness.summary.counts.get("deduped", 0) == 0
    assert {row.dedupe_basis for row in harness.snapshots.rows} == {"raw_bytes"}


def test_identical_refetch_is_deduped_and_recorded(tmp_path, registry) -> None:
    url = "https://example.test/page-1.json"
    body = fixture_text("paginated_page_1.json")
    harness = _run(
        tmp_path,
        registry,
        {url: body},
        content_type="application/json",
        dedupe={"mode": "identity", "identity_fields": ["acc_no"]},
    )
    assert harness.summary.counts["stored"] == 1

    # Same URL and bytes again with the row re-queued: exact matching answers first.
    harness.frontier.repo.set_state(url, "queued")
    again_summary = harness.orchestrator.crawl(limit=10)
    assert again_summary.counts.get("stored", 0) == 0
    assert again_summary.counts["deduped"] == 1
    entry = again_summary.metadata["dedupe"][0]
    assert entry["mode"] == "identity"
    # The same URL served the same bytes: that check is exact, whatever the mode.
    assert entry["basis"] == "raw_bytes"
    assert entry["reason"] == "unchanged_content"
    assert entry["similarity"] == 1.0
    assert isinstance(entry["compared_document_id"], int)


def test_same_identities_with_changed_wrapper_are_deduped(tmp_path, registry) -> None:
    original = "https://example.test/page-2.json"
    refetch = "https://example.test/page-2-refetch.json"
    harness = _run(
        tmp_path,
        registry,
        {
            original: fixture_text("paginated_page_2.json"),
            refetch: fixture_text("paginated_page_2_refetch.json"),
        },
        content_type="application/json",
        dedupe={"mode": "identity", "identity_fields": ["acc_no", "name"]},
    )

    assert harness.summary.counts["stored"] == 1
    assert harness.summary.counts["deduped"] == 1
    entry = harness.summary.metadata["dedupe"][0]
    assert entry["mode"] == "identity"
    assert entry["basis"] == "identity"
    assert entry["reason"] == "near_duplicate_identity"
    assert entry["similarity"] == 1.0
    assert isinstance(entry["compared_document_id"], int)
    assert entry["compared_url"] == original


def test_mirrored_pdf_text_is_deduped_by_simhash(tmp_path, registry) -> None:
    first = "https://alpha.test/placement-report.txt"
    second = "https://beta.test/placement-report-mirror.txt"
    a = fixture_text("mirror_a.txt")
    b = fixture_text("mirror_b.txt")
    assert hamming(simhash64(a), simhash64(b)) <= 3

    harness = _run(
        tmp_path,
        registry,
        {first: a, second: b},
        content_type="text/plain",
    )

    assert harness.summary.counts["stored"] == 1
    assert harness.summary.counts["deduped"] == 1
    entry = harness.summary.metadata["dedupe"][0]
    assert entry["mode"] == "simhash"
    assert entry["basis"] == "extracted_text"
    assert entry["reason"] == "near_duplicate_simhash"
    assert entry["hamming_distance"] <= 3
    assert 0.0 <= entry["similarity"] < 1.0
    assert isinstance(entry["compared_document_id"], int)
    assert entry["text_sha256"] == text_sha256(b)


def test_distinct_pdfs_sharing_boilerplate_are_not_deduped(tmp_path, registry) -> None:
    """The 2026-11-18 defect: distinct PDFs collapsed on container bytes.

    These two fixtures share a byte-identical container-boilerplate object and
    seven boilerplate lines; only the report body differs. Raw-byte simhash (the
    old input) puts them at distance 0. Extracted text must keep them apart.
    """
    first = "https://alpha.test/annual-report-2023-24.pdf"
    second = "https://beta.test/annual-report-2024-25.pdf"
    a = fixture_bytes("annual_report_2023.pdf")
    b = fixture_bytes("annual_report_2024.pdf")
    # What the old code hashed: the container bytes decoded as UTF-8.
    raw_distance = hamming(
        simhash64(a.decode("utf-8", "replace")), simhash64(b.decode("utf-8", "replace"))
    )
    assert raw_distance <= 3
    texts = {first: pdf_text_sample(a), second: pdf_text_sample(b)}
    assert texts[first] and texts[second]
    assert hamming(simhash64(texts[first]), simhash64(texts[second])) > 3

    harness = _run(
        tmp_path,
        registry,
        {first: a, second: b},
        content_type="application/pdf",
    )

    assert harness.summary.counts["fetched"] == 2
    assert harness.summary.counts["stored"] == 2
    assert harness.summary.counts.get("deduped", 0) == 0
    assert len(harness.snapshots.rows) == 2
    for row in harness.snapshots.rows:
        assert row.dedupe_basis == "extracted_text"
        assert row.text_sha256 is not None
    stored = harness.documents.rows.values()
    assert {row.dedupe_mode for row in stored} == {"simhash"}
    assert {row.dedupe_basis for row in stored} == {"extracted_text"}
    assert all(row.text_sha256 for row in stored)


def test_pdf_editions_of_one_template_are_not_collapsed(tmp_path, registry) -> None:
    """Approximate text similarity may not drop a different edition.

    These two policy PDFs differ only in the academic year (two characters), so
    their text simhash distance is 0. They are still different documents: the
    2026-11-18 recovery dropped exactly this pair on the first attempt.
    """
    first = "https://alpha.test/policy-2022-23.pdf"
    second = "https://alpha.test/policy-2023-24.pdf"
    a = pdf_text_sample(fixture_bytes("policy_2022.pdf"))
    b = pdf_text_sample(fixture_bytes("policy_2023.pdf"))
    assert a and b and a != b
    assert text_sha256(a) != text_sha256(b)
    assert hamming(simhash64(a), simhash64(b)) <= 3

    harness = _run(
        tmp_path,
        registry,
        {first: fixture_bytes("policy_2022.pdf"), second: fixture_bytes("policy_2023.pdf")},
        content_type="application/pdf",
    )

    assert harness.summary.counts["stored"] == 2
    assert harness.summary.counts.get("deduped", 0) == 0
    assert {row.text_sha256 for row in harness.snapshots.rows} == {
        text_sha256(a),
        text_sha256(b),
    }
    # The fingerprint is still recorded, so a review can see the near miss.
    assert {row.dedupe_simhash for row in harness.documents.rows.values()} == {
        f"{simhash64(a):016x}",
        f"{simhash64(b):016x}",
    }


def test_byte_identical_pdf_is_deduped_by_sha(tmp_path, registry) -> None:
    first = "https://alpha.test/prospectus.pdf"
    second = "https://beta.test/prospectus-copy.pdf"
    body = fixture_bytes("annual_report_2023.pdf")

    harness = _run(
        tmp_path,
        registry,
        {first: body, second: body},
        content_type="application/pdf",
    )

    assert harness.summary.counts["stored"] == 1
    assert harness.summary.counts["deduped"] == 1
    entry = harness.summary.metadata["dedupe"][0]
    assert entry["reason"] == "duplicate_sha256"
    assert entry["basis"] == "raw_bytes"
    assert entry["similarity"] == 1.0
    assert entry["compared_url"] == first
    assert isinstance(entry["compared_document_id"], int)
    # The dropped document keeps its justification even without a snapshot.
    dropped = harness.documents.rows[second]
    assert dropped.dedupe_reason == "duplicate_sha256"
    assert dropped.dedupe_basis == "raw_bytes"
    assert dropped.dedupe_compared_document_id == harness.documents.rows[first].id


def test_pdfs_with_identical_extracted_text_are_deduped_at_text_level(tmp_path, registry) -> None:
    """Different bytes (recompressed/mirrored) but the same text is one document."""
    first = "https://alpha.test/annual-report-2023-24.pdf"
    second = "https://mirror.test/annual-report-2023-24.pdf"
    a = fixture_bytes("annual_report_2023.pdf")
    b = fixture_bytes("annual_report_2023_mirror.pdf")
    assert a != b

    harness = _run(
        tmp_path,
        registry,
        {first: a, second: b},
        content_type="application/pdf",
    )

    assert harness.summary.counts["stored"] == 1
    assert harness.summary.counts["deduped"] == 1
    entry = harness.summary.metadata["dedupe"][0]
    assert entry["reason"] == "duplicate_text_sha256"
    assert entry["basis"] == "extracted_text"
    assert entry["similarity"] == 1.0
    assert entry["text_sha256"] == text_sha256(pdf_text_sample(a))
    assert entry["compared_url"] == first


def test_pdf_extraction_failure_falls_back_to_exact_only(tmp_path, registry) -> None:
    """No readable text layer means exact bytes only; store, never drop."""
    first = "https://alpha.test/broken-a.pdf"
    second = "https://beta.test/broken-b.pdf"
    third = "https://gamma.test/broken-a-copy.pdf"
    a = b"%PDF-1.4\n% deliberately malformed\n1 0 obj << >> endobj\ntrailer << >>\n"
    b = a + b"% second distinct broken body\n"
    assert pdf_text_sample(a) is None and pdf_text_sample(b) is None

    harness = _run(
        tmp_path,
        registry,
        {first: a, second: b, third: a},
        content_type="application/pdf",
    )

    assert harness.summary.counts["stored"] == 2
    assert harness.summary.counts["deduped"] == 1
    # Distinct unreadable PDFs are both kept; the byte-identical copy is still
    # caught, and the decision says it was made on raw bytes.
    entry = harness.summary.metadata["dedupe"][0]
    assert entry["url"] == third
    assert entry["reason"] == "duplicate_sha256"
    assert entry["basis"] == "raw_bytes"
    for row in harness.snapshots.rows:
        assert row.dedupe_basis == "raw_bytes"
        assert row.text_sha256 is None
    for row in harness.documents.rows.values():
        assert row.dedupe_mode == "exact"
        assert row.dedupe_basis == "raw_bytes"
        assert row.text_sha256 is None


def test_pdf_text_sample_is_bounded_and_fail_safe() -> None:
    body = fixture_bytes("annual_report_2023.pdf")
    full = pdf_text_sample(body)
    assert full and full.startswith("St Xavier")
    assert pdf_text_sample(body, max_chars=40) == full[:40]
    assert pdf_text_sample(body, max_pages=1) == full  # single-page fixture
    assert pdf_text_sample(b"not a pdf at all") is None
    assert pdf_text_sample(b"") is None
    assert is_pdf_payload("application/pdf", b"") is True
    assert is_pdf_payload("application/octet-stream", b"%PDF-1.7\n...") is True
    assert is_pdf_payload("text/html", b"<html></html>") is False


def test_simhash_threshold_changes_the_decision(tmp_path, registry) -> None:
    first = "https://alpha.test/a.txt"
    second = "https://beta.test/b.txt"
    a = fixture_text("mirror_a.txt")
    b = fixture_text("mirror_b.txt")
    distance = hamming(simhash64(a), simhash64(b))
    assert 0 < distance <= 64

    # At exactly the observed distance the mirror is a duplicate.
    at = tmp_path / "at"
    at.mkdir()
    at_distance = _run(
        at,
        registry,
        {first: a, second: b},
        content_type="text/plain",
        dedupe={"mode": "simhash", "simhash_threshold": distance},
    )
    assert at_distance.summary.counts["deduped"] == 1

    # One below it, the same pair must both be stored.
    below_dir = tmp_path / "below"
    below_dir.mkdir()
    below = _run(
        below_dir,
        registry,
        {first: a, second: b},
        content_type="text/plain",
        dedupe={"mode": "simhash", "simhash_threshold": distance - 1},
    )
    assert below.summary.counts.get("stored", 0) == 2
    assert below.summary.counts.get("deduped", 0) == 0


def test_dedupe_settings_are_registry_driven_and_validated(registry) -> None:
    nabh = registry.sources.by_id("nabh_hospitals")
    settings = dedupe_settings(nabh)
    assert settings == DedupeSettings(
        mode="identity",
        simhash_threshold=3,
        identity_fields=("acc_no", "name"),
    )
    # A source that does not declare a policy gets content-based auto mode.
    auto = dedupe_settings(FakeSource(id="undocumented"))
    assert auto.mode is None
    assert effective_mode(auto, content_type="application/json", text="{}") == "exact"
    assert effective_mode(auto, content_type="text/plain", text="prose") == "simhash"

    # The institution evidence crawl uses text fingerprints over extracted PDF
    # text, bounded so the decision stays inside the fetch path.
    sites = dedupe_settings(registry.sources.by_id("institution_sites"))
    assert sites.mode == "simhash"
    assert sites.simhash_threshold == 3
    assert sites.pdf_text_max_pages == 3
    assert sites.pdf_text_max_chars == 20000

    with pytest.raises(ValueError):
        dedupe_settings(FakeSource(id="bad", params={"dedupe": {"mode": "fuzzy"}}))
    with pytest.raises(ValueError):
        dedupe_settings(FakeSource(id="bad", params={"dedupe": {"text_max_pages": 0}}))


def test_row_identities_are_schema_agnostic() -> None:
    payload = json.dumps(
        {
            "rows": [{"acc_no": "A-1"}, {"acc_no": "A-2"}],
            "nested": json.dumps({"name": "Beta Hospital"}),
        }
    )
    identities = row_identities(payload, ["acc_no", "name"])
    assert identities == frozenset({"acc_no=a-1", "acc_no=a-2", "name=beta hospital"})
    # Unknown fields must yield "unknown", never a spurious duplicate token.
    assert row_identities(payload, ["missing"]) == frozenset()
