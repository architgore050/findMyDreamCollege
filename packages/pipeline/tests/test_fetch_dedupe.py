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
    row_identities,
    simhash64,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dedupe"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


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
    assert entry["reason"] == "near_duplicate_identity"
    assert entry["similarity"] == 1.0
    assert isinstance(entry["compared_document_id"], int)
    assert entry["compared_url"] == original


def test_mirrored_pdf_text_is_deduped_by_simhash(tmp_path, registry) -> None:
    first = "https://alpha.test/placement-report.pdf"
    second = "https://beta.test/placement-report-mirror.pdf"
    a = fixture_text("mirror_a.txt")
    b = fixture_text("mirror_b.txt")
    assert hamming(simhash64(a), simhash64(b)) <= 3

    harness = _run(
        tmp_path,
        registry,
        {first: a, second: b},
        content_type="application/pdf",
    )

    assert harness.summary.counts["stored"] == 1
    assert harness.summary.counts["deduped"] == 1
    entry = harness.summary.metadata["dedupe"][0]
    assert entry["mode"] == "simhash"
    assert entry["reason"] == "near_duplicate_simhash"
    assert entry["hamming_distance"] <= 3
    assert 0.0 <= entry["similarity"] < 1.0
    assert isinstance(entry["compared_document_id"], int)


def test_simhash_threshold_changes_the_decision(tmp_path, registry) -> None:
    first = "https://alpha.test/a.pdf"
    second = "https://beta.test/b.pdf"
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
        content_type="application/pdf",
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
        content_type="application/pdf",
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

    with pytest.raises(ValueError):
        dedupe_settings(FakeSource(id="bad", params={"dedupe": {"mode": "fuzzy"}}))


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
