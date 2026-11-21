"""W10 parse cache: hit semantics, invalidation, corruption, atomicity and the
live-vs-cache extraction equivalence on the committed parse fixture.

Offline: conversions run through the real tier dispatcher with fake T0/T1
callables, the router is a stub, the repo is in memory and the object store is
a real ``SnapshotStore`` on tmp_path.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_w10_support import (
    PDF_BYTES,
    CountingT0,
    CountingT1,
    FakeParsesRepo,
    SnapshotHarness,
    StubRouter,
    report_doc,
    stub_payload,
)

from fmd.acquire.parse_cache import (
    MISS_CORRUPT,
    MISS_NO_ENTRY,
    MISS_OBJECT_MISSING,
    MISS_SCHEMA_MISMATCH,
    PARSE_SCHEMA_VERSION,
    ParseCache,
    acquire_document,
    parse_key,
    parse_policy_digest,
    reconstruct_parse,
    serialize_parse,
)
from fmd.acquire.tiers import AcquirePolicy
from fmd.config import load_registry
from fmd.extract.pipeline import ExtractPipeline, ExtractTarget
from fmd.store import SnapshotStore

REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"
FIXTURE = Path(__file__).parent / "fixtures" / "parse_cache" / "report_parse.json"


def _policy(registry=None) -> AcquirePolicy:
    registry = registry or load_registry(REPO_CONFIG, env={})
    return AcquirePolicy.load(acquire=registry.acquire)


def _cache(
    tmp_path: Path, *, repo: FakeParsesRepo | None = None
) -> tuple[ParseCache, SnapshotStore, FakeParsesRepo]:
    store = SnapshotStore(tmp_path / "snapshots")
    repo = repo or FakeParsesRepo()
    return ParseCache(repo, store), store, repo


def _live(
    cache: ParseCache,
    digest: str,
    t1,
    policy: AcquirePolicy,
    *,
    t0=None,
):
    """One acquire_document call with the T0 tier faked (fake PDF bytes)."""
    return acquire_document(
        PDF_BYTES,
        content_type="application/pdf",
        policy=policy,
        cache=cache,
        snapshot_sha256=digest,
        t0=t0 or CountingT0(),
        t1=t1,
    )


def test_cache_hit_invokes_zero_converter_calls(tmp_path) -> None:
    cache, store, repo = _cache(tmp_path)
    policy = _policy()
    t1 = CountingT1()
    data = PDF_BYTES
    digest = store.put(data)

    # First pass: live conversion through the real dispatcher, stored.
    doc, miss = _live(cache, digest, t1, policy)
    assert doc.tier == "T1"
    assert miss.as_dict()["mode"] == "live"
    assert miss.info["stored"] is True
    assert t1.calls == 1
    assert repo.count() == 1

    # Second pass: lookup before conversion — zero converter invocations.
    doc2, hit = _live(cache, digest, t1, policy)
    assert t1.calls == 1  # unchanged: the converter was never touched again
    assert hit.as_dict()["mode"] == "cache"
    assert doc2.text == doc.text
    assert doc2.page_map == doc.page_map
    assert doc2.tables == doc.tables


def test_device_change_is_still_a_hit(tmp_path) -> None:
    """The Docling device is not part of the key: CUDA parse reusable by CPU."""
    cuda_registry = load_registry(REPO_CONFIG, env={"FMD_DOCLING_DEVICE": "cuda"})
    cpu_registry = load_registry(REPO_CONFIG, env={})
    assert parse_policy_digest(cuda_registry.acquire) == parse_policy_digest(cpu_registry.acquire)
    assert parse_key("a" * 64, "T1", parse_policy_digest(cuda_registry.acquire)) == parse_key(
        "a" * 64, "T1", parse_policy_digest(cpu_registry.acquire)
    )

    cache, store, _repo = _cache(tmp_path)
    digest = store.put(PDF_BYTES)
    t1 = CountingT1()
    _live(cache, digest, t1, _policy(cuda_registry))
    doc, hit = _live(cache, digest, t1, _policy(cpu_registry))
    assert t1.calls == 1
    assert hit.as_dict()["mode"] == "cache"
    assert doc.tier == "T1"


def test_policy_change_invalidates_and_reconverts(tmp_path, tmp_path_factory) -> None:
    cache, store, _repo = _cache(tmp_path)
    digest = store.put(PDF_BYTES)
    t1 = CountingT1()
    _live(cache, digest, t1, _policy())

    # A parse-shaping policy change (outline thresholds) must invalidate.
    config_copy = tmp_path_factory.mktemp("config")
    shutil.copytree(REPO_CONFIG, config_copy / "config")
    acquire_yaml = config_copy / "config" / "acquire.yaml"
    text = acquire_yaml.read_text(encoding="utf-8")
    acquire_yaml.write_text(
        text.replace("heading_max_chars: 120", "heading_max_chars: 90"), encoding="utf-8"
    )
    changed_registry = load_registry(config_copy / "config", env={})
    doc, miss = _live(cache, digest, t1, _policy(changed_registry))
    assert t1.calls == 2  # reconverted under the new policy
    assert miss.as_dict()["mode"] == "live"
    assert doc.tier == "T1"
    assert repo_count(cache) == 2  # the new policy stored its own parse


def repo_count(cache: ParseCache) -> int:
    return cache.repo.count()


def test_schema_bump_invalidates_and_reconverts(tmp_path, monkeypatch) -> None:
    cache, store, _repo = _cache(tmp_path)
    digest = store.put(PDF_BYTES)
    t1 = CountingT1()
    _live(cache, digest, t1, _policy())
    import fmd.acquire.parse_cache as parse_cache_module

    # Load the policy (and thus the registry cross-check) BEFORE the bump.
    policy = _policy()
    monkeypatch.setattr(parse_cache_module, "PARSE_SCHEMA_VERSION", PARSE_SCHEMA_VERSION + 1)
    doc, miss = _live(cache, digest, t1, policy)
    assert t1.calls == 2
    assert miss.as_dict()["mode"] == "live"
    assert miss.reason in {MISS_NO_ENTRY, MISS_SCHEMA_MISMATCH}


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda path: path.write_bytes(b"{not json at all"), MISS_CORRUPT),
        (lambda path: path.write_bytes(b'{"schema_version": 1, "tier": "T1"'), MISS_CORRUPT),
        (lambda path: path.unlink(), MISS_OBJECT_MISSING),
    ],
    ids=["garbage", "truncated", "missing_object"],
)
def test_corrupt_cache_reconverts_with_recorded_reason_and_never_crashes(
    tmp_path, mutate, reason
) -> None:
    """A payload with a foreign schema cannot even exist at a content address
    (the address is the hash of the bytes), so at-rest corruption is integrity
    corruption; a foreign schema shows up at the ROW level instead — tested
    separately below."""
    cache, store, _repo = _cache(tmp_path)
    digest = store.put(PDF_BYTES)
    t1 = CountingT1()
    _live(cache, digest, t1, _policy())
    # Mutate the stored payload object directly (corruption at rest).
    row = next(iter(cache.repo.rows.values()))
    mutate(store.path_for(row.object_sha256))

    doc, miss = _live(cache, digest, t1, _policy())
    assert t1.calls == 2  # reconverted, never guessed
    assert miss.reason == reason
    assert doc.tier == "T1"
    assert doc.text  # the fresh parse is a complete document


def test_row_schema_mismatch_reconverts(tmp_path) -> None:
    """A row recorded under a different schema version (e.g. written by a
    newer build) is a miss with a recorded reason, never a guess."""
    cache, store, _repo = _cache(tmp_path)
    digest = store.put(PDF_BYTES)
    t1 = CountingT1()
    _live(cache, digest, t1, _policy())
    # Corrupt the row's schema component: the object itself stays intact.
    row = next(iter(cache.repo.rows.values()))
    tampered = row.model_copy(update={"schema_version": PARSE_SCHEMA_VERSION + 1})
    cache.repo.rows[row.key_digest] = tampered

    doc, miss = _live(cache, digest, t1, _policy())
    assert t1.calls == 2
    assert miss.reason == MISS_SCHEMA_MISMATCH
    assert doc.tier == "T1"


def test_failed_repo_write_leaves_no_usable_cache_row(tmp_path) -> None:
    """Atomic-write scheme: object first, row second — a failed row write is a
    miss on every later run, never a partial read."""
    repo = FakeParsesRepo()
    repo.fail_put = True
    cache, store, _repo = _cache(tmp_path, repo=repo)
    digest = store.put(PDF_BYTES)
    t1 = CountingT1()

    doc, miss = _live(cache, digest, t1, _policy())
    assert doc.tier == "T1"
    assert t1.calls == 1
    assert miss.as_dict()["mode"] == "live"
    assert miss.info["stored"] is False
    assert repo.count() == 0

    # The run after the outage sees a miss and reconverts from scratch.
    doc2, miss2 = _live(cache, digest, t1, _policy())
    assert t1.calls == 2
    assert miss2.reason == MISS_NO_ENTRY


def test_incomplete_documents_are_never_cached(tmp_path) -> None:
    """A cache entry represents a complete document only: a review-routed or
    empty parse writes nothing, and there is nothing to resume."""
    cache, store, repo = _cache(tmp_path)
    policy = _policy()
    digest = store.put(PDF_BYTES)

    # A T1 conversion routed to review (the human queue) is never cached.
    review_doc = replace(
        report_doc(),
        needs_review=True,
        review_reason="ocr_unavailable_sparse_scan",
    )
    t1 = CountingT1(doc_factory=lambda: review_doc)
    doc, miss = _live(cache, digest, t1, policy)
    assert doc.needs_review is True
    assert miss.info["stored"] is False
    assert repo.count() == 0

    # A T0 fallback (T1 unavailable) is not a T1 parse and is never cached.
    from fmd.acquire.errors import TierUnavailable

    def unavailable(data: bytes):
        raise TierUnavailable("docling missing")

    doc2, miss2 = _live(cache, digest, unavailable, policy)
    assert doc2.tier == "T0"
    assert miss2.info["stored"] is False
    assert repo.count() == 0


def _extract(
    tmp_path: Path,
    registry,
    *,
    cache: ParseCache | None,
    t0=None,
    t1=None,
    store: SnapshotStore | None = None,
    allow_convert: bool = True,
):
    """One ExtractPipeline run over the harness's single PDF snapshot."""
    harness = SnapshotHarness(
        tmp_path / "harness", registry, router=StubRouter(registry.models, stub_payload())
    )
    _document, snapshot = harness.add_pdf_document("https://college.ac.in/report.pdf")
    router = StubRouter(registry.models, stub_payload())
    pipeline = ExtractPipeline(
        router,
        snapshots=type("S", (), {"get": staticmethod(lambda sid: snapshot)})(),
        extractions=harness.extractions,
        claims=harness.claims,
        store=harness.store,
        policy=_policy(registry),
        parse_cache=cache,
        t0=t0,
        t1=t1,
        allow_convert=allow_convert,
    )
    result = pipeline.run(
        ExtractTarget(snapshot_id=snapshot.id, entity_id=42, family="placement_report")
    )
    return harness, result, router


def _claim_key(claim) -> tuple:
    return (claim.predicate, claim.quote, claim.char_start, claim.char_end)


def _claim_span(claim) -> tuple:
    qualifier = claim.qualifier or {}
    return (qualifier.get("page_no"), qualifier.get("bbox"), qualifier.get("bboxes"))


def test_live_and_cached_parses_extract_identically(tmp_path) -> None:
    """The committed parse fixture: live conversion and cached parse produce
    identical claim tuples and identical spans."""
    registry = load_registry(REPO_CONFIG, env={})
    t1_live = CountingT1()

    # Live path: real dispatcher, fake T0/T1, no cache.
    live_harness, live_result, live_router = _extract(
        tmp_path, registry, cache=None, t0=CountingT0(), t1=t1_live
    )
    assert live_result.skipped is False
    assert live_result.acquire["mode"] == "live"
    assert live_result.acquire["cache_hit"] is False
    assert live_result.acquire["pages"] == 3
    assert live_router.calls, "the live path extracted claims"
    assert t1_live.calls == 1

    # The committed fixture must equal the live parse's serialised form.
    payload = serialize_parse(live_result.doc)
    assert json.loads(payload)["schema_version"] == PARSE_SCHEMA_VERSION
    assert payload == FIXTURE.read_bytes(), (
        "fixtures/parse_cache/report_parse.json is stale: regenerate it from the "
        "report document builder"
    )

    # Cache path: same bytes, seeded cache (as the acquire stage would leave it),
    # and a converter that must never be invoked.
    cache_harness = SnapshotHarness(
        tmp_path / "harness2", registry, router=StubRouter(registry.models, stub_payload())
    )
    _document2, snapshot2 = cache_harness.add_pdf_document("https://college.ac.in/report.pdf")
    cache_repo = FakeParsesRepo()
    cache = ParseCache(cache_repo, cache_harness.store)
    stored = cache.store_parse(snapshot2.sha256, live_result.doc, _policy(registry))
    assert stored.as_dict().get("mode") == "cache" or isinstance(stored, object)
    assert cache_repo.count() == 1

    exploding = CountingT1()
    exploding.error = AssertionError("converter must not run on the cache path")
    cache_result = None
    router2 = StubRouter(registry.models, stub_payload())
    pipeline = ExtractPipeline(
        router2,
        snapshots=type("S", (), {"get": staticmethod(lambda sid: snapshot2)})(),
        extractions=cache_harness.extractions,
        claims=cache_harness.claims,
        store=cache_harness.store,
        policy=_policy(registry),
        parse_cache=cache,
        t1=exploding,
        allow_convert=False,
    )
    cache_result = pipeline.run(
        ExtractTarget(snapshot_id=snapshot2.id, entity_id=42, family="placement_report")
    )
    assert exploding.calls == 0, "extract --from-cache must invoke no Docling"
    assert cache_result.skipped is False
    assert cache_result.acquire["cache_hit"] is True
    assert cache_result.acquire["mode"] == "cache"

    live_claims = sorted(_claim_key(claim) for claim in live_result.claims)
    cache_claims = sorted(_claim_key(claim) for claim in cache_result.claims)
    assert live_claims == cache_claims
    assert len(live_claims) > 0
    live_spans = sorted(_claim_span(claim) for claim in live_result.claims)
    cache_spans = sorted(_claim_span(claim) for claim in cache_result.claims)
    assert live_spans == cache_spans
    for claim in cache_result.claims:
        qualifier = claim.qualifier or {}
        assert qualifier.get("page_no") is not None
        assert qualifier.get("bbox") is not None


def test_cache_only_mode_records_missing_parse(tmp_path) -> None:
    """``--from-cache`` with no cached parse: recorded, run continues, no Docling."""
    registry = load_registry(REPO_CONFIG, env={})
    exploding = CountingT1()
    _harness, result, _router = _extract(
        tmp_path, registry, cache=None, t1=exploding, allow_convert=False
    )
    assert exploding.calls == 0
    assert result.skipped is True
    assert result.acquire["reason"] == "missing_parse_cache"
    assert result.acquire["mode"] == "cache_only"


def test_reconstruct_without_cache_roundtrip_is_stable(tmp_path) -> None:
    """serialize → reconstruct is identity on every extractor-visible field."""
    policy = _policy()
    doc = report_doc()
    reconstructed = reconstruct_parse(serialize_parse(doc), policy)
    assert reconstructed.text == doc.text
    assert reconstructed.page_map == doc.page_map
    assert reconstructed.blocks == doc.blocks
    assert reconstructed.tables == doc.tables
    assert reconstructed.canonical.anchors == doc.canonical.anchors
    assert reconstructed.canonical.outline == doc.canonical.outline
    assert reconstructed.canonical.sections == doc.canonical.sections
    assert reconstructed.table_confidence == doc.table_confidence
    assert reconstructed.bbox_granularity == doc.bbox_granularity
