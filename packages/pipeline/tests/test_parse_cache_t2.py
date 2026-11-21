"""Wave v1-w3: the T2 parse cache — engine-pinned keys, invalidation on an
engine-version bump, device exclusion, live-vs-cached extraction equality and
the honest refusals (HTTP engines, engine-meta drift, incomplete documents).

Offline: the tier dispatcher is faked at the T1 seam (a "converter" whose
factory returns the OCR-shaped document), the repo is in memory, the object
store is a real ``SnapshotStore`` on tmp_path. The engine fingerprint's
distribution version is patched, never downloaded.
"""

from __future__ import annotations

import json
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
    report_pages,
    stub_payload,
)

import fmd.acquire.parse_cache as parse_cache_module
from fmd.acquire.canonical import build_canonical
from fmd.acquire.outline import OutlinePolicy
from fmd.acquire.parse_cache import (
    MISS_ENGINE_MISMATCH,
    MISS_ENGINE_UNPINNED,
    MISS_NOT_PDF,
    ParseCache,
    acquire_document,
    parse_key,
    parse_policy_digest,
    reconstruct_parse,
    serialize_parse,
    t2_engine_fingerprint,
)
from fmd.acquire.types import ExtractedDoc, PageSpan, TextBlock
from fmd.config import load_registry
from fmd.extract.pipeline import ExtractPipeline, ExtractTarget

REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"
FAKE_ENGINE_VERSION = "3.10.0-test"


@pytest.fixture()
def pinned_engine(monkeypatch):
    """Pin the locally derived rapidocr version; return the fingerprint."""

    def _version(distribution: str) -> str | None:
        return FAKE_ENGINE_VERSION if distribution == "rapidocr" else None

    monkeypatch.setattr(parse_cache_module, "_distribution_version", _version)
    return _fingerprint(_policy())


def _policy(registry=None):
    registry = registry or load_registry(REPO_CONFIG, env={})
    from fmd.acquire.tiers import AcquirePolicy

    return AcquirePolicy.load(acquire=registry.acquire)


def _fingerprint(policy):
    fp = t2_engine_fingerprint(policy)
    assert fp is not None, "the committed registry must pin a cacheable T2 engine"
    return fp


def _t2_cache(
    tmp_path: Path, repo: FakeParsesRepo | None = None
) -> tuple[ParseCache, any, FakeParsesRepo]:
    from fmd.store import SnapshotStore

    store = SnapshotStore(tmp_path / "snapshots")
    repo = repo or FakeParsesRepo()
    return ParseCache(repo, store, tier="T2"), store, repo


def rapidocr_doc(
    engine_version: str = FAKE_ENGINE_VERSION, fingerprint: dict | None = None
) -> ExtractedDoc:
    """A T2 document exactly as ``tiers._ocr_doc`` builds it from the backend.

    Blocks carry ``granularity="block"`` and a ``block_id`` (OCR provenance),
    the engine_meta block mirrors ``rapidocr_backend._engine_meta`` (including
    the environment-specific ``paths`` key the store gate must ignore).
    """
    blocks = [
        TextBlock(
            page_no=1,
            text="Admission and Fee Structure 2026",
            bbox=(40.0, 40.0, 400.0, 60.0),
            granularity="block",
            block_id=0,
        ),
        TextBlock(
            page_no=1,
            text="The institute offers the B.Sc. Medical Laboratory Technology programme.",
            bbox=(40.0, 120.0, 500.0, 140.0),
            granularity="block",
            block_id=1,
        ),
    ]
    canonical = build_canonical(blocks, report_pages(), outline_policy=OutlinePolicy())
    fp = fingerprint or {
        "engine": "rapidocr",
        "engine_version": engine_version,
        "model_ids": {"det": "det-x", "rec": "rec-y", "rec_keys": "keys-z"},
    }
    return ExtractedDoc(
        text=canonical.text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=[],
        tier="T2",
        tool=str(fp["engine"]),
        tool_version=str(fp["engine_version"]),
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity="block",
        engine_meta={
            "engine": fp["engine"],
            "engine_version": fp["engine_version"],
            "model_ids": {**fp["model_ids"], "paths": {"rec": "/models/rec.onnx"}},
            "granularity": "block",
            "device": "cpu",
        },
    )


# --------------------------------------------------------------------- keying


def test_t1_key_layout_is_unchanged_and_t2_keys_are_separated():
    digest = "a" * 64
    policy_digest = parse_policy_digest(_policy(), tier="T1")
    # None and omitted are identical: the T1 layout is byte-stable from W10.
    assert parse_key(digest, "T1", policy_digest) == parse_key(digest, "T1", policy_digest, None)
    assert parse_key(digest, "T1", policy_digest) != parse_key(digest, "T2", policy_digest)


def test_t2_fingerprint_in_the_key_and_engine_bump_changes_it(pinned_engine):
    digest = "b" * 64
    t2_digest = parse_policy_digest(_policy(), tier="T2")
    base = parse_key(digest, "T2", t2_digest, pinned_engine)
    bumped = dict(pinned_engine, engine_version="9.9.9-next")
    assert base != parse_key(digest, "T2", t2_digest, bumped)
    ids_changed = dict(pinned_engine, model_ids=dict(pinned_engine["model_ids"], rec="other-rec"))
    assert base != parse_key(digest, "T2", t2_digest, ids_changed)


def test_t2_device_change_is_still_a_hit(tmp_path, monkeypatch, pinned_engine):
    """The device is excluded from the T2 key, as from the T1 key: a CUDA OCR
    parse is reusable by a CPU run and vice versa."""
    cuda_registry = load_registry(REPO_CONFIG, env={"FMD_OCR_DEVICE": "cuda"})
    cpu_registry = load_registry(REPO_CONFIG, env={})
    cuda_policy = _policy(cuda_registry)
    cpu_policy = _policy(cpu_registry)
    fp_cuda = _fingerprint(cuda_policy)
    fp_cpu = _fingerprint(cpu_policy)
    assert fp_cuda["engine_version"] == fp_cpu["engine_version"]
    assert fp_cuda["model_ids"] == fp_cpu["model_ids"]
    assert parse_policy_digest(cuda_registry.acquire, tier="T2") == parse_policy_digest(
        cpu_registry.acquire, tier="T2"
    )

    cache, store, _repo = _t2_cache(tmp_path)
    digest = store.put(PDF_BYTES)
    doc_factory = lambda: rapidocr_doc(fingerprint=fp_cpu)  # noqa: E731
    _live_t2(cache, digest, cpu_policy, doc_factory)
    doc, hit = _live_t2(cache, digest, cuda_policy, doc_factory)
    assert hit.as_dict()["mode"] == "cache"
    assert doc.tier == "T2"


def _live_t2(cache, digest, policy, doc_or_t1, *, t2_cache=None, raw_t1=False):
    """One acquire_document call with the tier dispatcher faked at T1's seam.

    ``doc_or_t1`` is either a document factory (wrapped in a counting T1 fake)
    or, with ``raw_t1=True``, a ready T1 callable (e.g. a raiser).
    """
    t1 = doc_or_t1 if raw_t1 else CountingT1(doc_factory=doc_or_t1)
    return acquire_document(
        PDF_BYTES,
        content_type="application/pdf",
        policy=policy,
        cache=cache,
        t2_cache=t2_cache,
        snapshot_sha256=digest,
        t0=CountingT0(),
        t1=t1,
    )


# -------------------------------------------------------------- invalidation


def test_engine_version_bump_invalidates_and_reconverts(tmp_path, monkeypatch, pinned_engine):
    """The explicit brief requirement: bumping the engine version must miss,
    reconvert and store under a new key — never reuse the stale parse."""
    cache, store, repo = _t2_cache(tmp_path)
    digest = store.put(PDF_BYTES)
    policy = _policy()

    def doc_factory() -> ExtractedDoc:
        # The document always reports the engine identity the run actually
        # used — here, whatever the (patched) config currently pins.
        return rapidocr_doc(fingerprint=_fingerprint(policy))

    _live_t2(cache, digest, policy, doc_factory)
    assert repo.count() == 1

    monkeypatch.setattr(
        parse_cache_module,
        "_distribution_version",
        lambda dist: "3.11.0-bumped" if dist == "rapidocr" else None,
    )
    doc, miss = _live_t2(cache, digest, policy, doc_factory)
    assert miss.as_dict()["mode"] == "live"
    assert miss.reason == "no_cache_entry"
    assert doc.tier == "T2"
    assert repo.count() == 2  # the bumped engine stored its own parse

    # And the bumped engine hits its own entry on the next run.
    _doc, hit = _live_t2(cache, digest, policy, doc_factory)
    assert hit.as_dict()["mode"] == "cache"


def test_model_id_change_invalidates(tmp_path, monkeypatch, pinned_engine, tmp_path_factory):
    """A model swap (different rec model id in the registry) is a new key."""
    cache, store, repo = _t2_cache(tmp_path)
    digest = store.put(PDF_BYTES)
    policy = _policy()
    _live_t2(cache, digest, policy, lambda: rapidocr_doc(fingerprint=pinned_engine))

    config_copy = tmp_path_factory.mktemp("config")
    import shutil

    shutil.copytree(REPO_CONFIG, config_copy / "config")
    acquire_yaml = config_copy / "config" / "acquire.yaml"
    text = acquire_yaml.read_text(encoding="utf-8")
    needle = "rec: devanagari_PP-OCRv5_rec_mobile.onnx"
    assert needle in text, "the committed registry pins the devanagari rec model"
    acquire_yaml.write_text(
        text.replace(needle, "rec: swapped_PP-OCRv5_rec.onnx"), encoding="utf-8"
    )
    swapped_policy = _policy(load_registry(config_copy / "config", env={}))

    def swapped_factory() -> ExtractedDoc:
        return rapidocr_doc(fingerprint=_fingerprint(swapped_policy))

    doc, miss = _live_t2(cache, digest, swapped_policy, swapped_factory)
    assert miss.as_dict()["mode"] == "live"
    assert miss.reason == "no_cache_entry"
    assert doc.tier == "T2"
    assert repo.count() == 2


# ------------------------------------------------------------------ equality


def test_t2_serialize_reconstruct_roundtrip_is_identity(pinned_engine):
    policy = _policy()
    doc = rapidocr_doc(fingerprint=pinned_engine)
    reconstructed = reconstruct_parse(serialize_parse(doc), policy)
    assert reconstructed.text == doc.text
    assert reconstructed.page_map == doc.page_map
    assert reconstructed.blocks == doc.blocks
    assert reconstructed.tables == doc.tables
    assert reconstructed.canonical.anchors == doc.canonical.anchors
    assert reconstructed.canonical.outline == doc.canonical.outline
    assert reconstructed.canonical.sections == doc.canonical.sections
    assert reconstructed.bbox_granularity == doc.bbox_granularity == "block"
    assert reconstructed.tool == doc.tool == "rapidocr"
    assert reconstructed.tool_version == doc.tool_version
    assert reconstructed.engine_meta == doc.engine_meta


def test_live_and_cached_t2_parses_extract_identically(tmp_path, pinned_engine):
    """The T2 twin of the W10 equality proof: a live OCR run and its cached
    replay produce identical claims and identical qualifier spans."""
    registry = load_registry(REPO_CONFIG, env={})
    t1_live = CountingT1(doc_factory=lambda: rapidocr_doc(fingerprint=pinned_engine))

    live_harness, live_result, live_router = _extract_t2(
        tmp_path / "live", registry, t1=t1_live
    )
    assert live_result.skipped is False
    assert live_result.acquire["mode"] == "live"
    assert live_router.calls, "the live path extracted claims"
    assert t1_live.calls == 1

    cache_harness = SnapshotHarness(
        tmp_path / "cached", registry, router=StubRouter(registry.models, stub_payload())
    )
    _document2, snapshot2 = cache_harness.add_pdf_document("https://college.ac.in/report.pdf")
    cache_repo = FakeParsesRepo()
    t2_cache = ParseCache(cache_repo, cache_harness.store, tier="T2")
    stored = t2_cache.store_parse(snapshot2.sha256, live_result.doc, _policy(registry))
    assert stored.as_dict()["mode"] == "cache"
    assert cache_repo.count() == 1

    exploding = CountingT1()
    exploding.error = AssertionError("converter must not run on the cache path")
    payload = json.loads(json.dumps(stub_payload()))
    pipeline = ExtractPipeline(
        StubRouter(registry.models, payload),
        snapshots=type("S", (), {"get": staticmethod(lambda sid: snapshot2)})(),
        extractions=cache_harness.extractions,
        claims=cache_harness.claims,
        store=cache_harness.store,
        policy=_policy(registry),
        parse_cache=None,
        t2_parse_cache=t2_cache,
        t1=exploding,
        allow_convert=False,
    )
    cache_result = pipeline.run(
        ExtractTarget(snapshot_id=snapshot2.id, entity_id=42, family="placement_report")
    )
    assert exploding.calls == 0, "a cached T2 parse must invoke no OCR run"
    assert cache_result.skipped is False
    assert cache_result.acquire["mode"] == "cache"
    assert cache_result.acquire["cache_hit"] is True

    def key(claim):
        return (claim.predicate, claim.quote, claim.char_start, claim.char_end)

    def span(claim):
        qualifier = claim.qualifier or {}
        return (qualifier.get("page_no"), qualifier.get("bbox"), qualifier.get("bboxes"))

    live_claims = sorted(key(claim) for claim in live_result.claims)
    cache_claims = sorted(key(claim) for claim in cache_result.claims)
    assert live_claims == cache_claims
    assert len(live_claims) > 0
    assert sorted(span(claim) for claim in live_result.claims) == sorted(
        span(claim) for claim in cache_result.claims
    )


def _extract_t2(tmp_path: Path, registry, *, t1):
    """One ExtractPipeline run over a snapshot whose conversion lands on T2."""
    harness = SnapshotHarness(
        tmp_path, registry, router=StubRouter(registry.models, stub_payload())
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
        parse_cache=None,
        t2_parse_cache=None,
        t0=CountingT0(),
        t1=t1,
        allow_convert=True,
    )
    result = pipeline.run(
        ExtractTarget(snapshot_id=snapshot.id, entity_id=42, family="placement_report")
    )
    return harness, result, router


# ------------------------------------------------------------------ refusals


def test_http_engine_t2_results_are_refused(tmp_path, monkeypatch):
    """qwen3vl/paddleocr have no locally pinnable engine version: their T2
    results are refused with a stable reason, never cached."""
    monkeypatch.setattr(parse_cache_module, "_distribution_version", lambda dist: None)
    registry = load_registry(REPO_CONFIG, env={"FMD_T2_BACKEND": "qwen3vl"})
    assert t2_engine_fingerprint(registry.acquire) is None
    policy = _policy(registry)
    cache, store, repo = _t2_cache(tmp_path)
    digest = store.put(PDF_BYTES)

    qwen_doc = replace(
        rapidocr_doc(),
        tool="qwen3vl",
        tool_version=None,
        engine_meta={
            "engine": "qwen3vl",
            "engine_version": None,
            "model_ids": {"model": "Qwen3-VL-8B-Instruct-FP8"},
        },
    )
    doc, miss = _live_t2(cache, digest, policy, lambda: qwen_doc)
    assert doc.tier == "T2"
    assert miss.info.get("stored") is False
    assert miss.info.get("store_refusal") == MISS_ENGINE_UNPINNED
    assert repo.count() == 0


def test_engine_meta_mismatch_refuses_to_cache(tmp_path, monkeypatch, pinned_engine):
    """The document's own provenance must agree with the locally derived
    fingerprint: env drift between the run and the config read refuses the
    store (the key must never describe an identity the run did not use)."""
    cache, store, repo = _t2_cache(tmp_path)
    digest = store.put(PDF_BYTES)
    policy = _policy()
    drifted_meta = dict(
        rapidocr_doc(fingerprint=pinned_engine).engine_meta, engine_version="3.9.0-other"
    )
    drifted = replace(
        rapidocr_doc(fingerprint=pinned_engine),
        tool_version="3.9.0-other",
        engine_meta=drifted_meta,
    )
    doc, miss = _live_t2(cache, digest, policy, lambda: drifted)
    assert doc.tier == "T2"
    assert miss.info.get("stored") is False
    assert miss.info.get("store_refusal") == MISS_ENGINE_MISMATCH
    assert repo.count() == 0


def test_incomplete_t2_documents_are_never_cached(tmp_path, pinned_engine):
    """``cacheable()`` refuses review-routed/empty parses before any store; a
    direct store call also refuses a T0 fallback result. Nothing is written."""
    cache, store, repo = _t2_cache(tmp_path)
    digest = store.put(PDF_BYTES)
    policy = _policy()

    review = replace(
        rapidocr_doc(fingerprint=pinned_engine),
        needs_review=True,
        review_reason="ocr_unavailable_sparse_scan",
    )
    doc, miss = _live_t2(cache, digest, policy, lambda: review)
    assert doc.needs_review is True
    assert miss.info.get("stored") is False
    assert miss.info.get("not_cacheable_tier") == "T2"
    assert repo.count() == 0

    from fmd.acquire.errors import TierUnavailable

    def unavailable(data: bytes):
        raise TierUnavailable("rapidocr missing")

    doc2, miss2 = _live_t2(cache, digest, policy, unavailable, raw_t1=True)
    assert doc2.tier == "T0"
    assert miss2.info.get("stored") is False
    assert repo.count() == 0

    # The store gate itself (bypassing cacheable()): a T0 result is refused.
    refusal = cache.store_parse(digest, doc2, policy)
    assert refusal.reason == MISS_NOT_PDF


def test_t2_parse_is_refused_by_a_t1_tier_cache(tmp_path, pinned_engine):
    """Cross-tier hardening (verifier finding): a T2 parse handed to a T1-tier
    cache must never be stored — a fingerprint-free T1 key would let an OCR
    parse survive an engine-version bump."""
    from fmd.store import SnapshotStore

    t1_repo = FakeParsesRepo()
    t1_cache = ParseCache(t1_repo, SnapshotStore(tmp_path / "t1store"), tier="T1")
    doc = rapidocr_doc(fingerprint=pinned_engine)
    refusal = t1_cache.store_parse("d" * 64, doc, _policy())
    assert refusal.reason == MISS_NOT_PDF
    assert "does not match cache tier" in (refusal.detail or "")
    assert t1_repo.count() == 0


def test_t2_cache_skips_the_ocr_run_on_repeat_acquisition(tmp_path, pinned_engine):
    """Second acquisition of the same scan: a verified T2 hit invokes no tier
    function at all — the OCR work is never repeated."""
    cache, store, _repo = _t2_cache(tmp_path)
    digest = store.put(PDF_BYTES)
    policy = _policy()
    t1 = CountingT1(doc_factory=lambda: rapidocr_doc(fingerprint=pinned_engine))
    _live_t2(cache, digest, policy, lambda: rapidocr_doc(fingerprint=pinned_engine), t2_cache=cache)
    # Same call shape as the stage: T1 cache first, T2 cache second.
    _doc, miss = _live_t2(
        ParseCache(FakeParsesRepo(), store),  # T1 cache: always a miss here
        digest,
        policy,
        t1,
        t2_cache=cache,
    )
    assert miss.as_dict()["mode"] == "cache"
    assert t1.calls == 0
