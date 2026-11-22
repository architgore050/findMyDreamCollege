"""Verdict-stream tests: mock router, harness scoring, audit trail (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fmd.eval.verdict import (
    MockVerdictRouter,
    ProfileFixture,
    VerdictError,
    VerdictObject,
    audit_verdict,
    load_profile_fixtures,
    product_top_n,
    run_verdict_fast,
    top3_overlap_with_reference,
    verdict_aggregate,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eval_verdict" / "profiles"
EXPECTED_VERDICTS = {
    "canary-01-correct": "correct",
    "canary-02-partially-correct": "partially_correct",
    "canary-03-incorrect": "incorrect",
    "canary-04-unknown": "unknown",
    "canary-05-correct-thin-negative": "correct",
}


def test_five_canary_profiles_run_on_mock_router() -> None:
    profiles = load_profile_fixtures(FIXTURES)
    assert {p.profile_id for p in profiles} == set(EXPECTED_VERDICTS)
    cases = run_verdict_fast(profiles)
    by_id = {case["profile_id"]: case for case in cases}
    for profile_id, expected in EXPECTED_VERDICTS.items():
        assert by_id[profile_id]["verdict"] == expected
    # top-3 overlap: 3 (agreeing), 1 (partial), 0, 0 (unknown unscored), 2 (thin negative dropped)
    assert [by_id[p]["top3_overlap"] for p in sorted(by_id)] == [3, 1, 0, 0, 2]
    assert all(case["audit"]["failures"] == [] for case in cases)


def test_harness_aggregate_scores_are_honest() -> None:
    cases = run_verdict_fast(load_profile_fixtures(FIXTURES))
    aggregate = verdict_aggregate(cases)
    assert aggregate["n_profiles"] == 5
    assert aggregate["agreement_rate"] == 0.4  # 2 correct of 5
    assert aggregate["unknown_coverage"] == 0.2
    assert aggregate["audit_failures"] == 0
    assert aggregate["top3_overlap_mean"] == 1.2


def test_audit_trail_every_pointer_carries_url_and_sha256() -> None:
    profiles = load_profile_fixtures(FIXTURES)
    for profile in profiles:
        verdict = MockVerdictRouter().respond(profile)
        audit = audit_verdict(profile, verdict)
        for pointer in verdict.evidence:
            assert pointer.url.startswith("https://")
            assert len(pointer.sha256) == 64
            assert pointer.quote
            assert pointer.fetched_at
        assert audit["n_external"] >= 1  # circularity guard satisfied
        # fixture-scoped sources use the RFC-2606 .invalid host on purpose
        assert any(pointer.url.endswith(".invalid/...") or ".invalid" in pointer.url
                   for pointer in verdict.evidence if pointer.external)


def test_real_nirf_pointer_sha256_matches_committed_truth_page() -> None:
    profiles = load_profile_fixtures(FIXTURES)
    canary = next(p for p in profiles if p.profile_id == "canary-01-correct")
    verdict = MockVerdictRouter().respond(canary)
    nirf_pointer = next(
        pointer for pointer in verdict.evidence if pointer.source_kind == "ranking"
    )
    repo_root = Path(__file__).resolve().parents[3]
    raw = repo_root / "data" / "truth" / "nirf" / "raw" / "2025_MedicalRanking.html"
    import hashlib

    assert hashlib.sha256(raw.read_bytes()).hexdigest() == nirf_pointer.sha256


def test_circularity_guard_fails_verdict_without_external_source() -> None:
    profiles = load_profile_fixtures(FIXTURES)
    profile = profiles[0]
    scripted = json.loads(json.dumps(profile.scripted_verdict))
    for pointer in scripted["evidence"]:
        pointer["external"] = False
    verdict = VerdictObject.model_validate(scripted)
    audit = audit_verdict(profile, verdict)
    assert any("circularity guard" in failure for failure in audit["failures"])


def test_external_pointer_inside_ingested_evidence_is_flagged() -> None:
    profiles = load_profile_fixtures(FIXTURES)
    profile = profiles[0]
    scripted = json.loads(json.dumps(profile.scripted_verdict))
    external = next(p for p in scripted["evidence"] if p["external"])
    external["url"] = profile.ingested_evidence_urls[0]
    verdict = VerdictObject.model_validate(scripted)
    audit = audit_verdict(profile, verdict)
    assert any("ingested evidence" in failure for failure in audit["failures"])


def test_unknown_verdict_requires_named_unknown_aspects() -> None:
    with pytest.raises(ValidationError, match="unknown_aspects"):
        VerdictObject.model_validate({
            "profile_id": "p",
            "verdict": "unknown",
            "rationale": "insufficient evidence",
            "as_of": "2026-11-22",
        })


def test_non_synthetic_profile_fixture_is_rejected() -> None:
    payload = json.loads((FIXTURES / "canary-01-correct.json").read_text(encoding="utf-8"))
    payload["synthetic"] = False
    with pytest.raises((ValidationError, VerdictError), match="synthetic: true"):
        ProfileFixture.model_validate(payload)


def test_product_top_n_reads_the_frozen_shortlist() -> None:
    profiles = load_profile_fixtures(FIXTURES)
    canary = next(p for p in profiles if p.profile_id == "canary-01-correct")
    assert product_top_n(canary, 3)[0] == "All India Institute of Medical Sciences, Delhi"
    assert top3_overlap_with_reference(canary, MockVerdictRouter().respond(canary)) == 3
