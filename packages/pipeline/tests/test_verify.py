"""Second-pass verifier tests (offline, deterministic, no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fmd.extract.pipeline import build_router
from fmd.verify.casefile import evaluate_fixture_dir, load_case, run_case
from fmd.verify.metrics import verification_metrics
from fmd.verify.policy import VerifyPolicy, cost_per_1k_claims
from fmd.verify.schemas import VerdictBundle
from fmd.verify.types import ClaimView
from fmd.verify.verifier import Verifier, bounded_window, bundle_claims

TESTS = Path(__file__).parent
CASE_DIR = TESTS / "fixtures" / "verify" / "fabrication_v1"
KEY_RE = re.compile(r'"key":\s*"([^"]+)"')


class ScriptedRouter:
    """Deterministic provider stub that records the exact prompts it saw."""

    def __init__(self, verdict: Any) -> None:
        self.verdict = verdict
        self.calls: list[dict[str, Any]] = []

    def call(self, purpose, schema, system, user, input_ref):
        keys = KEY_RE.findall(user)
        self.calls.append({"system": system, "user": user, "keys": keys})
        verdicts = self.verdict(keys, bool(re.search(r"PREVIOUS VERDICT", user)), len(self.calls))
        payload = {
            "judgements": [
                {"key": key, "verdict": verdicts[key], "reason": "scripted"} for key in keys
            ]
        }
        return VerdictBundle.model_validate(payload)


def _policy(**overrides: Any) -> VerifyPolicy:
    defaults: dict[str, Any] = {
        "window_chars": 1200,
        "bundle_fields": 8,
        "max_claims_per_run": 500,
        "max_usd_per_run": 0.0,
    }
    defaults.update(overrides)
    return VerifyPolicy(**defaults)


def _claims(count: int, *, text: str = "alpha beta gamma delta epsilon zeta") -> list[ClaimView]:
    return [
        ClaimView(
            key=f"k{index + 1}",
            predicate=f"field.{index + 1}",
            quote="alpha",
            char_start=0,
            char_end=5,
            claim_id=index + 1,
        )
        for index in range(count)
    ]


class RecordingClaims:
    def __init__(self) -> None:
        self.writes: list[tuple[int, Any]] = []

    def record_verification(self, claim_id: int, verification: Any) -> None:
        self.writes.append((claim_id, verification))


# --------------------------------------------------------------------------- window/bundling


def test_bounded_window_contains_span_and_respects_size() -> None:
    text = "start. " + ("word " * 400) + "END"
    start = text.index("END")
    lo, hi, window = bounded_window(text, start, start + 3, chars=120)
    assert lo <= start and hi >= start + 3
    assert text[lo:hi] == window
    assert "END" in window
    assert len(window) <= 120 + 20


def test_bounded_window_returns_whole_short_page() -> None:
    text = "a short page"
    assert bounded_window(text, 2, 7, chars=1000) == (0, len(text), text)


def test_bundling_only_shares_identical_windows() -> None:
    text = "A" * 50 + " target " + "B" * 50
    start = text.index("target")
    close = ClaimView("a", "p.a", "target", start, start + 6)
    far = ClaimView("b", "p.b", "target", start + 20, start + 26)
    bundles = bundle_claims([close, far], text, chars=30, max_fields=8)
    # Different windows never share a call.
    assert [len(bundle.views) for bundle in bundles] == [1, 1]
    same = bundle_claims([close, close], text, chars=30, max_fields=8)
    assert [len(bundle.views) for bundle in same] == [2]


# --------------------------------------------------------------------------- prompt contract


def test_verdict_schema_has_no_value_field() -> None:
    assert "value" not in VerdictBundle.model_fields
    judgement_fields = VerdictBundle.model_fields["judgements"].annotation
    assert judgement_fields is not None
    from fmd.verify.schemas import FieldJudgement

    assert "value" not in FieldJudgement.model_fields
    assert set(FieldJudgement.model_fields) == {"key", "verdict", "corrected_quote", "reason"}


def test_prompt_never_includes_the_claimed_value() -> None:
    router = ScriptedRouter(lambda keys, repair, _call: dict.fromkeys(keys, "supported"))
    verifier = Verifier(router, policy=_policy())
    claim = ClaimView("k1", "salary.median", "Median salary: INR 420000", 0, 28, value=420000)
    verifier.verify_claims([claim], text="Median salary: INR 420000 per annum")
    assert router.calls
    for call in router.calls:
        assert "420000 per annum" in call["user"]  # the quote/window may contain it
        assert '"value"' not in call["user"]


# --------------------------------------------------------------------------- synthetic fixture


def test_synthetic_fabrication_fixture_rejects_all_seven(registry) -> None:
    case = load_case(CASE_DIR)
    assert sum(1 for claim in case.claims if claim.expected != "correct") == 7
    router = build_router(registry.models, mock=True, fixtures_dir=str(case.fixtures_dir))
    verifier = Verifier(router)
    run, metrics = run_case(case, verifier)

    payload = metrics.as_dict()
    assert payload["fabrication_caught"] == 7
    assert payload["fabricated_total"] == 7
    assert payload["false_accepts"] == 0
    assert payload["correct_rejected"] == 0
    assert payload["supported"] == 4
    assert payload["rejected"] == 7
    assert payload["verified_rate"] == 1.0

    decisions = {decision.key: decision for decision in run.decisions}
    assert decisions["c01"].status == "supported"
    assert decisions["c11"].status == "rejected"
    assert decisions["c11"].repair_attempted is True
    assert decisions["c11"].verdict == "wrong_field"
    assert run.calls == 3  # two bundles + one repair
    assert run.repairs == 1
    assert payload["correct_rejected"] == 0


def test_eval_fixture_dir_reports_the_same_numbers(registry) -> None:
    payload = evaluate_fixture_dir(CASE_DIR, registry)
    assert payload["fabrication_caught"] == 7
    assert payload["correct_rejected"] == 0
    assert payload["verified_rate"] == 1.0


# --------------------------------------------------------------------------- repair handling


def test_repair_is_single_field_and_only_for_flagged_claims() -> None:
    def verdict(keys: list[str], repair: bool, call: int) -> dict[str, str]:
        if repair:
            return {"k2": "supported"}
        return {"k1": "supported", "k2": "contradicted", "k3": "supported"}

    router = ScriptedRouter(verdict)
    verifier = Verifier(router, policy=_policy(bundle_fields=8))
    text = "alpha beta gamma delta epsilon zeta"
    claims = _claims(3, text=text)
    run = verifier.verify_claims(claims, text=text)

    assert run.repairs == 1
    assert len(router.calls) == 2
    repair_call = router.calls[1]
    assert repair_call["keys"] == ["k2"]
    assert "PREVIOUS VERDICT: contradicted" in repair_call["user"]
    statuses = {decision.key: decision.status for decision in run.decisions}
    assert statuses == {"k1": "supported", "k2": "supported", "k3": "supported"}
    repaired = next(decision for decision in run.decisions if decision.key == "k2")
    assert repaired.repair_attempted is True
    assert repaired.reason == "supported_after_repair"


def test_repair_that_fails_rejects_the_claim_with_a_reason() -> None:
    def verdict(keys: list[str], repair: bool, call: int) -> dict[str, str]:
        return dict.fromkeys(keys, "wrong_field")

    router = ScriptedRouter(verdict)
    verifier = Verifier(router, policy=_policy())
    run = verifier.verify_claims(_claims(1), text="alpha beta gamma")
    decision = run.decisions[0]
    assert decision.status == "rejected"
    assert decision.verdict == "wrong_field"
    assert decision.reason == "verifier_wrong_field_after_repair"
    assert decision.repair_attempted is True
    assert run.repairs == 1


# --------------------------------------------------------------------------- budget discipline


def test_claim_cap_marks_the_remainder_unverified() -> None:
    router = ScriptedRouter(lambda keys, repair, _call: dict.fromkeys(keys, "supported"))
    claims = RecordingClaims()
    verifier = Verifier(
        router, claims=claims, policy=_policy(max_claims_per_run=2, bundle_fields=1)
    )
    run = verifier.verify_claims(_claims(5), text="alpha beta gamma delta epsilon zeta")

    assert run.budget_exhausted is True
    counts = run.counts()
    assert counts["supported"] == 2
    assert counts["unverified"] == 3
    assert [item[1].status for item in claims.writes].count("unverified") == 3
    assert all(item[1].notes["reason"] == "budget_exhausted" for item in claims.writes[2:])


def test_spend_cap_uses_configured_pricing() -> None:
    policy = _policy(
        max_usd_per_run=0.001,
        bundle_fields=1,
        input_tokens_per_call=1000,
        input_per_mtok=1.0,
    )
    assert policy.cost_per_call == 0.001
    router = ScriptedRouter(lambda keys, repair, _call: dict.fromkeys(keys, "supported"))
    verifier = Verifier(router, policy=policy)
    run = verifier.verify_claims(_claims(4), text="alpha beta gamma")
    assert run.budget_exhausted is True
    assert run.counts()["supported"] == 1
    assert run.counts()["unverified"] == 3


def test_cost_per_1k_claims_is_config_driven() -> None:
    policy = _policy(
        bundle_fields=8,
        input_tokens_per_call=900,
        output_tokens_per_call=120,
        input_per_mtok=0.10,
        output_per_mtok=0.40,
    )
    report = cost_per_1k_claims(policy)
    assert report["calls"] == 125
    assert report["input_tokens"] == 112_500
    assert report["output_tokens"] == 15_000
    expected = 125 * (900 / 1_000_000 * 0.10 + 120 / 1_000_000 * 0.40)
    assert report["usd"] == round(expected, 8)


# --------------------------------------------------------------------------- persistence


def test_verification_persists_only_its_own_columns() -> None:
    claims = RecordingClaims()
    router = ScriptedRouter(lambda keys, repair, _call: dict.fromkeys(keys, "supported"))
    verifier = Verifier(router, claims=claims, policy=_policy())
    verifier.verify_claims(_claims(2), text="alpha beta gamma")

    assert len(claims.writes) == 2
    for claim_id, verification in claims.writes:
        assert claim_id in {1, 2}
        assert verification.status == "supported"
        assert verification.verified_by is None  # scripted router has no profile
        assert verification.notes["prompt_version"] == "verify@v1"
        assert verification.notes["verdict"] == "supported"


def test_metrics_never_count_unverified_as_caught() -> None:
    class Decision:
        def __init__(self, key: str, status: str) -> None:
            self.key = key
            self.status = status
            self.verdict = None
            self.reason = None

    metrics = verification_metrics(
        [
            Decision("a", "supported"),
            Decision("b", "rejected"),
            Decision("c", "supported"),
            Decision("d", "unverified"),
        ],
        {"a": "fabricated", "b": "fabricated", "c": "correct", "d": "correct"},
    )
    assert metrics.fabrication_caught == 1
    assert metrics.false_accepts == 1
    assert metrics.correct_rejected == 0
    assert metrics.verified_rate == 3 / 4


def test_fixture_manifest_is_json_and_quotes_resolve() -> None:
    manifest = json.loads((CASE_DIR / "case.json").read_text(encoding="utf-8"))
    text = (CASE_DIR / manifest["text_file"]).read_text(encoding="utf-8")
    for claim in manifest["claims"]:
        assert claim["quote"] in text


# --------------------------------------------------------------------------- persistence (DB)


def test_claim_verification_roundtrip_never_mutates_content(registry, pg_database) -> None:
    from fmd.db.repositories import (
        ClaimCreate,
        ClaimsRepo,
        ClaimVerification,
        DocumentsRepo,
        DocumentUpsert,
        EntitiesRepo,
        EntityCreate,
        ExtractionCreate,
        ExtractionsRepo,
        SnapshotCreate,
        SnapshotsRepo,
    )

    entities = EntitiesRepo(pg_database)
    entity = entities.create(
        EntityCreate(kind="institution", canonical_name="Verification Fixture Institute")
    )
    document = DocumentsRepo(pg_database).upsert(
        DocumentUpsert(url="https://verify.example.ac.in/placement", domain="verify.example.ac.in")
    )
    snapshot = SnapshotsRepo(pg_database).create(
        SnapshotCreate(
            document_id=document.id,
            sha256="0" * 64,
            bytes_path="/tmp/does-not-need-to-exist",
            content_type="text/plain",
        )
    )
    text = "Placement rate: 80%"
    extraction = ExtractionsRepo(pg_database).create(
        ExtractionCreate(
            snapshot_id=snapshot.id,
            tool="fmd.extract",
            text=text,
            model="fixture-model",
            prompt_version="placement_report@v1",
        )
    )
    claims = ClaimsRepo(pg_database)
    claim = claims.create(
        ClaimCreate(
            extraction_id=extraction.id,
            entity_id=entity.id,
            predicate="placement.rate",
            value=0.8,
            unit="ratio",
            quote="Placement rate: 80%",
            char_start=0,
            char_end=len(text),
        )
    )
    assert claim.verification_status == "unverified"
    assert claim.id in {row.id for row in claims.list_unverified(limit=10_000)}

    updated = claims.record_verification(
        claim.id,
        ClaimVerification(
            status="rejected",
            verified_by="recorded-verifier",
            notes={"verdict": "absent", "prompt_version": "verify@v1"},
        ),
    )

    assert updated.verification_status == "rejected"
    assert updated.verified_by == "recorded-verifier"
    assert updated.verified_at is not None
    assert updated.verification_notes["verdict"] == "absent"
    # Content columns are byte-identical.
    assert (updated.value, updated.quote, updated.char_start, updated.char_end) == (
        claim.value,
        claim.quote,
        claim.char_start,
        claim.char_end,
    )
    assert updated.confidence == claim.confidence
    assert claim.id not in {row.id for row in claims.list_unverified(limit=10_000)}
    assert claim.id in {row.id for row in claims.list_review_queue(limit=10_000)}
