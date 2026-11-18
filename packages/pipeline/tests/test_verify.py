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
from fmd.verify.prompts import PROMPTS_DIR, load_prompt, prompt_versions
from fmd.verify.schemas import VerdictBundle
from fmd.verify.types import ClaimView
from fmd.verify.verifier import (
    Verifier,
    bounded_window,
    bundle_claims,
    field_payload,
    seed_span,
)

TESTS = Path(__file__).parent
CASE_DIR = TESTS / "fixtures" / "verify" / "fabrication_v1"
ROW_CASE_DIR = TESTS / "fixtures" / "verify" / "row_context_v2"
KEY_RE = re.compile(r'"key":\s*"([^"]+)"')

STANLEY = "Govt. Stanley Medical College & Hospital, Chennai - 600 001"
CHENGALPATTU = "Chengalpattu Medical College & Hospital, Chengalpattu- 603 001"
MLT_LABEL = "3. B.Sc. Medical Laboratory Technology"
MLT_QUOTE = f"{MLT_LABEL}\n20"

#: The Stanley/Chengalpattu shape, small enough to assert on directly. The same
#: label and the same number print under two institutions' blocks; only the row
#: context tells them apart.
ROW_PAGE = (
    "SEAT MATRIX FOR THE A.Y. 2024-2025\n"
    "S.N Name of the Institution Course Intake\n"
    "1 Madras Medical College, Chennai - 600 003\n"
    "2. B.Sc Radiotherapy Technology\n20\n"
    f"2 {STANLEY}\n"
    f"{MLT_QUOTE}\n"
    f"3 {CHENGALPATTU}\n"
    "1. B.Sc. Medical Laboratory Technology\n20\n"
)


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


def test_claim_view_carries_row_context_from_the_qualifier() -> None:
    claim = {
        "id": 7,
        "predicate": "seats.total",
        "quote": MLT_QUOTE,
        "char_start": 10,
        "char_end": 10 + len(MLT_QUOTE),
        "value": 20,
        "qualifier": {
            "disambiguation": "context_row",
            "context_quote": STANLEY,
            "context_char_start": 0,
            "context_char_end": len(STANLEY),
            "row": {
                "row_label": MLT_LABEL,
                "column_header": "Intake",
                "value_cell": "20",
                "table_page": 1,
            },
            "row_char_start": 10,
            "row_char_end": 60,
        },
    }
    view = ClaimView.from_claim(claim)
    assert view.context == STANLEY
    assert view.row_label == MLT_LABEL
    assert view.column_header == "Intake"
    assert view.context_span == (0, len(STANLEY))
    assert view.row_span == (10, 60)
    assert view.expected is None  # the verifier fills it from the registry
    # The claimed value and the printed value cell never reach the payload.
    payload = field_payload(view)
    assert payload["key"] == "7" and payload["quote"] == MLT_QUOTE
    assert payload["context"] == STANLEY
    assert payload["row_label"] == MLT_LABEL and payload["column_header"] == "Intake"
    assert "value" not in payload and "value_cell" not in payload


def test_field_payload_omits_absent_row_context() -> None:
    view = ClaimView("k1", "placement.rate", "80%", 0, 3)
    payload = field_payload(view)
    assert set(payload) == {"key", "field", "quote", "expected"}


def test_expected_type_comes_from_the_predicate_registry(registry) -> None:
    verifier = Verifier(
        ScriptedRouter(lambda keys, _repair, _call: dict.fromkeys(keys, "supported")),
        policy=_policy(),
        predicates=registry.predicates,
    )
    assert verifier._expected_type("seats.total") == "count"
    assert verifier._expected_type("seats.intake") == "count"
    assert verifier._expected_type("placement.eligible_count") == "count"
    assert verifier._expected_type("fee.tuition") == "money+period"
    assert verifier._expected_type("salary.median") == "money+period"
    assert verifier._expected_type("placement.rate") == "percentage+denominator"
    assert verifier._expected_type("accreditation.naac_grade") == "grade"
    assert verifier._expected_type("seats.intake_year") == "text"
    assert verifier._expected_type("predicates.not.in.the.registry") is None


def test_description_comes_from_the_predicate_registry(registry) -> None:
    """verify@v3: the verifier receives the registry's definition, scope included."""
    verifier = Verifier(
        ScriptedRouter(lambda keys, _repair, _call: dict.fromkeys(keys, "supported")),
        policy=_policy(),
        predicates=registry.predicates,
    )
    intake = verifier._description("seats.intake")
    assert intake and "THIS programme" in intake and "THIS institution" in intake
    total = verifier._description("seats.total")
    assert total and "aggregate" in total.lower()
    # Both seat predicates say what the other is not, so the model can tell a
    # row value from an aggregate.
    assert "aggregate" in intake.lower() and "never" in total.lower()
    # An alias resolves to its canonical row's description; an unknown
    # predicate has none.
    assert verifier._description("seats.sanctioned") == intake
    assert verifier._description("predicates.not.in.the.registry") is None


def test_field_payload_carries_the_description_and_never_the_value(registry) -> None:
    verifier = Verifier(
        ScriptedRouter(lambda keys, _repair, _call: dict.fromkeys(keys, "supported")),
        policy=_policy(),
        predicates=registry.predicates,
    )
    claim = ClaimView("k1", "seats.intake", "3. B.Sc. Medical Laboratory Technology\n20", 0, 40)
    filled = verifier._with_expected(claim)
    payload = field_payload(filled)
    assert payload["expected"] == "count"
    assert "THIS programme" in payload["description"]
    assert "value" not in payload


def test_verify_claims_fill_expected_but_never_the_value(registry) -> None:
    router = ScriptedRouter(lambda keys, _repair, _call: dict.fromkeys(keys, "supported"))
    verifier = Verifier(router, policy=_policy(), predicates=registry.predicates)
    claim = {
        "id": 1,
        "predicate": "fee.tuition",
        "quote": "Tuition fee 85000",
        "char_start": 0,
        "char_end": 17,
        "value": 85000,
        "qualifier": {"context_quote": STANLEY, "row": {"row_label": MLT_LABEL}},
    }
    verifier.verify_claims([claim], text="Tuition fee 85000")
    assert router.calls[0]["keys"] == ["claim-1"]
    user = router.calls[0]["user"]
    assert '"expected": "money+period"' in user
    assert '"description"' in user  # the registry's definition travels (verify@v3)
    assert '"context"' in user and STANLEY in user
    assert '"row_label"' in user
    assert '"column_header"' not in user  # qualifier did not resolve one
    assert '"value"' not in user and '"value_cell"' not in user


VERIFY_V2_CONTRACT_MARKERS = (
    "a bare value",
    "a label + value pair inside one table row",
    "a value plus unit and period",
    "does the value this quote wraps belong to this row and this column",
    "wrong_field",
    "printed currency and a printed period",
    "percentage+denominator",
    "never output a value",
    "corrected_quote",
)

REPAIR_V2_CONTRACT_MARKERS = (
    "a bare value",
    "a label + value pair inside one table row",
    "does the value this quote wraps belong to this row and this column",
    "genuinely wrong association",
    "printed currency and a printed period",
    "percentage+denominator",
    "never emit a value",
)


VERIFY_V3_CONTRACT_MARKERS = (
    "description",
    "the registry's one-sentence definition",
    "per-row vs aggregate",
    "judge whether the value the quote wraps belongs to the field as described",
    "a row-scoped quote is correct",
    "a single programme row's value presented under an aggregate field",
    "does the value this quote wraps belong to this row and this column",
    "wrong_field",
    "never output a value",
)

REPAIR_V3_CONTRACT_MARKERS = (
    "description",
    "one-sentence `description` of the field",
    "its meaning and scope",
    "a row-scoped quote is correct",
    "a single programme row's value under an aggregate field stays",
    "does the value this quote wraps belong to this row and this column",
    "genuinely wrong association",
    "never emit a value",
)


def test_v3_prompts_state_the_description_and_scope_contract() -> None:
    for name, markers in (
        ("verify", VERIFY_V3_CONTRACT_MARKERS),
        ("verify_repair", REPAIR_V3_CONTRACT_MARKERS),
    ):
        prompt = load_prompt(name, "v3")
        flattened = " ".join(prompt.text.split()).lower()
        for marker in markers:
            assert " ".join(marker.split()).lower() in flattened, (
                f"{name}@v3 is missing the contract statement {marker!r}"
            )


def test_v2_prompts_state_the_quote_shape_and_row_contract() -> None:
    for name, markers in (
        ("verify", VERIFY_V2_CONTRACT_MARKERS),
        ("verify_repair", REPAIR_V2_CONTRACT_MARKERS),
    ):
        prompt = load_prompt(name, "v2")
        flattened = " ".join(prompt.text.split()).lower()
        for marker in markers:
            assert " ".join(marker.split()).lower() in flattened, (
                f"{name}@v2 is missing the contract statement {marker!r}"
            )


def test_verify_prompt_family_is_complete_and_v3_is_latest(registry) -> None:
    """The verify family keeps its history: contiguous v1..vN, every file real."""
    for name in ("verify", "verify_repair"):
        versions = prompt_versions(name)
        assert versions, f"{name} must ship at least one prompt file"
        assert versions == [f"v{index}" for index in range(1, len(versions) + 1)], (
            f"{name} prompt history must be contiguous v1..vN with every earlier "
            f"version kept; got {versions}"
        )
        assert {"v1", "v2", "v3"}.issubset(set(versions))
        for version in versions:
            prompt = load_prompt(name, version)
            assert prompt.path == PROMPTS_DIR / name / f"{version}.md"
            assert prompt.path.is_file()
            assert prompt.text.strip(), f"{name}/{version}.md must not be empty"
            if version != "v1":  # v1 predates the version-label convention
                assert prompt.version in prompt.text, (
                    f"{name}/{version}.md must carry its version label in the body"
                )
        assert load_prompt(name).version == "v3"
    assert load_prompt("verify", "v1").ref == "verify@v1"
    assert load_prompt("verify", "v2").ref == "verify@v2"
    assert load_prompt("verify_repair", "v1").ref == "verify_repair@v1"
    assert load_prompt("verify_repair", "v2").ref == "verify_repair@v2"
    assert VerifyPolicy.load(registry.models).prompt_version == "v3"


# --------------------------------------------------------------------------- synthetic fixture


def test_synthetic_fabrication_fixture_rejects_all_seven(registry) -> None:
    case = load_case(CASE_DIR)
    assert sum(1 for claim in case.claims if claim.expected != "correct") == 7
    router = build_router(registry.models, mock=True, fixtures_dir=str(case.fixtures_dir))
    verifier = Verifier(router, predicates=registry.predicates)
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


# --------------------------------------------------------------------------- row-context case


def test_row_context_fixture_supports_the_stanley_claim_and_rejects_the_counterfactual(
    registry,
) -> None:
    """The wave-2 anti-regression, offline: same value 20, two rows, opposite fates.

    s01 is the real Stanley claim (label+value quote, context_row to the Stanley
    block): verify@v2 must accept what v1 discarded as wrong_field. s02 is the
    same quote asserted against the Chengalpattu block, whose MLT row is "1."
    not "3.": the wrong-row rule must survive the v2 relaxation, repair included.
    """
    case = load_case(ROW_CASE_DIR)
    assert len(case.claims) == 4
    router = build_router(registry.models, mock=True, fixtures_dir=str(case.fixtures_dir))
    verifier = Verifier(router, predicates=registry.predicates)
    run, metrics = run_case(case, verifier)

    decisions = {decision.key: decision for decision in run.decisions}
    assert decisions["s01"].status == "supported"
    assert decisions["s01"].row_context is True
    assert decisions["s01"].expected_type == "count"
    assert decisions["s01"].prompt_version == "verify@v3"
    # s01 is now seats.intake (W2 seat-semantics): the payload the model judged
    # described a per-programme field, so the row quote was correct.
    payload_by_key = {claim.view.key: claim.view.predicate for claim in case.claims}
    assert payload_by_key["s01"] == "seats.intake"

    assert decisions["s02"].status == "rejected"
    assert decisions["s02"].verdict == "wrong_field"
    assert decisions["s02"].repair_attempted is True
    assert decisions["s02"].reason == "verifier_wrong_field_after_repair"

    assert decisions["s03"].status == "rejected"
    assert decisions["s03"].verdict == "absent"
    assert decisions["s03"].repair_attempted is False  # absent rejects immediately
    assert decisions["s03"].expected_type == "money+period"

    assert decisions["s04"].status == "supported"  # bare value, no row context
    assert decisions["s04"].row_context is False

    payload = metrics.as_dict()
    assert payload["fabrication_caught"] == 2
    assert payload["fabricated_total"] == 2
    assert payload["false_accepts"] == 0
    assert payload["correct_rejected"] == 0
    assert payload["supported"] == 2
    assert payload["rejected"] == 2
    assert payload["verified_rate"] == 1.0
    assert run.calls == 2  # one bundle (all four share the small page) + one repair
    assert run.repairs == 1


def test_row_context_fixture_eval_reports_the_same_numbers(registry) -> None:
    payload = evaluate_fixture_dir(ROW_CASE_DIR, registry)
    assert payload["fabrication_caught"] == 2
    assert payload["false_accepts"] == 0
    assert payload["correct_rejected"] == 0
    assert payload["verified_rate"] == 1.0
    assert payload["calls"] == 2 and payload["repairs"] == 1


# --------------------------------------------------------------------------- seat semantics case

SEAT_CASE_DIR = TESTS / "fixtures" / "verify" / "seat_semantics_v3"


def test_seat_semantics_fixture_supports_the_row_intake_and_rejects_the_row_as_total(
    registry,
) -> None:
    """The seat split, offline: same value 20, two predicates, opposite fates.

    t01 is the Stanley claim as `seats.intake`: the registry description says
    the field is the sanctioned intake for THIS programme at THIS institution,
    so the row-scoped quote is correct and must be supported. t02 is the SAME
    quote asserted as `seats.total`: the description says that field is only
    for an aggregate the document itself states, so a row value stays
    wrong_field, repair included. t03 is the genuine aggregate (the Total row
    the Stanley block prints) and verifies as `seats.total`. t04 is a second
    row-scoped intake on the Chengalpattu block. t05 is the money guard: a
    bare amount with no printed period is `absent`, unchanged by the split.
    """
    case = load_case(SEAT_CASE_DIR)
    assert len(case.claims) == 5
    router = build_router(registry.models, mock=True, fixtures_dir=str(case.fixtures_dir))
    verifier = Verifier(router, predicates=registry.predicates)
    run, metrics = run_case(case, verifier)

    decisions = {decision.key: decision for decision in run.decisions}
    # Row-scoped seats.intake is supported — the Stanley fix this wave ships.
    assert decisions["t01"].status == "supported"
    assert decisions["t01"].row_context is True
    assert decisions["t01"].expected_type == "count"
    assert decisions["t01"].prompt_version == "verify@v3"

    # The same value labelled seats.total is rejected: wrong scope, not a
    # missing number.
    assert decisions["t02"].status == "rejected"
    assert decisions["t02"].verdict == "wrong_field"
    assert decisions["t02"].repair_attempted is True
    assert decisions["t02"].reason == "verifier_wrong_field_after_repair"

    # A genuine document-stated aggregate row still verifies as total.
    assert decisions["t03"].status == "supported"
    assert decisions["t03"].row_context is True
    assert decisions["t03"].expected_type == "count"

    # A second row-scoped intake claim is supported on its own row.
    assert decisions["t04"].status == "supported"

    # Money-without-period is still rejected immediately.
    assert decisions["t05"].status == "rejected"
    assert decisions["t05"].verdict == "absent"
    assert decisions["t05"].repair_attempted is False
    assert decisions["t05"].expected_type == "money+period"

    payload = metrics.as_dict()
    assert payload["fabrication_caught"] == 2
    assert payload["fabricated_total"] == 2
    assert payload["false_accepts"] == 0
    assert payload["correct_rejected"] == 0
    assert payload["supported"] == 3
    assert payload["rejected"] == 2
    assert payload["verified_rate"] == 1.0
    assert run.calls == 2  # one bundle (all five share the small page) + one repair
    assert run.repairs == 1


def test_seat_semantics_fixture_eval_reports_the_same_numbers(registry) -> None:
    payload = evaluate_fixture_dir(SEAT_CASE_DIR, registry)
    assert payload["fabrication_caught"] == 2
    assert payload["false_accepts"] == 0
    assert payload["correct_rejected"] == 0
    assert payload["verified_rate"] == 1.0
    assert payload["calls"] == 2 and payload["repairs"] == 1


def test_seat_semantics_fixture_quotes_and_contexts_resolve() -> None:
    manifest = json.loads((SEAT_CASE_DIR / "case.json").read_text(encoding="utf-8"))
    text = (SEAT_CASE_DIR / manifest["text_file"]).read_text(encoding="utf-8")
    for claim in manifest["claims"]:
        assert claim["quote"] in text
        if "context" in claim:
            assert claim["context"] in text
    # The discrimination pair: one quote, two different predicates.
    assert manifest["claims"][0]["quote"] == manifest["claims"][1]["quote"]
    assert manifest["claims"][0]["predicate"] != manifest["claims"][1]["predicate"]
    # The genuine aggregate quotes the document's own Total row.
    assert manifest["claims"][2]["quote"].startswith("Total")


def test_row_context_fixture_quotes_and_contexts_resolve() -> None:
    manifest = json.loads((ROW_CASE_DIR / "case.json").read_text(encoding="utf-8"))
    text = (ROW_CASE_DIR / manifest["text_file"]).read_text(encoding="utf-8")
    for claim in manifest["claims"]:
        assert claim["quote"] in text
        if "context" in claim:
            assert claim["context"] in text
    # The counterfactual pair: one quote, two different rows.
    assert manifest["claims"][0]["quote"] == manifest["claims"][1]["quote"]
    assert manifest["claims"][0]["context"] != manifest["claims"][1]["context"]


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


# --------------------------------------------------------------------------- row context (scripted)


class PayloadRouter:
    """Answers from the exact field payload it saw, recording every call."""

    def __init__(self, decide: Any) -> None:
        self.decide = decide
        self.calls: list[dict[str, Any]] = []

    def call(self, purpose, schema, system, user, input_ref):
        payload = json.loads(user[user.index("{") : user.rindex("}") + 1])
        fields = payload.get("fields", [payload])
        self.calls.append({"user": user, "repair": "PREVIOUS VERDICT" in user, "fields": fields})
        verdicts = {field["key"]: self.decide(field, self.calls[-1]["repair"]) for field in fields}
        return VerdictBundle.model_validate(
            {
                "judgements": [
                    {"key": key, "verdict": verdict, "reason": "scripted"}
                    for key, verdict in verdicts.items()
                ]
            }
        )


def _row_claim(key: str, context: str) -> ClaimView:
    start = ROW_PAGE.index(MLT_QUOTE)
    return ClaimView(
        key=key,
        predicate="seats.total",
        quote=MLT_QUOTE,
        char_start=start,
        char_end=start + len(MLT_QUOTE),
        value=20,
        context=context,
        row_label=MLT_LABEL,
        column_header="Intake",
        context_span=(ROW_PAGE.index(context), ROW_PAGE.index(context) + len(context)),
    )


def test_counterfactual_same_value_two_rows_one_supported_one_rejected(registry) -> None:
    """Same quote span, two different row contexts: the model must see both."""
    seen_contexts: list[str] = []

    def decide(field: dict[str, Any], _repair: bool) -> str:
        seen_contexts.append(field.get("context") or "")
        if field.get("context") == STANLEY:
            return "supported"
        return "wrong_field"

    router = PayloadRouter(decide)
    verifier = Verifier(router, policy=_policy(), predicates=registry.predicates)
    run = verifier.verify_claims(
        [_row_claim("stanley", STANLEY), _row_claim("chengalpattu", CHENGALPATTU)],
        text=ROW_PAGE,
    )
    decisions = {decision.key: decision for decision in run.decisions}
    assert decisions["stanley"].status == "supported"
    assert decisions["chengalpattu"].status == "rejected"
    assert decisions["chengalpattu"].verdict == "wrong_field"
    assert decisions["chengalpattu"].reason == "verifier_wrong_field_after_repair"
    # Both contexts reached the model; the claimed value never did.
    assert STANLEY in seen_contexts and CHENGALPATTU in seen_contexts
    for call in router.calls:
        assert '"value"' not in call["user"]
        for field in call["fields"]:
            assert field.get("column_header") == "Intake"
            assert field.get("row_label") == MLT_LABEL


def test_row_context_widens_the_window_to_cover_the_row() -> None:
    # The context anchor sits far above the quote but within the window budget.
    page = STANLEY + "\n" + ("padding text " * 20) + "\n" + MLT_QUOTE
    start = page.index(MLT_QUOTE)
    view = ClaimView(
        key="k1",
        predicate="seats.total",
        quote=MLT_QUOTE,
        char_start=start,
        char_end=start + len(MLT_QUOTE),
        context=STANLEY,
        context_span=(0, len(STANLEY)),
    )
    assert seed_span(view, chars=400) == (0, start + len(MLT_QUOTE))
    router = PayloadRouter(lambda _field, _repair: "supported")
    verifier = Verifier(router, policy=_policy(window_chars=400))
    verifier.verify_claims([view], text=page)
    # The window the model saw contains both the quote and the row context.
    window = router.calls[0]["user"].split("<<<\n")[1].split("\n>>>")[0]
    assert MLT_QUOTE in window and STANLEY in window


def test_row_context_never_blows_the_window_budget() -> None:
    """A block wider than window_chars falls back to the quote-centred window."""
    page = STANLEY + "\n" + ("padding " * 400) + MLT_QUOTE
    start = page.index(MLT_QUOTE)
    view = ClaimView(
        key="k1",
        predicate="seats.total",
        quote=MLT_QUOTE,
        char_start=start,
        char_end=start + len(MLT_QUOTE),
        context=STANLEY,
        context_span=(0, len(STANLEY)),
    )
    assert seed_span(view, chars=200) == (start, start + len(MLT_QUOTE))
    lo, hi, window = bounded_window(page, *seed_span(view, chars=200), chars=200)
    assert lo <= start and hi >= start + len(MLT_QUOTE)
    assert len(window) <= 200 + 20


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
        assert verification.notes["prompt_version"] == "verify@v3"
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
