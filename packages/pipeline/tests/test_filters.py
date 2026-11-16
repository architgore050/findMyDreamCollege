from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fmd.filters import IMPLEMENTATIONS, FilterContext, evaluate, evaluate_all

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def context(registry) -> FilterContext:
    return FilterContext.from_registry(registry, now=NOW)


def claim(predicate: str, value: object, **overrides) -> dict:
    payload = {
        "id": overrides.pop("id", None),
        "predicate": predicate,
        "value": value,
        "unit": None,
        "qualifier": {},
        "method": "extracted",
        "confidence": 1.0,
        "observed_at": NOW,
        "quote": "quote",
        "char_start": 0,
        "char_end": 1,
        "extraction_id": 1,
        "entity_id": 101,
    }
    payload.update(overrides)
    return payload


def programme(**overrides) -> dict:
    base = {
        "credential_level": "degree",
        "recognition_status": "recognised",
        "medium": "English",
        "claims": [],
    }
    base.update(overrides)
    return base


PROFILE = {
    "credential_levels": ["degree"],
    "marks": {"class12_aggregate_pct": 78, "pcb_pct": 76},
    "budget": {"currency": "INR", "total_ceiling": 900000},
    "target_intake_year": 2027,
    "travel_for_exams": "no",
    "medium_of_instruction": ["English"],
    "english_test": {"type": "IELTS", "score": 6.5, "status": "taken"},
}


def test_every_configured_filter_has_an_implementation(registry) -> None:
    assert list(IMPLEMENTATIONS) == registry.weights.filters


def test_all_eight_filters_pass_on_a_complete_record(context: FilterContext) -> None:
    record = programme(
        claims=[
            claim("exam_required", "mht_cet_pcb", qualifier={"exam_id": "mht_cet_pcb"}, id=1),
            claim(
                "exam_registration_closes",
                "2026-12-15T23:59:00",
                qualifier={"exam_id": "mht_cet_pcb"},
                observed_at=NOW - timedelta(days=1),
                id=2,
            ),
            claim("intake_cycles", [2026, 2027], id=3),
            claim("eligibility_requires_subjects", ["PCB"], id=4),
            claim("eligibility_min_aggregate_pct", 55, unit="percent", id=5),
            claim("tcoa_lo", 500000, unit="INR", id=6),
        ]
    )
    summary = evaluate_all(record, {**PROFILE, "travel_for_exams": "yes"}, context=context)
    assert summary.counts == {"pass": 8, "fail": 0, "unknown": 0}
    assert summary.excluded == ()
    assert summary.unverified == ()
    payload = summary.as_dict()
    assert payload["decisions"]["budget_ceiling"]["detail"]["tcoa_lo"] == 500000


# --------------------------------------------------------------------- each filter


def test_credential_level_pass_fail_unknown(context: FilterContext) -> None:
    passed = evaluate("credential_level", programme(), PROFILE, context=context)
    assert passed.result == "pass"
    failed = evaluate(
        "credential_level", programme(credential_level="diploma"), PROFILE, context=context
    )
    assert failed.result == "fail"
    assert "diploma" in failed.reason
    no_profile = evaluate(
        "credential_level", programme(), {**PROFILE, "credential_levels": []}, context=context
    )
    assert no_profile.result == "unknown"
    missing = evaluate(
        "credential_level",
        programme(credential_level=None),
        PROFILE,
        context=context,
    )
    assert missing.result == "unknown"


def test_recognition_status_pass_fail_unknown(context: FilterContext) -> None:
    assert evaluate("recognition_status", programme(), PROFILE, context=context).result == "pass"
    failed = evaluate(
        "recognition_status",
        programme(recognition_status="unrecognised"),
        PROFILE,
        context=context,
    )
    assert failed.result == "fail"
    conditional = evaluate(
        "recognition_status",
        programme(recognition_status="conditional"),
        PROFILE,
        context=context,
    )
    assert conditional.result == "unknown"

    incomplete = programme(
        recognition_status=None,
        claims=[
            claim(
                "recognition_status",
                "recognised",
                qualifier={"chain_resolved": False},
                id=11,
            )
        ],
    )
    decision = evaluate("recognition_status", incomplete, PROFILE, context=context)
    assert decision.result == "unknown"
    assert decision.claim_id == 11


def test_eligibility_pass_fail_unknown(context: FilterContext) -> None:
    record = programme(
        claims=[
            claim("eligibility_min_aggregate_pct", 55, unit="percent", id=21),
            claim("eligibility_min_pcb_pct", 50, unit="percent", id=22),
        ]
    )
    assert evaluate("eligibility", record, PROFILE, context=context).result == "pass"

    too_low = {**PROFILE, "marks": {"class12_aggregate_pct": 50, "pcb_pct": 76}}
    failed = evaluate("eligibility", record, too_low, context=context)
    assert failed.result == "fail"
    assert failed.claim_id == 21

    no_marks = {k: v for k, v in PROFILE.items() if k != "marks"}
    unknown = evaluate("eligibility", record, no_marks, context=context)
    assert unknown.result == "unknown"

    no_thresholds = programme()
    assert evaluate("eligibility", no_thresholds, PROFILE, context=context).result == "unknown"


def test_language_pass_fail_unknown_and_medium_exemption(context: FilterContext) -> None:
    no_test = programme(claims=[claim("language_test_required", "none", id=31)])
    assert evaluate("language", no_test, PROFILE, context=context).result == "pass"

    exempt = programme(medium="English")
    assert evaluate("language", exempt, PROFILE, context=context).result == "pass"

    hindi = programme(medium="Hindi")
    assert evaluate("language", hindi, PROFILE, context=context).result == "unknown"

    required = programme(
        medium="Hindi",
        claims=[
            claim(
                "language_test_required",
                "required",
                qualifier={"test": "ielts"},
                id=32,
            ),
            claim("language_min_score", 7.0, id=33),
        ],
    )
    failed = evaluate("language", required, PROFILE, context=context)
    assert failed.result == "fail"
    assert failed.claim_id == 32

    no_score = {
        **PROFILE,
        "english_test": {"type": "IELTS", "score": None, "status": "not_taken"},
    }
    assert evaluate("language", required, no_score, context=context).result == "unknown"

    no_requirement = programme(medium="Hindi")
    assert evaluate("language", no_requirement, PROFILE, context=context).result == "unknown"


def test_exam_registration_open_pass_fail_and_stale(context: FilterContext) -> None:
    def record(observed_at: datetime, closes: str = "2026-12-15T23:59:00") -> dict:
        return programme(
            claims=[
                claim("exam_required", "mht_cet_pcb", qualifier={"exam_id": "mht_cet_pcb"}),
                claim(
                    "exam_registration_closes",
                    closes,
                    qualifier={"exam_id": "mht_cet_pcb"},
                    observed_at=observed_at,
                    id=41,
                ),
            ]
        )

    fresh = evaluate(
        "exam_registration_open", record(NOW - timedelta(days=1)), PROFILE, context=context
    )
    assert fresh.result == "pass"

    closed = evaluate(
        "exam_registration_open",
        record(NOW - timedelta(days=1), closes="2026-11-01T23:59:00"),
        PROFILE,
        context=context,
    )
    assert closed.result == "fail"

    stale = evaluate(
        "exam_registration_open", record(NOW - timedelta(days=10)), PROFILE, context=context
    )
    assert stale.result == "unknown"
    assert stale.stale is True
    assert "TTL" in stale.reason

    no_window = evaluate(
        "exam_registration_open",
        programme(
            claims=[claim("exam_required", "mht_cet_pcb", qualifier={"exam_id": "mht_cet_pcb"})]
        ),
        PROFILE,
        context=context,
    )
    assert no_window.result == "unknown"


def test_exam_travel_pass_fail_unknown(context: FilterContext) -> None:
    merit = programme(
        claims=[claim("exam_required", "merit_class12", qualifier={"exam_id": "merit_class12"})]
    )
    assert evaluate("exam_travel", merit, PROFILE, context=context).result == "pass"

    travel = programme(
        claims=[claim("exam_required", "mht_cet_pcb", qualifier={"exam_id": "mht_cet_pcb"})]
    )
    assert evaluate("exam_travel", travel, PROFILE, context=context).result == "fail"
    allows = {**PROFILE, "travel_for_exams": "yes"}
    assert evaluate("exam_travel", travel, allows, context=context).result == "pass"

    unknown = {k: v for k, v in PROFILE.items() if k != "travel_for_exams"}
    assert evaluate("exam_travel", travel, unknown, context=context).result == "unknown"
    assert evaluate("exam_travel", programme(), PROFILE, context=context).result == "unknown"


def test_intake_year_available_pass_fail_unknown(context: FilterContext) -> None:
    record = programme(claims=[claim("intake_cycles", [2026, 2027], id=51)])
    assert evaluate("intake_year_available", record, PROFILE, context=context).result == "pass"
    assert evaluate("intake_year_available", record, PROFILE, context=context).claim_id == 51
    later = {**PROFILE, "target_intake_year": 2030}
    assert evaluate("intake_year_available", record, later, context=context).result == "fail"
    no_target = {k: v for k, v in PROFILE.items() if k != "target_intake_year"}
    assert evaluate("intake_year_available", record, no_target, context=context).result == "unknown"
    assert (
        evaluate("intake_year_available", programme(), PROFILE, context=context).result == "unknown"
    )


def test_budget_ceiling_compares_against_tcoa_lo_only(context: FilterContext) -> None:
    record = programme(claims=[claim("tcoa_lo", 500000, unit="INR", id=61)])
    assert evaluate("budget_ceiling", record, PROFILE, context=context).result == "pass"

    above = programme(claims=[claim("tcoa_lo", 1000000, unit="INR", id=62)])
    failed = evaluate("budget_ceiling", above, PROFILE, context=context)
    assert failed.result == "fail"
    assert failed.claim_id == 62

    no_tcoa = programme()
    assert evaluate("budget_ceiling", no_tcoa, PROFILE, context=context).result == "unknown"

    no_ceiling = {k: v for k, v in PROFILE.items() if k != "budget"}
    assert evaluate("budget_ceiling", record, no_ceiling, context=context).result == "unknown"

    annual_only = {
        **PROFILE,
        "budget": {"currency": "INR", "annual_ceiling": 200000},
    }
    assert evaluate("budget_ceiling", record, annual_only, context=context).result == "unknown"

    foreign = {**PROFILE, "budget": {"currency": "USD", "total_ceiling": 20000}}
    assert evaluate("budget_ceiling", record, foreign, context=context).result == "unknown"
