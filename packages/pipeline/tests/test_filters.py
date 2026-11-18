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



# ------------------------------------------------- w3 exam windows (spec 17)


def window_claims(*windows: dict) -> list[dict]:
    """Exam requirement + registration-close claims built from a compact spec.

    Each window dict: value, observed_at, id, cycle_year, precision,
    superseded_by, supersedes. ``exam_id`` is fixed to mht_cet_pcb.
    """

    claims = [
        claim("exam_required", "mht_cet_pcb", qualifier={"exam_id": "mht_cet_pcb"}, id=40)
    ]
    for window in windows:
        qualifier = {"exam_id": "mht_cet_pcb"}
        if window.get("cycle_year") is not None:
            qualifier["cycle_year"] = window["cycle_year"]
        if window.get("precision") is not None:
            qualifier["precision"] = window["precision"]
        if window.get("superseded_by") is not None:
            qualifier["superseded_by"] = window["superseded_by"]
        if window.get("supersedes") is not None:
            qualifier["supersedes"] = window["supersedes"]
        claims.append(
            claim(
                "exam_registration_closes",
                window["value"],
                qualifier=qualifier,
                observed_at=window["observed_at"],
                id=window["id"],
            )
        )
    return claims


def test_closed_required_exam_excludes_and_open_one_does_not(context: FilterContext) -> None:
    closed = programme(
        claims=window_claims(
            {"value": "2026-11-01", "observed_at": NOW - timedelta(days=1), "id": 41}
        )
    )
    summary = evaluate_all(closed, PROFILE, context=context)
    assert "exam_registration_open" in summary.excluded
    decision = summary.decisions["exam_registration_open"]
    assert decision.result == "fail"
    # The shortlist exclusion cites the deciding claim with a machine code.
    assert decision.claim_id == 41
    assert decision.observed_at == NOW - timedelta(days=1)
    assert decision.detail["exclusion_reason"] == "exam_registration_closed"
    assert decision.detail["exams"][0]["exam_id"] == "mht_cet_pcb"

    open_record = programme(
        claims=window_claims(
            {"value": "2026-12-15", "observed_at": NOW - timedelta(days=1), "id": 42}
        )
    )
    open_summary = evaluate_all(open_record, PROFILE, context=context)
    assert "exam_registration_open" not in open_summary.excluded
    assert open_summary.decisions["exam_registration_open"].result == "pass"
    assert open_summary.decisions["exam_registration_open"].claim_id == 42


def test_superseded_notification_reopens_the_window(context: FilterContext) -> None:
    """Append-only supersession: the retired window never decides (spec 17 §5)."""

    record = programme(
        claims=window_claims(
            {
                "value": "2026-11-01",
                "observed_at": NOW - timedelta(days=30),
                "id": 41,
                "superseded_by": 42,
            },
            {"value": "2027-03-15", "observed_at": NOW - timedelta(days=1), "id": 42},
        )
    )
    reopened = evaluate("exam_registration_open", record, PROFILE, context=context)
    assert reopened.result == "pass"
    assert reopened.claim_id == 42


def test_supersedes_qualifier_also_retires_the_old_window(context: FilterContext) -> None:
    record = programme(
        claims=window_claims(
            {"value": "2026-11-01", "observed_at": NOW - timedelta(days=2), "id": 41},
            {
                "value": "2027-03-15",
                "observed_at": NOW - timedelta(days=1),
                "id": 42,
                "supersedes": 41,
            },
        )
    )
    reopened = evaluate("exam_registration_open", record, PROFILE, context=context)
    assert reopened.result == "pass"
    assert reopened.claim_id == 42


def test_superseded_window_cannot_keep_the_exam_open(context: FilterContext) -> None:
    """The old row claims a far-future close; its replacement says closed."""

    record = programme(
        claims=window_claims(
            {
                "value": "2027-06-30",
                "observed_at": NOW - timedelta(days=40),
                "id": 41,
                "superseded_by": 42,
            },
            {"value": "2026-11-01", "observed_at": NOW - timedelta(days=1), "id": 42},
        )
    )
    decision = evaluate("exam_registration_open", record, PROFILE, context=context)
    assert decision.result == "fail"
    assert decision.claim_id == 42


def test_month_precision_yields_unknown_inside_the_month(registry) -> None:
    """"March 2027" is never a fabricated day (spec 17 §6)."""

    def month_record(observed: datetime) -> dict:
        return programme(
            claims=window_claims(
                {"value": "2027-03", "observed_at": observed, "id": 43, "precision": "month"}
            )
        )

    inside = datetime(2027, 3, 15, 9, 0, tzinfo=UTC)
    inside_ctx = FilterContext.from_registry(registry, now=inside)
    decision = evaluate(
        "exam_registration_open",
        month_record(inside - timedelta(days=1)),
        PROFILE,
        context=inside_ctx,
    )
    assert decision.result == "unknown"
    assert decision.stale is False
    assert decision.claim_id == 43
    assert "month" in decision.reason

    before_ctx = FilterContext.from_registry(registry, now=datetime(2027, 1, 10, tzinfo=UTC))
    assert (
        evaluate(
            "exam_registration_open",
            month_record(datetime(2027, 1, 9, tzinfo=UTC)),
            PROFILE,
            context=before_ctx,
        ).result
        == "pass"
    )

    after_ctx = FilterContext.from_registry(registry, now=datetime(2027, 4, 2, tzinfo=UTC))
    after = evaluate(
        "exam_registration_open",
        month_record(datetime(2027, 4, 1, tzinfo=UTC)),
        PROFILE,
        context=after_ctx,
    )
    assert after.result == "fail"
    assert after.detail["exclusion_reason"] == "exam_registration_closed"


def test_date_only_close_stays_open_through_the_printed_ist_day(registry) -> None:
    """A date-only close is honoured through the whole Asia/Kolkata day."""

    record = programme(
        claims=window_claims(
            {"value": "2027-03-12", "observed_at": datetime(2027, 3, 11, tzinfo=UTC), "id": 44}
        )
    )
    midday = FilterContext.from_registry(registry, now=datetime(2027, 3, 12, 10, 0, tzinfo=UTC))
    assert evaluate("exam_registration_open", record, PROFILE, context=midday).result == "pass"

    past_midnight_ist = FilterContext.from_registry(
        registry, now=datetime(2027, 3, 12, 18, 31, tzinfo=UTC)
    )
    assert (
        evaluate("exam_registration_open", record, PROFILE, context=past_midnight_ist).result
        == "fail"
    )


def test_printed_deadline_time_is_respected_not_shifted(registry) -> None:
    """5:00 pm IST on 12/03/2027 → 11:30Z; the printed instant decides."""

    record = programme(
        claims=window_claims(
            {
                "value": "2027-03-12T11:30:00+00:00",
                "observed_at": datetime(2027, 3, 11, tzinfo=UTC),
                "id": 45,
            }
        )
    )
    just_before = FilterContext.from_registry(
        registry, now=datetime(2027, 3, 12, 11, 29, tzinfo=UTC)
    )
    assert evaluate("exam_registration_open", record, PROFILE, context=just_before).result == (
        "pass"
    )
    just_after = FilterContext.from_registry(
        registry, now=datetime(2027, 3, 12, 11, 31, tzinfo=UTC)
    )
    assert evaluate("exam_registration_open", record, PROFILE, context=just_after).result == "fail"


def test_cycles_coexist_and_the_target_cycle_decides(context: FilterContext) -> None:
    """A 2026 late-round window says nothing about a 2027 intake (spec 17 §2)."""

    record = programme(
        claims=window_claims(
            {
                "value": "2026-08-01",
                "observed_at": NOW - timedelta(days=1),
                "id": 46,
                "cycle_year": 2026,
            },
            {
                "value": "2027-02-15",
                "observed_at": NOW - timedelta(days=2),
                "id": 47,
                "cycle_year": 2027,
            },
        )
    )
    decided = evaluate("exam_registration_open", record, PROFILE, context=context)
    assert decided.result == "pass"
    assert decided.claim_id == 47

    wrong_cycle = {**PROFILE, "target_intake_year": 2028}
    missing = evaluate("exam_registration_open", record, wrong_cycle, context=context)
    assert missing.result == "unknown"
    # No window for the target cycle: the decision cites the exam-required
    # claim that established the requirement, not a window.
    assert missing.claim_id == 40
    assert "2027" in missing.reason and "2028" in missing.reason

    no_target = {k: v for k, v in PROFILE.items() if k != "target_intake_year"}
    latest = evaluate("exam_registration_open", record, no_target, context=context)
    assert latest.result == "fail"
    assert latest.claim_id == 46


def test_a_sitting_date_never_gates_registration(context: FilterContext) -> None:
    """Registration closed but the exam sits in the future: still excluded."""

    record = programme(
        claims=[
            *window_claims(
                {"value": "2026-11-01", "observed_at": NOW - timedelta(days=1), "id": 48}
            ),
            claim(
                "exam.sitting_dates",
                "The examination will be conducted on 22 April 2027 and 23 April 2027",
                qualifier={"exam_id": "mht_cet_pcb"},
                observed_at=NOW - timedelta(days=1),
                id=49,
            ),
            claim(
                "exam.date",
                "2027-04-22",
                qualifier={"exam_id": "mht_cet_pcb"},
                observed_at=NOW - timedelta(days=1),
                id=50,
            ),
        ]
    )
    decision = evaluate("exam_registration_open", record, PROFILE, context=context)
    assert decision.result == "fail"
    assert decision.detail["exclusion_reason"] == "exam_registration_closed"
    assert decision.claim_id == 48


def test_stale_window_is_unknown_and_never_open(context: FilterContext) -> None:
    record = programme(
        claims=window_claims(
            {"value": "2027-02-15", "observed_at": NOW - timedelta(days=10), "id": 51}
        )
    )
    decision = evaluate("exam_registration_open", record, PROFILE, context=context)
    assert decision.result == "unknown"
    assert decision.stale is True
    assert decision.claim_id == 51
