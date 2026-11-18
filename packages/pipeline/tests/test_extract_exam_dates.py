"""Deterministic date parsing for exam windows: precision, timezone, no fabrication.

Spec 17 §4/§6: a full date is ``precision: day``; a month-only span is
``precision: month`` and never becomes a fabricated day. Printed Indian
deadlines are Asia/Kolkata; a printed time of day is stored as the equivalent
UTC instant beside the original printed text, and the printed day never shifts.
"""

from __future__ import annotations

from datetime import datetime

from fmd.extract.schemas import ExamNotification, ExtractedField
from fmd.extract.validate import _date_derived_value, validate_document
from fmd.filters.impl import _month_bounds

OBSERVED = datetime(2026, 11, 16, 9, 30)

EXACT_TEXT = (
    "Maharashtra State Common Entrance Test Cell, Government of Maharashtra\n"
    "MHT-CET (PCB) 2027 — Notification No. 12/2026\n"
    "\n"
    "Online application form submission on the official portal starts on\n"
    "12 January 2027. The window closes up to 5:00 pm on 12/03/2027.\n"
    "The examination will be conducted on 22 April 2027 and 23 April 2027 at\n"
    "examination centres across Maharashtra. Candidates must appear in person\n"
    "at the allotted centre; no remote attempt is provided.\n"
    "Application fee: INR 800 for the open category.\n"
)


def derived(text: str):
    return _date_derived_value(text, text)


# ------------------------------------------------------------------ precision


def test_full_date_is_day_precision_with_printed_token() -> None:
    derived_value = derived("registration starts 12 January 2027")

    assert derived_value.value == "2027-01-12"
    assert derived_value.precision == "day"
    assert derived_value.date_text == "12 January 2027"
    assert derived_value.reason is None


def test_month_only_span_is_month_precision_and_never_a_day() -> None:
    derived_value = derived("open up to March 2027")

    assert derived_value.value == "2027-03"
    assert derived_value.precision == "month"
    assert derived_value.date_text == "March 2027"
    assert derived_value.reason is None
    # No fabricated day anywhere in the value.
    assert len(str(derived_value.value)) == 7


def test_numeric_month_year_is_month_precision() -> None:
    derived_value = derived("window: 03/2027")

    assert derived_value.value == "2027-03"
    assert derived_value.precision == "month"


def test_a_full_date_never_degrades_to_month_precision() -> None:
    derived_value = derived("12 March 2027")

    assert derived_value.value == "2027-03-12"
    assert derived_value.precision == "day"


def test_non_month_words_are_not_dates() -> None:
    derived_value = derived("Session 2026-27")

    assert derived_value.value is None
    assert derived_value.reason == "unparseable_value"


def test_iso_date_stays_day_precision() -> None:
    derived_value = derived("2027-03-12")

    assert derived_value.value == "2027-03-12"
    assert derived_value.precision == "day"
    assert derived_value.date_text == "2027-03-12"


# ------------------------------------------------------------------ timezone


def test_printed_time_is_stored_as_utc_instant_without_day_shift() -> None:
    """5:00 pm IST on 12/03/2027 is 11:30Z — still the printed day."""

    derived_value = derived("closes up to 5:00 pm on 12/03/2027")

    assert derived_value.value == "2027-03-12T11:30:00+00:00"
    assert derived_value.precision == "day"
    assert derived_value.date_text == "12/03/2027 5:00 pm"


def test_twenty_four_hour_time_is_also_the_same_printed_day() -> None:
    derived_value = derived("window closes 22-04-2027 at 17:00")

    assert derived_value.value == "2027-04-22T11:30:00+00:00"
    assert derived_value.precision == "day"


def test_late_evening_ist_deadline_lands_on_the_printed_day_in_utc() -> None:
    """23:30 IST on the 12th is 18:00Z on the 12th: IST has no half-year shift."""

    derived_value = derived("closes 12/03/2027 at 11:30 pm")

    assert derived_value.value == "2027-03-12T18:00:00+00:00"


def test_month_bounds_are_ist_day_boundaries() -> None:
    bounds = _month_bounds("2027-03")

    assert bounds is not None
    start, end = bounds
    assert start.isoformat() == "2027-03-01T00:00:00+05:30"
    assert end.isoformat() == "2027-04-01T00:00:00+05:30"


def test_month_bounds_reject_non_months() -> None:
    assert _month_bounds("March 2027") is None
    assert _month_bounds("2027-13") is None
    assert _month_bounds("2027-03-12") is None


# ------------------------------------------------- validate_document plumbing


def validate(document, text: str):
    return validate_document(
        document,
        text,
        extraction_id=3,
        entity_id=9,
        observed_at=OBSERVED,
    )


def field(quote: str | None, *, confidence: float | None = 0.9) -> ExtractedField:
    return ExtractedField(quote=quote, confidence=confidence)


def notification(**overrides) -> ExamNotification:
    values = {
        "exam_name": field("MHT-CET (PCB) 2027"),
        "cycle_year": field("MHT-CET (PCB) 2027"),
        "scope": field("State Common Entrance Test Cell"),
        "state": field("Government of Maharashtra"),
        "registration_opens": field("starts on\n12 January 2027"),
        "registration_closes": field("closes up to 5:00 pm on 12/03/2027"),
        "exam_date": None,
        "sitting_dates": field("conducted on 22 April 2027 and 23 April 2027"),
        "requires_travel": field("no remote attempt is provided"),
        "fee_amount": field("Application fee: INR 800"),
    }
    values.update(overrides)
    return ExamNotification(**values)


def test_exact_notification_resolves_spans_and_qualifiers() -> None:
    result = validate(notification(), EXACT_TEXT)

    assert result.rejections == []
    predicates = {claim.predicate for claim in result.claims}
    assert predicates == {
        "exam.name",
        "exam.cycle_year",
        "exam.scope",
        "exam.state",
        "exam.registration_opens",
        "exam.registration_closes",
        "exam.sitting_dates",
        "exam.requires_travel",
        "exam.fee",
    }
    for claim in result.claims:
        assert EXACT_TEXT[claim.char_start : claim.char_end] == claim.quote

    closes = next(c for c in result.claims if c.predicate == "exam.registration_closes")
    assert closes.value == "2027-03-12T11:30:00+00:00"
    assert closes.qualifier["precision"] == "day"
    assert closes.qualifier["date_text"] == "12/03/2027 5:00 pm"
    assert closes.qualifier["tz"] == "Asia/Kolkata"

    sittings = next(c for c in result.claims if c.predicate == "exam.sitting_dates")
    assert "22 April 2027" in sittings.value
    # The sitting-date claim is informational text, never a gated date value.
    assert sittings.qualifier.get("precision") is None

    opens = next(c for c in result.claims if c.predicate == "exam.registration_opens")
    assert opens.value == "2027-01-12"
    assert opens.qualifier["precision"] == "day"
    assert opens.qualifier["date_text"] == "12 January 2027"

    fee = next(c for c in result.claims if c.predicate == "exam.fee")
    assert fee.value == 800
    assert fee.unit == "INR"


def test_month_precision_notification_keeps_the_month_and_the_printed_text() -> None:
    text = (
        "West Bengal Joint Entrance Examinations Board\n"
        "JENPAS-UG 2027 — Advance Notice\n"
        "\n"
        "Online application will commence from January 2027.\n"
        "The application window is expected to remain open up to March 2027.\n"
        "Date of examination: to be notified later on the Board website.\n"
    )
    document = notification(
        exam_name=field("JENPAS-UG 2027"),
        cycle_year=field("JENPAS-UG 2027"),
        scope=None,
        state=field("West Bengal Joint Entrance Examinations Board"),
        registration_opens=field("commence from January 2027"),
        registration_closes=field("open up to March 2027"),
        sitting_dates=None,
        requires_travel=None,
        fee_amount=None,
    )

    result = validate(document, text)

    assert result.rejections == []
    closes = next(c for c in result.claims if c.predicate == "exam.registration_closes")
    assert closes.value == "2027-03"
    assert closes.qualifier["precision"] == "month"
    assert closes.qualifier["date_text"] == "March 2027"
    assert "T" not in str(closes.value)


def test_registration_close_is_not_taken_from_a_sitting_sentence() -> None:
    """Only a sitting date is printed: registration_closes stays null."""

    text = (
        "State Board of Technical Education\n"
        "DPH Entrance Test 2027\n"
        "\n"
        "The examination will be conducted on 22 April 2027.\n"
    )
    document = notification(
        exam_name=field("DPH Entrance Test 2027"),
        cycle_year=field("DPH Entrance Test 2027"),
        scope=None,
        state=None,
        registration_opens=None,
        registration_closes=None,
        sitting_dates=field("conducted on 22 April 2027"),
        requires_travel=None,
        fee_amount=None,
    )

    result = validate(document, text)

    assert result.rejections == []
    predicates = {claim.predicate for claim in result.claims}
    assert "exam.registration_closes" not in predicates
    assert "exam.sitting_dates" in predicates
    sitting = next(c for c in result.claims if c.predicate == "exam.sitting_dates")
    assert "22 April 2027" in sitting.value
