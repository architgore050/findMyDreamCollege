from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fmd.extract.schemas import ExtractedField, FeeStructure, PlacementReport
from fmd.extract.validate import (
    Bands,
    Rejection,
    parse_number,
    validate_document,
)

TEXT = (
    "BSc Medical Laboratory Technology\n"
    "Placement Report 2024-25\n"
    "Eligible students: 120\n"
    "Students placed: 96\n"
    "Placement rate: 80%\n"
    "Median salary: INR 420000 per annum\n"
)
OBSERVED = datetime(2026, 11, 16, 9, 30, tzinfo=UTC)


def field(
    value,
    *,
    unit: str | None = None,
    quote: str | None = None,
    start: int | None = None,
    end: int | None = None,
    confidence: float | None = 0.9,
) -> ExtractedField:
    if quote is not None and start is None:
        found = TEXT.find(quote)
        start = found if found >= 0 else None
        end = start + len(quote) if start is not None else None
    return ExtractedField(
        value=value,
        unit=unit,
        quote=quote,
        char_start=start,
        char_end=end,
        confidence=confidence,
    )


def placement(**overrides) -> PlacementReport:
    values = {
        "cohort_year": field("2024-25", quote="2024-25"),
        "eligible_count": field(120, unit="count", quote="Eligible students: 120"),
        "placed_count": field(96, unit="count", quote="Students placed: 96"),
        "placement_rate": field("80%", unit="percent", quote="Placement rate: 80%"),
        "median_salary": field(420000, unit="INR", quote="Median salary: INR 420000 per annum"),
        "highest_package": None,
        "internship_counted_as_employment": None,
        "outcome_basis": None,
    }
    values.update(overrides)
    return PlacementReport(**values)


def validate(document, text: str = TEXT):
    return validate_document(
        document,
        text,
        extraction_id=7,
        entity_id=11,
        observed_at=OBSERVED,
    )


def test_valid_document_builds_claims_with_spans_and_qualifiers() -> None:
    result = validate(placement())

    assert result.rejections == []
    assert {claim.predicate for claim in result.claims} == {
        "placement.cohort_year",
        "placement.eligible_count",
        "placement.placed_count",
        "placement.rate",
        "salary.median",
    }
    rate = next(claim for claim in result.claims if claim.predicate == "placement.rate")
    assert rate.value == pytest.approx(0.8)
    assert rate.unit == "ratio"
    assert rate.qualifier["denominator"] == 120
    assert TEXT[rate.char_start : rate.char_end] == rate.quote
    salary = next(claim for claim in result.claims if claim.predicate == "salary.median")
    assert salary.qualifier["period"] == "per_year"
    assert salary.qualifier["as_of"] == "2026-11-16"
    assert all(claim.method == "extracted" for claim in result.claims)


def test_quote_absent_from_source_is_rejected_not_fixed() -> None:
    result = validate(placement(placed_count=field(97, unit="count", quote="Students placed: 97")))

    assert [claim.predicate for claim in result.claims].count("placement.placed_count") == 0
    rejection = next(
        item for item in result.rejections if item.predicate == "placement.placed_count"
    )
    assert rejection.reason == "span_mismatch"
    assert rejection.quote == "Students placed: 97"


def test_wrong_offsets_are_rewritten_to_the_exact_match() -> None:
    result = validate(
        placement(
            eligible_count=field(120, unit="count", quote="Eligible students: 120", start=0, end=5)
        )
    )

    assert result.rejections == []
    assert len(result.rewrites) == 1
    rewrite = result.rewrites[0]
    assert rewrite.before_start == 0
    assert TEXT[rewrite.after_start : rewrite.after_end] == "Eligible students: 120"
    claim = next(item for item in result.claims if item.predicate == "placement.eligible_count")
    assert claim.confidence == pytest.approx(0.9 * 0.95)


def test_fuzzy_match_at_or_above_threshold_rewrites_and_lowers_confidence() -> None:
    text = TEXT.replace("Placement rate: 80%", "Placement rate:  80%")
    result = validate_document(
        placement(),
        text,
        extraction_id=7,
        entity_id=11,
        observed_at=OBSERVED,
    )

    rate = next(item for item in result.claims if item.predicate == "placement.rate")
    assert text[rate.char_start : rate.char_end] == "Placement rate:  80%"
    assert rate.quote == "Placement rate:  80%"
    assert rate.confidence == pytest.approx(0.9 * 0.95)
    rewrite = next(item for item in result.rewrites if item.field == "placement_rate")
    assert rewrite.original_quote == "Placement rate: 80%"
    assert rewrite.corrected_quote == "Placement rate:  80%"
    assert 0.95 <= rewrite.ratio <= 1.0


def test_rate_without_denominator_is_rejected() -> None:
    result = validate(placement(eligible_count=None))

    rejection = next(item for item in result.rejections if item.predicate == "placement.rate")
    assert rejection.reason == "missing_denominator"
    assert "denominator" in rejection.detail


def test_missing_unit_is_rejected() -> None:
    result = validate(placement(placed_count=field(96, quote="Students placed: 96")))

    rejection = next(
        item for item in result.rejections if item.predicate == "placement.placed_count"
    )
    assert rejection.reason == "missing_unit"


def test_unparseable_unit_is_rejected() -> None:
    result = validate(
        placement(placed_count=field(96, unit="students per year", quote="Students placed: 96"))
    )

    rejection = next(
        item for item in result.rejections if item.predicate == "placement.placed_count"
    )
    assert rejection.reason == "unparseable_unit"


def test_money_without_currency_is_rejected() -> None:
    result = validate(
        placement(
            median_salary=field(420000, unit="count", quote="Median salary: INR 420000 per annum")
        )
    )

    rejection = next(item for item in result.rejections if item.predicate == "salary.median")
    assert rejection.reason == "unparseable_unit"


def test_out_of_band_value_is_rejected() -> None:
    result = validate(
        placement(eligible_count=field(120000, unit="count", quote="Eligible students: 120"))
    )

    rejection = next(
        item for item in result.rejections if item.predicate == "placement.eligible_count"
    )
    assert rejection.reason == "out_of_band"
    assert "plausibility band" in rejection.detail


def test_missing_quote_is_rejected() -> None:
    result = validate(placement(outcome_basis=field("included", quote=None)))

    rejection = next(item for item in result.rejections if item.predicate == "placement.basis")
    assert rejection.reason == "missing_quote"


def test_null_fields_emit_nothing_and_are_not_rejections() -> None:
    document = PlacementReport()

    result = validate(document)

    assert result.claims == []
    assert result.rejections == []
    assert result.rewrites == []


def test_value_inside_quote_but_quote_not_at_span_is_logged_as_rewrite() -> None:
    result = validate(placement(cohort_year=field("2024-25", quote="2024-25", start=999, end=1006)))

    assert result.rejections == []
    assert result.rewrites[0].field == "cohort_year"


def test_confidence_defaults_when_the_model_omits_it() -> None:
    result = validate(
        placement(
            cohort_year=field("2024-25", quote="2024-25", confidence=None),
        )
    )

    claim = next(item for item in result.claims if item.predicate == "placement.cohort_year")
    assert claim.confidence == pytest.approx(0.5)


def test_rejections_serialise_for_the_review_console() -> None:
    result = validate(placement(placed_count=field(96, quote="Students placed: 999")))

    rejection = result.rejections[0]
    assert isinstance(rejection, Rejection)
    payload = rejection.as_dict()
    assert payload["reason"] == "span_mismatch"
    assert payload["field"] == "placed_count"
    assert payload["predicate"] == "placement.placed_count"
    assert result.rejection_summary() == {"span_mismatch": 1}


def test_fee_money_claims_carry_period_and_currency() -> None:
    text = "Tuition fee: INR 85000 per year\nTotal first year: Rs. 1,50,000\n"
    document = FeeStructure(
        tuition_amount=ExtractedField(
            value="85000",
            unit="INR",
            quote="Tuition fee: INR 85000 per year",
            char_start=text.index("Tuition fee"),
            char_end=text.index("per year") + len("per year"),
            confidence=0.9,
        ),
        total_first_year=ExtractedField(
            value="1,50,000",
            unit="INR",
            quote="Total first year: Rs. 1,50,000",
            char_start=text.index("Total first year"),
            char_end=len(text.rstrip()),
            confidence=0.8,
        ),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.rejections == []
    values = {claim.predicate: claim for claim in result.claims}
    assert values["fee.tuition"].value == 85000
    assert values["fee.tuition"].qualifier["period"] == "per_year"
    assert values["fee.total_first_year"].value == 150000
    assert values["fee.total_first_year"].unit == "INR"


def test_parse_number_handles_printed_formats_but_refuses_magnitudes() -> None:
    assert parse_number("1,800 hours") == 1800
    assert parse_number("₹85,000") == 85000
    assert parse_number("80%") == 80
    assert parse_number(42) == 42
    assert parse_number("1.5 lakh") is None
    assert parse_number("not a number") is None
    assert parse_number(True) is None


def test_bands_come_from_yaml_not_python() -> None:
    bands = Bands.load()
    assert "INR" in bands.currencies
    assert "count" in bands.units
    assert bands.band("placement.rate").maximum == 1
