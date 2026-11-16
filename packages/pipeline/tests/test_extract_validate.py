"""Quote-first validation: deterministic spans, derived values, semantic guards."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fmd.extract.schemas import ExtractedField, FeeStructure, PlacementReport
from fmd.extract.validate import (
    Bands,
    Rejection,
    parse_number,
    resolve_quote,
    validate_document,
)

TEXT = (
    "BSc Medical Laboratory Technology\n"
    "Placement Report 2024-25\n"
    "Eligible students: 120\n"
    "Students placed: 96\n"
    "Placement rate: 80%\n"
    "Median salary: INR 420000 per annum\n"
    "Internships are included in the placement count.\n"
)
OBSERVED = datetime(2026, 11, 16, 9, 30, tzinfo=UTC)


def field(quote: str | None, *, confidence: float | None = 0.9) -> ExtractedField:
    return ExtractedField(quote=quote, confidence=confidence)


def placement(**overrides) -> PlacementReport:
    values = {
        "cohort_year": field("2024-25"),
        "eligible_count": field("Eligible students: 120"),
        "placed_count": field("Students placed: 96"),
        "placement_rate": field("Placement rate: 80%"),
        "median_salary": field("Median salary: INR 420000 per annum"),
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


def outcome_for(result, name: str):
    return next(item for item in result.outcomes if item.field == name)


# ------------------------------------------------------------------ exact resolution


def test_valid_document_derives_values_and_spans_from_quotes() -> None:
    result = validate(
        placement(outcome_basis=field("Internships are included in the placement count."))
    )

    assert result.rejections == []
    predicates = {claim.predicate for claim in result.claims}
    assert predicates == {
        "placement.cohort_year",
        "placement.eligible_count",
        "placement.placed_count",
        "placement.rate",
        "salary.median",
        "placement.basis",
    }
    rate = next(claim for claim in result.claims if claim.predicate == "placement.rate")
    assert rate.value == pytest.approx(0.8)
    assert rate.unit == "ratio"
    assert rate.qualifier["denominator"] == 120
    salary = next(claim for claim in result.claims if claim.predicate == "salary.median")
    assert salary.value == 420000
    assert salary.unit == "INR"
    assert salary.qualifier["period"] == "per_year"
    assert salary.qualifier["as_of"] == "2026-11-16"
    for claim in result.claims:
        assert TEXT[claim.char_start : claim.char_end] == claim.quote
        assert claim.method == "extracted"
    assert outcome_for(result, "placed_count").outcome == "resolved"


def test_exact_match_is_unique_and_offsets_are_computed_not_supplied() -> None:
    resolution = resolve_quote(TEXT, "Students placed: 96")

    assert resolution.status == "resolved"
    assert resolution.match is not None
    assert resolution.match.mode == "exact"
    assert TEXT[resolution.match.start : resolution.match.end] == "Students placed: 96"


def test_null_fields_emit_nothing_and_are_not_rejections() -> None:
    result = validate(PlacementReport())

    assert result.claims == []
    assert result.rejections == []
    assert result.rewrites == []
    assert {item.outcome for item in result.outcomes} == {"abstained"}


def test_confidence_defaults_when_the_model_omits_it() -> None:
    result = validate(placement(cohort_year=field("2024-25", confidence=None)))

    claim = next(item for item in result.claims if item.predicate == "placement.cohort_year")
    assert claim.confidence == pytest.approx(0.5)


# ------------------------------------------------------------------ normalisation


def test_whitespace_variant_resolves_via_normalisation() -> None:
    text = TEXT.replace("Placement rate: 80%", "Placement rate:  80%")
    result = validate(placement(), text)

    rate = next(item for item in result.claims if item.predicate == "placement.rate")
    assert rate.quote == "Placement rate:  80%"
    assert text[rate.char_start : rate.char_end] == rate.quote
    assert rate.confidence == pytest.approx(0.9 * 0.95)
    rewrite = next(item for item in result.rewrites if item.field == "placement_rate")
    assert rewrite.original_quote == "Placement rate: 80%"
    assert rewrite.corrected_quote == "Placement rate:  80%"


def test_nfkc_variant_resolves_to_the_source_span() -> None:
    text = TEXT.replace("Placement rate: 80%", "Placement rate: ８０％")
    result = validate(placement(), text)

    rate = next(item for item in result.claims if item.predicate == "placement.rate")
    assert rate.value == pytest.approx(0.8)
    assert rate.quote == "Placement rate: ８０％"
    assert text[rate.char_start : rate.char_end] == rate.quote


def test_devanagari_digits_and_rupee_spellings_resolve_and_parse() -> None:
    text = "ट्यूशन शुल्क: ₹ १,५०,००० प्रति वर्ष\n"
    document = FeeStructure(tuition_amount=field("शुल्क: Rs. 1,50,000 प्रति वर्ष"))

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.rejections == []
    claim = result.claims[0]
    assert claim.value == 150000
    assert claim.unit == "INR"
    assert claim.quote == "शुल्क: ₹ १,५०,००० प्रति वर्ष"
    assert text[claim.char_start : claim.char_end] == claim.quote


def test_indian_digit_grouping_is_removed_before_matching() -> None:
    text = "Total first year: ₹1,50,000\n"
    document = FeeStructure(total_first_year=field("Total first year: INR 150000"))

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.rejections == []
    claim = result.claims[0]
    assert claim.value == 150000
    assert claim.quote == "Total first year: ₹1,50,000"


# ------------------------------------------------------------------ ambiguity / repair


def test_duplicate_quote_is_ambiguous_and_never_first_match() -> None:
    text = "Students placed: 96\nStudents placed: 96\n"
    result = validate(placement(placed_count=field("Students placed: 96")), text)

    assert result.claims == []
    rejection = next(item for item in result.rejections if item.field == "placed_count")
    assert rejection.reason == "ambiguous_span"
    assert outcome_for(result, "placed_count").outcome == "ambiguous"


def test_one_repair_pass_recovers_an_edge_decorated_quote() -> None:
    result = validate(placement(cohort_year=field("(2024-25)")))

    claim = next(item for item in result.claims if item.predicate == "placement.cohort_year")
    assert claim.value == "2024-25"
    outcome = outcome_for(result, "cohort_year")
    assert outcome.outcome == "resolved"
    assert outcome.repaired is True
    assert claim.confidence == pytest.approx(0.9 * 0.95)


def test_unresolvable_quote_abstains_after_repair() -> None:
    result = validate(placement(placed_count=field("Students placed: 97")))

    assert "placement.placed_count" not in {claim.predicate for claim in result.claims}
    rejection = next(item for item in result.rejections if item.field == "placed_count")
    assert rejection.reason == "unresolved_span"
    assert outcome_for(result, "placed_count").outcome == "unresolved"


# ------------------------------------------------------------------ semantic guards


def test_salary_without_a_printed_period_is_rejected() -> None:
    text = "Median salary: INR 420000\n"
    document = PlacementReport(median_salary=field("Median salary: INR 420000"))

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.claims == []
    rejection = next(item for item in result.rejections if item.field == "median_salary")
    assert rejection.reason == "missing_period"
    assert outcome_for(result, "median_salary").outcome == "rejected"


def test_placed_greater_than_eligible_is_rejected_by_guard() -> None:
    text = "Eligible students: 120\nStudents placed: 130\n"
    document = PlacementReport(
        eligible_count=field("Eligible students: 120"),
        placed_count=field("Students placed: 130"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    rejection = next(item for item in result.rejections if item.field == "placed_count")
    assert rejection.reason == "guard_violation"
    assert rejection.guard == "placed_exceeds_eligible"
    assert outcome_for(result, "placed_count").outcome == "rejected_guard:placed_exceeds_eligible"
    assert "placement.placed_count" not in {claim.predicate for claim in result.claims}


def test_placement_rate_above_100_percent_is_rejected() -> None:
    text = "Eligible students: 120\nPlacement rate: 120%\n"
    document = PlacementReport(
        eligible_count=field("Eligible students: 120"),
        placement_rate=field("Placement rate: 120%"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    rejection = next(item for item in result.rejections if item.field == "placement_rate")
    assert rejection.guard == "placement_rate_above_100"
    outcome = outcome_for(result, "placement_rate")
    assert outcome.outcome == "rejected_guard:placement_rate_above_100"


def test_rate_inconsistent_with_printed_counts_is_rejected() -> None:
    text = "Eligible students: 120\nStudents placed: 96\nPlacement rate: 50%\n"
    document = PlacementReport(
        eligible_count=field("Eligible students: 120"),
        placed_count=field("Students placed: 96"),
        placement_rate=field("Placement rate: 50%"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    rejection = next(item for item in result.rejections if item.field == "placement_rate")
    assert rejection.guard == "rate_inconsistent_with_counts"


def test_seat_quotas_cannot_exceed_sanctioned_intake() -> None:
    from fmd.extract.schemas import SeatMatrix

    text = "Sanctioned intake: 60\nGovernment quota: 50\nManagement quota: 20\n"
    document = SeatMatrix(
        total_seats=field("Sanctioned intake: 60"),
        government_quota=field("Government quota: 50"),
        management_quota=field("Management quota: 20"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert {rejection.guard for rejection in result.rejections} == {
        "quotas_exceed_sanctioned_intake"
    }


def test_rate_without_denominator_is_rejected() -> None:
    result = validate(placement(eligible_count=None))

    rejection = next(item for item in result.rejections if item.predicate == "placement.rate")
    assert rejection.reason == "missing_denominator"


def test_money_without_a_printed_currency_is_rejected() -> None:
    text = "Median salary: 420000 per annum\n"
    document = PlacementReport(median_salary=field("Median salary: 420000 per annum"))

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    rejection = next(item for item in result.rejections if item.field == "median_salary")
    assert rejection.reason == "unparseable_unit"


def test_out_of_band_value_is_rejected() -> None:
    text = "Eligible students: 120000\n"
    document = PlacementReport(eligible_count=field("Eligible students: 120000"))

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    rejection = next(item for item in result.rejections if item.field == "eligible_count")
    assert rejection.reason == "out_of_band"
    assert "plausibility band" in rejection.detail


# ------------------------------------------------------------------ serialisation


def test_rejections_and_outcomes_serialise_for_the_review_console() -> None:
    result = validate(placement(placed_count=field("Students placed: 999")))

    rejection = result.rejections[0]
    assert isinstance(rejection, Rejection)
    payload = rejection.as_dict()
    assert payload["reason"] == "unresolved_span"
    assert payload["field"] == "placed_count"
    assert payload["predicate"] == "placement.placed_count"
    assert result.rejection_summary() == {"unresolved_span": 1}
    counts = result.outcome_counts()
    assert counts["unresolved"] == 1
    assert counts["resolved"] == 4


def test_fee_money_claims_carry_period_and_currency() -> None:
    text = "Tuition fee: INR 85000 per year\nTotal first year: Rs. 1,50,000\n"
    document = FeeStructure(
        tuition_amount=field("Tuition fee: INR 85000 per year"),
        total_first_year=field("Total first year: Rs. 1,50,000"),
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
