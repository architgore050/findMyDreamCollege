"""Quote-first validation: deterministic spans, derived values, semantic guards."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fmd.acquire.canonical import TextBlock, build_canonical
from fmd.acquire.types import PageSpan
from fmd.extract.schemas import ExtractedField, FeeStructure, PlacementReport, SeatMatrix
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
        intake=field("Sanctioned intake: 60"),
        government_quota=field("Government quota: 50"),
        management_quota=field("Management quota: 20"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert {rejection.guard for rejection in result.rejections} == {
        "quotas_exceed_sanctioned_intake"
    }


def test_aggregate_seats_also_bound_row_counts_when_no_intake_row_is_extracted() -> None:
    """A document-stated total still bounds a row's admitted count.

    `seats.total` is the aggregate scope (a printed Total row); it is a weaker
    bound than the row's own intake, but it is still an upper bound — the guard
    must not vanish just because the matrix has no per-row intake column.
    """
    from fmd.extract.schemas import SeatMatrix

    text = "Total: 136\nAdmitted: 200\n"
    document = SeatMatrix(
        aggregate_seats=field("Total: 136"),
        admitted=field("Admitted: 200"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    rejection = next(item for item in result.rejections if item.field == "admitted")
    assert rejection.guard == "admitted_exceeds_sanctioned_intake"


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


# ------------------------------------------------------------------ wave-2 money guards


def test_fee_money_without_any_period_is_rejected() -> None:
    text = "Tuition fee: INR 85000\n"
    document = FeeStructure(tuition_amount=field("Tuition fee: INR 85000"))

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.claims == []
    rejection = next(item for item in result.rejections if item.field == "tuition_amount")
    assert rejection.reason == "missing_period"
    assert "fee_period" in rejection.detail


def test_fee_period_sibling_supplies_the_period_with_provenance() -> None:
    text = "Tuition fee: INR 85000\nAll fees are charged per semester.\n"
    document = FeeStructure(
        fee_period=field("All fees are charged per semester"),
        tuition_amount=field("Tuition fee: INR 85000"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.rejections == []
    tuition = next(claim for claim in result.claims if claim.predicate == "fee.tuition")
    assert tuition.qualifier["period"] == "per_semester"
    assert tuition.qualifier["period_source"] == "fee.period"
    assert tuition.qualifier["fee_period"] == "semester"


def test_fee_period_sibling_cannot_be_mapped_still_rejects() -> None:
    text = "Tuition fee: INR 85000\nThe fee schedule is revised from time to time.\n"
    document = FeeStructure(
        fee_period=field("The fee schedule is revised from time to time"),
        tuition_amount=field("Tuition fee: INR 85000"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    rejection = next(item for item in result.rejections if item.field == "tuition_amount")
    assert rejection.reason == "missing_period"


def test_mess_and_other_fees_extract_with_their_own_period() -> None:
    text = "Mess charge: Rs. 2,400 per month\nTransport fee: Rs. 6,000 per year\n"
    document = FeeStructure(
        mess_amount=field("Mess charge: Rs. 2,400 per month"),
        other_amount=field("Transport fee: Rs. 6,000 per year"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.rejections == []
    values = {claim.predicate: claim for claim in result.claims}
    assert values["fee.mess"].value == 2400
    assert values["fee.mess"].qualifier["period"] == "per_month"
    assert values["fee.other"].value == 6000
    assert values["fee.other"].qualifier["period"] == "per_year"


def test_salary_period_sibling_supplies_the_period_with_provenance() -> None:
    text = "Median salary: INR 35000\nSalaries are quoted per month.\n"
    document = PlacementReport(
        salary_period=field("Salaries are quoted per month"),
        median_salary=field("Median salary: INR 35000"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.rejections == []
    salary = next(claim for claim in result.claims if claim.predicate == "salary.median")
    assert salary.value == 35000
    assert salary.qualifier["period"] == "per_month"
    assert salary.qualifier["period_source"] == "placement.salary_period"


def test_admitted_exceeding_sanctioned_intake_is_rejected() -> None:
    from fmd.extract.schemas import SeatMatrix

    text = "Sanctioned intake: 40\nAdmitted: 55\n"
    document = SeatMatrix(
        intake=field("Sanctioned intake: 40"),
        admitted=field("Admitted: 55"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    rejection = next(item for item in result.rejections if item.field == "admitted")
    assert rejection.guard == "admitted_exceeds_sanctioned_intake"
    assert outcome_for(result, "admitted").outcome == (
        "rejected_guard:admitted_exceeds_sanctioned_intake"
    )


def test_admitted_within_sanctioned_intake_and_duration_extract() -> None:
    from fmd.extract.schemas import SeatMatrix

    text = "Sanctioned intake: 40\nAdmitted: 38\nDuration (months): 24\n"
    document = SeatMatrix(
        intake=field("Sanctioned intake: 40"),
        admitted=field("Admitted: 38"),
        duration_months=field("Duration (months): 24"),
    )

    result = validate_document(document, text, extraction_id=1, entity_id=2, observed_at=OBSERVED)

    assert result.rejections == []
    values = {claim.predicate: claim for claim in result.claims}
    assert values["seats.admitted"].value == 38
    assert values["seats.duration_months"].value == 24
    assert values["seats.duration_months"].unit == "months"


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


# ------------------------------------------------------------------ W7 resolution scope


def test_document_scoped_claims_record_the_scope_and_the_page() -> None:
    pages = [PageSpan(page=1, char_start=0, char_end=len(TEXT))]
    result = validate_document(
        placement(),
        TEXT,
        extraction_id=7,
        entity_id=11,
        observed_at=OBSERVED,
        pages=pages,
    )

    assert result.rejections == []
    for claim in result.claims:
        assert claim.qualifier["resolution_scope"] == "document"
        assert claim.qualifier["disambiguation"] == "unique"
        assert claim.qualifier["scope_page_no"] == 1
    outcome = outcome_for(result, "eligible_count")
    assert outcome.scope == "document"
    assert outcome.char_start is not None
    assert TEXT[outcome.char_start : outcome.char_end] == outcome.quote


def test_claims_record_the_page_their_span_sits_on() -> None:
    text = TEXT + "Highest package: INR 600000 per annum\n"
    pages = [
        PageSpan(page=1, char_start=0, char_end=len(TEXT)),
        PageSpan(page=2, char_start=len(TEXT), char_end=len(text)),
    ]
    result = validate_document(
        placement(highest_package=field("Highest package: INR 600000 per annum")),
        text,
        extraction_id=7,
        entity_id=11,
        observed_at=OBSERVED,
        pages=pages,
    )

    assert result.rejections == []
    by_page = {
        claim.predicate: claim.qualifier["scope_page_no"] for claim in result.claims
    }
    assert by_page["placement.eligible_count"] == 1
    assert by_page["salary.highest_package"] == 2


def test_repaired_claims_still_record_their_scope_and_page() -> None:
    pages = [PageSpan(page=1, char_start=0, char_end=len(TEXT))]
    result = validate_document(
        placement(cohort_year=field("(2024-25)")),
        TEXT,
        extraction_id=7,
        entity_id=11,
        observed_at=OBSERVED,
        pages=pages,
    )

    claim = next(claim for claim in result.claims if claim.predicate == "placement.cohort_year")
    assert claim.qualifier["resolution_scope"] == "document"
    assert claim.qualifier["scope_page_no"] == 1
    assert outcome_for(result, "cohort_year").repaired is True


def test_ambiguous_quotes_still_reject_when_no_scope_geometry_is_supplied() -> None:
    text = "Students placed: 96\nStudents placed: 96\n"
    result = validate(placement(placed_count=field("Students placed: 96")), text)

    rejection = next(item for item in result.rejections if item.field == "placed_count")
    assert rejection.reason == "ambiguous_span"
    assert "no row context was supplied" in rejection.detail


def test_section_scoped_claim_records_heading_and_section_span() -> None:
    text = (
        "LAB AUDIT 2025-26\n"
        "Lab audit: 87 samples cleared.\n"
        "STUDENT AFFAIRS\n"
        "Lab audit: 87 samples cleared.\n"
        "Lab audit: 87 samples cleared.\n"
    )
    canonical = build_canonical([TextBlock(page_no=1, text=line) for line in text.splitlines()])
    document = SeatMatrix(
        programme_name=ExtractedField(quote="Lab audit: 87 samples cleared.", confidence=0.9)
    )

    result = validate_document(
        document,
        text,
        extraction_id=1,
        entity_id=2,
        observed_at=OBSERVED,
        pages=[PageSpan(page=1, char_start=0, char_end=len(text))],
        sections=canonical.sections,
    )

    assert len(result.claims) == 1
    claim = result.claims[0]
    qualifier = claim.qualifier
    assert qualifier["resolution_scope"] == "section"
    assert qualifier["disambiguation"] == "unique_section"
    assert qualifier["section_heading"] == "LAB AUDIT 2025-26"
    assert qualifier["scope_page_no"] == 1
    section_text = text[qualifier["scope_char_start"] : qualifier["scope_char_end"]]
    assert section_text.startswith("LAB AUDIT 2025-26")
    assert section_text.count("Lab audit: 87 samples cleared.") == 1
    # Provenance stays absolute: the span indexes the canonical document text.
    assert text[claim.char_start : claim.char_end] == "Lab audit: 87 samples cleared."
    assert outcome_for(result, "programme_name").scope == "section"
