from __future__ import annotations

import pytest

from fmd.eval.metrics import (
    GoldenRow,
    Prediction,
    PublishedNumber,
    interval_coverage,
    normalise_unit,
    provenance_coverage,
    rejection_correctness,
    score_extraction,
)

TEXT = (
    "Eligible students: 120. Students placed: 96. "
    "Placement rate: 80%. Median salary: INR 420000 per annum."
)


def _span(quote: str) -> tuple[int, int]:
    start = TEXT.index(quote)
    return start, start + len(quote)


def _prediction(field: str, value: object, unit: str | None, quote: str) -> Prediction:
    start, end = _span(quote)
    return Prediction(
        field=field,
        value=value,
        unit=unit,
        quote=quote,
        char_start=start,
        char_end=end,
    )


def _rows() -> list[GoldenRow]:
    return [
        GoldenRow(
            field="placement.eligible_count",
            value=120,
            unit="count",
            quote="Eligible students: 120",
        ),
        GoldenRow(
            field="placement.placed_count",
            value=96,
            unit="count",
            quote="Students placed: 96",
        ),
        GoldenRow(
            field="placement.rate",
            value=0.8,
            unit="ratio",
            quote="Placement rate: 80%",
            printed_value="80%",
            printed_unit="percent",
        ),
        GoldenRow(
            field="salary.median",
            value=420000,
            unit="inr",
            quote="Median salary: INR 420000 per annum",
        ),
        GoldenRow(
            field="salary.highest_package",
            value=None,
            expectation="abstain",
        ),
    ]


def test_span_precision_recall_f1_is_hand_computable() -> None:
    predictions = [
        _prediction("placement.eligible_count", 120, "count", "Eligible students: 120"),
        _prediction("placement.placed_count", 90, "count", "Students placed: 96"),
        _prediction("placement.rate", 0.8, "percent", "Placement rate: 80%"),
        _prediction("salary.median", 420000, "INR", "Median salary: INR 420000 per annum"),
    ]
    metrics = score_extraction(_rows(), predictions, TEXT)

    assert metrics.evaluated == 5
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (2, 2, 0)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert metrics.value_accuracy == pytest.approx(0.75)
    assert metrics.unit_normalisation_accuracy == pytest.approx(0.75)
    assert metrics.unit_checks == 4
    assert metrics.abstention_rate == pytest.approx(0.2)
    assert metrics.abstention_correctness == pytest.approx(1.0)
    assert metrics.unverified_excluded == 0


def test_exact_match_requires_the_quote_to_resolve_at_the_span() -> None:
    row = _rows()[0]
    broken = Prediction(
        field=row.field,
        value=120,
        unit="count",
        quote="Eligible students: 120",
        char_start=0,
        char_end=5,
    )
    metrics = score_extraction([row], [broken], TEXT)
    assert metrics.false_positives == 1
    assert metrics.per_field[0].span_ok is False
    assert metrics.per_field[0].value_ok is True


def test_missing_prediction_is_an_abstention_not_a_zero() -> None:
    metrics = score_extraction(_rows()[:4], [], TEXT)
    assert metrics.abstentions == 4
    assert metrics.abstention_rate == pytest.approx(1.0)
    assert metrics.abstention_correctness == pytest.approx(0.0)
    assert metrics.false_negatives == 4
    assert metrics.recall == pytest.approx(0.0)


def test_unit_normalisation_uses_registry_vocabulary() -> None:
    assert normalise_unit("inr") == "INR"
    assert normalise_unit("Percent") == "percent"
    assert normalise_unit(None) is None
    row = _rows()[3]
    prediction = _prediction("salary.median", 420000, "INR", "Median salary: INR 420000 per annum")
    metrics = score_extraction([row], [prediction], TEXT)
    assert metrics.true_positives == 1


def test_unverified_rows_are_excluded_and_counted() -> None:
    rows = [
        GoldenRow(
            field="fee.tuition",
            value=None,
            status="unverified",
            expectation="value",
        )
    ]
    metrics = score_extraction(rows, [], TEXT, unverified_excluded=1)
    assert metrics.evaluated == 0
    assert metrics.unverified_excluded == 1
    assert metrics.precision is None


def test_provenance_coverage_counts_only_resolved_published_numbers() -> None:
    start = TEXT.index("Eligible students: 120")
    numbers = [
        PublishedNumber(
            predicate="placement.eligible_count",
            value=120,
            quote="Eligible students: 120",
            char_start=start,
            char_end=start + len("Eligible students: 120"),
            text=TEXT,
        ),
        PublishedNumber(
            predicate="placement.placed_count",
            value=96,
            quote="Students placed: 96",
            char_start=0,
            char_end=4,
            text=TEXT,
        ),
        PublishedNumber(
            predicate="dsr25",
            value=0.22,
            method="modelled",
            quote="modelled",
            text=TEXT,
        ),
        PublishedNumber(
            predicate="placement.cohort_year",
            value="2024-25",
            quote="2024-25",
            text=TEXT,
        ),
    ]
    report = provenance_coverage(numbers)
    assert report.total == 2
    assert report.resolved == 1
    assert report.coverage == pytest.approx(0.5)
    assert report.modelled_excluded == 1
    assert report.non_numeric_excluded == 1
    assert report.gate_failed(1.0) is True
    assert report.gate_failed(0.5) is False


def test_provenance_coverage_requires_snapshot_metadata_when_asked() -> None:
    start = TEXT.index("Eligible students: 120")
    number = PublishedNumber(
        predicate="placement.eligible_count",
        value=120,
        quote="Eligible students: 120",
        char_start=start,
        char_end=start + len("Eligible students: 120"),
        text=TEXT,
    )
    assert provenance_coverage([number], require_snapshot=True).coverage == 0.0


def test_rejection_correctness_hand_computed() -> None:
    rows = [
        GoldenRow(
            field="placement.rate",
            value=0.8,
            expectation="reject",
            reason="missing denominator",
        ),
        GoldenRow(field="salary.median", value=420000, unit="INR", quote="x"),
        GoldenRow(field="placement.placed_count", value=96, unit="count", quote="y"),
    ]
    rejections = [{"field": "placement.rate"}, {"field": "placement.placed_count"}]
    metrics = rejection_correctness(rows, rejections)
    assert metrics.expected_rejects == 1
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (1, 1, 0)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)


def test_interval_coverage_hand_computed() -> None:
    report = interval_coverage(
        [
            (400000, 350000, 450000),
            (400000, 410000, 450000),
            (400000, 300000, 390000),
            (400000, 400000, 400000),
        ]
    )
    assert report.n == 4
    assert report.inside == 2
    assert report.coverage == pytest.approx(0.5)
    assert report.nominal == pytest.approx(0.8)
    assert interval_coverage([]).coverage is None


# ------------------------------------------------------------------- row context

ROW_TEXT = (
    "Inst A\n"
    "3. B.Sc. Medical Laboratory Technology 20\n"
    "\n"
    "Inst B\n"
    "3. B.Sc. Medical Laboratory Technology 20\n"
)


def _row_context_row() -> GoldenRow:
    return GoldenRow(
        field="seats.total",
        value=20,
        unit="count",
        quote="B.Sc. Medical Laboratory Technology 20",
        context="Inst A",
        disambiguation="context_row",
        row_label="3. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        value_cell="20",
        table_page=1,
    )


def _row_prediction(at: int) -> Prediction:
    quote = "B.Sc. Medical Laboratory Technology 20"
    return Prediction(
        field="seats.total",
        value=20,
        unit="count",
        quote=quote,
        char_start=at,
        char_end=at + len(quote),
        disambiguation="context_row",
        row_label="3. B.Sc. Medical Laboratory Technology",
    )


def test_correct_value_in_the_correct_row_is_a_true_positive() -> None:
    first = ROW_TEXT.index("B.Sc. Medical Laboratory Technology 20")
    metrics = score_extraction([_row_context_row()], [_row_prediction(first)], ROW_TEXT)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.row_context_rows == 1
    assert metrics.row_attribution_checks == 1
    assert metrics.row_attribution_correct == 1
    assert metrics.row_attribution_accuracy == pytest.approx(1.0)
    assert metrics.wrong_attributions == 0
    assert metrics.per_field[0].attribution_ok is True


def test_correct_value_in_the_wrong_row_is_catastrophic_not_a_miss() -> None:
    first = ROW_TEXT.index("B.Sc. Medical Laboratory Technology 20")
    second = ROW_TEXT.index("B.Sc. Medical Laboratory Technology 20", first + 1)
    metrics = score_extraction([_row_context_row()], [_row_prediction(second)], ROW_TEXT)

    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0, "wrong attribution must not be counted as a miss"
    assert metrics.row_attribution_checks == 1
    assert metrics.row_attribution_correct == 0
    assert metrics.row_attribution_accuracy == pytest.approx(0.0)
    assert metrics.wrong_attributions == 1
    result = metrics.per_field[0]
    assert result.value_ok is True
    assert result.attribution_ok is False
    assert result.wrong_attribution is True
    assert "wrong row attribution" in result.detail


def test_row_attribution_is_unmeasured_when_no_row_context_prediction_exists() -> None:
    metrics = score_extraction([_row_context_row()], [], ROW_TEXT)
    assert metrics.false_negatives == 1
    assert metrics.row_context_rows == 1
    assert metrics.row_attribution_checks == 0
    assert metrics.row_attribution_accuracy is None
    assert metrics.wrong_attributions == 0


def test_row_context_is_ignored_for_rows_without_context() -> None:
    row = GoldenRow(field="placement.eligible_count", value=120, unit="count", quote="120")
    prediction = Prediction(
        field=row.field,
        value=120,
        unit="count",
        quote="120",
        char_start=0,
        char_end=3,
        context="some unrelated heading",
        row_label="some unrelated row",
    )
    metrics = score_extraction([row], [prediction], "120")
    assert metrics.true_positives == 1
    assert metrics.row_context_rows == 0
    assert metrics.row_attribution_accuracy is None


def test_unverified_row_with_context_is_excluded_from_attribution() -> None:
    row = GoldenRow(
        field="seats.total",
        value=None,
        status="unverified",
        context="Inst A",
        disambiguation="context_row",
    )
    metrics = score_extraction([row], [], ROW_TEXT, unverified_excluded=1)
    assert metrics.evaluated == 0
    assert metrics.row_context_rows == 0
    assert metrics.unverified_excluded == 1
    assert metrics.row_attribution_accuracy is None
