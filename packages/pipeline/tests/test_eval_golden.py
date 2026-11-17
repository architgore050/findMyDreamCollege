from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmd.eval.golden import (
    DEFAULT_GOLDEN_DIR,
    GoldenError,
    load_golden,
)
from fmd.eval.rowctx import occurrence_spans

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_committed_golden_set_loads_and_splits_verified_from_unverified() -> None:
    golden = load_golden(DEFAULT_GOLDEN_DIR)

    assert golden.verified_rows == 16
    assert golden.unverified_rows == 1
    assert len(golden.entries) == 5
    assert {trace.traceability for trace in golden.traces} == {
        "fixture",
        "unverified",
    }
    assert golden.as_dict()["row_context_rows"] == 1
    assert golden.as_dict()["row_context_verified_rows"] == 1
    template = DEFAULT_GOLDEN_DIR / "programmes" / "_TEMPLATE.yaml"
    assert template.exists()
    assert all("template" not in entry.id for entry in golden.entries)


def test_the_real_seat_matrix_row_is_verified_against_one_named_row() -> None:
    golden = load_golden(DEFAULT_GOLDEN_DIR)
    entry = next(item for item in golden.entries if item.id == "tnmgrmu-ahs-ug-seat-matrix-2024-25")
    assert entry.source_url and entry.retrieved_at
    assert entry.fixture, "a verified row must cite a committed fixture"
    assert entry.family is None  # fixture traceability only; not measured offline

    year = entry.expected["seats.intake_year"]
    assert year.status == "verified"
    assert year.value == "A.Y. 2024-2025"
    assert year.quote == "A.Y. 2024-2025"

    seats = entry.expected["seats.total"]
    assert seats.status == "verified"
    assert seats.value == 20
    assert seats.quote == "B.Sc. Medical Laboratory Technology 20"
    assert seats.context == "Govt. Stanley Medical College & Hospital, Chennai - 600 001"
    assert seats.disambiguation == "context_row"
    assert seats.row is not None
    assert seats.row.row_label == "3. B.Sc. Medical Laboratory Technology"
    assert seats.row.column_header == "Intake"
    assert seats.row.value_cell == "20"
    assert seats.row.table_page == 1


def test_backward_compatible_rows_without_context_still_load() -> None:
    """Existing committed rows predate the context fields and must be untouched."""

    golden = load_golden(DEFAULT_GOLDEN_DIR)
    rows = [
        row
        for entry in golden.entries
        for row in entry.verified_rows.values()
        if row.context is None
    ]
    assert rows, "the committed set must still contain context-free verified rows"
    assert all(row.disambiguation is None for row in rows)
    assert all(row.row is None for row in rows)


def test_every_verified_quote_resolves_in_its_committed_fixture() -> None:
    golden = load_golden(DEFAULT_GOLDEN_DIR)
    for entry in golden.entries:
        if entry.fixture is None:
            continue
        text = (REPO_ROOT / entry.fixture).read_text(encoding="utf-8")
        for row in entry.verified_rows.values():
            if row.expectation != "value":
                continue
            assert row.quote is not None
            assert occurrence_spans(text, row.quote), f"{entry.id}.{row.quote!r} is not verbatim"


def test_verified_context_must_resolve_and_be_unique_within_its_row(tmp_path: Path) -> None:
    fixture = (
        "SEAT MATRIX FOR THE A.Y. 2024-2025\n"
        "2\tGovt. Stanley Medical College & Hospital, Chennai - 600 001\t"
        "3. B.Sc. Medical Laboratory Technology\t20\n"
        "\t4. B.Sc Critical Care Technology\t10\n"
        "3\tChengalpattu Medical College & Hospital, Chengalpattu- 603 001\t"
        "1. B.Sc. Medical Laboratory Technology\t20\n"
    )
    (tmp_path / "fixture.tsv").write_text(fixture, encoding="utf-8")
    entry = {
        "schema_version": 1,
        "id": "row-context-ok",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "fixture": "fixture.tsv",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. Medical Laboratory Technology 20",
                "status": "verified",
                "context": "Govt. Stanley Medical College & Hospital, Chennai - 600 001",
                "disambiguation": "context_row",
                "row": {
                    "row_label": "3. B.Sc. Medical Laboratory Technology",
                    "column_header": "Intake",
                    "value_cell": "20",
                    "table_page": 1,
                },
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    golden = load_golden(tmp_path, root=tmp_path)
    assert golden.verified_rows == 1


def test_fabricated_context_is_rejected(tmp_path: Path) -> None:
    fixture = (
        "2\tGovt. Stanley Medical College & Hospital, Chennai - 600 001\t"
        "3. B.Sc. Medical Laboratory Technology\t20\n"
        "3\tChengalpattu Medical College & Hospital, Chengalpattu- 603 001\t"
        "1. B.Sc. Medical Laboratory Technology\t20\n"
    )
    (tmp_path / "fixture.tsv").write_text(fixture, encoding="utf-8")
    entry = {
        "schema_version": 1,
        "id": "fabricated-context",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "fixture": "fixture.tsv",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. Medical Laboratory Technology 20",
                "status": "verified",
                # This college prints the same row, but is not in this fixture.
                "context": "Govt. Mohan Kumaramangalam Medical College & Hospital, Salem - 636 030",
                "disambiguation": "context_row",
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    with pytest.raises(GoldenError, match="context .* occurs 0 times"):
        load_golden(tmp_path, root=tmp_path)


def test_context_row_claim_that_does_not_disambiguate_is_rejected(tmp_path: Path) -> None:
    fixture = (
        "2\tGovt. Stanley Medical College & Hospital, Chennai - 600 001\t"
        "3. B.Sc. Medical Laboratory Technology\t20\n"
        "\t4. B.Sc Critical Care Technology\t20\n"
    )
    (tmp_path / "fixture.tsv").write_text(fixture, encoding="utf-8")
    entry = {
        "schema_version": 1,
        "id": "context-does-not-disambiguate",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "fixture": "fixture.tsv",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. Medical Laboratory Technology 20",
                "status": "verified",
                "context": "Govt. Stanley Medical College & Hospital, Chennai - 600 001",
                "disambiguation": "context_row",
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    with pytest.raises(GoldenError, match="already unique"):
        load_golden(tmp_path, root=tmp_path)


def test_context_row_with_ambiguous_block_is_rejected(tmp_path: Path) -> None:
    fixture = (
        "2\tGovt. Stanley Medical College & Hospital, Chennai - 600 001\t"
        "3. B.Sc. Medical Laboratory Technology\t20\n"
        "\t3. B.Sc. Medical Laboratory Technology\t20\n"
        "3\tChengalpattu Medical College & Hospital, Chengalpattu- 603 001\t"
        "1. B.Sc. Medical Laboratory Technology\t20\n"
    )
    (tmp_path / "fixture.tsv").write_text(fixture, encoding="utf-8")
    entry = {
        "schema_version": 1,
        "id": "ambiguous-block",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "fixture": "fixture.tsv",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. Medical Laboratory Technology 20",
                "status": "verified",
                "context": "Govt. Stanley Medical College & Hospital, Chennai - 600 001",
                "disambiguation": "context_row",
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    with pytest.raises(GoldenError, match="leaves 2 quote occurrences"):
        load_golden(tmp_path, root=tmp_path)


def test_verified_row_cannot_claim_context_row_without_a_context(tmp_path: Path) -> None:
    (tmp_path / "fixture.txt").write_text("seats 20\n", encoding="utf-8")
    entry = {
        "schema_version": 1,
        "id": "contextless-row-claim",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "fixture": "fixture.txt",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "seats 20",
                "status": "verified",
                "disambiguation": "context_row",
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    with pytest.raises(GoldenError, match="requires a non-empty context"):
        load_golden(tmp_path, root=tmp_path)


def test_loader_rejects_a_verified_quote_that_is_not_in_the_fixture(tmp_path: Path) -> None:
    (tmp_path / "fixture.txt").write_text("Eligible students: 120\n", encoding="utf-8")
    entry = {
        "schema_version": 1,
        "id": "bogus",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "fixture": "fixture.txt",
        "expected": {
            "placement.eligible_count": {
                "value": 120,
                "unit": "count",
                "quote": "Eligible students: 999",
                "status": "verified",
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    with pytest.raises(GoldenError, match="not verbatim"):
        load_golden(tmp_path, root=tmp_path)


def test_loader_forbids_a_value_on_an_unverified_row(tmp_path: Path) -> None:
    (tmp_path / "fixture.txt").write_text("nothing here\n", encoding="utf-8")
    entry = {
        "schema_version": 1,
        "id": "fabricated",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "fixture": "fixture.txt",
        "expected": {
            "salary.median": {
                "value": 420000,
                "unit": "INR",
                "quote": "Median salary: INR 420000",
                "status": "unverified",
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    with pytest.raises(GoldenError, match="unverified row must set value: null"):
        load_golden(tmp_path, root=tmp_path)


def test_loader_checks_abstention_absence_terms(tmp_path: Path) -> None:
    (tmp_path / "fixture.txt").write_text("Highest package: INR 900000\n", encoding="utf-8")
    entry = {
        "schema_version": 1,
        "id": "bad-abstention",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "fixture": "fixture.txt",
        "expected": {
            "salary.highest_package": {
                "value": None,
                "status": "verified",
                "expectation": "abstain",
                "absent_terms": ["Highest package"],
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    with pytest.raises(GoldenError, match="absent"):
        load_golden(tmp_path, root=tmp_path)


def test_loader_rejects_a_cited_verified_row_without_retrieved_at(tmp_path: Path) -> None:
    entry = {
        "schema_version": 1,
        "id": "cited-but-undated",
        "institution": "Unnamed",
        "programme": "BSc MLT",
        "source_url": "https://example.invalid/seat-matrix.pdf",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. Medical Laboratory Technology 20",
                "status": "verified",
            }
        },
    }
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    with pytest.raises(GoldenError, match="retrieved_at"):
        load_golden(tmp_path, root=tmp_path)


def test_measured_rows_are_only_the_placement_fixture() -> None:
    golden = load_golden(DEFAULT_GOLDEN_DIR)
    measured = [entry.id for entry in golden.entries if entry.family]
    assert measured == ["fixture-placement-report-2024-25"]
