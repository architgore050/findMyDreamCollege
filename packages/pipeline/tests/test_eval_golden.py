from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmd.eval.golden import (
    DEFAULT_GOLDEN_DIR,
    GoldenError,
    load_golden,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_committed_golden_set_loads_and_splits_verified_from_unverified() -> None:
    golden = load_golden(DEFAULT_GOLDEN_DIR)

    assert golden.verified_rows == 14
    assert golden.unverified_rows == 1
    assert len(golden.entries) == 4
    assert {trace.traceability for trace in golden.traces} == {"fixture", "unverified"}
    template = DEFAULT_GOLDEN_DIR / "programmes" / "_TEMPLATE.yaml"
    assert template.exists()
    assert all("template" not in entry.id for entry in golden.entries)


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
            assert row.quote in text, f"{entry.id}.{row.quote!r} is not verbatim"


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


def test_measured_rows_are_only_the_placement_fixture() -> None:
    golden = load_golden(DEFAULT_GOLDEN_DIR)
    measured = [entry.id for entry in golden.entries if entry.family]
    assert measured == ["fixture-placement-report-2024-25"]
