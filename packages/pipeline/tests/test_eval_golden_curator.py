"""Curation logic tests: proofs, fallback, check and list (no PDF required).

The PDF front-end (pdftext) is exercised in ``test_eval_golden_cli.py``; here the
same plan/prove/execute pipeline runs on hand-built :class:`PageText` objects,
so a missing extraction extra cannot hide a rule regression.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from fmd.eval.curator import (
    AddInputs,
    CuratorError,
    check_golden,
    execute_add,
    list_golden,
    parse_scalar,
    plan_add,
    prove_quote,
    resolve_candidate,
    row_gate,
    slugify,
    trim_words,
)
from fmd.eval.golden import load_golden
from fmd.eval.pdfpage import PageText, PdfMeta

NOW = date(2026, 11, 18)

# The canonical seat-matrix shape: two college records, the same course row and
# the same intake number in both, blank line between records.
RECORDS_TEXT = (
    "SEAT MATRIX FOR THE A.Y. 2024-2025\n"
    "1 College A Institute\n"
    "1. B.Sc. Medical Laboratory Technology 20\n"
    "2. B.Sc Cardiac Technology 20\n"
    "\n"
    "2 College B Institute\n"
    "1. B.Sc. Medical Laboratory Technology 20\n"
    "2. B.Sc Cardiac Technology 30\n"
)


def page_of(text: str = RECORDS_TEXT) -> PageText:
    lines = tuple(text.rstrip("\n").split("\n"))
    spans: list[tuple[int, int]] = []
    cursor = 0
    for line in lines:
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    return PageText(page=1, lines=lines, text=text, line_spans=tuple(spans))


def meta_of() -> PdfMeta:
    return PdfMeta(sha256="a" * 64, page_count=2, source="seat-matrix.pdf", tool_version="test")


def inputs_of(**overrides: object) -> AddInputs:
    base: dict[str, object] = {
        "source": "seat-matrix.pdf",
        "page": 1,
        "predicate": "seats.total",
        "value_raw": "20",
        "unit": "count",
        "institution": "College A Institute",
        "programme": "B.Sc. MLT",
        "golden_dir": Path("golden"),
        "root": Path("."),
    }
    base.update(overrides)
    return AddInputs(**base)  # type: ignore[arg-type]


def add_row(tmp_path: Path, page: PageText, **overrides: object):
    golden = tmp_path / "golden"
    overrides.setdefault("golden_dir", golden)
    overrides.setdefault("root", tmp_path)
    pick = overrides.pop("pick", None)
    inputs = inputs_of(**overrides)
    plan = plan_add(inputs, page)
    candidate = resolve_candidate(plan, pick)  # type: ignore[arg-type]
    return execute_add(plan, candidate, meta_of(), now=NOW)


# ------------------------------------------------------------------- prove_quote


def test_prove_quote_accepts_a_globally_unique_quote() -> None:
    proof = prove_quote(RECORDS_TEXT, "SEAT MATRIX FOR THE A.Y. 2024-2025")
    assert proof.ok
    assert proof.disambiguation == "unique"
    assert proof.detail["quote_occurrences"] == 1


def test_prove_quote_fails_ambiguous_quote_without_context() -> None:
    proof = prove_quote(RECORDS_TEXT, "B.Sc. Medical Laboratory Technology 20")
    assert not proof.ok
    assert proof.disambiguation is None
    assert proof.detail["quote_occurrences"] == 2
    assert "--context" in proof.detail["suggestion"]


def test_prove_quote_proves_context_row_like_the_loader(tmp_path: Path) -> None:
    proof = prove_quote(
        RECORDS_TEXT,
        "B.Sc. Medical Laboratory Technology 20",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
    )
    assert proof.ok
    assert proof.disambiguation == "context_row"
    assert proof.detail["quote_occurrences"] == 2
    assert proof.detail["quote_occurrences_in_block"] == 1
    assert proof.detail["row_label_occurrences_in_block"] == 1
    # The exact row must also load through the golden loader.
    entry = {
        "id": "row-context-ok",
        "institution": "College A Institute",
        "programme": "B.Sc. MLT",
        "fixture": "fixture.txt",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. Medical Laboratory Technology 20",
                "status": "verified",
                "context": "College A Institute",
                "disambiguation": "context_row",
                "row": {"row_label": "1. B.Sc. Medical Laboratory Technology"},
            }
        },
    }
    (tmp_path / "fixture.txt").write_text(RECORDS_TEXT, encoding="utf-8")
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    golden = load_golden(tmp_path, root=tmp_path)
    assert golden.verified_rows == 1


def test_prove_quote_rejects_unique_quote_with_context_as_mislabelled() -> None:
    proof = prove_quote(
        RECORDS_TEXT, "SEAT MATRIX FOR THE A.Y. 2024-2025", context="College A Institute"
    )
    assert not proof.ok
    assert "already globally unique" in proof.problems[0]
    assert "without --context" in proof.detail["suggestion"]


def test_prove_quote_rejects_a_context_that_does_not_disambiguate() -> None:
    ambiguous_block = RECORDS_TEXT.replace(
        "1. B.Sc. Medical Laboratory Technology 20\n2. B.Sc Cardiac Technology 30",
        "1. B.Sc. Medical Laboratory Technology 20\n1. B.Sc. Medical Laboratory Technology 20",
    )
    proof = prove_quote(
        ambiguous_block,
        "B.Sc. Medical Laboratory Technology 20",
        context="College B Institute",
    )
    assert not proof.ok
    assert "leaves 2 quote occurrences" in proof.problems[0]


def test_prove_quote_rejects_row_label_outside_the_block() -> None:
    proof = prove_quote(
        RECORDS_TEXT,
        "B.Sc. Medical Laboratory Technology 20",
        context="College B Institute",
        row_label="College A Institute",
    )
    assert not proof.ok
    assert "row label" in proof.problems[0]


def test_prove_quote_rejects_a_context_occurring_twice() -> None:
    text = RECORDS_TEXT.replace("2 College B Institute", "College A Institute (duplicate)")
    proof = prove_quote(
        text,
        "B.Sc. Medical Laboratory Technology 20",
        context="College A Institute",
    )
    assert not proof.ok
    assert "occurs 2 times" in proof.problems[0]


# ------------------------------------------------------------- scalar parsing


def test_parse_scalar_parses_numbers_bools_and_text() -> None:
    assert parse_scalar("20") == 20
    assert parse_scalar(" 20 ") == 20
    assert parse_scalar("0.8") == 0.8
    assert parse_scalar("true") is True
    assert parse_scalar("False") is False
    assert parse_scalar("A.Y. 2024-2025") == "A.Y. 2024-2025"
    assert parse_scalar('"2(f) & 12B"') == "2(f) & 12B"


def test_parse_scalar_refuses_null_and_empty_values() -> None:
    with pytest.raises(CuratorError, match="cannot be null"):
        parse_scalar("null")
    with pytest.raises(CuratorError, match="empty"):
        parse_scalar("   ")


def test_slugify_and_trim_words() -> None:
    assert slugify("Govt. Stanley Medical College & Hospital") == (
        "govt-stanley-medical-college-hospital"
    )
    assert trim_words("3. B.Sc. Medical Laboratory Technology 20", 1, 0) == (
        "B.Sc. Medical Laboratory Technology 20"
    )
    assert trim_words("one two", 1, 1) is None  # cannot trim everything
    assert trim_words("single", 1, 0) is None


# ----------------------------------------------------------------- add flow


def test_add_writes_verified_row_that_the_loader_accepts(tmp_path: Path) -> None:
    outcome = add_row(
        tmp_path,
        page_of(),
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        printed_value="20",
        trim_left=1,  # drop the serial '1.' like the extractor's row-label quote
        pick=2,  # candidates also include the A.Y. 2024-2025 title line (needle '20')
    )
    assert outcome.status == "verified"
    assert outcome.proof.disambiguation == "context_row"
    entry = load_golden(tmp_path / "golden", root=tmp_path)
    assert entry.verified_rows == 1
    row = entry.entries[0].expected["seats.total"]
    assert row.value == 20
    assert row.quote == "B.Sc. Medical Laboratory Technology 20"
    assert row.disambiguation == "context_row"
    assert row.row is not None and row.row.table_page == 1
    fixture = tmp_path / "golden" / "fixtures" / "college-a-institute_page01.txt"
    assert fixture.read_text(encoding="utf-8") == RECORDS_TEXT
    sidecar = json.loads(fixture.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["schema"] == "fmd-golden-fixture-meta-v1"
    assert sidecar["fixture_sha256"] == outcome.fixture_sha256
    assert sidecar["source_sha256"] == "a" * 64
    assert sidecar["retrieved_at"] == "2026-11-18"


def test_add_falls_back_to_unverified_when_uniqueness_fails(tmp_path: Path) -> None:
    outcome = add_row(tmp_path, page_of(), trim_left=1, pick=2)  # ambiguous, no context
    assert outcome.status == "unverified"
    entry = load_golden(tmp_path / "golden", root=tmp_path)  # must still load
    row = entry.entries[0].expected["seats.total"]
    assert row.status == "unverified"
    assert row.value is None and row.unit is None
    assert row.quote == "B.Sc. Medical Laboratory Technology 20"
    assert "occurs 2 times" in (row.note or "")
    assert entry.verified_rows == 0


def test_add_falls_back_to_unverified_when_context_is_mislabelled(tmp_path: Path) -> None:
    outcome = add_row(
        tmp_path,
        page_of(),
        value_raw="A.Y. 2024-2025",
        printed_value="A.Y. 2024-2025",
        predicate="seats.intake_year",
        unit=None,
        context="College A Institute",  # quote is already unique without it
        row_label="SEAT MATRIX FOR THE A.Y. 2024-2025",
        column_header="Year",
        pick=1,
    )
    assert outcome.status == "unverified"
    assert "already globally unique" in outcome.reasons[0]
    entry = load_golden(tmp_path / "golden", root=tmp_path)
    row = entry.entries[0].expected["seats.intake_year"]
    assert row.status == "unverified" and row.value is None


def test_add_merges_into_existing_entry_and_keeps_one_fixture(tmp_path: Path) -> None:
    add_row(
        tmp_path,
        page_of(),
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        pick=2,
    )
    outcome = add_row(
        tmp_path,
        page_of(),
        predicate="seats.intake_year",
        value_raw="A.Y. 2024-2025",
        printed_value="A.Y. 2024-2025",
        unit=None,
        entry_id="college-a-institute",
        pick=1,
    )
    assert outcome.status == "verified"
    entry = load_golden(tmp_path / "golden", root=tmp_path)
    assert sorted(entry.entries[0].expected) == ["seats.intake_year", "seats.total"]
    fixtures = list((tmp_path / "golden" / "fixtures").glob("*.txt"))
    assert len(fixtures) == 1


def test_add_refuses_a_duplicate_predicate_without_force(tmp_path: Path) -> None:
    add_row(
        tmp_path,
        page_of(),
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        pick=2,
    )
    with pytest.raises(CuratorError, match="already exists"):
        add_row(
            tmp_path,
            page_of(),
            context="College A Institute",
            row_label="1. B.Sc. Medical Laboratory Technology",
            column_header="Intake",
            pick=2,
        )


def test_add_force_replaces_a_predicate_on_a_generated_entry(tmp_path: Path) -> None:
    add_row(
        tmp_path,
        page_of(),
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        pick=2,
    )
    outcome = add_row(
        tmp_path,
        page_of(),
        value_raw="30",
        printed_value="30",
        force=True,
        # '30' is unique on the page, so no context: unique quotes reject a context.
    )
    assert outcome.status == "verified"
    assert outcome.proof.disambiguation == "unique"
    entry = load_golden(tmp_path / "golden", root=tmp_path)
    assert entry.entries[0].expected["seats.total"].value == 30


def test_add_dry_run_writes_nothing(tmp_path: Path) -> None:
    outcome = add_row(tmp_path, page_of(), dry_run=True, pick=1)
    assert outcome.status == "verified"
    assert outcome.entry_path is None and outcome.fixture_path is None
    assert not (tmp_path / "golden").exists()


def test_add_rejects_table_claim_without_row_identity(tmp_path: Path) -> None:
    with pytest.raises(CuratorError, match="--row-label and --column-header"):
        add_row(tmp_path, page_of(), context="College A Institute", pick=2)
    with pytest.raises(CuratorError, match="--context"):
        add_row(tmp_path, page_of(), row_label="1. B.Sc. Medical Laboratory Technology", pick=2)


def test_add_refuses_a_second_fixture_on_one_entry(tmp_path: Path) -> None:
    add_row(
        tmp_path,
        page_of(),
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        pick=2,
    )
    other_page = page_of(RECORDS_TEXT.replace("2024-2025", "2025-2026"))
    with pytest.raises(CuratorError, match="exactly one fixture"):
        add_row(
            tmp_path,
            PageText(
                page=2,
                lines=other_page.lines,
                text=other_page.text,
                line_spans=other_page.line_spans,
            ),
            entry_id="college-a-institute",
            context="College A Institute",
            row_label="1. B.Sc. Medical Laboratory Technology",
            column_header="Intake",
            pick=1,
        )


def test_add_refuses_drifted_committed_fixture(tmp_path: Path) -> None:
    add_row(
        tmp_path,
        page_of(),
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        pick=2,
    )
    fixture = tmp_path / "golden" / "fixtures" / "college-a-institute_page01.txt"
    fixture.write_text(RECORDS_TEXT.replace("College B", "College C"), encoding="utf-8")
    with pytest.raises(CuratorError, match="differs from the regenerated page text"):
        add_row(
            tmp_path,
            page_of(),
            predicate="seats.cardiac",
            value_raw="20",
            entry_id="college-a-institute",
            pick=2,
        )


# ------------------------------------------------------------ check and list


def _write_golden(tmp_path: Path, entries: list[dict]) -> Path:
    golden = tmp_path / "golden"
    (golden / "programmes").mkdir(parents=True)
    (golden / "fixtures").mkdir()
    for index, entry in enumerate(entries):
        (golden / "programmes" / f"entry{index}.yaml").write_text(
            yaml.safe_dump(entry), encoding="utf-8"
        )
    return golden


def test_check_reports_duplicate_exact_rows_as_failures(tmp_path: Path) -> None:
    entry = {
        "id": "dup-source",
        "institution": "College A Institute",
        "programme": "B.Sc. MLT",
        "fixture": "golden/fixtures/shared.txt",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. MLT 20",
                "status": "verified",
            }
        },
    }
    golden = _write_golden(tmp_path, [entry, {**entry, "id": "dup-copy"}])
    (tmp_path / "golden" / "fixtures" / "shared.txt").write_text("B.Sc. MLT 20\n", encoding="utf-8")
    report = check_golden(golden, root=tmp_path)
    assert not report.ok
    assert any("duplicate predicate row" in problem for problem in report.problems)
    assert report.payload["duplicates"]["failures"]


def test_check_warns_without_failing_on_same_entity_across_documents(tmp_path: Path) -> None:
    shared = {
        "institution": "College A Institute",
        "programme": "B.Sc. MLT",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. MLT 20",
                "status": "verified",
            }
        },
    }
    first = {**shared, "id": "doc-2024", "fixture": "golden/fixtures/a.txt"}
    second = {**shared, "id": "doc-2025", "fixture": "golden/fixtures/b.txt"}
    golden = _write_golden(tmp_path, [first, second])
    (tmp_path / "golden" / "fixtures" / "a.txt").write_text("B.Sc. MLT 20\n", encoding="utf-8")
    (tmp_path / "golden" / "fixtures" / "b.txt").write_text("B.Sc. MLT 20\n", encoding="utf-8")
    report = check_golden(golden, root=tmp_path)
    assert report.ok  # different fixtures/contexts: legitimate re-publication
    assert report.payload["duplicates"]["warnings"]
    assert not report.payload["duplicates"]["failures"]


def test_check_reports_fixture_presence_and_family_breakdown(tmp_path: Path) -> None:
    entry = {
        "id": "family-entry",
        "institution": "College A Institute",
        "programme": "B.Sc. MLT",
        "family": "seat_matrix",
        "fixture": "golden/fixtures/a.txt",
        "expected": {
            "seats.total": {
                "value": 20,
                "unit": "count",
                "quote": "B.Sc. MLT 20",
                "status": "verified",
            },
            "salary.median": {"value": None, "unit": None, "quote": None, "status": "unverified"},
        },
    }
    golden = _write_golden(tmp_path, [entry])
    (tmp_path / "golden" / "fixtures" / "a.txt").write_text("B.Sc. MLT 20\n", encoding="utf-8")
    report = check_golden(golden, root=tmp_path)
    assert report.ok
    assert report.payload["verified_rows"] == 1
    assert report.payload["unverified_rows"] == 1
    assert report.payload["by_family"]["seat_matrix"] == {
        "entries": 1,
        "verified": 1,
        "unverified": 1,
    }
    fixture_row = report.payload["fixtures"][0]
    assert fixture_row["match"] is None  # no sidecar: recorded by hand, nothing to compare
    assert fixture_row["present"] is True


def test_check_flags_a_tampered_fixture_against_the_sidecar(tmp_path: Path) -> None:
    outcome = add_row(
        tmp_path,
        page_of(),
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        pick=2,
    )
    fixture = tmp_path / "golden" / "fixtures" / "college-a-institute_page01.txt"
    fixture.write_text(fixture.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    report = check_golden(tmp_path / "golden", root=tmp_path)
    assert not report.ok
    assert "edited after curation" in report.problems[0]
    assert outcome.fixture_sha256 != report.payload["fixtures"][0]["sha256"]


def test_check_surfaces_loader_problems(tmp_path: Path) -> None:
    report = check_golden(tmp_path / "does-not-exist", root=tmp_path)
    assert not report.ok
    assert any("does not exist" in problem for problem in report.problems)


def test_list_reports_status_and_what_each_row_gates(tmp_path: Path) -> None:
    add_row(
        tmp_path,
        page_of(),
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        pick=2,
    )
    payload = list_golden(tmp_path / "golden", root=tmp_path)
    assert payload["totals"] == {"entries": 1, "verified_rows": 1, "unverified_rows": 0}
    entry = payload["entries"][0]
    assert entry["measured"] is False
    assert "not replayed" in entry["entry_gate"]
    row = entry["rows"][0]
    assert row["status"] == "verified" and row["value"] == 20
    assert "row attribution" in row["gate"]
    assert row["traceability"] == "fixture"


def test_row_gate_labels_every_expectation() -> None:
    from fmd.eval.golden import GoldenValue

    assert "excluded" in row_gate(GoldenValue(status="unverified"))
    assert "abstention" in row_gate(
        GoldenValue(status="verified", expectation="abstain", absent_terms=["x"])
    )
    assert "rejection" in row_gate(
        GoldenValue(status="verified", expectation="reject", reason="computed")
    )
    assert "extraction accuracy" in row_gate(GoldenValue(status="verified", value=1, quote="q"))
