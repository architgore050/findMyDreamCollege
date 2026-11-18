"""CLI tests for `fmd golden add/check/list` against hand-built PDFs.

No committed binaries: each test builds a minimal deterministic PDF (raw PDF
syntax, computed xref) that pdftext reads exactly like a real seat-matrix
document. Tests that need pdftext skip when the extraction extra is absent,
mirroring test_extract_pdf.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from fmd.cli import main
from fmd.eval.golden import REPO_ROOT, load_golden

HAS_PDFTEXT = importlib.util.find_spec("pdftext") is not None

PAGE_ONE = [
    (200, "GOVT STANLEY SEAT MATRIX A.Y. 2024-2025"),
    (72, "1  College A Institute"),
    (180, "1. B.Sc. Medical Laboratory Technology 20"),
    (180, "2. B.Sc Cardiac Technology 20"),
    (72, "2  College B Institute"),
    (180, "1. B.Sc. Medical Laboratory Technology 20"),
    (180, "2. B.Sc Cardiac Technology 30"),
]
PAGE_TWO = [(72, "APPENDIX: PAGE TWO")]


def build_pdf(pages: list[list[tuple[int, str]]]) -> bytes:
    """A minimal deterministic PDF: one Helvetica text object per line."""
    objects: list[bytes] = []
    n_pages = len(pages)
    kids = " ".join(f"{3 + i} 0 R" for i in range(n_pages))
    content_nos = list(range(3 + n_pages, 3 + 2 * n_pages))
    font_no = 3 + 2 * n_pages
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i in range(n_pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_nos[i]} 0 R "
            f"/Resources << /Font << /F1 {font_no} 0 R >> >> >>".encode()
        )
    for lines in pages:
        parts = ["BT /F1 12 Tf"]
        for index, (x, line) in enumerate(lines):
            y = 720 - 20 * index
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            parts.append(f"1 0 0 1 {x} {y} Tm ({escaped}) Tj")
        parts.append("ET")
        stream = " ".join(parts).encode("latin-1")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def source_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "seat-matrix.pdf"
    path.write_bytes(build_pdf([PAGE_ONE, PAGE_TWO]))
    return path


def add_args(source: Path, tmp_path: Path, **flags: str) -> list[str]:
    flags.setdefault("page", "1")
    argv = [
        "golden",
        "add",
        "--source",
        str(source),
        "--institution",
        "College A Institute",
        "--programme",
        "B.Sc. MLT",
        "--golden-dir",
        str(tmp_path / "golden"),
        "--root",
        str(tmp_path),
    ]
    for name, value in flags.items():
        argv += [f"--{name.replace(chr(95), '-')}"]
        if value:
            argv += [value]
    return argv


def write_predicate(argv: list[str], predicate: str, value: str, **extra: str) -> list[str]:
    return (
        argv
        + ["--predicate", predicate, "--value", value]
        + [
            part
            for name, flag_value in extra.items()
            for part in ([f"--{name.replace('_', '-')}"] + ([flag_value] if flag_value else []))
        ]
    )


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_typed_quote_is_rejected(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path), "seats.total", "20", quote="1. B.Sc. MLT 20"
    )
    assert main(argv) == 2
    assert "typed quote is rejected" in capsys.readouterr().err
    assert not (tmp_path / "golden").exists()


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_unique_quote_writes_a_verified_entry(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.intake_year",
        "A.Y. 2024-2025",
        printed_value="A.Y. 2024-2025",
        source_url="https://example.invalid/seat-matrix.pdf",
    )
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "status: verified" in out
    assert "sha256" in out  # source provenance is printed
    golden = load_golden(tmp_path / "golden", root=tmp_path)
    assert golden.verified_rows == 1 and golden.unverified_rows == 0
    entry = golden.entries[0]
    assert entry.source_url == "https://example.invalid/seat-matrix.pdf"
    assert entry.retrieved_at is not None
    fixture = tmp_path / "golden" / "fixtures" / "college-a-institute_page01.txt"
    assert fixture.exists()
    sidecar = json.loads(fixture.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["source_sha256"]
    assert sidecar["pages"] == 2
    assert sidecar["tool"] == "pdftext"
    text = fixture.read_text(encoding="utf-8")
    assert "1 College A Institute" in text and "\n2 College B Institute" in text


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_uniqueness_failure_falls_back_to_unverified(
    source_pdf: Path, tmp_path: Path, capsys
) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        trim_left="1",
        pick="2",
    )  # candidates: title, MLT-A, Cardiac-A, MLT-B (page-1 needle '20')
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert "status: unverified" in captured.out
    assert "occurs" in captured.err
    golden = load_golden(tmp_path / "golden", root=tmp_path)
    row = golden.entries[0].expected["seats.total"]
    assert row.status == "unverified"
    assert row.value is None
    assert row.quote == "B.Sc. Medical Laboratory Technology 20"


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_row_context_proof_writes_context_row(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        trim_left="1",
        pick="2",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
        value_cell="20",
    )
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "disambiguation: context_row" in out
    golden = load_golden(tmp_path / "golden", root=tmp_path)
    assert golden.as_dict()["row_context_verified_rows"] == 1
    row = golden.entries[0].expected["seats.total"]
    assert row.value == 20 and row.status == "verified"
    assert row.context == "College A Institute"
    assert row.row is not None
    assert row.row.column_header == "Intake" and row.row.table_page == 1


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_unique_quote_with_context_is_refused(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.intake_year",
        "A.Y. 2024-2025",
        printed_value="A.Y. 2024-2025",
        pick="1",
        context="College A Institute",
        row_label="GOVT STANLEY SEAT MATRIX A.Y. 2024-2025",
        column_header="Year",
    )
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert "already globally unique" in captured.err
    golden = load_golden(tmp_path / "golden", root=tmp_path)
    assert golden.unverified_rows == 1 and golden.verified_rows == 0


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_merges_a_second_predicate_into_the_entry(
    source_pdf: Path, tmp_path: Path, capsys
) -> None:
    first = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        trim_left="1",
        pick="2",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
    )
    assert main(first) == 0
    second = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.intake_year",
        "A.Y. 2024-2025",
        printed_value="A.Y. 2024-2025",
        pick="1",
        entry_id="college-a-institute",
    )
    assert main(second) == 0
    golden = load_golden(tmp_path / "golden", root=tmp_path)
    assert sorted(golden.entries[0].expected) == ["seats.intake_year", "seats.total"]
    assert len(list((tmp_path / "golden" / "fixtures").glob("*.txt"))) == 1
    assert (
        main(["golden", "check", "--golden", str(tmp_path / "golden"), "--root", str(tmp_path)])
        == 0
    )


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_refuses_a_duplicate_predicate_without_force(
    source_pdf: Path, tmp_path: Path, capsys
) -> None:
    first = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        trim_left="1",
        pick="2",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
    )
    assert main(first) == 0
    assert main(first) == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_refuses_to_merge_into_a_hand_curated_entry(
    source_pdf: Path, tmp_path: Path, capsys
) -> None:
    programmes = tmp_path / "golden" / "programmes"
    programmes.mkdir(parents=True)
    entry = {
        "id": "college-a-institute",
        "institution": "College A Institute",
        "programme": "B.Sc. MLT",
        "fixture": "golden/fixtures/college-a-institute_page01.txt",
        "expected": {},
    }
    (programmes / "college-a-institute.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        entry_id="college-a-institute",
        pick="2",
    )
    assert main(argv) == 2
    assert "not generated by" in capsys.readouterr().err


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_fixture_bytes_are_deterministic(source_pdf: Path, tmp_path: Path) -> None:
    from fmd.eval.pdfpage import extract_page, sha256_bytes

    page_a, _ = extract_page(str(source_pdf), 1)
    page_b, _ = extract_page(str(source_pdf), 1)
    assert page_a.text == page_b.text
    assert page_a.lines == page_b.lines

    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        trim_left="1",
        pick="2",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
    )
    first_dir = tmp_path / "run1"
    second_dir = tmp_path / "run2"
    assert main(argv + ["--golden-dir", str(first_dir / "golden"), "--root", str(first_dir)]) == 0
    assert main(argv + ["--golden-dir", str(second_dir / "golden"), "--root", str(second_dir)]) == 0
    fixture_a = (first_dir / "golden" / "fixtures" / "college-a-institute_page01.txt").read_bytes()
    fixture_b = (second_dir / "golden" / "fixtures" / "college-a-institute_page01.txt").read_bytes()
    assert sha256_bytes(fixture_a) == sha256_bytes(fixture_b)
    assert b"College B Institute" in fixture_a  # page text is committed as extracted


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_check_detects_a_tampered_fixture(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        trim_left="1",
        pick="2",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
    )
    assert main(argv) == 0
    fixture = tmp_path / "golden" / "fixtures" / "college-a-institute_page01.txt"
    fixture.write_bytes(fixture.read_bytes() + b"tampered\n")
    assert (
        main(["golden", "check", "--golden", str(tmp_path / "golden"), "--root", str(tmp_path)])
        == 1
    )
    captured = capsys.readouterr()
    assert "TAMPERED" in captured.out
    assert "does not match the recorded" in captured.err


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_check_passes_on_a_curated_directory(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        trim_left="1",
        pick="2",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
    )
    assert main(argv) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "golden",
                "check",
                "--json",
                "--golden",
                str(tmp_path / "golden"),
                "--root",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["verified_rows"] == 1
    assert payload["fixtures"][0]["match"] is True
    assert payload["context_checks"][0]["ok"] is True
    assert "seat_matrix" not in payload["by_family"]
    assert "(no family: loader checks only)" in payload["by_family"]


def test_check_passes_on_the_committed_golden_set(capsys) -> None:
    assert main(["golden", "check", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["verified_rows"] == 16
    assert payload["unverified_rows"] == 1
    unrecorded = {item["fixture"] for item in payload["fixtures"] if item["match"] is None}
    assert "data/golden/fixtures/tnmgrmu_ahs_ug_seat_matrix_2024_25_page1.tsv" in unrecorded


def test_list_reports_the_committed_set_status_and_gates(capsys) -> None:
    assert main(["golden", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_id = {entry["id"]: entry for entry in payload["entries"]}
    placement = by_id["fixture-placement-report-2024-25"]
    assert placement["measured"] is True
    assert "canary replay" in placement["entry_gate"]
    seat_row = next(
        row
        for row in by_id["tnmgrmu-ahs-ug-seat-matrix-2024-25"]["rows"]
        if row["predicate"] == "seats.intake"
    )
    assert "row attribution" in seat_row["gate"]
    unverified = next(row for row in placement["rows"] if row["predicate"] == "fee.tuition")
    assert "excluded" in unverified["gate"]
    assert unverified["status"] == "unverified"


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_list_output_after_curation(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        unit="count",
        trim_left="1",
        pick="2",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
    )
    assert main(argv) == 0
    capsys.readouterr()
    assert (
        main(["golden", "list", "--golden", str(tmp_path / "golden"), "--root", str(tmp_path)]) == 0
    )
    out = capsys.readouterr().out
    assert "college-a-institute" in out
    assert "seats.total: verified/value value=20 unit=count" in out
    assert "row attribution" in out
    assert "legend:" in out


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_page_out_of_range_and_missing_value(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(add_args(source_pdf, tmp_path), "seats.total", "20", page="9")
    assert main(argv) == 2
    assert "out of range" in capsys.readouterr().err
    argv = write_predicate(
        add_args(source_pdf, tmp_path), "seats.hostel_fee", "999999", printed_value="not printed"
    )
    assert main(argv) == 2
    assert "does not occur on page" in capsys.readouterr().err


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_pick_out_of_range_lists_candidates(source_pdf: Path, tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(source_pdf, tmp_path), "seats.total", "20", printed_value="20", pick="99"
    )
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "--pick 99 is out of range" in err
    assert "candidate line(s)" in err


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_prompted_interactively(source_pdf: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    answers = iter(
        [
            "count",  # unit (programme/source/page/predicate/value already given)
            "2",  # candidate pick among the needle '20' lines
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    argv = write_predicate(
        add_args(source_pdf, tmp_path),
        "seats.total",
        "20",
        printed_value="20",
        trim_left="1",
        interactive="",
        context="College A Institute",
        row_label="1. B.Sc. Medical Laboratory Technology",
        column_header="Intake",
    )
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "status: verified" in out
    golden = load_golden(tmp_path / "golden", root=tmp_path)
    assert golden.verified_rows == 1


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_add_from_an_unreachable_url_fails_cleanly(tmp_path: Path, capsys) -> None:
    argv = write_predicate(
        add_args(tmp_path / "missing.pdf", tmp_path),
        "seats.total",
        "20",
        source="http://127.0.0.1:9/no-such-document.pdf",
    )
    assert main(argv) == 2
    assert "could not download" in capsys.readouterr().err
    assert not (tmp_path / "golden").exists()


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_fetch_pdf_honours_robots_disallow(monkeypatch) -> None:
    import httpx

    from fmd.eval import pdfpage

    class _Response:
        def __init__(self, status_code: int, text: str = "") -> None:
            self.status_code = status_code
            self.text = text

    def fake_get(url, **_kwargs):
        if url.endswith("/robots.txt"):
            return _Response(200, "User-Agent: *\nDisallow: /restricted/\n")
        raise AssertionError("the document fetch must not run when robots disallows it")

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(Exception, match="robots.txt disallows"):
        pdfpage.fetch_pdf("https://example.invalid/restricted/seat-matrix.pdf")


def test_fetch_pdf_ignores_an_unreadable_robots_policy(monkeypatch) -> None:
    import httpx

    from fmd.eval import pdfpage

    def fake_get(url, **_kwargs):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(Exception, match="could not download"):
        pdfpage.fetch_pdf("http://127.0.0.1:9/no-such-document.pdf")


class _NoTty:
    def isatty(self) -> bool:
        return False


def test_golden_add_without_required_flags_exits_non_zero(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", _NoTty())  # never prompt in tests
    assert main(["golden", "add"]) == 2
    err = capsys.readouterr().err
    assert "missing required inputs" in err


def test_repo_root_fixture_recordings_stay_repo_relative(tmp_path: Path) -> None:
    assert (REPO_ROOT / "data" / "golden" / "programmes" / "_TEMPLATE.yaml").exists()
