from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from fmd.extract.pdf import (
    ExtractedDoc,
    PageSpan,
    ScanProfile,
    TierPolicy,
    TierUnavailable,
    extract_document,
    extract_pdf,
    probe,
    sample_for_adjudication,
    select_tier,
    t0_extract,
    t1_extract,
    t2_extract,
)

FIXTURES = Path(__file__).parent / "fixtures" / "extract"
PDF = FIXTURES / "placement_report.pdf"
HAS_PYPDFIUM = importlib.util.find_spec("pypdfium2") is not None


def _doc(text: str, *, tier: str = "T0", tool: str = "stub") -> ExtractedDoc:
    return ExtractedDoc(
        text=text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(text))],
        tables=[],
        tier=tier,
        tool=tool,
    )


@pytest.mark.skipif(
    not HAS_PYPDFIUM, reason="pypdfium2 is not installed; run `uv sync --extra extract`"
)
def test_born_digital_pdf_uses_t0_and_spans_resolve() -> None:
    doc = extract_document(PDF.read_bytes(), content_type="application/pdf")

    assert doc.tier == "T0"
    assert doc.tool == "pypdfium2"
    assert "Eligible students: 120" in doc.text
    span = doc.page_map[0]
    assert doc.text[span.char_start : span.char_end] == doc.text
    assert doc.fallback_from is None


def test_text_content_type_is_not_treated_as_pdf() -> None:
    doc = extract_document(b"plain evidence text", content_type="text/plain")
    assert doc.tool == "text-decoder"
    assert doc.text == "plain evidence text"


def test_html_scripts_are_stripped() -> None:
    data = b"<html><head><script>var x = 1;</script></head><body><p>Seats: 60</p></body></html>"
    doc = extract_document(data, content_type="text/html")
    assert "Seats: 60" in doc.text
    assert "var x" not in doc.text


def test_table_like_text_dispatches_to_t1() -> None:
    table_text = "\n".join(
        [
            "Programme              Seats   Year",
            "BSc MLT                 60      2025",
            "BSc MLT (lateral)       12      2025",
            "MSc MLT                 10      2025",
        ]
    )
    calls: list[str] = []

    def fake_t1(data: bytes) -> ExtractedDoc:
        calls.append("t1")
        return _doc("docling text", tier="T1", tool="docling")

    doc = extract_pdf(b"ignored", t0=lambda _: _doc(table_text), t1=fake_t1)

    assert calls == ["t1"]
    assert doc.tier == "T1"
    profile = probe(_doc(table_text))
    assert profile.tables_detected is True
    assert select_tier(profile) == "T1"


def test_empty_text_with_devanagari_headers_dispatches_to_t2() -> None:
    sparse = _doc("रिपोर्ट")
    policy = TierPolicy.load()
    profile = ScanProfile(
        page_count=1,
        char_count=7,
        chars_per_page=7.0,
        devanagari=True,
        tables_detected=False,
    )
    assert select_tier(profile, policy) == "T2"

    def fake_t2(data: bytes) -> ExtractedDoc:
        return _doc("OCR text", tier="T2", tool="mineru")

    doc = extract_pdf(b"ignored", t0=lambda _: sparse, t2=fake_t2)
    assert doc.tier == "T2"


def test_scanned_page_without_devanagari_dispatches_to_t1() -> None:
    profile = ScanProfile(
        page_count=2,
        char_count=4,
        chars_per_page=2.0,
        devanagari=False,
        tables_detected=False,
    )
    assert select_tier(profile) == "T1"


def test_t1_unavailable_falls_back_to_t0_text_with_an_explicit_note() -> None:
    table_text = "\n".join(
        [
            "Programme              Seats   Year",
            "BSc MLT                 60      2025",
            "BSc MLT (lateral)       12      2025",
            "MSc MLT                 10      2025",
        ]
    )

    def unavailable(_: bytes) -> ExtractedDoc:
        raise TierUnavailable("docling is not installed")

    doc = extract_pdf(b"ignored", t0=lambda _: _doc(table_text), t1=unavailable)

    assert doc.tier == "T0"
    assert doc.fallback_from == "T1"
    assert "docling is not installed" in doc.notes[0]
    assert doc.text == table_text


def test_unavailable_selected_tier_with_no_t0_text_fails_loudly() -> None:
    def unavailable(_: bytes) -> ExtractedDoc:
        raise TierUnavailable("docling is not installed")

    with pytest.raises(TierUnavailable) as excinfo:
        extract_pdf(b"ignored", t0=lambda _: _doc(""), t1=unavailable)
    assert "T0 also produced no text" in str(excinfo.value)


def test_docling_import_is_lazy_and_failure_names_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "docling", None)
    with pytest.raises(TierUnavailable) as excinfo:
        t1_extract(b"%PDF-1.4 nothing")
    assert "uv sync --extra extract" in str(excinfo.value)


def test_t2_is_interface_only_without_an_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FMD_MINERU_URL", raising=False)
    with pytest.raises(TierUnavailable) as excinfo:
        t2_extract(b"%PDF-1.4")
    assert "FMD_MINERU_URL" in str(excinfo.value)


def test_t2_accepts_an_injected_backend() -> None:
    class Backend:
        def extract(self, data: bytes) -> dict:
            return {
                "text": "scanned text",
                "tables": [{"rows": [["a", "b"]], "page": 1}],
                "tool": "fake-mineru",
            }

    doc = t2_extract(b"x", backend=Backend())
    assert doc.tier == "T2"
    assert doc.text == "scanned text"
    assert doc.tables[0].rows == [["a", "b"]]


def test_adjudication_sample_is_deterministic_and_bounded() -> None:
    items = [f"claim-{index}" for index in range(10_000)]
    first = sample_for_adjudication(items, rate=0.01)
    second = sample_for_adjudication(items, rate=0.01)
    assert first == second
    assert 50 <= len(first) <= 150
    assert sample_for_adjudication(items, rate=0.0) == []


def test_t0_requires_the_extract_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    with pytest.raises(TierUnavailable) as excinfo:
        t0_extract(b"%PDF-1.4")
    assert "uv sync --extra extract" in str(excinfo.value)
