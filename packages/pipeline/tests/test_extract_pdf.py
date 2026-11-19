from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from fmd.acquire.canonical import TextBlock, build_canonical
from fmd.acquire.errors import TierUnavailable
from fmd.acquire.tiers import (
    AcquirePolicy,
    TierPolicy,
    probe,
    select_tier,
    t0_extract,
    t1_extract,
    t2_extract,
)
from fmd.acquire.types import ExtractedDoc, PageGeometry, PageSpan, ScanProfile
from fmd.extract.pdf import extract_document, extract_pdf, sample_for_adjudication

FIXTURES = Path(__file__).parent / "fixtures" / "extract"
PDF = FIXTURES / "placement_report.pdf"
HAS_PDFTEXT = importlib.util.find_spec("pdftext") is not None


def _doc(text: str, *, tier: str = "T0", tool: str = "stub") -> ExtractedDoc:
    block = TextBlock(page_no=1, text=text, bbox=(10.0, 10.0, 500.0, 20.0))
    canonical = build_canonical(
        [block],
        [PageGeometry(1, 600.0, 800.0, (0.0, 0.0, 600.0, 800.0), 0, 0)],
    )
    return ExtractedDoc(
        text=canonical.text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(canonical.text))],
        tables=[],
        tier=tier,
        tool=tool,
        canonical=canonical,
        pages=canonical.pages,
        blocks=(block,),
        bbox_granularity=canonical.granularity,
    )


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_born_digital_pdf_uses_pdftext_with_provenance() -> None:
    doc = t0_extract(PDF.read_bytes())

    assert doc.tier == "T0"
    assert doc.tool == "pdftext"
    assert "Eligible students: 120" in doc.text
    assert doc.canonical is not None
    assert doc.bbox_granularity in {"glyph", "span"}
    assert doc.fallback_from is None

    hits = doc.canonical.resolve("Eligible students: 120")
    assert len(hits) == 1
    assert hits[0].page_no == 1
    assert hits[0].bbox[0] >= 0.0 and hits[0].bbox[2] <= 600.0
    assert doc.page_box(1) == (0.0, 0.0, 612.0, 792.0)


@pytest.mark.skipif(not HAS_PDFTEXT, reason="pdftext is not installed; run `uv sync --all-extras`")
def test_short_born_digital_document_escalates_to_t1_but_text_survives() -> None:
    calls: list[str] = []

    def fake_t1(data: bytes) -> ExtractedDoc:
        calls.append("t1")
        return _doc("docling text", tier="T1", tool="docling")

    doc = extract_document(PDF.read_bytes(), content_type="application/pdf", t1=fake_t1)

    assert calls == ["t1"]
    assert doc.tier == "T1"


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


def test_sparse_devanagari_text_dispatches_to_t2() -> None:
    sparse = _doc("रिपोर्ट")
    profile = ScanProfile(
        page_count=1,
        char_count=7,
        chars_per_page=7.0,
        devanagari=True,
        tables_detected=False,
    )
    assert select_tier(profile, AcquirePolicy.load()) == "T2"

    def fake_t2(data: bytes) -> ExtractedDoc:
        return _doc("OCR text", tier="T2", tool="paddleocr-vl")

    doc = extract_pdf(b"ignored", t0=lambda _: sparse, t2=fake_t2)
    assert doc.tier == "T2"


def test_scanned_page_without_text_layer_dispatches_to_t2() -> None:
    profile = ScanProfile(
        page_count=2,
        char_count=0,
        chars_per_page=0.0,
        devanagari=False,
        tables_detected=False,
        image_only=True,
    )
    assert select_tier(profile) == "T2"


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
        raise TierUnavailable("ocr endpoint missing")

    with pytest.raises(TierUnavailable) as excinfo:
        extract_pdf(b"ignored", t0=lambda _: _doc(""), t2=unavailable)
    assert "T0 also produced no text" in str(excinfo.value)


def test_docling_import_is_lazy_and_failure_names_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    with pytest.raises(TierUnavailable) as excinfo:
        t1_extract(b"%PDF-1.4 nothing")
    assert "uv sync --all-extras" in str(excinfo.value)


def test_t2_is_interface_only_without_an_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Endpoint-backed T2 (qwen3vl) without FMD_T2_URL names the env var."""
    from tests.test_acquire_tiers import _policy_with_t2_backend

    monkeypatch.delenv("FMD_T2_URL", raising=False)
    with pytest.raises(TierUnavailable) as excinfo:
        t2_extract(b"%PDF-1.4", policy=_policy_with_t2_backend("qwen3vl"))
    assert "FMD_T2_URL" in str(excinfo.value)


def test_t2_rapidocr_is_the_registry_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry default T2 is the in-process RapidOCR engine of record."""
    from fmd.acquire.tiers import AcquirePolicy, t2_backend

    monkeypatch.delenv("FMD_T2_BACKEND", raising=False)
    assert t2_backend(AcquirePolicy.load()) == "rapidocr"


def test_t2_accepts_an_injected_backend() -> None:
    class Backend:
        def extract(self, data: bytes) -> dict:
            return {
                "text": "scanned text",
                "tables": [{"rows": [["a", "b"]], "page": 1}],
                "tool": "fake-paddleocr",
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
    monkeypatch.setitem(sys.modules, "pdftext.extraction", None)
    with pytest.raises(TierUnavailable) as excinfo:
        t0_extract(b"%PDF-1.4")
    assert "uv sync --all-extras" in str(excinfo.value)


def test_tier_policy_thresholds_come_from_the_registry() -> None:
    policy = TierPolicy.load()
    acquire = policy.acquire
    assert policy.min_chars_per_page == acquire.tiers.text_min_chars_per_page
    assert policy.adjudication_sample_rate == acquire.tiers.adjudication_sample_rate
    # OCR tiers return text: claim extraction is a text task on the local
    # model, never the optional hosted vision profile (`hard_pdf`).
    assert policy.profile_for("T2") == "bulk_extract"
    assert policy.profile_for("T2b") == "bulk_extract"
