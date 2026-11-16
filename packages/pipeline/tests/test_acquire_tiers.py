from __future__ import annotations

import sys
from dataclasses import replace

import pytest

from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
from fmd.acquire.errors import TierUnavailable
from fmd.acquire.tiers import (
    AcquirePolicy,
    indic_self_check,
    plan_tiers,
    probe,
    run_tiers,
    t0_extract,
    t1_extract,
    t2_extract,
    t2b_extract,
)
from fmd.acquire.types import ExtractedDoc, PageSpan, ScanProfile, Table


def _policy() -> AcquirePolicy:
    return AcquirePolicy.load()


def _doc(
    text: str,
    *,
    tier: str = "T0",
    blocks: list[TextBlock] | None = None,
    tool: str = "stub",
    tables: list[Table] | None = None,
    two_column: bool = False,
) -> ExtractedDoc:
    if blocks is None:
        blocks = [TextBlock(page_no=1, text=text, bbox=(10, 10, 500, 20))]
    page = PageGeometry(
        page_no=1,
        width=600.0,
        height=800.0,
        bbox=(0.0, 0.0, 600.0, 800.0),
        char_start=0,
        char_end=0,
    )
    canonical = build_canonical(blocks, [page])
    return ExtractedDoc(
        text=canonical.text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(canonical.text))],
        tables=tables or [],
        tier=tier,
        tool=tool,
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity=canonical.granularity,
        two_column=two_column,
    )


LONG_TEXT = (
    "Programme: BSc Medical Laboratory Technology. Sanctioned intake: 60 seats. "
    "Duration: four years including a 1800-hour internship in semesters seven and eight. "
    "Eligible students: 120. Students placed: 96. Median salary INR 420000 per annum. "
    "The report counts internships separately from employment outcomes."
)


def test_clean_text_layer_routes_to_t0_only() -> None:
    calls: list[str] = []
    doc = _doc(LONG_TEXT)

    result = run_tiers(
        b"ignored",
        policy=_policy(),
        t0=lambda _: doc,
        t1=lambda _: calls.append("T1") or _doc("x", tier="T1"),
        t2=lambda _: calls.append("T2") or _doc("x", tier="T2"),
    )

    assert result.tier == "T0"
    assert calls == []
    assert plan_tiers(probe(doc)).tier == "T0"


def test_table_layout_routes_to_t1() -> None:
    profile = ScanProfile(
        page_count=1,
        char_count=400,
        chars_per_page=400.0,
        devanagari=False,
        tables_detected=True,
    )
    assert plan_tiers(profile).tier == "T1"


def test_two_column_layout_routes_to_t1() -> None:
    doc = replace(_doc(LONG_TEXT), two_column=True)
    assert plan_tiers(probe(doc)).tier == "T1"


def test_image_only_page_routes_to_t2() -> None:
    profile = ScanProfile(
        page_count=2,
        char_count=0,
        chars_per_page=0.0,
        devanagari=False,
        tables_detected=False,
        image_only=True,
    )
    assert plan_tiers(profile).tier == "T2"


def test_devanagari_routes_to_t2_even_with_a_text_layer() -> None:
    doc = _doc(LONG_TEXT + " शुल्क और पात्रता की जानकारी।")
    assert plan_tiers(probe(doc)).tier == "T2"


def test_low_table_confidence_escalates_to_t2() -> None:
    profile = ScanProfile(
        page_count=1,
        char_count=500,
        chars_per_page=500.0,
        devanagari=False,
        tables_detected=True,
        table_confidence=0.0,
    )
    assert plan_tiers(profile).tier == "T2"


def test_indic_self_check_scores_mojibake_low_and_hindi_high() -> None:
    assert indic_self_check("कुल शुल्क: ₹५०,००० प्रति वर्ष") > 0.5
    assert indic_self_check("क broken latin mojibake à¤•à¥") < 0.5


def test_t2_escalates_to_t2b_when_indic_self_check_fails() -> None:
    seen: list[str] = []
    t2_doc = _doc("शुल्क Total fees 50000 payable per year on registration", tier="T2")
    t2b_doc = _doc("कुल शुल्क ₹५०,००० प्रति वर्ष", tier="T2b")

    result = run_tiers(
        b"ignored",
        policy=_policy(),
        t0=lambda _: _doc(""),
        t1=lambda _: _doc("", tier="T1"),
        t2=lambda _: seen.append("T2") or t2_doc,
        t2b=lambda _: seen.append("T2b") or t2b_doc,
    )

    assert seen == ["T2", "T2b"]
    assert result.tier == "T2b"


def test_t2b_unavailable_routes_to_human_review() -> None:
    def unavailable(_: bytes) -> ExtractedDoc:
        raise TierUnavailable("no endpoint")

    result = run_tiers(
        b"ignored",
        policy=_policy(),
        t0=lambda _: _doc(""),
        t2=lambda _: _doc("शुल्क Total fees 50000 payable per year", tier="T2"),
        t2b=unavailable,
    )

    assert result.tier == "T3"
    assert result.needs_review is True
    assert "T2b unavailable" in result.notes[0]


def test_ocr_repetition_loop_routes_to_t3_before_claims() -> None:
    blocks = [
        TextBlock(page_no=1, text=LONG_TEXT, bbox=(10, 10, 500, 20)),
        TextBlock(page_no=1, text="Fee 50000 " * 71, bbox=(10, 30, 500, 40)),
    ]
    calls: list[str] = []
    result = run_tiers(
        b"ignored",
        policy=_policy(),
        t0=lambda _: _doc(""),
        t1=lambda _: calls.append("T1") or _doc("x", tier="T1"),
        t2=lambda _: calls.append("T2") or _doc("", tier="T2", blocks=blocks),
    )

    assert result.tier == "T3"
    assert result.ocr_anomaly is True
    assert result.needs_review is True
    assert result.guard_trips_by_name() == {"repetition_loop": 1}
    assert calls == ["T2"]


def test_t1_long_table_is_not_refused() -> None:
    table = "\n".join(
        f"BSc MLT year {row % 4 + 1} tuition {45000 + row * 315} lab {2500 + row * 17}"
        for row in range(20)
    )
    t1_blocks = [
        TextBlock(page_no=1, text="Fees payable at admission", bbox=(10, 10, 500, 20)),
        TextBlock(page_no=1, text="See table below", bbox=(10, 30, 500, 40)),
        TextBlock(page_no=1, text=table, bbox=(10, 50, 500, 700)),
    ]
    calls: list[str] = []

    result = run_tiers(
        b"ignored",
        policy=_policy(),
        t0=lambda _: _doc(LONG_TEXT + "\n50000 60 4\n45000 60 4\n40000 60 4"),
        t1=lambda _: calls.append("T1") or _doc("", tier="T1", blocks=t1_blocks),
        t2=lambda _: calls.append("T2") or _doc("x", tier="T2"),
    )

    assert result.tier == "T1"
    assert result.ocr_anomaly is False
    assert calls == ["T1"]


# --------------------------------------------------------------------------- lazy imports


def test_t0_missing_pdftext_names_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pdftext.extraction", None)
    with pytest.raises(TierUnavailable) as excinfo:
        t0_extract(b"%PDF-1.4")
    assert "uv sync --all-extras" in str(excinfo.value)


def test_t1_missing_docling_names_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    with pytest.raises(TierUnavailable) as excinfo:
        t1_extract(b"%PDF-1.4")
    assert "uv sync --all-extras" in str(excinfo.value)


def test_t2_without_endpoint_is_an_explicit_tier_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FMD_T2_URL", raising=False)
    with pytest.raises(TierUnavailable) as excinfo:
        t2_extract(b"%PDF-1.4")
    assert "FMD_T2_URL" in str(excinfo.value)


def test_t2b_without_endpoint_is_an_explicit_tier_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FMD_T2B_URL", raising=False)
    with pytest.raises(TierUnavailable) as excinfo:
        t2b_extract(b"%PDF-1.4")
    assert "FMD_T2B_URL" in str(excinfo.value)
    assert _policy().backend("T2b").model in str(excinfo.value)


# --------------------------------------------------------------------------- OCR contract


def test_t2_blocks_carry_block_boxes_and_page_geometry() -> None:
    class Backend:
        def extract(self, data: bytes) -> dict:
            return {
                "pages": [{"page": 1, "width": 600, "height": 800, "bbox": [0, 0, 600, 800]}],
                "blocks": [
                    {
                        "page": 1,
                        "block_order": 0,
                        "block_bbox": [10, 10, 200, 30],
                        "block_content": "Total fee INR 50000",
                    },
                    {
                        "page": 1,
                        "block_order": 1,
                        "block_bbox": [10, 40, 200, 60],
                        "block_content": "per year",
                    },
                ],
                "tool": "fake-paddleocr",
            }

    doc = t2_extract(b"x", backend=Backend(), policy=_policy())

    assert doc.tier == "T2"
    assert doc.bbox_granularity == "block"
    hits = doc.canonical.resolve("Total fee INR 50000") if doc.canonical else []
    assert hits and hits[0].granularity == "block"
    assert hits[0].bbox == (10.0, 10.0, 200.0, 30.0)


def test_t1_docling_document_provides_table_cells_and_uncertainty() -> None:
    class Box:
        def __init__(self, left: float, top: float, right: float, bottom: float) -> None:
            self.l, self.t, self.r, self.b = left, top, right, bottom

        def to_top_left_origin(self, page_height: float) -> Box:
            return self

    class Prov:
        page_no = 1
        bbox = Box(10, 10, 200, 30)

    class Item:
        def __init__(self, text: str, prov: Prov) -> None:
            self.text = text
            self.prov = [prov]
            self.data = None

    class Page:
        page_no = 1
        size = type("Size", (), {"width": 600.0, "height": 800.0})()

        def __init__(self) -> None:
            pass

    cells = [
        {
            "start_row_offset_idx": 0,
            "end_row_offset_idx": 1,
            "start_col_offset_idx": 0,
            "end_col_offset_idx": 2,
            "row_span": 1,
            "col_span": 2,
            "text": "Fees",
            "column_header": True,
            "bbox": Box(10, 10, 200, 30),
        },
        {
            "start_row_offset_idx": 1,
            "end_row_offset_idx": 2,
            "start_col_offset_idx": 0,
            "end_col_offset_idx": 1,
            "row_span": 1,
            "col_span": 1,
            "text": "50000",
            "column_header": False,
            "bbox": Box(10, 40, 100, 60),
        },
    ]

    class TableData:
        table_cells = cells
        num_rows = 2
        num_cols = 2

    class TableItem:
        text = None
        prov = [Prov()]
        bbox = None

        def __init__(self) -> None:
            self.data = TableData()

    class Document:
        pages = {1: Page()}

        def iterate_items(self):
            return [(Item("Programme: BSc MLT", Prov()), 0), (TableItem(), 0)]

    class Converter:
        def convert(self, path: str):
            return type("Result", (), {"document": Document()})()

    doc = t1_extract(b"%PDF-1.4", converter=Converter(), policy=_policy())

    assert doc.tier == "T1"
    assert doc.bbox_granularity == "span"
    assert doc.tables[0].column_header_uncertain is True
    assert doc.table_confidence == 0.0
    assert "Programme: BSc MLT" in doc.text
    assert "50000" in doc.text
