from __future__ import annotations

from fmd.eval.metrics import (
    BBoxReport,
    PublishedNumber,
    bbox_coverage,
    offset_resolvability,
    provenance_coverage,
)

TEXT = "Programme: BSc MLT\nTotal fee: INR 50000\nMedian salary: INR 420000"
FEE_START = TEXT.index("INR 50000")
FEE_END = FEE_START + len("INR 50000")
SALARY_START = TEXT.index("INR 420000")
SALARY_END = SALARY_START + len("INR 420000")


def _number(**overrides) -> PublishedNumber:
    defaults = {
        "predicate": "fee.total",
        "value": 50000,
        "quote": "INR 50000",
        "char_start": FEE_START,
        "char_end": FEE_END,
        "text": TEXT,
        "bbox": (50.0, 100.0, 200.0, 120.0),
        "bboxes": ((50.0, 100.0, 200.0, 120.0),),
        "page_no": 1,
        "page_box": (0.0, 0.0, 600.0, 800.0),
        "bbox_granularity": "glyph",
        "document_tier": "T0",
    }
    defaults.update(overrides)
    return PublishedNumber(**defaults)


def test_bbox_coverage_is_one_when_every_box_is_inside_the_page() -> None:
    report = bbox_coverage([_number(), _number()])

    assert report.coverage == 1.0
    assert report.by_granularity == {"glyph": 2}
    assert report.gate_failed() is False
    assert report.as_dict()["covered"] == 2


def test_missing_bbox_fails_the_t0_gate() -> None:
    report = bbox_coverage([_number(), _number(bbox=None, bboxes=())])

    assert report.coverage == 0.5
    assert report.failures[0]["reason"] == "missing_bbox"
    assert report.gate_failed() is True


def test_bbox_outside_the_page_is_a_failure_not_a_warning() -> None:
    outside = (700.0, 100.0, 800.0, 120.0)
    report = bbox_coverage([_number(bbox=outside, bboxes=(outside,))])

    assert report.coverage == 0.0
    assert report.failures[0]["reason"] == "bbox_outside_page"


def test_missing_page_box_is_reported_honestly() -> None:
    report = bbox_coverage([_number(page_box=None)])

    assert report.coverage == 0.0
    assert report.failures[0]["reason"] == "missing_page_box"


def test_ocr_pages_are_reportable_but_not_gated() -> None:
    report = bbox_coverage([_number(document_tier="T2", bbox=None, bboxes=())])

    assert report.coverage == 0.0
    assert report.gate_failed() is False


def test_multi_bbox_number_requires_every_box_inside() -> None:
    second = (650.0, 100.0, 700.0, 120.0)
    report = bbox_coverage([_number(bboxes=((50.0, 100.0, 200.0, 120.0), second))])

    assert report.coverage == 0.0
    assert report.failures[0]["reason"] == "bbox_outside_page"


def test_offset_resolvability_is_split_by_granularity() -> None:
    report = offset_resolvability(
        [
            _number(),
            _number(
                predicate="placement.median_salary",
                value=420000,
                quote="INR 420000",
                char_start=SALARY_START,
                char_end=SALARY_END,
                bbox_granularity="glyph",
            ),
            _number(
                predicate="fee.total",
                value=50000,
                quote="INR 99999",
                char_start=FEE_START,
                char_end=FEE_END,
                bbox_granularity="block",
                document_tier="T2",
            ),
        ]
    )

    assert report.resolvability == 2 / 3
    assert report.by_granularity["glyph"].resolvability == 1.0
    assert report.by_granularity["block"].resolvability == 0.0


def test_provenance_report_includes_bbox_and_offset_sections() -> None:
    report = provenance_coverage([_number()])
    payload = report.as_dict()

    assert payload["coverage"] == 1.0
    assert payload["bbox_coverage"]["coverage"] == 1.0
    assert payload["offset_resolvability"]["resolvability"] == 1.0
    assert payload["bbox_coverage"]["gate_tiers"] == ["T0", "T1"]


def test_published_number_reads_bbox_from_claim_qualifier() -> None:
    claim = {
        "predicate": "fee.total",
        "value": 50000,
        "quote": "INR 50000",
        "char_start": FEE_START,
        "char_end": FEE_END,
        "qualifier": {
            "bboxes": [
                {
                    "bbox": [50.0, 100.0, 200.0, 120.0],
                    "page_no": 1,
                    "bbox_granularity": "glyph",
                }
            ],
            "bbox": [50.0, 100.0, 200.0, 120.0],
            "page_no": 1,
            "bbox_granularity": "glyph",
        },
    }
    number = PublishedNumber.from_claim(claim, text=TEXT)

    assert number.bbox == (50.0, 100.0, 200.0, 120.0)
    assert number.page_no == 1
    assert number.bbox_granularity == "glyph"


def test_empty_bbox_report_is_vacuous_not_failing() -> None:
    report = BBoxReport()
    assert report.coverage is None
    assert report.gate_failed() is False


def test_pipeline_attaches_bboxes_without_collapsing_a_multi_span_quote() -> None:
    from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
    from fmd.acquire.types import ExtractedDoc, PageSpan
    from fmd.db.repositories import ClaimCreate
    from fmd.extract.pipeline import _attach_bbox_provenance

    blocks = [
        TextBlock(page_no=1, text="Total fee:", bbox=(10.0, 10.0, 60.0, 20.0)),
        TextBlock(page_no=1, text="INR 50000", bbox=(10.0, 10.0, 60.0, 20.0)),
    ]
    canonical = build_canonical(
        blocks, [PageGeometry(1, 600.0, 800.0, (0.0, 0.0, 600.0, 800.0), 0, 0)]
    )
    doc = ExtractedDoc(
        text=canonical.text,
        page_map=[PageSpan(page=1, char_start=0, char_end=len(canonical.text))],
        tables=[],
        tier="T0",
        tool="pdftext",
        canonical=canonical,
        pages=canonical.pages,
        blocks=tuple(blocks),
        bbox_granularity=canonical.granularity,
    )
    quote = "Total fee:\nINR 50000"
    claim = ClaimCreate(
        extraction_id=0,
        entity_id=1,
        predicate="fee.total",
        value=50000,
        unit="INR",
        quote=quote,
        char_start=0,
        char_end=len(quote),
    )

    adjusted = _attach_bbox_provenance(claim, doc)

    assert adjusted.qualifier is not None
    assert len(adjusted.qualifier["bboxes"]) == 2
    assert adjusted.qualifier["bbox_multi_span"] is True
    assert adjusted.qualifier["bbox_granularity"] == "span"
