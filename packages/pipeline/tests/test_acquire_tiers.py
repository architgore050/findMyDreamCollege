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


def test_offline_t1_without_artifacts_fails_fast_with_named_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline must never reach the Hub: no path => immediate TierUnavailable."""
    import time

    from fmd.acquire.errors import ArtifactsUnavailable

    monkeypatch.setenv("FMD_OFFLINE", "1")
    # Pinned so _enforce_offline_env()'s setdefault cannot leak Hub flags into
    # later tests through the real process environment.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.delenv("FMD_DOCLING_ARTIFACTS", raising=False)
    start = time.monotonic()
    with pytest.raises(ArtifactsUnavailable) as excinfo:
        t1_extract(b"%PDF-1.4")
    elapsed = time.monotonic() - start
    assert "artifacts_unavailable_offline" in str(excinfo.value)
    assert "FMD_DOCLING_ARTIFACTS" in str(excinfo.value)
    assert elapsed < 5.0


def test_offline_t1_with_empty_artifact_dir_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from fmd.acquire.errors import ArtifactsUnavailable

    monkeypatch.setenv("FMD_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("FMD_DOCLING_ARTIFACTS", str(tmp_path))
    with pytest.raises(ArtifactsUnavailable) as excinfo:
        t1_extract(b"%PDF-1.4")
    assert "artifacts_unavailable_offline" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


def test_offline_t1_artifact_gap_routes_to_t3_not_flat_t0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table question is not answered by T0 text just because the cache is gone."""
    from fmd.acquire.errors import ArtifactsUnavailable

    def unavailable(_: bytes) -> ExtractedDoc:
        raise ArtifactsUnavailable("artifacts_unavailable_offline: no cache")

    result = run_tiers(
        b"ignored",
        policy=_policy(),
        t0=lambda _: _doc(LONG_TEXT + "\n50000 60 4\n45000 60 4\n40000 60 4"),
        t1=unavailable,
    )

    assert result.tier == "T3"
    assert result.needs_review is True
    assert result.review_reason == "artifacts_unavailable_offline"
    assert result.fallback_from == "T1"
    assert result.as_review_item()["reasons"] == ["artifacts_unavailable_offline"]
    assert "T1 unavailable offline" in result.notes[0]


def test_t2_without_endpoint_is_an_explicit_tier_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An endpoint-backed T2 (qwen3vl) without FMD_T2_URL names the env var."""
    monkeypatch.delenv("FMD_T2_URL", raising=False)
    with pytest.raises(TierUnavailable) as excinfo:
        t2_extract(b"%PDF-1.4", policy=_policy_with_t2_backend("qwen3vl"))
    assert "FMD_T2_URL" in str(excinfo.value)


def test_t2_rapidocr_without_models_names_the_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process default (rapidocr) without models names the prefetch."""
    monkeypatch.delenv("FMD_RAPIDOCR_MODELS", raising=False)
    with pytest.raises(TierUnavailable) as excinfo:
        t2_extract(b"%PDF-1.4")
    assert "models_missing" in str(excinfo.value)
    assert "scripts/ocr_validate.py --prefetch" in str(excinfo.value)


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


# --------------------------------------------------------------------------- service failures


def test_t2_and_t2b_engines_come_from_the_registry() -> None:
    """T2 selects its engine from the matrix; T2b is the retained alternate."""
    policy = _policy()

    # Default on this host: in-process PP-OCR via the RapidOCR runner, with the
    # Devanagari rec model pinned as a first-class LangRec value.
    assert policy.backend("T2").backend == "rapidocr"
    assert policy.backend("T2").tool == "rapidocr"
    assert policy.backend("T2").device == "cpu"  # measured: llama-server resident
    models = policy.backend("T2").models
    assert models["det"] == "PP-OCRv6_det_small.onnx"
    assert models["rec"] == "devanagari_PP-OCRv5_rec_mobile.onnx"
    assert models["rec_keys"] == "ppocrv5_devanagari_dict.txt"
    assert models["lang_rec"] == "devanagari"
    assert models["sha256"]["rec"].startswith("d6f0a906")
    # Qwen3-VL stays fully wired: the Devanagari fallback and an alternate
    # backend, never the default, never removed.
    assert policy.backend("T2").devanagari_fallback == "qwen3vl"
    assert policy.backend("T2").service["backend"] == "qwen3vl"
    assert policy.backend("T2").service["served_name"] == "Qwen3-VL-8B-Instruct"
    assert policy.backend("T2").endpoint_env == "FMD_T2_URL"
    assert policy.backend("T2b").tool == "paddleocr-vl"
    assert policy.backend("T2b").endpoint_env == "FMD_T2B_URL"
    assert policy.backend("T2b").service["backend"] == "paddleocr"
    assert policy.backend("T2b").service["device"] == "cpu"


def _policy_with_t2_backend(name: str) -> AcquirePolicy:
    """The real registry with ``backends.T2.backend`` overridden (no mutation)."""
    from fmd.config import get_registry
    from fmd.config.models import AcquireBackend

    acquire = get_registry().acquire.model_copy(deep=True)
    spec = dict(acquire.backends["T2"].model_dump())
    spec["backend"] = name
    acquire.backends["T2"] = AcquireBackend(**spec)
    return AcquirePolicy.load(acquire=acquire)


@pytest.fixture
def qwen_t2_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve ``FMD_T2_BACKEND=qwen3vl`` for one test, registry cache included.

    ``run_tiers`` calls the real ``t2_extract`` without a policy argument (the
    tier loads the registry itself), so the backend is switched through the
    ``${FMD_T2_BACKEND:-rapidocr}`` interpolation and the registry cache is
    rebuilt — and rebuilt again on teardown so the default never leaks.
    """
    from fmd.config.loader import reset_registry_cache

    monkeypatch.setenv("FMD_T2_BACKEND", "qwen3vl")
    reset_registry_cache()
    yield
    monkeypatch.undo()
    reset_registry_cache()


@pytest.mark.parametrize("status", [429, 500, 503])
def test_ocr_service_http_error_is_tier_unavailable(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """429/500/503 from the OCR service is "unavailable", never an empty document."""
    import httpx

    from fmd.acquire import tiers as tiers_module

    def post(url: str, **kwargs: object) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            status, json={"detail": "backend_http_429: engine said no"}, request=request
        )

    monkeypatch.setenv("FMD_T2_URL", "http://ocr.test")
    monkeypatch.setattr(tiers_module.httpx, "post", post)

    with pytest.raises(TierUnavailable) as excinfo:
        t2_extract(
            b"%PDF-1.4", policy=_policy_with_t2_backend("qwen3vl")
        )
    assert "ocr.test" in str(excinfo.value)
    assert str(status) in str(excinfo.value)


def test_unavailable_vision_engine_degrades_loudly_not_silently(
    monkeypatch: pytest.MonkeyPatch, qwen_t2_backend: None
) -> None:
    """A page with usable T0 text falls back to T0 and carries the reason."""
    import httpx

    from fmd.acquire import tiers as tiers_module

    def post(url: str, **kwargs: object) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            503,
            json={"detail": "vision_unavailable: vLLM is not answering at http://host:8001/v1"},
            request=request,
        )

    monkeypatch.setenv("FMD_T2_URL", "http://ocr.test")
    monkeypatch.setattr(tiers_module.httpx, "post", post)

    result = run_tiers(
        b"ignored",
        policy=_policy_with_t2_backend("qwen3vl"),
        t0=lambda _: _doc(LONG_TEXT + " शुल्क और पात्रता की जानकारी।"),
        t2=t2_extract,
    )

    assert result.tier == "T0"
    assert result.fallback_from == "T2"
    # The service's /health carries the `vision_unavailable` label; the tier note
    # records that T2 could not run and why. `tiers._ocr_tier` formats an
    # httpx.HTTPStatusError, which omits the response body, so the label itself
    # lives on the service health probe (tests/test_ocr_service.py) -- the note
    # must still make the degradation visible and must not invent text.
    assert "T2 unavailable" in result.notes[0]
    assert "503" in result.notes[0]


def test_image_only_page_with_no_vision_engine_produces_no_claims(
    monkeypatch: pytest.MonkeyPatch, qwen_t2_backend: None
) -> None:
    """The scanned-page path: no OCR, no text, so no claims and a review item."""
    import httpx

    from fmd.acquire import tiers as tiers_module

    def post(url: str, **kwargs: object) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            503, json={"detail": "vision_unavailable: no weights"}, request=request
        )

    monkeypatch.setenv("FMD_T2_URL", "http://ocr.test")
    monkeypatch.setattr(tiers_module.httpx, "post", post)

    with pytest.raises(TierUnavailable) as excinfo:
        run_tiers(
            b"ignored",
            policy=_policy_with_t2_backend("qwen3vl"),
            t0=lambda _: _doc(""),
            t2=t2_extract,
        )
    message = str(excinfo.value)
    assert "cannot be extracted on this host" in message
    assert "503" in message


# --------------------------------------------------------------------------- device / OOM


def test_docling_device_defaults_to_cpu_and_is_env_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-LLM-resident hosts must not put Docling on the same GPU by default."""
    from fmd.acquire.tiers import docling_device

    monkeypatch.delenv("FMD_DOCLING_DEVICE", raising=False)
    assert docling_device(_policy()) == "cpu"
    monkeypatch.setenv("FMD_DOCLING_DEVICE", "cuda")
    assert docling_device(_policy()) == "cuda"


def test_invalid_docling_device_is_a_named_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from fmd.acquire.tiers import docling_device

    monkeypatch.setenv("FMD_DOCLING_DEVICE", "rocm")
    with pytest.raises(Exception) as excinfo:
        docling_device(_policy())
    assert "acquisition_failed:invalid_docling_device" in str(excinfo.value)


def test_docling_oom_does_not_fall_back_to_flat_t0_text() -> None:
    """A CUDA OOM is an acquisition failure, never a clean 0-claim extraction."""
    from fmd.acquire.errors import AcquisitionFailed

    class OomConverter:
        def convert(self, path: str) -> object:
            raise RuntimeError("CUDA out of memory. Tried to allocate 50.00 MiB")

    table_text = LONG_TEXT + "\n50000 60 4\n45000 60 4\n40000 60 4"

    with pytest.raises(AcquisitionFailed) as excinfo:
        run_tiers(
            b"ignored",
            policy=_policy(),
            t0=lambda _: _doc(table_text),
            t1=lambda _: t1_extract(b"%PDF-1.4", converter=OomConverter(), policy=_policy()),
        )
    assert excinfo.value.reason == "acquisition_failed:docling_cuda_oom"
    assert "out of memory" in str(excinfo.value)
