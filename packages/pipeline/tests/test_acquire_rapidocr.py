"""RapidOCR (PP-OCR) T2 backend — offline tests with a mocked runner.

No network, no model downloads, no onnxruntime import: the runner, the device
probe and the clock are injected, and page rendering is patched. The real
bring-up (GPU, weights, sha256) is exercised by the documented commands in the
runbook §3, not here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fmd.acquire.errors import AcquisitionFailed, TierUnavailable
from fmd.acquire.rapidocr_backend import (
    CUDA_UNAVAILABLE_NO_PROVIDER,
    CUDA_UNAVAILABLE_VRAM,
    ENGINE_NAME,
    LANG_REC_VALUES,
    NO_TEXT_LAYER,
    RapidOCRBackend,
    RapidOCRSettings,
    conjunct_risk,
    has_devanagari,
    normalise_devanagari_digits,
    vram_free_gib,
)
from fmd.acquire.tiers import (
    T2_BACKENDS,
    apply_anomaly_guard,
    run_tiers,
    t2_backend,
    t2_extract,
)
from fmd.acquire.types import ExtractedDoc
from tests.test_acquire_tiers import _doc, _policy, _policy_with_t2_backend

# --------------------------------------------------------------------------- fakes


class FakeOutput:
    """Shape-compatible stand-in for ``rapidocr``'s RapidOCROutput."""

    def __init__(
        self,
        txts: list[str],
        scores: list[float],
        boxes: list[list[list[float]]] | None = None,
    ) -> None:
        self.txts = tuple(txts)
        self.scores = tuple(scores)
        if boxes is None:
            boxes = [[[0.0, float(10 + 30 * i)], [100.0, float(10 + 30 * i)],
                      [100.0, float(30 + 30 * i)], [0.0, float(30 + 30 * i)]]
                     for i in range(len(txts))]
        self.boxes = np.asarray(boxes, dtype=float) if boxes else None


class FakeRunner:
    """Records the images it is called with; plays scripted outputs."""

    def __init__(self, outputs: list[FakeOutput]) -> None:
        self.outputs = list(outputs)
        self.calls: list[Any] = []

    def __call__(self, image: Any) -> FakeOutput:
        self.calls.append(image)
        if self.outputs:
            return self.outputs.pop(0)
        return FakeOutput([], [])


def _image(*, blank: bool = False) -> Any:
    if blank:
        return np.full((80, 200, 3), 255, dtype=np.uint8)
    # Non-uniform raster: a uniform image reads as "no ink" and is skipped.
    return (np.arange(80 * 200 * 3, dtype=np.uint8).reshape(80, 200, 3) % 251)


PINNED_FILES = (
    "PP-OCRv6_det_small.onnx",
    "devanagari_PP-OCRv5_rec_mobile.onnx",
    "ppocrv5_devanagari_dict.txt",
)


def _settings(tmp_path, **overrides: Any) -> RapidOCRSettings:
    """Settings rooted at ``tmp_path`` with the pinned model files present."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in PINNED_FILES:
        (tmp_path / name).write_bytes(b"fake-onnx-payload")
    defaults: dict[str, Any] = {
        "device": "cpu",
        "models_root": tmp_path,
        "det_model_id": "PP-OCRv6_det_small.onnx",
        "rec_model_id": "devanagari_PP-OCRv5_rec_mobile.onnx",
        "rec_keys_id": "ppocrv5_devanagari_dict.txt",
        "model_source": "modelscope:RapidAI/RapidOCR@v3.10.0",
        "max_pages": 10,
        "max_seconds": 600.0,
    }
    defaults.update(overrides)
    return RapidOCRSettings(**defaults)


@pytest.fixture
def models_root(tmp_path):
    """Pinned model files present under a fake artifacts root."""
    for name in PINNED_FILES:
        (tmp_path / name).write_bytes(b"fake-onnx-payload")
    return tmp_path


def _render(monkeypatch: pytest.MonkeyPatch, *images: Any) -> None:
    import fmd.acquire.rapidocr_backend as module

    monkeypatch.setattr(module, "_render_pdf", lambda data, **kwargs: list(images))


# --------------------------------------------------------------------------- helpers / hygiene


def test_devanagari_script_class_and_digits() -> None:
    """Devanagari is a first-class script class: routing, digits, conjuncts."""
    assert has_devanagari("कुल शुल्क ₹५०,०००")
    assert not has_devanagari("Total fee INR 50000")
    normalised, count = normalise_devanagari_digits("कुल शुल्क ५०,००० प्रति वर्ष")
    assert normalised == "कुल शुल्क 50,000 प्रति वर्ष"
    assert count == 5  # ५० (2) + ००० (3)
    unchanged, zero = normalise_devanagari_digits("Total fee 50000")
    assert (unchanged, zero) == ("Total fee 50000", 0)


def test_conjunct_risk_requires_low_confidence() -> None:
    """Conjunct clusters at low confidence flag the page; crisp pages do not."""
    assert conjunct_risk("शक्ति और विश्व", low_confidence=True)
    assert not conjunct_risk("शक्ति और विश्व", low_confidence=False)
    assert not conjunct_risk("Total fee 50000 per year", low_confidence=True)


def test_settings_reject_unknown_lang_rec(tmp_path) -> None:
    with pytest.raises(TierUnavailable):
        RapidOCRSettings(models_root=tmp_path, rec_model_id="r", rec_keys_id="d",
                         lang_rec="klingon")
    assert "devanagari" in LANG_REC_VALUES and "latin" in LANG_REC_VALUES


def test_vram_probe_returns_none_without_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    assert vram_free_gib() is None


# --------------------------------------------------------------------------- device


def test_device_defaults_to_cpu_from_the_registry() -> None:
    """CPU is the measured default (llama-server resident, ~688 MiB free)."""
    settings = RapidOCRSettings.from_spec(_policy().backend("T2"), env={})
    assert settings.device == "cpu"
    assert settings.min_free_vram_gib >= 1.5  # onnxruntime CUDA needs ~1.5–2 GiB


def test_cuda_without_provider_falls_back_to_cpu_with_reason(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "fmd.acquire.rapidocr_backend._onnx_providers", lambda: ["CPUExecutionProvider"]
    )
    backend = RapidOCRBackend(_settings(tmp_path, models_root=None, device="cuda"))
    device, reason = backend.resolve_device()
    assert (device, reason) == ("cpu", CUDA_UNAVAILABLE_NO_PROVIDER)


def test_cuda_below_vram_floor_falls_back_to_cpu_with_recorded_reason(
    monkeypatch, tmp_path
) -> None:
    """cuda is opt-in and never crashes a run: below the floor it records why.

    Measured host fact: with llama-server resident only ~688 MiB is free, far
    under the 2.0 GiB floor — the CUDA path fails by name and CPU answers.
    """
    monkeypatch.setattr(
        "fmd.acquire.rapidocr_backend._onnx_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    backend = RapidOCRBackend(
        _settings(tmp_path, device="cuda"),
        runner=FakeRunner([FakeOutput(["Total fee INR 50000"], [0.95])]),
        vram_probe=lambda: 0.67,  # measured free VRAM with the LLM server resident
    )
    _render(monkeypatch, _image())
    payload = backend.extract(b"pdf")

    assert backend.device == "cpu"
    assert backend.device_reason == CUDA_UNAVAILABLE_VRAM
    assert payload["device"] == "cpu"
    assert payload["device_reason"] == CUDA_UNAVAILABLE_VRAM
    assert payload["engine_meta"]["device_reason"] == CUDA_UNAVAILABLE_VRAM


def test_cuda_with_free_vram_is_honoured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "fmd.acquire.rapidocr_backend._onnx_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    seen: dict[str, str] = {}

    def factory(settings, device, models):
        seen["device"] = device
        return FakeRunner([FakeOutput(["ok"], [0.99])])

    backend = RapidOCRBackend(
        _settings(tmp_path, device="cuda"), runner_factory=factory, vram_probe=lambda: 8.0
    )
    _render(monkeypatch, _image())
    payload = backend.extract(b"pdf")

    assert backend.device == "cuda"
    assert backend.device_reason is None
    assert seen["device"] == "cuda"
    assert payload["engine_meta"]["device"] == "cuda"


def test_invalid_device_is_a_named_failure(tmp_path) -> None:
    from fmd.acquire.rapidocr_backend import AcquisitionFailed as AF  # noqa: F401

    with pytest.raises(TierUnavailable) as excinfo:
        RapidOCRSettings.from_spec(
            _policy_with_t2_backend("rapidocr").backend("T2"), env={"FMD_OCR_DEVICE": "tpu"}
        )
    assert "cpu|cuda" in str(excinfo.value)


# --------------------------------------------------------------------------- models


def test_missing_models_name_the_prefetch(tmp_path) -> None:
    """rec/dict absent under the configured root => named prefetch command."""
    root = tmp_path / "empty-root"
    _settings(root)  # writes the pinned files, then remove rec + dict
    (root / "devanagari_PP-OCRv5_rec_mobile.onnx").unlink()
    (root / "ppocrv5_devanagari_dict.txt").unlink()
    backend = RapidOCRBackend(_settings(tmp_path / "unused", models_root=root))
    with pytest.raises(TierUnavailable) as excinfo:
        backend.resolve_models()
    message = str(excinfo.value)
    assert "models_missing" in message
    assert "devanagari_PP-OCRv5_rec_mobile.onnx" in message
    assert "scripts/ocr_validate.py --prefetch" in message


def test_missing_det_names_the_install(tmp_path, monkeypatch) -> None:
    import sys

    root = tmp_path / "no-det"
    _settings(root)  # writes the pinned files, then remove the bundled det
    (root / "PP-OCRv6_det_small.onnx").unlink()
    monkeypatch.setitem(sys.modules, "rapidocr", None)  # no bundled fallback
    backend = RapidOCRBackend(_settings(tmp_path / "unused", models_root=root))
    with pytest.raises(TierUnavailable) as excinfo:
        backend.resolve_models()
    assert "PP-OCRv6_det_small.onnx" in str(excinfo.value)
    assert "uv pip install" in str(excinfo.value)


def test_missing_runner_names_the_install(tmp_path, monkeypatch) -> None:
    """Without the runner package the fix is named, not guessed at."""
    import sys

    monkeypatch.setitem(sys.modules, "rapidocr", None)  # import raises ImportError
    backend = RapidOCRBackend(_settings(tmp_path, models_root=None))
    with pytest.raises(TierUnavailable) as excinfo:
        backend._runner({})
    assert "uv pip install" in str(excinfo.value)
    assert "rapidocr>=3.10" in str(excinfo.value)


# --------------------------------------------------------------------------- extract


def test_extract_returns_blocks_engine_meta_and_pages(monkeypatch, tmp_path) -> None:
    runner = FakeRunner(
        [
            FakeOutput(
                ["Total fee", "INR ५०,००० per year"],
                [0.97, 0.93],
            )
        ]
    )
    backend = RapidOCRBackend(_settings(tmp_path), runner=runner)
    _render(monkeypatch, _image())
    payload = backend.extract(b"pdf")

    assert [block["block_content"] for block in payload["blocks"]] == [
        "Total fee",
        "INR 50,000 per year",
    ]
    assert all(block["page"] == 1 for block in payload["blocks"])
    assert payload["tool"] == ENGINE_NAME
    meta = payload["engine_meta"]
    assert meta["engine"] == "rapidocr"
    assert meta["model_ids"]["rec"] == "devanagari_PP-OCRv5_rec_mobile.onnx"
    assert meta["model_ids"]["rec_keys"] == "ppocrv5_devanagari_dict.txt"
    assert meta["granularity"] == "block"
    assert meta["lang_rec"] == "devanagari"
    page = meta["pages"][0]
    assert page["devanagari_digits_normalized"] == 5  # ५०,०००
    assert page["conjunct_risk"] is False
    assert payload["pages"][0]["width"] == 200


def test_confidence_floor_drops_and_counts(monkeypatch, tmp_path) -> None:
    """Boxes under the configured floor are dropped AND counted, never kept."""
    runner = FakeRunner(
        [FakeOutput(["crisp text", "smudged guess"], [0.95, 0.42])]
    )
    backend = RapidOCRBackend(_settings(tmp_path), runner=runner)
    _render(monkeypatch, _image())
    payload = backend.extract(b"pdf")

    assert [block["block_content"] for block in payload["blocks"]] == ["crisp text"]
    assert payload["engine_meta"]["pages"][0]["dropped_low_confidence"] == 1
    assert payload["engine_meta"]["dropped_low_confidence_total"] == 1


def test_conjunct_risk_flagged_on_low_confidence_devanagari(monkeypatch, tmp_path) -> None:
    """शक्ति→शक्ि / विश्व→विश्ष territory: conjuncts + weak scores => flag."""
    runner = FakeRunner(
        [
            FakeOutput(
                ["संस्था की शक्ति और विश्व स्तरीय मान्यता प्रमाण पत्र"],
                [0.71],
            )
        ]
    )
    backend = RapidOCRBackend(_settings(tmp_path, conjunct_risk_score_max=0.85), runner=runner)
    _render(monkeypatch, _image())
    payload = backend.extract(b"pdf")

    assert payload["engine_meta"]["pages"][0]["conjunct_risk"] is True
    assert 1 in payload["engine_meta"]["conjunct_risk_pages"]


def test_blank_pages_produce_no_text_layer_not_a_zero(monkeypatch, tmp_path) -> None:
    """OCR empty on an image-only page => no_text_layer, review — never a 0."""
    runner = FakeRunner([])
    backend = RapidOCRBackend(_settings(tmp_path), runner=runner)
    _render(monkeypatch, _image(blank=True))
    with pytest.raises(TierUnavailable) as excinfo:
        backend.extract(b"pdf")
    assert NO_TEXT_LAYER in str(excinfo.value)


def test_progress_line_per_page(monkeypatch, tmp_path, caplog) -> None:
    """A stall is visible within one page: a log line after every page."""
    import logging

    runner = FakeRunner([FakeOutput(["a"], [0.9]), FakeOutput(["b"], [0.9])])
    backend = RapidOCRBackend(_settings(tmp_path), runner=runner)
    _render(monkeypatch, _image(), _image())
    with caplog.at_level(logging.INFO, logger="fmd.acquire.rapidocr_backend"):
        backend.extract(b"pdf")
    lines = [r for r in caplog.records if "ocr_progress" in r.getMessage()]
    assert [r.getMessage().split("page=")[1].split(" ")[0] for r in lines] == ["1/2", "2/2"]


# --------------------------------------------------------------------------- budgets (D18)


def test_page_budget_aborts_with_stable_reason(monkeypatch, tmp_path) -> None:
    runner = FakeRunner([FakeOutput(["text"], [0.9]) for _ in range(3)])
    backend = RapidOCRBackend(_settings(tmp_path, max_pages=1), runner=runner)
    _render(monkeypatch, _image(), _image(), _image())
    with pytest.raises(AcquisitionFailed) as excinfo:
        backend.extract(b"pdf")
    assert "acquisition_budget_exceeded" in str(excinfo.value)


def test_time_budget_aborts_with_stable_reason(monkeypatch, tmp_path) -> None:
    """A slow page hits the wall-clock budget: named abort, no partial claims."""
    ticks = iter([0.0, 0.0, 1000.0, 1000.0, 1000.0])
    runner = FakeRunner([FakeOutput(["text"], [0.9]) for _ in range(3)])
    backend = RapidOCRBackend(
        _settings(tmp_path, max_seconds=900.0), runner=runner, clock=lambda: next(ticks)
    )
    _render(monkeypatch, _image(), _image())
    with pytest.raises(AcquisitionFailed) as excinfo:
        backend.extract(b"pdf")
    assert "acquisition_budget_exceeded" in str(excinfo.value)
    assert "no partial claims" in str(excinfo.value)


def test_run_tiers_never_falls_back_after_a_budget_abort(tmp_path) -> None:
    """An aborted OCR publishes no claims and no T0 fallback (D18)."""
    from fmd.acquire.errors import ACQUISITION_FAILED_PREFIX

    def aborted(_: bytes) -> ExtractedDoc:
        raise AcquisitionFailed(
            f"{ACQUISITION_FAILED_PREFIX}acquisition_budget_exceeded",
            "time budget exceeded at page 7/200",
        )

    with pytest.raises(AcquisitionFailed):
        run_tiers(b"pdf", policy=_policy(), t0=lambda _: _doc(""), t2=aborted)


# --------------------------------------------------------------------------- tier wiring


def test_registry_default_backend_is_rapidocr_and_matrix_is_closed() -> None:
    assert T2_BACKENDS == ("rapidocr", "qwen3vl", "paddleocr")
    assert t2_backend(_policy()) == "rapidocr"


def test_unknown_t2_backend_is_a_named_failure() -> None:
    policy = _policy_with_t2_backend("bogus")
    with pytest.raises(AcquisitionFailed) as excinfo:
        t2_backend(policy)
    assert "unknown_t2_backend" in str(excinfo.value)


def test_config_loader_rejects_unknown_backend(tmp_path) -> None:
    """The registry validation rejects a backend outside the matrix."""
    import shutil

    from fmd.config import ConfigError
    from fmd.config.loader import load_registry

    config_dir = tmp_path / "config"
    source = __import__("pathlib").Path(__file__).resolve().parents[3] / "config"
    shutil.copytree(source, config_dir)
    acquire = (config_dir / "acquire.yaml").read_text(encoding="utf-8")
    acquire = acquire.replace("backend: ${FMD_T2_BACKEND:-rapidocr}", "backend: turboocr")
    (config_dir / "acquire.yaml").write_text(acquire, encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir=config_dir)
    assert any("backends.T2.backend" in error for error in excinfo.value.errors)


def test_t2_extract_through_rapidocr_carries_engine_metadata(
    monkeypatch, tmp_path
) -> None:
    """t2_extract (registry default) → payload → ExtractedDoc.engine_meta."""
    import fmd.acquire.rapidocr_backend as backend_module
    from fmd.config.loader import reset_registry_cache

    monkeypatch.setenv("FMD_RAPIDOCR_MODELS", str(tmp_path))
    reset_registry_cache()
    try:
        for name in (
            "PP-OCRv6_det_small.onnx",
            "devanagari_PP-OCRv5_rec_mobile.onnx",
            "ppocrv5_devanagari_dict.txt",
        ):
            (tmp_path / name).write_bytes(b"fake")
        runner = FakeRunner([FakeOutput(["Total fee INR 50000"], [0.95])])
        monkeypatch.setattr(backend_module, "_render_pdf", lambda data, **kw: [_image()])
        monkeypatch.setattr(
            backend_module,
            "build_rapidocr_runner",
            lambda settings, device, models: runner,
        )

        doc = t2_extract(b"pdf")
        assert doc.tier == "T2"
        assert doc.tool == "rapidocr"
        assert doc.engine_meta is not None
        assert doc.engine_meta["engine"] == "rapidocr"
        assert doc.engine_meta["model_ids"]["det"] == "PP-OCRv6_det_small.onnx"
        assert doc.bbox_granularity == "block"
        structure = doc.to_structure()
        assert structure["engine_meta"]["engine"] == "rapidocr"
    finally:
        monkeypatch.delenv("FMD_RAPIDOCR_MODELS", raising=False)
        reset_registry_cache()


def test_http_payloads_get_synthesised_engine_meta() -> None:
    """Qwen3-VL/paddle HTTP payloads record engine + model even without a block."""
    from fmd.acquire.tiers import _engine_meta_of

    spec = _policy().backend("T2")
    meta = _engine_meta_of(
        {"engine": "qwen3vl", "tool_version": "0.33.0", "model": "Qwen3-VL-8B-Instruct"},
        spec,
    )
    assert meta["engine"] == "qwen3vl"
    assert meta["model_ids"] == {"model": "Qwen3-VL-8B-Instruct"}
    assert meta["granularity"] == "block"
    assert _engine_meta_of({"text": "x"}, spec) is None


def test_engine_provenance_lands_on_the_claim_qualifier() -> None:
    """engine + engine_version + model_ids persisted on the claim qualifier."""
    from pydantic import BaseModel

    from fmd.extract.pipeline import _attach_engine_provenance

    doc = _doc("Total fee INR 50000", tier="T2")
    doc = doc.__class__(**{**doc.__dict__, "engine_meta": {
        "engine": "rapidocr",
        "engine_version": "3.10.0",
        "model_ids": {"det": "PP-OCRv6_det_small.onnx",
                      "rec": "devanagari_PP-OCRv5_rec_mobile.onnx"},
        "device": "cpu",
        "device_reason": None,
        "granularity": "block",
        "pages": [{"page": 1, "conjunct_risk": True,
                   "dropped_low_confidence": 3,
                   "devanagari_digits_normalized": 4}],
    }})

    class Claim(BaseModel):
        qualifier: dict = {}

    claim = _attach_engine_provenance(Claim(qualifier={"page_no": 1}), doc)
    assert claim.qualifier["ocr_engine"] == "rapidocr"
    assert claim.qualifier["ocr_engine_version"] == "3.10.0"
    assert claim.qualifier["ocr_model_ids"]["rec"] == "devanagari_PP-OCRv5_rec_mobile.onnx"
    assert claim.qualifier["ocr_device"] == "cpu"
    assert claim.qualifier["conjunct_risk"] is True
    assert claim.qualifier["ocr_dropped_low_confidence"] == 3
    assert claim.qualifier["devanagari_digits_normalized"] == 4

    born_digital = _attach_engine_provenance(Claim(qualifier={}), _doc("x", tier="T0"))
    assert born_digital.qualifier == {}


def test_repetition_loop_in_rapidocr_payload_routes_to_t3() -> None:
    payload = {
        "tool": "rapidocr",
        "engine": "rapidocr",
        "pages": [{"page": 1, "width": 600, "height": 800, "bbox": [0, 0, 600, 800]}],
        "blocks": [
            {
                "page": 1,
                "block_order": 0,
                "block_bbox": [0, 0, 100, 20],
                "block_content": "Fee 50000 " * 71,
                "score": 0.95,
            }
        ],
        "engine_meta": {"engine": "rapidocr", "granularity": "block", "pages": []},
    }

    class Backend:
        def extract(self, data: bytes) -> dict:
            return payload

    doc = t2_extract(b"pdf", backend=Backend())
    assert doc.engine_meta is not None
    guarded = apply_anomaly_guard(doc, _policy())
    assert guarded.tier == "T3"
    assert guarded.needs_review is True
    assert guarded.guard_trips_by_name() == {"repetition_loop": 1}


# --------------------------------------------------------------------------- Devanagari capability


def test_devanagari_rec_model_id_is_in_the_runner_manifest() -> None:
    """The pinned Devanagari rec model id exists in the runner's model manifest.

    Skipped where the runner wheel is not installed (CI runs offline); the
    real check is `uv run pytest tests/test_acquire_rapidocr.py -q` on the GPU
    host after `uv pip install "rapidocr>=3.10" "onnxruntime>=1.32"`.
    """
    try:
        import rapidocr
    except ImportError:
        pytest.skip("rapidocr runner not installed in this environment (offline CI)")
    manifest = (
        __import__("pathlib").Path(rapidocr.__file__).resolve().parent
        / "default_models.yaml"
    )
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert "devanagari_PP-OCRv5_rec_mobile" in text
    assert "d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e" in text


def test_devanagari_input_routes_to_the_pinned_rec_model(
    tmp_path, monkeypatch
) -> None:
    """A Devanagari page is recognised by the pinned Devanagari rec model."""
    captured: dict[str, Any] = {}

    def factory(settings, device, models):
        captured["lang_type"] = settings.lang_rec
        captured["rec_model_path"] = models["rec"]
        captured["rec_keys_path"] = models["rec_keys"]
        return FakeRunner([FakeOutput(["कुल शुल्क ५०,०००"], [0.94])])

    backend = RapidOCRBackend(
        _settings(tmp_path), runner_factory=factory, vram_probe=lambda: None
    )
    _render(monkeypatch, _image())
    payload = backend.extract(b"pdf")

    assert captured["lang_type"] == "devanagari"
    assert captured["rec_model_path"].name == "devanagari_PP-OCRv5_rec_mobile.onnx"
    assert captured["rec_keys_path"].name == "ppocrv5_devanagari_dict.txt"
    assert has_devanagari(payload["blocks"][0]["block_content"])


def test_devanagari_page_through_the_pipeline_stays_guarded(
    tmp_path, monkeypatch
) -> None:
    """End-to-end: Devanagari scan → rapidocr → canonical text keeps the script."""
    runner = FakeRunner([FakeOutput(["कुल शुल्क ५०,००० प्रति वर्ष"], [0.94])])
    backend = RapidOCRBackend(_settings(tmp_path), runner=runner)
    _render(monkeypatch, _image())
    payload = backend.extract(b"pdf")
    assert any("शुल्क" in block["block_content"] for block in payload["blocks"])
