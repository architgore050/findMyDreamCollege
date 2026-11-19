"""RapidOCR (PP-OCR) backend — the T2 engine of record, in-process.

RapidOCR is the *runner*, not the model: an ONNX-runtime wrapper around
PaddleOCR PP-OCR detection + recognition graphs. The models it loads here are
pinned in ``config/acquire.yaml`` (``backends.T2.models``), sourced once from
ModelScope ``RapidAI/RapidOCR@v3.10.0``:

* det ``PP-OCRv6_det_small.onnx`` (bundled with the runner wheel);
* rec ``devanagari_PP-OCRv5_rec_mobile.onnx`` — sha256 ``d6f0a906…52c91e`` —
  with ``ppocrv5_devanagari_dict.txt``. Devanagari is a first-class LangRec
  value in the runner (``LangRec.DEVANAGARI``), not a transliteration trick.

Why this over a VLM: purpose-built detection+recognition is orders of
magnitude cheaper per page than a vision-language model. A VLM over a
200-page scan is exactly what stalled the corpus (D18). Qwen3-VL stays fully
wired and selectable — the Devanagari fallback and an alternate T2 backend —
and the PP-OCRv5-vs-Qwen preference is PROVISIONAL UNTIL MEASURED
(``scripts/ocr_validate.py --ground-truth`` decides; rule in the runbook §3.8).

Measured on the reference host (RTX 5080 Laptop, 16 GB):

* **device ``cpu`` is the default.** With llama-server resident (14.4 GiB)
  only ~688 MiB VRAM is free; onnxruntime CUDA needs ~1.5–2 GiB and OOMs.
  CPU measures 2.33 s/page (25.8 pages/min) — 200 pp ≈ 8 min.
* ``device: cuda`` is opt-in for batch runs with the LLM server stopped. If
  configured and the free VRAM is below the configured floor, the CUDA path
  fails with a recorded ``cuda_unavailable_vram`` reason and the backend
  falls back to CPU — the run is never crashed by a device preference.

Defects are handled, not hidden (all recorded per page in ``engine_meta``):

* boxes below the configured confidence floor are **dropped and counted**;
* virama/conjunct sequences are error-prone (शक्ति→शक्ि, विश्व→विश्ष): pages
  carrying conjuncts at low confidence get a ``conjunct_risk`` flag;
* Devanagari digits (०१२३) are untested in our corpus — when detected they
  are normalised to ASCII and the count is logged and recorded, never assumed.

Every scanned page records the engine, runner version, model ids and block
granularity in the payload (``engine_meta``), so a mixed-engine corpus stays
reproducible. Budget aborts raise ``AcquisitionFailed`` with the stable
``acquisition_budget_exceeded`` reason — no partial claims are published.

Tests never import onnxruntime or download models: the runner, the device
probe and the clock are injectable.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fmd.acquire.errors import (
    ACQUISITION_FAILED_PREFIX,
    AcquisitionFailed,
    TierUnavailable,
)

logger = logging.getLogger(__name__)

#: Engine name recorded on every claim qualifier / payload (never a version).
ENGINE_NAME = "rapidocr"
#: Stable review/failure codes (never prose).
NO_TEXT_LAYER = "no_text_layer"
CUDA_UNAVAILABLE_NO_PROVIDER = "cuda_unavailable_no_provider"
CUDA_UNAVAILABLE_VRAM = "cuda_unavailable_vram"
CUDA_PROBE_FAILED = "cuda_probe_failed"
#: D18 abort reason, raised as ``acquisition_failed:<code>``.
ACQUISITION_BUDGET_EXCEEDED = "acquisition_budget_exceeded"
#: Runner install pinned here and in infra/ocr/requirements.txt; the message
#: below is what a missing install names, so the fix is copy-pasteable.
INSTALL_COMMAND = 'uv pip install "rapidocr>=3.10" "onnxruntime>=1.32"'
#: One-time model prefetch (ModelScope RapidAI/RapidOCR@v3.10.0, sha256-checked).
PREFETCH_COMMAND = "uv run python scripts/ocr_validate.py --prefetch"

#: Devanagari block (U+0900–U+097F): the script class used for routing and for
#: the conjunct-risk heuristic. Mirrors ``fmd.acquire.tiers._DEVANAGARI``.
DEVANAGARI_RANGE = range(0x0900, 0x0980)
#: Devanagari digits U+0966–U+096F → ASCII. Normalisation is measured and
#: logged, never silent: the corpus does not yet cover digit-heavy scans.
_DEVANAGARI_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")
#: Consonant + virama + consonant: a conjunct cluster. Recognition drops or
#: mangles the virama on these (measured: शक्ति→शक्ि, विश्व→विश्ष).
_CONJUNCT_RE = re.compile(r"[\u0915-\u0939]\u094D[\u0915-\u0939]")
#: LangRec values this backend accepts. Devanagari is first-class in the
#: runner's own enum (``LangRec.DEVANAGARI``); Latin is the alternate.
LANG_REC_VALUES = ("devanagari", "latin")
#: First-party LangRec values, for the model manifest check.
LANG_REC_DEVANAGARI = "devanagari"

_TRUTHY = {"1", "true", "yes", "on"}
_DEVICES = {"cpu", "cuda"}
_NVIDIA_SMI = (
    "nvidia-smi",
    "--query-gpu=memory.free",
    "--format=csv,noheader,nounits",
)


def has_devanagari(text: str) -> bool:
    """Does ``text`` contain any Devanagari codepoint?"""
    return any(ord(char) in DEVANAGARI_RANGE for char in text)


def normalise_devanagari_digits(text: str) -> tuple[str, int]:
    """Devanagari digits → ASCII, with the number of replacements.

    Our corpus does not yet cover digit-heavy Devanagari scans, so digits are
    normalised (downstream numeric parsers are ASCII-only) and *logged* — the
    raw count travels on the page row so a reviewer can see it happened.
    """
    if not any(0x0966 <= ord(char) <= 0x096F for char in text):
        return text, 0
    normalised = text.translate(_DEVANAGARI_DIGIT_MAP)
    replaced = sum(
        1 for char in text if 0x0966 <= ord(char) <= 0x096F
    )
    return normalised, replaced


def conjunct_risk(text: str, *, low_confidence: bool) -> bool:
    """True when the page carries conjunct clusters at low confidence.

    ``low_confidence`` means the page dropped at least one box under the
    confidence floor or its mean box score sits under the configured
    ``conjunct_risk_score_max``. A crisp high-confidence page with conjuncts
    is *not* flagged; a low-confidence page with conjuncts is where the
    measured शक्ति→शक्ि / विश्व→विश्ष errors live.
    """
    return bool(_CONJUNCT_RE.search(text)) and low_confidence


def vram_free_gib() -> float | None:
    """Free VRAM of GPU 0 in GiB, or ``None`` when it cannot be measured.

    Reads ``nvidia-smi`` (the same source the operator uses); a missing binary
    or a driver error returns ``None`` and the caller treats the card as
    unusable — a card we cannot measure is not a card we can budget.
    """
    try:
        raw = subprocess.run(  # noqa: S603 — fixed argv, no shell
            _NVIDIA_SMI,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    first = raw.splitlines()[0].strip() if raw else ""
    try:
        return float(first) / 1024.0
    except ValueError:
        return None


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class RapidOCRSettings:
    """The ``backends.T2`` block (models, device, budgets) as pure data.

    Built by :meth:`from_spec` — Python never hardcodes a model id, device or
    budget; ``config/acquire.yaml`` is the registry.
    """

    device: str = "cpu"
    min_free_vram_gib: float = 2.0
    models_root: Path | None = None
    det_model_id: str = ""
    rec_model_id: str = ""
    rec_keys_id: str = ""
    model_source: str = ""
    model_sha256: dict[str, str] = field(default_factory=dict)
    lang_rec: str = LANG_REC_DEVANAGARI
    min_box_confidence: float = 0.7
    conjunct_risk_score_max: float = 0.85
    normalise_digits: bool = True
    max_pages: int = 250
    max_seconds: float = 900.0
    dpi: int = 200
    max_side_px: int = 2400

    def __post_init__(self) -> None:
        if self.lang_rec not in LANG_REC_VALUES:
            raise TierUnavailable(
                f"acquire.yaml: backends.T2.models.lang_rec {self.lang_rec!r} is not one of "
                f"{', '.join(LANG_REC_VALUES)} (LangRec values)"
            )
        if not self.rec_model_id or not self.rec_keys_id:
            raise TierUnavailable(
                "acquire.yaml: backends.T2.models must pin rec_model_id and rec_keys_id "
                f"(Devanagari is a first-class LangRec value; prefetch with {PREFETCH_COMMAND})"
            )

    @classmethod
    def from_spec(cls, spec: Any, env: dict[str, str] | None = None) -> RapidOCRSettings:
        """Parse the ``T2`` backend spec (+ env overrides) into settings."""
        env = dict(env if env is not None else {})
        models = getattr(spec, "models", None) or {}
        device = str(
            env.get("FMD_OCR_DEVICE")
            or getattr(spec, "device", None)
            or "cpu"
        ).strip().lower()
        base = device.split(":", 1)[0]
        if base not in _DEVICES:
            raise AcquisitionFailed(
                f"{ACQUISITION_FAILED_PREFIX}invalid_ocr_device",
                f"device {device!r} is not one of cpu|cuda|cuda:N",
            )
        root = str(
            env.get("FMD_RAPIDOCR_MODELS")
            or models.get("artifacts_root")
            or ""
        ).strip()
        sha = models.get("sha256") or {}
        return cls(
            device=device,
            min_free_vram_gib=float(getattr(spec, "min_free_vram_gib", 2.0) or 2.0),
            models_root=Path(root).expanduser() if root else None,
            det_model_id=str(models.get("det", "") or ""),
            rec_model_id=str(models.get("rec", "") or ""),
            rec_keys_id=str(models.get("rec_keys", "") or ""),
            model_source=str(models.get("source", "") or ""),
            model_sha256={str(k): str(v) for k, v in sha.items()} if isinstance(sha, dict) else {},
            lang_rec=str(models.get("lang_rec", LANG_REC_DEVANAGARI) or LANG_REC_DEVANAGARI),
            min_box_confidence=float(getattr(spec, "min_box_confidence", 0.7) or 0.7),
            conjunct_risk_score_max=float(
                getattr(spec, "conjunct_risk_score_max", 0.85) or 0.85
            ),
            normalise_digits=_as_bool(getattr(spec, "normalise_devanagari_digits", True)),
            max_pages=int(getattr(spec, "max_pages", 250) or 250),
            max_seconds=float(getattr(spec, "max_seconds", 900.0) or 900.0),
            dpi=int(getattr(spec, "dpi", 200) or 200),
            max_side_px=int(getattr(spec, "max_side_px", 2400) or 2400),
        )


# --------------------------------------------------------------------------- backend


class RapidOCRBackend:
    """In-process PP-OCR (detection + recognition) over the RapidOCR runner.

    ``extract(pdf_bytes)`` renders pages with pypdfium2, runs detection +
    recognition per page, applies the configured confidence floor, and returns
    the shared ``/extract`` payload shape (``pages`` + ``blocks``) plus an
    ``engine_meta`` block. One engine per page, one transcript.
    """

    name = ENGINE_NAME

    def __init__(
        self,
        settings: RapidOCRSettings,
        *,
        runner: Any = None,
        runner_factory: Callable[[RapidOCRSettings, str, dict[str, Path]], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        vram_probe: Callable[[], float | None] = vram_free_gib,
    ) -> None:
        self.settings = settings
        self._injected_runner = runner
        self._runner_factory = runner_factory
        self._built_runner: Any = None
        self._clock = clock
        self._vram_probe = vram_probe
        self.device: str = settings.device.split(":", 1)[0]
        self.device_reason: str | None = None

    # ------------------------------------------------------------- device

    def resolve_device(self) -> tuple[str, str | None]:
        """``(device, reason)`` with CPU fallback that never crashes a run.

        ``cpu`` is the default (measured: 2.33 s/page; a resident LLM server
        leaves ~688 MiB — onnxruntime CUDA OOMs there). ``cuda`` is opt-in:
        when the CUDA execution provider is missing, or free VRAM is below the
        configured floor, the reason is recorded and CPU answers instead.
        """
        device = self.settings.device
        if device == "cpu" or device.startswith("cpu"):
            return "cpu", None
        providers = _onnx_providers()
        if "CUDAExecutionProvider" not in providers:
            return "cpu", CUDA_UNAVAILABLE_NO_PROVIDER
        free = self._vram_probe()
        if free is None:
            return "cpu", CUDA_PROBE_FAILED
        if free < self.settings.min_free_vram_gib:
            return "cpu", CUDA_UNAVAILABLE_VRAM
        return device, None

    # ------------------------------------------------------------- models

    def resolve_models(self) -> dict[str, Path]:
        """Resolve pinned model ids to files, naming the prefetch on absence.

        The det model ships inside the runner wheel; the Devanagari rec model
        and its dictionary are prefetched once from ModelScope
        (RapidAI/RapidOCR@v3.10.0, sha256-pinned). Nothing is downloaded here.
        """
        models: dict[str, Path] = {}
        root = self.settings.models_root
        det = _find_model(self.settings.det_model_id, root, allow_bundled=True)
        if det is None:
            raise TierUnavailable(
                f"models_missing: det model {self.settings.det_model_id!r} not found under "
                f"{str(root) if root else 'FMD_RAPIDOCR_MODELS (unset)'}; it is bundled with "
                f"the runner wheel — install with {INSTALL_COMMAND}"
            )
        models["det"] = det
        for key, model_id in (
            ("rec", self.settings.rec_model_id),
            ("rec_keys", self.settings.rec_keys_id),
        ):
            path = _find_model(model_id, root, allow_bundled=False)
            if path is None:
                source = self.settings.model_source or "ModelScope RapidAI/RapidOCR@v3.10.0"
                raise TierUnavailable(
                    f"models_missing: {key} {model_id!r} not found under "
                    f"{str(root) if root else 'FMD_RAPIDOCR_MODELS (unset)'}; prefetch the "
                    f"pinned Devanagari models once with {PREFETCH_COMMAND} "
                    f"(source: {source})"
                )
            models[key] = path
        return models

    def _runner(self, models: dict[str, Path]) -> Any:
        if self._injected_runner is not None:
            return self._injected_runner
        if self._built_runner is not None:
            return self._built_runner
        try:
            import rapidocr  # noqa: F401 — presence check + version reporting

            del rapidocr
        except ImportError as exc:
            raise TierUnavailable(
                f"engine_not_installed: the RapidOCR runner is not installed ({exc}). "
                f"Install with {INSTALL_COMMAND}"
            ) from exc
        factory = self._runner_factory or build_rapidocr_runner
        self._built_runner = factory(self.settings, self.device, models)
        return self._built_runner

    # ------------------------------------------------------------- extract

    def extract(self, data: bytes) -> dict[str, Any]:
        """PDF bytes → the shared OCR payload with engine metadata."""
        self.device, self.device_reason = self.resolve_device()
        if self.device_reason:
            logger.warning(
                "rapidocr device=%s requested but %s — continuing on cpu (recorded)",
                self.settings.device,
                self.device_reason,
            )
        models = self.resolve_models()
        runner = self._runner(models)
        pages = _render_pdf(data, dpi=self.settings.dpi, max_side_px=self.settings.max_side_px)

        started = self._clock()
        page_rows: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        dropped_total = 0
        digits_total = 0
        conjunct_pages: list[int] = []
        scores_all: list[float] = []

        for index, image in enumerate(pages):
            page_no = index + 1
            if index >= self.settings.max_pages:
                raise AcquisitionFailed(
                    f"{ACQUISITION_FAILED_PREFIX}{ACQUISITION_BUDGET_EXCEEDED}",
                    f"page budget {self.settings.max_pages} reached before page {page_no}; "
                    "no partial claims are published (D18)",
                )
            elapsed = self._clock() - started
            if elapsed >= self.settings.max_seconds:
                raise AcquisitionFailed(
                    f"{ACQUISITION_FAILED_PREFIX}{ACQUISITION_BUDGET_EXCEEDED}",
                    f"time budget {self.settings.max_seconds:g}s exceeded at page "
                    f"{page_no}/{len(pages)} after {elapsed:.1f}s; no partial claims are "
                    "published (D18)",
                )
            page_blocks, page_row = self._run_page(
                page_no, image, runner, started=started, total_pages=len(pages)
            )
            page_rows.append(page_row)
            blocks.extend(page_blocks)
            dropped_total += page_row["dropped_low_confidence"]
            digits_total += page_row["devanagari_digits_normalized"]
            if page_row["conjunct_risk"]:
                conjunct_pages.append(page_no)
            scores_all.extend(page_row["kept_scores"])

        if not blocks:
            # Never let "the engine said nothing" read as a clean empty page:
            # an image-only page the engine cannot read goes to review (T3),
            # it does not become a zero-text success.
            raise TierUnavailable(
                f"{NO_TEXT_LAYER}: rapidocr found no text on {len(pages)} page(s); "
                "the page is unreadable at these settings — route the document "
                "to the human queue"
            )

        engine_meta = _engine_meta(
            self,
            models,
            page_rows=page_rows,
            dropped_total=dropped_total,
            digits_total=digits_total,
            conjunct_pages=conjunct_pages,
            scores=scores_all,
        )
        return {
            "tool": ENGINE_NAME,
            "tool_version": engine_meta["engine_version"],
            "engine": ENGINE_NAME,
            "engine_version": engine_meta["engine_version"],
            "model_ids": engine_meta["model_ids"],
            "device": self.device,
            "device_reason": self.device_reason,
            "pages": [
                {key: row[key] for key in ("page", "width", "height", "bbox", "blank")}
                for row in page_rows
            ],
            "blocks": blocks,
            "engine_meta": engine_meta,
        }

    def _run_page(
        self,
        page_no: int,
        image: Any,
        runner: Any,
        *,
        started: float,
        total_pages: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        blank = _is_blank(image)
        height, width = image.shape[0], image.shape[1]
        raw_texts: list[str] = []
        raw_scores: list[float] = []
        raw_boxes: list[tuple[float, float, float, float]] = []
        if not blank:
            try:
                output = runner(image)
            except Exception as exc:  # onnx/runtime failures are named, never silent
                raise AcquisitionFailed(
                    f"{ACQUISITION_FAILED_PREFIX}ocr_runner_failed",
                    f"rapidocr failed on page {page_no}: {type(exc).__name__}: {exc}",
                ) from exc
            texts, scores, boxes = _runner_output(output)
            raw_texts, raw_scores, raw_boxes = texts, scores, boxes

        kept: list[dict[str, Any]] = []
        dropped = 0
        kept_scores: list[float] = []
        digits_kept = 0
        for order, (text, score, box) in enumerate(
            zip(raw_texts, raw_scores, raw_boxes, strict=False)
        ):
            if score < self.settings.min_box_confidence:
                dropped += 1
                continue
            normalised, digits = normalise_devanagari_digits(str(text))
            digits_kept += digits
            kept.append(
                {
                    "page": page_no,
                    "block_order": order,
                    "block_bbox": [round(value, 2) for value in box],
                    "block_content": normalised,
                    "score": round(float(score), 4),
                }
            )
            kept_scores.append(float(score))
            if digits:
                logger.info(
                    "devanagari_digits page=%d normalised=%d", page_no, digits
                )

        page_text = "\n".join(block["block_content"] for block in kept)
        mean_score = sum(kept_scores) / len(kept_scores) if kept_scores else 0.0
        low_confidence = dropped > 0 or (
            bool(kept_scores) and mean_score < self.settings.conjunct_risk_score_max
        )
        risk = conjunct_risk(page_text, low_confidence=low_confidence)
        row = {
            "page": page_no,
            "width": int(width),
            "height": int(height),
            "bbox": [0, 0, int(width), int(height)],
            "blank": blank,
            "engine": ENGINE_NAME,
            "boxes": len(kept),
            "dropped_low_confidence": dropped,
            "devanagari_digits_normalized": digits_kept,
            "conjunct_risk": risk,
            "mean_confidence": round(mean_score, 4) if kept_scores else None,
            "kept_scores": kept_scores,
        }
        logger.info(
            "ocr_progress page=%d/%d boxes=%d dropped=%d conjunct_risk=%s elapsed=%.1fs",
            page_no,
            total_pages,
            row["boxes"],
            dropped,
            risk,
            self._clock() - started,
        )
        return kept, row


# --------------------------------------------------------------------------- runner glue


def build_rapidocr_runner(
    settings: RapidOCRSettings, device: str, models: dict[str, Path]
) -> Any:
    """Construct a ``rapidocr.RapidOCR`` with explicit pinned model paths.

    All parameters come from config; the runner never downloads at serve time
    (every path is resolved by :meth:`RapidOCRBackend.resolve_models` first).
    Param names verified against ``rapidocr`` 3.10 (``Global.*`` / ``Det.*`` /
    ``Rec.*`` / ``EngineConfig.onnxruntime.*``).
    """
    from rapidocr import RapidOCR

    params: dict[str, Any] = {
        # Keep every box: the confidence floor is applied (and counted) here,
        # not inside the runner.
        "Global.text_score": 0.0,
        "Global.log_level": "warning",
        "Det.model_path": str(models["det"]),
        "Rec.model_path": str(models["rec"]),
        "Rec.rec_keys_path": str(models["rec_keys"]),
        "Rec.lang_type": settings.lang_rec,
        "EngineConfig.onnxruntime.use_cuda": device.startswith("cuda"),
    }
    if device.startswith("cuda:"):
        params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] = int(device.split(":", 1)[1])
    return RapidOCR(params=_enum_params(params))


def _onnx_providers() -> list[str]:
    """Available onnxruntime execution providers, ``[]`` without onnxruntime."""
    try:
        import onnxruntime
    except ImportError:
        return []
    return list(onnxruntime.get_available_providers())


def _runner_output(output: Any) -> tuple[list[str], list[float], list[tuple[float, ...]]]:
    """Normalise a RapidOCROutput to (texts, scores, pixel bboxes)."""
    if output is None:
        return [], [], []
    texts = list(getattr(output, "txts", None) or [])
    scores = [float(s) for s in (getattr(output, "scores", None) or [])]
    boxes = getattr(output, "boxes", None)
    flattened: list[tuple[float, ...]] = []
    if boxes is not None:
        import numpy as np

        for box in np.asarray(boxes):
            xs, ys = box[:, 0], box[:, 1]
            flattened.append(
                (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            )
    return [str(t) for t in texts], scores, flattened


# --------------------------------------------------------------------------- rendering


def _render_pdf(data: bytes, *, dpi: int, max_side_px: int) -> list[Any]:
    """Render every page to a BGR ndarray at ``dpi`` (long side capped)."""
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise TierUnavailable(
            "T2 rendering needs pypdfium2 (and numpy): install the extraction extra with "
            "`uv sync --all-extras`"
        ) from exc
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        raise TierUnavailable(f"rapidocr could not open the PDF: {exc}") from exc
    images: list[Any] = []
    scale = dpi / 72.0
    for page in pdf:
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil().convert("RGB")
        array = np.asarray(pil)[:, :, ::-1]  # RGB → BGR (cv2 convention)
        long_side = max(array.shape[0], array.shape[1])
        if long_side > max_side_px:
            factor = max_side_px / long_side
            pil = pil.resize(
                (max(1, int(pil.width * factor)), max(1, int(pil.height * factor)))
            )
            array = np.asarray(pil)[:, :, ::-1]
        images.append(np.ascontiguousarray(array))
        page.close()
    pdf.close()
    return images


def _is_blank(image: Any) -> bool:
    """A page with a single uniform colour carries no ink."""
    if image is None or image.size == 0:
        return True
    return bool(image.min() == image.max())


# --------------------------------------------------------------------------- helpers


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY


def _find_model(model_id: str, root: Path | None, *, allow_bundled: bool) -> Path | None:
    """Locate a pinned model file: configured root first, runner bundle second.

    Only the detection model may fall back to the runner wheel's bundled copy;
    recognition models and dictionaries must exist at the configured root so
    the sha256-pinned artifacts (not a surprise download) are what runs.
    """
    if not model_id:
        return None
    if root is not None:
        candidate = root / model_id
        if candidate.is_file():
            return candidate
    if allow_bundled:
        try:
            import rapidocr as runner_pkg
        except ImportError:
            return None
        bundled = Path(runner_pkg.__file__).parent / "models" / model_id
        if bundled.is_file():
            return bundled
    return None


def _engine_meta(
    backend: RapidOCRBackend,
    models: dict[str, Path],
    *,
    page_rows: list[dict[str, Any]],
    dropped_total: int,
    digits_total: int,
    conjunct_pages: list[int],
    scores: list[float],
) -> dict[str, Any]:
    """The reproducibility block: engine, versions, model ids, device, flags."""
    pages_public = [
        {key: value for key, value in row.items() if key != "kept_scores"}
        for row in page_rows
    ]
    return {
        "engine": ENGINE_NAME,
        "engine_version": _version("rapidocr"),
        "onnxruntime_version": _version("onnxruntime"),
        "model_ids": {
            "det": backend.settings.det_model_id,
            "rec": backend.settings.rec_model_id,
            "rec_keys": backend.settings.rec_keys_id,
            "paths": {key: str(path) for key, path in models.items()},
        },
        "model_source": backend.settings.model_source,
        "model_sha256": backend.settings.model_sha256,
        "lang_rec": backend.settings.lang_rec,
        "granularity": "block",
        "device": backend.device,
        "device_reason": backend.device_reason,
        "min_box_confidence": backend.settings.min_box_confidence,
        "dropped_low_confidence_total": dropped_total,
        "devanagari_digits_normalized_total": digits_total,
        "conjunct_risk_pages": conjunct_pages,
        "confidence_min": round(min(scores), 4) if scores else None,
        "confidence_mean": (
            round(sum(scores) / len(scores), 4) if scores else None
        ),
        "pages": pages_public,
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = [
    "ACQUISITION_BUDGET_EXCEEDED",
    "CUDA_PROBE_FAILED",
    "CUDA_UNAVAILABLE_NO_PROVIDER",
    "CUDA_UNAVAILABLE_VRAM",
    "ENGINE_NAME",
    "INSTALL_COMMAND",
    "LANG_REC_DEVANAGARI",
    "LANG_REC_VALUES",
    "NO_TEXT_LAYER",
    "PREFETCH_COMMAND",
    "RapidOCRBackend",
    "RapidOCRSettings",
    "build_rapidocr_runner",
    "conjunct_risk",
    "has_devanagari",
    "normalise_devanagari_digits",
    "vram_free_gib",
]


def _enum_params(params: dict) -> dict:
    """rapidocr 3.10 requires Enum values, not strings (TypeError otherwise).

    Discovered the hard way: the runner was mocked in tests, so this only surfaced
    on the first real prefetch. Maps the string config to the package's enums when
    they are importable, and leaves already-Enum values untouched.
    """
    from contextlib import suppress

    try:
        from rapidocr.utils.typings import LangRec, ModelType, OCRVersion
    except Exception:  # older runner without these enums
        return params
    mapping = {
        "Det.ocr_version": OCRVersion,
        "Rec.ocr_version": OCRVersion,
        "Det.model_type": ModelType,
        "Rec.model_type": ModelType,
        "Rec.lang_type": LangRec,
    }
    out = dict(params)
    for key, enum_cls in mapping.items():
        value = out.get(key)
        if isinstance(value, str):
            # leave invalid values alone; the runner reports a clear error
            with suppress(ValueError):
                out[key] = enum_cls(value)
    return out
