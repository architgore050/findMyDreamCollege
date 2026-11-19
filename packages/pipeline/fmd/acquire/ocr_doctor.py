"""Install/runtime diagnostics for the in-process T2 OCR runner.

`fmd llm doctor` prints one line for OCR so a missing install or absent model
is visible instead of surfacing later as a silent ``TierUnavailable``:

``ocr  engine=rapidocr  device=cpu  models=present  provider=cpu  status=ok``

Everything here is a *probe*, never a repair: no download, no model load, no
network. ``models=present`` means the configured det/rec files (and the rec
dictionary) exist on disk under the registry's artifacts root
(``FMD_RAPIDOCR_MODELS``, the same env the backend reads); it is never inferred
from a successful import alone. The det graph may legitimately live inside the
runner wheel, exactly as ``RapidOCRBackend.resolve_models`` allows.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fmd.acquire.rapidocr_backend import (
    CUDA_PROBE_FAILED,
    CUDA_UNAVAILABLE_NO_PROVIDER,
    CUDA_UNAVAILABLE_VRAM,
    ENGINE_NAME,
    RapidOCRSettings,
    vram_free_gib,
)

#: Stable status reasons (never prose). ``ok`` is the only green value.
RAPIDOCR_MISSING = "rapidocr_missing"
ONNXRUNTIME_MISSING = "onnxruntime_missing"
MODELS_MISSING = "models_missing"
CONFIG_INVALID = "config_invalid"
STATUS_OK = "ok"

CUDA_PROVIDER = "CUDAExecutionProvider"


def _package_present(name: str) -> bool:
    """Is ``name`` installed? Uses ``find_spec`` so nothing is imported."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _package_models_dir(name: str) -> Path | None:
    """The runner wheel's bundled ``models/`` directory, without importing it."""
    try:
        found = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if found is None or not found.origin:
        return None
    return Path(found.origin).parent / "models"


def _find_file(model_id: str, root: Path | None, *, bundled: Path | None) -> Path | None:
    """Configured root first, runner bundle second (mirrors the backend)."""
    if not model_id:
        return None
    if root is not None:
        candidate = root / model_id
        if candidate.is_file():
            return candidate
    if bundled is not None:
        candidate = bundled / model_id
        if candidate.is_file():
            return candidate
    return None


def _format_line(
    *,
    device: str,
    models: str,
    provider: str,
    status: str,
) -> str:
    return (
        f"ocr  engine={ENGINE_NAME}  device={device}  models={models}  "
        f"provider={provider}  status={status}"
    )


def diagnose_ocr(
    spec: Any,
    *,
    env: Mapping[str, str] | None = None,
    providers: list[str] | None = None,
    vram_probe: Callable[[], float | None] | None = None,
    package_present: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Probe the configured T2 RapidOCR runner and return one diagnosable row.

    ``providers``/``vram_probe``/``package_present`` are injectable so tests
    exercise every reason without importing onnxruntime, touching nvidia-smi or
    reading the host filesystem.
    """
    env = dict(os.environ if env is None else env)
    present = package_present or _package_present
    rapidocr_present = present("rapidocr")
    onnxruntime_present = present("onnxruntime")

    try:
        settings = RapidOCRSettings.from_spec(spec, env=env)
        config_error: str | None = None
    except Exception as exc:  # config is external input; report, never crash doctor
        settings = None
        config_error = f"{type(exc).__name__}: {exc}"

    if settings is None:
        reason = CONFIG_INVALID
        return {
            "engine": ENGINE_NAME,
            "device": "unknown",
            "models": "missing",
            "provider": "cpu",
            "status": f"unavailable:{reason}",
            "reason": reason,
            "detail": config_error,
            "line": _format_line(
                device="unknown",
                models="missing",
                provider="cpu",
                status=f"unavailable:{reason}",
            ),
        }

    device = settings.device
    bundled = _package_models_dir("rapidocr") if rapidocr_present else None
    root = settings.models_root
    det = _find_file(settings.det_model_id, root, bundled=bundled)
    rec = _find_file(settings.rec_model_id, root, bundled=None)
    rec_keys = _find_file(settings.rec_keys_id, root, bundled=None)
    models_present = det is not None and rec is not None and rec_keys is not None
    models = "present" if models_present else "missing"

    if providers is not None:
        available = list(providers)
    elif onnxruntime_present:
        from fmd.acquire.rapidocr_backend import _onnx_providers

        available = _onnx_providers()
    else:
        available = []
    provider = "cuda" if CUDA_PROVIDER in available else "cpu"

    reason: str | None = None
    detail: str | None = None
    if not rapidocr_present:
        reason = RAPIDOCR_MISSING
        detail = "rapidocr is not installed; run `uv sync --extra ocr`"
    elif not onnxruntime_present:
        reason = ONNXRUNTIME_MISSING
        detail = "onnxruntime is not installed; run `uv sync --extra ocr`"
    elif not models_present:
        reason = MODELS_MISSING
        root_label = str(root) if root else "FMD_RAPIDOCR_MODELS (unset)"
        missing = [
            name
            for name, path in (
                (settings.det_model_id, det),
                (settings.rec_model_id, rec),
                (settings.rec_keys_id, rec_keys),
            )
            if path is None
        ]
        detail = f"not found under {root_label}: {', '.join(missing)}"
    elif device.startswith("cuda"):
        if CUDA_PROVIDER not in available:
            reason = CUDA_UNAVAILABLE_NO_PROVIDER
            detail = "onnxruntime reports no CUDAExecutionProvider; run on cpu"
        else:
            probe = vram_probe or vram_free_gib
            free = probe()
            if free is None:
                reason = CUDA_PROBE_FAILED
                detail = "free VRAM could not be measured (nvidia-smi)"
            elif free < settings.min_free_vram_gib:
                reason = CUDA_UNAVAILABLE_VRAM
                detail = f"{free:.2f} GiB free < min_free_vram_gib {settings.min_free_vram_gib:g}"

    status = STATUS_OK if reason is None else f"unavailable:{reason}"
    return {
        "engine": ENGINE_NAME,
        "device": device,
        "models": models,
        "provider": provider,
        "status": status,
        "reason": reason,
        "detail": detail,
        "line": _format_line(device=device, models=models, provider=provider, status=status),
    }


__all__ = [
    "CONFIG_INVALID",
    "CUDA_PROVIDER",
    "MODELS_MISSING",
    "ONNXRUNTIME_MISSING",
    "RAPIDOCR_MISSING",
    "STATUS_OK",
    "diagnose_ocr",
]
