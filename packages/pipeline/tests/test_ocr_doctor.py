"""`fmd llm doctor` OCR line — install and model presence made visible.

No network, no model download, no import of rapidocr/onnxruntime: package
presence, execution providers and VRAM are injected. Filesystem checks run
against ``tmp_path`` via the same ``FMD_RAPIDOCR_MODELS`` env the backend reads.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fmd.acquire import ocr_doctor
from fmd.acquire.ocr_doctor import (
    CONFIG_INVALID,
    MODELS_MISSING,
    ONNXRUNTIME_MISSING,
    RAPIDOCR_MISSING,
    diagnose_ocr,
)
from fmd.cli import main

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def _spec(*, device: str = "cpu", root: str = "", min_vram: float = 2.0) -> Any:
    return SimpleNamespace(
        device=device,
        min_free_vram_gib=min_vram,
        models={
            "det": "PP-OCRv6_det_small.onnx",
            "rec": "devanagari_PP-OCRv5_rec_mobile.onnx",
            "rec_keys": "ppocrv5_devanagari_dict.txt",
            "artifacts_root": root,
            "sha256": {},
        },
    )


def _write_models(root: Path) -> None:
    for name in (
        "PP-OCRv6_det_small.onnx",
        "devanagari_PP-OCRv5_rec_mobile.onnx",
        "ppocrv5_devanagari_dict.txt",
    ):
        (root / name).write_text("fixture", encoding="utf-8")


def _installed(name: str) -> bool:
    return True


def _missing(*names: str):
    return lambda name: name not in names


def _env(root: Path | None = None, **extra: str) -> dict[str, str]:
    env = {"FMD_RAPIDOCR_MODELS": str(root) if root else ""}
    env.update(extra)
    return env


def test_ocr_ok_when_runner_models_and_provider_are_present(tmp_path: Path) -> None:
    _write_models(tmp_path)
    row = diagnose_ocr(
        _spec(root=str(tmp_path)),
        env=_env(tmp_path),
        providers=["CPUExecutionProvider"],
        package_present=_installed,
    )
    assert row["line"] == (
        "ocr  engine=rapidocr  device=cpu  models=present  provider=cpu  status=ok"
    )
    assert row["models"] == "present"
    assert row["reason"] is None


def test_ocr_missing_runner_is_never_reported_green(tmp_path: Path) -> None:
    _write_models(tmp_path)
    row = diagnose_ocr(
        _spec(root=str(tmp_path)),
        env=_env(tmp_path),
        providers=["CPUExecutionProvider"],
        package_present=_missing("rapidocr"),
    )
    assert row["status"] == f"unavailable:{RAPIDOCR_MISSING}"
    assert row["models"] == "present"  # files exist; the runner does not
    assert "uv sync --extra ocr" in row["detail"]


def test_ocr_missing_onnxruntime_is_unavailable(tmp_path: Path) -> None:
    _write_models(tmp_path)
    row = diagnose_ocr(
        _spec(root=str(tmp_path)),
        env=_env(tmp_path),
        package_present=_missing("onnxruntime"),
    )
    assert row["status"] == f"unavailable:{ONNXRUNTIME_MISSING}"
    assert row["provider"] == "cpu"


def test_ocr_importable_runner_without_model_files_is_missing(tmp_path: Path) -> None:
    # Runner importable, artifacts root empty: det may be bundled, but rec and
    # the dictionary are not — presence is decided by files, never by import.
    row = diagnose_ocr(
        _spec(root=str(tmp_path)),
        env=_env(tmp_path),
        providers=["CPUExecutionProvider"],
        package_present=_installed,
    )
    assert row["status"] == f"unavailable:{MODELS_MISSING}"
    assert row["models"] == "missing"
    assert "devanagari_PP-OCRv5_rec_mobile.onnx" in row["detail"]


def test_ocr_cuda_requested_without_cuda_provider_falls_back_to_cpu(
    tmp_path: Path,
) -> None:
    _write_models(tmp_path)
    row = diagnose_ocr(
        _spec(device="cuda", root=str(tmp_path)),
        env=_env(tmp_path, FMD_OCR_DEVICE="cuda"),
        providers=["CPUExecutionProvider"],
        package_present=_installed,
    )
    assert row["device"] == "cuda"
    assert row["provider"] == "cpu"
    assert row["status"] == ("unavailable:" + ocr_doctor.CUDA_UNAVAILABLE_NO_PROVIDER)


def test_ocr_cuda_below_vram_floor_is_named(tmp_path: Path) -> None:
    _write_models(tmp_path)
    row = diagnose_ocr(
        _spec(device="cuda", root=str(tmp_path), min_vram=2.0),
        env=_env(tmp_path, FMD_OCR_DEVICE="cuda"),
        providers=["CUDAExecutionProvider"],
        vram_probe=lambda: 0.5,
        package_present=_installed,
    )
    assert row["provider"] == "cuda"
    assert row["status"] == "unavailable:cuda_unavailable_vram"


def test_ocr_config_error_is_reported_not_raised() -> None:
    broken = SimpleNamespace(device="cpu", models={})
    row = diagnose_ocr(broken, env={}, package_present=_installed)
    assert row["status"] == f"unavailable:{CONFIG_INVALID}"
    assert row["reason"] == CONFIG_INVALID


def test_diagnose_ocr_does_not_import_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_models(tmp_path)
    for name in ("rapidocr", "onnxruntime"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    diagnose_ocr(
        _spec(root=str(tmp_path)),
        env=_env(tmp_path),
        providers=["CPUExecutionProvider"],
        package_present=_installed,
    )
    assert "rapidocr" not in sys.modules
    assert "onnxruntime" not in sys.modules


def test_ocr_extra_is_declared_and_base_install_stays_light() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    base = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]
    ocr = extras["ocr"]
    assert "rapidocr>=3.10,<4" in ocr
    assert "onnxruntime>=1.32,<2" in ocr
    assert "pypdfium2>=5,<6" in ocr
    assert not any(
        dep.split(">")[0].split("=")[0].strip().lower() in {"rapidocr", "onnxruntime"}
        for dep in base
    )
    for dep in ocr:
        assert dep in extras["all"]
    # `uv sync --all-extras` composes one environment: the extract pin must not
    # contradict the OCR pin.
    assert "pypdfium2>=5,<6" in extras["extract"]


def test_llm_doctor_table_prints_the_ocr_line(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--config-dir", str(CONFIG_DIR), "llm", "doctor"])
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("ocr  engine=rapidocr")]
    assert len(lines) == 1
    assert " device=" in lines[0]
    assert " models=" in lines[0]
    assert " provider=" in lines[0]
    assert " status=" in lines[0]
