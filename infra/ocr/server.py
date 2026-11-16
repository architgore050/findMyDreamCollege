"""PaddleOCR-VL service: page image / PDF page -> ordered text blocks.

Why this exists: ``fmd.acquire.tiers`` (T2) posts ``{"pdf_base64": ...}`` to
``$FMD_T2_URL/extract`` and expects

    {"blocks": [{"block_bbox": [x0, y0, x1, y1],
                 "block_order": <int>,
                 "block_content": "<text>",
                 "page": <1-based page>}, ...],
     "pages":  [{"page": 1, "width": W, "height": H, "bbox": [0, 0, W, H]}],
     "tool": "...", "tool_version": "..."}

That is exactly the shape PaddleOCR's doc-parser results carry as
``parsing_res_list``; this module only serialises them and never reinterprets
text. Coordinates are page-image pixels at ``FMD_OCR_DPI`` (default 200) and the
matching ``pages[]`` width/height are the rendered image size, so a block box
always resolves inside its page box.

Endpoints
    GET  /health   -> {"status": "ok"|"degraded", "engine": ..., "model": ...}
    POST /extract  -> the contract above (pdf_base64 or image_base64)

Nothing here is a "model v1 default": the model id and DPI come from the
environment so the acquire tier can be repointed without editing code.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_ID = os.environ.get("FMD_OCR_MODEL", "PaddleOCR-VL-1.6")
DPI = int(os.environ.get("FMD_OCR_DPI", "200"))
MAX_PAGES = int(os.environ.get("FMD_OCR_MAX_PAGES", "40"))
DEVICE = os.environ.get("FMD_OCR_DEVICE", "gpu:0")

app = FastAPI(title="fmd-paddleocr-vl", version="1")

_pipeline = None
_load_error: str | None = None


def _load() -> object:
    """Lazily build the PaddleOCR-VL pipeline; keep the error for /health."""
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        if _pipeline is None:
            raise RuntimeError(_load_error)
        return _pipeline
    try:
        from paddleocr import PaddleOCRVL

        _pipeline = PaddleOCRVL(device=DEVICE)
    except Exception as exc:  # pragma: no cover - depends on host CUDA stack
        _load_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(_load_error) from exc
    return _pipeline


class ExtractRequest(BaseModel):
    pdf_base64: str | None = None
    image_base64: str | None = None


def _render_pdf(data: bytes) -> list[tuple[int, int, int, bytes]]:
    """PDF -> [(page_no, width_px, height_px, png_bytes)] using pdfium (BSD)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    out: list[tuple[int, int, int, bytes]] = []
    try:
        for index in range(min(len(pdf), MAX_PAGES)):
            page = pdf[index]
            bitmap = page.render(scale=DPI / 72.0)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            out.append((index + 1, image.width, image.height, buffer.getvalue()))
    finally:
        pdf.close()
    return out


def _as_dict(result: object) -> dict:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, dict):
        return payload
    if isinstance(result, dict):
        return result
    return {}


def _blocks_from(result: object, *, page_no: int, order_start: int) -> list[dict]:
    payload = _as_dict(result)
    res = payload.get("res", payload)
    items = res.get("parsing_res_list") or res.get("layout_parsing_result") or []
    blocks: list[dict] = []
    for offset, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = item.get("block_content") or item.get("block_text") or ""
        if not str(text).strip():
            continue
        bbox = item.get("block_bbox") or item.get("bbox")
        order = item.get("block_order")
        blocks.append(
            {
                "page": page_no,
                "block_order": int(order) + order_start if isinstance(order, int) else order_start + offset,
                "block_bbox": [float(value) for value in bbox] if bbox else None,
                "block_content": str(text),
                "block_label": item.get("block_label"),
            }
        )
    return blocks


def _run(image_bytes: bytes, *, page_no: int, order_start: int) -> list[dict]:
    pipeline = _load()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as handle:
        handle.write(image_bytes)
        handle.flush()
        try:
            results = pipeline.predict(handle.name)
        except TypeError:  # older signature: positional only
            results = pipeline.predict(handle.name, use_doc_orientation_classify=False)
    if isinstance(results, dict):
        results = [results]
    blocks: list[dict] = []
    for result in results or []:
        blocks.extend(_blocks_from(result, page_no=page_no, order_start=order_start + len(blocks)))
    return blocks


@app.get("/health")
def health() -> dict:
    try:
        _load()
    except Exception as exc:
        return {"status": "degraded", "engine": "paddleocr-vl", "model": MODEL_ID, "error": str(exc)}
    return {"status": "ok", "engine": "paddleocr-vl", "model": MODEL_ID}


@app.post("/extract")
def extract(request: ExtractRequest) -> dict:
    if request.pdf_base64:
        pages = _render_pdf(base64.b64decode(request.pdf_base64))
        images = [(page_no, width, height, data) for page_no, width, height, data in pages]
    elif request.image_base64:
        from PIL import Image

        raw = base64.b64decode(request.image_base64)
        size = Image.open(io.BytesIO(raw)).size
        images = [(1, size[0], size[1], raw)]
    else:
        raise HTTPException(status_code=422, detail="pdf_base64 or image_base64 is required")

    try:
        version = __import__("importlib.metadata").metadata.version("paddleocr")
    except Exception:
        version = None

    blocks: list[dict] = []
    page_rows: list[dict] = []
    order = 0
    for page_no, width, height, data in images:
        try:
            page_blocks = _run(data, page_no=page_no, order_start=order)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"OCR failed on page {page_no}: {exc}") from exc
        order += len(page_blocks)
        blocks.extend(page_blocks)
        page_rows.append(
            {"page": page_no, "width": width, "height": height, "bbox": [0, 0, width, height]}
        )

    return {
        "tool": "paddleocr-vl",
        "tool_version": version,
        "model": MODEL_ID,
        "pages": page_rows,
        "blocks": blocks,
    }
