"""OCR stage service (T2 / T2b): page image or PDF -> ordered text blocks.

``fmd.acquire.tiers`` posts ``{"pdf_base64": ...}`` (or ``{"image_base64": ...}``)
to ``$FMD_T2_URL/extract`` and expects exactly

    {"blocks": [{"page": 1, "block_bbox": [x0, y0, x1, y1],
                 "block_order": 0, "block_content": "<text>"}, ...],
     "pages":  [{"page": 1, "width": W, "height": H, "bbox": [0, 0, W, H]}],
     "tool": "...", "tool_version": "...", "model": "..."}

Two engines, one contract, chosen by ``config/acquire.yaml`` -> ``backends.T2.service``
(or ``T2b`` when ``FMD_OCR_TIER=T2b``):

``qwen3vl``
    Qwen3-VL-8B-Instruct served by vLLM's OpenAI-compatible endpoint. This is the
    default because vLLM has sm_120 (Blackwell) kernels and Qwen3-VL-8B is the
    only local option with a published real-scan Devanagari number (chrF++ 75.2,
    arXiv:2606.29213). The model is asked for a JSON block list; the boxes it
    returns are scaled from a 0..1000 grid onto the analysed raster, clamped to
    the page, and re-sorted into reading order. Coordinates are therefore
    *model-reported*: the tier is block-granular and deliberately excluded from
    the bbox gate in ``config/acquire.yaml:bbox.gate_tiers``.

``paddleocr``
    PaddleOCR-VL, kept as the alternative engine. PaddlePaddle's GPU wheels stop
    at sm_90, so on an sm_120 host this backend is CPU-only (correct, slow).
    Selected with ``FMD_OCR_BACKEND=paddleocr`` or ``FMD_OCR_TIER=T2b``.

Honesty rules (the reason this file is careful):

* weights missing, vLLM down, a model-name mismatch, or a 429/500 from the
  engine are **not** successes. ``GET /health`` reports ``status: unavailable``
  with a ``reason``, and ``POST /extract`` answers 503 with that same reason, so
  the acquire tier raises ``TierUnavailable`` and the page goes to the human
  queue instead of being published as an empty document.
* a page that produced no blocks is only acceptable when the raster is blank.
  A non-blank page with zero blocks is ``empty_result`` (503), never ``200``.

Endpoints
    GET  /health   -> {"status": "ok"|"unavailable", "reason": ..., ...}
    POST /extract  -> the contract above (pdf_base64 or image_base64)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

TOOL = "fmd-ocr"
ENGINES = ("qwen3vl", "paddleocr")
BBOX_SPACES = ("normalized_1000", "pixels")
# A block list wrapped in prose or a ```json fence is common; recover it.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class BackendUnavailable(RuntimeError):
    """The engine cannot answer: no weights, no endpoint, wrong served model."""


class BackendFailure(RuntimeError):
    """The engine answered, but with an error or an unusable body."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class Settings:
    """Service configuration: YAML registry first, environment on top."""

    tier: str = "T2"
    backend: str = ""
    served_name: str = ""
    vllm_url: str = ""
    dpi: int = 200
    max_pages: int = 40
    max_side_px: int = 2400
    timeout_s: float = 120.0
    json_mode: bool = True
    bbox_space: str = "normalized_1000"
    device: str = "cpu"
    hub_id: str = ""
    config_path: str = ""
    config_error: str = ""

    @classmethod
    def load(
        cls,
        *,
        env: dict[str, str] | None = None,
        config_path: str | Path | None = None,
    ) -> Settings:
        env = dict(os.environ if env is None else env)
        tier = env.get("FMD_OCR_TIER", "T2")
        block, path, error = _service_block(tier, env=env, config_path=config_path)

        def pick(key: str, default: Any) -> Any:
            value = env.get(key)
            return default if value is None or value == "" else value

        return cls(
            tier=tier,
            backend=str(pick("FMD_OCR_BACKEND", block.get("backend", ""))),
            served_name=str(pick("FMD_OCR_SERVED_NAME", block.get("served_name", ""))),
            vllm_url=str(pick("FMD_OCR_VLLM_URL", block.get("vllm_url", ""))),
            dpi=int(pick("FMD_OCR_DPI", block.get("dpi", 200))),
            max_pages=int(pick("FMD_OCR_MAX_PAGES", block.get("max_pages", 40))),
            max_side_px=int(pick("FMD_OCR_MAX_SIDE_PX", block.get("max_side_px", 2400))),
            timeout_s=float(pick("FMD_OCR_TIMEOUT_S", block.get("timeout_s", 120))),
            json_mode=_as_bool(pick("FMD_OCR_JSON_MODE", block.get("json_mode", True))),
            bbox_space=str(pick("FMD_OCR_BBOX_SPACE", block.get("bbox_space", "normalized_1000"))),
            device=str(pick("FMD_OCR_DEVICE", block.get("device", "cpu"))),
            hub_id=str(block.get("hub_id", "")),
            config_path=str(path or ""),
            config_error=error,
        )

    def validate_for_serving(self) -> None:
        if self.config_error:
            raise BackendUnavailable(f"config_unreadable: {self.config_error}")
        if self.backend not in ENGINES:
            raise BackendUnavailable(
                f"backend_unknown: {self.backend!r} is not one of {', '.join(ENGINES)}; "
                f"set it in {self.config_path or 'config/acquire.yaml'} or FMD_OCR_BACKEND"
            )
        if not self.served_name:
            raise BackendUnavailable("model_unset: no served name in config or FMD_OCR_SERVED_NAME")
        if self.bbox_space not in BBOX_SPACES:
            raise BackendUnavailable(
                f"bbox_space_unknown: {self.bbox_space!r} is not one of {', '.join(BBOX_SPACES)}"
            )
        if self.backend == "qwen3vl" and not self.vllm_url:
            raise BackendUnavailable("vision_endpoint_unset: FMD_OCR_VLLM_URL is empty")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _service_block(
    tier: str,
    *,
    env: dict[str, str],
    config_path: str | Path | None,
) -> tuple[dict[str, Any], Path | None, str]:
    """Read ``backends.<tier>.service`` from the registry. Never raises."""
    candidates: list[Path] = []
    explicit = config_path or env.get("FMD_OCR_CONFIG")
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.append(Path("/config/acquire.yaml"))
        candidates.append(Path(__file__).resolve().parents[2] / "config" / "acquire.yaml")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - PyYAML is a hard dep of the image
            return {}, candidate, f"ImportError: {exc}"
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            block = data["backends"][tier]["service"]
            if not isinstance(block, dict):
                raise TypeError(f"backends.{tier}.service is not a mapping")
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            # An unreadable registry is a health failure with a reason, not a crash.
            return {}, candidate, f"{type(exc).__name__}: {exc}"
        return dict(block), candidate, ""
    return {}, None, f"no acquire.yaml found; looked in {[str(p) for p in candidates]}"


# --------------------------------------------------------------------------- raster


@dataclass(frozen=True)
class PageImage:
    """One analysed page: PNG bytes plus the size the boxes refer to."""

    page: int
    width: int
    height: int
    png: bytes
    blank: bool


def _blank(image: Any) -> bool:
    """A page is blank when (almost) no pixel is ink. Deterministic, no model."""
    grey = image.convert("L")
    dark = sum(grey.histogram()[:200])
    return dark < max(32, (grey.width * grey.height) // 500)


def _encode(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _bounded(image: Any, max_side_px: int) -> Any:
    """Downscale so the long side fits; keeps aspect ratio. Never upscales."""
    longest = max(image.width, image.height)
    if longest <= max_side_px:
        return image
    scale = max_side_px / float(longest)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size)


def _render_pdf(data: bytes, settings: Settings) -> list[PageImage]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    pages: list[PageImage] = []
    try:
        for index in range(min(len(pdf), settings.max_pages)):
            page = pdf[index]
            image = page.render(scale=settings.dpi / 72.0).to_pil()
            image = _bounded(image, settings.max_side_px)
            pages.append(
                PageImage(
                    page=index + 1,
                    width=image.width,
                    height=image.height,
                    png=_encode(image),
                    blank=_blank(image),
                )
            )
    finally:
        pdf.close()
    return pages


def _render_image(data: bytes, settings: Settings) -> list[PageImage]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as opened:
        image = _bounded(opened.convert("RGB"), settings.max_side_px)
        return [
            PageImage(page=1, width=image.width, height=image.height, png=_encode(image), blank=_blank(image))
        ]


# --------------------------------------------------------------------------- parsing


def _text_of(item: dict[str, Any]) -> str:
    for key in ("block_content", "text", "content", "line"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _payload_blocks(content: str) -> list[Any]:
    """The engine's message -> block list. Recovers fenced/embedded JSON."""
    text = _FENCE.sub("", content.strip())
    data: Any = None
    for candidate in (text, _embedded_json(text)):
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
            break
        except (TypeError, ValueError):
            continue
    if data is None:
        raise BackendFailure("malformed_output", f"engine did not return JSON: {content[:200]!r}")
    if isinstance(data, dict):
        for key in ("blocks", "lines", "text_blocks", "result"):
            if isinstance(data.get(key), list):
                return data[key]
        raise BackendFailure("malformed_output", f"JSON has no block list: {content[:200]!r}")
    if isinstance(data, list):
        return data
    raise BackendFailure("malformed_output", f"JSON is a {type(data).__name__}: {content[:200]!r}")


def _embedded_json(text: str) -> str | None:
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        return None
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1] if end > start else None


def _scale_box(
    raw: Any, *, width: int, height: int, space: str
) -> tuple[float, float, float, float] | None:
    """Engine box -> page pixels, clamped. ``None`` when it cannot be trusted."""
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    # A box whose values exceed the 0..1000 grid is already in pixels; this is
    # the only fallback, and it is one-directional (never the reverse).
    normalised = space == "normalized_1000" and max(values) <= 1000.0
    if normalised:
        values = [values[0] / 1000 * width, values[1] / 1000 * height,
                  values[2] / 1000 * width, values[3] / 1000 * height]
    x0 = min(max(values[0], 0.0), float(width))
    y0 = min(max(values[1], 0.0), float(height))
    x1 = min(max(values[2], 0.0), float(width))
    y1 = min(max(values[3], 0.0), float(height))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return (x0, y0, x1, y1)


def normalise_blocks(
    items: list[Any],
    *,
    page: PageImage,
    order_start: int = 0,
    space: str = "normalized_1000",
) -> list[dict[str, Any]]:
    """Engine block list -> contract blocks in reading order.

    Ordering is by top edge then left edge (top-to-bottom, left-to-right). Blocks
    without a usable box keep the engine's order at the end. ``block_order`` is
    therefore monotonically increasing per page and the page's blocks sort the
    same way the caller's ``_ocr_doc`` sorts them.
    """
    candidates: list[tuple[int, str, tuple[float, float, float, float] | None]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = _text_of(item)
        if not text:
            continue
        box = _scale_box(
            item.get("block_bbox") or item.get("bbox"), width=page.width, height=page.height, space=space
        )
        candidates.append((index, text, box))
    candidates.sort(
        key=lambda entry: (
            (entry[2][1], entry[2][0]) if entry[2] is not None else (float("inf"), float(entry[0]))
        )
    )
    blocks: list[dict[str, Any]] = []
    for offset, (_, text, box) in enumerate(candidates):
        blocks.append(
            {
                "page": page.page,
                "block_order": order_start + offset,
                "block_bbox": list(box) if box is not None else None,
                "block_content": text,
            }
        )
    return blocks


# --------------------------------------------------------------------------- engines


class Qwen3VLBackend:
    """Qwen3-VL behind vLLM's OpenAI-compatible `/chat/completions`."""

    name = "qwen3vl"

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport

    # -- readiness -------------------------------------------------------
    def check(self) -> tuple[bool, str]:
        url = self.settings.vllm_url.rstrip("/") + "/models"
        try:
            with self._client() as client:
                response = client.get(url, timeout=5.0)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return False, (
                f"vision_unavailable: vLLM is not answering at {self.settings.vllm_url} ({exc}); "
                "start it with `scripts/serve_ocr.sh vision-start`"
            )
        served = [str(row.get("id")) for row in (payload or {}).get("data") or [] if isinstance(row, dict)]
        if self.settings.served_name not in served:
            return False, (
                f"model_not_served: {self.settings.served_name!r} is not loaded at "
                f"{self.settings.vllm_url} (served: {served or 'none'}); a text model on this "
                "port cannot read scans — start the vision weights"
            )
        return True, ""

    # -- extraction ------------------------------------------------------
    def extract_page(self, page: PageImage) -> list[Any]:
        body: dict[str, Any] = {
            "model": self.settings.served_name,
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + base64.b64encode(page.png).decode()},
                        },
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                },
            ],
        }
        if self.settings.json_mode:
            body["response_format"] = {"type": "json_object"}
        response = self._post(body)
        if response.status_code == 400 and self.settings.json_mode:
            # Older vLLM builds reject response_format for vision inputs; retry
            # once without it rather than declaring the page unreadable.
            body.pop("response_format", None)
            response = self._post(body)
        if response.status_code != 200:
            raise BackendFailure(
                f"backend_http_{response.status_code}",
                f"{self.settings.served_name} at {self.settings.vllm_url}: {response.text[:300]}",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendFailure("malformed_output", f"engine returned non-JSON: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendFailure("malformed_output", f"no choices[0].message.content: {payload}") from exc
        if isinstance(content, list):  # some builds answer with content parts
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return _payload_blocks(str(content))

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        url = self.settings.vllm_url.rstrip("/") + "/chat/completions"
        try:
            with self._client() as client:
                return client.post(url, json=body, timeout=self.settings.timeout_s)
        except httpx.HTTPError as exc:
            raise BackendFailure("backend_unreachable", str(exc)) from exc

    def _client(self) -> httpx.Client:
        return httpx.Client(transport=self._transport)


# Qwen3-VL needs the task, the output shape, and the coordinate rule in one
# prompt; anything vaguer produces markdown tables instead of block JSON.
_SYSTEM_PROMPT = (
    "You are an OCR engine for Indian college documents. Transcribe every text "
    "block on the page verbatim. Never translate, summarise, correct, or invent "
    "text. Preserve Devanagari, digits and punctuation exactly as printed."
)
_USER_PROMPT = (
    'Return JSON only: {"blocks": [{"bbox": [x0, y0, x1, y1], "text": "..."}]}. '
    "One entry per text block, in natural reading order (top to bottom, then left "
    "to right). bbox is the block's rectangle on the image scaled to a 0-1000 grid "
    "for width and height. If the page has no text, return {\"blocks\": []}."
)


class PaddleBackend:
    """PaddleOCR-VL. GPU-less on sm_120; correct on CPU, slow."""

    name = "paddleocr"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: Any = None

    def check(self) -> tuple[bool, str]:
        try:
            self._load()
        except BackendUnavailable as exc:
            return False, str(exc)
        return True, ""

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from paddleocr import PaddleOCRVL
        except Exception as exc:  # ImportError, or a broken CUDA stack at import
            raise BackendUnavailable(
                f"engine_not_installed: paddleocr is not importable ({type(exc).__name__}: {exc}); "
                "this backend is CPU-only on sm_120 hosts"
            ) from exc
        try:
            self._pipeline = PaddleOCRVL(device=self.settings.device)
        except Exception as exc:
            raise BackendUnavailable(
                f"engine_start_failed: PaddleOCR-VL could not start on {self.settings.device} "
                f"({type(exc).__name__}: {exc}); Paddle GPU wheels stop at sm_90, use CPU"
            ) from exc
        return self._pipeline

    def extract_page(self, page: PageImage) -> list[Any]:
        pipeline = self._load()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as handle:
            handle.write(page.png)
            handle.flush()
            try:
                results = pipeline.predict(handle.name)
            except Exception as exc:
                raise BackendFailure("backend_failed", f"PaddleOCR predict failed: {exc}") from exc
        if isinstance(results, dict):
            results = [results]
        items: list[Any] = []
        for result in results or []:
            payload = getattr(result, "json", None)
            if callable(payload):
                payload = payload()
            if not isinstance(payload, dict):
                payload = result if isinstance(result, dict) else {}
            res = payload.get("res", payload)
            items.extend(res.get("parsing_res_list") or res.get("layout_parsing_result") or [])
        return items


def backend_for(settings: Settings) -> Qwen3VLBackend | PaddleBackend:
    settings.validate_for_serving()
    if settings.backend == "qwen3vl":
        return Qwen3VLBackend(settings)
    return PaddleBackend(settings)


# --------------------------------------------------------------------------- app


class ExtractRequest(BaseModel):
    pdf_base64: str | None = None
    image_base64: str | None = None


def create_app(
    settings: Settings | None = None,
    *,
    backend: Any = None,
) -> FastAPI:
    """Build the app. Tests pass a settings object and/or a fake backend."""
    resolved = settings or Settings.load()
    app = FastAPI(title="fmd-ocr", version="2")
    app.state.settings = resolved
    app.state.backend = backend

    def get_backend() -> Any:
        if app.state.backend is None:
            app.state.backend = backend_for(app.state.settings)
        return app.state.backend

    @app.get("/health")
    def health() -> dict[str, Any]:
        row: dict[str, Any] = {
            "status": "unavailable",
            "tool": TOOL,
            "engine": resolved.backend or "unset",
            "model": resolved.served_name or None,
            "tier": resolved.tier,
            "config": resolved.config_path or None,
        }
        try:
            engine = get_backend()
        except BackendUnavailable as exc:
            row["reason"] = str(exc)
            return row
        ready, reason = engine.check()
        row["engine"] = engine.name
        if not ready:
            row["reason"] = reason
            return row
        row["status"] = "ok"
        return row

    @app.post("/extract")
    def extract(request: ExtractRequest) -> dict[str, Any]:
        try:
            engine = get_backend()
        except BackendUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        ready, reason = engine.check()
        if not ready:
            raise HTTPException(status_code=503, detail=reason)

        try:
            if request.pdf_base64:
                pages = _render_pdf(base64.b64decode(request.pdf_base64), resolved)
            elif request.image_base64:
                pages = _render_image(base64.b64decode(request.image_base64), resolved)
            else:
                raise HTTPException(status_code=422, detail="pdf_base64 or image_base64 is required")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"could not decode input: {exc}") from exc
        if not pages:
            raise HTTPException(status_code=422, detail="input contained no pages")

        blocks: list[dict[str, Any]] = []
        page_rows: list[dict[str, Any]] = []
        order = 0
        unreadable: list[int] = []
        for page in pages:
            page_blocks: list[dict[str, Any]] = []
            if not page.blank:
                try:
                    raw = engine.extract_page(page)
                except BackendUnavailable as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except BackendFailure as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                page_blocks = normalise_blocks(
                    raw, page=page, order_start=order, space=resolved.bbox_space
                )
                if not page_blocks:
                    unreadable.append(page.page)
            order += len(page_blocks)
            blocks.extend(page_blocks)
            page_rows.append(
                {
                    "page": page.page,
                    "width": page.width,
                    "height": page.height,
                    "bbox": [0, 0, page.width, page.height],
                    "blank": page.blank,
                }
            )
        if unreadable:
            # Never let "the engine said nothing" look like a clean empty page.
            raise HTTPException(
                status_code=503,
                detail=(
                    f"empty_result: no blocks on non-blank page(s) {unreadable} from "
                    f"{resolved.served_name or resolved.backend}; use a clearer scan or route "
                    "the document to the human queue"
                ),
            )

        return {
            "tool": TOOL,
            "engine": resolved.backend,
            "tool_version": _version(str(getattr(engine, "name", resolved.backend))),
            "model": resolved.served_name,
            "pages": page_rows,
            "blocks": blocks,
        }

    return app


def _version(engine: str) -> str | None:
    """Version of the locally installed engine package, when there is one.

    The vLLM vision backend runs in another process: its build id is not ours to
    report, so `tool_version` stays None and `model` carries the served id.
    """
    import importlib.metadata

    distribution = {"paddleocr": "paddleocr"}.get(engine)
    if distribution is None:
        return None
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


app = create_app()
