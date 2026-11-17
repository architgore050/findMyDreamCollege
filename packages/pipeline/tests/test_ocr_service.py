"""Contract tests for the OCR stage service (``infra/ocr/server.py``).

Everything here is offline. The vision engine is either a fake backend or an
``httpx.MockTransport`` replaying vLLM responses, so CI needs no GPU, no weights,
and no network. The single test that talks to a live service is skipped unless
``FMD_OCR_E2E_URL`` is set, and it skips with the service's own reason when the
weights are not there.

The service contract is what ``fmd.acquire.tiers._ocr_doc`` consumes:
``POST /extract -> {blocks: [{page, block_bbox, block_order, block_content}],
pages: [{page, width, height, bbox}]}``.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = REPO_ROOT / "infra" / "ocr" / "server.py"
CONFIG_PATH = REPO_ROOT / "config" / "acquire.yaml"
FIXTURE = REPO_ROOT / "infra" / "ocr" / "fixtures" / "scan_sample.png"
E2E_URL = os.environ.get("FMD_OCR_E2E_URL", "").strip()


def _load_server() -> Any:
    spec = importlib.util.spec_from_file_location("fmd_ocr_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


server = _load_server()


def _settings(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "tier": "T2",
        "backend": "qwen3vl",
        "served_name": "Qwen3-VL-8B-Instruct",
        "vllm_url": "http://vllm.test/v1",
        "dpi": 200,
        "max_pages": 2,
        "max_side_px": 2400,
        "timeout_s": 5.0,
        "json_mode": True,
        "bbox_space": "normalized_1000",
        "device": "cpu",
        "config_path": str(CONFIG_PATH),
    }
    values.update(overrides)
    return server.Settings(**values)


class FakeBackend:
    """Replays a fixed block list; records the pages it was given."""

    name = "qwen3vl"

    def __init__(
        self,
        items: list[Any] | None = None,
        *,
        ready: tuple[bool, str] = (True, ""),
        failure: Exception | None = None,
    ) -> None:
        self.items = items if items is not None else []
        self.ready = ready
        self.failure = failure
        self.calls: list[Any] = []

    def check(self) -> tuple[bool, str]:
        return self.ready

    def extract_page(self, page: Any) -> list[Any]:
        self.calls.append(page)
        if self.failure is not None:
            raise self.failure
        return self.items


def _client(backend: Any, **overrides: Any) -> TestClient:
    return TestClient(server.create_app(_settings(**overrides), backend=backend))


def _image_b64(path: Path = FIXTURE) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _png(width: int, height: int, *, ink: bool) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (255, 255, 255))
    if ink:
        ImageDraw.Draw(image).line([(10, 10), (width - 10, height - 10)], fill=(0, 0, 0), width=4)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _post_image(client: TestClient, data: bytes) -> Any:
    return client.post("/extract", json={"image_base64": base64.b64encode(data).decode("ascii")})


BLOCKS = [
    {"bbox": [50, 400, 950, 460], "text": "शुल्क और पात्रता की जानकारी"},
    {"bbox": [40, 80, 900, 130], "text": "Total fee: INR 50000 per year"},
    {"bbox": [40, 250, 800, 300], "text": "Sanctioned intake: 60 seats"},
]


# --------------------------------------------------------------------------- contract


def test_extract_returns_the_documented_shape() -> None:
    response = _post_image(_client(FakeBackend(BLOCKS)), FIXTURE.read_bytes())

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"tool", "engine", "model", "pages", "blocks"}
    assert payload["tool"] == "fmd-ocr"
    assert payload["model"] == "Qwen3-VL-8B-Instruct"

    page = payload["pages"][0]
    assert (page["page"], page["width"], page["height"]) == (1, 600, 800)
    assert page["bbox"] == [0, 0, 600, 800]

    assert len(payload["blocks"]) == 3
    for block in payload["blocks"]:
        assert block["page"] == 1
        assert isinstance(block["block_order"], int)
        assert len(block["block_bbox"]) == 4
        assert block["block_content"].strip()
        x0, y0, x1, y1 = block["block_bbox"]
        assert 0 <= x0 < x1 <= 600
        assert 0 <= y0 < y1 <= 800


def test_blocks_are_restored_to_reading_order_and_order_is_monotonic() -> None:
    payload = _client(FakeBackend(BLOCKS)).post(
        "/extract", json={"image_base64": _image_b64()}
    ).json()

    blocks = payload["blocks"]
    assert [block["block_order"] for block in blocks] == [0, 1, 2]
    tops = [block["block_bbox"][1] for block in blocks]
    assert tops == sorted(tops)
    assert blocks[0]["block_content"].startswith("Total fee")
    assert blocks[-1]["block_content"].startswith("शुल्क")


def test_a_degenerate_box_is_reported_as_null_never_invented() -> None:
    items = [
        {"bbox": [0, 0, 0, 0], "text": "degenerate"},
        {"bbox": [90, 90, 110, 110], "text": "normalised"},
    ]
    payload = _client(FakeBackend(items)).post(
        "/extract", json={"image_base64": _image_b64()}
    ).json()

    by_text = {block["block_content"]: block for block in payload["blocks"]}
    assert set(by_text) == {"degenerate", "normalised"}
    assert by_text["degenerate"]["block_bbox"] is None
    assert by_text["normalised"]["block_bbox"] == [54.0, 72.0, 66.0, 88.0]
    for block in payload["blocks"]:
        if block["block_bbox"] is not None:
            x0, y0, x1, y1 = block["block_bbox"]
            assert 0 <= x0 < x1 <= 600 and 0 <= y0 < y1 <= 800


def test_a_box_already_in_pixels_is_not_scaled_down_again() -> None:
    items = [{"bbox": [100, 200, 1800, 400], "text": "pixel box"}]
    client = _client(FakeBackend(items), max_side_px=4000)
    payload = _post_image(client, _png(2000, 3000, ink=True)).json()

    assert payload["pages"][0]["width"] == 2000
    x0, y0, x1, y1 = payload["blocks"][0]["block_bbox"]
    assert (x0, y0, x1, y1) == (100.0, 200.0, 1800.0, 400.0)


def test_page_with_no_text_blocks_is_not_a_successful_extraction() -> None:
    response = _post_image(_client(FakeBackend([])), FIXTURE.read_bytes())

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail.startswith("empty_result:")
    assert "human queue" in detail


def test_blank_page_is_an_empty_page_not_a_failure() -> None:
    backend = FakeBackend(BLOCKS)
    payload = _post_image(_client(backend), _png(300, 400, ink=False)).json()

    assert payload["blocks"] == []
    assert payload["pages"][0]["blank"] is True
    assert backend.calls == [], "a blank raster must not be sent to the engine"


def test_oversized_page_is_downscaled_before_the_engine_sees_it() -> None:
    backend = FakeBackend(BLOCKS)
    payload = _post_image(_client(backend, max_side_px=500), _png(4000, 2000, ink=True)).json()

    page = payload["pages"][0]
    assert max(page["width"], page["height"]) == 500
    assert (page["width"], page["height"]) == (500, 250)
    assert backend.calls[0].width == 500
    x0, y0, x1, y1 = payload["blocks"][0]["block_bbox"]
    assert 0 <= x0 < x1 <= 500 and 0 <= y0 < y1 <= 500


def test_pdf_without_pages_is_rejected() -> None:
    response = _client(FakeBackend(BLOCKS)).post("/extract", json={})

    assert response.status_code == 422


# --------------------------------------------------------------------------- failure mapping


def _vllm_transport(*, status: int, model_id: str = "Qwen3-VL-8B-Instruct") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model_id}]})
        return httpx.Response(status, json={"error": {"message": "engine said no"}})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("status", [429, 500, 502])
def test_engine_rate_limit_and_server_errors_become_503(status: int) -> None:
    settings = _settings()
    backend = server.Qwen3VLBackend(settings, transport=_vllm_transport(status=status))
    client = TestClient(server.create_app(settings, backend=backend))

    response = _post_image(client, FIXTURE.read_bytes())

    assert response.status_code == 503
    assert f"backend_http_{status}" in response.json()["detail"]


def test_unreachable_engine_is_503_not_an_empty_document() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    settings = _settings()
    backend = server.Qwen3VLBackend(settings, transport=httpx.MockTransport(handler))
    client = TestClient(server.create_app(settings, backend=backend))

    assert _post_image(client, FIXTURE.read_bytes()).status_code == 503
    health = client.get("/health").json()
    assert health["status"] == "unavailable"
    assert "vision_unavailable" in health["reason"]


def test_free_text_instead_of_json_is_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "Qwen3-VL-8B-Instruct"}]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "I cannot read this scan."}}]},
        )

    settings = _settings()
    backend = server.Qwen3VLBackend(settings, transport=httpx.MockTransport(handler))
    client = TestClient(server.create_app(settings, backend=backend))
    response = _post_image(client, FIXTURE.read_bytes())

    assert response.status_code == 503
    assert "malformed_output" in response.json()["detail"]


def test_fenced_json_and_content_parts_are_accepted() -> None:
    body = json.dumps({"blocks": [{"bbox": [10, 20, 300, 60], "text": "Fee INR 50000"}]})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "Qwen3-VL-8B-Instruct"}]})
        fenced = {"type": "text", "text": f"```json\n{body}\n```"}
        return httpx.Response(200, json={"choices": [{"message": {"content": [fenced]}}]})

    settings = _settings()
    backend = server.Qwen3VLBackend(settings, transport=httpx.MockTransport(handler))
    client = TestClient(server.create_app(settings, backend=backend))
    payload = _post_image(client, FIXTURE.read_bytes()).json()

    assert [block["block_content"] for block in payload["blocks"]] == ["Fee INR 50000"]


def test_backend_unavailable_is_503_with_the_reason() -> None:
    backend = FakeBackend(
        BLOCKS, failure=server.BackendUnavailable("engine_start_failed: sm_90 wheel on sm_120")
    )
    client = TestClient(server.create_app(_settings(), backend=backend))

    health = client.get("/health").json()
    assert health["status"] == "ok"  # check() is the readiness gate, not the page call
    response = _post_image(client, FIXTURE.read_bytes())
    assert response.status_code == 503
    assert "sm_90" in response.json()["detail"]


# --------------------------------------------------------------------------- health


def test_health_ok_when_the_served_model_matches_config() -> None:
    client = TestClient(server.create_app(_settings(), backend=FakeBackend(BLOCKS)))

    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["engine"] == "qwen3vl"
    assert payload["model"] == "Qwen3-VL-8B-Instruct"
    assert "reason" not in payload


def test_health_unavailable_when_a_text_model_is_on_the_vision_port() -> None:
    settings = _settings()
    backend = server.Qwen3VLBackend(
        settings, transport=_vllm_transport(status=200, model_id="Qwen3.5-9B-AWQ")
    )
    payload = TestClient(server.create_app(settings, backend=backend)).get("/health").json()

    assert payload["status"] == "unavailable"
    assert "model_not_served" in payload["reason"]
    assert "Qwen3.5-9B-AWQ" in payload["reason"]


def test_health_unavailable_when_the_registry_cannot_be_read(tmp_path: Path) -> None:
    settings = server.Settings.load(env={"FMD_OCR_CONFIG": str(tmp_path / "missing.yaml")})
    assert settings.config_error
    payload = TestClient(server.create_app(settings, backend=None)).get("/health").json()

    assert payload["status"] == "unavailable"
    assert "config_unreadable" in payload["reason"]


def test_health_unavailable_for_an_unknown_backend() -> None:
    settings = _settings(backend="tesseract", config_path=str(CONFIG_PATH))
    payload = TestClient(server.create_app(settings, backend=None)).get("/health").json()

    assert payload["status"] == "unavailable"
    assert "backend_unknown" in payload["reason"]


# --------------------------------------------------------------------------- registry


def test_service_defaults_are_read_from_the_registry() -> None:
    primary = server.Settings.load(env={}, config_path=CONFIG_PATH)
    assert primary.backend == "qwen3vl"
    assert primary.served_name == "Qwen3-VL-8B-Instruct"
    assert primary.bbox_space == "normalized_1000"
    assert ":8001" in primary.vllm_url
    assert primary.hub_id == "Qwen3-VL-8B-Instruct-FP8"

    alternate = server.Settings.load(env={"FMD_OCR_TIER": "T2b"}, config_path=CONFIG_PATH)
    assert alternate.backend == "paddleocr"
    assert alternate.device == "cpu", "Paddle GPU wheels stop at sm_90 on this host"

    overridden = server.Settings.load(
        env={
            "FMD_OCR_TIER": "T2",
            "FMD_OCR_BACKEND": "paddleocr",
            "FMD_OCR_VLLM_URL": "http://elsewhere:9000/v1",
            "FMD_OCR_MAX_SIDE_PX": "1200",
        },
        config_path=CONFIG_PATH,
    )
    assert overridden.backend == "paddleocr"
    assert overridden.vllm_url == "http://elsewhere:9000/v1"
    assert overridden.max_side_px == 1200


# --------------------------------------------------------------------------- live service


@pytest.mark.skipif(
    not E2E_URL,
    reason=(
        "T2 is interface-complete, weights-dependent: Qwen3-VL-8B-Instruct is ~10.6 GB of "
        "FP8 weights and tests never download them. Bring the service up with "
        "`scripts/serve_ocr.sh vision-start && scripts/serve_ocr.sh start`, then run "
        "`FMD_OCR_E2E_URL=http://localhost:8110 uv run pytest -q "
        "tests/test_ocr_service.py::test_golden_scan_through_a_live_service`."
    ),
)
def test_golden_scan_through_a_live_service() -> None:
    health = httpx.get(E2E_URL + "/health", timeout=10).json()
    if health.get("status") != "ok":
        pytest.skip(
            f"OCR weights absent: service says {health.get('status')} — {health.get('reason')}"
        )

    response = httpx.post(
        E2E_URL + "/extract",
        json={"image_base64": _image_b64()},
        timeout=300,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    text = "\n".join(block["block_content"] for block in payload["blocks"])

    assert "50000" in text
    assert "शुल्क" in text
    orders = [block["block_order"] for block in payload["blocks"]]
    assert orders == sorted(orders)
