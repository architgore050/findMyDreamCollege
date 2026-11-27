"""Network-policy conformance, offline.

Every scenario uses ``httpx.MockTransport`` and an injected sleeper, so no test
opens a socket and none can hang. Assertions are on *bounded* attempts and
elapsed time, not on wall-clock sleeps.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import types
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from fmd.fetch.ladder import (
    FetchLadder,
    HttpxFetcher,
)
from fmd.fetch.politeness import PolitenessManager
from fmd.fetch.robots import RobotsCache
from fmd.fetch.types import FetchRequest
from fmd.net.policy import (
    BrowserPolicy,
    NetworkGuard,
    NetworkUnavailable,
    RetryPolicy,
    Timeouts,
    classify_browser_error,
    classify_exception,
    make_client,
    parse_retry_after,
    request_with_retries,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class RecordingSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.delays)


class Clock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds


def _dns_handler(request: httpx.Request) -> httpx.Response:
    exc = httpx.ConnectError("[Errno -2] Name or service not known", request=request)
    exc.__cause__ = socket.gaierror(-2, "Name or service not known")
    raise exc


def _connect_timeout_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("connect timed out", request=request)


def _read_timeout_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("read timed out", request=request)


def _truncated_handler(request: httpx.Request) -> httpx.Response:
    class _ShortStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"short"
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )

    return httpx.Response(200, headers={"content-length": "100"}, stream=_ShortStream())


class _FailingStream(httpx.SyncByteStream):
    def __init__(self, request: httpx.Request) -> None:
        self._request = request

    def __iter__(self):
        yield b"partial"
        raise httpx.ReadTimeout("read timed out", request=self._request)


def test_classification_taxonomy() -> None:
    request = httpx.Request("GET", "https://college.ac.in/x")
    dns = httpx.ConnectError("Name or service not known", request=request)
    dns.__cause__ = socket.gaierror(-2, "Name or service not known")
    assert classify_exception(dns).kind == "dns_failure"
    assert classify_exception(dns).retryable is False

    assert classify_exception(httpx.ConnectTimeout("x", request=request)).kind == (
        "connect_timeout"
    )
    assert classify_exception(httpx.ReadTimeout("x", request=request)).retryable is True
    assert (
        classify_exception(httpx.ConnectError("certificate verify failed", request=request)).kind
        == "tls_error"
    )
    assert (
        classify_exception(
            httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body", request=request
            )
        ).kind
        == "truncated_body"
    )
    assert classify_exception(httpx.ReadError("reset", request=request)).kind == (
        "connection_reset"
    )


def test_dns_failure_fails_fast_without_retry() -> None:
    sleeper = RecordingSleeper()
    tier = HttpxFetcher(
        transport=httpx.MockTransport(_dns_handler),
        retry=RetryPolicy(max_attempts=3),
        sleeper=sleeper,
        guard=NetworkGuard(threshold=3),
    )
    calls = {"n": 0}

    def counted(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _dns_handler(request)

    tier._client = make_client(
        timeouts=Timeouts(connect=1.0), transport=httpx.MockTransport(counted)
    )
    tier._owns_client = True

    with pytest.raises(httpx.ConnectError):
        tier.fetch(FetchRequest(url="https://college.ac.in/page"))

    assert calls["n"] == 1, "a DNS failure must not be retried"
    assert sleeper.delays == []
    tier.close()


def test_connect_timeout_is_retried_but_bounded() -> None:
    sleeper = RecordingSleeper()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _connect_timeout_handler(request)

    tier = HttpxFetcher(
        transport=httpx.MockTransport(handler),
        retry=RetryPolicy(max_attempts=3),
        sleeper=sleeper,
        guard=NetworkGuard(threshold=10),
    )
    with pytest.raises(httpx.ConnectTimeout):
        tier.fetch(FetchRequest(url="https://college.ac.in/page"))

    assert calls["n"] == 3
    assert len(sleeper.delays) == 2
    assert sleeper.total < 3.0, "backoff must stay bounded"
    tier.close()


def test_read_timeout_mid_body_is_one_bounded_failure() -> None:
    sleeper = RecordingSleeper()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_FailingStream(request))

    client = make_client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.ReadTimeout):
            request_with_retries(
                client,
                "GET",
                "https://college.ac.in/big.pdf",
                retry=RetryPolicy(max_attempts=2),
                sleep=sleeper,
            )
    finally:
        client.close()
    assert len(sleeper.delays) == 1


def test_429_numeric_retry_after_is_honoured() -> None:
    sleeper = RecordingSleeper()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "2.5"})
        return httpx.Response(200, content=b"ok")

    client = make_client(transport=httpx.MockTransport(handler))
    try:
        response = request_with_retries(
            client,
            "GET",
            "https://searx.example/search",
            retry=RetryPolicy(max_attempts=2),
            sleep=sleeper,
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert sleeper.delays == [2.5]


def test_429_http_date_retry_after_is_honoured() -> None:
    sleeper = RecordingSleeper()
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=10), usegmt=True)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": when})
        return httpx.Response(200, content=b"ok")

    client = make_client(transport=httpx.MockTransport(handler))
    try:
        response = request_with_retries(
            client,
            "GET",
            "https://searx.example/search",
            retry=RetryPolicy(max_attempts=2),
            sleep=sleeper,
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert len(sleeper.delays) == 1
    assert 8.0 <= sleeper.delays[0] <= 10.0


def test_retry_after_cap_and_parse() -> None:
    assert parse_retry_after("99999") == 99999.0
    sleeper = RecordingSleeper()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "99999"})

    client = make_client(transport=httpx.MockTransport(handler))
    try:
        response = request_with_retries(
            client,
            "GET",
            "https://x.example/",
            retry=RetryPolicy(max_attempts=2, max_retry_after=30.0),
            sleep=sleeper,
        )
    finally:
        client.close()
    assert response.status_code == 429
    assert sleeper.delays == [30.0]


def test_5xx_then_success() -> None:
    sleeper = RecordingSleeper()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=b"ok")

    client = make_client(transport=httpx.MockTransport(handler))
    try:
        response = request_with_retries(
            client,
            "GET",
            "https://x.example/",
            retry=RetryPolicy(max_attempts=2),
            sleep=sleeper,
        )
    finally:
        client.close()
    assert response.status_code == 200
    assert calls["n"] == 2


def test_truncated_body_is_retryable() -> None:
    sleeper = RecordingSleeper()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _truncated_handler(request)

    client = make_client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.RemoteProtocolError):
            request_with_retries(
                client,
                "GET",
                "https://x.example/",
                retry=RetryPolicy(max_attempts=2),
                sleep=sleeper,
            )
    finally:
        client.close()
    assert calls["n"] == 2


def test_total_outage_trips_the_breaker() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _connect_timeout_handler(request)

    guard = NetworkGuard(threshold=2)
    client = make_client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.ConnectTimeout):
            request_with_retries(
                client,
                "GET",
                "https://a.example/",
                retry=RetryPolicy(max_attempts=1),
                guard=guard,
            )
        with pytest.raises(NetworkUnavailable) as excinfo:
            request_with_retries(
                client,
                "GET",
                "https://b.example/",
                retry=RetryPolicy(max_attempts=1),
                guard=guard,
            )
    finally:
        client.close()

    assert "network_unavailable" in str(excinfo.value)
    assert guard.state()["tripped"] is True
    assert calls["n"] == 2, "the third call must not touch the network"


def test_ladder_aborts_on_outage_instead_of_per_item_failures() -> None:
    clock = Clock()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _connect_timeout_handler(request)

    tier = HttpxFetcher(
        transport=httpx.MockTransport(handler),
        retry=RetryPolicy(max_attempts=1),
        sleeper=RecordingSleeper(),
    )
    ladder = FetchLadder(
        fetchers=[tier],
        robots=RobotsCache(fetch=lambda url: (404, ""), clock=clock),
        politeness=PolitenessManager(clock=clock, jitter=0.0),
        clock=clock,
        sleeper=lambda seconds: clock.advance(seconds),
        guard=NetworkGuard(threshold=2),
    )

    first = ladder.fetch("https://a.example/page")
    assert first.status == 0
    assert first.error is not None and first.error.startswith("connect_timeout")

    # The second failure trips the breaker inside the tier; the third URL never
    # reaches the network at all.
    with pytest.raises(NetworkUnavailable):
        ladder.fetch("https://b.example/page")
    with pytest.raises(NetworkUnavailable):
        ladder.fetch("https://c.example/page")
    assert calls["n"] == 2


def test_adaptive_backoff_is_per_host_and_capped() -> None:
    clock = Clock()
    politeness = PolitenessManager(clock=clock, jitter=0.0)
    url = "https://college.ac.in/page"

    politeness.note_timeout(url)
    first = politeness.delay_for("https://college.ac.in/other")
    politeness.note_timeout(url)
    second = politeness.delay_for("https://college.ac.in/other")

    assert first == 2.0
    assert second == 4.0
    # A different host is unaffected.
    assert politeness.delay_for("https://other.ac.in/x") == 0.0

    for _ in range(20):
        politeness.note_timeout(url)
    assert politeness.delay_for("https://college.ac.in/other") <= 300.0
    # An HTTP answer resets the host's timeout streak.
    politeness.record_response(url, 200)
    assert politeness.delay_for("https://college.ac.in/other") == 0.0


# ---------------------------------------------------------------- browser tier


def test_browser_error_taxonomy() -> None:
    assert classify_browser_error("net::ERR_NAME_NOT_RESOLVED").kind == "dns_failure"
    assert classify_browser_error("net::ERR_INTERNET_DISCONNECTED").kind == ("network_unreachable")
    assert classify_browser_error("net::ERR_CONNECTION_TIMED_OUT").kind == ("connect_timeout")
    assert classify_browser_error("Timeout 20000ms exceeded").retryable is True
    assert classify_browser_error("Target crashed").kind == "browser_crash"
    assert classify_browser_error("Frame was detached").retryable is True
    assert classify_browser_error("something else").retryable is False


def test_browser_policy_is_env_driven() -> None:
    policy = BrowserPolicy.from_env(
        {
            "FMD_BROWSER_NAV_TIMEOUT_MS": "9000",
            "FMD_BROWSER_BLOCK_ASSETS": "0",
            "FMD_BROWSER_MAX_CONCURRENCY_PER_DOMAIN": "1",
        }
    )
    assert policy.navigation_timeout_ms == 9000
    assert policy.block_assets is False
    assert policy.max_concurrency_per_domain == 1
    # A bad value falls back to the safe default instead of raising mid-run.
    assert (
        BrowserPolicy.from_env({"FMD_BROWSER_NAV_TIMEOUT_MS": "nope"}).navigation_timeout_ms
        == 20_000
    )


def test_mcp_browser_refuses_post_before_any_server_call() -> None:
    from fmd.fetch.mcp_browser import InvisiblePlaywrightMCPFetcher

    def _no_session() -> None:
        raise AssertionError("no session may be created for a non-GET request")

    tier = InvisiblePlaywrightMCPFetcher(session_factory=_no_session)
    with pytest.raises(Exception) as excinfo:
        tier.fetch(FetchRequest(url="https://x.example/", method="POST", body=b"x"))
    assert "cannot faithfully replay" in str(excinfo.value)


# ------------------------------------------------------------ download helpers


def _load_hf_download(monkeypatch) -> types.ModuleType:
    stub = types.ModuleType("huggingface_hub")
    stub.HfApi = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    spec = importlib.util.spec_from_file_location(
        "hf_download_under_test", REPO_ROOT / "scripts" / "hf_download.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_range_ignored_resume_restarts_never_appends(monkeypatch) -> None:
    module = _load_hf_download(monkeypatch)
    action, _reason = module.plan_resume(1_000, 200, None, 2_000)
    assert action == "restart"


def test_range_resume_requires_matching_offset_and_total(monkeypatch) -> None:
    module = _load_hf_download(monkeypatch)
    assert module.plan_resume(500, 206, "bytes 500-999/1000", 1000)[0] == "append"
    assert module.plan_resume(500, 206, "bytes 0-999/1000", 1000)[0] == "error"
    assert module.plan_resume(500, 206, "bytes 500-999/2000", 1000)[0] == "error"
    assert module.plan_resume(500, 206, None, 1000)[0] == "error"
    assert module.plan_resume(0, 200, None, 1000)[0] == "append"


def test_download_retry_after_parses_seconds_and_dates(monkeypatch) -> None:
    module = _load_hf_download(monkeypatch)
    assert module.parse_retry_after("12") == 12.0
    assert module.parse_retry_after("99999") == module.MAX_RETRY_AFTER
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=5), usegmt=True)
    assert 3.0 <= module.parse_retry_after(when) <= 5.0
