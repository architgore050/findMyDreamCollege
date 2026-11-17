"""One HTTP policy for every outbound call (brief §7, runbook §7).

The hostel-wifi failure modes this module exists for:

* a DNS lookup that hangs for 8 s on three hosts instead of failing at once;
* a read that never ends because the server drip-feeds a body;
* retries that hammer a host that already said ``429`` with ``Retry-After``;
* a total outage that turns one run into hundreds of per-item timeouts.

The policy is deliberately boring:

======================  =====================================================
connect timeout         4 s (covers DNS + TCP + TLS; DNS failures fail fast)
read/write timeout      20 s default, scales with ``max_tokens`` for LLMs
pool timeout            4 s
retries                 2 extra attempts (3 total) for transient faults
backoff                 full-jitter exponential, 0.5 s → 30 s
``Retry-After``         numeric seconds *and* HTTP-date, capped at 300 s
outage breaker          3 consecutive transport failures → ``network_unavailable``
======================  =====================================================

Every class here is injectable (clock, sleeper, RNG), so tests assert bounded
attempts without sleeping and without touching the network.
"""

from __future__ import annotations

import random
import socket
import ssl
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

#: Reason string every stage reports when the network itself is gone. It is a
#: distinct value on purpose: "no network" is not a per-document failure and
#: must not be recorded as one.
NETWORK_UNAVAILABLE = "network_unavailable"

#: Statuses worth a retry. 4xx other than 408/425/429 never is: auth, robots and
#: not-found answers do not improve by asking again.
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Faults that indicate the *network* is down (not one host misbehaving). Three
#: in a row trip the run-level breaker so a stage aborts instead of failing
#: every remaining item for minutes.
OUTAGE_KINDS: frozenset[str] = frozenset(
    {"dns_failure", "connect_timeout", "connection_error", "network_unreachable"}
)


@dataclass(frozen=True)
class Timeouts:
    """Connect/read/write/pool timeouts in seconds."""

    connect: float = 4.0
    read: float = 20.0
    write: float = 20.0
    pool: float = 4.0
    #: Optional wall-clock ceiling for the whole exchange. ``None`` lets the
    #: per-phase timeouts govern, which is what streaming downloads want.
    total: float | None = None

    def as_httpx(self) -> httpx.Timeout:
        if self.total is not None:
            return httpx.Timeout(
                self.total,
                connect=self.connect,
                read=self.read,
                write=self.write,
                pool=self.pool,
            )
        return httpx.Timeout(connect=self.connect, read=self.read, write=self.write, pool=self.pool)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retries with full jitter (``random.uniform(0, delay)``)."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    #: A server ``Retry-After`` is honoured but capped: a hostile or wrong value
    #: must not stall a run for an hour.
    max_retry_after: float = 300.0


DEFAULT_TIMEOUTS = Timeouts()
DEFAULT_RETRY = RetryPolicy()


class NetworkUnavailable(RuntimeError):
    """The network is down; stop the stage instead of failing item by item."""

    reason = NETWORK_UNAVAILABLE

    def __init__(self, message: str) -> None:
        super().__init__(f"{NETWORK_UNAVAILABLE}: {message}")


@dataclass(frozen=True)
class NetworkFault:
    """A classified transport/HTTP failure and its handling policy."""

    kind: str
    reason: str
    retryable: bool
    message: str
    status: int | None = None


def _caused_by(exc: BaseException, wanted: tuple[type[BaseException], ...]) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, wanted):
            return True
        current = current.__cause__ or current.__context__
    return False


_DNS_MARKERS = (
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "no address associated with hostname",
    "name resolution",
    "dns",
)


def _looks_like_dns(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _DNS_MARKERS)


def classify_exception(exc: BaseException) -> NetworkFault:
    """Map an ``httpx`` transport exception to a stable reason + retry policy.

    ``dns_failure`` is deliberately **not** retryable: on a bad link every
    retry is another multi-second stall. The caller records it with the outage
    breaker, which aborts the stage.
    """
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, httpx.ConnectTimeout):
        return NetworkFault("connect_timeout", "connect_timeout", True, text)
    if isinstance(exc, httpx.ReadTimeout):
        return NetworkFault("read_timeout", "read_timeout", True, text)
    if isinstance(exc, httpx.WriteTimeout):
        return NetworkFault("write_timeout", "write_timeout", True, text)
    if isinstance(exc, httpx.PoolTimeout):
        return NetworkFault("pool_timeout", "pool_timeout", True, text)
    if isinstance(exc, httpx.ConnectError):
        if _caused_by(exc, (socket.gaierror,)) or _looks_like_dns(str(exc)):
            return NetworkFault("dns_failure", "dns_failure", False, text)
        lowered = str(exc).lower()
        if (
            _caused_by(exc, (ssl.SSLError,))
            or "ssl" in lowered
            or "certificate" in lowered
            or "tls" in lowered
        ):
            return NetworkFault("tls_error", "tls_error", False, text)
        if "network is unreachable" in lowered:
            return NetworkFault("network_unreachable", NETWORK_UNAVAILABLE, True, text)
        return NetworkFault("connection_error", "connection_error", True, text)
    if isinstance(exc, httpx.RemoteProtocolError):
        lowered = str(exc).lower()
        if "incomplete" in lowered or "peer closed" in lowered or "content-length" in lowered:
            return NetworkFault("truncated_body", "truncated_body", True, text)
        return NetworkFault("connection_reset", "connection_reset", True, text)
    if isinstance(exc, httpx.ReadError | httpx.WriteError):
        return NetworkFault("connection_reset", "connection_reset", True, text)
    if isinstance(exc, httpx.ProxyError):
        return NetworkFault("proxy_error", "proxy_error", False, text)
    if isinstance(exc, httpx.TransportError):
        return NetworkFault("connection_error", "connection_error", True, text)
    return NetworkFault("unexpected_error", "unexpected_error", False, text)


def classify_status(status: int, headers: dict[str, str] | None = None) -> NetworkFault:
    """Classify an HTTP answer. Only 429/408/425/5xx are retryable."""
    detail = f"HTTP {status}"
    if status == 429:
        return NetworkFault("rate_limited", "rate_limited", True, detail, status=status)
    if status in (408, 425):
        return NetworkFault("request_timeout", "request_timeout", True, detail, status=status)
    if 500 <= status < 600:
        return NetworkFault("server_error", "server_error", True, detail, status=status)
    if status in (401, 403):
        return NetworkFault("forbidden", "forbidden", False, detail, status=status)
    if status in (404, 410):
        return NetworkFault("not_found", "not_found", False, detail, status=status)
    return NetworkFault("http_error", "http_error", False, detail, status=status)


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse ``Retry-After`` as delta-seconds or an HTTP-date.

    Kept here (not in the crawl layer) so the LLM router, discovery clients and
    the fetch ladder all read the header the same way.
    """
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    return max(0.0, (when - now).total_seconds())


def backoff_delay(
    attempt: int,
    *,
    retry: RetryPolicy = DEFAULT_RETRY,
    retry_after: float | None = None,
    rng: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before ``attempt`` (1-based) is retried.

    Full jitter (``rng() * ceiling``) decorrelates parallel callers and avoids a
    thundering herd after an outage. A server ``Retry-After`` wins, capped.
    """
    if retry_after is not None and retry_after > 0:
        return min(retry_after, retry.max_retry_after)
    ceiling = min(retry.max_delay, retry.base_delay * (2 ** max(0, attempt - 1)))
    return max(0.0, rng() * ceiling)


class NetworkGuard:
    """Run-level outage breaker.

    Counts consecutive *network* faults (DNS, connect timeout, unreachable).
    Three in a row within the window means the link is gone: every subsequent
    call raises :class:`NetworkUnavailable` until an explicit ``reset()``.
    A successful response clears the streak, so one flaky host cannot abort a
    run while other hosts still answer.
    """

    def __init__(
        self,
        *,
        threshold: int = 3,
        window_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = max(1, threshold)
        self.window_s = window_s
        self._clock = clock
        self._events: list[tuple[float, str, str]] = []
        self.tripped = False
        self.reason: str | None = None

    def check(self) -> None:
        if self.tripped:
            raise NetworkUnavailable(
                self.reason or "the network is unavailable; aborting instead of retrying every item"
            )

    def record_success(self) -> None:
        if not self.tripped:
            self._events.clear()

    def record_failure(self, target: str, fault: NetworkFault) -> None:
        if fault.kind not in OUTAGE_KINDS:
            return
        now = self._clock()
        self._events = [event for event in self._events if now - event[0] <= self.window_s]
        self._events.append((now, target, fault.kind))
        if len(self._events) < self.threshold:
            return
        self.tripped = True
        hosts = sorted({target for _, target, _ in self._events})
        kinds = sorted({kind for _, _, kind in self._events})
        self.reason = (
            f"{self.threshold} consecutive transport failures ({', '.join(kinds)}) "
            f"across {', '.join(hosts[:5])}. The network is down; aborting this stage "
            "instead of failing every remaining item. Retry the run when the link is back."
        )

    def reset(self) -> None:
        self._events.clear()
        self.tripped = False
        self.reason = None

    def state(self) -> dict[str, Any]:
        return {"tripped": self.tripped, "failures": len(self._events), "reason": self.reason}


def make_client(
    *,
    timeouts: Timeouts = DEFAULT_TIMEOUTS,
    transport: httpx.BaseTransport | None = None,
    headers: dict[str, str] | None = None,
    base_url: str = "",
    follow_redirects: bool = True,
    http2: bool = True,
    limits: httpx.Limits | None = None,
) -> httpx.Client:
    """Build a pooled, timeout-bound client.

    This is the *only* place a real ``httpx.Client`` is constructed. Call sites
    must not pass their own limits: a bare client defaults to five keep-alive
    connections per host, which on a choppy link turns into reconnect storms.
    """
    if headers is None:
        from fmd.settings import get_settings

        headers = {"User-Agent": get_settings().user_agent}
    options: dict[str, Any] = {
        "timeout": timeouts.as_httpx(),
        "headers": headers,
        "follow_redirects": follow_redirects,
        "limits": limits
        or httpx.Limits(max_connections=16, max_keepalive_connections=8, keepalive_expiry=30.0),
        "transport": transport,
    }
    if base_url:
        options["base_url"] = base_url
    if http2:
        try:
            return httpx.Client(http2=True, **options)
        except ImportError:  # h2 extra absent; HTTP/1.1 is fine
            pass
    return httpx.Client(http2=False, **options)


def request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    retry: RetryPolicy = DEFAULT_RETRY,
    retry_statuses: Iterable[int] = RETRYABLE_STATUSES,
    headers: dict[str, str] | None = None,
    content: bytes | None = None,
    json: Any | None = None,
    params: dict[str, Any] | None = None,
    guard: NetworkGuard | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    clock: Callable[[], float] = time.monotonic,
) -> httpx.Response:
    """Issue one request, retrying only transient faults.

    Raises the original ``httpx`` exception when retries are exhausted and
    :class:`NetworkUnavailable` when the outage breaker trips. ``MockTransport``
    tests inject ``sleep`` so elapsed time stays bounded.
    """
    if guard is not None:
        guard.check()
    statuses = frozenset(retry_statuses)
    target = urlsplit(url).hostname or url
    attempts = max(1, retry.max_attempts)
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.request(
                method,
                url,
                headers=headers,
                content=content,
                json=json,
                params=params,
            )
        except httpx.HTTPError as exc:
            last_exc = exc
            fault = classify_exception(exc)
            if guard is not None:
                guard.record_failure(target, fault)
                guard.check()
            if not fault.retryable or attempt >= attempts:
                raise
            sleep(backoff_delay(attempt, retry=retry, rng=rng))
            continue
        except OSError as exc:  # a socket error that escaped httpx
            fault = classify_exception(exc)
            if guard is not None:
                guard.record_failure(target, fault)
                guard.check()
            if not fault.retryable or attempt >= attempts:
                raise
            sleep(backoff_delay(attempt, retry=retry, rng=rng))
            continue

        if response.status_code in statuses and attempt < attempts:
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            delay = backoff_delay(attempt, retry=retry, retry_after=retry_after, rng=rng)
            # Drain the body without buffering it; the retry supersedes it.
            response.close()
            sleep(delay)
            continue
        if guard is not None:
            guard.record_success()
        return response

    assert last_exc is not None  # pragma: no cover - loop always returns or raises
    raise last_exc


#: --- LLM-specific timeout scaling -------------------------------------------

#: Worst-case decoding speed assumed when sizing a read timeout. A 26B MoE on
#: this host decodes ~100 tok/s; 16 tok/s leaves a 6x margin for a busy server
#: without turning a dead endpoint into a five-minute wait.
LLM_TOKENS_PER_SECOND = 16.0
LLM_MIN_READ_S = 30.0
LLM_MAX_READ_S = 600.0


def llm_timeouts(
    max_tokens: int | None,
    *,
    connect: float = 4.0,
    pool: float = 4.0,
    minimum_read: float = LLM_MIN_READ_S,
    maximum_read: float = LLM_MAX_READ_S,
) -> Timeouts:
    """Read timeout that scales with the completion budget.

    A fixed 120 s is wrong in both directions: too short for a 4k-token local
    completion on CPU, needlessly long for a 200-token classification. The
    estimate is ``30 s + max_tokens / 16``, capped at 600 s.
    """
    budget = max(0, int(max_tokens or 0))
    read = min(maximum_read, max(minimum_read, minimum_read + budget / LLM_TOKENS_PER_SECOND))
    return Timeouts(
        connect=connect,
        read=read,
        write=min(read, 60.0),
        pool=pool,
        total=None,
    )


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"})


def is_local_endpoint(url: str) -> bool:
    """True for a loopback/host-alias endpoint (the single-GPU llama.cpp server)."""
    host = (urlsplit(url).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.startswith("127.")


def hf_download_env() -> dict[str, str]:
    """Env defaults for Hugging Face artifact fetches (Docling, prefetch).

    ``huggingface_hub`` uses these for its own metadata/download calls. They are
    intentionally conservative for a slow link: a dead endpoint surfaces in
    10 s instead of the 10-minute default, and a stalled transfer is retried
    rather than slept on. Export these before a prefetch in a network-poor
    environment; the pipeline only *reads* artifacts from cache.
    """
    return {
        "HF_HUB_ETAG_TIMEOUT": "10",
        "HF_HUB_DOWNLOAD_TIMEOUT": "30",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


# ---------------------------------------------------------------------------
# Browser tier (patchright / Playwright) policy
# ---------------------------------------------------------------------------

#: Browser navigations are the slowest, most failure-prone outbound path on a
#: bad link: DNS, TLS, JS challenge, then dozens of subresources. Every value is
#: environment-switchable so a host can tune without a code change; defaults
#: assume <1 MiB/s and frequent drops.
BROWSER_ENV_DEFAULTS: dict[str, str] = {
    "FMD_BROWSER_NAV_TIMEOUT_MS": "20000",
    "FMD_BROWSER_ACTION_TIMEOUT_MS": "5000",
    "FMD_BROWSER_SELECTOR_TIMEOUT_MS": "5000",
    "FMD_BROWSER_NETWORK_IDLE_TIMEOUT_MS": "8000",
    "FMD_BROWSER_WAIT_NETWORK_IDLE": "0",
    "FMD_BROWSER_MAX_PAGE_BYTES": str(8 * 1024 * 1024),
    "FMD_BROWSER_BLOCK_ASSETS": "1",
    "FMD_BROWSER_MAX_CONCURRENCY_PER_DOMAIN": "1",
    "FMD_BROWSER_HARD_KILL_GRACE_MS": "3000",
    "FMD_BROWSER_LAUNCH_TIMEOUT_MS": "15000",
    "FMD_BROWSER_PROFILE_DIR": "",
}


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    try:
        return max(0, int(env.get(key, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BrowserPolicy:
    """Explicit timeouts and load caps for the patchright tier.

    ``navigation_timeout`` bounds ``page.goto``; ``network_idle_timeout`` is a
    *best-effort* extra wait and is off by default because a page with a dead
    analytics beacon would otherwise never be idle. ``selector_timeout`` is for
    callers that need a specific element. ``max_page_bytes`` refuses a runaway
    page instead of buffering it.
    """

    navigation_timeout_ms: int = 20_000
    action_timeout_ms: int = 5_000
    selector_timeout_ms: int = 5_000
    network_idle_timeout_ms: int = 8_000
    wait_network_idle: bool = False
    max_page_bytes: int = 8 * 1024 * 1024
    block_assets: bool = True
    max_concurrency_per_domain: int = 1
    hard_kill_grace_ms: int = 3_000
    launch_timeout_ms: int = 15_000
    profile_dir: str = ""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> BrowserPolicy:
        values = {**BROWSER_ENV_DEFAULTS, **(env if env is not None else {})}
        return cls(
            navigation_timeout_ms=_env_int(values, "FMD_BROWSER_NAV_TIMEOUT_MS", 20_000),
            action_timeout_ms=_env_int(values, "FMD_BROWSER_ACTION_TIMEOUT_MS", 5_000),
            selector_timeout_ms=_env_int(values, "FMD_BROWSER_SELECTOR_TIMEOUT_MS", 5_000),
            network_idle_timeout_ms=_env_int(values, "FMD_BROWSER_NETWORK_IDLE_TIMEOUT_MS", 8_000),
            wait_network_idle=_env_bool(values, "FMD_BROWSER_WAIT_NETWORK_IDLE", False),
            max_page_bytes=_env_int(values, "FMD_BROWSER_MAX_PAGE_BYTES", 8 * 1024 * 1024),
            block_assets=_env_bool(values, "FMD_BROWSER_BLOCK_ASSETS", True),
            max_concurrency_per_domain=_env_int(
                values, "FMD_BROWSER_MAX_CONCURRENCY_PER_DOMAIN", 1
            ),
            hard_kill_grace_ms=_env_int(values, "FMD_BROWSER_HARD_KILL_GRACE_MS", 3_000),
            launch_timeout_ms=_env_int(values, "FMD_BROWSER_LAUNCH_TIMEOUT_MS", 15_000),
            profile_dir=values.get("FMD_BROWSER_PROFILE_DIR", ""),
        )


_BROWSER_ERROR_MAP: tuple[tuple[tuple[str, ...], NetworkFault], ...] = (
    (
        ("err_name_not_resolved", "name not resolved", "getaddrinfo"),
        NetworkFault("dns_failure", "dns_failure", False, "browser: DNS resolution failed"),
    ),
    (
        ("err_internet_disconnected", "internet_disconnected", "network is unreachable"),
        NetworkFault(
            "network_unreachable", NETWORK_UNAVAILABLE, True, "browser: network disconnected"
        ),
    ),
    (
        ("err_connection_timed_out", "err_timed_out", "timeout", "timed out"),
        NetworkFault("connect_timeout", "connect_timeout", True, "browser: navigation timed out"),
    ),
    (
        ("err_connection_reset", "err_connection_closed", "connection reset"),
        NetworkFault("connection_reset", "connection_reset", True, "browser: connection reset"),
    ),
    (
        ("err_connection_refused", "err_connection_failed"),
        NetworkFault("connection_error", "connection_error", True, "browser: connection failed"),
    ),
    (
        ("err_cert", "ssl", "certificate"),
        NetworkFault("tls_error", "tls_error", False, "browser: TLS/certificate error"),
    ),
    (
        ("target crashed", "page crashed", "crashed"),
        NetworkFault("browser_crash", "browser_crash", False, "browser: renderer crashed"),
    ),
    (
        ("frame was detached", "frame detached", "detached"),
        NetworkFault("detached_frame", "detached_frame", True, "browser: frame detached"),
    ),
    (
        ("target closed", "browser has been closed"),
        NetworkFault("browser_closed", "browser_closed", True, "browser: closed mid-navigation"),
    ),
)


def classify_browser_error(message: str) -> NetworkFault:
    """Map a Playwright/patchright error string onto the HTTP taxonomy.

    Playwright surfaces network failures as error *strings*, not typed
    exceptions, so this is the browser equivalent of
    :func:`classify_exception`. Unknown failures are non-retryable: a browser
    tier is the last escalation, and guessing "retry" risks a zombie Chromium.
    """
    lowered = (message or "").lower()
    for markers, fault in _BROWSER_ERROR_MAP:
        if any(marker in lowered for marker in markers):
            return fault
    return NetworkFault(
        "browser_error", "browser_error", False, f"browser: {message or 'unknown failure'}"
    )
