"""MCP browser tier: ``invisible-playwright-mcp`` over stdio (v1/free-web-stack).

The third rung of the fetch ladder. Replaces the in-process PatchrightFetcher:
instead of driving a local Chromium from this process, the tier talks to the
`invisible-playwright-mcp <https://github.com/oss-invisible/playwright_mcp>`_
server (MIT, Python >= 3.11) over stdio with the official ``mcp`` SDK. The
server owns the browser (one page, stateful); we only navigate and read.

Contract highlights (docs/mcp-server.md of the pinned upstream):

* ``browser_open`` is the only opener; other tools refuse while closed. It
  returns ``{opened: bool, cause?: str}`` — ``cause`` is one of
  ``profile_locked | no_display | missing_libraries | sandbox_blocked |
  launch_failed | downloading | download_failed``. Every ``opened=false`` is
  reported honestly (``mcp_browser_<cause>``); nothing is retried in silence.
* ``browser_navigate(url, wait_until)`` answers with one sentence —
  ``navigated to <final-url> (HTTP <status>)`` — in the text blocks and in
  ``structuredContent.result`` (the 0.74.0 wire shape; an explicit
  ``status``/``url`` field answer is still parsed for older shapes), so
  redirects and 404s are visible. ``wait_until`` defaults to
  ``networkidle`` (``FMD_BROWSER_MCP_WAIT_UNTIL``:
  ``domcontentloaded|load|networkidle``) so a JS-rendered page is read
  after the network settles, not as its static shell; a networkidle
  timeout or navigation error falls back ONCE to ``domcontentloaded``
  (logged, never silent).
* ``browser_read_html(mode="full")`` returns the page HTML (structure, noise
  removed, uncapped).

Politeness is unchanged: the robots decision and the per-domain rate limit
happen *before* this tier is called (the ladder's job), no auth/CAPTCHA bypass,
and the browser presents its own truthful identity. All MCP calls are
serialized with one lock: the server is stateful with a single page, so two
concurrent fetches would race each other's navigation.

Config (env, no secrets): ``FMD_BROWSER_MCP_ENABLED`` (default ``1``),
``FMD_BROWSER_MCP_COMMAND`` (default ``uvx``), ``FMD_BROWSER_MCP_ARGS``
(default ``invisible-playwright-mcp==0.74.0`` — the version the
2026-11-27 real-server matrix validated; upgrade deliberately).
``STEALTHFOX_*`` and ``INVISIBLE_MCP_HOME`` env vars are passed through to
the spawned server unchanged, plus the display/session vars a headed
browser needs (``DISPLAY``, ``XAUTHORITY``, ``WAYLAND_DISPLAY``,
``XDG_RUNTIME_DIR``, ``DBUS_SESSION_BUS_ADDRESS``). The default is headed
everywhere: ``STEALTHFOX_HEADLESS=0`` in ``.env.example`` AND a spawn-time
``setdefault("STEALTHFOX_HEADLESS", "0")`` so an unset variable cannot
fall back to upstream's own headless=1; an explicit headless=1 without
Xvfb fails a preflight naming the fix.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass
from typing import Any

from fmd.fetch.ladder import BrowserNavigationError
from fmd.fetch.types import (
    FetchRequest,
    FetchResult,
    TierUnavailable,
    UnsupportedReplay,
)
from fmd.net.policy import BrowserPolicy, NetworkGuard, classify_browser_error

logger = logging.getLogger(__name__)

#: Ladder tier name (recorded on every snapshot and failure entry).
TIER_MCP_BROWSER = "invisible_playwright_mcp"

#: Causes the upstream contract expects for ``browser_open`` failures; used
#: verbatim in the honest error string so the failure log stays scannable.
_OPEN_FAILURE_PREFIX = "mcp_browser_"

#: The enumerated causes upstream documents for ``opened=false`` (0.74.0:
#: profile_locked | no_display | missing_libraries | sandbox_blocked |
#: launch_failed | downloading | download_failed). The text fallback scans
#: for exactly these tokens; anything structured passes through verbatim.
_OPEN_CAUSES = (
    "profile_locked",
    "no_display",
    "missing_libraries",
    "sandbox_blocked",
    "launch_failed",
    "downloading",
    "download_failed",
)

#: Env var names passed through to the spawned MCP server, unchanged.
_PASSTHROUGH_ENV_PREFIXES = ("STEALTHFOX_", "INVISIBLE_MCP_")

#: Display/session env vars a headed browser needs on top of the prefixes
#: above. Without them a headed server loses DISPLAY and Firefox reports
#: "no display" (2026-11-27 real-host finding #3).
_DISPLAY_ENV_VARS = (
    "DISPLAY",
    "XAUTHORITY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
)

#: Upstream 0.74.0 answers ``browser_navigate`` with one sentence in the text
#: blocks and in ``structuredContent.result``: "navigated to <url> (HTTP <n>)".
#: Strict on purpose: the final URL (no spaces) and exactly a three-digit
#: status; anything else falls through to the explicit-field shape below.
_NAVIGATION_SENTENCE = re.compile(r"navigated to (?P<url>\S+) \(HTTP (?P<status>\d{3})\)")

#: Navigate wait strategies upstream accepts. The default waits for the
#: network to settle before reading, so a JS-rendered page is not read as
#: its static shell (2026-11-27 real-host finding: domcontentloaded returned
#: the stale shell in 1 of 10 JS fetches).
_WAIT_UNTIL_CHOICES = ("domcontentloaded", "load", "networkidle")
_WAIT_UNTIL_DEFAULT = "networkidle"


@dataclass(frozen=True)
class ToolAnswer:
    """The one shape of a tool call this module reasons about.

    ``structured`` is the server's ``structuredContent`` (may be ``None``);
    ``text`` is the concatenated text blocks (the fallback every tool provides).
    Both the real SDK session and the test fakes produce this, so the parsing
    below is exercised identically offline.
    """

    structured: dict[str, Any] | None
    text: str
    is_error: bool = False


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "off", ""}


def _passthrough_env() -> dict[str, str]:
    """Default-safe environment plus the server's own env vars, unchanged."""
    env: dict[str, str] = {}
    try:
        from mcp.client.stdio import get_default_environment

        env.update(get_default_environment())
    except Exception:  # noqa: BLE001 - env best-effort; never breaks the fetch
        env.update({"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", "")})
    for key, value in os.environ.items():
        if key.startswith(_PASSTHROUGH_ENV_PREFIXES) and value != "":
            env[key] = value
    for key in _DISPLAY_ENV_VARS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    # Runtime default for the spawned server (2026-11-27 real-host finding):
    # upstream itself defaults to headless=1 when the variable is unset, and
    # that needs Xvfb — it failed with mcp_browser_no_display on a headed
    # host whose .env was simply absent. Make the documented default real:
    # explicit host values win, unset means headed (STEALTHFOX_HEADLESS=0).
    env.setdefault("STEALTHFOX_HEADLESS", "0")
    return env


def _preflight_display() -> None:
    """STEALTHFOX_HEADLESS=1 runs the browser under a virtual display.

    Upstream headless still needs Xvfb; without it the server dies with a
    bare browser error. Fail *before* spawning, with the actual fix. Only an
    explicitly requested headless mode triggers this — the committed default
    is headed (``STEALTHFOX_HEADLESS=0``), so tests and display-less hosts
    that never opted in are unaffected.
    """
    raw = os.environ.get("STEALTHFOX_HEADLESS", "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return
    if shutil.which("Xvfb") is not None or shutil.which("xvfb-run") is not None:
        return
    raise TierUnavailable(
        "browser MCP tier preflight: STEALTHFOX_HEADLESS=1 requires Xvfb and none "
        "is installed. Install xvfb (e.g. `apt install xvfb`) or set "
        "STEALTHFOX_HEADLESS=0 to run headed on the operator's display."
    )


def _structured_field(structured: dict[str, Any] | None, names: tuple[str, ...]) -> Any:
    if not structured:
        return None
    lowered = {str(key).lower(): value for key, value in structured.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _answer_field(answer: ToolAnswer, names: tuple[str, ...]) -> Any:
    value = _structured_field(answer.structured, names)
    if value is not None:
        return value
    lowered = answer.text.lower()
    for name in names:
        marker = name.lower() + ":"
        if marker in lowered:
            return answer.text.split(marker, 1)[1].split("\n", 1)[0].strip()
    return None


class RealMcpSession:
    """One spawned MCP server over stdio, driven from a background loop.

    The ``mcp`` SDK is async; the fetch ladder is synchronous. The session
    starts one daemon thread with its own event loop, enters the stdio +
    session contexts there, and forwards ``call_tool`` calls via
    ``run_coroutine_threadsafe``. ``close()`` unwinds the contexts cleanly.
    """

    def __init__(self, *, command: str, args: list[str], call_timeout_s: float = 60.0) -> None:
        import asyncio

        self._asyncio = asyncio
        self._command = command
        self._args = list(args)
        self._call_timeout_s = call_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop: asyncio.Event | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        asyncio = self._asyncio
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ModuleNotFoundError as exc:
            raise TierUnavailable(
                "browser MCP tier is unavailable: the official `mcp` SDK is not "
                "installed. Install the crawl extra: `uv sync --extra crawl`."
            ) from exc
        params = StdioServerParameters(
            command=self._command, args=self._args, env=_passthrough_env()
        )
        self._stop = asyncio.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run,
            args=(stdio_client, ClientSession, params),
            daemon=True,
            name="fmd-mcp-browser",
        )
        self._thread.start()
        connect_timeout = max(5.0, self._call_timeout_s)
        if not self._ready.wait(connect_timeout):
            raise TierUnavailable(
                f"browser MCP server ({self._command} {' '.join(self._args)}) did not "
                f"initialize within {connect_timeout:.0f}s"
            )
        if self._error is not None:
            self.close()
            if isinstance(self._error, TierUnavailable):
                raise self._error
            raise TierUnavailable(
                f"browser MCP server failed to initialize: {self._error}"
            ) from self._error

    def _run(self, stdio_client, client_session, params) -> None:  # noqa: ANN001 - SDK types
        asyncio = self._asyncio
        asyncio.set_event_loop(self._loop)

        async def lifespan() -> None:
            async with (
                stdio_client(params) as (read, write),
                client_session(read, write) as session,
            ):
                await session.initialize()
                self._session = session
                self._ready.set()
                assert self._stop is not None
                await self._stop.wait()

        try:
            self._loop.run_until_complete(lifespan())
        except BaseException as exc:  # transport died or never came up
            self._error = exc
            self._ready.set()
        finally:
            self._session = None
            self._loop.close()

    def close(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            self._loop = None
            return
        if self._stop is not None:
            loop.call_soon_threadsafe(self._stop.set)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
            if thread.is_alive() and not loop.is_closed():  # pragma: no cover - hung shutdown
                loop.call_soon_threadsafe(loop.stop)
        self._loop = None
        self._thread = None
        self._session = None

    # ------------------------------------------------------------------ calls
    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolAnswer:
        session = self._session
        if session is None:
            raise RuntimeError("browser MCP session is not open")
        future = self._asyncio.run_coroutine_threadsafe(
            session.call_tool(name, arguments or {}), self._loop
        )
        try:
            result = future.result(self._call_timeout_s)
        except TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"browser MCP tool {name!r} timed out") from exc
        structured = getattr(result, "structuredContent", None)
        blocks: list[str] = []
        for block in result.content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                blocks.append(text)
        return ToolAnswer(
            structured=dict(structured) if isinstance(structured, dict) else None,
            text="\n".join(blocks),
            is_error=bool(getattr(result, "isError", False)),
        )


class InvisiblePlaywrightMCPFetcher:
    """Third tier: the invisible-playwright-mcp server (stealth Firefox).

    One server process per fetcher, one page, one navigation at a time (the
    lock). ``fetch`` walks the contract: ``browser_open`` once, then
    ``browser_navigate`` + ``browser_read_html(mode="full")`` for every GET.
    """

    name = TIER_MCP_BROWSER

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        enabled: bool | None = None,
        session_factory: Any | None = None,
        call_timeout_s: float = 60.0,
        wait_until: str | None = None,
        policy: BrowserPolicy | None = None,
        guard: NetworkGuard | None = None,
    ) -> None:
        self._enabled_override = enabled
        self._command = command or os.environ.get("FMD_BROWSER_MCP_COMMAND", "uvx").strip() or "uvx"
        if args is not None:
            self._args = [str(part) for part in args]
        else:
            raw_args = os.environ.get("FMD_BROWSER_MCP_ARGS", "invisible-playwright-mcp==0.74.0")
            self._args = [part for part in str(raw_args).split() if part]
        if wait_until is None:
            wait_until = os.environ.get("FMD_BROWSER_MCP_WAIT_UNTIL", _WAIT_UNTIL_DEFAULT)
        wait_until = str(wait_until).strip().lower()
        if wait_until not in _WAIT_UNTIL_CHOICES:
            raise TierUnavailable(
                "browser MCP tier preflight: FMD_BROWSER_MCP_WAIT_UNTIL="
                f"{wait_until!r} is not one of {'|'.join(_WAIT_UNTIL_CHOICES)}."
            )
        self._wait_until = wait_until
        self._session_factory = session_factory
        self._call_timeout_s = call_timeout_s
        self.policy = policy or BrowserPolicy.from_env()
        self.guard = guard
        self._lock = threading.Lock()
        self._session: Any | None = None
        self._opened = False

    # ------------------------------------------------------------------ internals
    def _is_enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        return _env_flag("FMD_BROWSER_MCP_ENABLED", "1")

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        if self._session_factory is not None:
            self._session = self._session_factory()
            return self._session
        _preflight_display()
        session = RealMcpSession(
            command=self._command, args=self._args, call_timeout_s=self._call_timeout_s
        )
        try:
            session.start()
        except (FileNotFoundError, PermissionError) as exc:
            raise TierUnavailable(
                f"browser MCP command not found: {self._command!r} "
                f"({' '.join(self._args)}). Install invisible-playwright-mcp or set "
                "FMD_BROWSER_MCP_COMMAND."
            ) from exc
        self._session = session
        return session

    def _call(self, name: str, arguments: dict[str, Any]) -> ToolAnswer:
        """One MCP tool call.

        Transport failures raise (nothing usable came back). A *tool-level*
        error (``isError=true``) is returned as-is so the caller can read its
        ``structuredContent``: upstream reports ``browser_open`` refusals
        exactly that way (``{opened: false, cause: no_display}``), and raising
        here would throw the cause away before ``_ensure_browser_open`` sees it.
        """
        try:
            answer = self._session.call_tool(name, arguments)
        except TimeoutError as exc:
            raise BrowserNavigationError(
                classify_browser_error(str(exc)), f"{name} timed out"
            ) from exc
        except Exception as exc:  # transport died or the server crashed mid-call
            raise BrowserNavigationError(
                classify_browser_error(str(exc)), f"{name} failed: {exc}"
            ) from exc
        return answer

    @staticmethod
    def _raise_tool_error(answer: ToolAnswer, name: str) -> None:
        """Raise on a tool-level error for calls whose answers carry no
        structured refusal worth preserving (navigate / read_html)."""
        if not answer.is_error:
            return
        raise BrowserNavigationError(
            classify_browser_error(answer.text), f"{name} refused: {answer.text[:200]}"
        )

    @staticmethod
    def _open_cause(answer: ToolAnswer) -> str:
        """The upstream contract names a cause for every ``opened=false``.

        Take it from ``structuredContent`` first (the real 0.74.0 error shape),
        fall back to a known-cause scan of the text for older servers, and
        keep the result a safe token — it becomes part of the failure-log
        error string.
        """
        raw = _answer_field(answer, ("cause",))
        cause = str(raw).strip() if raw is not None else ""
        if not cause:
            lowered = answer.text.lower()
            cause = next((c for c in _OPEN_CAUSES if c in lowered), "unknown_cause")
        safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in cause)
        return safe[:64] or "unknown_cause"

    def _ensure_browser_open(self, url: str) -> FetchResult | None:
        """Open the server's browser once; None means ready to navigate.

        An ``opened=false`` answer — structured or ``isError`` — is never
        retried silently: the caller gets an honest status-0 result naming the
        cause, and the next fetch will attempt a fresh ``browser_open``.
        """
        if self._opened:
            return None
        answer = self._call("browser_open", {})
        opened = _answer_field(answer, ("opened",))
        if opened is False or (answer.is_error and opened is None):
            # The real refusal shape (0.74.0) is isError=true with
            # structuredContent {opened: false, cause: ...}: report the cause
            # verbatim (mcp_browser_no_display etc.), never a generic error.
            self._opened = False
            return self._honest(url, 0, f"{_OPEN_FAILURE_PREFIX}{self._open_cause(answer)}")
        if opened is not True:
            # A malformed open answer is a server-side problem, not a page.
            self._opened = False
            return self._honest(url, 0, f"{_OPEN_FAILURE_PREFIX}bad_open_answer")
        self._opened = True
        return None

    def _honest(self, url: str, status: int, error: str, method: str = "GET") -> FetchResult:
        return FetchResult(
            url=url,
            final_url=url,
            status=status,
            fetcher_tier=self.name,
            error=error,
            method=method,
        )

    @staticmethod
    def _parse_navigation(answer: ToolAnswer, url: str) -> tuple[int, str]:
        """Extract ``(status, final_url)`` from a ``browser_navigate`` answer.

        The pinned upstream (0.74.0) speaks a sentence — "navigated to <url>
        (HTTP <status>)" — in the text blocks and under
        ``structuredContent.result``; it carries no explicit status/url
        fields. That sentence is parsed first. The older explicit-field shape
        (``status:``/``url:`` keys) stays supported for backward
        compatibility, and an answer that carries neither is the honest
        browser error it always was.
        """
        match = _NAVIGATION_SENTENCE.search(answer.text)
        if match is None:
            result_field = _structured_field(answer.structured, ("result",))
            if isinstance(result_field, str):
                match = _NAVIGATION_SENTENCE.search(result_field)
        if match is not None and match.group("url").startswith(("http://", "https://")):
            return int(match.group("status")), match.group("url")
        status_raw = _answer_field(
            answer, ("status", "http_status", "status_code", "httpstatuscode")
        )
        final_url = _answer_field(answer, ("url", "final_url", "landed_url", "finalurl"))
        if isinstance(final_url, str) and final_url.startswith(("http://", "https://")):
            landed = final_url
        else:
            landed = url
        try:
            status = int(str(status_raw).strip())
        except (TypeError, ValueError):
            raise BrowserNavigationError(
                classify_browser_error(answer.text),
                "navigate answer carries no parseable HTTP status",
            ) from None
        return status, landed

    # ------------------------------------------------------------------ TierFetcher
    def fetch(self, request: FetchRequest) -> FetchResult:
        method = str(request.method or "GET").upper()
        if method != "GET":
            # browser_navigate only issues GET. Faking a POST would execute the
            # page's own JavaScript and render whatever it decides, not replay
            # the persisted request/body — refuse explicitly instead.
            raise UnsupportedReplay(
                f"{self.name} cannot faithfully replay {method} requests; a browser "
                "navigation only issues GET. Refusing rather than downgrading."
            )
        if not self._is_enabled():
            raise TierUnavailable("browser MCP tier is disabled (FMD_BROWSER_MCP_ENABLED=0).")
        if self.guard is not None:
            self.guard.check()
        # The server is stateful with exactly one page: serialize every call.
        with self._lock:
            try:
                self._get_session()
            except (ModuleNotFoundError, FileNotFoundError, PermissionError):
                raise
            except TierUnavailable:
                raise
            not_ready = self._ensure_browser_open(request.url)
            if not_ready is not None:
                return not_ready
            try:
                navigate = self._call(
                    "browser_navigate",
                    {"url": request.url, "wait_until": self._wait_until},
                )
                self._raise_tool_error(navigate, "browser_navigate")
            except BrowserNavigationError:
                if self._wait_until != "networkidle":
                    raise
                # One bounded fallback: networkidle can time out on a page
                # with a live beacon (or the navigation can error while
                # waiting). Retry ONCE with domcontentloaded and read what
                # rendered — an explicit, logged degradation, never a
                # silently swallowed error over a known-stale shell.
                logger.warning(
                    "browser_navigate wait_until=networkidle failed for %s; "
                    "falling back once to domcontentloaded",
                    request.url,
                )
                navigate = self._call(
                    "browser_navigate",
                    {"url": request.url, "wait_until": "domcontentloaded"},
                )
                self._raise_tool_error(navigate, "browser_navigate")
            status, final_url = self._parse_navigation(navigate, request.url)
            page = self._call("browser_read_html", {"mode": "full"})
            self._raise_tool_error(page, "browser_read_html")
            body = page.text.encode("utf-8")
        if self.policy.max_page_bytes and len(body) > self.policy.max_page_bytes:
            raise BrowserNavigationError(
                classify_browser_error("page too large"),
                f"page is {len(body)} bytes (cap {self.policy.max_page_bytes})",
            )
        if self.guard is not None:
            self.guard.record_success()
        return FetchResult(
            url=request.url,
            final_url=final_url,
            status=status,
            headers={},
            body=body,
            fetcher_tier=self.name,
            method=method,
        )

    def close(self) -> None:
        """Terminate the spawned server (or the injected session)."""
        with self._lock:
            session = self._session
            self._session = None
            self._opened = False
        if session is None:
            return
        closer = getattr(session, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 - close must never raise
                logger.info("browser MCP close failed: %s", exc)
