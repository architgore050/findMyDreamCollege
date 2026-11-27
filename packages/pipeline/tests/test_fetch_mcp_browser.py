"""MCP browser tier tests (offline: fake sessions + a stdio JSON-RPC fake server).

No real browser engine, no network. The fake-server roundtrip proves the
spawn/wire path (initialize -> tools/list -> tools/call over stdio) with
``sys.executable`` as the command; the session-fake tests prove call order,
parsing, and failure honesty. The committed fixtures under
``fixtures/mcp/`` are the REAL wire shapes captured by the 2026-11-27 local
real-server matrix (invisible-playwright-mcp 0.74.0) and are fed straight
through the parser, so an upstream contract change fails loudly offline.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from fmd.fetch.ladder import TIER_MCP_BROWSER
from fmd.fetch.mcp_browser import InvisiblePlaywrightMCPFetcher, RealMcpSession, ToolAnswer
from fmd.fetch.types import FetchRequest, TierUnavailable, UnsupportedReplay

FAKE_SERVER = Path(__file__).resolve().parent / "fixtures" / "mcp" / "fake_mcp_server.py"
MCP_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mcp"


def mcp_fixture(name: str) -> dict:
    return json.loads((MCP_FIXTURES / name).read_text(encoding="utf-8"))


def answer_from_fixture(fixture: dict) -> ToolAnswer:
    """Build the one ToolAnswer shape from a committed wire fixture."""
    text = "\n".join(
        block["text"]
        for block in fixture.get("content", [])
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )
    structured = fixture.get("structuredContent")
    return ToolAnswer(
        structured=dict(structured) if isinstance(structured, dict) else None,
        text=text,
        is_error=bool(fixture.get("isError", False)),
    )


class FakeSession:
    """Scripted MCP session: pops one scripted answer per call, records order."""

    def __init__(self, script: list[ToolAnswer], *, slow_call_s: float = 0.0) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        self.slow_call_s = slow_call_s
        self._active = 0
        self.max_active = 0
        self._mutex = threading.Lock()

    def call_tool(self, name: str, arguments: dict | None = None) -> ToolAnswer:
        with self._mutex:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.slow_call_s:
                import time

                time.sleep(self.slow_call_s)
            self.calls.append((name, dict(arguments or {})))
            if not self.script:
                raise AssertionError("script exhausted; add a ToolAnswer")
            return self.script.pop(0)
        finally:
            with self._mutex:
                self._active -= 1

    def close(self) -> None:
        self.closed = True


def open_ok() -> ToolAnswer:
    return ToolAnswer(structured={"opened": True}, text="opened")


def navigate_ok(url: str, status: int = 200, final_url: str | None = None) -> ToolAnswer:
    return ToolAnswer(
        structured={"status": status, "url": final_url or url},
        text=f"status: {status}\nurl: {final_url or url}",
    )


def navigate_timeout() -> ToolAnswer:
    """A tool-level networkidle timeout, like upstream reports it."""
    return ToolAnswer(
        structured=None,
        text="TimeoutError: page.waitForNavigation: timeout 30000ms exceeded (networkidle)",
        is_error=True,
    )


def read_html(html: str) -> ToolAnswer:
    return ToolAnswer(structured=None, text=html)


def make_fetcher(session: FakeSession, **kwargs) -> InvisiblePlaywrightMCPFetcher:
    return InvisiblePlaywrightMCPFetcher(session_factory=lambda: session, **kwargs)


# ------------------------------------------------------------------- call order + shape


def test_fetch_call_order_is_open_navigate_read_html() -> None:
    url = "https://college.ac.in/page"
    session = FakeSession(
        [open_ok(), navigate_ok(url), read_html("<html><body>hello</body></html>")]
    )
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    assert [name for name, _ in session.calls] == [
        "browser_open",
        "browser_navigate",
        "browser_read_html",
    ]
    assert session.calls[0] == ("browser_open", {})
    assert session.calls[1] == ("browser_navigate", {"url": url, "wait_until": "networkidle"})
    assert session.calls[2] == ("browser_read_html", {"mode": "full"})
    assert result.status == 200
    assert result.ok
    assert result.final_url == url
    assert result.body == b"<html><body>hello</body></html>"
    assert result.fetcher_tier == TIER_MCP_BROWSER
    assert result.method == "GET"
    # a second fetch must NOT re-open the browser (the server is stateful)
    session.script = [navigate_ok(url), read_html("<html>2</html>")]
    fetcher.fetch(FetchRequest(url=url))
    assert [name for name, _ in session.calls][3:] == ["browser_navigate", "browser_read_html"]


def test_navigate_404_and_redirect_are_parsed_from_structured_content() -> None:
    url = "https://college.ac.in/missing"
    redirect_target = "https://college.ac.in/login"
    session = FakeSession(
        [open_ok(), navigate_ok(url, status=404), navigate_ok(url, 302, redirect_target)]
    )
    fetcher = make_fetcher(session)
    session.script.insert(2, read_html("<html></html>"))

    result = fetcher.fetch(FetchRequest(url=url))
    assert result.status == 404
    assert result.terminal

    session.script = [navigate_ok(url, 302, redirect_target), read_html("<html></html>")]
    result = fetcher.fetch(FetchRequest(url=url))
    assert result.status == 302
    assert result.final_url == redirect_target


def test_navigate_status_falls_back_to_text_answer() -> None:
    url = "https://college.ac.in/text-only"
    session = FakeSession(
        [
            open_ok(),
            ToolAnswer(structured=None, text=f"status: 404\nurl: {url}"),
            read_html("<html>gone</html>"),
        ]
    )
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    assert result.status == 404
    assert result.final_url == url


def test_navigate_without_a_parseable_status_is_a_browser_error() -> None:
    from fmd.fetch.ladder import BrowserNavigationError

    url = "https://college.ac.in/odd"
    session = FakeSession([open_ok(), ToolAnswer(structured={}, text="no status here")])
    fetcher = make_fetcher(session)

    with pytest.raises(BrowserNavigationError, match="no parseable HTTP status"):
        fetcher.fetch(FetchRequest(url=url))


# ------------------------------------------------------------------- real 0.74.0 wire shape


def test_navigate_real_sentence_shape_is_parsed_from_the_committed_fixture() -> None:
    """The exact 0.74.0 wire shape: 'navigated to <url> (HTTP <status>)' in the
    text block AND in structuredContent.result, with no explicit fields."""
    fixture = mcp_fixture("navigate_real_answer_200.json")
    url = str(fixture["requested_url"])
    session = FakeSession([open_ok(), answer_from_fixture(fixture), read_html("<html>ok</html>")])
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    assert result.status == 200
    assert result.ok
    assert result.final_url == "http://127.0.0.1:18765/ok"
    assert result.fetcher_tier == TIER_MCP_BROWSER


def test_navigate_real_sentence_shape_reports_the_redirect_final_url() -> None:
    """A 302 target lands on the final page: the sentence carries the final
    URL and final status, so final_url reflects where the page actually is."""
    fixture = mcp_fixture("navigate_real_answer_redirect.json")
    requested = str(fixture["requested_url"])
    session = FakeSession([open_ok(), answer_from_fixture(fixture), read_html("<html>ok</html>")])
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=requested))

    assert result.status == 200
    assert result.ok
    assert result.final_url == "http://127.0.0.1:18765/ok"
    assert result.final_url != requested


def test_navigate_real_sentence_shape_keeps_404_terminal() -> None:
    fixture = mcp_fixture("navigate_real_answer_404.json")
    url = str(fixture["requested_url"])
    session = FakeSession([open_ok(), answer_from_fixture(fixture), read_html("<html>gone</html>")])
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    assert result.status == 404
    assert result.terminal
    assert result.final_url == url


def test_navigate_sentence_in_structured_result_only_is_parsed_too() -> None:
    """Some servers put the sentence only in structuredContent.result (empty
    text blocks); the parser must not depend on the text fallback."""
    session = FakeSession(
        [
            open_ok(),
            ToolAnswer(
                structured={"result": "navigated to http://127.0.0.1:18765/ok (HTTP 200)"},
                text="",
            ),
            read_html("<html>ok</html>"),
        ]
    )
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url="http://127.0.0.1:18765/ok"))

    assert result.status == 200
    assert result.final_url == "http://127.0.0.1:18765/ok"


def test_navigate_sentence_with_a_relative_url_falls_back_to_field_parsing() -> None:
    """A sentence whose 'URL' is not http(s) is not trusted; the older
    explicit-field shape (still sent by older builds) must keep working."""
    session = FakeSession(
        [
            open_ok(),
            ToolAnswer(
                structured={"status": 200, "url": "https://college.ac.in/page"},
                text="navigated to /page (HTTP 200)",
            ),
            read_html("<html>ok</html>"),
        ]
    )
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url="https://college.ac.in/page"))

    assert result.status == 200
    assert result.final_url == "https://college.ac.in/page"


# ------------------------------------------------------------------- wait strategy (JS race)


def test_navigate_waits_for_networkidle_by_default() -> None:
    """2026-11-27 real-host finding: domcontentloaded + immediate read
    returned the stale static shell in 1 of 10 JS fetches. The default wait
    strategy must be networkidle so the network settles before reading."""
    url = "https://college.ac.in/js"
    session = FakeSession([open_ok(), navigate_ok(url), read_html("<html>rendered</html>")])
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    assert session.calls[1] == ("browser_navigate", {"url": url, "wait_until": "networkidle"})
    assert result.body == b"<html>rendered</html>"


def test_wait_until_is_configurable_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FMD_BROWSER_MCP_WAIT_UNTIL", "load")
    url = "https://college.ac.in/page"
    session = FakeSession([open_ok(), navigate_ok(url), read_html("<html>x</html>")])
    fetcher = make_fetcher(session)

    fetcher.fetch(FetchRequest(url=url))

    assert session.calls[1] == ("browser_navigate", {"url": url, "wait_until": "load"})


def test_invalid_wait_until_is_a_preflight_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TierUnavailable, match="FMD_BROWSER_MCP_WAIT_UNTIL"):
        InvisiblePlaywrightMCPFetcher(session_factory=lambda: FakeSession([]), wait_until="bogus")

    monkeypatch.setenv("FMD_BROWSER_MCP_WAIT_UNTIL", "bogus")
    with pytest.raises(TierUnavailable, match="FMD_BROWSER_MCP_WAIT_UNTIL"):
        InvisiblePlaywrightMCPFetcher(session_factory=lambda: FakeSession([]))


def test_networkidle_timeout_falls_back_to_domcontentloaded_once() -> None:
    """A networkidle timeout (tool error or transport) triggers exactly one
    explicit fallback navigation with domcontentloaded, then the read — the
    page is re-navigated, never silently accepted as a stale shell."""
    url = "https://college.ac.in/js"
    session = FakeSession(
        [
            open_ok(),
            navigate_timeout(),
            navigate_ok(url),
            read_html("<html>rendered late</html>"),
        ]
    )
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    wait_strategies = [
        call[1]["wait_until"] for call in session.calls if call[0] == "browser_navigate"
    ]
    assert wait_strategies == ["networkidle", "domcontentloaded"]
    assert result.status == 200
    assert result.body == b"<html>rendered late</html>"


def test_networkidle_transport_error_falls_back_too() -> None:
    """The same fallback covers a raised timeout (e.g. the call-level
    timeout), not only isError tool answers."""

    class TimeoutSession(FakeSession):
        def call_tool(self, name: str, arguments: dict | None = None) -> ToolAnswer:
            first_navigate = name == "browser_navigate" and not any(
                call[0] == "browser_navigate" for call in self.calls
            )
            if first_navigate:
                # Record the attempt like the base would, then fail at the
                # transport layer (the fetcher never sees an answer).
                self.calls.append((name, dict(arguments or {})))
                raise TimeoutError("browser MCP tool 'browser_navigate' timed out")
            return super().call_tool(name, arguments)

    url = "https://college.ac.in/js"
    session = TimeoutSession([open_ok(), navigate_ok(url), read_html("<html>rendered</html>")])
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    wait_strategies = [
        call[1]["wait_until"] for call in session.calls if call[0] == "browser_navigate"
    ]
    assert wait_strategies == ["networkidle", "domcontentloaded"]
    assert result.ok


def test_fallback_is_bounded_to_one_retry() -> None:
    """The fallback never loops: a second failure propagates as the honest
    browser error (exactly two navigations, no third call)."""
    from fmd.fetch.ladder import BrowserNavigationError

    url = "https://college.ac.in/js"
    session = FakeSession([open_ok(), navigate_timeout(), navigate_timeout()])
    fetcher = make_fetcher(session)

    with pytest.raises(BrowserNavigationError):
        fetcher.fetch(FetchRequest(url=url))

    assert len([call for call in session.calls if call[0] == "browser_navigate"]) == 2


def test_explicit_wait_until_gets_no_fallback() -> None:
    """An operator-configured wait_until is honoured as chosen: its failure
    propagates immediately instead of a surprise domcontentloaded retry."""
    from fmd.fetch.ladder import BrowserNavigationError

    url = "https://college.ac.in/page"
    session = FakeSession([open_ok(), navigate_timeout()])
    fetcher = make_fetcher(session, wait_until="load")

    with pytest.raises(BrowserNavigationError):
        fetcher.fetch(FetchRequest(url=url))

    wait_strategies = [
        call[1]["wait_until"] for call in session.calls if call[0] == "browser_navigate"
    ]
    assert wait_strategies == ["load"]


# ------------------------------------------------------------------- honesty


def test_browser_open_failure_cause_is_reported_honestly_and_not_cached() -> None:
    url = "https://college.ac.in/page"
    session = FakeSession(
        [ToolAnswer(structured={"opened": False, "cause": "downloading"}, text="downloading")]
    )
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    assert result.status == 0
    assert result.error == "mcp_browser_downloading"
    assert not result.ok
    # the failure must not poison the next attempt: a fresh open is tried
    session.script = [open_ok(), navigate_ok(url), read_html("<html>ok</html>")]
    result = fetcher.fetch(FetchRequest(url=url))
    assert result.ok


def test_browser_open_real_error_fixture_preserves_the_structured_cause() -> None:
    """The real 0.74.0 refusal: isError=true with structuredContent
    {opened: false, cause: no_display}. The cause must reach the honest
    FetchResult (mcp_browser_no_display), not be swallowed by a generic
    tool-error raise before the open path can read it."""
    fixture = mcp_fixture("browser_open_error_no_display.json")
    url = "https://college.ac.in/page"
    session = FakeSession([answer_from_fixture(fixture)])
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url=url))

    assert result.status == 0
    assert result.error == "mcp_browser_no_display"
    assert not result.ok
    assert result.body == b""
    # the failure must not poison the next attempt either
    session.script = [open_ok(), navigate_ok(url), read_html("<html>ok</html>")]
    assert fetcher.fetch(FetchRequest(url=url)).ok


def test_browser_open_error_cause_is_scanned_from_text_when_unstructured() -> None:
    session = FakeSession(
        [
            ToolAnswer(
                structured=None,
                text="Error: engine still downloading, retry later",
                is_error=True,
            )
        ]
    )
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url="https://college.ac.in/page"))

    assert result.status == 0
    assert result.error == "mcp_browser_downloading"


def test_browser_open_tool_error_without_any_known_cause_is_still_honest() -> None:
    session = FakeSession([ToolAnswer(structured=None, text="boom", is_error=True)])
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url="https://college.ac.in/page"))

    assert result.status == 0
    assert result.error == "mcp_browser_unknown_cause"


def test_navigate_tool_error_still_raises_a_browser_navigation_error() -> None:
    from fmd.fetch.ladder import BrowserNavigationError

    url = "https://college.ac.in/page"
    # The first refusal triggers the one bounded fallback; the SECOND refusal
    # is the honest raise (and proves the fallback is bounded).
    session = FakeSession(
        [
            open_ok(),
            ToolAnswer(structured=None, text="net::ERR_CONNECTION_REFUSED at login", is_error=True),
            ToolAnswer(
                structured=None, text="net::ERR_CONNECTION_REFUSED at login again", is_error=True
            ),
        ]
    )
    fetcher = make_fetcher(session)

    with pytest.raises(BrowserNavigationError, match="browser_navigate refused"):
        fetcher.fetch(FetchRequest(url=url))


def test_stdio_fake_server_preserves_the_open_error_cause_over_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK must hand isError results back as results (not raise), so the
    structured cause survives the real spawn path. The trigger var is
    STEALTHFOX_-prefixed on purpose: that is also what proves the fetcher's
    env passthrough forwards it to the spawned server."""
    pytest.importorskip("mcp", reason="the official mcp SDK (crawl extra) is not installed")
    monkeypatch.setenv("STEALTHFOX_FAKE_MCP_OPEN_ERROR", "no_display")
    url = "https://college.ac.in/fake"
    fetcher = InvisiblePlaywrightMCPFetcher(
        command=sys.executable, args=[str(FAKE_SERVER)], call_timeout_s=30.0
    )
    try:
        result = fetcher.fetch(FetchRequest(url=url))
        assert result.status == 0
        assert result.error == "mcp_browser_no_display"
    finally:
        fetcher.close()


def test_malformed_open_answer_is_honest_too() -> None:
    session = FakeSession([ToolAnswer(structured=None, text="???")])
    fetcher = make_fetcher(session)

    result = fetcher.fetch(FetchRequest(url="https://college.ac.in/page"))

    assert result.status == 0
    assert result.error == "mcp_browser_bad_open_answer"


def test_non_get_refused_before_any_mcp_call() -> None:
    session = FakeSession([])
    fetcher = make_fetcher(session)

    with pytest.raises(UnsupportedReplay, match="cannot faithfully replay POST"):
        fetcher.fetch(FetchRequest(url="https://college.ac.in/api", method="POST", body=b"x"))
    assert session.calls == []


def test_disabled_tier_raises_tier_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMD_BROWSER_MCP_ENABLED", "0")
    fetcher = InvisiblePlaywrightMCPFetcher(session_factory=lambda: FakeSession([]))

    with pytest.raises(TierUnavailable, match="disabled"):
        fetcher.fetch(FetchRequest(url="https://college.ac.in/page"))


def test_missing_command_raises_tier_unavailable() -> None:
    # No session factory: the real spawn path runs and the missing command
    # surfaces as TierUnavailable (never a crash, never a fake page).
    fetcher = InvisiblePlaywrightMCPFetcher(
        command="/nonexistent/fmd-mcp-command", args=[], call_timeout_s=5
    )

    with pytest.raises(TierUnavailable, match="command not found|failed to initialize"):
        fetcher.fetch(FetchRequest(url="https://college.ac.in/page"))


def test_close_terminates_the_session_and_next_fetch_reopens() -> None:
    url = "https://college.ac.in/page"
    session = FakeSession([open_ok(), navigate_ok(url), read_html("<html>a</html>")])
    fetcher = make_fetcher(session)
    fetcher.fetch(FetchRequest(url=url))

    fetcher.close()
    assert session.closed

    session.script = [open_ok(), navigate_ok(url), read_html("<html>b</html>")]
    result = fetcher.fetch(FetchRequest(url=url))
    assert result.ok
    assert [name for name, _ in session.calls] == [
        "browser_open",
        "browser_navigate",
        "browser_read_html",
        "browser_open",
        "browser_navigate",
        "browser_read_html",
    ]


# ------------------------------------------------------------------- serialization


def test_lock_serializes_concurrent_fetches() -> None:
    url = "https://college.ac.in/page"
    session = FakeSession([], slow_call_s=0.02)
    fetcher = make_fetcher(session)
    first = [open_ok(), navigate_ok(url), read_html("<html>1</html>")]
    second = [navigate_ok(url), read_html("<html>2</html>")]
    session.script = first + second

    threads = [
        threading.Thread(target=lambda: fetcher.fetch(FetchRequest(url=url))),
        threading.Thread(target=lambda: fetcher.fetch(FetchRequest(url=url))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert session.max_active == 1, "the stateful one-page server must never see concurrent calls"
    assert [name for name, _ in session.calls] == [
        "browser_open",
        "browser_navigate",
        "browser_read_html",
        "browser_navigate",
        "browser_read_html",
    ]


# ------------------------------------------------------------------- real spawn path


def test_stdio_fake_server_roundtrip_proves_the_spawn_wire_path() -> None:
    pytest.importorskip("mcp", reason="the official mcp SDK (crawl extra) is not installed")
    url = "https://college.ac.in/fake"
    fetcher = InvisiblePlaywrightMCPFetcher(
        command=sys.executable,
        args=[str(FAKE_SERVER)],
        call_timeout_s=30.0,
    )
    try:
        result = fetcher.fetch(FetchRequest(url=url))
        assert result.status == 200
        assert result.fetcher_tier == TIER_MCP_BROWSER
        assert f"fake page for {url}".encode() in result.body
        missing = fetcher.fetch(FetchRequest(url=url + "/missing"))
        assert missing.status == 404
    finally:
        fetcher.close()


def test_real_session_refuses_a_missing_command_directly() -> None:
    pytest.importorskip("mcp", reason="the official mcp SDK (crawl extra) is not installed")
    session = RealMcpSession(command="/nonexistent/fmd-mcp-command", args=[], call_timeout_s=5)
    with pytest.raises(TierUnavailable, match="command not found|failed to initialize"):
        session.start()


# ------------------------------------------------------------------- display env


def test_passthrough_env_forwards_display_vars_and_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fmd.fetch.mcp_browser import _passthrough_env

    monkeypatch.setenv("STEALTHFOX_HEADLESS", "0")
    monkeypatch.setenv("STEALTHFOX_SEED", "abc123")
    monkeypatch.setenv("INVISIBLE_MCP_HOME", "/opt/invisible")
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("XAUTHORITY", "/run/x/.Xauthority")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")

    env = _passthrough_env()

    assert env["STEALTHFOX_HEADLESS"] == "0"
    assert env["STEALTHFOX_SEED"] == "abc123"
    assert env["INVISIBLE_MCP_HOME"] == "/opt/invisible"
    assert env["DISPLAY"] == ":1"
    assert env["XAUTHORITY"] == "/run/x/.Xauthority"
    assert env["WAYLAND_DISPLAY"] == "wayland-0"
    assert env["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"


def test_passthrough_env_defaults_headless_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-11-27 real-host finding: upstream's own default when the variable
    is UNSET is headless=1, which needs Xvfb and failed with
    mcp_browser_no_display. The spawn env must make the documented headed
    default real."""
    from fmd.fetch.mcp_browser import _passthrough_env

    monkeypatch.delenv("STEALTHFOX_HEADLESS", raising=False)

    env = _passthrough_env()

    assert env["STEALTHFOX_HEADLESS"] == "0"


def test_passthrough_env_keeps_explicit_headless_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fmd.fetch.mcp_browser import _passthrough_env

    monkeypatch.setenv("STEALTHFOX_HEADLESS", "1")
    assert _passthrough_env()["STEALTHFOX_HEADLESS"] == "1"

    monkeypatch.setenv("STEALTHFOX_HEADLESS", "0")
    assert _passthrough_env()["STEALTHFOX_HEADLESS"] == "0"


def test_passthrough_env_drops_unset_and_empty_display_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fmd.fetch.mcp_browser import _passthrough_env

    for name in (
        "DISPLAY",
        "XAUTHORITY",
        "WAYLAND_DISPLAY",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", "")  # set but empty: not a usable display

    env = _passthrough_env()

    assert "DISPLAY" not in env
    assert "XAUTHORITY" not in env
    assert "WAYLAND_DISPLAY" not in env


def test_headless_without_xvfb_fails_preflight_naming_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct preflight contract, plus the real-spawn wiring: fetch() on the
    actual spawn path (no session factory) must surface the preflight, not a
    bare browser error."""
    from fmd.fetch import mcp_browser

    monkeypatch.setenv("STEALTHFOX_HEADLESS", "1")
    monkeypatch.setattr(mcp_browser.shutil, "which", lambda name: None)

    with pytest.raises(TierUnavailable, match="xvfb|STEALTHFOX_HEADLESS"):
        mcp_browser._preflight_display()

    fetcher = mcp_browser.InvisiblePlaywrightMCPFetcher(command="true", args=[])
    with pytest.raises(TierUnavailable, match="xvfb|STEALTHFOX_HEADLESS"):
        fetcher.fetch(FetchRequest(url="https://college.ac.in/page"))


def test_headless_with_xvfb_available_passes_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fmd.fetch import mcp_browser

    monkeypatch.setenv("STEALTHFOX_HEADLESS", "1")
    monkeypatch.setattr(
        mcp_browser.shutil, "which", lambda name: "/usr/bin/Xvfb" if name == "Xvfb" else None
    )

    assert mcp_browser._preflight_display() is None


def test_preflight_only_triggers_for_explicit_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset or 0: no Xvfb required (the committed default is headed)."""
    from fmd.fetch import mcp_browser

    monkeypatch.setattr(mcp_browser.shutil, "which", lambda name: None)
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv("STEALTHFOX_HEADLESS", raising=False)
        else:
            monkeypatch.setenv("STEALTHFOX_HEADLESS", value)
        assert mcp_browser._preflight_display() is None
