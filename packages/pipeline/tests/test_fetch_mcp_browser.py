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
    assert session.calls[1] == ("browser_navigate", {"url": url, "wait_until": "domcontentloaded"})
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
