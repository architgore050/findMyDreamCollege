"""A tiny stdio JSON-RPC fake of the invisible-playwright-mcp server (offline).

Speaks just enough of the MCP wire protocol (newline-delimited JSON-RPC 2.0)
to prove the fetcher's spawn/wire path without downloading a browser engine or
touching the network: ``initialize`` -> ``notifications/initialized`` ->
``tools/list`` -> ``tools/call``. It answers the four tools the fetcher uses
with deterministic content driven by the URL.
"""

from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {"name": "browser_open", "inputSchema": {"type": "object", "properties": {}}},
    {
        "name": "browser_navigate",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "wait_until": {"type": "string"}},
        },
    },
    {
        "name": "browser_read_html",
        "inputSchema": {"type": "object", "properties": {"mode": {"type": "string"}}},
    },
    {"name": "browser_status", "inputSchema": {"type": "object", "properties": {}}},
]


def page_for(url: str) -> str:
    return f"<html><body><p>fake page for {url}</p></body></html>"


def handle_call(name: str, arguments: dict) -> dict:
    if name == "browser_open":
        return {"structuredContent": {"opened": True, "browser": "main"}, "text": "opened"}
    if name == "browser_navigate":
        url = str(arguments.get("url") or "")
        # A 404 path is visible through the structured answer, like upstream.
        status = 404 if url.rstrip("/").endswith("/missing") else 200
        LAST_URL["value"] = url
        return {
            "structuredContent": {"status": status, "url": url},
            "text": f"status: {status}\nurl: {url}",
        }
    if name == "browser_read_html":
        mode = str(arguments.get("mode") or "form")
        if mode != "full":
            return {"isError": True, "text": "only mode=full is supported by the fake"}
        # The fetcher cannot know the URL here; the fake keeps a last URL.
        return {"text": LAST_URL["value"] and page_for(LAST_URL["value"])}
    if name == "browser_status":
        return {"structuredContent": {"open": True, "url": LAST_URL["value"]}, "text": "ok"}
    return {"isError": True, "text": f"unknown tool {name}"}


LAST_URL = {"value": ""}


def reply(request_id, result=None, error=None) -> dict:
    message: dict = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    return message


def serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method", "")
        request_id = message.get("id")
        if request_id is None:
            # A notification (e.g. notifications/initialized): no reply.
            continue
        if method == "initialize":
            params = message.get("params") or {}
            sys.stdout.write(
                json.dumps(
                    reply(
                        request_id,
                        {
                            "protocolVersion": params.get(
                                "protocolVersion", PROTOCOL_VERSION
                            ),
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fake-invisible-playwright-mcp", "version": "0"},
                        },
                    )
                )
                + "\n"
            )
        elif method == "tools/list":
            sys.stdout.write(json.dumps(reply(request_id, {"tools": TOOLS})) + "\n")
        elif method == "tools/call":
            params = message.get("params") or {}
            answer = handle_call(str(params.get("name")), dict(params.get("arguments") or {}))
            text = str(answer.get("text", ""))
            result = {
                "content": [{"type": "text", "text": text}],
                "isError": bool(answer.get("isError", False)),
            }
            if "structuredContent" in answer:
                result["structuredContent"] = answer["structuredContent"]
            sys.stdout.write(json.dumps(reply(request_id, result)) + "\n")
        else:
            sys.stdout.write(
                json.dumps(
                    reply(
                        request_id,
                        error={"code": -32601, "message": f"method not found: {method}"},
                    )
                )
                + "\n"
            )
        sys.stdout.flush()


if __name__ == "__main__":
    serve()