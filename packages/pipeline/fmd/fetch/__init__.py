"""HTTP fetch ladder — workstream B (brief §7).

TODO(WS-B): httpx -> curl_cffi -> patchright escalation, robots/politeness,
conditional re-crawl. WS-A intentionally ships no implementation.
"""

from __future__ import annotations


def fetch(url: str) -> bytes:
    raise NotImplementedError("fetch ladder is workstream B (see brief §7)")
