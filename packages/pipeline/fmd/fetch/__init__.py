"""HTTP fetch ladder — workstream B (brief §7).

Public surface:

* :class:`~fmd.fetch.ladder.FetchLadder` — tier escalation + politeness.
* :class:`~fmd.fetch.robots.RobotsCache` — protego robots decisions.
* :class:`~fmd.fetch.politeness.PolitenessManager` — per-domain rate limiting.
* :class:`~fmd.fetch.failure_log.FetchFailure` — append-only scrape-failure log
  (``record_failure`` / ``read_failures`` / ``aggregate``, ``fmd failures``).
* :class:`~fmd.fetch.mcp_browser.InvisiblePlaywrightMCPFetcher` — stealth browser
  tier over the invisible-playwright-mcp server (stdio, official ``mcp`` SDK).
* :class:`~fmd.fetch.firecrawl_local.FirecrawlLocalFetcher` — free, local-only
  Firecrawl tier (``FIRECRAWL_CLOUD=off``; cloud answers fail closed).
* :func:`~fmd.fetch.dedupe.sha256_hex` / ``simhash64`` — snapshot dedupe.
"""

from __future__ import annotations

from fmd.fetch.dedupe import (
    hamming,
    near_duplicate,
    normalise_bytes,
    sha256_hex,
    simhash64,
)
from fmd.fetch.failure_log import (
    FetchFailure,
    aggregate,
    failure_kind_for,
    institution_hint,
    is_failure,
    read_failures,
    record_failure,
)
from fmd.fetch.firecrawl_local import (
    FirecrawlLocalClient,
    FirecrawlLocalFetcher,
    MapLink,
)
from fmd.fetch.ladder import (
    CurlCffiFetcher,
    FetchLadder,
    HttpxFetcher,
    fetch,
)
from fmd.fetch.mcp_browser import InvisiblePlaywrightMCPFetcher, RealMcpSession
from fmd.fetch.politeness import (
    PolitenessManager,
    registrable_domain,
)
from fmd.fetch.robots import RobotsCache
from fmd.fetch.types import (
    BLOCK_STATUSES,
    TERMINAL_STATUSES,
    FetchRequest,
    FetchResult,
    RobotsDecision,
    TierFetcher,
    TierUnavailable,
    UnsupportedReplay,
    http_date,
)

__all__ = [
    "BLOCK_STATUSES",
    "TERMINAL_STATUSES",
    "CurlCffiFetcher",
    "FetchFailure",
    "FetchLadder",
    "FetchRequest",
    "FetchResult",
    "FirecrawlLocalClient",
    "FirecrawlLocalFetcher",
    "HttpxFetcher",
    "InvisiblePlaywrightMCPFetcher",
    "MapLink",
    "PolitenessManager",
    "RealMcpSession",
    "RobotsCache",
    "RobotsDecision",
    "TierFetcher",
    "TierUnavailable",
    "UnsupportedReplay",
    "aggregate",
    "failure_kind_for",
    "fetch",
    "hamming",
    "http_date",
    "institution_hint",
    "is_failure",
    "near_duplicate",
    "normalise_bytes",
    "read_failures",
    "record_failure",
    "registrable_domain",
    "sha256_hex",
    "simhash64",
]
