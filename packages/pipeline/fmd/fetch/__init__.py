"""HTTP fetch ladder — workstream B (brief §7).

Public surface:

* :class:`~fmd.fetch.ladder.FetchLadder` — tier escalation + politeness.
* :class:`~fmd.fetch.robots.RobotsCache` — protego robots decisions.
* :class:`~fmd.fetch.politeness.PolitenessManager` — per-domain rate limiting.
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
from fmd.fetch.ladder import (
    CurlCffiFetcher,
    FetchLadder,
    HttpxFetcher,
    PatchrightFetcher,
    fetch,
)
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
    "FetchLadder",
    "FetchRequest",
    "FetchResult",
    "HttpxFetcher",
    "PatchrightFetcher",
    "PolitenessManager",
    "RobotsCache",
    "RobotsDecision",
    "TierFetcher",
    "TierUnavailable",
    "UnsupportedReplay",
    "fetch",
    "hamming",
    "http_date",
    "near_duplicate",
    "normalise_bytes",
    "registrable_domain",
    "sha256_hex",
    "simhash64",
]
