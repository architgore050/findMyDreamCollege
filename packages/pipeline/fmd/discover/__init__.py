"""Discovery: SearXNG, Common Crawl, sitemaps — workstream B (brief §7, §8).

Every backend emits :class:`~fmd.discover.base.DocumentRef` leads only. No
parsing, no claims, no scoring. Query templates are registry data.
"""

from __future__ import annotations

from fmd.discover.base import (
    DiscoveryUnavailable,
    DocumentRef,
    dedupe_refs,
    params_from_source,
)
from fmd.discover.common_crawl import CommonCrawlIndex
from fmd.discover.searxng import SearxngClient, SearxngUnavailable, templates_from_source
from fmd.discover.sitemaps import SitemapDiscovery

__all__ = [
    "CommonCrawlIndex",
    "DiscoveryUnavailable",
    "DocumentRef",
    "SearxngClient",
    "SearxngUnavailable",
    "SitemapDiscovery",
    "dedupe_refs",
    "params_from_source",
    "templates_from_source",
]
