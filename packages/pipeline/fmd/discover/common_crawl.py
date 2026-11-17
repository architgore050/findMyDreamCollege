"""Common Crawl URL-index discovery via DuckDB (brief §7).

Sweeps the columnar URL index for PDFs on the domain suffixes of the enabled
country. All of that — the index glob, the domain suffixes and the PDF URL
patterns — is registry data in ``config/sources.yaml``; this module contains no
country suffix. This is a discovery backend only: it emits leads and never
evidence. duckdb is an optional dependency (``crawl`` / ``er`` extras), so
absence degrades to :class:`DiscoveryUnavailable` with an actionable message
instead of an ImportError traceback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from fmd.discover.base import (
    DiscoveryUnavailable,
    DocumentRef,
    dedupe_refs,
    params_from_source,
)

logger = logging.getLogger(__name__)

DEFAULT_INDEX_GLOB = (
    "https://data.commoncrawl.org/cc-index/table/cc-main/warc/crawl=*/subset=warc/*.parquet"
)
DEFAULT_SOURCE_ID = "common_crawl"
#: duckdb's httpfs settings. A cold parquet scan can otherwise sit in a socket
#: read for minutes after the link drops. `http_retries` is deliberately small:
#: a retried HTTP range is expensive and the index query is resumable by row.
DUCKDB_HTTP_TIMEOUT_S = 20
DUCKDB_HTTP_RETRIES = 2
DUCKDB_HTTP_RETRY_WAIT_MS = 1000

QueryRunner = Callable[[str, Sequence[str], int], Iterable[tuple[Any, ...]]]


def _import_duckdb():
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise DiscoveryUnavailable(
            "Common Crawl discovery needs duckdb. Install it with "
            "`uv sync --extra crawl` (duckdb is declared in the crawl and er extras)."
        ) from exc
    return duckdb


def _require(values: Sequence[str], name: str) -> tuple[str, ...]:
    cleaned = tuple(str(value) for value in values if str(value).strip())
    if not cleaned:
        raise DiscoveryUnavailable(
            f"Common Crawl discovery has no {name}. Add `params.{name}` to the "
            "common_crawl_index source entry in config/sources.yaml (registry data, "
            "not code)."
        )
    return cleaned


class CommonCrawlIndex:
    def __init__(
        self,
        index_glob: str = DEFAULT_INDEX_GLOB,
        *,
        source_id: str = DEFAULT_SOURCE_ID,
        query_runner: QueryRunner | None = None,
        domains: Sequence[str] = (),
        pdf_url_patterns: Sequence[str] = (),
        fetch_bytes: Callable[[str], bytes] | None = None,
    ) -> None:
        self.index_glob = index_glob
        self.source_id = source_id
        self._query_runner = query_runner
        self.domains = tuple(domains)
        self.pdf_url_patterns = tuple(pdf_url_patterns)
        self._fetch_bytes = fetch_bytes

    @classmethod
    def from_source(
        cls, source: Any, *, query_runner: QueryRunner | None = None
    ) -> CommonCrawlIndex:
        """Build from a registry/seeded source entry; params stay in config."""
        params = params_from_source(source)
        source_id = getattr(source, "id", None)
        if source_id is None and isinstance(source, dict):
            source_id = source.get("id")
        return cls(
            str(params.get("index_glob") or DEFAULT_INDEX_GLOB),
            source_id=str(source_id or DEFAULT_SOURCE_ID),
            query_runner=query_runner,
            domains=tuple(params.get("domain_suffixes") or ()),
            pdf_url_patterns=tuple(params.get("pdf_url_patterns") or ()),
        )

    # ------------------------------------------------------------------ duckdb
    def _sql(self, domains: Sequence[str], limit: int) -> str:
        # `read_parquet` is fed by placeholder; patterns are SQL literals.
        clauses = " OR ".join(f"url LIKE '%.{suffix}/%'" for suffix in domains)
        pdf = "(" + " OR ".join(f"url LIKE '{pattern}'" for pattern in self.pdf_url_patterns) + ")"
        escaped = self.index_glob.replace("'", "''")
        return (
            f"SELECT url FROM read_parquet('{escaped}') "
            f"WHERE ({clauses}) AND {pdf} "
            f"LIMIT {int(limit)}"
        )

    def _run_duckdb(
        self, _index_glob: str, domains: Sequence[str], limit: int
    ) -> list[tuple[Any, ...]]:
        # `_index_glob` satisfies the QueryRunner signature; `_sql` reads the
        # instance's glob so an injected runner can ignore it.
        duckdb = _import_duckdb()
        try:
            conn = duckdb.connect()
            try:
                conn.execute("INSTALL httpfs")
                conn.execute("LOAD httpfs")
                conn.execute(f"SET http_timeout={int(DUCKDB_HTTP_TIMEOUT_S)}")
                conn.execute(f"SET http_retries={int(DUCKDB_HTTP_RETRIES)}")
                conn.execute(f"SET http_retry_wait_ms={int(DUCKDB_HTTP_RETRY_WAIT_MS)}")
            except Exception as exc:  # pragma: no cover - extension/network dependent
                logger.warning("duckdb httpfs unavailable: %s", exc)
                raise DiscoveryUnavailable(
                    "Common Crawl discovery needs the duckdb httpfs extension and "
                    "network access to data.commoncrawl.org."
                ) from exc
            try:
                return list(conn.execute(self._sql(domains, limit)).fetchall())
            finally:
                conn.close()
        except DiscoveryUnavailable:
            raise
        except Exception as exc:  # pragma: no cover - network/parquet dependent
            raise DiscoveryUnavailable(f"Common Crawl index query failed: {exc}") from exc

    # ---------------------------------------------------------------- discovery
    def discover(
        self,
        *,
        domains: Sequence[str] | None = None,
        limit: int = 500,
    ) -> list[DocumentRef]:
        domains = _require(
            tuple(domains) if domains is not None else self.domains, "domain_suffixes"
        )
        _require(self.pdf_url_patterns, "pdf_url_patterns")
        runner = self._query_runner or self._run_duckdb
        try:
            rows = runner(self.index_glob, domains, limit)
        except DiscoveryUnavailable:
            raise
        except ModuleNotFoundError as exc:
            raise DiscoveryUnavailable(str(exc)) from exc
        except Exception as exc:
            raise DiscoveryUnavailable(f"Common Crawl index query failed: {exc}") from exc

        refs: list[DocumentRef] = []
        for row in rows:
            if not row:
                continue
            url = row[0]
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            refs.append(
                DocumentRef(
                    url=url,
                    source_id=self.source_id,
                    discovery_method="common_crawl_index",
                    metadata={"index": self.index_glob},
                )
            )
        return dedupe_refs(refs)
