"""Postgres connection handling (psycopg 3 pool).

All SQL lives in :mod:`fmd.db.migrations` or :mod:`fmd.db.repositories`, never in
the CLI, API or workstream modules.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from fmd.settings import get_settings


class DatabaseError(RuntimeError):
    """Base class for database-layer failures."""


class Database:
    """A lazily-opened pool with dict rows and optional schema pinning."""

    def __init__(
        self,
        url: str | None = None,
        *,
        search_path: str | None = None,
        min_size: int = 1,
        max_size: int = 5,
        open_now: bool = True,
    ) -> None:
        self.url = url or os.environ.get("DATABASE_URL") or get_settings().database_url
        self.search_path = search_path
        kwargs: dict[str, Any] = {"row_factory": dict_row}
        if search_path:
            if not search_path.replace("_", "").isalnum():
                raise DatabaseError(f"unsafe schema name {search_path!r}")
            kwargs["options"] = f"-c search_path={search_path}"
        self.pool = ConnectionPool(
            conninfo=self.url,
            min_size=min_size,
            max_size=max_size,
            kwargs=kwargs,
            open=False,
        )
        if open_now:
            self.open()

    def open(self, timeout: float = 10.0) -> None:
        try:
            self.pool.open(wait=True, timeout=timeout)
        except Exception as exc:  # pragma: no cover - depends on environment
            raise DatabaseError(
                f"could not connect to Postgres at {self._safe_url()}: {exc}. "
                "Is `docker compose -f infra/docker-compose.yml up -d db` running and "
                "DATABASE_URL correct?"
            ) from exc

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def connection(self) -> Iterator[Connection[Any]]:
        """Checkout a connection; commit on success, roll back on error."""
        with self.pool.connection() as conn:
            yield conn

    def ping(self) -> bool:
        try:
            with self.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _safe_url(self) -> str:
        # Never echo credentials into logs or HTTP responses.
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(self.url)
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


_database: Database | None = None


def get_database() -> Database:
    """Process-wide default database (used by CLI/API)."""
    global _database
    if _database is None:
        _database = Database()
    return _database


def reset_database() -> None:
    """Test helper: drop the process-wide pooled database."""
    global _database
    if _database is not None:
        _database.close()
    _database = None
