"""Forward-only SQL migration runner.

Migrations are ``db/migrations/NNNN_name.sql`` applied in filename order inside a
single transaction guarded by an advisory lock. Applied checksums are verified;
editing an applied migration is a hard error (AGENTS.md, schema changes).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from fmd.db.connection import Database, DatabaseError
from fmd.paths import resolve_migrations_dir

# Stable arbitrary key for pg_advisory_xact_lock.
_MIGRATION_LOCK_KEY = 0x464D4400


class MigrationError(DatabaseError):
    pass


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str


def discover_migrations(migrations_dir: str | Path | None = None) -> list[Migration]:
    directory = (
        Path(migrations_dir) if migrations_dir is not None else resolve_migrations_dir()
    )
    if not directory.is_dir():
        raise MigrationError(f"migrations directory does not exist: {directory}")
    files = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix == ".sql"),
        key=lambda p: p.name,
    )
    if not files:
        raise MigrationError(f"no *.sql migrations found in {directory}")
    migrations: list[Migration] = []
    seen: set[str] = set()
    for path in files:
        if not path.name[:1].isdigit():
            raise MigrationError(
                f"migration {path.name!r} must start with a number (e.g. 0002_add_x.sql)"
            )
        if path.stem in seen:
            raise MigrationError(f"duplicate migration version {path.stem!r}")
        seen.add(path.stem)
        migrations.append(
            Migration(
                version=path.stem,
                path=path,
                checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return migrations


def applied_migrations(database: Database) -> dict[str, str]:
    with database.connection() as conn:
        exists = conn.execute(
            "SELECT to_regclass('schema_migrations') AS table_name"
        ).fetchone()
        if not exists or exists["table_name"] is None:
            return {}
        rows = conn.execute(
            "SELECT version, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
    return {row["version"]: row["checksum"] for row in rows}


def run_migrations(
    database: Database, migrations_dir: str | Path | None = None
) -> list[str]:
    """Apply pending migrations atomically; return the versions applied now."""
    migrations = discover_migrations(migrations_dir)
    applied_now: list[str] = []

    with database.connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     text PRIMARY KEY,
                checksum    text NOT NULL,
                applied_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        rows = conn.execute(
            "SELECT version, checksum FROM schema_migrations"
        ).fetchall()
        applied = {row["version"]: row["checksum"] for row in rows}

        for migration in migrations:
            if migration.version in applied:
                if applied[migration.version] != migration.checksum:
                    raise MigrationError(
                        f"migration {migration.version} was modified after it was applied "
                        f"({migration.path}). Migrations are forward-only: add a new "
                        "numbered migration instead of editing this file."
                    )
                continue
            sql = migration.path.read_text(encoding="utf-8")
            try:
                conn.execute(sql)
            except Exception as exc:
                raise MigrationError(
                    f"migration {migration.version} failed: {exc}. The whole run was "
                    "rolled back; fix the migration and re-run."
                ) from exc
            conn.execute(
                "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                (migration.version, migration.checksum),
            )
            applied_now.append(migration.version)

    return applied_now
