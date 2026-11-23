"""Append-only scrape-failure log (FMD-CLOUD-025, PR A).

The log answers one question: *which good universities did the scraper lose
out on, and why?* Every failed fetch is appended as one JSON line to a monthly
``failures-YYYYMM.jsonl`` file; ``fmd failures`` aggregates it **by
institution** — the institution is the thing we lose, not a percentage.

Failure taxonomy (``kind``):

* ``robots_disallowed``     — a real policy block (we honour robots, invariant 9).
* ``domain_quarantined``    — the politeness manager quarantined the domain.
* ``all_tiers_blocked``     — every tier got a block signal: the site is fighting us.
* ``http_404`` / ``http_410`` — the page is gone.
* ``http_error``            — any other >= 400 status.
* ``status_0``              — no HTTP answer at all.
* ``<fault.kind>``          — transport failures (``dns_failure``, ``connection_error`` …)
                              via :func:`fmd.net.policy.classify_exception`.

Logging is best-effort and never raises: the fetch path must not break because
evidence of a failure could not be written. The directory defaults to
``<repo>/data/scrape-failures`` (committed: the log is append-only evidence,
invariant 8) and can be redirected with ``FMD_SCRAPE_FAILURES_DIR``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields as dataclass_fields
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fmd.fetch.types import FetchResult

#: Repo root: ``packages/pipeline/fmd/fetch/failure_log.py`` -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[4]

#: Longest ``kind`` token kept from an error string (keeps the log scannable).
KIND_MAX_CHARS = 64

#: Sample URLs kept per institution by :func:`aggregate`.
SAMPLE_URLS_MAX = 3


@dataclass(frozen=True)
class FetchFailure:
    """One scrape failure, appended verbatim to the monthly JSONL file."""

    ts: str
    url: str
    final_url: str | None = None
    status: int | None = None
    error: str | None = None
    kind: str = ""
    tier: str | None = None
    robots_reason: str | None = None
    method: str = "GET"
    elapsed_ms: float | None = None
    institution: str | None = None
    stage: str | None = None

    def __post_init__(self) -> None:
        # ``ts`` is ISO8601 UTC in the log; accept a datetime so callers can
        # pass ``utcnow()`` directly.
        if isinstance(self.ts, datetime):
            object.__setattr__(self, "ts", iso_utc(self.ts))


def iso_utc(value: datetime) -> str:
    """ISO8601 UTC string (sortable, so ``--since`` is a plain string compare)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- config
def failures_dir() -> Path:
    """Where the JSONL log lives (env override, else ``<repo>/data/scrape-failures``)."""
    env = os.environ.get("FMD_SCRAPE_FAILURES_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return REPO_ROOT / "data" / "scrape-failures"


@lru_cache(maxsize=1)
def _source_prefixes() -> tuple[tuple[str, str], ...]:
    """``(base_url, name)`` pairs from the source registry, longest first.

    Best-effort: a missing or invalid registry yields an empty tuple so the
    fetch path never breaks on logging. Cached for the process lifetime.
    """
    try:
        from fmd.config import load_registry

        registry = load_registry()
    except Exception:  # noqa: BLE001 - logging must never depend on config
        return ()
    pairs: list[tuple[str, str]] = []
    for source in registry.sources.sources:
        base_url = (getattr(source, "base_url", None) or "").strip()
        name = (getattr(source, "name", None) or "").strip()
        if base_url and name:
            pairs.append((base_url, name))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return tuple(pairs)


def institution_hint(url: str) -> str | None:
    """The source ``name`` whose ``base_url`` is the longest prefix of ``url``.

    Unmatched URLs return ``None`` and aggregate into the ``unresolved`` bucket.
    """
    if not url:
        return None
    for base_url, name in _source_prefixes():
        if url.startswith(base_url):
            return name
    return None


# --------------------------------------------------------------------------- classify
def is_failure(result: FetchResult) -> bool:
    """True when the fetch did not deliver a usable answer.

    200 and 304 are successes. A transport error, any status >= 400, or a
    quarantined domain is a failure (404/410 included: the page is gone).
    """
    return bool(result.error) or result.status >= 400 or bool(result.quarantined)


def failure_kind_for(result: FetchResult) -> str:
    """Stable ``kind`` token for the failure taxonomy (module docstring)."""
    error = (result.error or "").strip()
    if error:
        token = error.split(":", 1)[0].strip()
        return token[:KIND_MAX_CHARS] or "error"
    if result.status == 404:
        return "http_404"
    if result.status == 410:
        return "http_410"
    if result.status >= 400:
        return "http_error"
    return "status_0"


# --------------------------------------------------------------------------- log I/O
def month_token(ts: str) -> str:
    """``YYYYMM`` from an ISO timestamp string (used for the monthly file name)."""
    return (ts or "")[:7].replace("-", "") or datetime.now(UTC).strftime("%Y%m")


def record_failure(failure: FetchFailure) -> None:
    """Append one JSON line to ``<failures_dir>/failures-YYYYMM.jsonl``.

    Best-effort by contract: any error (disk, permissions, serialization) is
    swallowed so the fetch path cannot break on logging.
    """
    try:
        directory = failures_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"failures-{month_token(failure.ts)}.jsonl"
        line = json.dumps(asdict(failure), ensure_ascii=False, sort_keys=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:  # noqa: BLE001 - logging must never break the fetch path
        return


def read_failures(since: str | None = None) -> list[FetchFailure]:
    """Every logged failure across month files, oldest first.

    ``since`` is an ISO timestamp; records older than it (plain string compare,
    valid because every ``ts`` is ISO8601 UTC) are dropped. Malformed lines are
    skipped, never fatal.
    """
    directory = failures_dir()
    if not directory.is_dir():
        return []
    known = {f.name for f in dataclass_fields(FetchFailure)}
    out: list[FetchFailure] = []
    for path in sorted(directory.glob("failures-*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            try:
                out.append(FetchFailure(**{k: v for k, v in record.items() if k in known}))
            except TypeError:
                continue
    return out


# --------------------------------------------------------------------------- aggregate
def _bump(kinds: dict[str, int], kind: str) -> None:
    kinds[kind] = kinds.get(kind, 0) + 1


def _sorted_kinds(kinds: dict[str, int]) -> dict[str, int]:
    return dict(sorted(kinds.items(), key=lambda item: (-item[1], item[0])))


def aggregate(failures: list[FetchFailure], top_n: int = 10) -> dict:
    """Group failures BY INSTITUTION (the thing we lose), not by percentage.

    ``by_institution`` is sorted by ``(-n, name)`` and truncated to ``top_n``;
    failures with no institution hint land in ``unresolved``.
    """
    by_institution: dict[str, dict] = {}
    unresolved_n = 0
    unresolved_kinds: dict[str, int] = {}
    for failure in failures:
        institution = (failure.institution or "").strip() or None
        kind = (failure.kind or "").strip() or "unknown"
        if institution is None:
            unresolved_n += 1
            _bump(unresolved_kinds, kind)
            continue
        entry = by_institution.setdefault(
            institution,
            {"n": 0, "kinds": {}, "first_ts": failure.ts, "last_ts": failure.ts, "sample_urls": []},
        )
        entry["n"] += 1
        _bump(entry["kinds"], kind)
        if failure.ts < entry["first_ts"]:
            entry["first_ts"] = failure.ts
        if failure.ts > entry["last_ts"]:
            entry["last_ts"] = failure.ts
        if failure.url and failure.url not in entry["sample_urls"]:
            if len(entry["sample_urls"]) < SAMPLE_URLS_MAX:
                entry["sample_urls"].append(failure.url)

    ranked = sorted(by_institution.items(), key=lambda item: (-item[1]["n"], item[0]))
    truncated = {
        name: {
            "n": entry["n"],
            "kinds": _sorted_kinds(entry["kinds"]),
            "first_ts": entry["first_ts"],
            "last_ts": entry["last_ts"],
            "sample_urls": list(entry["sample_urls"]),
        }
        for name, entry in ranked[: max(0, top_n)]
    }
    return {
        "total": len(failures),
        "by_institution": truncated,
        "unresolved": {"n": unresolved_n, "kinds": _sorted_kinds(unresolved_kinds)},
        "top_institutions": [name for name, _ in ranked[: max(0, top_n)]],
    }
