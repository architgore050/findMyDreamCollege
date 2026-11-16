"""Shared fetch-layer value types (brief §7).

Nothing here performs I/O. The types are deliberately plain so tests can build
responses without a network stack and so every tier returns the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import format_datetime
from typing import Protocol, runtime_checkable

# Statuses that indicate an anti-bot / rate-limit block. These are the *only*
# signals that may escalate to a stealth tier (brief §7).
BLOCK_STATUSES: frozenset[int] = frozenset({403, 429, 503})
TERMINAL_STATUSES: frozenset[int] = frozenset({404, 410})


def http_date(value: datetime | str | None) -> str | None:
    """Render a stored ``documents.last_modified`` as an HTTP-date header value.

    ``documents.last_modified`` is a ``timestamptz``; ``If-Modified-Since`` needs
    an IMF-fixdate string in GMT. Passing a string through unchanged keeps this
    usable for a value read straight off a response header.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return format_datetime(value.astimezone(UTC), usegmt=True)


class TierUnavailable(RuntimeError):
    """A stealth tier is not installed.

    The base install must keep working without the ``crawl`` extra, so tier
    imports happen lazily and this error names the extra to install.
    """


class FetchError(RuntimeError):
    """Transport-level failure returned by a tier."""


@dataclass(frozen=True)
class RobotsDecision:
    """The logged answer to "may this crawler fetch this URL?"."""

    url: str
    allowed: bool
    reason: str
    crawl_delay: float | None = None
    sitemaps: tuple[str, ...] = ()
    status: int | None = None


@dataclass(frozen=True)
class FetchRequest:
    """A conditional GET request."""

    url: str
    etag: str | None = None
    last_modified: str | None = None
    referer: str | None = None

    @classmethod
    def revalidate(
        cls,
        url: str,
        *,
        etag: str | None = None,
        last_modified: datetime | str | None = None,
        referer: str | None = None,
    ) -> FetchRequest:
        """Build a conditional GET from stored validators (``documents`` row).

        A 304 answer means the caller must not write a snapshot but must still
        bump ``documents.last_seen``.
        """
        return cls(
            url=url,
            etag=etag,
            last_modified=http_date(last_modified),
            referer=referer,
        )


@dataclass
class FetchResult:
    """One fetch attempt, including the tier that produced it.

    ``status == 0`` means no HTTP response was received (transport failure,
    robots refusal, or quarantine). ``final_url`` is followed redirects.
    """

    url: str
    final_url: str
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    fetcher_tier: str = "none"
    elapsed_ms: float = 0.0
    robots: RobotsDecision | None = None
    error: str | None = None
    quarantined: bool = False

    @property
    def not_modified(self) -> bool:
        return self.status == 304

    @property
    def costs_budget(self) -> bool:
        """A 304 is a revalidation, not a fresh page; it costs no crawl budget."""
        return self.status != 304 and self.status != 0

    @property
    def blocked(self) -> bool:
        return self.status in BLOCK_STATUSES or "cf-mitigated" in self.headers

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        encoding = "utf-8"
        content_type = self.headers.get("content-type", "")
        marker = "charset="
        if marker in content_type.lower():
            encoding = content_type.lower().split(marker, 1)[1].split(";", 1)[0].strip()
        return self.body.decode(encoding, errors="replace")


@runtime_checkable
class TierFetcher(Protocol):
    """A single rung of the ladder."""

    name: str

    def fetch(self, request: FetchRequest) -> FetchResult:  # pragma: no cover - protocol
        ...


class Clock(Protocol):
    """Injectable time source; tests use a deterministic fake."""

    def now(self):  # pragma: no cover - protocol
        ...

    def monotonic(self) -> float:  # pragma: no cover - protocol
        ...
