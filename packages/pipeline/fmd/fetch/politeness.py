"""Politeness: registrable-domain identity, token buckets, circuit breaker.

Rules from the brief and AGENTS.md invariant 9:

* key on the *registrable* domain (tldextract), not the hostname; two sites under
  different multi-label public suffixes are different domains and must be tested;
* default rate is 1 rps with jitter, plus ``Crawl-delay`` when larger;
* ``Retry-After`` on 429/503 is honoured;
* five consecutive 5xx responses quarantine the domain for six hours;
* state is persisted through :class:`fmd.frontier.state.DomainStateStore`.

tldextract is configured with ``suffix_list_urls=()`` so the bundled public
suffix snapshot is used and no network request is ever made.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from urllib.parse import urlsplit

from fmd.fetch.types import TierUnavailable
from fmd.frontier.state import DomainState, DomainStateStore, InMemoryDomainStateStore

logger = logging.getLogger(__name__)

FIVE_XX_QUARANTINE_HOURS = 6
MAX_CONSECUTIVE_5XX = 5
DEFAULT_CRAWL_DELAY_WHEN_NO_ROBOTS = 1.0


class Clock(Protocol):
    def now(self) -> datetime:  # pragma: no cover - protocol
        ...

    def monotonic(self) -> float:  # pragma: no cover - protocol
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        import time

        return time.monotonic()


_extractor = None


def _tld_extract():
    global _extractor
    if _extractor is None:
        try:
            import tldextract
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
            raise TierUnavailable(
                "registrable-domain politeness requires tldextract. Install the crawl "
                "extra: `uv sync --extra crawl`."
            ) from exc

        # No network: use the package's bundled suffix snapshot.
        _extractor = tldextract.TLDExtract(suffix_list_urls=())
    return _extractor


def registrable_domain(url: str) -> str:
    """Return the eTLD+1 for a URL, including multi-label public suffixes."""
    host = urlsplit(url).hostname or ""
    if not host:
        return ""
    extracted = _tld_extract()(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return host.lower()


@dataclass
class TokenBucket:
    """One-token bucket. ``delay(now)`` is the wait before a request may start."""

    rate: float
    tokens: float = 1.0
    updated_at: datetime | None = None

    def _refill(self, now: datetime) -> None:
        if self.updated_at is None:
            self.updated_at = now
            return
        elapsed = max(0.0, (now - self.updated_at).total_seconds())
        self.tokens = min(1.0, self.tokens + elapsed * self.rate)
        self.updated_at = now

    def delay(self, now: datetime) -> float:
        self._refill(now)
        if self.tokens >= 1.0:
            return 0.0
        return max(0.0, (1.0 - self.tokens) / self.rate)

    def consume(self, now: datetime) -> None:
        self._refill(now)
        self.tokens = max(0.0, self.tokens - 1.0)


class PolitenessManager:
    """Per-domain token bucket + crawl-delay + quarantine."""

    def __init__(
        self,
        store: DomainStateStore | None = None,
        *,
        clock: Clock | None = None,
        default_rps: float = 1.0,
        jitter: float = 0.1,
        random_fn: Callable[[], float] | None = None,
        quarantine_hours: float = FIVE_XX_QUARANTINE_HOURS,
    ) -> None:
        self.store = store or InMemoryDomainStateStore()
        self.clock = clock or SystemClock()
        self.default_rps = default_rps
        self.jitter = jitter
        self._random = random_fn or random.random
        self.quarantine_hours = quarantine_hours
        self._buckets: dict[str, TokenBucket] = {}

    # ------------------------------------------------------------------ helpers
    def state(self, domain: str) -> DomainState:
        return self.store.load(domain)

    def _save(self, state: DomainState) -> None:
        self.store.save(state)

    def _bucket(self, domain: str, state: DomainState) -> TokenBucket:
        bucket = self._buckets.get(domain)
        rate = state.rps or self.default_rps
        if bucket is None or bucket.rate != rate:
            bucket = TokenBucket(rate=rate)
            self._buckets[domain] = bucket
        return bucket

    def _jittered(self, delay: float) -> float:
        if delay <= 0:
            return 0.0
        # Token buckets already rate-limit; jitter only decorrelates starts.
        return delay + self._random() * self.jitter

    # ------------------------------------------------------------------- public
    def set_rate(self, domain: str, rps: float) -> None:
        state = self.state(domain)
        state.rps = rps
        self._save(state)
        self._buckets.pop(domain, None)

    def set_crawl_delay(self, url_or_domain: str, delay: float | None) -> None:
        domain = url_or_domain if "://" not in url_or_domain else registrable_domain(url_or_domain)
        if not domain:
            return
        state = self.state(domain)
        state.crawl_delay = delay
        self._save(state)

    def is_quarantined(self, url_or_domain: str) -> bool:
        domain = url_or_domain if "://" not in url_or_domain else registrable_domain(url_or_domain)
        state = self.state(domain)
        until = state.quarantined_until
        if until is None:
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=UTC)
        return until > self.clock.now()

    def quarantine(self, url_or_domain: str, hours: float | None = None) -> datetime:
        domain = url_or_domain if "://" not in url_or_domain else registrable_domain(url_or_domain)
        state = self.state(domain)
        until = self.clock.now() + timedelta(hours=hours or self.quarantine_hours)
        state.quarantined_until = until
        self._save(state)
        logger.warning("domain %s quarantined until %s", domain, until.isoformat())
        return until

    def delay_for(self, url: str, *, now: datetime | None = None) -> float:
        """Seconds the caller should wait before issuing this request."""
        domain = registrable_domain(url)
        if not domain:
            return 0.0
        now = now or self.clock.now()
        state = self.state(domain)
        until = state.quarantined_until
        if until is not None:
            if until.tzinfo is None:
                until = until.replace(tzinfo=UTC)
            if until > now:
                return (until - now).total_seconds()
        bucket = self._bucket(domain, state)
        delay = bucket.delay(now)
        if state.crawl_delay and state.last_request_at is not None:
            last = state.last_request_at
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            since_last = (now - last).total_seconds()
            delay = max(delay, state.crawl_delay - since_last)
        return self._jittered(max(0.0, delay))

    def note_request(self, url: str, *, at: datetime | None = None) -> float:
        """Consume a token and return the wait that was applied."""
        domain = registrable_domain(url)
        if not domain:
            return 0.0
        at = at or self.clock.now()
        delay = self.delay_for(url, now=at)
        # Reserve at the time the request will actually happen.
        effective = at + timedelta(seconds=delay)
        state = self.state(domain)
        bucket = self._bucket(domain, state)
        bucket.consume(effective)
        state.last_request_at = effective
        self._save(state)
        return delay

    def record_response(
        self,
        url: str,
        status: int,
        *,
        retry_after: float | None = None,
    ) -> bool:
        """Update the circuit breaker. Returns True when quarantine was entered."""
        domain = registrable_domain(url)
        if not domain:
            return False
        state = self.state(domain)
        now = self.clock.now()
        if 500 <= status < 600:
            state.consecutive_5xx += 1
        elif status != 0:
            state.consecutive_5xx = 0
        quarantined = False
        if state.consecutive_5xx >= MAX_CONSECUTIVE_5XX:
            state.quarantined_until = now + timedelta(hours=self.quarantine_hours)
            state.consecutive_5xx = 0
            quarantined = True
            logger.warning(
                "domain %s quarantined after %d consecutive 5xx",
                domain,
                MAX_CONSECUTIVE_5XX,
            )
        if retry_after is not None and retry_after > 0:
            until = now + timedelta(seconds=retry_after)
            if state.quarantined_until is None or until > state.quarantined_until:
                state.quarantined_until = until
        self._save(state)
        return quarantined

    def clear_quarantine(self, url_or_domain: str) -> None:
        domain = url_or_domain if "://" not in url_or_domain else registrable_domain(url_or_domain)
        state = self.state(domain)
        state.quarantined_until = None
        state.consecutive_5xx = 0
        self._save(state)


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse ``Retry-After`` as delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    from email.utils import parsedate_to_datetime

    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    return max(0.0, (when - now).total_seconds())
