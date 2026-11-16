"""robots.txt handling (brief §7, invariant 9).

Policies are parsed with ``protego`` and cached per host for 24 hours. The
cache can be backed by the persisted ``domain_politeness`` row so a restart
does not re-fetch every robots file. A missing robots.txt means *allow* but
with a conservative default delay. Robots is never bypassed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from fmd.fetch.politeness import (
    DEFAULT_CRAWL_DELAY_WHEN_NO_ROBOTS,
    SystemClock,
    registrable_domain,
)
from fmd.fetch.types import RobotsDecision, TierUnavailable
from fmd.frontier.state import DomainStateStore, InMemoryDomainStateStore

logger = logging.getLogger(__name__)

ROBOTS_TTL = timedelta(hours=24)
ROBOTS_MAX_BYTES = 512_000

RobotsFetcher = Callable[[str], tuple[int, str]]


class Clock(Protocol):
    def now(self) -> datetime:  # pragma: no cover - protocol
        ...


@dataclass
class _CachedPolicy:
    text: str
    status: int
    fetched_at: datetime
    crawl_delay: float | None
    sitemaps: tuple[str, ...]


def _robots_url(url: str) -> str:
    parts = urlsplit(url)
    netloc = parts.netloc
    return urlunsplit((parts.scheme or "https", netloc, "/robots.txt", "", ""))


def _parse(
    text: str, url: str, user_agent: str, *, status: int, fetched_at: datetime
) -> _CachedPolicy:
    try:
        from protego import Protego
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise TierUnavailable(
            "robots.txt parsing requires protego. Install the crawl extra: "
            "`uv sync --extra crawl` (or `pip install 'fmd[crawl]'`)."
        ) from exc

    policy = Protego.parse(text)
    try:
        crawl_delay = policy.crawl_delay(user_agent)
    except Exception:  # pragma: no cover - protego is permissive
        crawl_delay = None
    sitemaps = tuple(getattr(policy, "sitemaps", ()) or ())
    return _CachedPolicy(
        text=text,
        status=status,
        fetched_at=fetched_at,
        crawl_delay=float(crawl_delay) if crawl_delay is not None else None,
        sitemaps=sitemaps,
    )


class RobotsCache:
    def __init__(
        self,
        fetch: RobotsFetcher | None = None,
        *,
        user_agent: str | None = None,
        ttl: timedelta = ROBOTS_TTL,
        clock: Clock | None = None,
        store: DomainStateStore | None = None,
        conservative_delay: float = DEFAULT_CRAWL_DELAY_WHEN_NO_ROBOTS,
    ) -> None:
        self._fetch = fetch or self._httpx_fetch
        if user_agent is None:
            from fmd.settings import get_settings

            user_agent = get_settings().user_agent
        self.user_agent = user_agent
        self.ttl = ttl
        self.clock = clock or SystemClock()
        self.store = store or InMemoryDomainStateStore()
        self.conservative_delay = conservative_delay
        self._policies: dict[str, _CachedPolicy] = {}
        self._raw: dict[str, str] = {}

    @staticmethod
    def _httpx_fetch(url: str) -> tuple[int, str]:
        import httpx

        from fmd.settings import get_settings

        response = httpx.get(
            url,
            headers={"User-Agent": get_settings().user_agent},
            follow_redirects=True,
            timeout=10.0,
        )
        return response.status_code, response.text[:ROBOTS_MAX_BYTES]

    # -------------------------------------------------------------------- cache
    def _cached(self, host: str, domain: str) -> _CachedPolicy | None:
        policy = self._policies.get(host)
        if policy is not None:
            if self.clock.now() - policy.fetched_at < self.ttl:
                return policy
            self._policies.pop(host, None)
        state = self.store.load(domain)
        fetched_at = state.robots_fetched_at
        if fetched_at is not None and state.robots_json is not None:
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=UTC)
            if self.clock.now() - fetched_at < self.ttl:
                text = str(state.robots_json.get("text", ""))
                status = int(state.robots_json.get("status", 200))
                policy = _parse(
                    text,
                    "",
                    self.user_agent,
                    status=status,
                    fetched_at=fetched_at,
                )
                self._policies[host] = policy
                return policy
        return None

    def _store(self, domain: str, policy: _CachedPolicy) -> None:
        state = self.store.load(domain)
        state.robots_fetched_at = policy.fetched_at
        state.robots_json = {"text": policy.text, "status": policy.status}
        state.crawl_delay = policy.crawl_delay
        state.sitemaps = policy.sitemaps
        self.store.save(state)

    # ------------------------------------------------------------------- public
    def crawl_delay(self, url: str) -> float | None:
        decision = self.decision(url)
        return decision.crawl_delay

    def sitemaps(self, url: str) -> tuple[str, ...]:
        return self.decision(url).sitemaps

    def decision(self, url: str) -> RobotsDecision:
        parts = urlsplit(url)
        host = parts.netloc.lower()
        domain = registrable_domain(url)
        policy = self._cached(host, domain)
        if policy is None:
            robots = _robots_url(url)
            try:
                status, text = self._fetch(robots)
            except Exception as exc:  # network failure -> conservative allow
                logger.warning("robots.txt fetch failed for %s: %s", robots, exc)
                decision = RobotsDecision(
                    url=url,
                    allowed=True,
                    reason="robots_unavailable",
                    crawl_delay=self.conservative_delay,
                )
                self._log(decision)
                return decision
            if status in (401, 403):
                # An access-controlled robots file is not permission to crawl.
                decision = RobotsDecision(
                    url=url,
                    allowed=False,
                    reason="robots_forbidden",
                    status=status,
                )
                self._log(decision)
                return decision
            if status >= 400:
                policy = _parse(
                    "", url, self.user_agent, status=status, fetched_at=self.clock.now()
                )
                self._policies[host] = policy
                self._store(domain, policy)
                decision = RobotsDecision(
                    url=url,
                    allowed=True,
                    reason="no_robots_txt",
                    crawl_delay=self.conservative_delay,
                    status=status,
                )
                self._log(decision)
                return decision
            text = text[:ROBOTS_MAX_BYTES]
            policy = _parse(text, url, self.user_agent, status=status, fetched_at=self.clock.now())
            self._policies[host] = policy
            self._raw[host] = text
            self._store(domain, policy)

        # Re-parse per URL: protego policies are cheap and URL-specific.
        try:
            from protego import Protego
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise TierUnavailable(
                "robots.txt parsing requires protego. Install the crawl extra."
            ) from exc
        parsed = Protego.parse(policy.text)
        allowed = parsed.can_fetch(url, self.user_agent)
        delay = policy.crawl_delay
        if delay is None and policy.status >= 400:
            delay = self.conservative_delay
        decision = RobotsDecision(
            url=url,
            allowed=allowed,
            reason=(
                "disallowed_by_robots"
                if not allowed
                else "no_robots_txt"
                if policy.status >= 400
                else "allowed"
            ),
            crawl_delay=delay,
            sitemaps=policy.sitemaps,
            status=policy.status,
        )
        self._log(decision)
        return decision

    @staticmethod
    def _log(decision: RobotsDecision) -> None:
        logger.info(
            "robots decision url=%s allowed=%s reason=%s status=%s delay=%s",
            decision.url,
            decision.allowed,
            decision.reason,
            decision.status,
            decision.crawl_delay,
        )
