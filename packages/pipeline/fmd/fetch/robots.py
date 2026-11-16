"""robots.txt handling (brief §7, invariant 9).

Robots is never bypassed, but a robots.txt that is *unreachable* must not be
silently read as "Disallow: /". The policy below is explicit and follows RFC
9309 plus the fetch ladder's block signals:

===================================  ==================  =====================
robots.txt answer                    decision            cached?
===================================  ==================  =====================
2xx, parses                          honour it exactly   yes, ``ROBOTS_TTL``
4xx other than 401/403 (404, 410…)   ``unavailable_allow`` with a
                                     conservative delay (RFC 9309's
                                     "unavailable" case) yes, ``ROBOTS_TTL``
401/403 after ladder escalation      ``unreachable_disallow``: abstain, never
                                     guess allow        in-process only; retried
                                     on the next run
5xx / network / transport failure    ``unreachable_disallow``: abstain
                                                        in-process only; retried
                                                        on the next run
===================================  ==================  =====================

"Abstain" means no document is published for that URL *and* the reason is
recorded, so a run summary can say "N sources blocked by robots" separately from
"N sources failed". A transient failure is never persisted as a permanent
policy: it is remembered only for ``TRANSIENT_TTL`` inside this process and is
re-fetched on the next run.

robots.txt is fetched with the same truthful UA as every other request. On a
block signal (403/429/503 or ``cf-mitigated``) it escalates through the *same*
tier ladder as page fetches: a 403 from CloudFront is a UA/TLS/appearance block,
not a robots ``Disallow``, so it deserves the ladder's second and third tiers
before we abstain.

Policies are parsed with ``protego`` and cached per host for 24 hours. The
cache can be backed by the persisted ``domain_politeness`` row so a restart
does not re-fetch every robots file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from fmd.fetch.politeness import (
    DEFAULT_CRAWL_DELAY_WHEN_NO_ROBOTS,
    SystemClock,
    registrable_domain,
)
from fmd.fetch.types import (
    BLOCK_STATUSES,
    FetchRequest,
    RobotsDecision,
    TierFetcher,
    TierUnavailable,
)
from fmd.frontier.state import DomainStateStore, InMemoryDomainStateStore

logger = logging.getLogger(__name__)

ROBOTS_TTL = timedelta(hours=24)
# A transient failure is retried on the next run, but is reused inside a long
# run so one blocked domain does not re-escalate on every URL.
TRANSIENT_TTL = timedelta(minutes=10)
ROBOTS_MAX_BYTES = 512_000

# RFC 9309 §2.3.1.3: 4xx (other than 401/403) means "unavailable" => allow.
# 401/403 is an access control answer, not permission to crawl.
ACCESS_CONTROLLED_STATUSES: frozenset[int] = frozenset({401, 403})

CASE_POLICY = "policy"
CASE_UNAVAILABLE_ALLOW = "unavailable_allow"
CASE_UNREACHABLE_DISALLOW = "unreachable_disallow"

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
    case: str
    reason: str
    transient: bool = False


def _robots_url(url: str) -> str:
    parts = urlsplit(url)
    netloc = parts.netloc
    return urlunsplit((parts.scheme or "https", netloc, "/robots.txt", "", ""))


def _parse(
    text: str,
    url: str,
    user_agent: str,
    *,
    status: int,
    fetched_at: datetime,
    case: str = CASE_POLICY,
    reason: str = "allowed",
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
        case=case,
        reason=reason,
    )


class RobotsCache:
    def __init__(
        self,
        fetch: RobotsFetcher | None = None,
        *,
        fetchers: Iterable[TierFetcher] | None = None,
        user_agent: str | None = None,
        ttl: timedelta = ROBOTS_TTL,
        transient_ttl: timedelta = TRANSIENT_TTL,
        clock: Clock | None = None,
        store: DomainStateStore | None = None,
        conservative_delay: float = DEFAULT_CRAWL_DELAY_WHEN_NO_ROBOTS,
    ) -> None:
        # `fetch` is the single-shot seam used by tests and simple callers.
        # When it is absent, robots.txt goes through the tier ladder so a 403
        # can escalate instead of being mistaken for `Disallow: /`.
        self._fetch = fetch
        self._fetchers = None if fetchers is None else list(fetchers)
        if user_agent is None:
            from fmd.settings import get_settings

            user_agent = get_settings().user_agent
        self.user_agent = user_agent
        self.ttl = ttl
        self.transient_ttl = transient_ttl
        self.clock = clock or SystemClock()
        self.store = store or InMemoryDomainStateStore()
        self.conservative_delay = conservative_delay
        self._policies: dict[str, _CachedPolicy] = {}
        self._raw: dict[str, str] = {}
        self._cases: dict[str, str] = {}
        self._blocked_domains: set[str] = set()
        self._unreachable_domains: set[str] = set()
        self._counts: dict[str, int] = {
            "allowed": 0,
            "blocked_by_robots": 0,
            CASE_UNAVAILABLE_ALLOW: 0,
            CASE_UNREACHABLE_DISALLOW: 0,
        }

    def _ladder_fetchers(self) -> list[TierFetcher]:
        if self._fetchers is None:
            # Imported lazily: `fmd.fetch.ladder` imports this module.
            from fmd.fetch.ladder import default_fetchers

            self._fetchers = default_fetchers()
        return self._fetchers

    def _fetch_robots(self, url: str) -> tuple[int, str]:
        """Fetch one robots.txt, escalating on the same block signals as pages.

        A block (403/429/503/``cf-mitigated``) is a UA/appearance signal, not a
        robots ``Disallow``, so it advances one rung. 401/403 that survive every
        available tier are returned as-is and the caller abstains. 5xx and
        transport failures are terminal for this run: never escalate a broken
        origin into a browser, and never treat its silence as permission.
        """
        if self._fetch is not None:
            return self._fetch(url)

        fetchers = self._ladder_fetchers()
        last: tuple[int, str] = (0, "")
        for index, fetcher in enumerate(fetchers):
            try:
                result = fetcher.fetch(FetchRequest(url=url))
            except TierUnavailable as exc:
                logger.info("robots tier %s unavailable for %s: %s", fetcher.name, url, exc)
                continue
            except Exception as exc:  # transport failure -> abstain, no escalate
                logger.warning("robots fetch failed for %s via %s: %s", url, fetcher.name, exc)
                return 0, ""
            last = (result.status, result.text)
            if result.blocked and index < len(fetchers) - 1:
                logger.info(
                    "robots block signal %s from tier %s for %s; escalating",
                    result.status,
                    fetcher.name,
                    url,
                )
                continue
            return last
        return last

    # -------------------------------------------------------------------- cache
    def _cached(self, host: str, domain: str) -> _CachedPolicy | None:
        policy = self._policies.get(host)
        if policy is not None:
            ttl = self.transient_ttl if policy.transient else self.ttl
            if self.clock.now() - policy.fetched_at < ttl:
                return policy
            self._policies.pop(host, None)
        state = self.store.load(domain)
        fetched_at = state.robots_fetched_at
        if fetched_at is not None and state.robots_json is not None:
            # A transient failure is never reused across runs.
            if state.robots_json.get("transient"):
                return None
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
                    case=str(state.robots_json.get("case", CASE_POLICY)),
                    reason=str(state.robots_json.get("reason", "allowed")),
                )
                self._policies[host] = policy
                return policy
        return None

    def _store(self, domain: str, policy: _CachedPolicy) -> None:
        state = self.store.load(domain)
        state.robots_fetched_at = policy.fetched_at
        state.robots_json = {
            "text": policy.text,
            "status": policy.status,
            "case": policy.case,
            "reason": policy.reason,
            "transient": policy.transient,
        }
        state.crawl_delay = policy.crawl_delay
        state.sitemaps = policy.sitemaps
        self.store.save(state)

    def _remember(self, host: str, domain: str, policy: _CachedPolicy) -> None:
        self._policies[host] = policy
        self._store(domain, policy)

    # ------------------------------------------------------------------- public
    def crawl_delay(self, url: str) -> float | None:
        decision = self.decision(url)
        return decision.crawl_delay

    def sitemaps(self, url: str) -> tuple[str, ...]:
        return self.decision(url).sitemaps

    def case_for(self, url: str) -> str | None:
        """The last robots case recorded for this URL's domain.

        Run summaries report ``blocked_by_robots`` and ``unreachable_disallow``
        separately; ``case_for`` lets a caller attribute one source.
        """
        return self._cases.get(registrable_domain(url))

    def summary(self) -> dict[str, int]:
        """Decision counts for a run summary (URL decisions, not domains).

        ``blocked_by_robots`` is an explicit robots ``Disallow``;
        ``unreachable_disallow`` is an abstention (401/403/5xx/network) and
        belongs with failures, not with policy blocks.
        """
        return dict(self._counts)

    def blocked_domains(self) -> tuple[str, ...]:
        """Registrable domains with at least one explicit robots ``Disallow``."""
        return tuple(sorted(self._blocked_domains))

    def unreachable_domains(self) -> tuple[str, ...]:
        """Registrable domains whose robots.txt could not be obtained."""
        return tuple(sorted(self._unreachable_domains))

    def decision(self, url: str) -> RobotsDecision:
        parts = urlsplit(url)
        host = parts.netloc.lower()
        domain = registrable_domain(url)
        policy = self._cached(host, domain)
        if policy is not None and policy.case == CASE_UNREACHABLE_DISALLOW:
            # In-process reuse of an abstention (the persisted copy is marked
            # transient and ignored by `_cached` on the next run).
            decision = RobotsDecision(
                url=url,
                allowed=False,
                reason=CASE_UNREACHABLE_DISALLOW,
                status=policy.status or None,
            )
            self._record(domain, decision)
            return decision
        if policy is None:
            robots = _robots_url(url)
            try:
                status, text = self._fetch_robots(robots)
            except Exception as exc:  # network failure -> abstain, never allow
                logger.warning("robots.txt fetch failed for %s: %s", robots, exc)
                status, text = 0, ""
            fetched_at = self.clock.now()

            if (
                status == 0
                or status in BLOCK_STATUSES
                or status in ACCESS_CONTROLLED_STATUSES
                or status >= 500
            ):
                # Explicitly forbidden, rate-limited, broken, or unreachable.
                # RFC 9309 has no "assume allow" for this; abstain and record
                # why. Transient: the next run tries again (with escalation)
                # rather than treating a temporary block as a permanent policy.
                policy = _CachedPolicy(
                    text="",
                    status=status,
                    fetched_at=fetched_at,
                    crawl_delay=self.conservative_delay,
                    sitemaps=(),
                    case=CASE_UNREACHABLE_DISALLOW,
                    reason=CASE_UNREACHABLE_DISALLOW,
                    transient=True,
                )
                self._remember(host, domain, policy)
                decision = RobotsDecision(
                    url=url,
                    allowed=False,
                    reason=CASE_UNREACHABLE_DISALLOW,
                    status=status or None,
                )
                self._record(domain, decision)
                return decision

            if status >= 400:
                # 404/410 and other 4xx: the RFC 9309 "unavailable" case.
                # Allow, but conservatively — the site has not granted anything.
                policy = _parse(
                    "",
                    url,
                    self.user_agent,
                    status=status,
                    fetched_at=fetched_at,
                    case=CASE_UNAVAILABLE_ALLOW,
                    reason=CASE_UNAVAILABLE_ALLOW,
                )
                self._remember(host, domain, policy)
                decision = RobotsDecision(
                    url=url,
                    allowed=True,
                    reason=CASE_UNAVAILABLE_ALLOW,
                    crawl_delay=self.conservative_delay,
                    status=status,
                )
                self._record(domain, decision)
                return decision

            # 2xx (or a non-error response): honour the policy verbatim.
            text = text[:ROBOTS_MAX_BYTES]
            policy = _parse(
                text,
                url,
                self.user_agent,
                status=status,
                fetched_at=fetched_at,
                case=CASE_POLICY,
                reason="allowed",
            )
            self._raw[host] = text
            self._remember(host, domain, policy)

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
        if delay is None and policy.case == CASE_UNAVAILABLE_ALLOW:
            delay = self.conservative_delay
        reason = "disallowed_by_robots" if not allowed else policy.reason
        decision = RobotsDecision(
            url=url,
            allowed=allowed,
            reason=reason,
            crawl_delay=delay,
            sitemaps=policy.sitemaps,
            status=policy.status,
        )
        self._record(domain, decision)
        return decision

    def _record(self, domain: str, decision: RobotsDecision) -> None:
        if decision.reason == "disallowed_by_robots":
            self._counts["blocked_by_robots"] += 1
            self._cases[domain] = "disallowed_by_robots"
            self._blocked_domains.add(domain)
        elif decision.reason == CASE_UNAVAILABLE_ALLOW:
            self._counts[CASE_UNAVAILABLE_ALLOW] += 1
            self._cases[domain] = CASE_UNAVAILABLE_ALLOW
        elif not decision.allowed:
            self._counts[CASE_UNREACHABLE_DISALLOW] += 1
            self._cases[domain] = CASE_UNREACHABLE_DISALLOW
            self._unreachable_domains.add(domain)
        else:
            self._counts["allowed"] += 1
            self._cases[domain] = "allowed"
        self._log(decision)

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
