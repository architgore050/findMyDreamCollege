"""Shared offline helpers for adapter tests (no network, deterministic)."""

from __future__ import annotations

from pathlib import Path

from fmd.config.models import DiscoveryParams
from fmd.fetch.types import FetchResult, RobotsDecision

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def with_params(source, **updates):
    current = source.params.model_dump(exclude_none=True) if source.params else {}
    return source.model_copy(update={"params": DiscoveryParams(**{**current, **updates})})


class FakeRobots:
    def __init__(self, *, allowed: bool = True, reason: str = "allowed") -> None:
        self.allowed = allowed
        self.reason = reason

    def decision(self, url: str) -> RobotsDecision:
        return RobotsDecision(url=url, allowed=self.allowed, reason=self.reason)


class FakePoliteness:
    def note_request(self, url: str) -> float:
        return 0.0


class FakeLadder:
    """Stand-in for ``FetchLadder``; returns pre-baked responses by URL."""

    def __init__(
        self,
        responses: FetchResult | dict[str, FetchResult],
        *,
        allowed: bool = True,
        reason: str = "allowed",
    ) -> None:
        self.responses = responses
        self.robots = FakeRobots(allowed=allowed, reason=reason)
        self.politeness = FakePoliteness()
        self.sleeper = lambda _delay: None
        self.calls: list[str] = []

    def fetch(self, url: str, **_kwargs) -> FetchResult:
        self.calls.append(url)
        result = self.responses.get(url) if isinstance(self.responses, dict) else self.responses
        if result is None:
            result = failed_result(url, status=404, error="http_404")
        if result.robots is None:
            result.robots = self.robots.decision(url)
        return result


def ok_result(url: str, text: str, *, status: int = 200, content_type: str = "text/html"):
    return FetchResult(
        url=url,
        final_url=url,
        status=status,
        headers={"content-type": content_type},
        body=text.encode("utf-8"),
        fetcher_tier="httpx",
    )


def failed_result(url: str, *, status: int = 0, error: str | None = "unreachable"):
    return FetchResult(
        url=url,
        final_url=url,
        status=status,
        body=b"",
        fetcher_tier="httpx",
        error=error,
    )
