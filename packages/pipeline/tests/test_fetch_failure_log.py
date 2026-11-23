"""Offline tests for the append-only scrape-failure log (FMD-CLOUD-025 PR A).

Covers the module (record/read, failure taxonomy, institution hint,
aggregation) and the FetchLadder logging wrapper (single exit for every
failure outcome, successes never logged). The FakeTier/FakeClock pattern is
the one used by ``tests/test_fetch_ladder.py``; the log directory is always
redirected to a tmp path via ``FMD_SCRAPE_FAILURES_DIR`` so no test writes to
the repository.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fmd.fetch.failure_log import (
    FetchFailure,
    aggregate,
    failure_kind_for,
    institution_hint,
    is_failure,
    read_failures,
    record_failure,
)
from fmd.fetch.ladder import FetchLadder
from fmd.fetch.politeness import PolitenessManager
from fmd.fetch.robots import RobotsCache
from fmd.fetch.types import FetchRequest, FetchResult, RobotsDecision
from fmd.frontier.state import InMemoryDomainStateStore

# A real registry source (config/sources.yaml) used for prefix matching.
NIRF_BASE_URL = "https://www.nirfindia.org/"
NIRF_NAME = "NIRF rankings (institution-level proxy only)"


# ----------------------------------------------------------------- fixtures/helpers
class FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono


class FakeTier:
    """One ladder rung answering with canned statuses (test_fetch_ladder pattern)."""

    def __init__(self, name: str, statuses: list[int]) -> None:
        self.name = name
        self.statuses = list(statuses)
        self.calls = 0

    def fetch(self, request: FetchRequest) -> FetchResult:
        self.calls += 1
        status = self.statuses.pop(0) if self.statuses else 200
        return FetchResult(
            url=request.url,
            final_url=request.url,
            status=status,
            headers={"content-type": "text/html"},
            body=b"ok" if status == 200 else b"",
            fetcher_tier=self.name,
        )


class ExplodingTier:
    name = "httpx"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def fetch(self, request: FetchRequest) -> FetchResult:
        raise self.exc


class RobotsAlwaysDenied:
    def decision(self, url: str) -> RobotsDecision:
        return RobotsDecision(url=url, allowed=False, reason="disallowed_by_robots")


def permissive_robots() -> RobotsCache:
    """robots.txt answers 404 -> RFC 9309 "unavailable" => allow (ladder-test pattern)."""
    return RobotsCache(fetch=lambda url: (404, ""), clock=FakeClock())


def make_ladder(tiers, *, robots=None, stage: str | None = None) -> FetchLadder:
    clock = FakeClock()
    return FetchLadder(
        fetchers=tiers,
        robots=robots if robots is not None else permissive_robots(),
        politeness=PolitenessManager(clock=clock, jitter=0.0, store=InMemoryDomainStateStore()),
        clock=clock,
        sleeper=lambda seconds: None,
        stage=stage,
    )


@pytest.fixture
def failures_dir(tmp_path, monkeypatch) -> str:
    directory = tmp_path / "scrape-failures"
    monkeypatch.setenv("FMD_SCRAPE_FAILURES_DIR", str(directory))
    return str(directory)


# ------------------------------------------------------------------- module tests
def test_record_and_read_roundtrip(failures_dir: str) -> None:
    first = FetchFailure(
        ts="2026-01-05T10:00:00+00:00",
        url="https://www.nirfindia.org/missing",
        final_url="https://www.nirfindia.org/missing",
        status=404,
        kind="http_404",
        tier="httpx",
        method="GET",
        elapsed_ms=12.5,
        institution=NIRF_NAME,
        stage="crawl",
    )
    second = FetchFailure(
        ts="2026-02-01T09:30:00+00:00",
        url="https://bfuhs.ac.in/dead",
        error="dns_failure: name or service not known",
        kind="dns_failure",
        status=0,
        tier="httpx",
        institution="BFUHS (Punjab)",
    )
    record_failure(first)
    record_failure(second)  # a different month -> a second JSONL file

    from pathlib import Path

    files = sorted(Path(failures_dir).glob("failures-*.jsonl"))
    assert [path.name for path in files] == ["failures-202601.jsonl", "failures-202602.jsonl"]

    loaded = read_failures()
    assert loaded == [first, second]
    # `since` is a plain ISO string compare.
    assert [f.url for f in read_failures(since="2026-02-01T00:00:00+00:00")] == [second.url]


def test_is_failure_kind_matrix() -> None:
    def result(status: int = 0, error: str | None = None, quarantined: bool = False):
        return FetchResult(
            url="https://example.org/x",
            final_url="https://example.org/x",
            status=status,
            error=error,
            quarantined=quarantined,
        )

    # 200/304 are not failures.
    assert is_failure(result(status=200)) is False
    assert is_failure(result(status=304)) is False
    # 404/410/other 4xx/5xx are failures with stable kinds.
    assert is_failure(result(status=404)) is True
    assert failure_kind_for(result(status=404)) == "http_404"
    assert failure_kind_for(result(status=410)) == "http_410"
    assert failure_kind_for(result(status=500)) == "http_error"
    assert failure_kind_for(result(status=403)) == "http_error"
    # An error string wins, kind = first token before ":".
    assert failure_kind_for(result(error="robots_disallowed")) == "robots_disallowed"
    assert failure_kind_for(result(error="dns_failure: name or service not known")) == (
        "dns_failure"
    )
    # Quarantine alone is a failure; no status and no error -> status_0.
    assert is_failure(result(status=200, quarantined=True)) is True
    assert failure_kind_for(result(status=0)) == "status_0"


def test_institution_hint() -> None:
    # Longest prefix match against the registry's base_url values -> source name.
    assert institution_hint("https://www.nirfindia.org/rankings/2026.html") == NIRF_NAME
    assert institution_hint(NIRF_BASE_URL) == NIRF_NAME
    # A URL no source claims is unresolved.
    assert institution_hint("https://unmatched.example.org/page") is None
    assert institution_hint("") is None


def test_aggregate_grouping() -> None:
    failures = [
        FetchFailure(
            ts="2026-03-01T10:00:00+00:00",
            url="https://www.nirfindia.org/a",
            status=404,
            kind="http_404",
            institution=NIRF_NAME,
        ),
        FetchFailure(
            ts="2026-03-03T11:00:00+00:00",
            url="https://www.nirfindia.org/b",
            error="robots_disallowed",
            kind="robots_disallowed",
            status=0,
            institution=NIRF_NAME,
        ),
        FetchFailure(
            ts="2026-03-02T09:00:00+00:00",
            url="https://bfuhs.ac.in/c",
            error="dns_failure: down",
            kind="dns_failure",
            status=0,
            institution="BFUHS (Punjab)",
        ),
        FetchFailure(
            ts="2026-03-04T12:00:00+00:00",
            url="https://nowhere.example.org/d",
            kind="status_0",
        ),
    ]
    agg = aggregate(failures, top_n=10)
    assert agg["total"] == 4
    # Sorted by (-n, name).
    assert agg["top_institutions"] == [NIRF_NAME, "BFUHS (Punjab)"]
    nirf = agg["by_institution"][NIRF_NAME]
    assert nirf["n"] == 2
    assert nirf["kinds"] == {"http_404": 1, "robots_disallowed": 1}
    assert nirf["first_ts"] == "2026-03-01T10:00:00+00:00"
    assert nirf["last_ts"] == "2026-03-03T11:00:00+00:00"
    assert nirf["sample_urls"] == ["https://www.nirfindia.org/a", "https://www.nirfindia.org/b"]
    bfuhs = agg["by_institution"]["BFUHS (Punjab)"]
    assert bfuhs["n"] == 1
    assert bfuhs["kinds"] == {"dns_failure": 1}
    # No hint -> the unresolved bucket, never an invented institution.
    assert agg["unresolved"] == {"n": 1, "kinds": {"status_0": 1}}
    assert "unresolved" not in agg["by_institution"]
    # top_n truncates the institution list.
    truncated = aggregate(failures, top_n=1)
    assert truncated["top_institutions"] == [NIRF_NAME]
    assert list(truncated["by_institution"]) == [NIRF_NAME]


# ------------------------------------------------------------------- ladder tests
def test_ladder_logs_404_failure(failures_dir: str) -> None:
    ladder = make_ladder([FakeTier("httpx", [404])], stage="crawl")
    result = ladder.fetch("https://www.nirfindia.org/missing-page")
    assert result.status == 404

    records = read_failures()
    assert len(records) == 1
    record = records[0]
    assert record.kind == "http_404"
    assert record.status == 404
    assert record.url == "https://www.nirfindia.org/missing-page"
    assert record.institution == NIRF_NAME
    assert record.stage == "crawl"
    assert record.tier == "httpx"
    assert record.method == "GET"


def test_ladder_logs_transport_failure(failures_dir: str) -> None:
    ladder = make_ladder([ExplodingTier(Exception("connection dropped"))])
    result = ladder.fetch("https://www.nirfindia.org/page")
    # The ladder maps the transport exception onto the shared fault taxonomy.
    assert result.status == 0
    assert result.error is not None

    records = read_failures()
    assert len(records) == 1
    record = records[0]
    assert record.status == 0
    assert record.kind and record.kind != ""
    assert record.kind == result.error.split(":", 1)[0]
    assert record.url == "https://www.nirfindia.org/page"


def test_ladder_logs_robots_disallowed(failures_dir: str) -> None:
    ladder = make_ladder([FakeTier("httpx", [200])], robots=RobotsAlwaysDenied())
    result = ladder.fetch("https://www.nirfindia.org/forbidden")
    assert result.error == "robots_disallowed"

    records = read_failures()
    assert len(records) == 1
    record = records[0]
    assert record.kind == "robots_disallowed"
    assert record.robots_reason == "disallowed_by_robots"
    assert record.tier == "robots"
    assert record.status == 0


def test_ladder_success_not_logged(failures_dir: str) -> None:
    ladder = make_ladder([FakeTier("httpx", [200])])
    result = ladder.fetch("https://www.nirfindia.org/rankings")
    assert result.status == 200
    assert read_failures() == []


def test_ladder_logs_stage_abort_exception(failures_dir: str) -> None:
    """A stage-level raise (no result) is logged too, then re-raised unchanged."""
    from fmd.net.policy import NetworkFault, NetworkUnavailable

    ladder = make_ladder([FakeTier("httpx", [200])])
    # The outage breaker trips after three DNS failures in a row: every later
    # fetch raises NetworkUnavailable before any tier attempt.
    fault = NetworkFault("dns_failure", "dns_failure", False, "name or service not known")
    ladder.guard.record_failure("host-a", fault)
    ladder.guard.record_failure("host-b", fault)
    ladder.guard.record_failure("host-c", fault)
    with pytest.raises(NetworkUnavailable):
        ladder.fetch("https://www.nirfindia.org/page")

    records = read_failures()
    assert len(records) == 1
    assert records[0].status == 0
    assert records[0].kind == "NetworkUnavailable"
    assert records[0].institution == NIRF_NAME


def test_month_token_and_ts_coercion() -> None:
    failure = FetchFailure(ts=datetime(2026, 5, 7, tzinfo=UTC), url="https://x", kind="http_404")
    assert failure.ts.startswith("2026-05-07T")
    assert failure.ts.endswith("+00:00")
    from fmd.fetch.failure_log import month_token

    assert month_token(failure.ts) == "202605"
    assert month_token("").startswith(datetime.now(UTC).strftime("%Y"))


def test_read_failures_skips_malformed_lines(failures_dir: str) -> None:
    from pathlib import Path

    good = FetchFailure(ts="2026-04-01T00:00:00+00:00", url="https://a", kind="http_404")
    record_failure(good)
    path = next(Path(failures_dir).glob("failures-*.jsonl"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    assert read_failures() == [good]


def test_duplicate_404s_grow_the_log(failures_dir: str) -> None:
    """Two failures to the same host append two records; the log grows, never mutates."""
    ladder = make_ladder([FakeTier("httpx", [404, 404])])
    ladder.fetch("https://www.nirfindia.org/one")
    ladder.fetch("https://www.nirfindia.org/two")
    records = read_failures()
    assert [record.url for record in records] == [
        "https://www.nirfindia.org/one",
        "https://www.nirfindia.org/two",
    ]
    assert all(record.ts and record.kind == "http_404" for record in records)
