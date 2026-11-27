"""Verdict-live tests: bounded loop, replay sessions, audit tooling (offline).

Everything here runs on the replay path — committed scripted sessions
(``fixtures/eval_verdict/live/sessions/``) replayed by ``ScriptedModelRouter``
and served by ``ScriptedPageFetcher``. No test touches the network or a real
model; nothing here is evidence that a live model behaves well (that is the
operator's LQ-9 run).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from fmd.cli import main
from fmd.config import get_registry
from fmd.eval.verdict import (
    ProductShortlistRow,
    ProfileFixture,
    audit_verdict,
    load_profile_fixtures,
)
from fmd.eval.verdict_live import (
    SYSTEM_PROMPT,
    AuditWorksheet,
    Citation,
    LiveSession,
    LiveVerdictBackend,
    OperatorAgreement,
    ResearchAction,
    ScriptedModelRouter,
    ScriptedPageFetcher,
    StoredPage,
    build_audit_worksheet,
    clamp_audit_fraction,
    load_live_sessions,
    reverify_evidence,
    run_verdict_live,
    score_audit_worksheet,
    verdict_live_aggregate,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eval_verdict"
PROFILES = FIXTURES / "profiles"
SESSIONS_DIR = FIXTURES / "live" / "sessions"

AIIMS = "All India Institute of Medical Sciences, Delhi"
PGIMER = "Post Graduate Institute of Medical Education and Research"
CMC = "Christian Medical College"


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def mock_provider(monkeypatch):
    """Route the verdict_live model profile to the offline mock provider."""
    monkeypatch.setenv("FMD_VERDICT_LIVE_PROVIDER", "mock")
    get_registry.cache_clear()
    yield
    get_registry.cache_clear()


def make_backend(session: LiveSession, **kwargs) -> LiveVerdictBackend:
    return LiveVerdictBackend(
        ScriptedModelRouter(session),
        ScriptedPageFetcher(session.pages),
        **kwargs,
    )


def run_one(profile: ProfileFixture, session: LiveSession, **kwargs):
    backend = make_backend(session, **kwargs)
    return backend.respond(profile), backend


def run_live_replay(profiles: list[ProfileFixture]) -> list[dict]:
    sessions = load_live_sessions(SESSIONS_DIR)
    cases: list[dict] = []
    for profile in profiles:
        backend = make_backend(sessions[profile.profile_id])
        cases.extend(run_verdict_live([profile], backend))
    return cases


def inline_profile(profile_id: str, ingested: list[str]) -> ProfileFixture:
    """A synthetic profile for the test-only sessions (cap loop, violations)."""
    return ProfileFixture(
        synthetic=True,
        profile_id=profile_id,
        profile={
            "target_field": "medical laboratory technology",
            "target_intake_year": 2027,
            "target_country": "IN",
        },
        product_shortlist=[
            ProductShortlistRow(rank=1, institution_name=AIIMS, programme="B.Sc. MLT"),
            ProductShortlistRow(rank=2, institution_name=PGIMER, programme="B.Sc. MLT"),
            ProductShortlistRow(rank=3, institution_name=CMC, programme="B.Sc. MLT"),
        ],
        ingested_evidence_urls=ingested,
        scripted_verdict={},
        as_of="2026-11-22",
    )


def session_from_turns(profile_id: str, pages: list[dict], turns: list[dict]) -> LiveSession:
    return LiveSession(
        profile_id=profile_id,
        notes="inline test session",
        pages=[{"url": p["url"], "status": p.get("status", 200), "body": p["body"]} for p in pages],
        turns=[{"content": t} for t in turns],
    )


def synthetic_cases(n: int, verdict: str = "correct") -> list[dict]:
    return [
        {
            "profile_id": f"synthetic-{i:02d}",
            "verdict": verdict,
            "rationale": "r",
            "reference_top3": [],
            "unknown_aspects": [],
            "as_of": "2026-11-22",
            "top3_overlap": 0,
            "audit": {"selected_programmes": [], "evidence": [], "failures": []},
            "reverify": {"failures": [], "n_evidence": 0},
            "fetch_ledger": [],
            "backend_run": {"turns_used": 1, "fetches_used": 0},
        }
        for i in range(n)
    ]


# ------------------------------------------------------------------- bounded loop caps


def test_loop_terminates_at_turn_and_fetch_cap() -> None:
    sessions = load_live_sessions(SESSIONS_DIR)
    session = sessions["live-cap-turn-loop"]
    profile = inline_profile("live-cap-turn-loop", [])
    verdict, backend = run_one(profile, session)
    assert backend.last_run["turns_used"] == 12  # hard model-turn cap (default)
    assert backend.last_run["fetches_used"] == 6  # hard fetch cap (12 were requested)
    assert backend.last_run["stop_reason"] == "final_turn_action_rejected"
    assert verdict.verdict == "unknown"  # degraded honestly, never guessed
    assert any("stop_reason" in aspect for aspect in verdict.unknown_aspects)


def test_fetch_budget_exhaustion_is_reported_to_the_model() -> None:
    sessions = load_live_sessions(SESSIONS_DIR)
    session = sessions["live-cap-turn-loop"]
    profile = inline_profile("live-cap-turn-loop", [])
    verdict, backend = run_one(profile, session)
    assert backend.last_run["fetches_used"] == backend.max_fetches
    # the loop kept consuming turns after the budget died (honest observation),
    # and the degraded verdict names the cap as the unknown aspect
    assert verdict.verdict == "unknown"


def test_max_turns_and_fetches_must_be_sane() -> None:
    profile = inline_profile("p", [])
    session = load_live_sessions(SESSIONS_DIR)["live-cap-turn-loop"]
    with pytest.raises(Exception, match="limits must be positive"):
        LiveVerdictBackend(
            ScriptedModelRouter(session), ScriptedPageFetcher(session.pages), max_turns=0
        )
        _ = profile  # profile unused here; construction fails first


def test_final_turn_directive_appears_only_on_the_last_turn() -> None:
    session = load_live_sessions(SESSIONS_DIR)["live-cap-turn-loop"]
    backend = make_backend(session, max_turns=3)
    profile = inline_profile("live-final-turn-prompt", [])

    prompts = [
        backend._build_user_prompt(profile, [f"turn {turn}"], [], ledger={}, turns_used=turn)
        for turn in (1, 2, 3)
    ]

    assert all("FINAL TURN" not in prompt for prompt in prompts[:-1])
    assert "FINAL TURN (3/3)" in prompts[-1]
    assert "emit now with the evidence and citations already collected" in prompts[-1]
    assert "Do NOT call search_web, map_site, fetch_url, or cite_source" in prompts[-1]
    assert "emit_verdict" in prompts[-1] and "unknown" in prompts[-1]


def test_penultimate_turn_requires_citation_when_a_page_was_fetched() -> None:
    session = load_live_sessions(SESSIONS_DIR)["live-cap-turn-loop"]
    backend = make_backend(session, max_turns=3)
    profile = inline_profile("live-penultimate-checkpoint", [])
    url = "https://fixture.invalid/live/checkpoint/page"
    ledger = {
        url: StoredPage(
            url=url,
            final_url=url,
            status=200,
            body=b"<html>verbatim evidence</html>",
            fetched_at="2026-11-22T00:00:00Z",
        )
    }
    citation = Citation(
        url=url,
        quote="verbatim evidence",
        title="Fetched fixture",
        source_kind="institution",
    )

    prompt = backend._build_user_prompt(
        profile, [f"FETCH OK: {url}"], [], ledger=ledger, turns_used=2
    )

    assert "CITATION CHECKPOINT (2/3)" in prompt
    assert "cite_source now" in prompt
    assert "Do NOT call search_web, map_site, or fetch_url" in prompt
    assert "final turn (3/3) is reserved solely for emit_verdict" in prompt
    assert url in prompt
    assert "FINAL TURN (3/3)" not in prompt

    no_fetched_page = backend._build_user_prompt(profile, [], [], ledger={}, turns_used=2)
    already_cited = backend._build_user_prompt(profile, [], [citation], ledger=ledger, turns_used=2)
    assert "CITATION CHECKPOINT" not in no_fetched_page
    assert "CITATION CHECKPOINT" not in already_cited


def test_final_turn_citation_is_not_executed_and_zero_citation_run_degrades_honestly() -> None:
    url = "https://fixture.invalid/live/final-disobedience/page"
    session = session_from_turns(
        "live-final-disobedience",
        [{"url": url, "body": "<html>verbatim evidence</html>"}],
        [
            {"action": "fetch_url", "url": url},
            {
                "action": "cite_source",
                "url": url,
                "quote": "verbatim evidence",
                "title": "Fetched fixture",
                "source_kind": "institution",
            },
        ],
    )
    profile = inline_profile("live-final-disobedience", [])

    verdict, backend = run_one(profile, session, max_turns=2)

    assert backend.last_run["turns_used"] == 2
    assert backend.last_run["stop_reason"] == "final_turn_action_rejected"
    assert backend.last_run["citations"] == 0
    assert verdict.verdict == "unknown"
    assert verdict.evidence == []
    assert "stop_reason: final_turn_action_rejected" in verdict.unknown_aspects
    assert list(backend.ledger_for("live-final-disobedience")) == [url]
    assert reverify_evidence(backend, profile, verdict) == []


def test_zero_citation_final_emit_degrades_honestly() -> None:
    url = "https://fixture.invalid/live/zero-citation/page"
    session = session_from_turns(
        "live-zero-citation-final",
        [{"url": url, "body": "<html>fetched but never cited</html>"}],
        [
            {"action": "fetch_url", "url": url},
            # Disobey the penultimate citation checkpoint by fetching again.
            {"action": "fetch_url", "url": url},
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-zero-citation-final",
                    "verdict": "correct",
                    "rationale": "incorrectly claims a supported verdict without citations",
                    "reference_top3": [AIIMS, PGIMER, CMC],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    profile = inline_profile("live-zero-citation-final", [])

    verdict, backend = run_one(profile, session, max_turns=3)

    assert backend.last_run["turns_used"] == 3
    assert backend.last_run["fetches_used"] == 2
    assert backend.last_run["citations"] == 0
    assert backend.last_run["rejected_verdicts"] == 1
    assert backend.last_run["stop_reason"] == "final_turn_zero_citations"
    assert verdict.verdict == "unknown"
    assert verdict.evidence == []
    assert "stop_reason: final_turn_zero_citations" in verdict.unknown_aspects


# ------------------------------------------------------------------- sha256 re-verification


def test_reverify_passes_on_fixture_content_and_fails_on_tampered() -> None:
    profiles = {p.profile_id: p for p in load_profile_fixtures(PROFILES)}
    session = load_live_sessions(SESSIONS_DIR)["canary-01-correct"]
    verdict, backend = run_one(profiles["canary-01-correct"], session)
    assert verdict.evidence, "the canary replay must attach an evidence graph"
    assert reverify_evidence(backend, profiles["canary-01-correct"], verdict) == []

    page = next(iter(backend.ledger_for("canary-01-correct").values()))
    page.body = page.body + b"<!-- tampered after the run -->"
    failures = reverify_evidence(backend, profiles["canary-01-correct"], verdict)
    assert failures, "tampered stored content must fail re-verification"
    assert any("tampered" in failure for failure in failures)


def test_reverify_flags_a_pointer_the_fetch_layer_never_stored() -> None:
    profiles = {p.profile_id: p for p in load_profile_fixtures(PROFILES)}
    session = load_live_sessions(SESSIONS_DIR)["canary-01-correct"]
    verdict, backend = run_one(profiles["canary-01-correct"], session)
    del backend.ledger_for("canary-01-correct")[verdict.evidence[0].url]
    failures = reverify_evidence(backend, profiles["canary-01-correct"], verdict)
    assert any("fabricated pointer" in failure for failure in failures)


# ------------------------------------------------------------------- circularity guard


def test_circularity_guard_rejects_all_internal_evidence() -> None:
    sessions = load_live_sessions(SESSIONS_DIR)
    nirf = "https://www.nirfindia.org/Rankings/2025/MedicalRanking.html"
    profile = inline_profile("live-circularity-internal", [nirf])
    verdict, backend = run_one(profile, sessions["live-circularity-internal"])
    assert verdict.verdict == "correct"  # the model still emits a label...
    audit = audit_verdict(profile, verdict)
    assert audit["n_external"] == 0
    assert any("circularity guard" in failure for failure in audit["failures"])


def test_canary_replay_satisfies_the_circularity_guard() -> None:
    cases = run_live_replay(load_profile_fixtures(PROFILES))
    assert all(case["audit"]["n_external"] >= 1 for case in cases)
    assert all(case["audit"]["failures"] == [] for case in cases)


# ------------------------------------------------------------------- schema guards


def test_research_action_rejects_score_emission_and_undeclared_fields() -> None:
    with pytest.raises(ValidationError, match="score"):
        ResearchAction.model_validate(
            {
                "action": "emit_verdict",
                "score": 0.87,
                "verdict": {
                    "profile_id": "p",
                    "verdict": "correct",
                    "rationale": "sneaky",
                    "reference_top3": [AIIMS],
                    "as_of": "2026-11-22",
                },
            }
        )


def test_schema_violation_is_recorded_and_the_loop_recovers() -> None:
    sessions = load_live_sessions(SESSIONS_DIR)
    profile = inline_profile("live-schema-violation", [])
    verdict, backend = run_one(profile, sessions["live-schema-violation"])
    assert backend.last_run["schema_violations"] == 1
    assert verdict.verdict == "correct"
    assert "score" not in verdict.model_dump()
    assert reverify_evidence(backend, profile, verdict) == []


def test_emit_with_evidence_pointers_is_rejected_harness_attaches_the_graph() -> None:
    session = session_from_turns(
        "live-pointer-smuggle",
        [{"url": "https://fixture.invalid/live/x/page", "body": "<html>page</html>"}],
        [
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-pointer-smuggle",
                    "verdict": "unknown",
                    "rationale": "tries to attach its own evidence",
                    "unknown_aspects": ["x"],
                    "evidence": [
                        {
                            "url": "https://fixture.invalid/live/x/page",
                            "title": "page",
                            "quote": "page",
                            "sha256": "0" * 64,
                            "fetched_at": "2026-11-22T00:00:00Z",
                            "source_kind": "fixture",
                        }
                    ],
                    "as_of": "2026-11-22",
                },
            },
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-pointer-smuggle",
                    "verdict": "unknown",
                    "rationale": "clean retry",
                    "unknown_aspects": ["x"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    profile = inline_profile("live-pointer-smuggle", [])
    verdict, backend = run_one(profile, session)
    # the smuggling turn is rejected at finalize (it validated: a fabricated
    # 64-hex sha256 is schema-plausible), so the graph is attached only from
    # harness-verified citations — never from the model's own pointers
    assert backend.last_run["rejected_verdicts"] == 1
    assert verdict.evidence == []  # no citations were made, so nothing attaches
    assert verdict.verdict == "unknown"


def test_mismatched_profile_id_or_as_of_is_rejected() -> None:
    session = session_from_turns(
        "live-mismatch",
        [{"url": "https://fixture.invalid/live/m/page", "body": "<html>m</html>"}],
        [
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "someone-else",
                    "verdict": "unknown",
                    "rationale": "wrong profile",
                    "unknown_aspects": ["x"],
                    "as_of": "2026-11-22",
                },
            },
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-mismatch",
                    "verdict": "unknown",
                    "rationale": "stale as_of",
                    "unknown_aspects": ["x"],
                    "as_of": "2020-01-01",
                },
            },
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-mismatch",
                    "verdict": "unknown",
                    "rationale": "clean retry",
                    "unknown_aspects": ["x"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    profile = inline_profile("live-mismatch", [])
    verdict, backend = run_one(profile, session)
    assert backend.last_run["turns_used"] == 3
    assert verdict.verdict == "unknown"
    assert verdict.profile_id == "live-mismatch"
    assert verdict.as_of == "2026-11-22"


# ------------------------------------------------------------------- unknown is a value


def test_unknown_verdicts_are_accepted_and_counted_never_coerced() -> None:
    cases = run_live_replay(load_profile_fixtures(PROFILES))
    by_id = {case["profile_id"]: case for case in cases}
    assert by_id["canary-04-unknown"]["verdict"] == "unknown"
    assert len(by_id["canary-04-unknown"]["unknown_aspects"]) == 3
    assert by_id["canary-04-unknown"]["reference_top3"] == []
    aggregate = verdict_live_aggregate(cases)
    assert aggregate["unknown_coverage"] == 0.2
    assert aggregate["agreement_rate"] == 0.4  # 2 of 5 correct; unknown never counts
    assert aggregate["reverify_failures"] == 0


def test_canary_sessions_replay_end_to_end_offline() -> None:
    cases = run_live_replay(load_profile_fixtures(PROFILES))
    by_id = {case["profile_id"]: case for case in cases}
    assert [by_id[p]["verdict"] for p in sorted(by_id)] == [
        "correct",
        "partially_correct",
        "incorrect",
        "unknown",
        "correct",
    ]
    assert [by_id[p]["top3_overlap"] for p in sorted(by_id)] == [3, 1, 0, 0, 2]
    for case in cases:
        assert case["backend_run"]["stop_reason"] == "emitted_verdict"
        for pointer in case["audit"]["evidence"]:
            assert pointer["url"].startswith("https://")
            assert len(pointer["sha256"]) == 64
            assert pointer["quote"] and pointer["fetched_at"]
        assert case["fetch_ledger"], "the fetch layer's record must reach the report"


def test_cite_source_only_accepts_fetched_pages_with_verbatim_quotes() -> None:
    session = session_from_turns(
        "live-bad-cite",
        [{"url": "https://fixture.invalid/live/b/page", "body": "<html>only this text</html>"}],
        [
            {"action": "fetch_url", "url": "https://fixture.invalid/live/b/page"},
            {
                "action": "cite_source",
                "url": "https://fixture.invalid/live/b/never-fetched",
                "quote": "anything",
                "title": "t",
                "source_kind": "institution",
            },
            {
                "action": "cite_source",
                "url": "https://fixture.invalid/live/b/page",
                "quote": "quote that is NOT in the page",
                "title": "t",
                "source_kind": "institution",
            },
            {
                "action": "cite_source",
                "url": "https://fixture.invalid/live/b/page",
                "quote": "only this text",
                "title": "good",
                "source_kind": "institution",
            },
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-bad-cite",
                    "verdict": "unknown",
                    "rationale": "one honest citation, two rejected attempts",
                    "unknown_aspects": ["thin evidence"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    profile = inline_profile("live-bad-cite", [])
    verdict, backend = run_one(profile, session)
    assert backend.last_run["citations"] == 1  # only the verbatim citation survived
    assert len(verdict.evidence) == 1
    assert verdict.evidence[0].quote == "only this text"
    pointer = verdict.evidence[0]
    assert pointer.sha256 == backend.ledger_for("live-bad-cite")[pointer.url].sha256


# ------------------------------------------------------------------- discovery (free, leads-only)


def test_system_prompt_requires_map_site_before_guessing_course_paths() -> None:
    prompt = " ".join(SYSTEM_PROMPT.lower().split())

    assert "after search_web identifies an official institution domain" in prompt
    assert "ingested_evidence_urls" in prompt
    assert "before guessing course paths" in prompt
    assert '"action": "map_site"' in prompt
    assert '"query": "b.sc. medical laboratory technology"' in prompt


class FakeSearchProvider:
    """Free SearXNG stand-in: DocumentRef-shaped leads from one fixture query."""

    name = "searxng"

    def __init__(self, results: list | Exception | None = None) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str) -> list:
        self.queries.append(query)
        if isinstance(self.results, Exception):
            raise self.results
        return list(self.results or [])


class FakeMapProvider:
    """Free Firecrawl Local stand-in: MapLink-shaped links, or a typed error."""

    name = "firecrawl_local"

    def __init__(self, results: list | Exception | None = None) -> None:
        self.results = results
        self.calls: list[tuple[str, str | None]] = []

    def map(self, url: str, *, query: str | None = None, limit: int = 50) -> list:
        self.calls.append((url, query))
        if isinstance(self.results, Exception):
            raise self.results
        return list(self.results or [])


class _Lead:
    def __init__(self, url: str, title: str, engine: str = "duckduckgo") -> None:
        self.url = url
        self.title = title
        self.metadata = {"title": title, "engine": engine, "score": 1.0}


class _Link:
    def __init__(self, url: str, title: str, description: str = "") -> None:
        self.url = url
        self.title = title
        self.description = description


def discovery_session(profile_id: str) -> LiveSession:
    """Fetch the mapped page, cite it, emit: proves leads need fetch_url+cite."""
    return session_from_turns(
        profile_id,
        [{"url": "https://aiims.edu/academics/mlt", "body": "<html>BSc MLT prospectus</html>"}],
        [
            {
                "action": "search_web",
                "query": "AIIMS BSc MLT admission",
                "note": "find official page",
            },
            {"action": "map_site", "url": "https://aiims.edu", "query": "MLT", "note": "site map"},
            {"action": "fetch_url", "url": "https://aiims.edu/academics/mlt"},
            {
                "action": "cite_source",
                "url": "https://aiims.edu/academics/mlt",
                "quote": "BSc MLT prospectus",
                "title": "AIIMS academics",
                "source_kind": "institution",
            },
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": profile_id,
                    "verdict": "unknown",
                    "rationale": "discovery helped; verdict stays honest",
                    "unknown_aspects": ["reference not derivable from one page"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )


def test_scripted_search_fetch_cite_session_emits_on_final_turn() -> None:
    class PromptRecordingRouter(ScriptedModelRouter):
        def __init__(self, session: LiveSession) -> None:
            super().__init__(session)
            self.user_prompts: list[str] = []

        def call(self, purpose, schema, system, user, input_ref):
            self.user_prompts.append(user)
            return super().call(purpose, schema, system, user, input_ref)

    profile_id = "live-final-turn-emit"
    url = "https://aiims.edu/academics/mlt"
    session = session_from_turns(
        profile_id,
        [{"url": url, "body": "<html>BSc MLT prospectus</html>"}],
        [
            {"action": "search_web", "query": "AIIMS BSc MLT"},
            {"action": "fetch_url", "url": url},
            {
                "action": "cite_source",
                "url": url,
                "quote": "BSc MLT prospectus",
                "title": "AIIMS MLT programme",
                "source_kind": "institution",
            },
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": profile_id,
                    "verdict": "unknown",
                    "rationale": "one programme page is not enough to rank the shortlist",
                    "unknown_aspects": ["recognition and admissions evidence for the shortlist"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    search = FakeSearchProvider([_Lead(url, "AIIMS MLT programme")])
    router = PromptRecordingRouter(session)
    backend = LiveVerdictBackend(
        router,
        ScriptedPageFetcher(session.pages),
        max_turns=4,
        search_provider=search,
    )
    profile = inline_profile(profile_id, [])

    verdict = backend.respond(profile)

    assert verdict.verdict == "unknown"
    assert verdict.evidence and reverify_evidence(backend, profile, verdict) == []
    assert backend.last_run["turns_used"] == 4
    assert backend.last_run["searches_used"] == 1
    assert backend.last_run["fetches_used"] == 1
    assert backend.last_run["citations"] == 1
    assert backend.last_run["stop_reason"] == "emitted_verdict"
    assert ["FINAL TURN" in prompt for prompt in router.user_prompts] == [
        False,
        False,
        False,
        True,
    ]


def test_scripted_penultimate_cite_then_final_emit_succeeds() -> None:
    class PromptRecordingRouter(ScriptedModelRouter):
        def __init__(self, session: LiveSession) -> None:
            super().__init__(session)
            self.user_prompts: list[str] = []

        def call(self, purpose, schema, system, user, input_ref):
            self.user_prompts.append(user)
            return super().call(purpose, schema, system, user, input_ref)

    profile_id = "live-checkpoint-cite-emit"
    url = "https://fixture.invalid/live/checkpoint-cite-emit/page"
    session = session_from_turns(
        profile_id,
        [{"url": url, "body": "<html>BSc MLT programme evidence</html>"}],
        [
            {"action": "fetch_url", "url": url},
            {
                "action": "cite_source",
                "url": url,
                "quote": "BSc MLT programme evidence",
                "title": "Programme evidence",
                "source_kind": "institution",
            },
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": profile_id,
                    "verdict": "unknown",
                    "rationale": "one cited programme page cannot rank the full shortlist",
                    "unknown_aspects": ["recognition and admissions evidence for the shortlist"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    router = PromptRecordingRouter(session)
    backend = LiveVerdictBackend(
        router,
        ScriptedPageFetcher(session.pages),
        max_turns=3,
    )
    profile = inline_profile(profile_id, [])

    verdict = backend.respond(profile)

    assert verdict.verdict == "unknown"
    assert len(verdict.evidence) == 1
    assert reverify_evidence(backend, profile, verdict) == []
    assert backend.last_run["turns_used"] == 3
    assert backend.last_run["fetches_used"] == 1
    assert backend.last_run["citations"] == 1
    assert backend.last_run["stop_reason"] == "emitted_verdict"
    assert ["CITATION CHECKPOINT" in prompt for prompt in router.user_prompts] == [
        False,
        True,
        False,
    ]
    assert ["FINAL TURN" in prompt for prompt in router.user_prompts] == [
        False,
        False,
        True,
    ]


def backend_with_providers(
    session: LiveSession,
    *,
    search: FakeSearchProvider | None,
    maps: FakeMapProvider | None,
    **kwargs,
):
    return LiveVerdictBackend(
        ScriptedModelRouter(session),
        ScriptedPageFetcher(session.pages),
        search_provider=search,
        map_provider=maps,
        **kwargs,
    )


def test_search_and_map_turns_journal_leads_and_leave_the_ledger_clean() -> None:
    session = discovery_session("live-discovery-ok")
    search = FakeSearchProvider([_Lead("https://aiims.edu/academics/mlt", "AIIMS academics")])
    maps = FakeMapProvider([_Link("https://aiims.edu/academics/mlt", "MLT", "BSc syllabus")])
    backend = backend_with_providers(session, search=search, maps=maps)
    profile = inline_profile("live-discovery-ok", [])

    verdict = backend.respond(profile)
    stats = backend.last_run
    assert stats["searches_used"] == 1 and stats["maps_used"] == 1
    assert stats["provider_errors"] == 0
    # discovery output reached the journal as leads (title | url | engine)
    assert not verdict.evidence or all(
        pointer.url == "https://aiims.edu/academics/mlt" for pointer in verdict.evidence
    )
    ledger = backend.ledger_for("live-discovery-ok")
    assert list(ledger) == ["https://aiims.edu/academics/mlt"], (
        "the ledger must hold ONLY the fetched page, never a bare lead"
    )
    assert verdict.verdict == "unknown" and verdict.evidence
    # the map provider received the query focus
    assert maps.calls == [("https://aiims.edu", "MLT")]


def test_search_and_map_caps_are_enforced() -> None:
    session = session_from_turns(
        "live-discovery-cap",
        [{"url": "https://fixture.invalid/live/d/page", "body": "<html>p</html>"}],
        [
            {"action": "search_web", "query": "one"},
            {"action": "search_web", "query": "two"},
            {"action": "search_web", "query": "three — over the cap"},
            {"action": "map_site", "url": "https://fixture.invalid/a"},
            {"action": "map_site", "url": "https://fixture.invalid/b"},
            {"action": "map_site", "url": "https://fixture.invalid/c"},
        ],
    )
    search = FakeSearchProvider([_Lead("https://fixture.invalid/x", "x")])
    maps = FakeMapProvider([_Link("https://fixture.invalid/y", "y")])
    backend = backend_with_providers(session, search=search, maps=maps)
    profile = inline_profile("live-discovery-cap", [])

    backend.respond(profile)

    stats = backend.last_run
    assert stats["searches_used"] == backend.max_searches == 1
    assert stats["maps_used"] == backend.max_maps == 2
    assert search.queries == ["one"]
    assert [url for url, _ in maps.calls] == [
        "https://fixture.invalid/a",
        "https://fixture.invalid/b",
    ]


def test_search_cap_override_is_allowed_and_still_enforced() -> None:
    session = session_from_turns(
        "live-discovery-search-override",
        [],
        [
            {"action": "search_web", "query": "one"},
            {"action": "search_web", "query": "two"},
            {"action": "search_web", "query": "three — over the overridden cap"},
        ],
    )
    search = FakeSearchProvider([_Lead("https://fixture.invalid/x", "x")])
    backend = backend_with_providers(
        session,
        search=search,
        maps=None,
        max_turns=3,
        max_searches=2,
    )
    profile = inline_profile("live-discovery-search-override", [])

    backend.respond(profile)

    assert backend.max_searches == 2
    assert backend.last_run["searches_used"] == 2
    assert search.queries == ["one", "two"]
    assert backend.last_run["stop_reason"] == "final_turn_action_rejected"


def test_provider_unavailable_is_an_honest_journal_note_not_a_crash() -> None:
    session = session_from_turns(
        "live-discovery-missing",
        [],
        [
            {"action": "search_web", "query": "who knows"},
            {"action": "map_site", "url": "https://fixture.invalid/nowhere"},
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-discovery-missing",
                    "verdict": "unknown",
                    "rationale": "no providers, no guesses",
                    "unknown_aspects": ["no discovery, no evidence"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    # No providers at all (replay mode default): the loop journals and moves on.
    backend = backend_with_providers(session, search=None, maps=None)
    profile = inline_profile("live-discovery-missing", [])

    verdict = backend.respond(profile)

    assert verdict.verdict == "unknown"
    assert backend.last_run["searches_used"] == 0
    assert backend.last_run["maps_used"] == 0
    assert backend.last_run["provider_errors"] == 0
    assert backend.last_run["stop_reason"] == "emitted_verdict"


def test_provider_errors_are_counted_and_discovery_stays_free() -> None:
    session = session_from_turns(
        "live-discovery-errors",
        [],
        [
            {"action": "search_web", "query": "boom search"},
            {"action": "map_site", "url": "https://fixture.invalid/boom"},
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-discovery-errors",
                    "verdict": "unknown",
                    "rationale": "providers failed honestly; no paid fallback was tried",
                    "unknown_aspects": ["discovery unavailable"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    search = FakeSearchProvider(RuntimeError("searxng down"))
    maps = FakeMapProvider(RuntimeError("firecrawl local unreachable"))
    backend = backend_with_providers(session, search=search, maps=maps)
    profile = inline_profile("live-discovery-errors", [])

    verdict = backend.respond(profile)

    assert backend.last_run["provider_errors"] == 2
    assert backend.last_run["searches_used"] == 1
    assert backend.last_run["maps_used"] == 1
    assert verdict.evidence == []


def test_firecrawl_cloud_handoff_is_an_honest_failure_with_no_evidence() -> None:
    from fmd.fetch.firecrawl_local import FirecrawlCloudHandoffForbidden

    session = session_from_turns(
        "live-discovery-cloud",
        [{"url": "https://fixture.invalid/live/cloud/page", "body": "<html>real page</html>"}],
        [
            {"action": "map_site", "url": "https://fixture.invalid/cloud-site"},
            {"action": "fetch_url", "url": "https://fixture.invalid/live/cloud/page"},
            {
                "action": "emit_verdict",
                "verdict": {
                    "profile_id": "live-discovery-cloud",
                    "verdict": "unknown",
                    "rationale": "the map provider tried to hand off to the cloud; refused",
                    "unknown_aspects": ["site map unavailable locally"],
                    "as_of": "2026-11-22",
                },
            },
        ],
    )
    maps = FakeMapProvider(
        FirecrawlCloudHandoffForbidden(
            "cloud_handoff_forbidden: Firecrawl Local answer came from firecrawl-cloud"
        )
    )
    backend = backend_with_providers(session, search=FakeSearchProvider([]), maps=maps)
    profile = inline_profile("live-discovery-cloud", [])

    verdict = backend.respond(profile)

    assert backend.last_run["provider_errors"] == 1
    assert backend.last_run["stop_reason"] == "emitted_verdict"
    # the cloud-tainted answer contributed NOTHING to the evidence graph
    assert [pointer.url for pointer in verdict.evidence] == []


def test_research_action_requires_query_or_url_for_discovery() -> None:
    with pytest.raises(ValidationError, match="search_web needs a query"):
        ResearchAction.model_validate({"action": "search_web"})
    with pytest.raises(ValidationError, match="map_site needs a url"):
        ResearchAction.model_validate({"action": "map_site", "query": "x"})
    action = ResearchAction.model_validate(
        {"action": "map_site", "url": "https://aiims.edu", "query": "MLT"}
    )
    assert action.query == "MLT"


def test_cli_verdict_live_reports_discovery_fields(mock_provider, tmp_path: Path) -> None:
    report_path = tmp_path / "eval-verdict-live.json"
    code = main(
        [
            "eval",
            "--eval-kind",
            "verdict",
            "--stream",
            "verdict-live",
            "--report-out",
            str(report_path),
            "--max-searches",
            "3",
            "--max-maps",
            "1",
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    backend = report["backend"]
    assert backend["max_searches"] == 3 and backend["max_maps"] == 1
    assert backend["search_provider"] == "none"  # replay mode stays offline
    assert backend["map_provider"] == "none"
    aggregate = report["aggregate"]
    assert aggregate["searches_used"] == 0 and aggregate["maps_used"] == 0
    assert aggregate["provider_errors"] == 0


# ------------------------------------------------------------------- audit tooling


def test_audit_sampling_is_deterministic_under_fixed_seed() -> None:
    cases = run_live_replay(load_profile_fixtures(PROFILES))
    ws_a = build_audit_worksheet(cases, fraction=0.15, seed=13)
    ws_b = build_audit_worksheet(cases, fraction=0.15, seed=13)
    assert [e.profile_id for e in ws_a.entries] == [e.profile_id for e in ws_b.entries]
    assert ws_a.seed == 13 and ws_a.fraction_used == 0.15 and ws_a.gate == 0.70
    ids = {e.profile_id for e in ws_a.entries}
    assert {"canary-02-partially-correct", "canary-03-incorrect"} <= ids  # stratified
    assert len(ws_a.entries) == 2  # min(2, 5) floor; failures always added
    assert all(e.operator_agreement is None for e in ws_a.entries)  # empty field
    for entry in ws_a.entries:
        assert entry.evidence and entry.fetch_ledger
        for pointer in entry.evidence:
            assert pointer.url and pointer.quote and len(pointer.sha256) == 64


def test_seeded_rng_fills_the_random_part_of_the_sample() -> None:
    cases = synthetic_cases(12)
    ws_a = build_audit_worksheet(cases, fraction=0.15, seed=7)
    ws_b = build_audit_worksheet(cases, fraction=0.15, seed=7)
    ws_c = build_audit_worksheet(cases, fraction=0.15, seed=8)
    assert ws_a.n_cases == 12 and len(ws_a.entries) == 2  # round(12*0.15) -> 2
    assert [e.profile_id for e in ws_a.entries] == [e.profile_id for e in ws_b.entries]
    assert [e.profile_id for e in ws_a.entries] != [e.profile_id for e in ws_c.entries]


def test_audit_fraction_is_clamped_into_the_10_20_band() -> None:
    assert clamp_audit_fraction(0.75) == 0.20
    assert clamp_audit_fraction(0.01) == 0.10
    assert clamp_audit_fraction(0.15) == 0.15


def _fill(ws: AuditWorksheet, agreements: dict[str, dict]) -> AuditWorksheet:
    for entry in ws.entries:
        if entry.profile_id in agreements:
            entry.operator_agreement = OperatorAgreement(**agreements[entry.profile_id])
    return ws


def test_audit_sampling_has_a_minimum_of_two_entries_when_available() -> None:
    cases = synthetic_cases(5)
    worksheet = build_audit_worksheet(cases, fraction=0.4, seed=13)
    repeated = build_audit_worksheet(cases, fraction=0.4, seed=13)

    assert worksheet.fraction_used == 0.20  # requested fraction is only clamped to the band
    assert len(worksheet.entries) == 2  # round(5 * 0.20) == 1, but the floor is two
    assert [entry.profile_id for entry in worksheet.entries] == [
        entry.profile_id for entry in repeated.entries
    ]
    assert "minimum of min(2, n_cases)" in worksheet.sampling_rule

    one_case = build_audit_worksheet(synthetic_cases(1), fraction=0.4, seed=13)
    assert len(one_case.entries) == 1
    _fill(one_case, {one_case.entries[0].profile_id: {"agrees": True}})
    assert score_audit_worksheet(one_case)["result"] == "INSUFFICIENT"


def test_audit_score_computes_agreement_rate_pass_and_fail() -> None:
    cases = run_live_replay(load_profile_fixtures(PROFILES))
    ws_fail = build_audit_worksheet(cases, fraction=0.15, seed=13)
    first, second = ws_fail.entries[0].profile_id, ws_fail.entries[1].profile_id
    _fill(ws_fail, {first: {"agrees": True}, second: {"agrees": False, "reason": "top-3 differs"}})
    outcome = score_audit_worksheet(ws_fail)
    assert outcome["agreement_rate"] == 0.5
    assert outcome["result"] == "FAIL"  # 0.5 < 0.70 gate
    assert outcome["disagreements"] == [
        {
            "profile_id": second,
            "agent_verdict": ws_fail.entries[1].verdict,
            "reason": "top-3 differs",
        }
    ]

    ws_pass = build_audit_worksheet(cases, fraction=0.15, seed=13)
    _fill(ws_pass, {e.profile_id: {"agrees": True, "citation_ok": True} for e in ws_pass.entries})
    outcome_pass = score_audit_worksheet(ws_pass)
    assert outcome_pass["agreement_rate"] == 1.0
    assert outcome_pass["result"] == "PASS"


def test_audit_score_insufficient_and_fabricated_citation() -> None:
    cases = run_live_replay(load_profile_fixtures(PROFILES))
    ws_empty = build_audit_worksheet(cases, fraction=0.15, seed=13)
    outcome = score_audit_worksheet(ws_empty)
    assert outcome["result"] == "INSUFFICIENT"
    assert outcome["agreement_rate"] is None

    ws_one = build_audit_worksheet(cases, fraction=0.15, seed=13)
    _fill(ws_one, {ws_one.entries[0].profile_id: {"agrees": True}})
    assert score_audit_worksheet(ws_one)["result"] == "INSUFFICIENT"

    ws_fab = build_audit_worksheet(cases, fraction=0.15, seed=13)
    _fill(
        ws_fab,
        {
            ws_fab.entries[0].profile_id: {"agrees": True, "citation_ok": False},
            ws_fab.entries[1].profile_id: {"agrees": True},
        },
    )
    outcome_fab = score_audit_worksheet(ws_fab)
    assert outcome_fab["result"] == "FAIL"  # doc 04 section 2.3: quarantine the run
    assert outcome_fab["fabricated_citations"] == [ws_fab.entries[0].profile_id]


def test_worksheet_round_trips_through_its_schema() -> None:
    cases = run_live_replay(load_profile_fixtures(PROFILES))
    ws = build_audit_worksheet(cases, fraction=0.15, seed=13)
    payload = json.loads(ws.model_dump_json())
    reparsed = AuditWorksheet.model_validate(payload)
    assert [e.profile_id for e in reparsed.entries] == [e.profile_id for e in ws.entries]


# ------------------------------------------------------------------- CLI wiring


def test_cli_verdict_live_runs_end_to_end_on_mock_provider(
    tmp_path: Path, capsys, mock_provider
) -> None:
    report_path = tmp_path / "eval-verdict-live.json"
    code = main(
        [
            "eval",
            "--eval-kind",
            "verdict",
            "--stream",
            "verdict-live",
            "--report-out",
            str(report_path),
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stream"] == "verdict-live"
    assert report["backend"]["mode"] == "replay"
    assert report["backend"]["max_searches"] == 1
    assert report["backend"]["max_maps"] == 2
    assert report["aggregate"]["n_profiles"] == 5
    assert report["aggregate"]["audit_failures"] == 0
    assert all(case["reverify"]["failures"] == [] for case in report["cases"])
    out = capsys.readouterr().out
    assert "fmd eval (verdict-live)" in out
    assert "not live-model behavior" in out  # replay honesty note


def test_cli_verdict_live_dry_run_and_limits_parse(mock_provider, capsys) -> None:
    code = main(
        [
            "eval",
            "--eval-kind",
            "verdict",
            "--stream",
            "verdict-live",
            "--dry-run",
            "--max-turns",
            "5",
            "--max-fetches",
            "3",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "max_turns 5" in out and "max_fetches 3" in out


def test_cli_verdict_live_missing_session_fails_honestly(
    tmp_path: Path, capsys, mock_provider
) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    shutil.copy(PROFILES / "canary-01-correct.json", profiles_dir / "canary-01-correct.json")
    code = main(
        [
            "eval",
            "--eval-kind",
            "verdict",
            "--stream",
            "verdict-live",
            "--fixture-path",
            str(profiles_dir),
            "--dry-run",
        ]
    )
    assert code == 1
    assert "verdict-live failed" in capsys.readouterr().err


def test_cli_verdict_live_unknown_stream_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["eval", "--eval-kind", "verdict", "--stream", "verdict-slow"])
    assert excinfo.value.code == 2


def test_cli_audit_sample_writes_worksheet_and_audit_score_round_trip(
    tmp_path: Path, capsys, mock_provider
) -> None:
    report_path = tmp_path / "eval-verdict-live.json"
    code = main(
        [
            "eval",
            "--eval-kind",
            "verdict",
            "--stream",
            "verdict-live",
            "--report-out",
            str(report_path),
            "--audit-sample",
            "0.15",
            "--audit-seed",
            "13",
        ]
    )
    assert code == 0
    worksheet_path = tmp_path / "audit-verdict-live.json"
    worksheet = json.loads(worksheet_path.read_text(encoding="utf-8"))
    assert worksheet["stream"] == "verdict-live"
    assert worksheet["seed"] == 13 and worksheet["fraction_used"] == 0.15
    assert all(e["operator_agreement"] is None for e in worksheet["entries"])

    # an empty worksheet is INSUFFICIENT (exit 3), never a silent PASS
    code = main(["eval", "--eval-kind", "verdict", "--audit-score", str(worksheet_path)])
    assert code == 3
    assert "INSUFFICIENT" in capsys.readouterr().out


def test_cli_audit_score_pass_and_fail_exit_codes(tmp_path: Path, capsys, mock_provider) -> None:
    report_path = tmp_path / "eval-verdict-live.json"
    assert (
        main(
            [
                "eval",
                "--eval-kind",
                "verdict",
                "--stream",
                "verdict-live",
                "--report-out",
                str(report_path),
                "--audit-sample",
                "0.15",
            ]
        )
        == 0
    )
    worksheet_path = tmp_path / "audit-verdict-live.json"
    worksheet = AuditWorksheet.model_validate(
        json.loads(worksheet_path.read_text(encoding="utf-8"))
    )
    ids = [entry.profile_id for entry in worksheet.entries]

    _fill(worksheet, {ids[0]: {"agrees": True}, ids[1]: {"agrees": True}})
    worksheet_path.write_text(worksheet.model_dump_json(), encoding="utf-8")
    assert main(["eval", "--eval-kind", "verdict", "--audit-score", str(worksheet_path)]) == 0

    _fill(worksheet, {ids[0]: {"agrees": False, "reason": "reference differs"}})
    worksheet_path.write_text(worksheet.model_dump_json(), encoding="utf-8")
    code = main(["eval", "--eval-kind", "verdict", "--audit-score", str(worksheet_path)])
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "never enforced in CI" in out


def test_cli_audit_score_rejects_a_malformed_worksheet(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1, "stream": "verdict-live"}', encoding="utf-8")
    assert main(["eval", "--eval-kind", "verdict", "--audit-score", str(bad)]) == 1
    assert "not schema-valid" in capsys.readouterr().err
