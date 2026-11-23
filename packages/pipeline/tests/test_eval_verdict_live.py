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
    AuditWorksheet,
    LiveSession,
    LiveVerdictBackend,
    OperatorAgreement,
    ResearchAction,
    ScriptedModelRouter,
    ScriptedPageFetcher,
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
    assert backend.last_run["turns_used"] == 8  # hard model-turn cap
    assert backend.last_run["fetches_used"] == 6  # hard fetch cap (12 were requested)
    assert backend.last_run["stop_reason"] == "turn_cap_reached"
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


# ------------------------------------------------------------------- audit tooling


def test_audit_sampling_is_deterministic_under_fixed_seed() -> None:
    cases = run_live_replay(load_profile_fixtures(PROFILES))
    ws_a = build_audit_worksheet(cases, fraction=0.15, seed=13)
    ws_b = build_audit_worksheet(cases, fraction=0.15, seed=13)
    assert [e.profile_id for e in ws_a.entries] == [e.profile_id for e in ws_b.entries]
    assert ws_a.seed == 13 and ws_a.fraction_used == 0.15 and ws_a.gate == 0.70
    ids = {e.profile_id for e in ws_a.entries}
    assert {"canary-02-partially-correct", "canary-03-incorrect"} <= ids  # stratified
    assert len(ws_a.entries) == 2  # target max(1, round(5*0.15)); failures always added
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
            "eval", "--eval-kind", "verdict", "--stream", "verdict-live",
            "--report-out", str(report_path),
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stream"] == "verdict-live"
    assert report["backend"]["mode"] == "replay"
    assert report["aggregate"]["n_profiles"] == 5
    assert report["aggregate"]["audit_failures"] == 0
    assert all(case["reverify"]["failures"] == [] for case in report["cases"])
    out = capsys.readouterr().out
    assert "fmd eval (verdict-live)" in out
    assert "NEVER" in out  # the honesty note about the live stream


def test_cli_verdict_live_dry_run_and_limits_parse(mock_provider, capsys) -> None:
    code = main(
        [
            "eval", "--eval-kind", "verdict", "--stream", "verdict-live",
            "--dry-run", "--max-turns", "5", "--max-fetches", "3",
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
            "eval", "--eval-kind", "verdict", "--stream", "verdict-live",
            "--fixture-path", str(profiles_dir), "--dry-run",
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
            "eval", "--eval-kind", "verdict", "--stream", "verdict-live",
            "--report-out", str(report_path),
            "--audit-sample", "0.15", "--audit-seed", "13",
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
    assert main(
        [
            "eval", "--eval-kind", "verdict", "--stream", "verdict-live",
            "--report-out", str(report_path), "--audit-sample", "0.15",
        ]
    ) == 0
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
