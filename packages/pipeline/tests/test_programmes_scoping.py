"""W9 Tiers 2+3 — the semantic classifier, its cache and the learning loop.

Offline by contract: the model is a fake router (a counting double that
returns canned :class:`EquivalenceResponse` payloads) or the committed replay
provider; no network, no model download. The behaviours pinned here are the
conservative-by-construction ones from 14 §B2:

* MBBS/Nursing never reach the classifier at all (Tier 1 excludes them);
* a no-evidence row is ``uncertain`` without a single provider call;
* a semantic ``yes`` must cite evidence, or it is downgraded to ``uncertain``;
* an unavailable model degrades to ``uncertain`` → ``needs_review``;
* the cache makes the second identical name free (zero provider calls);
* a human decision (Tier 3) wins over rules and classifier on the next run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fmd.config import load_registry
from fmd.programmes.classifier import (
    EquivalenceCandidate,
    classify,
    decision_cache_key,
)
from fmd.programmes.service import (
    ProgrammeScoper,
    ScopeInput,
    build_scoper,
    evidence_from_claims,
    record_manual_decision,
)
from fmd.programmes.taxonomy import ScopeDecision, Taxonomy

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture(scope="module")
def registry():
    return load_registry(CONFIG_PATH, env={})


class FakeRouter:
    """Counts provider calls; answers from a per-key table or a default."""

    def __init__(self, answers: dict[str, dict] | None = None, default: dict | None = None):
        self.answers = dict(answers or {})
        self.default = default
        self.calls: list[dict] = []

    def call(self, *, purpose, schema, system, user, input_ref):
        self.calls.append({"purpose": purpose, "input_ref": input_ref, "user": user})
        decisions = []
        for key in input_ref.split(","):
            payload = self.answers.get(key, self.default)
            assert payload is not None, f"no canned answer for {key}"
            decisions.append({"key": key, **payload})
        return schema.model_validate({"decisions": decisions})


class FakeStore:
    """In-memory DecisionStore (the DB repo's duck twin)."""

    def __init__(self) -> None:
        self.rows: dict[str, SimpleNamespace] = {}

    def get(self, cache_key: str):
        return self.rows.get(cache_key)

    def put(self, payload: dict):
        record = SimpleNamespace(**payload)
        self.rows[payload["cache_key"]] = record
        return record


def _candidate(name: str, *, country: str | None = "AU", evidence: tuple[str, ...] = ()) -> (
    tuple
):
    config = load_registry(CONFIG_PATH, env={}).programmes
    normalised = " ".join(name.casefold().split())
    key = decision_cache_key(normalised, f"institution:{country or ''}", config.definition.version)
    candidate = EquivalenceCandidate(
        key=key,
        normalised_name=normalised,
        raw_name=name,
        institution="Test University",
        institution_type=f"institution:{country or ''}",
        country=country,
        evidence=evidence,
    )
    return candidate, key


# ------------------------------------------------------------------- Tier 2


def test_semantic_yes_admits_with_cited_evidence(registry) -> None:
    candidate, key = _candidate("Bachelor of Biomedical Science (Laboratory Medicine)")
    evidence = (
        'claim 12 [curriculum.internship_months] 6 — "6-month clinical rotation '
        'in haematology, clinical biochemistry and microbiology"',
    )
    candidate = EquivalenceCandidate(
        key=candidate.key,
        normalised_name=candidate.normalised_name,
        raw_name=candidate.raw_name,
        institution=candidate.institution,
        institution_type=candidate.institution_type,
        country=candidate.country,
        evidence=evidence,
    )
    router = FakeRouter(
        answers={
            candidate.key: {
                "equivalent": "yes",
                "confidence": 0.86,
                "rationale": "laboratory medicine with clinical rotations",
                "evidence": ["claim 12", "Laboratory Medicine"],
            }
        }
    )
    store = FakeStore()
    taxonomy = Taxonomy(registry.programmes)
    decisions, notes = classify(
        [candidate], taxonomy=taxonomy, config=registry.programmes, router=router, store=store
    )
    decision = decisions[key]
    assert decision.in_scope and not decision.needs_review
    assert decision.tier == "semantic"
    assert decision.reason == "semantic_equivalent"
    assert "claim 12" in decision.evidence
    assert decision.model is None  # the router records the model, not the caller
    # The decision is persisted for the next run.
    assert key in store.rows
    assert store.rows[key].decision == "in_scope"
    assert not router.calls[0]["purpose"].startswith("default")


def test_semantic_yes_without_evidence_is_downgraded_to_uncertain(registry) -> None:
    candidate, key = _candidate(
        "Bachelor of Applied Something Unfamiliar",
        evidence=('claim 3 [curriculum.medium] English — "two semesters"',),
    )
    router = FakeRouter(
        answers={
            key: {
                "equivalent": "yes",
                "confidence": 0.9,
                "rationale": "sounds right",
                "evidence": [],
            }
        }
    )
    decisions, notes = classify(
        [candidate],
        taxonomy=Taxonomy(registry.programmes),
        config=registry.programmes,
        router=router,
    )
    decision = decisions[key]
    assert not decision.in_scope
    assert decision.needs_review
    assert decision.reason == "semantic_uncertain"
    assert any("downgraded" in note for note in notes)


def test_semantic_no_excludes(registry) -> None:
    candidate, key = _candidate("Bachelor of Wellness Studies")
    evidence = ('claim 9 [curriculum.medium] English — "taught in English"',)
    candidate = candidate.__class__(
        key=candidate.key,
        normalised_name=candidate.normalised_name,
        raw_name=candidate.raw_name,
        institution=candidate.institution,
        institution_type=candidate.institution_type,
        country=candidate.country,
        evidence=evidence,
    )
    router = FakeRouter(
        answers={
            key: {
                "equivalent": "no",
                "confidence": 0.8,
                "rationale": "no laboratory practice component",
                "evidence": ["claim 9"],
            }
        }
    )
    decisions, _notes = classify(
        [candidate],
        taxonomy=Taxonomy(registry.programmes),
        config=registry.programmes,
        router=router,
    )
    decision = decisions[key]
    assert not decision.in_scope
    assert not decision.needs_review
    assert decision.reason == "semantic_not_equivalent"


def test_no_evidence_and_no_signal_is_uncertain_without_a_provider_call(registry) -> None:
    candidate, key = _candidate("Bachelor of Health Sciences")
    router = FakeRouter()
    decisions, notes = classify(
        [candidate],
        taxonomy=Taxonomy(registry.programmes),
        config=registry.programmes,
        router=router,
    )
    decision = decisions[key]
    assert not decision.in_scope and decision.needs_review
    assert decision.reason == "semantic_no_evidence_uncertain"
    assert router.calls == []  # nothing to cite, so nothing to ask
    assert notes == []


def test_unavailable_model_degrades_to_uncertain_needs_review(registry) -> None:
    candidate, key = _candidate("Bachelor of Laboratory Practice")
    decisions, notes = classify(
        [candidate],
        taxonomy=Taxonomy(registry.programmes),
        config=registry.programmes,
        router=None,
    )
    decision = decisions[key]
    assert not decision.in_scope and decision.needs_review
    assert decision.reason == "semantic_unavailable_uncertain"
    assert any("no model router" in note for note in notes)


def test_provider_error_degrades_to_uncertain_not_a_crash(registry) -> None:
    class ExplodingRouter:
        def call(self, **_kwargs):
            raise RuntimeError("endpoint exploded")

    candidate, key = _candidate("Bachelor of Laboratory Practice")
    decisions, notes = classify(
        [candidate],
        taxonomy=Taxonomy(registry.programmes),
        config=registry.programmes,
        router=ExplodingRouter(),
    )
    assert decisions[key].needs_review
    assert any("model unavailable" in note for note in notes)


def test_cached_semantic_decision_makes_zero_provider_calls(registry) -> None:
    candidate, key = _candidate("Bachelor of Biomedical Science (Laboratory Medicine)")
    evidence = ('claim 12 [curriculum.medium] English — "clinical laboratory"',)
    candidate = EquivalenceCandidate(
        key=candidate.key,
        normalised_name=candidate.normalised_name,
        raw_name=candidate.raw_name,
        institution=candidate.institution,
        institution_type=candidate.institution_type,
        country=candidate.country,
        evidence=evidence,
    )
    store = FakeStore()
    store.rows[key] = SimpleNamespace(
        cache_key=key,
        decision="in_scope",
        tier="semantic",
        confidence=0.9,
        rationale="cached yes",
        evidence=["claim 12"],
        model="gemma-test",
        prompt_version="v1",
        cached=True,
        needs_review=False,
        flags=[],
        decided_by="classifier",
    )
    router = FakeRouter()
    decisions, _notes = classify(
        [candidate],
        taxonomy=Taxonomy(registry.programmes),
        config=registry.programmes,
        router=router,
        store=store,
    )
    decision = decisions[key]
    assert decision.in_scope and decision.cached
    assert decision.model == "gemma-test"
    assert router.calls == []  # cache hit: zero provider calls
    assert decision.prompt_version == "v1"


def test_manual_decision_wins_over_classifier(registry) -> None:
    candidate, key = _candidate("Bachelor of Laboratory Practice")
    store = FakeStore()
    store.rows[key] = SimpleNamespace(
        cache_key=key,
        decision="in_scope",
        tier="manual",
        confidence=None,
        rationale="human checked the syllabus",
        evidence=["human note"],
        model=None,
        prompt_version=None,
        cached=False,
        needs_review=False,
        flags=[],
        decided_by="human",
    )
    router = FakeRouter(default={"equivalent": "no", "evidence": ["claim 1"]})
    decisions, _notes = classify(
        [candidate],
        taxonomy=Taxonomy(registry.programmes),
        config=registry.programmes,
        router=router,
        store=store,
    )
    decision = decisions[key]
    assert decision.tier == "manual"
    assert decision.in_scope and not decision.needs_review
    assert router.calls == []  # the classifier is not even asked


def test_one_call_per_new_name_not_per_row(registry) -> None:
    config = registry.programmes
    taxonomy = Taxonomy(config)
    router = FakeRouter(default={"equivalent": "no", "confidence": 0.7, "evidence": ["claim 1"]})
    normalised = "bachelor of laboratory practice"
    key = decision_cache_key(normalised, "institution:AU", config.definition.version)
    candidates = [
        EquivalenceCandidate(
            key=key,
            normalised_name=normalised,
            raw_name=f"Bachelor of Laboratory Practice ({i})",
            institution_type="institution:AU",
            country="AU",
            evidence=("claim 1 [curriculum.medium] English",),
        )
        for i in range(5)
    ]
    decisions, _notes = classify(
        candidates, taxonomy=taxonomy, config=config, router=router
    )
    # One shared decision for the one name; five rows did not mean five calls,
    # and the prompt carried the name once.
    assert len(decisions) == 1
    assert len(router.calls) == 1
    assert router.calls[0]["user"].count("key=") == 1


# ------------------------------------------------------- three-tier scoper


def _scoper(registry, *, store=None, router=None) -> ProgrammeScoper:
    return ProgrammeScoper(
        Taxonomy(registry.programmes), registry.programmes, store=store, router=router
    )


def test_scoper_rules_decide_mbbs_without_the_model(registry) -> None:
    scoper = _scoper(registry, router=FakeRouter())
    inp = ScopeInput(
        institution_id=1,
        raw_name="MBBS - Bachelor of Medicine and Bachelor of Surgery",
        normalised_name="mbbs",
        institution_kind="institution",
        country="IN",
    )
    [(got, decision)] = scoper.scope_all([inp])
    assert got is inp
    assert decision.tier == "rule"
    assert not decision.in_scope
    assert decision.reason.startswith("rule_exclude:")


def test_scoper_manual_decision_beats_rules(registry) -> None:
    store = FakeStore()
    config = registry.programmes
    key = decision_cache_key(
        "b sc medical laboratory technology", "institution:IN", config.definition.version
    )
    store.rows[key] = SimpleNamespace(
        cache_key=key,
        decision="in_scope",
        tier="manual",
        confidence=None,
        rationale="verified syllabus",
        evidence=["note"],
        model=None,
        prompt_version=None,
        cached=False,
        needs_review=False,
        flags=[],
        decided_by="human",
    )
    scoper = _scoper(registry, store=store, router=FakeRouter())
    inp = ScopeInput(
        institution_id=1,
        raw_name="B.Sc. Medical Laboratory Technology",
        normalised_name="b sc mlt",
        institution_kind="institution",
        country="IN",
    )
    [(got, decision)] = scoper.scope_all([inp])
    assert decision.tier == "manual"
    assert decision.in_scope


def test_scoper_semantic_fallback_for_stub_country(registry) -> None:
    router = FakeRouter(
        default={"equivalent": "yes", "confidence": 0.8, "evidence": ["claim 4", "laboratory"]}
    )
    scoper = _scoper(registry, router=router)
    inp = ScopeInput(
        institution_id=7,
        raw_name="Bachelor of Biomedical Science (Laboratory Medicine)",
        normalised_name="bachelor biomedical science laboratory medicine",
        institution_kind="institution",
        country="AU",
        claims=('claim 4 [curriculum.medium] English — "clinical laboratory practice"',),
    )
    [(got, decision)] = scoper.scope_all([inp])
    assert got is inp
    assert decision.tier == "semantic"
    assert decision.in_scope
    assert decision.evidence
    assert len(router.calls) == 1


def test_scoper_unknown_country_uncertain_with_flag(registry) -> None:
    scoper = _scoper(registry, router=FakeRouter())
    inp = ScopeInput(
        institution_id=9,
        raw_name="Bachelor of Cura Hyperbarica",
        normalised_name="bachelor cura hyperbarica",
        institution_kind="institution",
        country=None,
    )
    [(_got, decision)] = scoper.scope_all([inp])
    assert not decision.in_scope and decision.needs_review
    assert decision.reason == "semantic_no_evidence_uncertain"


def test_evidence_from_claims_renders_citable_lines() -> None:
    claims = [
        SimpleNamespace(id=47, predicate="seats.intake", value=20, quote="20 seats"),
        SimpleNamespace(
            id=52,
            predicate="curriculum.internship_months",
            value=6,
            quote="six months in clinical laboratory",
        ),
    ]
    evidence = evidence_from_claims(claims)
    assert len(evidence) == 1  # seats say nothing about the field
    assert evidence[0].startswith("claim 52 [curriculum.internship_months]")
    assert "six months in clinical laboratory" in evidence[0]


# ------------------------------------------------- replay provider (mock path)


def test_replay_provider_serves_the_classifier_prompt(tmp_path, monkeypatch) -> None:
    """End-to-end through the committed replay fixture plumbing (offline).

    The fixture name is the exact prompt hash the router computes; the test
    writes it into a throw-away fixtures dir and points a replay provider at
    it, so the whole ``ModelRouter`` path runs without a network.
    """
    from fmd.config.models import ProviderConfig
    from fmd.llm.providers import ReplayProvider, build_provider
    from fmd.llm.router import prompt_hash

    registry_config = load_registry(CONFIG_PATH, env={})
    taxonomy = Taxonomy(registry_config.programmes)
    candidate, key = _candidate("Bachelor of Biomedical Science (Laboratory Medicine)")
    evidence = ('claim 12 [curriculum.medium] English — "laboratory medicine"',)
    candidate = EquivalenceCandidate(
        key=candidate.key,
        normalised_name=candidate.normalised_name,
        raw_name=candidate.raw_name,
        institution=candidate.institution,
        institution_type=candidate.institution_type,
        country=candidate.country,
        evidence=evidence,
    )
    system, user = __import__(
        "fmd.programmes.classifier", fromlist=["build_prompt"]
    ).build_prompt(
        [
            SimpleNamespace(
                key=candidate.key,
                programme_name=candidate.raw_name,
                institution=candidate.institution,
                country=candidate.country,
                department=None,
                evidence=list(candidate.evidence),
            )
        ],
        taxonomy.definition_text(),
        registry_config.programmes.classifier.prompt_version,
    )
    sha = prompt_hash(system, user)
    fixtures = tmp_path / "programme_equivalence"
    fixtures.mkdir()
    (fixtures / f"{sha}.json").write_text(
        json.dumps(
            {
                "content": {
                    "decisions": [
                        {
                            "key": key,
                            "equivalent": "yes",
                            "confidence": 0.75,
                            "rationale": "laboratory medicine",
                            "evidence": ["claim 12"],
                        }
                    ]
                },
                "tokens_in": 10,
                "tokens_out": 5,
                "model": "replay-test",
            }
        ),
        encoding="utf-8",
    )
    provider = build_provider(
        "replay_test",
        ProviderConfig(kind="replay", fixtures_dir=str(tmp_path)),
    )
    assert isinstance(provider, ReplayProvider)
    router = SimpleNamespace(
        call=lambda **kwargs: provider.complete(
            model=kwargs["schema"].__name__ and "replay-test",
            system=kwargs["system"],
            user=kwargs["user"],
            schema=kwargs["schema"],
            profile=None,
            prompt_sha256=prompt_hash(kwargs["system"], kwargs["user"]),
            purpose=kwargs["purpose"],
        )
    )
    decisions, _notes = classify(
        [candidate],
        taxonomy=taxonomy,
        config=registry_config.programmes,
        router=router,
    )
    decision = decisions[key]
    assert decision.in_scope
    assert decision.evidence == ("claim 12",)
    assert isinstance(decision, ScopeDecision)


def test_build_prompt_is_deterministic() -> None:
    from fmd.programmes.classifier import EquivalenceItem, build_prompt

    registry_config = load_registry(CONFIG_PATH, env={})
    taxonomy = Taxonomy(registry_config.programmes)
    items = [
        EquivalenceItem(key="k2", programme_name="B", evidence=["claim 2"]),
        EquivalenceItem(key="k1", programme_name="A", evidence=[]),
    ]
    system1, user1 = build_prompt(items, taxonomy.definition_text(), "v1")
    system2, user2 = build_prompt(list(reversed(items)), taxonomy.definition_text(), "v1")
    assert (system1, user1) == (system2, user2)
    assert "claim 2" in user1
    assert "evidence: (none)" in user1


def test_record_manual_decision_persists_and_applies(registry) -> None:
    """Tier 3: a human decision is written as a manual decision (fake repo)."""
    config = registry.programmes

    class FakeRepo:
        def __init__(self, row):
            self.row = row
            self.decisions: dict[str, SimpleNamespace] = {}
            self.updates: list[dict] = []

        def get(self, _programme_id):
            return self.row

        def scope_decision_get(self, key):
            return self.decisions.get(key)

        def scope_decision_put(self, create):
            record = SimpleNamespace(**create.model_dump())
            self.decisions[create.cache_key] = record
            return record, True

        def update_scope(self, programme_id, **kwargs):
            self.updates.append({"programme_id": programme_id, **kwargs})
            for key, value in kwargs.items():
                setattr(self.row, key, value)
            return self.row

    from fmd.programmes.service import SCOPE_UNCERTAIN_REASON

    row = SimpleNamespace(
        id=5,
        institution_id=1,
        name="Bachelor of Laboratory Practice",
        review_reasons=[SCOPE_UNCERTAIN_REASON, "possible_duplicate_programme_name:X"],
        in_scope=False,
        scope_tier="semantic",
        scope_reason="semantic_uncertain",
    )
    repo = FakeRepo(row)
    entities = SimpleNamespace(
        get=lambda _id: SimpleNamespace(id=1, kind="institution", country="AU")
    )
    scoper = build_scoper(registry, None, store=None, router=None)
    outcome = record_manual_decision(
        repo,
        entities,
        scoper.taxonomy,
        config,
        5,
        equivalent=True,
        note="syllabus verified against AIMS guidelines",
    )
    assert outcome.equivalent and outcome.in_scope
    assert outcome.previous is None
    assert outcome.row_updated
    stored = next(iter(repo.decisions.values()))
    assert stored.tier == "manual"
    assert stored.decided_by == "human"
    assert stored.decision == "in_scope"
    assert stored.note == "syllabus verified against AIMS guidelines"
    # scope_uncertain cleared, other review reasons preserved.
    update = repo.updates[0]
    assert update["needs_review"] is True
    assert SCOPE_UNCERTAIN_REASON not in update["review_reasons"]
    assert "possible_duplicate_programme_name:X" in update["review_reasons"]
    assert update["scope_tier"] == "manual"
