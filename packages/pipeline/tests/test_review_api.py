"""Review console (WS-G) contract: queue, actions, audit, ER authority.

Offline tests cover the admin gate and degradation states. The action, audit
and ER-authority tests run against a live Postgres (``pg_database``, same
skip-loudly contract as ``test_db.py``): a review decision is a database
transition, and a fixture cannot prove that supersede-not-mutate happened.

No network, no model calls. The admin token is monkeypatched per test.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import fmd.api as api_module
from fmd.api import _database_or_503, app
from fmd.db import (
    AliasCreate,
    AliasesRepo,
    ClaimCreate,
    ClaimsRepo,
    DocumentsRepo,
    DocumentUpsert,
    EntitiesRepo,
    EntityCreate,
    ExtractionCreate,
    ExtractionsRepo,
    ReviewAuditRepo,
    SnapshotCreate,
    SnapshotsRepo,
)
from fmd.db.repositories import ClaimVerification
from fmd.er.config import ERConfig
from fmd.er.service import resolve_with_audit
from fmd.er.types import Alias, EntityCandidate, Mention

TOKEN = "review-console-test-token"
HEADERS = {"X-Admin-Token": TOKEN}
ENVELOPE_KEYS = {"scorer_version", "weights_version", "claims_hash", "generated_at"}

client = TestClient(app)


@pytest.fixture
def admin_token(monkeypatch) -> str:
    monkeypatch.setenv("FMD_ADMIN_TOKEN", TOKEN)
    return TOKEN


@pytest.fixture
def review_db(pg_database):
    """Route the app's database dependency at the isolated test schema."""
    app.dependency_overrides[_database_or_503] = lambda: pg_database
    yield pg_database
    app.dependency_overrides.pop(_database_or_503, None)


# --------------------------------------------------------------------------- seeds


def _suffix() -> str:
    return uuid.uuid4().hex[:10]


def _entity(database, name: str, state: str = "Delhi") -> int:
    return (
        EntitiesRepo(database)
        .create(EntityCreate(kind="institution", canonical_name=name, country="IN", state=state))
        .id
    )


def _extraction(database, entity_id: int, text: str) -> int:
    document = DocumentsRepo(database).upsert(
        DocumentUpsert(
            url=f"https://example.ac.in/{_suffix()}.pdf",
            domain="example.ac.in",
        )
    )
    snapshot = SnapshotsRepo(database).create(
        SnapshotCreate(
            document_id=document.id,
            sha256="a" * 64,
            bytes_path=f"/tmp/{_suffix()}.bin",
            fetcher_tier="direct",
        )
    )
    return (
        ExtractionsRepo(database)
        .create(ExtractionCreate(snapshot_id=snapshot.id, tool="fmd.extract", text=text))
        .id
    )


def _claim(
    database,
    extraction_id: int,
    entity_id: int,
    *,
    quote: str,
    text: str,
    confidence: float,
    predicate: str = "seats.total",
    value: int = 20,
) -> int:
    char_start = text.index(quote)
    claim = ClaimsRepo(database).create(
        ClaimCreate(
            extraction_id=extraction_id,
            entity_id=entity_id,
            predicate=predicate,
            value=value,
            quote=quote,
            char_start=char_start,
            char_end=char_start + len(quote),
            confidence=confidence,
        )
    )
    return claim.id


TEXT = "The B.Sc. MLT programme admits 20 students per year. Total intake is 40 seats."


@pytest.fixture
def seeded_claims(review_db):
    """Three claims on one entity: two unverified (low/mid confidence) and one
    machine-supported.

    The supported claim needs a quote that also occurs verbatim so a correction
    can re-locate it (``Total intake is 40 seats.`` in the extraction text).
    """
    database = review_db
    entity_id = _entity(database, f"Review Seed University {_suffix()}")
    extraction_id = _extraction(database, entity_id, TEXT)
    low_id = _claim(
        database, extraction_id, entity_id, quote="20 students", text=TEXT, confidence=0.2
    )
    mid_id = _claim(
        database,
        extraction_id,
        entity_id,
        quote="per year",
        text=TEXT,
        confidence=0.55,
        predicate="curriculum.duration_years",
        value=1,
    )
    supported_id = _claim(
        database,
        extraction_id,
        entity_id,
        quote="Total intake is 40 seats.",
        text=TEXT,
        confidence=0.9,
        predicate="seats.intake",
        value=40,
    )
    # Machine-supported verdict, as the verifier would leave it (verified_by=model).
    ClaimsRepo(database).record_verification(
        supported_id,
        ClaimVerification(
            status="supported",
            verified_by="mock-verifier",
            notes={"verdict": "supported", "reason": "quote matches the window"},
        ),
    )
    return {
        "entity_id": entity_id,
        "extraction_id": extraction_id,
        "low": low_id,
        "mid": mid_id,
        "supported": supported_id,
    }


def _er_config() -> ERConfig:
    return ERConfig.from_mapping(
        {"blocking": {"backend": "deterministic"}, "similarity": {"use_embeddings": False}}
    )


def _candidates_from_db(database, entity_ids: list[int]) -> list[EntityCandidate]:
    """Build the candidate universe exactly the way the planner does."""
    entities = EntitiesRepo(database)
    aliases = AliasesRepo(database)
    candidates: list[EntityCandidate] = []
    for entity_id in entity_ids:
        row = entities.get(entity_id)
        assert row is not None
        candidates.append(
            EntityCandidate(
                entity_id=entity_id,
                canonical_name=row.canonical_name,
                state=row.state,
                aliases=[
                    Alias(
                        alias=alias.alias,
                        alias_type=alias.alias_type,
                        confidence=alias.confidence if alias.confidence is not None else 1.0,
                        source_id=alias.source_id,
                    )
                    for alias in aliases.list_for_entity(entity_id)
                ],
            )
        )
    return candidates


# --------------------------------------------------------------------------- admin gate (offline)


def test_missing_token_is_403_on_every_review_endpoint(admin_token) -> None:
    # Token configured but not supplied: a permission verdict, not a challenge.
    for method, path in (
        ("get", "/api/review/queue"),
        ("get", "/api/review/audit"),
        ("post", "/api/review/claims/1"),
        ("post", "/api/review/aliases/1"),
        ("post", "/api/review/documents/1"),
    ):
        response = getattr(client, method)(path)
        assert response.status_code == 403, path
        body = response.json()
        assert body["error"]["code"] == "forbidden"
        assert body.keys() >= ENVELOPE_KEYS


def test_wrong_token_is_403_and_bearer_is_accepted(admin_token, monkeypatch) -> None:
    def boom() -> None:
        raise api_module.DatabaseError("no db")

    monkeypatch.setattr(api_module, "get_database", boom)
    assert client.get("/api/review/queue", headers={"X-Admin-Token": "nope"}).status_code == 403
    # The gate runs before the database dependency: with no DB at all, a valid
    # token gets the database verdict, not an auth one.
    assert client.get("/api/review/queue", headers=HEADERS).status_code == 503
    bearer = client.get("/api/review/queue", headers={"Authorization": f"Bearer {TOKEN}"})
    assert bearer.status_code == 503  # past the gate, blocked on the database


def test_unset_admin_token_disables_review(admin_token, monkeypatch) -> None:
    monkeypatch.delenv("FMD_ADMIN_TOKEN", raising=False)
    response = client.get("/api/review/queue")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "admin_disabled"


def test_queue_without_database_is_503(admin_token, monkeypatch) -> None:
    def boom() -> None:
        raise api_module.DatabaseError("cannot connect")

    monkeypatch.setattr(api_module, "get_database", boom)
    response = client.get("/api/review/queue", headers=HEADERS)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_unavailable"


# --------------------------------------------------------------------------- claim queue


def test_queue_orders_low_confidence_first_and_paginates(
    admin_token, seeded_claims, review_db
) -> None:
    response = client.get("/api/review/queue", params={"kind": "claim"}, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body.keys() >= ENVELOPE_KEYS
    assert body["kind"] == "claim"
    assert body["total"] >= 2
    confidences = [item["confidence"] for item in body["items"]]
    assert confidences == sorted(confidences)
    ids = [item["id"] for item in body["items"]]
    # Other DB-backed test files share the session schema, so the queue is
    # filtered to this fixture's entity: its two unverified claims queue in
    # confidence order, and the machine-supported one does not queue at all.
    own = [item for item in body["items"] if item["entity_id"] == seeded_claims["entity_id"]]
    assert [seeded_claims["low"], seeded_claims["mid"]] == [item["id"] for item in own]
    assert seeded_claims["supported"] not in ids
    # Provenance for the console columns: URL, snapshot hash, fetch tier.
    sample = body["items"][0]
    assert {
        "quote",
        "predicate",
        "url",
        "sha256",
        "snapshot_id",
        "fetcher_tier",
        "verification_status",
    } <= sample.keys()

    page = client.get(
        "/api/review/queue",
        params={"kind": "claim", "limit": 1, "offset": 0},
        headers=HEADERS,
    ).json()
    assert len(page["items"]) == 1
    next_page = client.get(
        "/api/review/queue",
        params={"kind": "claim", "limit": 1, "offset": 1},
        headers=HEADERS,
    ).json()
    assert next_page["items"][0]["id"] != page["items"][0]["id"]
    assert page["items"][0]["confidence"] <= next_page["items"][0]["confidence"]


# --------------------------------------------------------------------------- accept / reject


def test_accept_then_double_accept_is_idempotent(admin_token, seeded_claims, review_db) -> None:
    claim_id = seeded_claims["low"]
    first = client.post(
        f"/api/review/claims/{claim_id}", json={"action": "accept"}, headers=HEADERS
    )
    assert first.status_code == 200
    body = first.json()
    assert body["claim"]["verification_status"] == "supported"
    assert body["claim"]["verified_by"] == "human"
    assert body["already_done"] is False
    assert body["audit_id"] is not None
    # The claim leaves the queue once a human accepted it.
    queue = client.get("/api/review/queue", params={"kind": "claim"}, headers=HEADERS).json()
    assert claim_id not in [item["id"] for item in queue["items"]]

    second = client.post(
        f"/api/review/claims/{claim_id}", json={"action": "accept"}, headers=HEADERS
    )
    assert second.status_code == 200
    assert second.json()["already_done"] is True
    assert second.json()["audit_id"] is None

    audits = ReviewAuditRepo(review_db).list_for_subject("claim", claim_id)
    assert len(audits) == 1
    assert audits[0].decided_by == "human"
    assert audits[0].previous_state["verification_status"] == "unverified"
    assert audits[0].previous_state["verified_by"] is None


def test_rejecting_a_supported_claim_requires_a_note(admin_token, seeded_claims, review_db) -> None:
    claim_id = seeded_claims["supported"]
    missing = client.post(
        f"/api/review/claims/{claim_id}", json={"action": "reject"}, headers=HEADERS
    )
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "missing_note"

    ok = client.post(
        f"/api/review/claims/{claim_id}",
        json={"action": "reject", "note": "quote is from the previous session's brochure"},
        headers=HEADERS,
    )
    assert ok.status_code == 200
    claim = ok.json()["claim"]
    assert claim["verification_status"] == "rejected"
    assert claim["verified_by"] == "human"
    # The verifier's notes survive the override, readable under previous_verification.
    notes = claim["verification_notes"]
    assert notes["human_review"]["action"] == "reject"
    assert notes["previous_verification"]["verdict"] == "supported"

    audits = ReviewAuditRepo(review_db).list_for_subject("claim", claim_id)
    assert audits[0].action == "reject"
    assert audits[0].previous_state["verification_status"] == "supported"
    assert audits[0].previous_state["verified_by"] == "mock-verifier"


def test_reject_an_unverified_claim_without_note_is_allowed(
    admin_token, seeded_claims, review_db
) -> None:
    claim_id = seeded_claims["low"]
    response = client.post(
        f"/api/review/claims/{claim_id}", json={"action": "reject"}, headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json()["claim"]["verification_status"] == "rejected"


# --------------------------------------------------------------------------- correction


def test_correction_supersedes_without_mutating(admin_token, seeded_claims, review_db) -> None:
    repo = ClaimsRepo(review_db)
    claim_id = seeded_claims["low"]
    before = repo.get(claim_id)
    assert before is not None

    corrected_quote = "Total intake is 40 seats."
    response = client.post(
        f"/api/review/claims/{claim_id}",
        json={
            "action": "correct",
            "corrected_quote": corrected_quote,
            "note": "the low-confidence span was the per-year sentence, not the intake row",
        },
        headers=HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    replacement = body["claim"]
    superseded = body["superseded_claim"]

    # The replacement carries the corrected quote, a resolvable span, human support.
    assert replacement["quote"] == corrected_quote
    assert replacement["id"] != claim_id
    assert replacement["verification_status"] == "supported"
    assert replacement["verified_by"] == "human"
    assert replacement["char_end"] - replacement["char_start"] == len(corrected_quote)
    assert replacement["superseded_by"] is None

    # The original row is untouched apart from the supersede pointer.
    assert superseded["id"] == claim_id
    assert superseded["superseded_by"] == replacement["id"]
    after = repo.get(claim_id)
    assert after is not None
    assert (after.value, after.quote, after.char_start, after.char_end, after.confidence) == (
        before.value,
        before.quote,
        before.char_start,
        before.char_end,
        before.confidence,
    )
    assert after.verification_status == before.verification_status

    # The replacement's span resolves in the extraction text (invariant 3).
    extraction = ExtractionsRepo(review_db).get(replacement["extraction_id"])
    assert extraction is not None
    assert extraction.text[replacement["char_start"] : replacement["char_end"]] == corrected_quote

    audits = ReviewAuditRepo(review_db).list_for_subject("claim", claim_id)
    assert audits[0].action == "correct"
    assert audits[0].previous_state["verification_status"] == before.verification_status

    # The queue now serves the replacement (unverified -> queue) not the old row.
    queue = client.get("/api/review/queue", params={"kind": "claim"}, headers=HEADERS).json()
    ids = [item["id"] for item in queue["items"]]
    assert claim_id not in ids


def test_correction_validation(admin_token, seeded_claims, review_db) -> None:
    claim_id = seeded_claims["low"]
    no_quote = client.post(
        f"/api/review/claims/{claim_id}", json={"action": "correct", "note": "x"}, headers=HEADERS
    )
    assert no_quote.status_code == 422
    assert no_quote.json()["error"]["code"] == "missing_corrected_quote"

    unresolvable = client.post(
        f"/api/review/claims/{claim_id}",
        json={"action": "correct", "corrected_quote": "not in the document at all", "note": "x"},
        headers=HEADERS,
    )
    assert unresolvable.status_code == 422
    assert unresolvable.json()["error"]["code"] == "unresolvable_corrected_quote"

    no_note = client.post(
        f"/api/review/claims/{claim_id}",
        json={"action": "correct", "corrected_quote": "Total intake is 40 seats."},
        headers=HEADERS,
    )
    assert no_note.status_code == 422
    assert no_note.json()["error"]["code"] == "missing_note"


def test_actions_on_a_superseded_claim_conflict(admin_token, seeded_claims, review_db) -> None:
    claim_id = seeded_claims["low"]
    first = client.post(
        f"/api/review/claims/{claim_id}",
        json={"action": "correct", "corrected_quote": "Total intake is 40 seats.", "note": "n"},
        headers=HEADERS,
    )
    assert first.status_code == 200
    for action in ("accept", "reject", "correct"):
        response = client.post(
            f"/api/review/claims/{claim_id}",
            json={"action": action, "corrected_quote": "Total intake is 40 seats.", "note": "n"},
            headers=HEADERS,
        )
        assert response.status_code == 409, action
        assert response.json()["error"]["code"] == "already_superseded"


def test_unknown_claim_is_404(admin_token, review_db) -> None:
    response = client.post("/api/review/claims/999999", json={"action": "accept"}, headers=HEADERS)
    assert response.status_code == 404


def test_invalid_action_is_422(admin_token, seeded_claims, review_db) -> None:
    response = client.post(
        f"/api/review/claims/{seeded_claims['low']}", json={"action": "delete"}, headers=HEADERS
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- aliases


def _alias(database, entity_id: int, alias: str, alias_type: str = "manual") -> int:
    return (
        AliasesRepo(database)
        .create(
            AliasCreate(entity_id=entity_id, alias=alias, alias_type=alias_type, confidence=None)
        )
        .id
    )


def test_alias_confirm_blocks_auto_merge(admin_token, review_db) -> None:
    """The acceptance criterion: a confirmed manual alias is ER-authoritative.

    Chain: pending manual alias in the queue -> confirm via the API -> the
    candidate universe the planner builds (``AliasesRepo.list_for_entity``) ->
    ER keeps the pair locked. A mention that exactly matches the confirmed
    alias resolves unattended; a *different* mention that merely scores high
    against the same entity is declined to adjudication instead of auto-merging.
    """
    database = review_db
    suffix = _suffix()
    locked_id = _entity(database, f"Locked University {suffix}")
    other_id = _entity(database, f"Other College {suffix}")
    # A human-proposed manual alias: the institution's abbreviation, not its
    # canonical name, so the canonical mention cannot exact-match it.
    alias_id = _alias(database, locked_id, f"LUX Medical College {suffix}")

    queue = client.get("/api/review/queue", params={"kind": "alias"}, headers=HEADERS).json()
    assert queue["kind"] == "alias"
    queued = [item for item in queue["items"] if item["id"] == alias_id]
    assert queued and queued[0]["review_status"] == "pending"
    assert queued[0]["entity_name"] == f"Locked University {suffix}"

    confirmed = client.post(
        f"/api/review/aliases/{alias_id}", json={"action": "confirm"}, headers=HEADERS
    )
    assert confirmed.status_code == 200
    alias = confirmed.json()["alias"]
    assert alias["review_status"] == "confirmed"
    assert alias["confirmed_by"] == "human"
    assert alias["confirmed_at"] is not None

    candidates = _candidates_from_db(database, [locked_id, other_id])
    exact = resolve_with_audit(
        [Mention(text=f"LUX Medical College {suffix}", state="Delhi", entity_id=other_id)],
        candidates,
        config=_er_config(),
    ).resolutions[0]
    assert exact.chosen is not None
    assert exact.chosen.entity_id == locked_id
    assert exact.audit.decided_by == "manual"
    assert "manual_alias_authoritative" in exact.audit.notes

    # The canonical name of the locked entity no longer auto-accepts: the pair
    # stays with the human.
    canonical = resolve_with_audit(
        [Mention(text=f"Locked University {suffix}", state="Delhi", entity_id=locked_id)],
        [c for c in candidates if c.entity_id == locked_id],
        config=_er_config(),
    ).resolutions[0]
    assert canonical.chosen is None
    assert canonical.audit.decided_by == "none"
    assert "manual_canonical_lock" in canonical.audit.notes


def test_alias_reject_removes_the_suggestion_from_resolution(admin_token, review_db) -> None:
    database = review_db
    suffix = _suffix()
    entity_id = _entity(database, f"Rejected Alias University {suffix}")
    alias_id = _alias(database, entity_id, f"Totally Wrong Name {suffix}")

    rejected = client.post(
        f"/api/review/aliases/{alias_id}", json={"action": "reject"}, headers=HEADERS
    )
    assert rejected.status_code == 200
    assert rejected.json()["alias"]["review_status"] == "rejected"

    # The ER-facing read no longer hands the rejected alias to the planner.
    assert AliasesRepo(database).list_for_entity(entity_id) == []
    # And it left the pending queue.
    queue = client.get("/api/review/queue", params={"kind": "alias"}, headers=HEADERS).json()
    assert alias_id not in [item["id"] for item in queue["items"]]


def test_alias_decision_is_idempotent_and_flips_conflict(admin_token, review_db) -> None:
    database = review_db
    entity_id = _entity(database, f"Alias Flow {_suffix()}")
    confirm_id = _alias(database, entity_id, "Alias Flow A")
    reject_id = _alias(database, entity_id, "Alias Flow B")

    first = client.post(
        f"/api/review/aliases/{confirm_id}", json={"action": "confirm"}, headers=HEADERS
    )
    again = client.post(
        f"/api/review/aliases/{confirm_id}", json={"action": "confirm"}, headers=HEADERS
    )
    assert first.status_code == 200 and first.json()["already_done"] is False
    assert again.status_code == 200 and again.json()["already_done"] is True
    assert len(ReviewAuditRepo(database).list_for_subject("alias", confirm_id)) == 1

    flip = client.post(
        f"/api/review/aliases/{confirm_id}", json={"action": "reject"}, headers=HEADERS
    )
    assert flip.status_code == 409
    assert flip.json()["error"]["code"] == "decision_conflict"

    client.post(f"/api/review/aliases/{reject_id}", json={"action": "reject"}, headers=HEADERS)
    assert (
        client.post(
            f"/api/review/aliases/{reject_id}", json={"action": "reject"}, headers=HEADERS
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/api/review/aliases/{reject_id}", json={"action": "confirm"}, headers=HEADERS
        ).status_code
        == 409
    )


def test_unknown_alias_is_404(admin_token, review_db) -> None:
    assert (
        client.post(
            "/api/review/aliases/999999", json={"action": "confirm"}, headers=HEADERS
        ).status_code
        == 404
    )


# --------------------------------------------------------------------------- documents


def _needs_review_document(database) -> int:
    document = DocumentsRepo(database).upsert(
        DocumentUpsert(
            url=f"https://ambiguous.ac.in/{_suffix()}.pdf",
            domain="ambiguous.ac.in",
        )
    )
    DocumentsRepo(database).set_plan(
        document.id,
        extraction_family="fee_structure",
        entity_id=None,
        plan_state="needs_review",
        plan_confidence=0.3,
        plan_reason="two institutions on this domain",
    )
    return document.id


def test_document_link_entity_resolves_the_abstention(admin_token, review_db) -> None:
    database = review_db
    document_id = _needs_review_document(database)
    entity_id = _entity(database, f"Link Target {_suffix()}")

    queue = client.get("/api/review/queue", params={"kind": "document"}, headers=HEADERS).json()
    queued = [item for item in queue["items"] if item["id"] == document_id]
    assert queued and queued[0]["plan_state"] == "needs_review"

    missing_entity = client.post(
        f"/api/review/documents/{document_id}", json={"action": "link_entity"}, headers=HEADERS
    )
    assert missing_entity.status_code == 422
    assert missing_entity.json()["error"]["code"] == "missing_entity"

    unknown_entity = client.post(
        f"/api/review/documents/{document_id}",
        json={"action": "link_entity", "entity_id": 999999},
        headers=HEADERS,
    )
    assert unknown_entity.status_code == 422
    assert unknown_entity.json()["error"]["code"] == "unknown_entity"

    linked = client.post(
        f"/api/review/documents/{document_id}",
        json={"action": "link_entity", "entity_id": entity_id, "note": "the second name"},
        headers=HEADERS,
    )
    assert linked.status_code == 200
    document = linked.json()["document"]
    assert document["plan_state"] == "linked"
    assert document["entity_id"] == entity_id
    # The planner's own fields survive a review decision.
    assert document["extraction_family"] == "fee_structure"

    repeat = client.post(
        f"/api/review/documents/{document_id}",
        json={"action": "link_entity", "entity_id": entity_id},
        headers=HEADERS,
    )
    assert repeat.status_code == 200
    assert repeat.json()["already_done"] is True

    audits = ReviewAuditRepo(database).list_for_subject("document", document_id)
    assert len(audits) == 1
    assert audits[0].previous_state["plan_state"] == "needs_review"
    assert audits[0].previous_state["entity_id"] is None

    # The resolved document left the queue.
    queue = client.get("/api/review/queue", params={"kind": "document"}, headers=HEADERS).json()
    assert document_id not in [item["id"] for item in queue["items"]]


def test_document_mark_not_applicable(admin_token, review_db) -> None:
    database = review_db
    document_id = _needs_review_document(database)
    response = client.post(
        f"/api/review/documents/{document_id}",
        json={"action": "mark_not_applicable", "note": "a navigation page, not evidence"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    document = response.json()["document"]
    assert document["plan_state"] == "not_applicable"
    assert "navigation page" in (document["plan_reason"] or "")

    flip = client.post(
        f"/api/review/documents/{document_id}",
        json={"action": "link_entity", "entity_id": _entity(database, f"X {_suffix()}")},
        headers=HEADERS,
    )
    assert flip.status_code == 409


def test_unknown_document_is_404(admin_token, review_db) -> None:
    assert (
        client.post(
            "/api/review/documents/999999", json={"action": "mark_not_applicable"}, headers=HEADERS
        ).status_code
        == 404
    )


# --------------------------------------------------------------------------- audit read


def test_audit_endpoint_returns_the_trail(admin_token, seeded_claims, review_db) -> None:
    claim_id = seeded_claims["low"]
    client.post(f"/api/review/claims/{claim_id}", json={"action": "accept"}, headers=HEADERS)
    response = client.get(
        "/api/review/audit",
        params={"kind": "claim", "subject_id": claim_id},
        headers=HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body.keys() >= ENVELOPE_KEYS
    assert body["items"]
    audit = body["items"][0]
    assert audit["kind"] == "claim"
    assert audit["subject_id"] == claim_id
    assert audit["action"] == "accept"
    assert audit["decided_by"] == "human"
    assert audit["previous_state"]["verification_status"] == "unverified"
    assert audit["created_at"] is not None
    assert datetime.fromisoformat(audit["created_at"]).tzinfo is not None


def test_audit_endpoint_requires_the_token(admin_token) -> None:
    assert (
        client.get("/api/review/audit", params={"kind": "claim", "subject_id": 1}).status_code
        == 403
    )
