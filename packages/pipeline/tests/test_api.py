from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import fmd.api as api_module
from fmd.api import app
from fmd.db import DatabaseError
from fmd.db.repositories import EntitiesRepo, EntityCreate, ProgrammeCreate, ProgrammesRepo

client = TestClient(app)
ENVELOPE_KEYS = {"scorer_version", "weights_version", "claims_hash", "generated_at"}


class _FakeDatabase:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def open(self, timeout: float = 1.0) -> None:
        pass

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass


def test_health_envelope(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "Database", _FakeDatabase)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "up"
    assert body.keys() >= ENVELOPE_KEYS
    assert len(body["claims_hash"]) == 64


def test_sources_lists_registry_with_trust() -> None:
    response = client.get("/api/sources")
    assert response.status_code == 200
    body = response.json()
    assert body.keys() >= ENVELOPE_KEYS
    assert body["items"]
    sample = body["items"][0]
    assert {"id", "trust", "adapter", "enabled", "last_crawl"} <= sample.keys()
    assert body["exclusions"]


def test_database_unavailable_is_503(monkeypatch) -> None:
    def boom() -> None:
        raise DatabaseError("cannot connect")

    monkeypatch.setattr(api_module, "get_database", boom)
    response = client.get("/api/programmes")
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "database_unavailable"
    assert body.keys() >= ENVELOPE_KEYS


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/review/queue"),
        ("post", "/api/review/claims/1"),
        ("post", "/api/review/aliases/1"),
        ("post", "/api/review/documents/1"),
    ],
)
def test_review_endpoints_are_gated_and_no_longer_501(method: str, path: str, monkeypatch) -> None:
    # WS-G has shipped: the review surface is no longer the 501 stub. Without a
    # token the gate answers 403 before anything else; the full action contract
    # lives in test_review_api.py.
    monkeypatch.delenv("FMD_ADMIN_TOKEN", raising=False)
    kwargs = {"headers": {"X-Admin-Token": "test-token"}} if path.endswith("/1") else {}
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "admin_disabled"
    assert body.keys() >= ENVELOPE_KEYS


def test_engine_endpoints_require_a_database(monkeypatch) -> None:
    def boom() -> None:
        raise DatabaseError("cannot connect")

    monkeypatch.setattr(api_module, "get_database", boom)
    for response in (
        client.post("/api/rankings", json={"profile": {}}),
        client.get("/api/programmes/1/cost"),
        client.get("/api/compare", params={"ids": "1,2"}),
        client.post("/api/exclusions", json={"profile": {}}),
    ):
        assert response.status_code == 503
        assert response.json().keys() >= ENVELOPE_KEYS


def test_exams_is_a_post_and_degrades_without_db(monkeypatch) -> None:
    def boom() -> None:
        raise DatabaseError("cannot connect")

    monkeypatch.setattr(api_module, "get_database", boom)
    response = client.post("/api/exams", json={"profile": {"travel_for_exams": "no"}})
    assert response.status_code == 200
    body = response.json()
    assert body.keys() >= ENVELOPE_KEYS
    assert body["items"]
    assert body["profile_persisted"] is False
    for item in body["items"]:
        assert {"status", "observed_at", "stale", "requires_travel", "deciding_claim_id"} <= (
            item.keys()
        )
        assert item["status"] in {"open", "closed", "unverified"}

    # The old GET contract is gone; profiles live in the request body.
    assert client.get("/api/exams").status_code == 405


def test_admin_write_endpoints_are_gated(monkeypatch) -> None:
    # The admin gate resolves before the database dependency, so every case here
    # is asserted with no database reachable (the DB-down 503 must never shadow
    # the auth verdict, and auth must never need a live Postgres).
    def boom() -> None:
        raise DatabaseError("cannot connect")

    monkeypatch.delenv("FMD_ADMIN_TOKEN", raising=False)
    assert client.post("/api/scoring/rerun").status_code == 503

    monkeypatch.setenv("FMD_ADMIN_TOKEN", "test-token")
    assert client.post("/api/scoring/rerun").status_code == 401
    assert client.post("/api/scoring/rerun", headers={"X-Admin-Token": "wrong"}).status_code == 401

    # Valid token but no database: 503 database_unavailable, not a crash.
    monkeypatch.setattr(api_module, "get_database", boom)
    authorized = client.post("/api/scoring/rerun", headers={"X-Admin-Token": "test-token"})
    assert authorized.status_code == 503
    assert authorized.json()["error"]["code"] == "database_unavailable"


def test_validation_errors_never_echo_the_profile() -> None:
    """A malformed body must not reflect the submitted profile back.

    FastAPI's validation errors carry the offending ``input`` verbatim; the
    422 handler strips it, so family financing data never reaches the response
    (or any log that captures response bodies). The database dependency is
    overridden because FastAPI resolves dependencies before validating the
    body — a down database would otherwise answer 503 first.
    """

    app.dependency_overrides[api_module._database_or_503] = lambda: object()
    try:
        sentinel = "SENTINEL-ECHO-42fmd"
        response = client.post(
            "/api/rankings",
            json={
                "profile": {"regions": [sentinel], "acceptance_marker": sentinel},
                "weights": "not-an-object",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
        assert sentinel not in response.text
        assert "regions" not in response.text
    finally:
        app.dependency_overrides.clear()


def test_db_backed_programme_reads(pg_database) -> None:
    suffix = uuid.uuid4().hex[:10]
    institution = EntitiesRepo(pg_database).create(
        EntityCreate(
            kind="institution",
            canonical_name=f"API Institute {suffix}",
            country="IN",
            state="MH",
        )
    )
    programme = ProgrammesRepo(pg_database).create(
        ProgrammeCreate(
            institution_id=institution.id,
            name=f"BSc MLT {suffix}",
            credential_level="degree",
            recognition_status="recognised",
        )
    )

    app.dependency_overrides[api_module._database_or_503] = lambda: pg_database
    try:
        listed = client.get("/api/programmes", params={"state": "MH", "q": "BSc MLT"})
        assert listed.status_code == 200
        body = listed.json()
        assert body.keys() >= ENVELOPE_KEYS
        assert any(item["id"] == programme.id for item in body["items"])
        assert body["total"] >= 1
        assert body["facets"]["credential_level"]["degree"] >= 1

        detail = client.get(f"/api/programmes/{programme.id}")
        assert detail.status_code == 200
        detail_body = detail.json()
        assert detail_body["programme"]["id"] == programme.id
        assert detail_body["institution"]["id"] == institution.id
        assert detail_body["claims"] == []
        assert detail_body["claims_by_predicate"] == {}

        evidence = client.get(f"/api/programmes/{programme.id}/evidence")
        assert evidence.status_code == 200
        assert evidence.json()["total"] == 0

        missing = client.get("/api/programmes/99999999")
        assert missing.status_code == 404
    finally:
        app.dependency_overrides.clear()


def _seed_engine_programme(database, suffix: str):
    import hashlib

    from fmd.db.repositories import (
        ClaimCreate,
        ClaimsRepo,
        CostItemCreate,
        CostItemsRepo,
        DocumentsRepo,
        DocumentUpsert,
        ExtractionCreate,
        ExtractionsRepo,
        SnapshotCreate,
        SnapshotsRepo,
    )

    institution = EntitiesRepo(database).create(
        EntityCreate(
            kind="institution",
            canonical_name=f"Engine Institute {suffix}",
            country="IN",
            state="MH",
        )
    )
    programme = ProgrammesRepo(database).create(
        ProgrammeCreate(
            institution_id=institution.id,
            name=f"BSc MLT Engine {suffix}",
            credential_level="degree",
            recognition_status="recognised",
            duration_months=48,
        )
    )
    Costs = CostItemsRepo(database)
    for category, amount in (("tuition", 100000), ("hostel", 60000), ("mess", 40000)):
        Costs.create(
            CostItemCreate(
                programme_id=programme.id,
                category=category,
                label=category,
                amount=amount,
                currency="INR",
                period="per_year",
            )
        )

    document = DocumentsRepo(database).upsert(
        DocumentUpsert(url=f"https://example.ac.in/{suffix}.pdf", domain="example.ac.in")
    )
    data = b"%PDF engine evidence"
    digest = hashlib.sha256(data).hexdigest()
    snapshot = SnapshotsRepo(database).create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=f"objects/{digest[:2]}/{digest}",
            fetcher_tier="httpx",
            content_type="application/pdf",
        )
    )
    extraction = ExtractionsRepo(database).create(
        ExtractionCreate(
            snapshot_id=snapshot.id,
            tool="test",
            text="engine evidence text",
            prompt_version="v1",
        )
    )
    claims = ClaimsRepo(database)
    for index, (predicate, value, unit) in enumerate(
        [
            ("recognition_status", "recognised", None),
            ("naac_grade", "A", None),
            ("internship_hours", 1200, "hours"),
            ("p25_first_year_salary", 300000, "INR"),
            ("city_safety_index", 70, "percent"),
        ]
    ):
        claims.create(
            ClaimCreate(
                extraction_id=extraction.id,
                entity_id=institution.id,
                predicate=predicate,
                value=value,
                unit=unit,
                qualifier={"cohort_year": 2025},
                quote="engine evidence text",
                char_start=0,
                char_end=5,
                confidence=1.0,
                method="extracted",
            )
        )
        del index
    return institution, programme


def test_engine_endpoints_against_real_db(pg_database, monkeypatch) -> None:
    suffix = uuid.uuid4().hex[:10]
    institution, programme = _seed_engine_programme(pg_database, suffix)
    second = ProgrammesRepo(pg_database).create(
        ProgrammeCreate(
            institution_id=institution.id,
            name=f"BSc MLT Engine B {suffix}",
            credential_level="degree",
            recognition_status="recognised",
            duration_months=48,
        )
    )
    monkeypatch.setattr(api_module, "get_database", lambda: pg_database)
    app.dependency_overrides[api_module._database_or_503] = lambda: pg_database
    profile = {
        "credential_levels": ["degree"],
        "budget": {"currency": "INR", "total_ceiling": 5_000_000},
        "target_intake_year": 2026,
        "travel_for_exams": "yes",
        "financing": {
            "loan_required": True,
            "collateral_available": True,
            "annual_family_income": 900_000,
        },
    }
    try:
        ranked = client.post("/api/rankings", json={"profile": profile})
        assert ranked.status_code == 200
        body = ranked.json()
        assert body.keys() >= ENVELOPE_KEYS
        assert body["profile_persisted"] is False
        assert set(body["weights"]) == {
            "outcomes",
            "affordability_roi",
            "clinical_training",
            "academic_quality",
            "student_life_safety",
            "admission_realism",
            "financing_feasibility",
        }
        item = next(row for row in body["items"] if row["programme"]["id"] == programme.id)
        assert {
            "programme",
            "score",
            "composite_band",
            "evidence_tier",
            "filters",
            "insights",
        } <= item.keys()
        assert item["composite_band"]["method"]
        assert item["filters"]["counts"]
        assert item["score"]["sub_scores"]["outcomes"]["evidence_tier"]
        for sub_score in item["score"]["sub_scores"].values():
            for component in sub_score["components"]:
                assert component["method"]

        override = dict.fromkeys(body["weights"], 0.0)
        override["outcomes"] = 100.0
        overridden = client.post("/api/rankings", json={"profile": profile, "weights": override})
        assert overridden.status_code == 200
        assert "+override." in overridden.json()["weights_version"]
        bad = client.post("/api/rankings", json={"profile": profile, "weights": {"outcomes": 10}})
        assert bad.status_code == 422
        assert bad.json()["error"]["code"] == "invalid_weights"

        cost = client.post(
            f"/api/programmes/{programme.id}/cost",
            json={"profile": profile, "living": "hostel"},
        )
        assert cost.status_code == 200
        cost_body = cost.json()
        assert cost_body.keys() >= ENVELOPE_KEYS
        assert cost_body["available"] is True
        assert cost_body["profile_persisted"] is False
        assert set(cost_body["scenarios"]) == {"hostel", "independent"}
        assert set(cost_body["scenario_totals"]) == {"hostel", "independent"}
        assert cost_body["non_additive_scenarios"] is True
        assert cost_body["assumed"]["loan_rate"]["assumed"] is True
        assert cost_body["assumed"]["loan_terms"]["assumed"] is True
        assert cost_body["dsr25"]["band"] in {
            "comfortable",
            "manageable",
            "stressed",
            "not_financeable",
            "unknown",
        }
        assert cost_body["tcoa"]["method"]
        assert cost_body["affordability_gap"]["value"] is not None
        for scenario in cost_body["scenario_totals"].values():
            assert scenario["dsr25"]["band"]
            assert scenario["emi"]["method"]
            assert scenario["tcoa"]["value"] is not None
        assert cost_body["unknown_reasons"] == []

        independent = client.post(
            f"/api/programmes/{programme.id}/cost",
            json={"profile": profile, "living": "independent"},
        )
        assert independent.status_code == 200
        assert independent.json()["living"] == "independent"

        filters = client.post(f"/api/programmes/{programme.id}/filters", json={"profile": profile})
        assert filters.status_code == 200
        filters_body = filters.json()
        assert filters_body["needs_profile"] is False
        assert filters_body["profile_persisted"] is False
        assert len(filters_body["filters"]) == 8
        assert filters_body["order"] == [item["filter_id"] for item in filters_body["filters"]]
        for decision in filters_body["filters"]:
            assert decision["result"] in {"pass", "fail", "unknown"}
            assert "reason" in decision and "claim_id" in decision
        assert isinstance(filters_body["would_exclude"], bool)

        needs_profile = client.get(f"/api/programmes/{programme.id}/cost")
        assert needs_profile.status_code == 200
        assert needs_profile.json()["needs_profile"] is True
        assert needs_profile.json()["available"] is False
        assert "tcoa" not in needs_profile.json()

        no_profile_filters = client.get(f"/api/programmes/{programme.id}/filters")
        assert no_profile_filters.status_code == 200
        assert no_profile_filters.json()["needs_profile"] is True
        assert no_profile_filters.json()["filters"] == []

        compare = client.get("/api/compare", params={"ids": f"{programme.id},{second.id}"})
        assert compare.status_code == 200
        compare_body = compare.json()
        assert compare_body["matrix"]
        assert compare_body["comparisons"]
        assert isinstance(compare_body["not_comparable_pairs"], list)

        compared = client.get("/api/compare", params={"ids": f"{programme.id}"})
        assert compared.status_code == 422

        excluded = client.post("/api/exclusions", json={"profile": profile})
        assert excluded.status_code == 200
        ex_body = excluded.json()
        assert {"histogram", "excluded", "excluded_total"} <= ex_body.keys()
        assert ex_body["profile_persisted"] is False

        exams = client.post("/api/exams", json={"profile": profile})
        assert exams.status_code == 200
        assert exams.json()["items"]
        del institution
    finally:
        app.dependency_overrides.clear()


def test_programme_cost_partial_profile_lists_unknown_reasons(pg_database, monkeypatch) -> None:
    suffix = uuid.uuid4().hex[:10]
    _institution, programme = _seed_engine_programme(pg_database, suffix)
    monkeypatch.setattr(api_module, "get_database", lambda: pg_database)
    app.dependency_overrides[api_module._database_or_503] = lambda: pg_database
    try:
        response = client.post(
            f"/api/programmes/{programme.id}/cost",
            json={
                "profile": {"financing": {"family_income_band": "4.5-8L"}},
                "living": "hostel",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is True
        assert body["affordability_gap"]["value"] is None
        assert body["affordability_gap"]["annual_family_income"] is None
        joined = " | ".join(body["unknown_reasons"])
        assert "income:" in joined
        assert "collateral:" in joined
        # An income band is not silently turned into a midpoint.
        assert not any(reason.startswith("salary:") for reason in body["unknown_reasons"])
    finally:
        app.dependency_overrides.clear()


def test_programme_cost_and_filters_reject_malformed_bodies(pg_database) -> None:
    app.dependency_overrides[api_module._database_or_503] = lambda: pg_database
    try:
        suffix = uuid.uuid4().hex[:10]
        _institution, programme = _seed_engine_programme(pg_database, suffix)
        for payload in (
            {"living": "hostel"},
            {"profile": "not-an-object", "living": "hostel"},
            {"profile": {}, "living": "hotel"},
        ):
            response = client.post(f"/api/programmes/{programme.id}/cost", json=payload)
            assert response.status_code == 422, payload
        malformed = client.post(f"/api/programmes/{programme.id}/filters", json={})
        assert malformed.status_code == 422
        assert malformed.json()["error"]["code"] == "validation_error"
    finally:
        app.dependency_overrides.clear()


def _table_names(database) -> list[str]:
    with database.connection() as conn:
        rows = conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()
    return [str(row["table_name"]) for row in rows]


def test_profile_is_never_persisted(pg_database) -> None:
    import json

    suffix = uuid.uuid4().hex[:10]
    sentinel = f"SENTINEL-PROFILE-{suffix}"
    _institution, programme = _seed_engine_programme(pg_database, suffix)
    app.dependency_overrides[api_module._database_or_503] = lambda: pg_database
    profile = {
        "target_field": sentinel,
        "regions": [sentinel],
        "financing": {
            "loan_required": True,
            "collateral_available": True,
            "annual_family_income": 750_000,
        },
    }
    try:
        cost = client.post(
            f"/api/programmes/{programme.id}/cost",
            json={"profile": profile, "living": "independent"},
        )
        assert cost.status_code == 200
        assert sentinel not in cost.text

        filters = client.post(f"/api/programmes/{programme.id}/filters", json={"profile": profile})
        assert filters.status_code == 200
        assert sentinel not in filters.text

        with pg_database.connection() as conn:
            llm_rows = conn.execute("SELECT count(*) AS total FROM llm_calls").fetchone()
            assert int(llm_rows["total"]) == 0
            for table in _table_names(pg_database):
                hit = conn.execute(
                    f'SELECT count(*) AS total FROM "{table}" WHERE "{table}"::text LIKE %s',
                    (f"%{sentinel}%",),
                ).fetchone()
                assert int(hit["total"]) == 0, table
            assert json.dumps(profile) not in cost.text
    finally:
        app.dependency_overrides.clear()
