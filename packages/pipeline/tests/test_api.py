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
    assert ENVELOPE_KEYS <= body.keys()
    assert len(body["claims_hash"]) == 64


def test_sources_lists_registry_with_trust() -> None:
    response = client.get("/api/sources")
    assert response.status_code == 200
    body = response.json()
    assert ENVELOPE_KEYS <= body.keys()
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
    assert ENVELOPE_KEYS <= body.keys()


@pytest.mark.parametrize(
    ("method", "path", "workstream"),
    [
        ("get", "/api/programmes/1/cost", "WS-F"),
        ("get", "/api/rankings", "WS-F"),
        ("get", "/api/compare", "WS-F"),
        ("get", "/api/exams", "WS-F"),
        ("get", "/api/review/queue", "WS-G"),
    ],
)
def test_unimplemented_endpoints_are_501(method: str, path: str, workstream: str) -> None:
    response = getattr(client, method)(path)
    assert response.status_code == 501
    body = response.json()
    assert body["error"]["workstream"] == workstream
    assert "not implemented" in body["error"]["message"]
    assert ENVELOPE_KEYS <= body.keys()


def test_admin_write_endpoints_are_gated(monkeypatch) -> None:
    monkeypatch.delenv("FMD_ADMIN_TOKEN", raising=False)
    assert client.post("/api/scoring/rerun").status_code == 503

    monkeypatch.setenv("FMD_ADMIN_TOKEN", "test-token")
    assert client.post("/api/scoring/rerun").status_code == 401
    assert (
        client.post("/api/scoring/rerun", headers={"X-Admin-Token": "wrong"}).status_code
        == 401
    )
    allowed = client.post("/api/scoring/rerun", headers={"X-Admin-Token": "test-token"})
    assert allowed.status_code == 501
    assert allowed.json()["error"]["workstream"] == "WS-F"


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
        assert ENVELOPE_KEYS <= body.keys()
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
