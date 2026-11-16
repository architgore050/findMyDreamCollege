"""FastAPI surface (brief §17).

WS-A implements health, sources and DB-backed programme reads. Endpoints owned
by other workstreams return HTTP 501 with the owning workstream named. Every
response carries ``scorer_version``, ``weights_version``, ``claims_hash`` and
``generated_at``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fmd import __version__
from fmd.config import ConfigError, Registry, get_registry
from fmd.db.connection import Database, DatabaseError, get_database
from fmd.db.repositories import (
    ClaimsRepo,
    CostItemsRepo,
    EntitiesRepo,
    ExamsRepo,
    ProgrammeFilter,
    ProgrammesRepo,
    ScoresRepo,
)
from fmd.settings import get_settings

EMPTY_CLAIMS_HASH = hashlib.sha256(b"").hexdigest()


# --------------------------------------------------------------------------- helpers


def _registry() -> Registry | None:
    try:
        return get_registry()
    except ConfigError:
        return None


def _meta(claims_hash: str | None = None) -> dict[str, Any]:
    registry = _registry()
    return {
        "scorer_version": registry.scorer_version if registry else "unknown",
        "weights_version": registry.weights_version if registry else "unknown",
        "claims_hash": claims_hash or EMPTY_CLAIMS_HASH,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _envelope(payload: dict[str, Any], claims_hash: str | None = None) -> dict[str, Any]:
    return {**payload, **_meta(claims_hash)}


def _claims_hash(claims: list[Any]) -> str:
    """Deterministic hash of exactly the claims used to build a response."""
    canonical = [
        {
            "id": claim.id,
            "entity_id": claim.entity_id,
            "predicate": claim.predicate,
            "value": claim.value,
            "quote": claim.quote,
            "char_start": claim.char_start,
            "char_end": claim.char_end,
            "method": claim.method,
        }
        for claim in claims
    ]
    canonical.sort(key=lambda item: item["id"])
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _error(
    status_code: int,
    code: str,
    message: str,
    *,
    workstream: str | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if workstream:
        error["workstream"] = workstream
    return JSONResponse(status_code=status_code, content=_envelope({"error": error}))


def _database_or_503() -> Database:
    try:
        return get_database()
    except DatabaseError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "database_unavailable",
                "message": (
                    f"database unavailable: {exc}. Start Postgres with "
                    "`docker compose -f infra/docker-compose.yml up -d db` and run "
                    "`fmd migrate`."
                ),
            },
        ) from exc


DatabaseDep = Annotated[Database, Depends(_database_or_503)]


def _programme_json(programme: Any) -> dict[str, Any]:
    return programme.model_dump(mode="json")


# --------------------------------------------------------------------------- app

app = FastAPI(
    title="findMyDreamCollege API",
    version=__version__,
    description="Sourced programme evidence for a user-supplied profile (v1: India).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in get_settings().cors_origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        code = str(detail.get("code", "http_error"))
        message = str(detail.get("message", detail))
        workstream = detail.get("workstream")
    else:
        code = "http_error"
        message = str(detail)
        workstream = None
    return _error(exc.status_code, code, message, workstream=workstream)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(422, "validation_error", str(exc.errors()))


async def _database_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return _error(
        503,
        "database_unavailable",
        f"database unavailable: {exc}. Start Postgres with "
        "`docker compose -f infra/docker-compose.yml up -d db` and run `fmd migrate`.",
    )


app.add_exception_handler(DatabaseError, _database_error_handler)
app.add_exception_handler(psycopg.Error, _database_error_handler)


def require_admin(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
) -> None:
    """Single shared token; no user accounts in v1 (brief §2)."""
    expected = get_settings().admin_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "admin_disabled",
                "message": "admin endpoints are disabled: FMD_ADMIN_TOKEN is not set",
            },
        )
    supplied = x_admin_token
    if supplied is None and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized", "message": "invalid or missing admin token"},
        )


# --------------------------------------------------------------------------- WS-A reads


@app.get("/api/health")
def health() -> dict[str, Any]:
    registry = _registry()
    database_status = "down"
    try:
        settings = get_settings()
        probe = Database(settings.database_url, open_now=False)
        try:
            probe.open(timeout=1.0)
            database_status = "up" if probe.ping() else "down"
        finally:
            probe.close()
    except Exception:
        database_status = "down"

    payload: dict[str, Any] = {
        "status": "ok" if database_status == "up" and registry is not None else "degraded",
        "version": __version__,
        "database": database_status,
    }
    if registry is not None:
        payload["config"] = {
            "config_dir": str(registry.config_dir),
            "enabled_countries": registry.enabled_country_codes(),
            "sources_enabled": len(registry.enabled_sources()),
        }
    else:
        payload["config_error"] = "config registry failed to load; run `fmd llm-ping` or check logs"
    return _envelope(payload)


@app.get("/api/sources")
def list_sources() -> dict[str, Any]:
    registry = _registry()
    if registry is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "config_invalid", "message": "config registry failed to load"},
        )
    items = []
    for source in registry.sources.sources:
        items.append(
            {
                "id": source.id,
                "name": source.name,
                "base_url": source.base_url,
                "kind": source.kind,
                "format": source.format,
                "cadence_days": source.cadence_days,
                "trust": source.trust,
                "adapter": source.adapter,
                "enabled": source.enabled,
                "requires_manual_review": source.requires_manual_review,
                "notes": source.notes,
                # WS-B owns crawl telemetry; the field is explicit rather than omitted.
                "last_crawl": None,
            }
        )
    exclusions = [
        {"id": item.id, "name": item.name, "reason": item.reason}
        for item in registry.sources.exclusions
    ]
    return _envelope({"items": items, "exclusions": exclusions})


@app.get("/api/programmes")
def list_programmes(
    database: DatabaseDep,
    credential_level: str | None = Query(default=None),
    recognition_status: str | None = Query(default=None),
    state: str | None = Query(default=None),
    institution_id: int | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    filters = ProgrammeFilter(
        credential_level=credential_level,
        recognition_status=recognition_status,
        state=state,
        institution_id=institution_id,
        query=q,
        limit=limit,
        offset=offset,
    )
    repo = ProgrammesRepo(database)
    items = repo.list(filters)
    total = repo.count(filters)
    return _envelope(
        {
            "items": [_programme_json(item) for item in items],
            "total": total,
            "limit": limit,
            "offset": offset,
            "facets": repo.facets(filters),
            # Hard-filter exclusion histograms are WS-F (brief §12).
            "exclusions": {"available": False, "owner": "WS-F"},
        }
    )


@app.get("/api/programmes/{programme_id}")
def get_programme(
    programme_id: int,
    database: DatabaseDep,
) -> dict[str, Any]:
    programme = ProgrammesRepo(database).get(programme_id)
    if programme is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": f"programme {programme_id} not found"},
        )
    institution = EntitiesRepo(database).get(programme.institution_id)
    # §6 has no programme->entity link: programme-level claims are attached to the
    # awarding institution entity and disambiguated by claims.qualifier_json.
    claims = ClaimsRepo(database).list_for_entity(programme.institution_id)
    by_predicate: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        by_predicate.setdefault(claim.predicate, []).append(claim.model_dump(mode="json"))
    costs = CostItemsRepo(database).list_for_programme(programme_id)
    exams = ExamsRepo(database).list(country=institution.country if institution else None)
    score = ScoresRepo(database).latest_for_entity(programme.institution_id)

    payload = {
        "programme": _programme_json(programme),
        "institution": institution.model_dump(mode="json") if institution else None,
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "claims_by_predicate": by_predicate,
        "costs": [item.model_dump(mode="json") for item in costs],
        "exams": [exam.model_dump(mode="json") for exam in exams],
        "score": score.model_dump(mode="json") if score else None,
    }
    return _envelope(payload, claims_hash=_claims_hash(claims))


@app.get("/api/programmes/{programme_id}/evidence")
def programme_evidence(
    programme_id: int,
    database: DatabaseDep,
    predicate: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    programme = ProgrammesRepo(database).get(programme_id)
    if programme is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": f"programme {programme_id} not found"},
        )
    page = ClaimsRepo(database).evidence_page(
        entity_id=programme.institution_id,
        predicate=predicate,
        limit=limit,
        offset=offset,
    )
    return _envelope(
        {
            "items": [item.model_dump(mode="json") for item in page.items],
            "total": page.total,
            "limit": limit,
            "offset": offset,
        },
        claims_hash=_claims_hash(page.items),
    )


# --------------------------------------------------------------------------- other workstreams


def _not_implemented(endpoint: str, workstream: str, section: str) -> JSONResponse:
    return _error(
        501,
        "not_implemented",
        f"{endpoint} is not implemented ({workstream}, see brief §{section})",
        workstream=workstream,
    )


@app.get("/api/programmes/{programme_id}/cost")
def programme_cost(programme_id: int) -> JSONResponse:
    return _not_implemented("/api/programmes/{id}/cost", "WS-F", "13")


@app.get("/api/rankings")
def rankings() -> JSONResponse:
    return _not_implemented("/api/rankings", "WS-F", "11")


@app.get("/api/compare")
def compare() -> JSONResponse:
    return _not_implemented("/api/compare", "WS-F", "11")


@app.get("/api/exams")
def exams() -> JSONResponse:
    return _not_implemented("/api/exams", "WS-F", "12.5")


@app.get("/api/review/queue")
def review_queue() -> JSONResponse:
    return _not_implemented("/api/review/queue", "WS-G", "18.7")


@app.post("/api/review/claims/{claim_id}")
def review_claim(claim_id: int, _: None = Depends(require_admin)) -> JSONResponse:
    return _not_implemented("/api/review/claims/{id}", "WS-G", "18.7")


@app.post("/api/scoring/rerun")
def scoring_rerun(_: None = Depends(require_admin)) -> JSONResponse:
    return _not_implemented("/api/scoring/rerun", "WS-F", "11")
