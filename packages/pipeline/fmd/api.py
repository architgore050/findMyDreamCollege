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
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
from fmd.score.config import ScoringConfig
from fmd.settings import get_settings

EMPTY_CLAIMS_HASH = hashlib.sha256(b"").hexdigest()


# --------------------------------------------------------------------------- helpers


def _registry() -> Registry | None:
    try:
        return get_registry()
    except ConfigError:
        return None


def _meta(
    claims_hash: str | None = None, weights_version: str | None = None
) -> dict[str, Any]:
    registry = _registry()
    return {
        "scorer_version": registry.scorer_version if registry else "unknown",
        "weights_version": weights_version
        or (registry.weights_version if registry else "unknown"),
        "claims_hash": claims_hash or EMPTY_CLAIMS_HASH,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _envelope(
    payload: dict[str, Any],
    claims_hash: str | None = None,
    weights_version: str | None = None,
) -> dict[str, Any]:
    return {**payload, **_meta(claims_hash, weights_version)}


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


# --------------------------------------------------------------------------- WS-F engine
#
# Profiles arrive in the request body (the browser keeps them in localStorage)
# and are never persisted server-side. Weight overrides are validated here and
# never written back to config.


class RankingsRequest(BaseModel):
    profile: dict[str, Any]
    weights: dict[str, float] | None = None
    limit: int = Field(default=100, ge=1, le=500)


class ProfileRequest(BaseModel):
    profile: dict[str, Any]
    limit: int = Field(default=200, ge=1, le=500)


class ProgrammeCostRequest(BaseModel):
    """Profile travels in the body only (never in a URL, never persisted)."""

    profile: dict[str, Any]
    living: Literal["hostel", "independent"] = "hostel"


class ProgrammeFiltersRequest(BaseModel):
    profile: dict[str, Any]


class WeightOverrideError(ValueError):
    pass


def _resolved_scoring(
    registry: Registry, override: Mapping[str, float] | None
) -> tuple[ScoringConfig, str]:
    config = ScoringConfig.from_config(registry.weights)
    problems = config.validate()
    if problems:
        raise HTTPException(
            status_code=500,
            detail={"code": "config_invalid", "message": "; ".join(problems)},
        )
    if not override:
        return config, config.weights_version

    try:
        normalized = {str(key): float(value) for key, value in override.items()}
    except (TypeError, ValueError) as exc:
        raise WeightOverrideError(f"weights must be numbers: {exc}") from exc
    unknown = sorted(set(normalized) - set(config.weights))
    if unknown:
        raise WeightOverrideError(
            "unknown sub-score weights: " + ", ".join(unknown)
        )
    missing = sorted(set(config.weights) - set(normalized))
    if missing:
        raise WeightOverrideError(
            "weight override must cover every sub-score; missing: " + ", ".join(missing)
        )
    total = sum(normalized.values())
    if any(value < 0 for value in normalized.values()):
        raise WeightOverrideError("weight overrides must be non-negative")
    if abs(total - 100.0) > 1e-6:
        raise WeightOverrideError(f"weight overrides sum to {total:g}, expected 100")

    from dataclasses import replace

    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    version = f"{config.weights_version}+override.{digest}"
    return replace(config, weights=normalized, weights_version=version), version


def _programme_view(programme: Any, claims: Sequence[Any]) -> dict[str, Any]:
    payload = _programme_json(programme)
    payload["claims"] = claims
    return payload


def _claim_number(claims: Sequence[Any], *predicates: str) -> float | None:
    from fmd.score.evidence import ClaimIndex

    index = ClaimIndex.build(claims)
    for predicate in predicates:
        value = index.number(predicate)
        if value is not None:
            return value
    return None


def _positive_number(value: Any) -> float | None:
    """Parse a profile-supplied number; UNKNOWN/booleans/strings are not numbers."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if number > 0 else None


def _financing_dict(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    financing = profile.get("financing")
    return financing if isinstance(financing, Mapping) else {}


def _family_income(profile: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Resolve an annual family income *only* from an explicit point value.

    A profile may state ``financing.family_income_band`` instead. Turning a band
    into a midpoint would invent a number, so the gap stays UNKNOWN and the
    reason says exactly why. Bands themselves live in the profile schema/config,
    never as Python constants.
    """

    financing = _financing_dict(profile)
    for raw in (
        profile.get("family_income_annual"),
        financing.get("annual_family_income"),
        (profile.get("budget") if isinstance(profile.get("budget"), Mapping) else {}).get(
            "annual_family_income"
        ),
    ):
        value = _positive_number(raw)
        if value is not None:
            return value, None
    band = financing.get("family_income_band")
    if isinstance(band, str) and band.strip().lower() not in {"", "unknown"}:
        return None, (
            "income: the profile states a family-income band, not a point value; "
            "affordability_gap stays UNKNOWN rather than inventing a midpoint"
        )
    return None, "income: the profile does not supply annual family income"


def _financing_context(
    registry: Registry,
    claims: Sequence[Any],
    profile: Mapping[str, Any],
) -> tuple[Any, float | None, list[str]]:
    """Loan terms + explicit family income, with honest UNKNOWN reasons."""

    from fmd.cost import loan_terms_from_config

    reasons: list[str] = []
    loan = loan_terms_from_config(registry.cost_rules)
    rate = _claim_number(claims, "loan_interest_rate", "loan_rate", "annual_loan_rate")
    if rate is not None:
        if rate > 1.0:
            rate = rate / 100.0
        loan = loan.model_copy(
            update={"annual_rate": rate, "assumed_rate": False, "rate_source": "claim"}
        )
    if loan.annual_rate is None:
        reasons.append(
            "loan_rate: no published rate claim and no configured assumption; "
            "debt, EMI and DSR₂₅ stay UNKNOWN"
        )

    financing = _financing_dict(profile)
    loan_required = financing.get("loan_required")
    if isinstance(loan_required, str):
        loan_required = loan_required.strip().lower() not in {"no", "false", "unknown", ""}
    if loan_required is False:
        loan = loan.model_copy(update={"principal": 0.0})
    else:
        collateral = financing.get("collateral_available")
        if isinstance(collateral, str):
            collateral = collateral.strip().lower() not in {"unknown", "", "prefer_not_to_say"}
        if collateral is None:
            reasons.append(
                "collateral: the profile does not state collateral availability; "
                "lender sanction eligibility is not modelled and no approval is implied"
            )

    family_income, income_reason = _family_income(profile)
    if income_reason is not None:
        reasons.append(income_reason)
    return loan, family_income, reasons


def _cost_snapshot(
    database: Database,
    registry: Registry,
    programme: Any,
    claims: Sequence[Any],
    *,
    living: str,
    profile: Mapping[str, Any] | None = None,
) -> Any | None:
    """Build a cost snapshot from stored cost items; ``None`` when unmodellable."""

    from fmd.cost import (
        CostInputs,
        CostItem,
        CostModelError,
        CostRules,
        compute,
        student_work_from_config,
    )

    records = CostItemsRepo(database).list_for_programme(programme.id)
    if not records:
        return None
    items = [
        CostItem(
            category=record.category,
            label=record.label,
            amount=float(record.amount),
            currency=record.currency,
            period=record.period,
            method="extracted",
            confidence=float(record.confidence if record.confidence is not None else 1.0),
            basis=record.basis,
            source_claim_id=record.source_claim_id,
        )
        for record in records
    ]
    try:
        rules = CostRules.from_config(registry.cost_rules)
    except CostModelError:
        return None

    institution = EntitiesRepo(database).get(programme.institution_id)
    loan, family_income, _reasons = _financing_context(registry, claims, profile or {})

    kwargs: dict[str, Any] = {
        "items": items,
        "living": living,
        "loan": loan,
        "p25_first_year_salary": _claim_number(claims, "p25_first_year_salary"),
        "family_income_annual": family_income,
    }
    if programme.duration_months:
        kwargs["duration_months"] = programme.duration_months
    if institution is not None and institution.country:
        with suppress(CostModelError):
            kwargs["student_work"] = student_work_from_config(
                registry.cost_rules, institution.country
            )
    try:
        return compute(CostInputs(**kwargs), rules)
    except (CostModelError, ValueError):
        return None


def _score_row(
    database: Database,
    registry: Registry,
    config: ScoringConfig,
    programme: Any,
    claims: Sequence[Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    from fmd.filters import evaluate_all
    from fmd.score import score_programme

    view = _programme_view(programme, claims)
    summary = evaluate_all(view, profile)
    cost = _cost_snapshot(database, registry, programme, claims, living="hostel", profile=profile)
    result = score_programme(
        programme.id,
        view,
        profile,
        claims,
        config=config,
        cost=cost,
        cost_rules=registry.cost_rules,
        filter_decisions=list(summary.decisions.values()),
    )
    composite = result.composite.as_dict()
    return {
        "programme": _programme_json(programme),
        "score": result.as_dict(),
        "composite_band": {
            "point": composite["point"],
            "lo": composite["lo"],
            "hi": composite["hi"],
            "method": "weighted_composite",
            "bound": composite["bound"],
        },
        "evidence_tier": result.evidence_tier,
        "filters": summary.as_dict(),
        "excluded": bool(summary.excluded),
        "insights": None if result.insights is None else result.insights.as_dict(),
        "cost": None if cost is None else cost.render(),
    }


def _all_programmes(database: Database, limit: int) -> list[Any]:
    return ProgrammesRepo(database).list(ProgrammeFilter(limit=limit, offset=0))


def _claims_for(database: Database, programme: Any) -> list[Any]:
    return ClaimsRepo(database).list_for_entity(programme.institution_id)


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, float, float]:
        point = row["composite_band"]["point"]
        return (
            1 if row["excluded"] else 0,
            -(point if point is not None else -1.0),
            -row["composite_band"]["hi"],
        )

    return sorted(rows, key=key)


@app.post("/api/rankings")
def rankings(
    request: RankingsRequest,
    database: DatabaseDep,
) -> dict[str, Any]:
    registry = _registry()
    if registry is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "config_invalid", "message": "config registry failed to load"},
        )
    try:
        config, version = _resolved_scoring(registry, request.weights)
    except WeightOverrideError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_weights", "message": str(exc)},
        ) from exc

    programmes = _all_programmes(database, request.limit)
    rows: list[dict[str, Any]] = []
    all_claims: list[Any] = []
    for programme in programmes:
        claims = _claims_for(database, programme)
        all_claims.extend(claims)
        rows.append(_score_row(database, registry, config, programme, claims, request.profile))

    return _envelope(
        {
            "items": _sorted_rows(rows),
            "total": len(rows),
            "weights": dict(config.weights),
            "profile_persisted": False,
        },
        claims_hash=_claims_hash(all_claims),
        weights_version=version,
    )


def _programme_or_404(database: Database, programme_id: int) -> Any:
    programme = ProgrammesRepo(database).get(programme_id)
    if programme is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": f"programme {programme_id} not found"},
        )
    return programme


def _registry_or_500() -> Registry:
    registry = _registry()
    if registry is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "config_invalid", "message": "config registry failed to load"},
        )
    return registry


def _scenario_totals(snapshot: Any) -> dict[str, Any]:
    """Per-living-scenario metrics from one already-computed snapshot."""

    return {
        "living": snapshot.living,
        "living_cost": snapshot.scenarios[snapshot.living].model_dump(mode="json"),
        "tcoa": snapshot.tcoa.model_dump(mode="json"),
        "tcoa_first_year": snapshot.tcoa_first_year.model_dump(mode="json"),
        "net_price": snapshot.net_price.model_dump(mode="json"),
        "aid_total": snapshot.aid_total.model_dump(mode="json"),
        "loan_principal": snapshot.loan_principal.model_dump(mode="json"),
        "family_out_of_pocket": snapshot.family_out_of_pocket.model_dump(mode="json"),
        "student_work_offset": snapshot.student_work_offset.model_dump(mode="json"),
        "debt_at_repayment": snapshot.debt_at_repayment.model_dump(mode="json"),
        "emi": snapshot.emi.model_dump(mode="json"),
        "dsr25": snapshot.dsr25.model_dump(mode="json"),
        "affordability_gap": snapshot.affordability_gap.model_dump(mode="json"),
        "methods": dict(snapshot.methods),
        "assumptions": list(snapshot.assumptions),
        "assumed": dict(snapshot.assumptions_flags),
        "display": dict(snapshot.display),
    }


def _cost_unknown_reasons(
    claims: Sequence[Any],
    financing_reasons: Sequence[str],
) -> list[str]:
    reasons = list(financing_reasons)
    if _claim_number(claims, "p25_first_year_salary") is None:
        reasons.append(
            "salary: no P25 first-year salary claim for the placement city; "
            "DSR₂₅ stays UNKNOWN rather than assuming a salary"
        )
    return reasons


@app.get("/api/programmes/{programme_id}/cost")
def programme_cost_needs_profile(
    programme_id: int,
    database: DatabaseDep,
    living: str = Query(default="hostel"),
) -> dict[str, Any]:
    """GET is a pointer, not a number: family financing lives in the profile."""

    if living not in {"hostel", "independent"}:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_living_scenario",
                "message": "living must be 'hostel' or 'independent'",
            },
        )
    _programme_or_404(database, programme_id)
    programme = _programme_or_404(database, programme_id)
    claims = _claims_for(database, programme)
    return _envelope(
        {
            "programme_id": programme_id,
            "living": living,
            "available": False,
            "needs_profile": True,
            "profile_persisted": False,
            "reason": (
                "cost depends on the family financing profile; POST "
                "{profile, living} to this endpoint. No family-specific number is "
                "published without one."
            ),
            "scenarios": {},
            "scenario_totals": {},
            "unknown_reasons": [
                "profile: no profile was supplied, so DSR₂₅, the affordability gap and "
                "EMI cannot be computed"
            ],
        },
        claims_hash=_claims_hash(claims),
    )


@app.post("/api/programmes/{programme_id}/cost")
def programme_cost(
    programme_id: int,
    request: ProgrammeCostRequest,
    database: DatabaseDep,
) -> dict[str, Any]:
    programme = _programme_or_404(database, programme_id)
    registry = _registry_or_500()
    claims = _claims_for(database, programme)

    _loan, _income, financing_reasons = _financing_context(
        registry, claims, request.profile
    )
    reasons = _cost_unknown_reasons(claims, financing_reasons)

    snapshots: dict[str, Any] = {}
    for scenario in ("hostel", "independent"):
        snapshot = _cost_snapshot(
            database,
            registry,
            programme,
            claims,
            living=scenario,
            profile=request.profile,
        )
        if snapshot is not None:
            snapshots[scenario] = snapshot

    if not snapshots:
        return _envelope(
            {
                "programme_id": programme_id,
                "living": request.living,
                "available": False,
                "needs_profile": False,
                "profile_persisted": False,
                "reason": (
                    "no complete cost evidence; refusing to model a TCOA from nothing"
                ),
                "scenarios": {},
                "scenario_totals": {},
                "unknown_reasons": reasons,
            },
            claims_hash=_claims_hash(claims),
        )

    selected = snapshots.get(request.living) or next(iter(snapshots.values()))
    totals = {name: _scenario_totals(snapshot) for name, snapshot in snapshots.items()}
    selected_totals = totals[selected.living]

    assumed: dict[str, Any] = {}
    for block in totals.values():
        assumed.update(block["assumed"])
    if selected_totals["emi"]["method"] == "computed_emi":
        assumed.setdefault(
            "loan_terms",
            {
                "assumed": True,
                "source": "assumption",
                "note": (
                    "tenure and moratorium are modelled defaults, not published "
                    "sanction terms"
                ),
            },
        )

    payload: dict[str, Any] = {
        "programme_id": programme_id,
        "living": selected.living,
        "available": True,
        "needs_profile": False,
        "profile_persisted": False,
        "non_additive_scenarios": True,
        "note": "hostel and independent are complete alternatives; never sum them",
        "currency": selected.currency,
        "duration_months": selected.duration_months,
        # Living components per scenario (old shape, still complete alternatives).
        "scenarios": {
            name: selected.scenarios[name].model_dump(mode="json")
            for name in totals
            if name in selected.scenarios
        },
        # Full metrics per scenario: TCOA, EMI, DSR₂₅ and affordability each.
        "scenario_totals": totals,
        "assumed": assumed,
        "unknown_reasons": reasons,
        "fx_band_pct": selected.fx_band_pct,
        "bands": selected.bands,
        "dsr25_bands": selected.bands,
    }
    for key in (
        "tcoa",
        "tcoa_first_year",
        "net_price",
        "aid_total",
        "loan_principal",
        "family_out_of_pocket",
        "student_work_offset",
        "debt_at_repayment",
        "emi",
        "dsr25",
        "affordability_gap",
        "methods",
        "assumptions",
        "display",
    ):
        payload[key] = selected_totals[key]
    return _envelope(payload, claims_hash=_claims_hash(claims))


@app.get("/api/programmes/{programme_id}/filters")
def programme_filters_needs_profile(programme_id: int, database: DatabaseDep) -> dict[str, Any]:
    """No profile means no decisions: UNKNOWN is not a pass and not a guess."""

    _programme_or_404(database, programme_id)
    return _envelope(
        {
            "programme_id": programme_id,
            "needs_profile": True,
            "profile_persisted": False,
            "would_exclude": False,
            "order": [],
            "counts": {"pass": 0, "fail": 0, "unknown": 0},
            "filters": [],
            "excluded": [],
            "unverified": [],
            "reason": (
                "hard filters compare this programme against a profile; POST {profile} "
                "to this endpoint. No filter is evaluated without one."
            ),
        }
    )


@app.post("/api/programmes/{programme_id}/filters")
def programme_filters(
    programme_id: int,
    request: ProgrammeFiltersRequest,
    database: DatabaseDep,
) -> dict[str, Any]:
    from fmd.filters import FilterContext, evaluate_all

    programme = _programme_or_404(database, programme_id)
    registry = _registry_or_500()
    claims = _claims_for(database, programme)
    context = FilterContext.from_registry(registry)
    summary = evaluate_all(_programme_view(programme, claims), request.profile, context=context)
    decisions = [
        summary.decisions[filter_id].as_dict() for filter_id in summary.order
    ]
    return _envelope(
        {
            "programme_id": programme_id,
            "needs_profile": False,
            "profile_persisted": False,
            "order": list(summary.order),
            "counts": summary.counts,
            "filters": decisions,
            "would_exclude": bool(summary.excluded),
            "excluded": list(summary.excluded),
            "unverified": list(summary.unverified),
        },
        claims_hash=_claims_hash(claims),
    )


@app.get("/api/compare")
def compare_programmes(
    database: DatabaseDep,
    ids: str = Query(..., description="comma-separated programme ids"),
) -> dict[str, Any]:
    from fmd.score.aggregate import compare, pareto_matrix

    try:
        requested = [int(item) for item in ids.split(",") if item.strip()]
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_ids", "message": "ids must be comma-separated integers"},
        ) from exc
    requested = list(dict.fromkeys(requested))
    if len(requested) < 2:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_ids", "message": "at least two programme ids are required"},
        )
    registry = _registry()
    if registry is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "config_invalid", "message": "config registry failed to load"},
        )
    config, version = _resolved_scoring(registry, None)
    repo = ProgrammesRepo(database)
    rows: dict[str, dict[str, Any]] = {}
    programmes: dict[str, Any] = {}
    all_claims: list[Any] = []
    for programme_id in requested:
        programme = repo.get(programme_id)
        if programme is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "not_found", "message": f"programme {programme_id} not found"},
            )
        claims = _claims_for(database, programme)
        all_claims.extend(claims)
        view = _programme_view(programme, claims)
        from fmd.score import score_programme

        result = score_programme(
            programme.id, view, {}, claims, config=config, cost_rules=registry.cost_rules
        )
        programmes[str(programme_id)] = programme
        rows[str(programme_id)] = {
            key: value.as_dict() for key, value in result.sub_scores.items()
        }

    matrix = pareto_matrix(rows)
    comparisons: dict[str, Any] = {}
    for left in sorted(rows):
        for right in sorted(rows):
            if left >= right:
                continue
            verdict = compare(rows[left], rows[right])
            differing = [
                dimension
                for dimension in rows[left]
                if rows[left][dimension]["point"] is not None
                and rows[right][dimension]["point"] is not None
                and rows[left][dimension]["point"] != rows[right][dimension]["point"]
            ]
            comparisons[f"{left}|{right}"] = {
                "verdict": verdict,
                "differing_sub_scores": differing,
                "differing_claims": {
                    "left": {
                        dimension: [
                            component["claim_id"]
                            for component in rows[left][dimension]["components"]
                            if component["claim_id"] is not None
                        ]
                        for dimension in differing
                    },
                    "right": {
                        dimension: [
                            component["claim_id"]
                            for component in rows[right][dimension]["components"]
                            if component["claim_id"] is not None
                        ]
                        for dimension in differing
                    },
                },
            }

    return _envelope(
        {
            "ids": requested,
            "matrix": matrix,
            "comparisons": comparisons,
            "not_comparable_pairs": [
                key for key, item in comparisons.items() if item["verdict"] == "not_comparable"
            ],
        },
        claims_hash=_claims_hash(all_claims),
        weights_version=version,
    )


def _exam_status(
    *,
    now: datetime,
    closes: datetime | None,
    observed_at: datetime | None,
    ttl_days: int,
) -> tuple[str, bool, str]:
    if closes is None:
        return "unverified", False, "no registration window extracted"
    if observed_at is None:
        return "unverified", False, "window has no observed_at; cannot verify freshness"
    observed = observed_at
    if (now.tzinfo is None) != (observed.tzinfo is None):
        observed = observed.replace(tzinfo=UTC) if observed.tzinfo is None else observed
        now = now.replace(tzinfo=UTC) if now.tzinfo is None else now
    age_days = (now - observed).total_seconds() / 86_400.0
    if age_days > ttl_days:
        return "unverified", True, f"window observed {age_days:.1f}d ago exceeds TTL {ttl_days}d"
    close_at = closes
    if (now.tzinfo is None) != (close_at.tzinfo is None):
        close_at = close_at.replace(tzinfo=UTC) if close_at.tzinfo is None else close_at
    if now < close_at:
        return "open", False, f"registration closes {close_at.date().isoformat()}"
    return "closed", False, f"registration closed {close_at.date().isoformat()}"


@app.post("/api/exams")
def exam_windows(request: ProfileRequest) -> dict[str, Any]:
    registry = _registry()
    if registry is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "config_invalid", "message": "config registry failed to load"},
        )
    try:
        database: Database | None = get_database()
    except DatabaseError:
        database = None

    policy = registry.weights.filter_policy
    ttl = int(policy.stale_exam_window_ttl_days)
    now = datetime.now(UTC)
    profile = request.profile
    travel_pref = str(profile.get("travel_for_exams", "")).strip().lower()
    forbids_travel = travel_pref in {"no", "false", "none", "not_allowed", "avoids"}

    windows: dict[str, Any] = {}
    if database is not None:
        repo = ExamsRepo(database)
        for exam in registry.exams.exams:
            found = repo.list_windows(exam.id)
            if found:
                windows[exam.id] = found[0]

    items: list[dict[str, Any]] = []
    for exam in registry.exams.exams:
        window = windows.get(exam.id)
        closes = getattr(window, "registration_closes", None) if window else None
        observed = getattr(window, "observed_at", None) if window else None
        if window is None and exam.registration is not None:
            closes = _parse_exam_datetime(exam.registration.closes)
        status, stale, reason = _exam_status(
            now=now, closes=closes, observed_at=observed, ttl_days=ttl
        )
        requires_travel = bool(exam.requires_travel and not exam.remote_options)
        blocked = requires_travel and forbids_travel
        if blocked and status == "open":
            status = "closed"
            reason = f"{reason}; profile forbids required travel"
        items.append(
            {
                "exam_id": exam.id,
                "name": exam.name,
                "country": exam.country,
                "status": status,
                "observed_at": observed.isoformat() if observed else None,
                "stale": stale,
                "ttl_days": ttl,
                "requires_travel": requires_travel,
                "remote_options": bool(exam.remote_options),
                "travel_blocked": blocked,
                "deciding_claim_id": getattr(window, "source_claim_id", None)
                if window
                else None,
                "reason": reason,
            }
        )

    return _envelope(
        {
            "items": items,
            "counts": {
                "open": sum(1 for item in items if item["status"] == "open"),
                "closed": sum(1 for item in items if item["status"] == "closed"),
                "unverified": sum(1 for item in items if item["status"] == "unverified"),
            },
            "profile_persisted": False,
        }
    )


def _parse_exam_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("/", "-"))
    except ValueError:
        return None


@app.post("/api/exclusions")
def exclusions(
    request: ProfileRequest,
    database: DatabaseDep,
) -> dict[str, Any]:
    registry = _registry()
    if registry is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "config_invalid", "message": "config registry failed to load"},
        )
    config, version = _resolved_scoring(registry, None)
    from fmd.filters import evaluate_all

    programmes = _all_programmes(database, request.limit)
    histogram: dict[str, int] = {}
    unknown_histogram: dict[str, int] = {}
    excluded: list[dict[str, Any]] = []
    all_claims: list[Any] = []
    for programme in programmes:
        claims = _claims_for(database, programme)
        all_claims.extend(claims)
        summary = evaluate_all(_programme_view(programme, claims), request.profile)
        for filter_id, decision in summary.decisions.items():
            if decision.result == "fail":
                histogram[filter_id] = histogram.get(filter_id, 0) + 1
            elif decision.result == "unknown":
                unknown_histogram[filter_id] = unknown_histogram.get(filter_id, 0) + 1
        if summary.excluded:
            deciding = [
                summary.decisions[filter_id].as_dict() for filter_id in summary.excluded
            ]
            excluded.append(
                {
                    "programme": _programme_json(programme),
                    "deciding_filters": deciding,
                    "deciding_claim_ids": [
                        decision["claim_id"]
                        for decision in deciding
                        if decision["claim_id"] is not None
                    ],
                }
            )
    return _envelope(
        {
            "histogram": histogram,
            "unknown_histogram": unknown_histogram,
            "excluded": excluded,
            "excluded_total": len(excluded),
            "evaluated_total": len(programmes),
            "profile_persisted": False,
        },
        claims_hash=_claims_hash(all_claims),
        weights_version=version,
    )


# --------------------------------------------------------------------------- review (WS-G)


def _not_implemented(endpoint: str, workstream: str, section: str) -> JSONResponse:
    return _error(
        501,
        "not_implemented",
        f"{endpoint} is not implemented ({workstream}, see brief §{section})",
        workstream=workstream,
    )


@app.get("/api/review/queue")
def review_queue() -> JSONResponse:
    return _not_implemented("/api/review/queue", "WS-G", "18.7")


@app.post("/api/review/claims/{claim_id}")
def review_claim(claim_id: int, _: None = Depends(require_admin)) -> JSONResponse:
    return _not_implemented("/api/review/claims/{id}", "WS-G", "18.7")


@app.post("/api/scoring/rerun")
def scoring_rerun(
    database: DatabaseDep,
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    registry = _registry()
    if registry is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "config_invalid", "message": "config registry failed to load"},
        )
    from fmd.score.persist import persist_score

    config, version = _resolved_scoring(registry, None)
    programmes = _all_programmes(database, 500)
    written = 0
    skipped = 0
    from fmd.score import score_programme

    for programme in programmes:
        claims = _claims_for(database, programme)
        scored = score_programme(
            programme.id,
            _programme_view(programme, claims),
            {},
            claims,
            config=config,
            cost_rules=registry.cost_rules,
        )
        latest = ScoresRepo(database).latest_for_entity(programme.institution_id)
        if (
            latest is not None
            and latest.claims_hash == scored.claims_hash
            and latest.scorer_version == scored.scorer_version
            and latest.weights_version == scored.weights_version
        ):
            skipped += 1
            continue
        persist_score(database, scored)
        written += 1
    return _envelope(
        {"written": written, "skipped": skipped, "total": len(programmes)},
        weights_version=version,
    )
