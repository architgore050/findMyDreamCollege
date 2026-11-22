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
    AliasesRepo,
    ClaimCreate,
    ClaimsRepo,
    ClaimVerification,
    CostItemsRepo,
    DocumentsRepo,
    EntitiesRepo,
    ExamsRepo,
    ExtractionsRepo,
    ProgrammeFilter,
    ProgrammesRepo,
    ReviewAuditCreate,
    ReviewAuditRepo,
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


def _meta(claims_hash: str | None = None, weights_version: str | None = None) -> dict[str, Any]:
    registry = _registry()
    return {
        "scorer_version": registry.scorer_version if registry else "unknown",
        "weights_version": weights_version or (registry.weights_version if registry else "unknown"),
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


def _validation_message(exc: RequestValidationError) -> str:
    """Compact 422 message that never echoes the submitted payload.

    ``exc.errors()`` carries the offending ``input`` verbatim; a malformed body
    would otherwise reflect the caller's profile (family financing data) back
    into the response and any logs that capture it.
    """

    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        parts.append(f"{location or 'body'}: {error.get('msg', 'invalid')}")
    return "; ".join(parts) if parts else "request body failed validation"


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(422, "validation_error", _validation_message(exc))


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


@app.get("/api/methodology")
def methodology() -> JSONResponse:
    """PLAN v1-w9 (a): the trust surface, generated from the registry.

    Every number on this page is read from ``config/`` at request time —
    weights, versions, evidence-tier policy, predicate counts, source counts
    and the exclusion registry. Nothing is hand-copied: a change to
    ``config/weights.yaml`` changes this page (and its envelope version)
    without an edit here. The dispute path is documented in the payload so
    the UI and the API agree on it.
    """
    registry = _registry()
    if registry is None:
        return _error(503, "registry_unavailable", "config registry is not loaded")
    weights = registry.weights
    sources = list(registry.sources.sources)
    exclusions = [
        {"id": item.id, "name": item.name, "reason": item.reason}
        for item in registry.sources.exclusions
    ]
    by_trust: dict[str, int] = {}
    enabled_by_trust: dict[str, int] = {}
    for source in sources:
        trust = str(getattr(source, "trust", "unknown"))
        by_trust[trust] = by_trust.get(trust, 0) + 1
        if getattr(source, "enabled", False):
            enabled_by_trust[trust] = enabled_by_trust.get(trust, 0) + 1
    payload = {
        "methodology": {
            "weights": {name: float(value) for name, value in weights.weights.items()},
            "weights_sum": round(sum(float(v) for v in weights.weights.values()), 6),
            "component_policy": {
                "min_claims_for_score": weights.min_claims_for_score,
                "min_claims_for_confidence": weights.min_claims_for_confidence,
                "min_components_for_composite": weights.min_components_for_composite,
                "single_claim_interval_half_width": weights.single_claim_interval_half_width,
            },
            "evidence_tiers": {
                name: tier.model_dump(mode="json")
                for name, tier in weights.evidence_tiers.items()
            },
            "predicates_version": registry.predicates.version,
            "predicates_count": len(registry.predicates.predicates),
            "predicate_aliases": len(registry.predicates.aliases),
            "sources": {
                "total": len(sources),
                "enabled": sum(1 for s in sources if getattr(s, "enabled", False)),
                "by_trust": by_trust,
                "enabled_by_trust": enabled_by_trust,
            },
            "source_exclusions": exclusions,
            "value_policy": [
                "UNKNOWN is a value: a component with no usable claim is reported "
                "UNKNOWN, never imputed, never rendered as zero.",
                "Every published number carries the claim id that produced it; "
                "click through to the claim to see the verbatim quote, its span "
                "in the source document and the source file itself.",
                "Evidence tiers (well_evidenced / partial / thin / insufficient) "
                "are computed from n_claims and n_sources as configured in "
                "config/weights.yaml — they are policy, not judgement calls.",
                "Composite scores are published only when the configured minimum "
                "of components is known; otherwise the interval is reported with "
                "the unknown components listed.",
                "Excluded sources are recorded, with reasons, in an append-only "
                "registry: corrections are appended under a new id, history is "
                "never rewritten.",
            ],
            "dispute_path": [
                "Every rendered number links to the claim that produced it; the "
                "claim page shows the verbatim quote, its character span, the "
                "sha256 of the source snapshot and a link to the source document.",
                "To dispute a number, follow that link to the source document and "
                "compare it with the quote; then open an issue at "
                "https://github.com/architgore050/findMyDreamCollege/issues with "
                "the claim id — the claim is corrected or superseded (never "
                "silently rewritten) and the correction carries its own "
                "observed_at and evidence.",
                "Operators can supersede or reject claims in the review console "
                "(/review); superseded claims keep their history.",
            ],
        },
    }
    return _envelope(payload)


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
        raise WeightOverrideError("unknown sub-score weights: " + ", ".join(unknown))
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

    _loan, _income, financing_reasons = _financing_context(registry, claims, request.profile)
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
                "reason": ("no complete cost evidence; refusing to model a TCOA from nothing"),
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
                    "tenure and moratorium are modelled defaults, not published sanction terms"
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
    decisions = [summary.decisions[filter_id].as_dict() for filter_id in summary.order]
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
        rows[str(programme_id)] = {key: value.as_dict() for key, value in result.sub_scores.items()}

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
    precision: str | None = None,
    month_value: str | None = None,
) -> tuple[str, bool, str]:
    """The /api/exams verdict for one stored window.

    Mirrors the ``exam_registration_open`` filter's rules (the filter is
    authoritative; this projection exists so the exams page can explain
    itself): stale beyond TTL ⇒ unverified, a month-precision window is
    unverified inside the printed month rather than given a fabricated day,
    and a day-precision close never shifts the printed day (date-only closes
    stay open through the whole IST day; printed times are IST).
    """

    from fmd.filters.impl import _close_instant, _month_bounds

    if closes is None and month_value is None:
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
    if precision == "month":
        bounds = _month_bounds(month_value if month_value is not None else closes)
        if bounds is None:
            return (
                "unverified",
                False,
                f"month-precision window {month_value or closes!r} unparseable",
            )
        start, end = bounds
        if now >= end:
            return (
                "closed",
                False,
                f"registration closed by end of {month_value or start.date().isoformat()[:7]} "
                "(month precision)",
            )
        if now >= start:
            return (
                "unverified",
                False,
                "closes sometime in "
                f"{month_value or start.date().isoformat()[:7]} (month precision); "
                "no day is printed — verify with the authority",
            )
        return (
            "open",
            False,
            f"registration closes sometime in {month_value or start.date().isoformat()[:7]}, "
            "after now (month precision)",
        )
    close_at = _close_instant(closes)
    if close_at is None:
        return "unverified", False, "closing date unparseable"
    if now < close_at:
        return "open", False, f"registration closes {close_at.date().isoformat()}"
    return "closed", False, f"registration closed {close_at.date().isoformat()}"


def _profile_target_cycle(profile: Mapping[str, Any]) -> int | None:
    target = profile.get("target_intake_year")
    text = str(target) if target is not None else ""
    return int(text[:4]) if text[:4].isdigit() else None


def _window_claim_for(
    claims_by_id: Mapping[int, Any], row: Any
) -> tuple[str | None, str | None, Any]:
    """(precision, printed date_text, raw claim value) cited by a window row."""

    claim = claims_by_id.get(row.source_claim_id) if row.source_claim_id else None
    if claim is None:
        return None, None, None
    qualifier = claim.qualifier or {}
    return qualifier.get("precision"), qualifier.get("date_text"), claim.value


def _programmes_by_required_exam(database: Database) -> dict[str, list[int]]:
    """exam_id → programme ids whose extracted claims require that exam."""

    mapping: dict[str, set[int]] = {}
    repo = ProgrammesRepo(database)
    claims_repo = ClaimsRepo(database)
    for programme in repo.list(ProgrammeFilter(limit=1000, offset=0)):
        for claim in claims_repo.list_for_entity(programme.institution_id):
            if claim.predicate != "exam_required":
                continue
            exam_id = (claim.qualifier or {}).get("exam_id") or claim.value
            if exam_id:
                mapping.setdefault(str(exam_id), set()).add(programme.id)
    return {exam_id: sorted(ids) for exam_id, ids in sorted(mapping.items())}


def _deciding_window_row(
    rows: Sequence[Any], target_cycle: int | None
) -> Any | None:
    """The cycle that decides an exam for this profile: the target intake
    year's window when one exists, else the latest observed cycle."""

    if not rows:
        return None
    if target_cycle is not None:
        matching = [row for row in rows if row.cycle_year == target_cycle]
        if matching:
            rows = matching
    return max(
        rows,
        key=lambda row: (
            row.observed_at.isoformat() if row.observed_at is not None else "",
            row.cycle_year,
        ),
    )


def _deadline_sort_instant(
    closes: datetime | None, precision: str | None, month_value: str | None
) -> datetime | None:
    """Earliest instant at which the window may close — the act-before bound.

    Day precision: the close instant itself. Month precision: the month's
    first IST instant, because a window printed only as "March 2027" may close
    on the 1st; sorting by that bound never shows an already-maybe-closed
    window as the safe next action.
    """

    from fmd.filters.impl import _close_instant, _month_bounds

    if precision == "month":
        bounds = _month_bounds(month_value if month_value is not None else closes)
        return bounds[0] if bounds else None
    return _close_instant(closes)


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
    target_cycle = _profile_target_cycle(profile)

    windows_by_exam: dict[str, list[Any]] = {}
    claims_by_id: dict[int, Any] = {}
    programmes_by_exam: dict[str, list[int]] = {}
    if database is not None:
        repo = ExamsRepo(database)
        claims_repo = ClaimsRepo(database)
        for exam in registry.exams.exams:
            windows_by_exam[exam.id] = repo.list_windows(exam.id)
        for rows in windows_by_exam.values():
            for row in rows:
                if row.source_claim_id is not None and row.source_claim_id not in claims_by_id:
                    claim = claims_repo.get(int(row.source_claim_id))
                    if claim is not None:
                        claims_by_id[row.source_claim_id] = claim
        programmes_by_exam = _programmes_by_required_exam(database)

    items: list[dict[str, Any]] = []
    deadlines: list[dict[str, Any]] = []
    for exam in registry.exams.exams:
        rows = windows_by_exam.get(exam.id, [])
        cycles: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: item.cycle_year):
            precision, date_text, claim_value = _window_claim_for(claims_by_id, row)
            status, stale, reason = _exam_status(
                now=now,
                closes=row.registration_closes,
                observed_at=row.observed_at,
                ttl_days=ttl,
                precision=precision,
                month_value=claim_value if precision == "month" else None,
            )
            cycles.append(
                {
                    "cycle_year": row.cycle_year,
                    "registration_opens": (
                        row.registration_opens.isoformat() if row.registration_opens else None
                    ),
                    "registration_closes": (
                        row.registration_closes.isoformat() if row.registration_closes else None
                    ),
                    "sitting_dates": row.sitting_dates,
                    "observed_at": row.observed_at.isoformat() if row.observed_at else None,
                    "stale": stale,
                    "status": status,
                    "precision": precision,
                    "date_text": date_text,
                    "month_value": claim_value if precision == "month" else None,
                    "source_claim_id": row.source_claim_id,
                    "reason": reason,
                }
            )

        deciding = _deciding_window_row(rows, target_cycle)
        precision: str | None = None
        date_text: str | None = None
        month_value: str | None = None
        closes: datetime | None = None
        observed: datetime | None = None
        source_claim_id: int | None = None
        if deciding is not None:
            precision, date_text, claim_value = _window_claim_for(claims_by_id, deciding)
            if precision == "month":
                # Month precision: the stored projection may hold a boundary
                # datetime, but the claim only ever said the month. Never
                # publish a fabricated day — the month rides in month_value.
                month_value = claim_value if isinstance(claim_value, str) else None
            else:
                closes = deciding.registration_closes
            observed = deciding.observed_at
            source_claim_id = deciding.source_claim_id
        status, stale, reason = _exam_status(
            now=now,
            closes=closes,
            observed_at=observed,
            ttl_days=ttl,
            precision=precision,
            month_value=month_value,
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
                "scope": exam.scope,
                "status": status,
                "cycle_year": deciding.cycle_year if deciding is not None else None,
                "closes": closes.isoformat() if closes else None,
                "closes_text": date_text,
                "precision": precision,
                "sitting_dates": (
                    deciding.sitting_dates if deciding is not None else None
                ),
                "observed_at": observed.isoformat() if observed else None,
                "stale": stale,
                "ttl_days": ttl,
                "requires_travel": requires_travel,
                "remote_options": bool(exam.remote_options),
                "travel_blocked": blocked,
                "deciding_claim_id": source_claim_id,
                "reason": reason,
                "cycles": cycles,
                "programme_ids": programmes_by_exam.get(exam.id, []),
            }
        )

        # Next actionable deadlines: every cycle that can still be acted on
        # (open, or unverified-but-future — the action there is verification).
        if not blocked:
            for cycle in cycles:
                cycle_closes = (
                    _parse_exam_datetime(cycle["registration_closes"])
                    if cycle["registration_closes"]
                    else None
                )
                sort_at = _deadline_sort_instant(
                    cycle_closes, cycle["precision"], cycle["month_value"]
                )
                if sort_at is None or sort_at <= now or cycle["status"] == "closed":
                    continue
                deadlines.append(
                    {
                        "exam_id": exam.id,
                        "name": exam.name,
                        "cycle_year": cycle["cycle_year"],
                        "deadline": (
                            cycle_closes.isoformat()
                            if cycle_closes and cycle["precision"] != "month"
                            else None
                        ),
                        "deadline_month": (
                            cycle["month_value"] if cycle["precision"] == "month" else None
                        ),
                        "act_before": sort_at.isoformat(),
                        "precision": cycle["precision"],
                        "date_text": cycle["date_text"],
                        "status": cycle["status"],
                        "stale": cycle["stale"],
                        "claim_id": cycle["source_claim_id"],
                        "reason": cycle["reason"],
                    }
                )

    deadlines.sort(
        key=lambda entry: (
            entry["act_before"],
            entry["exam_id"],
            entry["cycle_year"],
        )
    )

    return _envelope(
        {
            "items": items,
            "counts": {
                "open": sum(1 for item in items if item["status"] == "open"),
                "closed": sum(1 for item in items if item["status"] == "closed"),
                "unverified": sum(1 for item in items if item["status"] == "unverified"),
            },
            "next_deadlines": deadlines,
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
            deciding = [summary.decisions[filter_id].as_dict() for filter_id in summary.excluded]
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
#
# The review console is the correction path for everything a machine decided on
# its own (brief §18.7). Three queues, three verbs, one audit log:
#
#   * claims   — accept | reject | correct. Accept/reject write only the
#     verification columns (0005); a correction never mutates content, it
#     inserts a superseding claim and points superseded_by at the old row
#     (AGENTS.md data rules).
#   * aliases  — confirm | reject. Confirming arms the ER manual-alias lock;
#     rejecting hides the suggestion from the resolution candidate universe.
#   * documents— link_entity | mark_not_applicable. Resolves a planner
#     abstention stored as plan_state='needs_review'.
#
# Every accepted request appends one review_audit row with decided_by='human',
# a timestamp and the previous state, so no machine error is unrecoverable.


REVIEW_DECIDER = "human"


def require_review_admin(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
) -> None:
    """Admin gate for the review surface (no user accounts, brief §2).

    Answering **403** (not the scoring-rerun gate's 401) is deliberate: there
    is no login flow to re-attempt with credentials, so a missing or wrong
    token is a final permission verdict — "this browser may not decide
    evidence" — and the console renders it as such. Like ``require_admin``,
    this resolves before any database dependency so an unauthorised caller
    never makes the API probe Postgres.
    """
    expected = get_settings().admin_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "admin_disabled",
                "message": "review endpoints are disabled: FMD_ADMIN_TOKEN is not set",
            },
        )
    supplied = x_admin_token
    if supplied is None and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "forbidden",
                "message": "review endpoints require the admin token (X-Admin-Token)",
            },
        )


class ClaimReviewActionRequest(BaseModel):
    action: Literal["accept", "reject", "correct"]
    corrected_quote: str | None = None
    note: str | None = None


class AliasReviewActionRequest(BaseModel):
    action: Literal["confirm", "reject"]
    note: str | None = None


class DocumentReviewActionRequest(BaseModel):
    action: Literal["link_entity", "mark_not_applicable"]
    entity_id: int | None = None
    note: str | None = None


def _review_notes(
    previous: dict[str, Any] | None,
    *,
    action: str,
    note: str | None,
    previous_status: str,
) -> dict[str, Any]:
    """Compose the verification_notes payload for one human decision.

    Machine notes are preserved under a key instead of overwritten: a reviewer
    who overturns the verifier must leave the verifier's reasoning readable.
    """
    payload: dict[str, Any] = {
        "human_review": {
            "action": action,
            "note": note,
            "decided_by": REVIEW_DECIDER,
            "decided_at": datetime.now(UTC).isoformat(),
            "previous_status": previous_status,
        }
    }
    if previous:
        payload["previous_verification"] = previous
    return payload


def _previous_claim_state(claim: Any) -> dict[str, Any]:
    return {
        "verification_status": claim.verification_status,
        "verified_by": claim.verified_by,
        "superseded_by": claim.superseded_by,
        "confidence": claim.confidence,
    }


def _queue_claim_items(database: Database, items: Sequence[Any]) -> list[dict[str, Any]]:
    """Claims joined to the snapshot that proves each one (console columns)."""
    repo = ClaimsRepo(database)
    enriched: list[dict[str, Any]] = []
    for claim in items:
        evidence = repo.evidence_with_provenance(claim.id)
        payload = (evidence or claim).model_dump(mode="json")
        enriched.append(payload)
    return enriched


def _queue_alias_items(database: Database, items: Sequence[Any]) -> list[dict[str, Any]]:
    entities = EntitiesRepo(database)
    enriched: list[dict[str, Any]] = []
    for alias in items:
        payload = alias.model_dump(mode="json")
        entity = entities.get(alias.entity_id)
        payload["entity_name"] = entity.canonical_name if entity else None
        enriched.append(payload)
    return enriched


@app.get("/api/review/queue")
def review_queue(
    _admin: Annotated[None, Depends(require_review_admin)],
    database: DatabaseDep,
    kind: Literal["claim", "alias", "document"] = Query(default="claim"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """One paginated adjudication queue; least-confident work first."""
    if kind == "claim":
        page = ClaimsRepo(database).review_queue(limit=limit, offset=offset)
        items = _queue_claim_items(database, page.items)
    elif kind == "alias":
        page = AliasesRepo(database).review_queue(limit=limit, offset=offset)
        items = _queue_alias_items(database, page.items)
    else:
        page = DocumentsRepo(database).review_queue(limit=limit, offset=offset)
        items = [item.model_dump(mode="json") for item in page.items]
    return _envelope(
        {
            "kind": kind,
            "items": items,
            "total": page.total,
            "limit": limit,
            "offset": offset,
            "ordering": (
                "confidence ASC" if kind == "claim" else "queue position (oldest pending first)"
            ),
        }
    )


@app.get("/api/review/audit")
def review_audit(
    _admin: Annotated[None, Depends(require_review_admin)],
    database: DatabaseDep,
    kind: Literal["claim", "alias", "document"] = Query(...),
    subject_id: int = Query(..., ge=1),
) -> dict[str, Any]:
    """The decision trail for one queue item, newest first."""
    audits = ReviewAuditRepo(database).list_for_subject(kind, subject_id)
    return _envelope(
        {
            "kind": kind,
            "subject_id": subject_id,
            "items": [audit.model_dump(mode="json") for audit in audits],
        }
    )


def _already_done(
    claim: Any,
    *,
    status: str,
) -> bool:
    return claim.verification_status == status and claim.verified_by == REVIEW_DECIDER


@app.post("/api/review/claims/{claim_id}")
def review_claim_action(
    claim_id: int,
    request: ClaimReviewActionRequest,
    _admin: Annotated[None, Depends(require_review_admin)],
    database: DatabaseDep,
) -> dict[str, Any]:
    repo = ClaimsRepo(database)
    claim = repo.get(claim_id)
    if claim is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": f"claim {claim_id} not found"},
        )

    if request.action == "correct":
        if claim.superseded_by is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "already_superseded",
                    "message": f"claim {claim_id} was already superseded by "
                    f"claim {claim.superseded_by}",
                },
            )
        if not request.corrected_quote or not request.corrected_quote.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_corrected_quote",
                    "message": "a correction must carry corrected_quote; the original "
                    "claim is never mutated",
                },
            )
        if not request.note:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_note",
                    "message": "a correction requires a note explaining what was wrong "
                    "and why the new quote is right",
                },
            )
        return _correct_claim(database, claim, request)

    target_status = "supported" if request.action == "accept" else "rejected"
    if claim.superseded_by is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "already_superseded",
                "message": f"claim {claim_id} was superseded by claim "
                f"{claim.superseded_by}; review the superseding claim instead",
            },
        )
    if request.action == "reject" and claim.verification_status == "supported" and not request.note:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "missing_note",
                "message": "rejecting a supported claim requires a note naming the "
                "reason; the verifier's verdict must not be overturned silently",
            },
        )
    if _already_done(claim, status=target_status):
        # Idempotent: a repeated click re-reads the queue and must not append
        # a second audit row for a decision that was already taken.
        return _envelope(
            {
                "claim": claim.model_dump(mode="json"),
                "action": request.action,
                "already_done": True,
                "audit_id": None,
            },
            claims_hash=repo.claims_hash(claim.entity_id),
        )

    audit_repo = ReviewAuditRepo(database)
    audit = audit_repo.record(
        ReviewAuditCreate(
            kind="claim",
            subject_id=claim_id,
            action=request.action,
            previous_state=_previous_claim_state(claim),
            details={"note": request.note},
        )
    )
    updated = repo.record_verification(
        claim_id,
        ClaimVerification(
            status=target_status,
            verified_by=REVIEW_DECIDER,
            notes=_review_notes(
                claim.verification_notes,
                action=request.action,
                note=request.note,
                previous_status=claim.verification_status,
            ),
        ),
    )
    return _envelope(
        {
            "claim": updated.model_dump(mode="json"),
            "action": request.action,
            "already_done": False,
            "audit_id": audit.id,
        },
        claims_hash=repo.claims_hash(claim.entity_id),
    )


def _correct_claim(
    database: Database,
    claim: Any,
    request: ClaimReviewActionRequest,
) -> dict[str, Any]:
    """Append a superseding claim with the corrected quote; touch nothing else.

    Invariant 3 (every published number resolves to a span) applies to the
    replacement too: the corrected quote must locate inside the extraction's
    text, otherwise there is no honest span and the correction is refused.
    """
    corrected = request.corrected_quote or ""
    extraction = ExtractionsRepo(database).get(claim.extraction_id)
    char_start: int | None = None
    if extraction is not None:
        located = extraction.text.find(corrected)
        if located >= 0:
            char_start = located
    if char_start is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unresolvable_corrected_quote",
                "message": "corrected_quote does not occur in the extraction text; a "
                "correction must stay resolvable to the same snapshot span",
            },
        )

    audit = ReviewAuditRepo(database).record(
        ReviewAuditCreate(
            kind="claim",
            subject_id=claim.id,
            action="correct",
            previous_state=_previous_claim_state(claim),
            details={
                "note": request.note,
                "corrected_quote": corrected,
                "superseded_by_created": True,
            },
        )
    )
    replacement, superseded = ClaimsRepo(database).correct_claim(
        claim.id,
        ClaimCreate(
            extraction_id=claim.extraction_id,
            entity_id=claim.entity_id,
            predicate=claim.predicate,
            value=claim.value,
            unit=claim.unit,
            qualifier=claim.qualifier,
            quote=corrected,
            char_start=char_start,
            char_end=char_start + len(corrected),
            confidence=1.0,
            observed_at=claim.observed_at,
            method=claim.method,
        ),
        verified_by=REVIEW_DECIDER,
        notes=_review_notes(
            claim.verification_notes,
            action="correct",
            note=request.note,
            previous_status=claim.verification_status,
        ),
    )
    return _envelope(
        {
            "claim": replacement.model_dump(mode="json"),
            "superseded_claim": superseded.model_dump(mode="json"),
            "action": "correct",
            "already_done": False,
            "audit_id": audit.id,
        },
        claims_hash=ClaimsRepo(database).claims_hash(claim.entity_id),
    )


def _alias_previous_state(alias: Any) -> dict[str, Any]:
    return {
        "review_status": alias.review_status,
        "confirmed_by": alias.confirmed_by,
        "alias_type": alias.alias_type,
    }


@app.post("/api/review/aliases/{alias_id}")
def review_alias_action(
    alias_id: int,
    request: AliasReviewActionRequest,
    _admin: Annotated[None, Depends(require_review_admin)],
    database: DatabaseDep,
) -> dict[str, Any]:
    repo = AliasesRepo(database)
    alias = repo.get(alias_id)
    if alias is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": f"alias {alias_id} not found"},
        )
    target_status = "confirmed" if request.action == "confirm" else "rejected"
    if alias.review_status == target_status:
        return _envelope(
            {
                "alias": alias.model_dump(mode="json"),
                "action": request.action,
                "already_done": True,
                "audit_id": None,
            }
        )
    # A decision is terminal: flipping it is a conflict, raised before any
    # audit row is written so the log only ever records applied actions.
    if alias.review_status in {"confirmed", "rejected"}:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "decision_conflict",
                "message": f"alias {alias_id} was already {alias.review_status}; the "
                "decision trail is in /api/review/audit",
            },
        )

    audit = ReviewAuditRepo(database).record(
        ReviewAuditCreate(
            kind="alias",
            subject_id=alias_id,
            action=request.action,
            previous_state=_alias_previous_state(alias),
            details={"note": request.note},
        )
    )
    updated = (
        repo.confirm(alias_id, confirmed_by=REVIEW_DECIDER)
        if request.action == "confirm"
        else repo.reject(alias_id)
    )
    return _envelope(
        {
            "alias": updated.model_dump(mode="json"),
            "action": request.action,
            "already_done": False,
            "audit_id": audit.id,
        }
    )


def _document_previous_state(document: Any) -> dict[str, Any]:
    return {
        "plan_state": document.plan_state,
        "entity_id": document.entity_id,
        "plan_confidence": document.plan_confidence,
    }


def _decision_conflict(document_id: int, state: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "decision_conflict",
            "message": f"document {document_id} was already decided ({state}); the "
            "decision trail is in /api/review/audit",
        },
    )


@app.post("/api/review/documents/{document_id}")
def review_document_action(
    document_id: int,
    request: DocumentReviewActionRequest,
    _admin: Annotated[None, Depends(require_review_admin)],
    database: DatabaseDep,
) -> dict[str, Any]:
    repo = DocumentsRepo(database)
    document = repo.get(document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": f"document {document_id} not found"},
        )
    note = request.note
    if request.action == "link_entity":
        if request.entity_id is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_entity",
                    "message": "link_entity requires entity_id; the console never guesses "
                    "an entity link",
                },
            )
        entity = EntitiesRepo(database).get(request.entity_id)
        if entity is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unknown_entity",
                    "message": f"entity {request.entity_id} does not exist",
                },
            )
        if document.plan_state == "linked" and document.entity_id == request.entity_id:
            return _envelope(
                {
                    "document": document.model_dump(mode="json"),
                    "action": request.action,
                    "already_done": True,
                    "audit_id": None,
                }
            )
        if document.plan_state == "not_applicable":
            raise _decision_conflict(document_id, "not_applicable")
    elif document.plan_state == "not_applicable":
        return _envelope(
            {
                "document": document.model_dump(mode="json"),
                "action": request.action,
                "already_done": True,
                "audit_id": None,
            }
        )
    elif document.plan_state == "linked":
        raise _decision_conflict(document_id, "linked")

    audit = ReviewAuditRepo(database).record(
        ReviewAuditCreate(
            kind="document",
            subject_id=document_id,
            action=request.action,
            previous_state=_document_previous_state(document),
            details={"note": note, "entity_id": request.entity_id},
        )
    )
    if request.action == "link_entity":
        assert request.entity_id is not None
        reason = f"resolved by human review (link_entity -> entity {request.entity_id})" + (
            f"; note: {note}" if note else ""
        )
        updated = repo.review_link_entity(document_id, request.entity_id, reason=reason)
    else:
        reason = "resolved by human review (mark_not_applicable)" + (
            f"; note: {note}" if note else ""
        )
        updated = repo.review_mark_not_applicable(document_id, reason=reason)

    if updated is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "decision_conflict",
                "message": f"document {document_id} is no longer in a state that allows "
                f"{request.action}; reload the queue",
            },
        )
    return _envelope(
        {
            "document": updated.model_dump(mode="json"),
            "action": request.action,
            "already_done": False,
            "audit_id": audit.id,
        }
    )


@app.post("/api/scoring/rerun")
def scoring_rerun(
    # Admin gate resolves before the database dependency: an unauthorised call
    # must answer 401/503 without first probing (or waiting on) Postgres.
    _admin: Annotated[None, Depends(require_admin)],
    database: DatabaseDep,
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
