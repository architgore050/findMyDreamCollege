"""Thin, typed repositories. The only place SQL strings live (besides migrations).

Repositories do not compute scores, validate spans or make product decisions;
they persist and read rows. Callers own transactions per method via the pooled
connection context.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from fmd.db.connection import Database


class RowModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _json(value: Any) -> Jsonb | None:
    return None if value is None else Jsonb(value)


# --------------------------------------------------------------------------- records


class CountrySeed(RowModel):
    code: str
    name: str
    enabled: bool
    currency: str
    tier: str
    languages: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] | None = None


class CountryRecord(CountrySeed):
    updated_at: datetime | None = None


class SourceSeed(RowModel):
    id: str
    name: str
    base_url: str | None = None
    kind: str
    format: str
    cadence_days: int
    trust: str
    adapter: str
    enabled: bool
    notes: str | None = None
    requires_manual_review: bool = False
    seeds: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class SourceRecord(SourceSeed):
    updated_at: datetime | None = None


class SourceExclusionSeed(RowModel):
    id: str
    name: str
    reason: str


class EntityCreate(RowModel):
    kind: Literal["institution", "programme", "hospital", "exam", "lender", "city"]
    canonical_name: str
    country: str | None = None
    state: str | None = None
    city: str | None = None
    status: str = "active"


class EntityRecord(EntityCreate):
    id: int
    created_at: datetime | None = None


class AliasCreate(RowModel):
    entity_id: int
    alias: str
    alias_type: Literal[
        "acronym", "former_name", "transliteration", "misspelling", "local_name"
    ]
    source_id: str | None = None
    confidence: float | None = 1.0


class AliasRecord(AliasCreate):
    id: int


class DocumentUpsert(RowModel):
    url: str
    domain: str
    kind: str | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    etag: str | None = None
    last_modified: datetime | None = None
    robots_decision: str | None = None
    robots_checked_at: datetime | None = None
    http_status: int | None = None


class DocumentRecord(DocumentUpsert):
    id: int


class SnapshotCreate(RowModel):
    document_id: int
    sha256: str
    bytes_path: str
    fetched_at: datetime | None = None
    fetcher_tier: str | None = None
    content_type: str | None = None
    headers: dict[str, Any] | None = None


class SnapshotRecord(SnapshotCreate):
    id: int


class ExtractionCreate(RowModel):
    snapshot_id: int
    tool: str
    tool_version: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    page_range: str | None = None
    text: str
    structure: dict[str, Any] | None = None


class ExtractionRecord(ExtractionCreate):
    id: int
    created_at: datetime | None = None


Method = Literal["extracted", "derived", "modelled", "user_provided"]


class ClaimCreate(RowModel):
    extraction_id: int
    entity_id: int
    predicate: str
    value: Any
    unit: str | None = None
    qualifier: dict[str, Any] | None = None
    quote: str
    char_start: int
    char_end: int
    confidence: float = 1.0
    observed_at: datetime | None = None
    method: Method = "extracted"


class ClaimRecord(ClaimCreate):
    id: int
    superseded_by: int | None = None


class ClaimEvidence(ClaimRecord):
    url: str
    fetched_at: datetime | None = None
    sha256: str
    snapshot_id: int


class ClaimEvidencePage(RowModel):
    items: list[ClaimEvidence]
    total: int


class ProgrammeCreate(RowModel):
    institution_id: int
    name: str
    credential_level: Literal["degree", "diploma", "lateral_entry", "certificate"]
    duration_months: int | None = None
    internship_months: float | None = None
    affiliated_university_id: int | None = None
    recognition_status: Literal[
        "recognised", "conditional", "unrecognised", "unknown"
    ] = "unknown"
    seats: int | None = None
    medium: str | None = None


class ProgrammeRecord(ProgrammeCreate):
    id: int
    created_at: datetime | None = None


class ProgrammeFilter(RowModel):
    credential_level: str | None = None
    recognition_status: str | None = None
    institution_id: int | None = None
    state: str | None = None
    query: str | None = None
    limit: int = 50
    offset: int = 0


class ScoreCreate(RowModel):
    entity_id: int
    scorer_version: str
    weights_version: str
    claims_hash: str
    sub_scores: dict[str, Any]
    evidence_tier: Literal["well_evidenced", "partial", "thin", "insufficient"]
    composite: float | None = None
    composite_lo: float | None = None
    composite_hi: float | None = None
    computed_at: datetime | None = None


class ScoreRecord(ScoreCreate):
    id: int


class ExamSeed(RowModel):
    id: str
    name: str
    country: str
    scope: Literal["national", "state", "institution"]
    provider: str
    registration_opens: datetime | None = None
    registration_closes: datetime | None = None
    earliest_sitting: date | None = None
    latest_sitting: date | None = None
    requires_travel: bool = True
    remote_options: bool = False
    fee_amount: Decimal | None = None
    currency: str | None = None
    source_claim_id: int | None = None


class ExamRecord(ExamSeed):
    pass


class ExamWindowCreate(RowModel):
    exam_id: str
    cycle_year: int
    registration_opens: datetime | None = None
    registration_closes: datetime | None = None
    sitting_dates: list[Any] | None = None
    source_claim_id: int | None = None
    observed_at: datetime | None = None


class ExamWindowRecord(ExamWindowCreate):
    id: int


class CostItemCreate(RowModel):
    programme_id: int
    category: Literal[
        "tuition",
        "hostel",
        "mess",
        "rent",
        "utilities",
        "food",
        "transport",
        "equipment",
        "exam_fee",
        "visa",
        "insurance",
        "misc",
    ]
    label: str
    amount: Decimal
    currency: str
    period: Literal["one_time", "per_year", "per_month"]
    basis: str | None = None
    source_claim_id: int | None = None
    confidence: float | None = None


class CostItemRecord(CostItemCreate):
    id: int


class FinancingOptionCreate(RowModel):
    scope: Literal["domestic", "overseas"]
    lender: str
    scheme: str
    rate_min: Decimal | None = None
    rate_max: Decimal | None = None
    collateral_rules: str | None = None
    max_amount: Decimal | None = None
    moratorium_months: int | None = None
    tenure_months: int | None = None
    subsidy_rules: str | None = None
    source_claim_id: int | None = None
    observed_at: datetime | None = None


class FinancingOptionRecord(FinancingOptionCreate):
    id: int


class LLMCallLog(RowModel):
    purpose: str
    prompt_sha256: str
    prompt_path: str | None
    model: str
    provider: str
    params: dict[str, Any] | None = None
    raw_response: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    cost_usd: float = 0.0
    cache_hit: bool = False


class LLMCallRecord(LLMCallLog):
    id: int
    created_at: datetime | None = None


class FrontierUpsert(RowModel):
    url: str
    domain: str
    priority: int = 0
    next_allowed_at: datetime | None = None
    state: Literal[
        "queued", "in_progress", "done", "failed", "quarantined"
    ] = "queued"


class FrontierRecord(FrontierUpsert):
    attempts: int = 0
    tier_used: str | None = None
    ban_score: float | None = None
    yield_score: float | None = None
    updated_at: datetime | None = None


# --------------------------------------------------------------------------- repos


class CountriesRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def upsert(self, seed: CountrySeed) -> CountryRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO countries
                    (code, name, enabled, currency, tier, languages_json, capabilities_json,
                     updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (code) DO UPDATE SET
                    name = EXCLUDED.name,
                    enabled = EXCLUDED.enabled,
                    currency = EXCLUDED.currency,
                    tier = EXCLUDED.tier,
                    languages_json = EXCLUDED.languages_json,
                    capabilities_json = EXCLUDED.capabilities_json,
                    updated_at = now()
                RETURNING *
                """,
                (
                    seed.code,
                    seed.name,
                    seed.enabled,
                    seed.currency,
                    seed.tier,
                    _json(seed.languages),
                    _json(seed.capabilities),
                ),
            ).fetchone()
        assert row is not None
        return _country(row)

    def get(self, code: str) -> CountryRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM countries WHERE code = %s", (code,)).fetchone()
        return _country(row) if row else None

    def list(self, enabled_only: bool = False) -> list[CountryRecord]:
        sql = "SELECT * FROM countries"
        if enabled_only:
            sql += " WHERE enabled"
        sql += " ORDER BY code"
        with self.db.connection() as conn:
            rows = conn.execute(sql).fetchall()
        return [_country(row) for row in rows]


class SourcesRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def upsert(self, seed: SourceSeed) -> SourceRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO sources
                    (id, name, base_url, kind, format, cadence_days, trust, adapter, enabled,
                     notes, requires_manual_review, seeds_json, config_json, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    base_url = EXCLUDED.base_url,
                    kind = EXCLUDED.kind,
                    format = EXCLUDED.format,
                    cadence_days = EXCLUDED.cadence_days,
                    trust = EXCLUDED.trust,
                    adapter = EXCLUDED.adapter,
                    enabled = EXCLUDED.enabled,
                    notes = EXCLUDED.notes,
                    requires_manual_review = EXCLUDED.requires_manual_review,
                    seeds_json = EXCLUDED.seeds_json,
                    config_json = EXCLUDED.config_json,
                    updated_at = now()
                RETURNING *
                """,
                (
                    seed.id,
                    seed.name,
                    seed.base_url,
                    seed.kind,
                    seed.format,
                    seed.cadence_days,
                    seed.trust,
                    seed.adapter,
                    seed.enabled,
                    seed.notes,
                    seed.requires_manual_review,
                    _json(seed.seeds),
                    _json(seed.config),
                ),
            ).fetchone()
        assert row is not None
        return _source(row)

    def get(self, source_id: str) -> SourceRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM sources WHERE id = %s", (source_id,)).fetchone()
        return _source(row) if row else None

    def list(self) -> list[SourceRecord]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM sources ORDER BY id").fetchall()
        return [_source(row) for row in rows]


class SourceExclusionsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def upsert(self, seed: SourceExclusionSeed) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO source_exclusions (id, name, reason, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, reason = EXCLUDED.reason, updated_at = now()
                """,
                (seed.id, seed.name, seed.reason),
            )


class EntitiesRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, create: EntityCreate) -> EntityRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO entities (kind, canonical_name, country, state, city, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    create.kind,
                    create.canonical_name,
                    create.country,
                    create.state,
                    create.city,
                    create.status,
                ),
            ).fetchone()
        assert row is not None
        return _entity(row)

    def get(self, entity_id: int) -> EntityRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM entities WHERE id = %s", (entity_id,)).fetchone()
        return _entity(row) if row else None

    def find(self, kind: str, canonical_name: str) -> EntityRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE kind = %s AND canonical_name = %s ORDER BY id LIMIT 1",
                (kind, canonical_name),
            ).fetchone()
        return _entity(row) if row else None

    def list(self, kind: str | None = None, limit: int = 100, offset: int = 0) -> list[EntityRecord]:
        if kind is None:
            sql = "SELECT * FROM entities ORDER BY id LIMIT %s OFFSET %s"
            params: tuple[Any, ...] = (limit, offset)
        else:
            sql = "SELECT * FROM entities WHERE kind = %s ORDER BY id LIMIT %s OFFSET %s"
            params = (kind, limit, offset)
        with self.db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_entity(row) for row in rows]


class AliasesRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, create: AliasCreate) -> AliasRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO entity_aliases
                    (entity_id, alias, alias_type, source_id, confidence)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    create.entity_id,
                    create.alias,
                    create.alias_type,
                    create.source_id,
                    create.confidence,
                ),
            ).fetchone()
        assert row is not None
        return AliasRecord.model_validate(row)

    def list_for_entity(self, entity_id: int) -> list[AliasRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM entity_aliases WHERE entity_id = %s ORDER BY id",
                (entity_id,),
            ).fetchall()
        return [AliasRecord.model_validate(row) for row in rows]


class DocumentsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def upsert(self, document: DocumentUpsert) -> DocumentRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO documents
                    (url, domain, kind, first_seen, last_seen, etag, last_modified,
                     robots_decision, robots_checked_at, http_status)
                VALUES (%s, %s, %s, COALESCE(%s, now()), COALESCE(%s, now()), %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    domain = EXCLUDED.domain,
                    kind = COALESCE(EXCLUDED.kind, documents.kind),
                    last_seen = COALESCE(EXCLUDED.last_seen, now()),
                    etag = COALESCE(EXCLUDED.etag, documents.etag),
                    last_modified = COALESCE(EXCLUDED.last_modified, documents.last_modified),
                    robots_decision = COALESCE(EXCLUDED.robots_decision, documents.robots_decision),
                    robots_checked_at = COALESCE(EXCLUDED.robots_checked_at,
                                                 documents.robots_checked_at),
                    http_status = COALESCE(EXCLUDED.http_status, documents.http_status)
                RETURNING *
                """,
                (
                    document.url,
                    document.domain,
                    document.kind,
                    document.first_seen,
                    document.last_seen,
                    document.etag,
                    document.last_modified,
                    document.robots_decision,
                    document.robots_checked_at,
                    document.http_status,
                ),
            ).fetchone()
        assert row is not None
        return _document(row)

    def get(self, document_id: int) -> DocumentRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = %s", (document_id,)
            ).fetchone()
        return _document(row) if row else None

    def get_by_url(self, url: str) -> DocumentRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE url = %s", (url,)).fetchone()
        return _document(row) if row else None

    def list_by_domain(self, domain: str, limit: int = 100) -> list[DocumentRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE domain = %s ORDER BY last_seen DESC NULLS LAST LIMIT %s",
                (domain, limit),
            ).fetchall()
        return [_document(row) for row in rows]


class SnapshotsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, snapshot: SnapshotCreate) -> SnapshotRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO snapshots
                    (document_id, sha256, bytes_path, fetched_at, fetcher_tier,
                     content_type, headers_json)
                VALUES (%s, %s, %s, COALESCE(%s, now()), %s, %s, %s)
                ON CONFLICT (document_id, sha256) DO UPDATE SET
                    bytes_path = EXCLUDED.bytes_path
                RETURNING *
                """,
                (
                    snapshot.document_id,
                    snapshot.sha256,
                    snapshot.bytes_path,
                    snapshot.fetched_at,
                    snapshot.fetcher_tier,
                    snapshot.content_type,
                    _json(snapshot.headers),
                ),
            ).fetchone()
        assert row is not None
        return _snapshot(row)

    def get(self, snapshot_id: int) -> SnapshotRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE id = %s", (snapshot_id,)
            ).fetchone()
        return _snapshot(row) if row else None

    def latest_for_document(self, document_id: int) -> SnapshotRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE document_id = %s ORDER BY fetched_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
        return _snapshot(row) if row else None

    def list_for_document(self, document_id: int) -> list[SnapshotRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE document_id = %s ORDER BY fetched_at DESC",
                (document_id,),
            ).fetchall()
        return [_snapshot(row) for row in rows]


class ExtractionsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, extraction: ExtractionCreate) -> ExtractionRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO extractions
                    (snapshot_id, tool, tool_version, model, prompt_version, page_range, text,
                     structure_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    extraction.snapshot_id,
                    extraction.tool,
                    extraction.tool_version,
                    extraction.model,
                    extraction.prompt_version,
                    extraction.page_range,
                    extraction.text,
                    _json(extraction.structure),
                ),
            ).fetchone()
        assert row is not None
        return _extraction(row)

    def get(self, extraction_id: int) -> ExtractionRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM extractions WHERE id = %s", (extraction_id,)
            ).fetchone()
        return _extraction(row) if row else None

    def list_for_snapshot(self, snapshot_id: int) -> list[ExtractionRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM extractions WHERE snapshot_id = %s ORDER BY id",
                (snapshot_id,),
            ).fetchall()
        return [_extraction(row) for row in rows]


class ClaimsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, claim: ClaimCreate) -> ClaimRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO claims
                    (extraction_id, entity_id, predicate, value_json, unit, qualifier_json,
                     quote, char_start, char_end, confidence, observed_at, method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    claim.extraction_id,
                    claim.entity_id,
                    claim.predicate,
                    _json(claim.value),
                    claim.unit,
                    _json(claim.qualifier),
                    claim.quote,
                    claim.char_start,
                    claim.char_end,
                    claim.confidence,
                    claim.observed_at,
                    claim.method,
                ),
            ).fetchone()
        assert row is not None
        return _claim(row)

    def get(self, claim_id: int) -> ClaimRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM claims WHERE id = %s", (claim_id,)).fetchone()
        return _claim(row) if row else None

    def list_for_entity(
        self,
        entity_id: int,
        predicate: str | None = None,
        include_superseded: bool = False,
        limit: int = 500,
        offset: int = 0,
    ) -> list[ClaimRecord]:
        conditions = ["entity_id = %s"]
        params: list[Any] = [entity_id]
        if predicate is not None:
            conditions.append("predicate = %s")
            params.append(predicate)
        if not include_superseded:
            conditions.append("superseded_by IS NULL")
        params.extend([limit, offset])
        sql = (
            "SELECT * FROM claims WHERE "
            + " AND ".join(conditions)
            + " ORDER BY observed_at DESC NULLS LAST, id DESC LIMIT %s OFFSET %s"
        )
        with self.db.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_claim(row) for row in rows]

    def evidence_page(
        self,
        entity_id: int,
        predicate: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ClaimEvidencePage:
        """Claims joined to the snapshot URL/timestamp/hash that proves them."""
        conditions = ["c.entity_id = %s", "c.superseded_by IS NULL"]
        params: list[Any] = [entity_id]
        if predicate is not None:
            conditions.append("c.predicate = %s")
            params.append(predicate)
        where = " AND ".join(conditions)
        count_sql = f"SELECT count(*) AS n FROM claims c WHERE {where}"  # noqa: S608
        page_sql = (
            "SELECT c.*, d.url, s.fetched_at, s.sha256, s.id AS snapshot_id "
            "FROM claims c "
            "JOIN extractions x ON x.id = c.extraction_id "
            "JOIN snapshots s ON s.id = x.snapshot_id "
            "JOIN documents d ON d.id = s.document_id "
            f"WHERE {where} "
            "ORDER BY c.observed_at DESC NULLS LAST, c.id DESC LIMIT %s OFFSET %s"
        )
        with self.db.connection() as conn:
            total_row = conn.execute(count_sql, tuple(params)).fetchone()
            rows = conn.execute(
                page_sql, tuple([*params, limit, offset])
            ).fetchall()
        items = [
            ClaimEvidence(
                **_claim(row).model_dump(),
                url=row["url"],
                fetched_at=row["fetched_at"],
                sha256=row["sha256"],
                snapshot_id=row["snapshot_id"],
            )
            for row in rows
        ]
        return ClaimEvidencePage(items=items, total=int(total_row["n"]) if total_row else 0)

    def supersede(self, old_claim_id: int, new_claim_id: int) -> ClaimRecord:
        """Point the old row at its replacement. This is the only claim UPDATE."""
        if old_claim_id == new_claim_id:
            raise ValueError("a claim cannot supersede itself")
        with self.db.connection() as conn:
            new_row = conn.execute(
                "SELECT entity_id FROM claims WHERE id = %s", (new_claim_id,)
            ).fetchone()
            if new_row is None:
                raise ValueError(f"replacement claim {new_claim_id} does not exist")
            old_row = conn.execute(
                "SELECT entity_id FROM claims WHERE id = %s", (old_claim_id,)
            ).fetchone()
            if old_row is None:
                raise ValueError(f"claim {old_claim_id} does not exist")
            if old_row["entity_id"] != new_row["entity_id"]:
                raise ValueError("cannot supersede a claim for a different entity")
            row = conn.execute(
                """
                UPDATE claims SET superseded_by = %s
                WHERE id = %s AND superseded_by IS NULL
                RETURNING *
                """,
                (new_claim_id, old_claim_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"claim {old_claim_id} is already superseded")
        return _claim(row)

    def count_for_entity(self, entity_id: int, predicate: str | None = None) -> int:
        if predicate is None:
            sql = "SELECT count(*) AS n FROM claims WHERE entity_id = %s AND superseded_by IS NULL"
            params: tuple[Any, ...] = (entity_id,)
        else:
            sql = (
                "SELECT count(*) AS n FROM claims WHERE entity_id = %s AND predicate = %s "
                "AND superseded_by IS NULL"
            )
            params = (entity_id, predicate)
        with self.db.connection() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0


class ProgrammesRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, programme: ProgrammeCreate) -> ProgrammeRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO programmes
                    (institution_id, name, credential_level, duration_months,
                     internship_months, affiliated_university_id, recognition_status, seats,
                     medium)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    programme.institution_id,
                    programme.name,
                    programme.credential_level,
                    programme.duration_months,
                    programme.internship_months,
                    programme.affiliated_university_id,
                    programme.recognition_status,
                    programme.seats,
                    programme.medium,
                ),
            ).fetchone()
        assert row is not None
        return _programme(row)

    def get(self, programme_id: int) -> ProgrammeRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM programmes WHERE id = %s", (programme_id,)
            ).fetchone()
        return _programme(row) if row else None

    def list_for_institution(self, institution_id: int) -> list[ProgrammeRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM programmes WHERE institution_id = %s ORDER BY id",
                (institution_id,),
            ).fetchall()
        return [_programme(row) for row in rows]

    def list(self, filters: ProgrammeFilter) -> list[ProgrammeRecord]:
        where, params = _programme_where(filters)
        params.extend([filters.limit, filters.offset])
        sql = (
            "SELECT p.* FROM programmes p JOIN entities e ON e.id = p.institution_id"
            + where
            + " ORDER BY p.id LIMIT %s OFFSET %s"
        )
        with self.db.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_programme(row) for row in rows]

    def count(self, filters: ProgrammeFilter) -> int:
        where, params = _programme_where(filters)
        sql = (
            "SELECT count(*) AS n FROM programmes p JOIN entities e ON e.id = p.institution_id"
            + where
        )
        with self.db.connection() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        return int(row["n"]) if row else 0

    def facets(self, filters: ProgrammeFilter) -> dict[str, dict[str, int]]:
        """Counts for filter chips; WS-F adds hard-filter exclusion histograms."""
        where, params = _programme_where(filters)
        base = (
            " FROM programmes p JOIN entities e ON e.id = p.institution_id"
            + where
        )
        facet_sql = {
            "credential_level": (
                "SELECT p.credential_level AS key, count(*) AS n" + base
                + " GROUP BY p.credential_level ORDER BY p.credential_level"
            ),
            "recognition_status": (
                "SELECT p.recognition_status AS key, count(*) AS n" + base
                + " GROUP BY p.recognition_status ORDER BY p.recognition_status"
            ),
            "state": (
                "SELECT e.state AS key, count(*) AS n" + base
                + " GROUP BY e.state ORDER BY e.state"
            ),
        }
        facets: dict[str, dict[str, int]] = {}
        with self.db.connection() as conn:
            for name, sql in facet_sql.items():
                rows = conn.execute(sql, tuple(params)).fetchall()
                facets[name] = {
                    (row["key"] if row["key"] is not None else "unknown"): int(row["n"])
                    for row in rows
                }
        return facets


def _programme_where(filters: ProgrammeFilter) -> tuple[str, list[Any]]:
    conditions: list[str] = []
    params: list[Any] = []
    if filters.credential_level:
        conditions.append("p.credential_level = %s")
        params.append(filters.credential_level)
    if filters.recognition_status:
        conditions.append("p.recognition_status = %s")
        params.append(filters.recognition_status)
    if filters.institution_id is not None:
        conditions.append("p.institution_id = %s")
        params.append(filters.institution_id)
    if filters.state:
        conditions.append("e.state = %s")
        params.append(filters.state)
    if filters.query:
        conditions.append("p.name ILIKE %s")
        params.append(f"%{filters.query}%")
    return (" WHERE " + " AND ".join(conditions) if conditions else ""), params


class ScoresRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, score: ScoreCreate) -> ScoreRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO scores
                    (entity_id, scorer_version, weights_version, claims_hash, computed_at,
                     sub_scores_json, composite, composite_lo, composite_hi, evidence_tier)
                VALUES (%s, %s, %s, %s, COALESCE(%s, now()), %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    score.entity_id,
                    score.scorer_version,
                    score.weights_version,
                    score.claims_hash,
                    score.computed_at,
                    _json(score.sub_scores),
                    score.composite,
                    score.composite_lo,
                    score.composite_hi,
                    score.evidence_tier,
                ),
            ).fetchone()
        assert row is not None
        return _score(row)

    def latest_for_entity(self, entity_id: int) -> ScoreRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM scores WHERE entity_id = %s ORDER BY computed_at DESC, id DESC LIMIT 1",
                (entity_id,),
            ).fetchone()
        return _score(row) if row else None

    def list_for_entity(self, entity_id: int) -> list[ScoreRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scores WHERE entity_id = %s ORDER BY computed_at DESC, id DESC",
                (entity_id,),
            ).fetchall()
        return [_score(row) for row in rows]


class ExamsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def upsert(self, exam: ExamSeed) -> ExamRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO exams
                    (id, name, country, scope, provider, registration_opens,
                     registration_closes, earliest_sitting, latest_sitting, requires_travel,
                     remote_options, fee_amount, currency, source_claim_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    country = EXCLUDED.country,
                    scope = EXCLUDED.scope,
                    provider = EXCLUDED.provider,
                    requires_travel = EXCLUDED.requires_travel,
                    remote_options = EXCLUDED.remote_options,
                    fee_amount = COALESCE(EXCLUDED.fee_amount, exams.fee_amount),
                    currency = COALESCE(EXCLUDED.currency, exams.currency)
                RETURNING *
                """,
                (
                    exam.id,
                    exam.name,
                    exam.country,
                    exam.scope,
                    exam.provider,
                    exam.registration_opens,
                    exam.registration_closes,
                    exam.earliest_sitting,
                    exam.latest_sitting,
                    exam.requires_travel,
                    exam.remote_options,
                    exam.fee_amount,
                    exam.currency,
                    exam.source_claim_id,
                ),
            ).fetchone()
        assert row is not None
        return ExamRecord.model_validate(row)

    def get(self, exam_id: str) -> ExamRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM exams WHERE id = %s", (exam_id,)).fetchone()
        return ExamRecord.model_validate(row) if row else None

    def list(self, country: str | None = None) -> list[ExamRecord]:
        if country is None:
            sql, params = "SELECT * FROM exams ORDER BY id", ()
        else:
            sql, params = "SELECT * FROM exams WHERE country = %s ORDER BY id", (country,)
        with self.db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [ExamRecord.model_validate(row) for row in rows]

    def upsert_window(self, window: ExamWindowCreate) -> ExamWindowRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO exam_windows
                    (exam_id, cycle_year, registration_opens, registration_closes,
                     sitting_dates_json, source_claim_id, observed_at)
                VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, now()))
                ON CONFLICT (exam_id, cycle_year) DO UPDATE SET
                    registration_opens = EXCLUDED.registration_opens,
                    registration_closes = EXCLUDED.registration_closes,
                    sitting_dates_json = EXCLUDED.sitting_dates_json,
                    source_claim_id = EXCLUDED.source_claim_id,
                    observed_at = EXCLUDED.observed_at
                RETURNING *
                """,
                (
                    window.exam_id,
                    window.cycle_year,
                    window.registration_opens,
                    window.registration_closes,
                    _json(window.sitting_dates),
                    window.source_claim_id,
                    window.observed_at,
                ),
            ).fetchone()
        assert row is not None
        return ExamWindowRecord.model_validate(row)

    def list_windows(self, exam_id: str) -> list[ExamWindowRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM exam_windows WHERE exam_id = %s ORDER BY cycle_year DESC",
                (exam_id,),
            ).fetchall()
        return [ExamWindowRecord.model_validate(row) for row in rows]


class CostItemsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, item: CostItemCreate) -> CostItemRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO cost_items
                    (programme_id, category, label, amount, currency, period, basis,
                     source_claim_id, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    item.programme_id,
                    item.category,
                    item.label,
                    item.amount,
                    item.currency,
                    item.period,
                    item.basis,
                    item.source_claim_id,
                    item.confidence,
                ),
            ).fetchone()
        assert row is not None
        return CostItemRecord.model_validate(row)

    def list_for_programme(self, programme_id: int) -> list[CostItemRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM cost_items WHERE programme_id = %s ORDER BY id",
                (programme_id,),
            ).fetchall()
        return [CostItemRecord.model_validate(row) for row in rows]


class FinancingOptionsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, option: FinancingOptionCreate) -> FinancingOptionRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO financing_options
                    (scope, lender, scheme, rate_min, rate_max, collateral_rules, max_amount,
                     moratorium_months, tenure_months, subsidy_rules, source_claim_id,
                     observed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()))
                RETURNING *
                """,
                (
                    option.scope,
                    option.lender,
                    option.scheme,
                    option.rate_min,
                    option.rate_max,
                    option.collateral_rules,
                    option.max_amount,
                    option.moratorium_months,
                    option.tenure_months,
                    option.subsidy_rules,
                    option.source_claim_id,
                    option.observed_at,
                ),
            ).fetchone()
        assert row is not None
        return FinancingOptionRecord.model_validate(row)

    def list(self, scope: str | None = None) -> list[FinancingOptionRecord]:
        if scope is None:
            sql, params = "SELECT * FROM financing_options ORDER BY id", ()
        else:
            sql, params = (
                "SELECT * FROM financing_options WHERE scope = %s ORDER BY id",
                (scope,),
            )
        with self.db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [FinancingOptionRecord.model_validate(row) for row in rows]


class LLMCallsRepo:
    """Audit sink implementing the router's ``LLMCallSink`` protocol."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def record(self, call: LLMCallLog) -> int:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO llm_calls
                    (purpose, prompt_sha256, prompt_path, model, provider, params_json,
                     raw_response, tokens_in, tokens_out, latency_ms, cost_usd, cache_hit)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    call.purpose,
                    call.prompt_sha256,
                    call.prompt_path,
                    call.model,
                    call.provider,
                    _json(call.params),
                    call.raw_response,
                    call.tokens_in,
                    call.tokens_out,
                    call.latency_ms,
                    call.cost_usd,
                    call.cache_hit,
                ),
            ).fetchone()
        return int(row["id"]) if row else 0

    def list_for_purpose(self, purpose: str, limit: int = 100) -> list[LLMCallRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_calls WHERE purpose = %s ORDER BY created_at DESC, id DESC LIMIT %s",
                (purpose, limit),
            ).fetchall()
        return [LLMCallRecord.model_validate(row) for row in rows]

    def spend_since(self, since: datetime) -> Decimal:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(sum(cost_usd), 0) AS total FROM llm_calls WHERE created_at >= %s",
                (since,),
            ).fetchone()
        return Decimal(row["total"]) if row else Decimal(0)


class CrawlFrontierRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def enqueue(self, item: FrontierUpsert) -> FrontierRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO crawl_frontier (url, domain, priority, next_allowed_at, state)
                VALUES (%s, %s, %s, COALESCE(%s, now()), %s)
                ON CONFLICT (url) DO UPDATE SET
                    domain = EXCLUDED.domain,
                    priority = EXCLUDED.priority,
                    next_allowed_at = LEAST(crawl_frontier.next_allowed_at,
                                            EXCLUDED.next_allowed_at),
                    state = CASE WHEN crawl_frontier.state = 'done'
                                 THEN crawl_frontier.state ELSE EXCLUDED.state END,
                    updated_at = now()
                RETURNING *
                """,
                (item.url, item.domain, item.priority, item.next_allowed_at, item.state),
            ).fetchone()
        assert row is not None
        return FrontierRecord.model_validate(row)

    def get(self, url: str) -> FrontierRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM crawl_frontier WHERE url = %s", (url,)
            ).fetchone()
        return FrontierRecord.model_validate(row) if row else None

    def list_due(self, at: datetime, limit: int = 50) -> list[FrontierRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM crawl_frontier
                WHERE state = 'queued' AND next_allowed_at <= %s
                ORDER BY priority DESC, next_allowed_at ASC LIMIT %s
                """,
                (at, limit),
            ).fetchall()
        return [FrontierRecord.model_validate(row) for row in rows]

    def set_state(
        self,
        url: str,
        state: Literal["queued", "in_progress", "done", "failed", "quarantined"],
        tier_used: str | None = None,
        attempts: int | None = None,
    ) -> FrontierRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE crawl_frontier SET
                    state = %s,
                    tier_used = COALESCE(%s, tier_used),
                    attempts = COALESCE(%s, attempts),
                    updated_at = now()
                WHERE url = %s
                RETURNING *
                """,
                (state, tier_used, attempts, url),
            ).fetchone()
        return FrontierRecord.model_validate(row) if row else None

    def count_by_state(self) -> dict[str, int]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT state, count(*) AS n FROM crawl_frontier GROUP BY state"
            ).fetchall()
        return {row["state"]: int(row["n"]) for row in rows}


# --------------------------------------------------------------------------- mappers


def _country(row: dict[str, Any]) -> CountryRecord:
    return CountryRecord(
        code=row["code"],
        name=row["name"],
        enabled=row["enabled"],
        currency=row["currency"],
        tier=row["tier"],
        languages=row["languages_json"] or [],
        capabilities=row["capabilities_json"],
        updated_at=row["updated_at"],
    )


def _source(row: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        id=row["id"],
        name=row["name"],
        base_url=row["base_url"],
        kind=row["kind"],
        format=row["format"],
        cadence_days=row["cadence_days"],
        trust=row["trust"],
        adapter=row["adapter"],
        enabled=row["enabled"],
        notes=row["notes"],
        requires_manual_review=row["requires_manual_review"],
        seeds=row["seeds_json"] or [],
        config=row["config_json"] or {},
        updated_at=row["updated_at"],
    )


def _entity(row: dict[str, Any]) -> EntityRecord:
    return EntityRecord.model_validate(row)


def _document(row: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord.model_validate(row)


def _snapshot(row: dict[str, Any]) -> SnapshotRecord:
    return SnapshotRecord(
        document_id=row["document_id"],
        sha256=row["sha256"],
        bytes_path=row["bytes_path"],
        fetched_at=row["fetched_at"],
        fetcher_tier=row["fetcher_tier"],
        content_type=row["content_type"],
        headers=row["headers_json"],
        id=row["id"],
    )


def _extraction(row: dict[str, Any]) -> ExtractionRecord:
    return ExtractionRecord(
        snapshot_id=row["snapshot_id"],
        tool=row["tool"],
        tool_version=row["tool_version"],
        model=row["model"],
        prompt_version=row["prompt_version"],
        page_range=row["page_range"],
        text=row["text"],
        structure=row["structure_json"],
        id=row["id"],
        created_at=row["created_at"],
    )


def _claim(row: dict[str, Any]) -> ClaimRecord:
    return ClaimRecord(
        extraction_id=row["extraction_id"],
        entity_id=row["entity_id"],
        predicate=row["predicate"],
        value=row["value_json"],
        unit=row["unit"],
        qualifier=row["qualifier_json"],
        quote=row["quote"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        confidence=row["confidence"],
        observed_at=row["observed_at"],
        method=row["method"],
        superseded_by=row["superseded_by"],
        id=row["id"],
    )


def _programme(row: dict[str, Any]) -> ProgrammeRecord:
    return ProgrammeRecord.model_validate(row)


def _score(row: dict[str, Any]) -> ScoreRecord:
    return ScoreRecord(
        entity_id=row["entity_id"],
        scorer_version=row["scorer_version"],
        weights_version=row["weights_version"],
        claims_hash=row["claims_hash"],
        sub_scores=row["sub_scores_json"],
        composite=row["composite"],
        composite_lo=row["composite_lo"],
        composite_hi=row["composite_hi"],
        evidence_tier=row["evidence_tier"],
        computed_at=row["computed_at"],
        id=row["id"],
    )
