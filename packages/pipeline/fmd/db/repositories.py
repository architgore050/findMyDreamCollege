"""Thin, typed repositories. The only place SQL strings live (besides migrations).

Repositories do not compute scores, validate spans or make product decisions;
they persist and read rows. Callers own transactions per method via the pooled
connection context.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fmd.db.connection import Database, DatabaseError


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
    normalised_name: str | None = None
    blocking_key: str | None = None
    # Inventory facts (migration 0004): the website the institution publishes
    # and the type band the inventory source states. Both stay NULL when the
    # source is silent — never guessed.
    website_url: str | None = None
    institution_type: str | None = None


class EntityRecord(EntityCreate):
    id: int
    merged_into_entity_id: int | None = None
    merged_at: datetime | None = None
    created_at: datetime | None = None


class AliasCreate(RowModel):
    entity_id: int
    alias: str
    alias_type: Literal[
        "acronym", "former_name", "transliteration", "misspelling", "local_name", "manual"
    ]
    source_id: str | None = None
    confidence: float | None = 1.0
    span_start: int | None = None
    span_end: int | None = None
    quote: str | None = None
    source_document_id: int | None = None
    source_extraction_id: int | None = None
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None


AliasReviewStatus = Literal["pending", "confirmed", "rejected"]


class AliasRecord(AliasCreate):
    id: int
    # Review-console state machine (migration 0008). Pending is the default for
    # every mined suggestion; only a human action moves it.
    review_status: AliasReviewStatus = "pending"


class ERAuditCreate(RowModel):
    """One append-only merge/non-merge decision (brief §10)."""

    subject_entity_id: int
    candidate_entity_id: int
    band: Literal["accept", "reject", "adjudicate"]
    score: float
    features: dict[str, Any] = Field(default_factory=dict)
    thresholds_version: str
    model: str | None = None
    model_verdict: str | None = None
    model_reason: str | None = None
    decided_by: Literal["auto", "model", "human"]
    claims_hash: str | None = None


class ERAuditRecord(ERAuditCreate):
    id: int
    created_at: datetime | None = None


@dataclass(frozen=True)
class UnmergeResult:
    """Outcome of :meth:`EntitiesRepo.unmerge`.

    A conflict is a *result*, not an exception: nothing is moved and ``drift``
    lists every piece of evidence that no longer matches what the merge moved.
    The pair is iterable as ``(entity, audit)`` so existing callers that unpack
    two values keep working.
    """

    entity: EntityRecord
    audit: ERAuditRecord | None = None
    conflict: bool = False
    drift: tuple[str, ...] = ()
    moved_back: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.conflict

    def __iter__(self):
        yield self.entity
        yield self.audit


DocumentRole = Literal["inventory", "navigation", "evidence"]


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
    http_method: Literal["GET", "POST"] = "GET"
    http_body: dict[str, Any] | None = None
    request_headers: dict[str, Any] | None = None

    # Crawl/extract plan, persisted as columns (migration 0003). ``plan_state`` is
    # planned | linked | needs_review | abstained; a null ``entity_id`` with state
    # ``needs_review`` is an abstention a human must resolve, never a guess.
    source_id: str | None = None
    extraction_family: str | None = None
    entity_id: int | None = None
    plan_state: (
        Literal["planned", "linked", "needs_review", "abstained", "not_applicable"] | None
    ) = None
    plan_confidence: float | None = None
    plan_reason: str | None = None
    # What this document *is* (migration 0004). ``None`` on an update means
    # "leave the stored role alone"; a fresh insert becomes ``evidence``.
    role: DocumentRole | None = None

    # Dedupe decision (migration 0007). Written after the decision is made so a
    # dropped document keeps the justification even though it has no snapshot.
    text_sha256: str | None = None
    dedupe_mode: Literal["exact", "simhash", "identity"] | None = None
    dedupe_basis: Literal["raw_bytes", "extracted_text", "identity"] | None = None
    dedupe_similarity: float | None = None
    dedupe_reason: str | None = None
    dedupe_compared_document_id: int | None = None
    dedupe_simhash: str | None = None

    @classmethod
    def from_document_ref(cls, ref: Any, *, domain: str | None = None) -> DocumentUpsert:
        """Accept a ``DocumentRef`` (keeping POST shape from ``metadata``).

        WS-C adapters put ``request_method`` / ``request_body`` /
        ``request_headers`` in ``DocumentRef.metadata``. Persisting that shape is
        what makes an admin-ajax POST source replayable.
        """
        metadata = getattr(ref, "metadata", None) or {}
        method = str(metadata.get("request_method") or metadata.get("http_method") or "GET").upper()
        if method not in {"GET", "POST"}:
            raise ValueError(f"unsupported request method {method!r} (GET or POST)")
        body = metadata.get("request_body", metadata.get("http_body"))
        headers = metadata.get("request_headers") or metadata.get("headers")
        host = domain
        if host is None:
            from urllib.parse import urlsplit

            host = (urlsplit(getattr(ref, "url", "")).hostname or "").lower()
        if not host:
            raise ValueError(f"cannot derive a domain for {getattr(ref, 'url', ref)!r}")
        return cls(
            url=ref.url,
            domain=host,
            source_id=getattr(ref, "source_id", None),
            http_method=method,
            http_body=body if isinstance(body, dict) else None,
            request_headers=headers if isinstance(headers, dict) else None,
        )


class DocumentRecord(DocumentUpsert):
    id: int


class InstitutionDomainSeed(RowModel):
    """A registrable domain discovered from an inventory website field."""

    domain: str
    seed_url: str
    entity_id: int | None = None
    discovered_by: str | None = None
    priority: int = 0
    reason: str | None = None


class InstitutionDomainRecord(InstitutionDomainSeed):
    state: Literal["pending", "queued", "planned", "skipped"] = "pending"
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    last_queued_at: datetime | None = None


class SnapshotCreate(RowModel):
    document_id: int
    sha256: str
    bytes_path: str
    fetched_at: datetime | None = None
    fetcher_tier: str | None = None
    content_type: str | None = None
    headers: dict[str, Any] | None = None
    #: sha256 of the normalised text the dedupe decision used (migration 0007).
    text_sha256: str | None = None
    #: Which bytes the decision was made over: raw_bytes | extracted_text | identity.
    dedupe_basis: str | None = None


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


class SectionNodeCreate(RowModel):
    """One heading of a document's section tree (W8 section-structure).

    ``node_key`` is the 1-based ordinal in document order (stable across
    re-derivations of the same text); ``parent_key`` refers to the parent's
    ``node_key`` and is resolved to a database id on insert.
    """

    node_key: int
    level: int
    title: str
    char_start: int
    char_end: int
    heading_char_start: int
    heading_char_end: int
    page_no: int | None = None
    bbox: list[float] | None = None
    parent_key: int | None = None


class SectionTreeWrite(RowModel):
    """A whole-tree write: sections are derived structure, replaced wholesale."""

    document_id: int
    extractor: str
    extractor_version: str | None = None
    nodes: list[SectionNodeCreate]


class SectionNodeRecord(SectionNodeCreate):
    """A persisted heading; ``id``/``parent_id`` are database identities."""

    id: int
    document_id: int
    parent_id: int | None = None


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


VerificationStatus = Literal["unverified", "supported", "rejected"]


class ClaimVerification(RowModel):
    """The only claim columns a verifier may write (migration 0005).

    ``status`` is the final disposition. The verifier's raw verdict
    (``supported | contradicted | absent | wrong_field``), the prompt version,
    whether a repair was attempted, and any ``corrected_quote`` live in
    ``notes`` because they must stay reviewable without changing the claim.
    """

    status: VerificationStatus
    verified_by: str | None = None
    verified_at: datetime | None = None
    notes: dict[str, Any] | None = None


class ClaimRecord(ClaimCreate):
    id: int
    superseded_by: int | None = None
    verification_status: VerificationStatus = "unverified"
    verified_by: str | None = None
    verified_at: datetime | None = None
    verification_notes: dict[str, Any] | None = None


class ClaimEvidence(ClaimRecord):
    url: str
    fetched_at: datetime | None = None
    sha256: str
    snapshot_id: int
    # Which fetch tier produced the snapshot (ladder diagnostic, review console).
    fetcher_tier: str | None = None


class ClaimEvidencePage(RowModel):
    items: list[ClaimEvidence]
    total: int


class ClaimReviewPage(RowModel):
    """Low-confidence claims first; the human decides, nothing is dropped."""

    items: list[ClaimRecord]
    total: int


class AliasReviewPage(RowModel):
    items: list[AliasRecord]
    total: int


class DocumentReviewPage(RowModel):
    items: list[DocumentRecord]
    total: int


ReviewKind = Literal["claim", "alias", "document"]


class ReviewAuditCreate(RowModel):
    """One human review action, append-only (migration 0008).

    ``previous_state`` is the mutable review-relevant fields of the subject
    *before* the action, so every decision is reversible by inspection even
    though the console itself never rewrites history. ``decided_by`` is pinned
    to ``'human'``: machine verdicts live in ``er_audit`` and
    ``claims.verification_*``, never in this table.
    """

    kind: ReviewKind
    subject_id: int
    action: str
    previous_state: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] | None = None
    decided_by: Literal["human"] = "human"


class ReviewAuditRecord(ReviewAuditCreate):
    id: int
    created_at: datetime | None = None


class ProgrammeCreate(RowModel):
    institution_id: int
    name: str
    credential_level: Literal["degree", "diploma", "lateral_entry", "certificate"]
    duration_months: int | None = None
    internship_months: float | None = None
    affiliated_university_id: int | None = None
    recognition_status: Literal["recognised", "conditional", "unrecognised", "unknown"] = "unknown"
    seats: int | None = None
    medium: str | None = None
    #: ER-normalised grouping key; the repo derives a stable fallback when a
    #: caller predates migration 0006.
    normalised_name: str | None = None
    #: Published attribute -> claim id that proves it (migration 0006).
    field_sources: dict[str, str] = {}
    needs_review: bool = False
    review_reasons: list[str] = []
    #: Predicate family that supplied the programme name (curriculum/seats/…).
    source_family: str | None = None
    #: Three-tier scoping outcome (migration 0010). Out-of-scope rows are kept
    #: for audit but excluded from scoring/ranking/shortlist; `uncertain` rows
    #: are out of scope AND needs_review until a human decides.
    in_scope: bool = True
    scope_reason: str | None = None
    scope_tier: Literal["rule", "semantic", "manual"] | None = None
    scope_evidence: list[str] = []
    scope_model: str | None = None
    scope_prompt_version: str | None = None
    scope_cached: bool = False
    scope_decided_at: datetime | None = None


class ProgrammeRecord(ProgrammeCreate):
    id: int
    created_at: datetime | None = None


class ProgrammeFilter(RowModel):
    credential_level: str | None = None
    recognition_status: str | None = None
    institution_id: int | None = None
    state: str | None = None
    query: str | None = None
    #: None = no scoping filter (all rows); True = in-scope only; False = the
    #: out-of-scope audit set (excluded rows, including `uncertain`).
    in_scope: bool | None = None
    limit: int = 50
    offset: int = 0


class ProgrammeFieldEvidence(RowModel):
    """One ``field_sources`` entry joined to the claim that proves it.

    ``predicate`` is the ``field_sources`` key (the predicate that produced the
    attribute); the remaining columns come from the claim and its provenance
    chain, so ``fmd programmes inspect`` can show field, claim id, quote and
    page without a second query. A claim that no longer resolves (append-only
    claims should never vanish) yields NULLs rather than an error.
    """

    predicate: str
    claim_id: int
    value: Any = None
    unit: str | None = None
    quote: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    confidence: float | None = None
    verification_status: str | None = None
    page_range: str | None = None
    url: str | None = None
    fetched_at: datetime | None = None
    sha256: str | None = None


class ProgrammeScopeDecisionCreate(RowModel):
    """One durable scoping decision (classifier cache + manual learning loop).

    ``cache_key`` is the sha256 over (normalised name, institution type,
    definition version): one model call per new *name*, not per row. Human
    decisions use the same key space with ``decided_by='human'`` and
    ``tier='manual'`` so Tier 1 consumes them before any classifier call.
    """

    cache_key: str
    normalised_name: str
    raw_name: str | None = None
    institution_type: str | None = None
    country: str | None = None
    decision: Literal["in_scope", "out_of_scope", "uncertain"]
    tier: Literal["rule", "semantic", "manual"]
    confidence: float | None = None
    rationale: str | None = None
    evidence: list[str] = []
    model: str | None = None
    prompt_version: str | None = None
    cached: bool = False
    needs_review: bool = False
    flags: list[str] = []
    definition_version: str
    decided_by: Literal["classifier", "human"] = "classifier"
    note: str | None = None


class ProgrammeScopeDecision(ProgrammeScopeDecisionCreate):
    id: int
    created_at: datetime | None = None


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
    # Derived, not a column: a merge records the score ids it invalidated in
    # ``er_audit.features_json``; the row survives but must not be trusted.
    stale: bool = False


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

    @model_validator(mode="before")
    @classmethod
    def _map_sitting_dates_column(cls, data: Any) -> Any:
        # The row column is ``sitting_dates_json``; without this mapping the
        # jsonb sitting dates were silently dropped by ``extra="ignore"``.
        if isinstance(data, dict) and "sitting_dates" not in data and "sitting_dates_json" in data:
            data = {**data, "sitting_dates": data["sitting_dates_json"]}
        return data


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


class PipelineRunCreate(RowModel):
    """One stage execution in the SQL run ledger (migration 0003)."""

    run_id: str
    stage: Literal["crawl", "extract", "score", "worker", "ingest"]
    started_at: datetime
    finished_at: datetime | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["running", "succeeded", "failed", "dry_run"] = "running"


class PipelineRunRecord(PipelineRunCreate):
    pass


class FrontierUpsert(RowModel):
    url: str
    domain: str
    priority: int = 0
    next_allowed_at: datetime | None = None
    state: Literal["queued", "in_progress", "done", "failed", "quarantined"] = "queued"
    http_method: Literal["GET", "POST"] = "GET"
    http_body: dict[str, Any] | None = None
    request_headers: dict[str, Any] | None = None
    source_id: str | None = None
    extraction_family: str | None = None
    entity_id: int | None = None

    @classmethod
    def from_document_ref(cls, ref: Any, **fields: Any) -> FrontierUpsert:
        """Enqueue a ``DocumentRef`` without losing its request shape.

        ``http_method`` / ``http_body`` / ``request_headers`` persist in
        ``crawl_frontier`` so a queued ``admin-ajax`` POST page is still
        replayable after a restart. Conditional validators (``etag`` /
        ``last_modified``) become ``If-None-Match`` / ``If-Modified-Since``
        request headers: the frontier has no validator columns, and this is the
        form the next fetch needs anyway.
        """
        metadata = getattr(ref, "metadata", None) or {}
        method = str(metadata.get("request_method") or metadata.get("http_method") or "GET").upper()
        if method not in {"GET", "POST"}:
            raise ValueError(f"unsupported request method {method!r} (GET or POST)")
        body = metadata.get("request_body", metadata.get("http_body"))
        raw_headers = metadata.get("request_headers") or metadata.get("headers")
        headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}
        etag = getattr(ref, "etag", None) or metadata.get("etag")
        last_modified = getattr(ref, "last_modified", None) or metadata.get("last_modified")
        lowered = {key.lower() for key in headers}
        if etag and "if-none-match" not in lowered:
            headers["If-None-Match"] = str(etag)
        if last_modified and "if-modified-since" not in lowered:
            headers["If-Modified-Since"] = str(last_modified)
        from urllib.parse import urlsplit

        # Discovery may already know the target; carry it as a column, never as
        # a synthetic request header. The planner refines it after the fetch.
        family = metadata.get("family") or metadata.get("document_family")
        if family and "extraction_family" not in fields:
            fields["extraction_family"] = str(family)
        if metadata.get("entity_id") is not None and "entity_id" not in fields:
            with contextlib.suppress(TypeError, ValueError):
                fields["entity_id"] = int(metadata["entity_id"])

        return cls(
            url=ref.url,
            domain=fields.pop("domain", None) or (urlsplit(ref.url).hostname or ""),
            priority=fields.pop("priority", getattr(ref, "priority", 0) or 0),
            http_method=method,
            http_body=body if isinstance(body, dict) else None,
            request_headers=headers or None,
            source_id=fields.pop("source_id", None) or getattr(ref, "source_id", None),
            **fields,
        )


class FrontierRecord(FrontierUpsert):
    attempts: int = 0
    tier_used: str | None = None
    ban_score: float | None = None
    yield_score: float | None = None
    updated_at: datetime | None = None


class DomainPolitenessRecord(RowModel):
    domain: str
    rps: float | None = None
    crawl_delay: float | None = None
    robots: dict[str, Any] | None = None
    robots_fetched_at: datetime | None = None
    quarantined_until: datetime | None = None


class DocumentParsePut(RowModel):
    """One parse-cache lookup row (migration 0011).

    ``key_digest`` is the caller-computed cache key (sha256 over snapshot sha,
    tier, policy digest and payload schema version); the components are stored
    beside it so a row is auditable without recomputing the key.
    """

    key_digest: str
    snapshot_sha256: str
    tier: str
    policy_digest: str
    schema_version: int
    object_sha256: str
    pages: int
    parser: str
    parser_version: str | None = None


class DocumentParseRecord(DocumentParsePut):
    created_at: datetime | None = None


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


class AppendOnlyViolation(RuntimeError):
    """An append-only registry was asked to change an existing row.

    Corrections are appended under a NEW id (``<id>@<date-of-correction>``);
    rewriting history is the one thing a trust surface must never do.
    """


class SourceExclusionsRepo:
    """The source-exclusion registry (PLAN v1-w9c): append-only.

    ``append`` inserts a new exclusion, or accepts an identical re-seed as a
    no-op. An attempt to CHANGE an existing row's name or reason raises
    :class:`AppendOnlyViolation` — the correction must be appended under a new
    id so the registry keeps its history.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    def append(self, seed: SourceExclusionSeed) -> str:
        """Append one exclusion; returns ``"appended"`` or ``"unchanged"``."""
        with self.db.connection() as conn:
            inserted = conn.execute(
                """
                INSERT INTO source_exclusions (id, name, reason, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                (seed.id, seed.name, seed.reason),
            ).fetchone()
            if inserted is not None:
                return "appended"
            row = conn.execute(
                "SELECT name, reason FROM source_exclusions WHERE id = %s", (seed.id,)
            ).fetchone()
        if row is None:  # pragma: no cover - conflict without a row is impossible
            raise AppendOnlyViolation(
                f"source exclusion {seed.id!r} conflicted but is missing"
            )
        if row["name"] != seed.name or row["reason"] != seed.reason:
            raise AppendOnlyViolation(
                f"source exclusion {seed.id!r} already exists with a different "
                f"name/reason; append a correction under a new id "
                f"(for example {seed.id}@<date>) instead of rewriting it"
            )
        return "unchanged"


class EntitiesRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, create: EntityCreate) -> EntityRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO entities
                    (kind, canonical_name, country, state, city, status, normalised_name,
                     blocking_key, website_url, institution_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    create.kind,
                    create.canonical_name,
                    create.country,
                    create.state,
                    create.city,
                    create.status,
                    create.normalised_name,
                    create.blocking_key,
                    create.website_url,
                    create.institution_type,
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
                "SELECT * FROM entities WHERE kind = %s AND canonical_name = %s "
                "ORDER BY id LIMIT 1",
                (kind, canonical_name),
            ).fetchone()
        return _entity(row) if row else None

    def find_normalised(self, kind: str, normalised_name: str) -> EntityRecord | None:
        """Idempotent inventory match on the ER-normalised name (migration 0002)."""
        if not normalised_name:
            return None
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE kind = %s AND normalised_name = %s "
                "ORDER BY id LIMIT 1",
                (kind, normalised_name),
            ).fetchone()
        return _entity(row) if row else None

    def update_details(
        self,
        entity_id: int,
        *,
        state: str | None = None,
        city: str | None = None,
        institution_type: str | None = None,
        website_url: str | None = None,
        normalised_name: str | None = None,
        blocking_key: str | None = None,
    ) -> EntityRecord | None:
        """Fill inventory facts without clearing what a human/source already set.

        A supplied value wins (an inventory refresh is newer than the stored
        row); an omitted value is left untouched. Never deletes evidence.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE entities SET
                    state = COALESCE(%s, state),
                    city = COALESCE(%s, city),
                    institution_type = COALESCE(%s, institution_type),
                    website_url = COALESCE(%s, website_url),
                    normalised_name = COALESCE(normalised_name, %s),
                    blocking_key = COALESCE(blocking_key, %s)
                WHERE id = %s
                RETURNING *
                """,
                (
                    state,
                    city,
                    institution_type,
                    website_url,
                    normalised_name,
                    blocking_key,
                    entity_id,
                ),
            ).fetchone()
        return _entity(row) if row else None

    def find_by_blocking_key(
        self, blocking_key: str, kind: str | None = None, limit: int = 50
    ) -> list[EntityRecord]:
        sql = "SELECT * FROM entities WHERE blocking_key = %s AND merged_into_entity_id IS NULL"
        params: list[Any] = [blocking_key]
        if kind is not None:
            sql += " AND kind = %s"
            params.append(kind)
        sql += " ORDER BY id LIMIT %s"
        params.append(limit)
        with self.db.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_entity(row) for row in rows]

    def list(
        self,
        kind: str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_merged: bool = True,
    ) -> list[EntityRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if kind is not None:
            conditions.append("kind = %s")
            params.append(kind)
        if not include_merged:
            conditions.append("merged_into_entity_id IS NULL")
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.extend([limit, offset])
        with self.db.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM entities{where} ORDER BY id LIMIT %s OFFSET %s",  # noqa: S608
                tuple(params),
            ).fetchall()
        return [_entity(row) for row in rows]

    # ------------------------------------------------------------------ merge audit
    def record_audit(self, audit: ERAuditCreate) -> ERAuditRecord:
        if audit.subject_entity_id == audit.candidate_entity_id:
            raise ValueError("an entity cannot be audited against itself")
        if not 0.0 <= audit.score <= 1.0:
            raise ValueError("audit score must be in [0, 1]")
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO er_audit
                    (subject_entity_id, candidate_entity_id, band, score, features_json,
                     thresholds_version, model, model_verdict, model_reason, decided_by,
                     claims_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    audit.subject_entity_id,
                    audit.candidate_entity_id,
                    audit.band,
                    audit.score,
                    _json(audit.features),
                    audit.thresholds_version,
                    audit.model,
                    audit.model_verdict,
                    audit.model_reason,
                    audit.decided_by,
                    audit.claims_hash,
                ),
            ).fetchone()
        assert row is not None
        return _er_audit(row)

    def merge(
        self,
        entity_id: int,
        into_entity_id: int,
        *,
        score: float = 1.0,
        features: dict[str, Any] | None = None,
        thresholds_version: str = "manual",
        model: str | None = None,
        model_verdict: str | None = None,
        model_reason: str | None = None,
        decided_by: Literal["auto", "model", "human"] = "human",
        claims_hash: str | None = None,
    ) -> tuple[EntityRecord, ERAuditRecord]:
        """Merge ``entity_id`` into ``into_entity_id`` and move its evidence.

        In one transaction this:

        * reparents ``claims.entity_id``, ``entity_aliases.entity_id``,
          ``programmes.institution_id`` and ``programmes.affiliated_university_id``;
        * records the exact ids it moved (plus the dependent ``cost_items`` /
          ``financing_options`` rows and the target's prior claim ids) in
          ``er_audit.features_json.evidence`` so :meth:`unmerge` can move exactly
          those rows back;
        * records the ``scores`` rows the merge invalidated in
          ``features_json.scores_invalidated`` — they are kept, never deleted.

        Claim content (value/quote/span/confidence/...) is never touched: only
        the owning entity changes.

        Reversible: :meth:`unmerge` moves the recorded evidence back and appends
        a ``reject`` audit row, so the full history is preserved.
        """
        if entity_id == into_entity_id:
            raise ValueError("cannot merge an entity into itself")
        if not 0.0 <= score <= 1.0:
            raise ValueError("merge score must be in [0, 1]")
        with self.db.connection() as conn:
            subject = conn.execute("SELECT * FROM entities WHERE id = %s", (entity_id,)).fetchone()
            target = conn.execute(
                "SELECT * FROM entities WHERE id = %s", (into_entity_id,)
            ).fetchone()
            if subject is None:
                raise ValueError(f"entity {entity_id} does not exist")
            if target is None:
                raise ValueError(f"target entity {into_entity_id} does not exist")
            if subject["merged_into_entity_id"] is not None:
                raise ValueError(f"entity {entity_id} is already merged")
            if target["merged_into_entity_id"] is not None:
                raise ValueError(
                    f"target entity {into_entity_id} is itself merged; merge into the survivor"
                )
            evidence = self._reparent_evidence(conn, entity_id, into_entity_id)
            invalidated = self._invalidate_scores(conn, entity_id, into_entity_id)
            row = conn.execute(
                """
                UPDATE entities
                SET merged_into_entity_id = %s, merged_at = now(), status = 'merged'
                WHERE id = %s
                RETURNING *
                """,
                (into_entity_id, entity_id),
            ).fetchone()
            payload = dict(features or {})
            payload.update(
                {
                    "operation": "merge",
                    "evidence": evidence,
                    "scores_invalidated": invalidated,
                }
            )
            audit_row = conn.execute(
                """
                INSERT INTO er_audit
                    (subject_entity_id, candidate_entity_id, band, score, features_json,
                     thresholds_version, model, model_verdict, model_reason, decided_by,
                     claims_hash)
                VALUES (%s, %s, 'accept', %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    entity_id,
                    into_entity_id,
                    score,
                    _json(payload),
                    thresholds_version,
                    model,
                    model_verdict,
                    model_reason,
                    decided_by,
                    claims_hash,
                ),
            ).fetchone()
        assert row is not None and audit_row is not None
        return _entity(row), _er_audit(audit_row)

    # ------------------------------------------------------------------ merge internals
    @staticmethod
    def _ids(rows: list[Any]) -> list[int]:
        return [int(row["id"]) for row in rows]

    def _reparent_evidence(self, conn: Any, source: int, target: int) -> dict[str, Any]:
        """Move every entity-owned row from ``source`` to ``target``.

        Returns the exact id sets that moved. Dependent rows that follow by FK
        (``cost_items`` via ``programme_id``, ``financing_options`` via
        ``source_claim_id``) are recorded so :meth:`unmerge` can verify they did
        not drift, but they need no UPDATE: their parent moved.
        """
        target_claims_before = self._ids(
            conn.execute(
                "SELECT id FROM claims WHERE entity_id = %s ORDER BY id", (target,)
            ).fetchall()
        )
        claim_ids = self._ids(
            conn.execute(
                "SELECT id FROM claims WHERE entity_id = %s ORDER BY id", (source,)
            ).fetchall()
        )
        if claim_ids:
            conn.execute("UPDATE claims SET entity_id = %s WHERE entity_id = %s", (target, source))

        # An alias already present on the target would violate the
        # (entity_id, alias, alias_type) unique index. Those rows stay on the
        # source (nothing is deleted) and are recorded explicitly.
        alias_conflicts = self._ids(
            conn.execute(
                """
                SELECT s.id
                FROM entity_aliases s
                JOIN entity_aliases t
                  ON t.entity_id = %s AND t.alias = s.alias AND t.alias_type = s.alias_type
                WHERE s.entity_id = %s
                ORDER BY s.id
                """,
                (target, source),
            ).fetchall()
        )
        all_aliases = self._ids(
            conn.execute(
                "SELECT id FROM entity_aliases WHERE entity_id = %s ORDER BY id", (source,)
            ).fetchall()
        )
        conflict_set = set(alias_conflicts)
        alias_ids = [alias_id for alias_id in all_aliases if alias_id not in conflict_set]
        if alias_ids:
            conn.execute(
                "UPDATE entity_aliases SET entity_id = %s WHERE id = ANY(%s)",
                (target, alias_ids),
            )

        programme_ids = self._ids(
            conn.execute(
                "SELECT id FROM programmes WHERE institution_id = %s ORDER BY id", (source,)
            ).fetchall()
        )
        affiliated_ids = self._ids(
            conn.execute(
                "SELECT id FROM programmes WHERE affiliated_university_id = %s ORDER BY id",
                (source,),
            ).fetchall()
        )
        if programme_ids:
            conn.execute(
                "UPDATE programmes SET institution_id = %s WHERE institution_id = %s",
                (target, source),
            )
        if affiliated_ids:
            conn.execute(
                "UPDATE programmes SET affiliated_university_id = %s "
                "WHERE affiliated_university_id = %s",
                (target, source),
            )

        cost_item_ids = (
            self._ids(
                conn.execute(
                    "SELECT id FROM cost_items WHERE programme_id = ANY(%s) ORDER BY id",
                    (programme_ids,),
                ).fetchall()
            )
            if programme_ids
            else []
        )
        financing_ids = (
            self._ids(
                conn.execute(
                    "SELECT id FROM financing_options WHERE source_claim_id = ANY(%s) ORDER BY id",
                    (claim_ids,),
                ).fetchall()
            )
            if claim_ids
            else []
        )
        return {
            "claims": claim_ids,
            "aliases": alias_ids,
            "aliases_left_on_source": alias_conflicts,
            "programmes": programme_ids,
            "affiliated_programmes": affiliated_ids,
            "cost_items": cost_item_ids,
            "financing_options": financing_ids,
            "target_claims_before": target_claims_before,
        }

    @staticmethod
    def _invalidate_scores(conn: Any, source: int, target: int) -> dict[str, list[int]]:
        """Record (never delete) the score rows a membership change invalidates."""
        invalidated: dict[str, list[int]] = {}
        for key, entity_id in (("source", source), ("target", target)):
            invalidated[key] = [
                int(row["id"])
                for row in conn.execute(
                    "SELECT id FROM scores WHERE entity_id = %s ORDER BY id", (entity_id,)
                ).fetchall()
            ]
        return invalidated

    def _unmerge_drift(
        self, conn: Any, source: int, target: int, evidence: dict[str, Any]
    ) -> list[str]:
        """Every way the current evidence no longer matches what the merge moved."""
        moved = evidence.get("evidence", evidence)
        target_claims_before = evidence.get(
            "target_claims_before", moved.get("target_claims_before") or []
        )
        drift: list[str] = []

        claim_ids = [int(value) for value in moved.get("claims") or []]
        expected = {*(int(value) for value in target_claims_before), *claim_ids}
        current_target_claims = {
            int(row["id"])
            for row in conn.execute(
                "SELECT id FROM claims WHERE entity_id = %s", (target,)
            ).fetchall()
        }
        new_claims = sorted(current_target_claims - expected)
        if new_claims:
            drift.append(
                f"claims attached to the survivor (entity {target}) after the merge: {new_claims}"
            )
        missing_claims = sorted(set(claim_ids) - current_target_claims)
        if missing_claims:
            drift.append(
                f"claims moved by the merge are no longer owned by entity {target}: "
                f"{missing_claims}"
            )

        alias_ids = [int(value) for value in moved.get("aliases") or []]
        current_target_aliases = {
            int(row["id"])
            for row in conn.execute(
                "SELECT id FROM entity_aliases WHERE entity_id = %s", (target,)
            ).fetchall()
        }
        missing_aliases = sorted(set(alias_ids) - current_target_aliases)
        if missing_aliases:
            drift.append(
                f"aliases moved by the merge are no longer owned by entity {target}: "
                f"{missing_aliases}"
            )

        programme_ids = [int(value) for value in moved.get("programmes") or []]
        affiliated_ids = [int(value) for value in moved.get("affiliated_programmes") or []]
        programme_ownership = {
            int(row["id"]): (row["institution_id"], row["affiliated_university_id"])
            for row in conn.execute(
                "SELECT id, institution_id, affiliated_university_id FROM programmes "
                "WHERE id = ANY(%s)",
                (programme_ids + affiliated_ids,),
            ).fetchall()
        }
        for programme_id in programme_ids:
            ownership = programme_ownership.get(programme_id)
            if ownership is None or ownership[0] != target:
                drift.append(f"programme {programme_id} is no longer owned by entity {target}")
        for programme_id in affiliated_ids:
            ownership = programme_ownership.get(programme_id)
            if ownership is None or ownership[1] != target:
                drift.append(
                    f"programme {programme_id} is no longer affiliated with entity {target}"
                )

        # cost_items / financing_options follow their parent; a changed parent
        # means the evidence set is no longer the one the merge captured.
        for row in conn.execute(
            "SELECT id, programme_id FROM cost_items WHERE id = ANY(%s)",
            ([int(value) for value in moved.get("cost_items") or []],),
        ).fetchall():
            if int(row["programme_id"]) not in {
                int(value) for value in moved.get("programmes") or []
            }:
                drift.append(f"cost item {row['id']} no longer belongs to a moved programme")
        for row in conn.execute(
            "SELECT id, source_claim_id FROM financing_options WHERE id = ANY(%s)",
            ([int(value) for value in moved.get("financing_options") or []],),
        ).fetchall():
            if row["source_claim_id"] not in set(claim_ids):
                drift.append(f"financing option {row['id']} no longer cites a moved claim")
        return drift

    def unmerge(
        self, entity_id: int, *, decided_by: Literal["auto", "human"] = "human"
    ) -> UnmergeResult:
        """Move the merge's evidence back and append a ``reject`` audit row.

        This is a true inverse of :meth:`merge`: only the row ids recorded in the
        merge audit move back, and nothing moves if the evidence has drifted
        (e.g. new claims arrived on the survivor). A conflict is returned with a
        ``drift`` list instead of being applied silently.
        """
        with self.db.connection() as conn:
            subject = conn.execute("SELECT * FROM entities WHERE id = %s", (entity_id,)).fetchone()
            if subject is None:
                raise ValueError(f"entity {entity_id} does not exist")
            previous = subject["merged_into_entity_id"]
            if previous is None:
                raise ValueError(f"entity {entity_id} is not merged")
            previous = int(previous)
            merge_audit = conn.execute(
                """
                SELECT * FROM er_audit
                WHERE subject_entity_id = %s AND candidate_entity_id = %s
                  AND band = 'accept'
                  AND features_json->>'operation' = 'merge'
                ORDER BY id DESC LIMIT 1
                """,
                (entity_id, previous),
            ).fetchone()
            if merge_audit is None:
                raise ValueError(
                    f"entity {entity_id} has no merge audit row recording moved evidence; "
                    "refusing to guess what to move back"
                )
            evidence = (merge_audit["features_json"] or {}).get("evidence")
            if not evidence:
                raise ValueError(
                    f"merge audit for entity {entity_id} does not record the moved evidence; "
                    "refusing to guess what to move back"
                )

            drift = self._unmerge_drift(conn, entity_id, previous, evidence)
            if drift:
                return UnmergeResult(entity=_entity(subject), conflict=True, drift=tuple(drift))

            moved = evidence.get("evidence", evidence)
            moved_back: dict[str, list[int]] = {}
            for key in ("claims", "aliases"):
                ids = [int(value) for value in moved.get(key) or []]
                if ids:
                    table = "claims" if key == "claims" else "entity_aliases"
                    column = "entity_id"
                    cursor = conn.execute(
                        f"UPDATE {table} SET {column} = %s WHERE id = ANY(%s) "  # noqa: S608
                        f"AND {column} = %s",
                        (entity_id, ids, previous),
                    )
                    if cursor.rowcount != len(ids):
                        raise ValueError(
                            f"unmerge of entity {entity_id} moved {cursor.rowcount} of "
                            f"{len(ids)} {key}; refusing a partial revert"
                        )
                    moved_back[key] = ids
            programme_ids = [int(value) for value in moved.get("programmes") or []]
            if programme_ids:
                cursor = conn.execute(
                    "UPDATE programmes SET institution_id = %s WHERE id = ANY(%s) "
                    "AND institution_id = %s",
                    (entity_id, programme_ids, previous),
                )
                if cursor.rowcount != len(programme_ids):
                    raise ValueError(
                        f"unmerge of entity {entity_id} moved {cursor.rowcount} of "
                        f"{len(programme_ids)} programmes; refusing a partial revert"
                    )
                moved_back["programmes"] = programme_ids
            affiliated_ids = [int(value) for value in moved.get("affiliated_programmes") or []]
            if affiliated_ids:
                cursor = conn.execute(
                    "UPDATE programmes SET affiliated_university_id = %s WHERE id = ANY(%s) "
                    "AND affiliated_university_id = %s",
                    (entity_id, affiliated_ids, previous),
                )
                if cursor.rowcount != len(affiliated_ids):
                    raise ValueError(
                        f"unmerge of entity {entity_id} reverted {cursor.rowcount} of "
                        f"{len(affiliated_ids)} affiliations; refusing a partial revert"
                    )
                moved_back["affiliated_programmes"] = affiliated_ids

            invalidated = self._invalidate_scores(conn, entity_id, previous)
            row = conn.execute(
                """
                UPDATE entities
                SET merged_into_entity_id = NULL, merged_at = NULL, status = 'active'
                WHERE id = %s
                RETURNING *
                """,
                (entity_id,),
            ).fetchone()
            payload = {
                "operation": "unmerge",
                "merged_into": previous,
                "moved_back": moved_back,
                "scores_invalidated": invalidated,
            }
            audit_row = conn.execute(
                """
                INSERT INTO er_audit
                    (subject_entity_id, candidate_entity_id, band, score, features_json,
                     thresholds_version, decided_by)
                VALUES (%s, %s, 'reject', 0.0, %s, 'manual', %s)
                RETURNING *
                """,
                (entity_id, previous, _json(payload), decided_by),
            ).fetchone()
        assert row is not None and audit_row is not None
        return UnmergeResult(entity=_entity(row), audit=_er_audit(audit_row), moved_back=moved_back)

    def audit_for_pair(
        self, subject_entity_id: int, candidate_entity_id: int, limit: int = 50
    ) -> list[ERAuditRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM er_audit WHERE subject_entity_id = %s "
                "AND candidate_entity_id = %s ORDER BY created_at DESC, id DESC LIMIT %s",
                (subject_entity_id, candidate_entity_id, limit),
            ).fetchall()
        return [_er_audit(row) for row in rows]

    def audit_recent(self, limit: int = 50, since: datetime | None = None) -> list[ERAuditRecord]:
        if since is None:
            sql = "SELECT * FROM er_audit ORDER BY created_at DESC, id DESC LIMIT %s"
            params: tuple[Any, ...] = (limit,)
        else:
            sql = (
                "SELECT * FROM er_audit WHERE created_at >= %s "
                "ORDER BY created_at DESC, id DESC LIMIT %s"
            )
            params = (since, limit)
        with self.db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_er_audit(row) for row in rows]


class AliasesRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, create: AliasCreate) -> AliasRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO entity_aliases
                    (entity_id, alias, alias_type, source_id, confidence, span_start,
                     span_end, quote, source_document_id, source_extraction_id,
                     confirmed_by, confirmed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    create.entity_id,
                    create.alias,
                    create.alias_type,
                    create.source_id,
                    create.confidence,
                    create.span_start,
                    create.span_end,
                    create.quote,
                    create.source_document_id,
                    create.source_extraction_id,
                    create.confirmed_by,
                    create.confirmed_at,
                ),
            ).fetchone()
        assert row is not None
        return AliasRecord.model_validate(row)

    def list_for_entity(self, entity_id: int) -> list[AliasRecord]:
        """Aliases that still count for resolution, oldest first.

        This is the read the ER candidate universe consumes, so an alias a
        human *rejected* (review_status, migration 0008) stops resolving
        mentions immediately: a rejected suggestion must not keep boosting
        similarity or locking a pair. Confirmed and still-pending rows stay
        visible — pending is the pre-review default, not a distrust verdict.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM entity_aliases WHERE entity_id = %s "
                "AND review_status <> 'rejected' ORDER BY id",
                (entity_id,),
            ).fetchall()
        return [AliasRecord.model_validate(row) for row in rows]

    def get(self, alias_id: int) -> AliasRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM entity_aliases WHERE id = %s", (alias_id,)).fetchone()
        return AliasRecord.model_validate(row) if row else None

    def review_queue(self, limit: int = 100, offset: int = 0) -> AliasReviewPage:
        """Alias suggestions awaiting a human decision, oldest first."""
        with self.db.connection() as conn:
            total = conn.execute(
                "SELECT count(*) AS n FROM entity_aliases WHERE review_status = 'pending'"
            ).fetchone()
            rows = conn.execute(
                "SELECT * FROM entity_aliases WHERE review_status = 'pending' "
                "ORDER BY id LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
        return AliasReviewPage(
            items=[AliasRecord.model_validate(row) for row in rows],
            total=int(total["n"]) if total else 0,
        )

    def confirm(self, alias_id: int, *, confirmed_by: str) -> AliasRecord:
        """Record the human approval of an alias suggestion.

        For a ``manual`` alias this is what arms the ER lock (types.py:
        manual aliases are authoritative). Raises ``ValueError`` for a missing
        or already-decided row so a stale console cannot silently flip a
        decision that was already taken.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE entity_aliases SET
                    review_status = 'confirmed',
                    confirmed_by = %s,
                    confirmed_at = COALESCE(confirmed_at, now())
                WHERE id = %s AND review_status IN ('pending', 'confirmed')
                RETURNING *
                """,
                (confirmed_by, alias_id),
            ).fetchone()
        if row is None:
            raise ValueError(f"alias {alias_id} does not exist or was rejected")
        return AliasRecord.model_validate(row)

    def reject(self, alias_id: int) -> AliasRecord:
        """Record the human refusal of an alias suggestion (append-only audit
        lives in ``review_audit``; the row keeps its evidence columns)."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE entity_aliases SET review_status = 'rejected'
                WHERE id = %s AND review_status IN ('pending', 'rejected')
                RETURNING *
                """,
                (alias_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"alias {alias_id} does not exist or was confirmed")
        return AliasRecord.model_validate(row)

    def create_if_missing(self, create: AliasCreate) -> AliasRecord | None:
        """Idempotent alias write; returns ``None`` when the alias already exists."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO entity_aliases
                    (entity_id, alias, alias_type, source_id, confidence, span_start,
                     span_end, quote, source_document_id, source_extraction_id,
                     confirmed_by, confirmed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (entity_id, alias, alias_type) DO NOTHING
                RETURNING *
                """,
                (
                    create.entity_id,
                    create.alias,
                    create.alias_type,
                    create.source_id,
                    create.confidence,
                    create.span_start,
                    create.span_end,
                    create.quote,
                    create.source_document_id,
                    create.source_extraction_id,
                    create.confirmed_by,
                    create.confirmed_at,
                ),
            ).fetchone()
        return AliasRecord.model_validate(row) if row else None

    def list_manual_for_entity(self, entity_id: int) -> list[AliasRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM entity_aliases WHERE entity_id = %s AND alias_type = 'manual' "
                "ORDER BY id",
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
                     robots_decision, robots_checked_at, http_status, http_method, http_body,
                     request_headers, source_id, extraction_family, entity_id, plan_state,
                     plan_confidence, plan_reason, role, text_sha256, dedupe_mode, dedupe_basis,
                     dedupe_similarity, dedupe_reason, dedupe_compared_document_id,
                     dedupe_simhash)
                VALUES (%s, %s, %s, COALESCE(%s, now()), COALESCE(%s, now()), %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, 'evidence'),
                        %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    domain = EXCLUDED.domain,
                    kind = COALESCE(EXCLUDED.kind, documents.kind),
                    last_seen = COALESCE(EXCLUDED.last_seen, now()),
                    etag = COALESCE(EXCLUDED.etag, documents.etag),
                    last_modified = COALESCE(EXCLUDED.last_modified, documents.last_modified),
                    robots_decision = COALESCE(EXCLUDED.robots_decision, documents.robots_decision),
                    robots_checked_at = COALESCE(EXCLUDED.robots_checked_at,
                                                 documents.robots_checked_at),
                    http_status = COALESCE(EXCLUDED.http_status, documents.http_status),
                    http_method = CASE
                        WHEN EXCLUDED.http_method = 'GET' AND documents.http_method = 'POST'
                             AND EXCLUDED.http_body IS NULL
                        THEN documents.http_method ELSE EXCLUDED.http_method END,
                    http_body = COALESCE(EXCLUDED.http_body, documents.http_body),
                    request_headers = COALESCE(EXCLUDED.request_headers,
                                               documents.request_headers),
                    source_id = COALESCE(EXCLUDED.source_id, documents.source_id),
                    extraction_family = COALESCE(EXCLUDED.extraction_family,
                                                 documents.extraction_family),
                    entity_id = COALESCE(EXCLUDED.entity_id, documents.entity_id),
                    plan_state = COALESCE(EXCLUDED.plan_state, documents.plan_state),
                    plan_confidence = COALESCE(EXCLUDED.plan_confidence,
                                               documents.plan_confidence),
                    plan_reason = COALESCE(EXCLUDED.plan_reason, documents.plan_reason),
                    role = COALESCE(EXCLUDED.role, documents.role),
                    -- A recorded decision replaces the previous one wholesale:
                    -- a stored document whose reason is now '' must not keep
                    -- showing the earlier drop. Rows with no decision (304,
                    -- terminal) carry dedupe_mode NULL and leave it untouched.
                    text_sha256 = CASE WHEN EXCLUDED.dedupe_mode IS NOT NULL
                                       THEN EXCLUDED.text_sha256 ELSE documents.text_sha256 END,
                    dedupe_mode = CASE WHEN EXCLUDED.dedupe_mode IS NOT NULL
                                       THEN EXCLUDED.dedupe_mode ELSE documents.dedupe_mode END,
                    dedupe_basis = CASE WHEN EXCLUDED.dedupe_mode IS NOT NULL
                                        THEN EXCLUDED.dedupe_basis ELSE documents.dedupe_basis END,
                    dedupe_similarity = CASE WHEN EXCLUDED.dedupe_mode IS NOT NULL
                                             THEN EXCLUDED.dedupe_similarity
                                             ELSE documents.dedupe_similarity END,
                    dedupe_reason = CASE WHEN EXCLUDED.dedupe_mode IS NOT NULL
                                         THEN EXCLUDED.dedupe_reason
                                         ELSE documents.dedupe_reason END,
                    dedupe_compared_document_id = CASE
                        WHEN EXCLUDED.dedupe_mode IS NOT NULL
                        THEN EXCLUDED.dedupe_compared_document_id
                        ELSE documents.dedupe_compared_document_id END,
                    dedupe_simhash = CASE WHEN EXCLUDED.dedupe_mode IS NOT NULL
                                          THEN EXCLUDED.dedupe_simhash
                                          ELSE documents.dedupe_simhash END
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
                    document.http_method,
                    _json(document.http_body),
                    _json(document.request_headers),
                    document.source_id,
                    document.extraction_family,
                    document.entity_id,
                    document.plan_state,
                    document.plan_confidence,
                    document.plan_reason,
                    document.role,
                    document.text_sha256,
                    document.dedupe_mode,
                    document.dedupe_basis,
                    document.dedupe_similarity,
                    document.dedupe_reason,
                    document.dedupe_compared_document_id,
                    document.dedupe_simhash,
                ),
            ).fetchone()
        assert row is not None
        return _document(row)

    def set_plan(
        self,
        document_id: int,
        *,
        source_id: str | None = None,
        extraction_family: str | None,
        entity_id: int | None,
        plan_state: str,
        plan_confidence: float | None = None,
        plan_reason: str | None = None,
        role: str | None = None,
    ) -> DocumentRecord | None:
        """Persist a :class:`fmd.plan.DocumentPlan` on the document row.

        Deliberately an explicit UPDATE rather than ``upsert``: a plan is a
        decision about an existing document, and ``plan_state`` must be able to
        move backwards (e.g. ``linked`` → ``needs_review``) when evidence changes.
        ``role`` is only written when the planner supplies it; omitting it keeps
        the crawl-time classification.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE documents SET
                    source_id = COALESCE(%s, source_id),
                    extraction_family = %s,
                    entity_id = %s,
                    plan_state = %s,
                    plan_confidence = %s,
                    plan_reason = %s,
                    role = COALESCE(%s, role)
                WHERE id = %s
                RETURNING *
                """,
                (
                    source_id,
                    extraction_family,
                    entity_id,
                    plan_state,
                    plan_confidence,
                    plan_reason,
                    role,
                    document_id,
                ),
            ).fetchone()
        return _document(row) if row else None

    def set_role(self, document_id: int, role: str) -> DocumentRecord | None:
        """Re-classify a document without touching its plan (migration 0004)."""
        with self.db.connection() as conn:
            row = conn.execute(
                "UPDATE documents SET role = %s WHERE id = %s RETURNING *",
                (role, document_id),
            ).fetchone()
        return _document(row) if row else None

    def count_unplanned(self) -> dict[str, int]:
        """Documents extraction will not read, bucketed by *honest* reason.

        Inventory and navigation documents were never extraction candidates, so
        they are counted as ``not_applicable`` — not as a family-detection
        failure. Only evidence documents can be ``missing_family``,
        ``needs_review`` or ``unresolved_entity``.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT role, plan_state, extraction_family, entity_id, count(*) AS n "
                "FROM documents GROUP BY role, plan_state, extraction_family, entity_id"
            ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            total = int(row["n"])
            role = row["role"] or "evidence"
            if role != "evidence":
                reason = "not_applicable"
            elif row["extraction_family"] is None:
                reason = "missing_family"
            elif row["plan_state"] == "needs_review":
                reason = "needs_review"
            elif row["entity_id"] is None:
                reason = "unresolved_entity"
            else:
                continue
            counts[reason] = counts.get(reason, 0) + total
        return counts

    def plan_evidence_rows(self, *, limit: int = 2000) -> list[dict[str, Any]]:
        """Evidence-role documents with their plan columns, for diagnosis.

        Wave v1-w5 (``classify-report``): the read-only inputs the family
        report replays — no decisions are made or written here. Ordered by id
        so a ``--limit`` cut is stable across runs.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT id, url, role, plan_state, plan_reason, "
                "extraction_family, entity_id, source_id "
                "FROM documents WHERE role = 'evidence' OR role IS NULL "
                "ORDER BY id LIMIT %s",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_by_role(self, role: str, *, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        """Documents of one role, oldest first (inventory ingest queue)."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE role = %s ORDER BY id LIMIT %s OFFSET %s",
                (role, limit, offset),
            ).fetchall()
        return [_document(row) for row in rows]

    def list_plan_candidates(
        self, *, limit: int = 500, force: bool = False, planner_version: str | None = None
    ) -> list[DocumentRecord]:
        """Documents the planning stage (:mod:`fmd.orchestrate.plan`) must decide.

        The default queue is everything that *should* be extractable but is not:

        * ``plan_state IS NULL`` — never planned. Snapshot recovery, backfills
          and re-crawls that stored documents without running the planner land
          here; extraction skips them entirely (``list_extractable`` requires a
          family and an entity).
        * ``plan_state = 'planned'`` — Phase A deferrals (``defer_for_text``)
          still waiting for readable text.

        With ``force=True`` the queue widens to *decided* documents
        (``linked`` / ``needs_review`` / ``abstained`` / ``not_applicable``)
        that may be stale at the current planner version: rows whose
        ``plan_reason`` is missing or planner-shaped but carries no
        ``plan_v<planner_version>`` stamp. Reasons the planner does not write —
        the free prose human review leaves on a decision — are never returned,
        so a manual link or exclusion cannot be overwritten by a re-plan.
        """
        if not force:
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE plan_state IS NULL OR plan_state = 'planned' "
                    "ORDER BY id LIMIT %s",
                    (limit,),
                ).fetchall()
            return [_document(row) for row in rows]

        stale_filter = ""
        params: list[Any] = [limit]
        if planner_version:
            # Coarse SQL exclusion of rows already stamped at the current
            # version; the caller re-checks precisely via fmd.plan.versioning.
            # IS DISTINCT FROM keeps crawl-era rows (plan_reason IS NULL):
            # a plain NOT (reason = …) would evaluate to NULL and drop them.
            stale_filter = (
                " AND plan_reason IS DISTINCT FROM %s"
                " AND (plan_reason IS NULL OR plan_reason NOT LIKE %s)"
            )
            params = [f"plan_v{planner_version}", f"%|plan_v{planner_version}", limit]
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE plan_state IN "
                "('linked', 'needs_review', 'abstained', 'not_applicable') "
                # psycopg formats the whole statement, so literal % is doubled.
                "AND (plan_reason IS NULL OR plan_reason LIKE '%%|plan_v%%' "
                "OR plan_reason LIKE 'plan_v%%' "
                "OR plan_reason IN ('family_undetected', 'ambiguous_family', "
                "'entity_undetected', 'entity_unresolved', 'entity_ambiguous') "
                "OR plan_reason LIKE 'unsupported_family:%%')"
                + stale_filter
                + " ORDER BY id LIMIT %s",
                tuple(params),
            ).fetchall()
        return [_document(row) for row in rows]

    def counts_by_role(self) -> dict[str, int]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT role, count(*) AS n FROM documents GROUP BY role ORDER BY role"
            ).fetchall()
        return {str(row["role"]): int(row["n"]) for row in rows}

    def list_needs_review(self, limit: int = 100) -> list[DocumentRecord]:
        """Ambiguous plans awaiting a human decision (WS-G review queue source)."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE plan_state = 'needs_review' "
                "ORDER BY last_seen DESC NULLS LAST, id LIMIT %s",
                (limit,),
            ).fetchall()
        return [_document(row) for row in rows]

    def review_queue(self, limit: int = 100, offset: int = 0) -> DocumentReviewPage:
        """Ambiguous documents to resolve, least-confident plan first.

        Ordered deterministically: the planner's weakest link comes first, then
        most-recently-seen, then id, so pagination is stable while the queue
        changes underneath a reviewer.
        """
        with self.db.connection() as conn:
            total = conn.execute(
                "SELECT count(*) AS n FROM documents WHERE plan_state = 'needs_review'"
            ).fetchone()
            rows = conn.execute(
                "SELECT * FROM documents WHERE plan_state = 'needs_review' "
                "ORDER BY plan_confidence ASC NULLS LAST, last_seen DESC NULLS LAST, id "
                "LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
        return DocumentReviewPage(
            items=[_document(row) for row in rows],
            total=int(total["n"]) if total else 0,
        )

    def review_link_entity(
        self,
        document_id: int,
        entity_id: int,
        *,
        reason: str | None = None,
    ) -> DocumentRecord | None:
        """Resolve an ambiguous plan by naming the entity, human-decided.

        Deliberately not :meth:`set_plan`, which would null out
        ``extraction_family``; a review decision must only write what it
        decided (entity link + state), never re-decide the extraction plan.
        ``plan_reason`` records that a human resolved it, keeping the planner's
        original reason readable in the audit row instead.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE documents SET
                    entity_id = %s,
                    plan_state = 'linked',
                    plan_confidence = 1.0,
                    plan_reason = %s
                WHERE id = %s AND plan_state IN ('needs_review', 'linked')
                RETURNING *
                """,
                (entity_id, reason or "resolved by human review (link_entity)", document_id),
            ).fetchone()
        return _document(row) if row else None

    def review_mark_not_applicable(
        self,
        document_id: int,
        *,
        reason: str | None = None,
    ) -> DocumentRecord | None:
        """Take an ambiguous document out of the extraction plan by decision."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE documents SET
                    plan_state = 'not_applicable',
                    plan_confidence = 1.0,
                    plan_reason = %s
                WHERE id = %s AND plan_state IN ('needs_review', 'not_applicable')
                RETURNING *
                """,
                (reason or "resolved by human review (mark_not_applicable)", document_id),
            ).fetchone()
        return _document(row) if row else None

    def get(self, document_id: int) -> DocumentRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = %s", (document_id,)).fetchone()
        return _document(row) if row else None

    def get_by_url(self, url: str) -> DocumentRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE url = %s", (url,)).fetchone()
        return _document(row) if row else None

    def list_by_domain(self, domain: str, limit: int = 100) -> list[DocumentRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE domain = %s "
                "ORDER BY last_seen DESC NULLS LAST LIMIT %s",
                (domain, limit),
            ).fetchall()
        return [_document(row) for row in rows]


class ReviewAuditRepo:
    """Append-only log of every human review action (migration 0008).

    One row per action with the subject's previous review-relevant state, so
    the console can show the trail next to each item and an operator can
    reconstruct what the queue looked like when a decision was taken. There is
    no update and no delete: a mistaken action is corrected by a further
    action, never by editing the log.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    def record(self, audit: ReviewAuditCreate) -> ReviewAuditRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO review_audit
                    (kind, subject_id, action, previous_state, details_json, decided_by)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    audit.kind,
                    audit.subject_id,
                    audit.action,
                    _json(audit.previous_state) if audit.previous_state else "{}",
                    _json(audit.details),
                    audit.decided_by,
                ),
            ).fetchone()
        assert row is not None
        return ReviewAuditRecord.model_validate(row)

    def list_for_subject(
        self,
        kind: str,
        subject_id: int,
        *,
        limit: int = 50,
    ) -> list[ReviewAuditRecord]:
        """The trail for one queue item, newest decision first."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM review_audit WHERE kind = %s AND subject_id = %s "
                "ORDER BY id DESC LIMIT %s",
                (kind, subject_id, limit),
            ).fetchall()
        return [ReviewAuditRecord.model_validate(row) for row in rows]


class InstitutionDomainsRepo:
    """The institution-site crawl queue (migration 0004).

    Inventory ingest writes a row per discovered domain; TODO-free by design,
    the state machine is explicitly ``pending -> queued -> planned -> skipped``.
    A re-ingest never resets a domain that is already queued or planned, so an
    idempotent run does not duplicate crawl work.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    def upsert(self, seed: InstitutionDomainSeed) -> InstitutionDomainRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO institution_domains
                    (domain, seed_url, entity_id, discovered_by, priority, reason, state,
                     first_seen, last_seen)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending', now(), now())
                ON CONFLICT (domain) DO UPDATE SET
                    seed_url = EXCLUDED.seed_url,
                    entity_id = COALESCE(EXCLUDED.entity_id, institution_domains.entity_id),
                    discovered_by = COALESCE(EXCLUDED.discovered_by,
                                             institution_domains.discovered_by),
                    priority = EXCLUDED.priority,
                    reason = COALESCE(EXCLUDED.reason, institution_domains.reason),
                    last_seen = now()
                RETURNING *
                """,
                (
                    seed.domain,
                    seed.seed_url,
                    seed.entity_id,
                    seed.discovered_by,
                    seed.priority,
                    seed.reason,
                ),
            ).fetchone()
        assert row is not None
        return _institution_domain(row)

    def upsert_many(self, seeds: Iterable[InstitutionDomainSeed]) -> int:
        return sum(1 for seed in seeds if self.upsert(seed) is not None)

    def list_candidates(self, limit: int = 500) -> list[InstitutionDomainRecord]:
        """Every domain still worth crawling, highest priority first."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM institution_domains WHERE state IN ('pending', 'queued') "
                "ORDER BY priority DESC, domain LIMIT %s",
                (limit,),
            ).fetchall()
        return [_institution_domain(row) for row in rows]

    def list_by_state(self, state: str, limit: int = 500) -> list[InstitutionDomainRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM institution_domains WHERE state = %s "
                "ORDER BY priority DESC, domain LIMIT %s",
                (state, limit),
            ).fetchall()
        return [_institution_domain(row) for row in rows]

    def mark(self, domains: Iterable[str], state: str, *, reason: str | None = None) -> int:
        names = [domain for domain in dict.fromkeys(domains) if domain]
        if not names:
            return 0
        with self.db.connection() as conn:
            row = conn.execute(
                "UPDATE institution_domains SET state = %s, "
                "reason = COALESCE(%s, reason), "
                "last_queued_at = CASE WHEN %s = 'queued' THEN now() ELSE last_queued_at END "
                "WHERE domain = ANY(%s)",
                (state, reason, state, names),
            )
        return int(getattr(row, "rowcount", 0) or 0)

    def set_priority(self, domain: str, priority: int, *, reason: str | None = None) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE institution_domains SET priority = %s, "
                "reason = COALESCE(%s, reason) WHERE domain = %s",
                (priority, reason, domain),
            )

    def counts_by_state(self) -> dict[str, int]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT state, count(*) AS n FROM institution_domains GROUP BY state ORDER BY state"
            ).fetchall()
        return {str(row["state"]): int(row["n"]) for row in rows}


class SnapshotsRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, snapshot: SnapshotCreate) -> SnapshotRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO snapshots
                    (document_id, sha256, bytes_path, fetched_at, fetcher_tier,
                     content_type, headers_json, text_sha256, dedupe_basis)
                VALUES (%s, %s, %s, COALESCE(%s, now()), %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, sha256) DO UPDATE SET
                    bytes_path = EXCLUDED.bytes_path,
                    text_sha256 = COALESCE(EXCLUDED.text_sha256, snapshots.text_sha256),
                    dedupe_basis = COALESCE(EXCLUDED.dedupe_basis, snapshots.dedupe_basis)
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
                    snapshot.text_sha256,
                    snapshot.dedupe_basis,
                ),
            ).fetchone()
        assert row is not None
        return _snapshot(row)

    def get(self, snapshot_id: int) -> SnapshotRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM snapshots WHERE id = %s", (snapshot_id,)).fetchone()
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

    def list_without_extraction(
        self, tool_version: str | None = None, limit: int = 100
    ) -> list[SnapshotRecord]:
        """Snapshots with no extraction at (or above) ``tool_version``.

        The orchestrator's resumable extract queue: a snapshot that already has an
        extraction at the requested version is not re-processed. Passing ``None``
        returns every snapshot (``extract --force``), still in a stable order.
        """
        sql = "SELECT * FROM snapshots s"
        params: list[Any] = []
        if tool_version is not None:
            sql += (
                " WHERE NOT EXISTS (SELECT 1 FROM extractions x"
                " WHERE x.snapshot_id = s.id AND x.tool_version = %s)"
            )
            params.append(tool_version)
        sql += " ORDER BY s.id LIMIT %s"
        params.append(limit)
        with self.db.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_snapshot(row) for row in rows]

    def list_extractable(
        self, tool_version: str | None = None, limit: int = 100
    ) -> list[SnapshotRecord]:
        """Snapshots whose document is planned: family + entity in columns.

        This is the extract queue after migration 0003. A document that is not
        planned (no family, no entity, or ``needs_review``) is never selected, so
        it cannot be silently extracted against the wrong entity.
        """
        sql = (
            "SELECT s.* FROM snapshots s"
            " JOIN documents d ON d.id = s.document_id"
            " WHERE d.role = 'evidence'"
            " AND d.extraction_family IS NOT NULL AND d.entity_id IS NOT NULL"
            " AND (d.plan_state IS NULL OR d.plan_state <> 'needs_review')"
        )
        params: list[Any] = []
        if tool_version is not None:
            sql += (
                " AND NOT EXISTS (SELECT 1 FROM extractions x"
                " WHERE x.snapshot_id = s.id AND x.tool_version = %s)"
            )
            params.append(tool_version)
        sql += " ORDER BY s.id LIMIT %s"
        params.append(limit)
        with self.db.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
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


class SectionsRepo:
    """The document section tree (W8 section-structure, migration 0009).

    Sections are derived structure, never evidence: a write replaces the
    document's whole tree in one transaction (delete + insert keyed by the
    stable ``node_key``), so a re-extraction can never leave a stale tree or a
    half-updated one. Lookups use the (document_id, char_start, level DESC)
    index plus a parent_id walk — never a linear scan per claim.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    def replace_for_document(self, tree: SectionTreeWrite) -> int:
        """Write the document's tree wholesale; returns the node count."""
        with self.db.connection() as conn:
            conn.execute(
                "DELETE FROM document_sections WHERE document_id = %s",
                (tree.document_id,),
            )
            id_by_key: dict[int, int] = {}
            for node in tree.nodes:
                parent_id = id_by_key.get(node.parent_key) if node.parent_key else None
                if node.parent_key is not None and parent_id is None:
                    raise DatabaseError(
                        f"section node {node.node_key} references parent_key "
                        f"{node.parent_key} which is not already written"
                    )
                row = conn.execute(
                    """
                    INSERT INTO document_sections
                        (document_id, parent_id, node_key, level, title,
                         char_start, char_end, heading_char_start, heading_char_end,
                         page_no, bbox_json, extractor, extractor_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        tree.document_id,
                        parent_id,
                        node.node_key,
                        node.level,
                        node.title,
                        node.char_start,
                        node.char_end,
                        node.heading_char_start,
                        node.heading_char_end,
                        node.page_no,
                        _json(list(node.bbox) if node.bbox is not None else None),
                        tree.extractor,
                        tree.extractor_version,
                    ),
                ).fetchone()
                assert row is not None
                id_by_key[node.node_key] = int(row["id"])
        return len(tree.nodes)

    def delete_for_document(self, document_id: int) -> int:
        with self.db.connection() as conn:
            rows = conn.execute(
                "DELETE FROM document_sections WHERE document_id = %s RETURNING id",
                (document_id,),
            ).fetchall()
        return len(rows)

    def list_for_document(self, document_id: int) -> list[SectionNodeRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM document_sections WHERE document_id = %s ORDER BY node_key",
                (document_id,),
            ).fetchall()
        return [_section_node(row) for row in rows]

    def breadcrumb(self, document_id: int, char_start: int, char_end: int) -> dict[str, Any] | None:
        """Deepest section containing ``[char_start, char_end)`` + its chain.

        One indexed query: the anchor picks the narrowest containing node
        (deepest level first), the recursive part walks ``parent_id`` to the
        root. Returns ``{"node": SectionNodeRecord, "breadcrumb": [titles]}``
        or ``None`` when no section contains the span.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                WITH RECURSIVE chain AS (
                    (SELECT * FROM document_sections
                      WHERE document_id = %s
                        AND char_start <= %s AND char_end >= %s
                      ORDER BY level DESC, char_start DESC
                      LIMIT 1)
                  UNION ALL
                    SELECT s.* FROM document_sections s
                    JOIN chain c ON s.id = c.parent_id
                )
                SELECT * FROM chain ORDER BY level ASC
                """,
                (document_id, char_start, char_end),
            ).fetchall()
        if not rows:
            return None
        nodes = [_section_node(row) for row in rows]
        return {"node": nodes[-1], "breadcrumb": [node.title for node in nodes]}


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

    # -------------------------------------------------------------- verification
    def record_verification(
        self,
        claim_id: int,
        verification: ClaimVerification,
    ) -> ClaimRecord:
        """Write only the verification columns; claim content stays immutable.

        Raises ``ValueError`` for a missing claim so a verifier cannot believe it
        recorded a judgement that was never stored.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE claims SET
                    verification_status = %s,
                    verified_by = %s,
                    verified_at = COALESCE(%s, now()),
                    verification_notes = %s
                WHERE id = %s
                RETURNING *
                """,
                (
                    verification.status,
                    verification.verified_by,
                    verification.verified_at,
                    _json(verification.notes),
                    claim_id,
                ),
            ).fetchone()
        if row is None:
            raise ValueError(f"claim {claim_id} does not exist")
        return _claim(row)

    def record_verifications(
        self, items: Iterable[tuple[int, ClaimVerification]]
    ) -> list[ClaimRecord]:
        """Persist a batch in the given order; one statement per claim."""
        return [
            self.record_verification(claim_id, verification) for claim_id, verification in items
        ]

    def list_unverified(self, limit: int = 500) -> list[ClaimRecord]:
        """Pending verification queue, oldest first (deterministic)."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE verification_status = 'unverified' "
                "AND superseded_by IS NULL ORDER BY id LIMIT %s",
                (limit,),
            ).fetchall()
        return [_claim(row) for row in rows]

    def list_review_queue(self, limit: int = 100) -> list[ClaimRecord]:
        """Claims a human should look at: rejected, then still-unverified."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE verification_status <> 'supported' "
                "AND superseded_by IS NULL "
                "ORDER BY CASE verification_status WHEN 'rejected' THEN 0 ELSE 1 END, id "
                "LIMIT %s",
                (limit,),
            ).fetchall()
        return [_claim(row) for row in rows]

    def review_queue(self, limit: int = 100, offset: int = 0) -> ClaimReviewPage:
        """The human review queue (WS-G): lowest-confidence claims first.

        Everything not positively supported and not superseded queues —
        including machine-rejected rows, because a verifier can be wrong and
        the console is the only correction path. Pagination is keyed on the
        same ordering as the queue index (migration 0008) so pages are stable.
        """
        with self.db.connection() as conn:
            total = conn.execute(
                "SELECT count(*) AS n FROM claims "
                "WHERE verification_status <> 'supported' AND superseded_by IS NULL"
            ).fetchone()
            rows = conn.execute(
                "SELECT * FROM claims "
                "WHERE verification_status <> 'supported' AND superseded_by IS NULL "
                "ORDER BY confidence ASC, id ASC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
        return ClaimReviewPage(
            items=[_claim(row) for row in rows],
            total=int(total["n"]) if total else 0,
        )

    def evidence_with_provenance(self, claim_id: int) -> ClaimEvidence | None:
        """One claim joined to the snapshot that proves it (review console).

        The console shows the quote next to its URL, fetch time and hash; the
        chain is the same one :meth:`evidence_page` exposes, read for a single
        claim without scanning an entity.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT c.*, d.url, s.fetched_at, s.sha256, s.id AS snapshot_id, "
                "s.fetcher_tier "
                "FROM claims c "
                "JOIN extractions x ON x.id = c.extraction_id "
                "JOIN snapshots s ON s.id = x.snapshot_id "
                "JOIN documents d ON d.id = s.document_id "
                "WHERE c.id = %s",
                (claim_id,),
            ).fetchone()
        if row is None:
            return None
        return ClaimEvidence(
            **_claim(row).model_dump(),
            url=row["url"],
            fetched_at=row["fetched_at"],
            sha256=row["sha256"],
            snapshot_id=row["snapshot_id"],
        )

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
            "SELECT c.*, d.url, s.fetched_at, s.sha256, s.id AS snapshot_id, s.fetcher_tier "
            "FROM claims c "
            "JOIN extractions x ON x.id = c.extraction_id "
            "JOIN snapshots s ON s.id = x.snapshot_id "
            "JOIN documents d ON d.id = s.document_id "
            f"WHERE {where} "
            "ORDER BY c.observed_at DESC NULLS LAST, c.id DESC LIMIT %s OFFSET %s"
        )
        with self.db.connection() as conn:
            total_row = conn.execute(count_sql, tuple(params)).fetchone()
            rows = conn.execute(page_sql, (*params, limit, offset)).fetchall()
        items = [
            ClaimEvidence(
                **_claim(row).model_dump(),
                url=row["url"],
                fetched_at=row["fetched_at"],
                sha256=row["sha256"],
                snapshot_id=row["snapshot_id"],
                fetcher_tier=row.get("fetcher_tier"),
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

    def correct_claim(
        self,
        old_claim_id: int,
        replacement: ClaimCreate,
        *,
        verified_by: str,
        notes: dict[str, Any] | None = None,
    ) -> tuple[ClaimRecord, ClaimRecord]:
        """Human correction (review console): INSERT the replacement, link, done.

        One connection, one transaction: either the replacement exists and the
        old row points at it, or neither happened — a half-corrected claim
        (orphan replacement, or a superseded row with no successor) would be
        exactly the kind of silent drift the console exists to prevent. The old
        row's content columns are never touched; only ``superseded_by`` and the
        verification columns are written, and the replacement is born
        ``supported`` with ``verified_by`` naming who decided.
        """
        if replacement.quote == "" or replacement.char_end <= replacement.char_start:
            raise ValueError("the replacement claim must carry a resolvable quote span")
        with self.db.connection() as conn:
            old_row = conn.execute(
                "SELECT entity_id FROM claims WHERE id = %s", (old_claim_id,)
            ).fetchone()
            if old_row is None:
                raise ValueError(f"claim {old_claim_id} does not exist")
            if old_row["entity_id"] != replacement.entity_id:
                raise ValueError("cannot supersede a claim for a different entity")
            new_row = conn.execute(
                """
                INSERT INTO claims
                    (extraction_id, entity_id, predicate, value_json, unit, qualifier_json,
                     quote, char_start, char_end, confidence, observed_at, method,
                     verification_status, verified_by, verified_at, verification_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'supported', %s, now(), %s)
                RETURNING *
                """,
                (
                    replacement.extraction_id,
                    replacement.entity_id,
                    replacement.predicate,
                    _json(replacement.value),
                    replacement.unit,
                    _json(replacement.qualifier),
                    replacement.quote,
                    replacement.char_start,
                    replacement.char_end,
                    replacement.confidence,
                    replacement.observed_at,
                    replacement.method,
                    verified_by,
                    _json(notes),
                ),
            ).fetchone()
            assert new_row is not None
            linked_row = conn.execute(
                """
                UPDATE claims SET superseded_by = %s
                WHERE id = %s AND superseded_by IS NULL
                RETURNING *
                """,
                (new_row["id"], old_claim_id),
            ).fetchone()
            if linked_row is None:
                raise ValueError(f"claim {old_claim_id} is already superseded")
        return _claim(new_row), _claim(linked_row)

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

    def claims_hash(self, entity_id: int) -> str:
        """Deterministic hash of the live claims for one entity.

        Written to ``er_audit.claims_hash`` so a merge decision can be tied to
        the exact evidence visible when it was taken (mirrors ``scores``).
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT predicate, value_json, unit, qualifier_json, quote, char_start,
                       char_end, confidence, method, observed_at
                FROM claims
                WHERE entity_id = %s AND superseded_by IS NULL
                ORDER BY id
                """,
                (entity_id,),
            ).fetchall()
        payload = [
            {
                "predicate": row["predicate"],
                "value": row["value_json"],
                "unit": row["unit"],
                "qualifier": row["qualifier_json"],
                "quote": row["quote"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "confidence": row["confidence"],
                "method": row["method"],
                "observed_at": row["observed_at"].isoformat()
                if row["observed_at"] is not None
                else None,
            }
            for row in rows
        ]
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ProgrammesRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def create(self, programme: ProgrammeCreate) -> ProgrammeRecord:
        normalised = programme.normalised_name or _fallback_normalised_name(programme.name)
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO programmes
                    (institution_id, name, credential_level, duration_months,
                     internship_months, affiliated_university_id, recognition_status, seats,
                     medium, normalised_name, field_sources, needs_review, review_reasons,
                     source_family, in_scope, scope_reason, scope_tier, scope_evidence,
                     scope_model, scope_prompt_version, scope_cached, scope_decided_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s)
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
                    normalised,
                    _json(programme.field_sources),
                    programme.needs_review,
                    _json(programme.review_reasons),
                    programme.source_family,
                    programme.in_scope,
                    programme.scope_reason,
                    programme.scope_tier,
                    _json(programme.scope_evidence),
                    programme.scope_model,
                    programme.scope_prompt_version,
                    programme.scope_cached,
                    programme.scope_decided_at,
                ),
            ).fetchone()
        assert row is not None
        return _programme(row)

    def upsert(self, programme: ProgrammeCreate) -> tuple[ProgrammeRecord, bool]:
        """Insert or refresh one (institution, normalised name) row.

        Refresh means *replace with the materialised draft*, including NULLs:
        an attribute whose claim disappeared must not linger. Returns
        ``(record, created)``. Deterministic and idempotent over the same
        evidence.
        """
        normalised = programme.normalised_name or _fallback_normalised_name(programme.name)
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO programmes
                    (institution_id, name, credential_level, duration_months,
                     internship_months, affiliated_university_id, recognition_status, seats,
                     medium, normalised_name, field_sources, needs_review, review_reasons,
                     source_family, in_scope, scope_reason, scope_tier, scope_evidence,
                     scope_model, scope_prompt_version, scope_cached, scope_decided_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (institution_id, normalised_name) DO UPDATE SET
                    name = EXCLUDED.name,
                    credential_level = EXCLUDED.credential_level,
                    duration_months = EXCLUDED.duration_months,
                    internship_months = EXCLUDED.internship_months,
                    affiliated_university_id = EXCLUDED.affiliated_university_id,
                    recognition_status = EXCLUDED.recognition_status,
                    seats = EXCLUDED.seats,
                    medium = EXCLUDED.medium,
                    field_sources = EXCLUDED.field_sources,
                    needs_review = EXCLUDED.needs_review,
                    review_reasons = EXCLUDED.review_reasons,
                    source_family = EXCLUDED.source_family,
                    in_scope = EXCLUDED.in_scope,
                    scope_reason = EXCLUDED.scope_reason,
                    scope_tier = EXCLUDED.scope_tier,
                    scope_evidence = EXCLUDED.scope_evidence,
                    scope_model = EXCLUDED.scope_model,
                    scope_prompt_version = EXCLUDED.scope_prompt_version,
                    scope_cached = EXCLUDED.scope_cached,
                    scope_decided_at = EXCLUDED.scope_decided_at
                RETURNING *, (xmax = 0) AS inserted
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
                    normalised,
                    _json(programme.field_sources),
                    programme.needs_review,
                    _json(programme.review_reasons),
                    programme.source_family,
                    programme.in_scope,
                    programme.scope_reason,
                    programme.scope_tier,
                    _json(programme.scope_evidence),
                    programme.scope_model,
                    programme.scope_prompt_version,
                    programme.scope_cached,
                    programme.scope_decided_at,
                ),
            ).fetchone()
        assert row is not None
        created = bool(row.pop("inserted", False))
        return _programme(row), created

    def materialisation_claims(self) -> list[dict[str, Any]]:
        """Evidence claims a programme may be materialised from.

        Only positively verified, unsuperseded, non-modelled claims whose
        document is ``role='evidence'`` and whose entity is an institution.
        A navigation/inventory payload is an index, never evidence.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.entity_id, c.predicate, c.value_json AS value, c.unit,
                       c.qualifier_json AS qualifier, c.confidence, c.observed_at,
                       c.method, c.extraction_id, c.superseded_by, c.quote,
                       c.verification_status, d.role AS document_role
                FROM claims c
                JOIN extractions x ON x.id = c.extraction_id
                JOIN snapshots s ON s.id = x.snapshot_id
                JOIN documents d ON d.id = s.document_id
                JOIN entities e ON e.id = c.entity_id
                WHERE c.verification_status = 'supported'
                  AND c.superseded_by IS NULL
                  AND c.method <> 'modelled'
                  AND d.role = 'evidence'
                  AND e.kind = 'institution'
                ORDER BY c.id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        """Programme totals used by a run summary (not by scoring)."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE needs_review) AS needs_review,
                       count(*) FILTER (WHERE duration_months IS NOT NULL) AS with_duration,
                       count(*) FILTER (WHERE internship_months IS NOT NULL) AS with_internship,
                       count(*) FILTER (WHERE seats IS NOT NULL) AS with_seats,
                       count(*) FILTER (WHERE recognition_status <> 'unknown') AS with_recognition
                FROM programmes
                """
            ).fetchone()
        return {key: int(value or 0) for key, value in dict(row).items()}

    def get(self, programme_id: int) -> ProgrammeRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM programmes WHERE id = %s", (programme_id,)).fetchone()
        return _programme(row) if row else None

    def get_by_key(self, institution_id: int, normalised_name: str) -> ProgrammeRecord | None:
        """The stored row for one (institution, ER-normalised name) key."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM programmes WHERE institution_id = %s AND normalised_name = %s",
                (institution_id, normalised_name),
            ).fetchone()
        return _programme(row) if row else None

    def evidence_for_programme(self, programme_id: int) -> list[ProgrammeFieldEvidence]:
        """Every ``field_sources`` entry joined to its claim, quote and page.

        One row per proving predicate, ordered by predicate key so the output
        is stable. This is the reviewer's audit path for
        ``fmd programmes inspect``: attribute -> claim id -> quote -> URL.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT fs.key AS predicate,
                       fs.value::bigint AS claim_id,
                       c.value_json AS value, c.unit, c.quote,
                       c.char_start, c.char_end, c.confidence,
                       c.verification_status,
                       x.page_range, d.url, s.fetched_at, s.sha256
                FROM programmes p
                CROSS JOIN LATERAL jsonb_each_text(p.field_sources) AS fs(key, value)
                LEFT JOIN claims c ON c.id = fs.value::bigint
                LEFT JOIN extractions x ON x.id = c.extraction_id
                LEFT JOIN snapshots s ON s.id = x.snapshot_id
                LEFT JOIN documents d ON d.id = s.document_id
                WHERE p.id = %s
                ORDER BY fs.key
                """,
                (programme_id,),
            ).fetchall()
        return [ProgrammeFieldEvidence(**dict(row)) for row in rows]

    def list_for_institution(self, institution_id: int) -> list[ProgrammeRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM programmes WHERE institution_id = %s ORDER BY id",
                (institution_id,),
            ).fetchall()
        return [_programme(row) for row in rows]

    def list_all(self) -> list[ProgrammeRecord]:
        """Every programme row, id order (requalification reads them all)."""
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM programmes ORDER BY id").fetchall()
        return [_programme(row) for row in rows]

    def update_scope(
        self,
        programme_id: int,
        *,
        in_scope: bool,
        scope_reason: str | None,
        scope_tier: str | None,
        scope_evidence: list[str],
        scope_model: str | None,
        scope_prompt_version: str | None,
        scope_cached: bool,
        needs_review: bool,
        review_reasons: list[str],
        scope_decided_at: datetime | None = None,
        normalised_name: str | None = None,
        name: str | None = None,
    ) -> ProgrammeRecord | None:
        """Persist one row's scoping outcome (materialise writes via upsert).

        Only scope, review and (when requalifying pre-W9 rows) grouping-key
        columns change here: materialised attributes and their
        ``field_sources`` are owned by the materialisation write path.
        ``normalised_name``/``name`` are the requalify rekey: a row grouped
        before the numbering-strip fix moves to its recomputed key (the merge
        step has already removed any row that holds the target key).
        """
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE programmes SET
                    in_scope = %s,
                    scope_reason = %s,
                    scope_tier = %s,
                    scope_evidence = %s,
                    scope_model = %s,
                    scope_prompt_version = %s,
                    scope_cached = %s,
                    scope_decided_at = COALESCE(%s, now()),
                    needs_review = %s,
                    review_reasons = %s,
                    normalised_name = COALESCE(%s, normalised_name),
                    name = COALESCE(%s, name)
                WHERE id = %s
                RETURNING *
                """,
                (
                    in_scope,
                    scope_reason,
                    scope_tier,
                    _json(scope_evidence),
                    scope_model,
                    scope_prompt_version,
                    scope_cached,
                    scope_decided_at,
                    needs_review,
                    _json(review_reasons),
                    normalised_name,
                    name,
                    programme_id,
                ),
            ).fetchone()
        return _programme(row) if row else None

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
        base = " FROM programmes p JOIN entities e ON e.id = p.institution_id" + where
        facet_sql = {
            "credential_level": (
                "SELECT p.credential_level AS key, count(*) AS n"
                + base
                + " GROUP BY p.credential_level ORDER BY p.credential_level"
            ),
            "recognition_status": (
                "SELECT p.recognition_status AS key, count(*) AS n"
                + base
                + " GROUP BY p.recognition_status ORDER BY p.recognition_status"
            ),
            "state": (
                "SELECT e.state AS key, count(*) AS n" + base + " GROUP BY e.state ORDER BY e.state"
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

    # ------------------------------------------------------------- scope (0010)

    def scope_decision_get(self, cache_key: str) -> ProgrammeScopeDecision | None:
        """The stored decision for one cache key (classifier cache or manual)."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM programme_scope_decisions WHERE cache_key = %s",
                (cache_key,),
            ).fetchone()
        return _scope_decision(row) if row else None

    def scope_decision_put(
        self, decision: ProgrammeScopeDecisionCreate
    ) -> tuple[ProgrammeScopeDecision, bool]:
        """Insert (or refresh) the decision for a cache key. Never deletes.

        A human decision overwrites the classifier's for the same key — manual
        always wins — but the previous row's *content* is preserved in the
        caller's audit trail (the CLI prints it); the table keeps one row per
        key so lookups stay exact. Returns ``(record, created)``.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO programme_scope_decisions
                    (cache_key, normalised_name, raw_name, institution_type, country,
                     decision, tier, confidence, rationale, evidence, model, prompt_version,
                     cached, needs_review, flags, definition_version, decided_by, note)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cache_key) DO UPDATE SET
                    normalised_name = EXCLUDED.normalised_name,
                    raw_name = EXCLUDED.raw_name,
                    institution_type = EXCLUDED.institution_type,
                    country = EXCLUDED.country,
                    decision = EXCLUDED.decision,
                    tier = EXCLUDED.tier,
                    confidence = EXCLUDED.confidence,
                    rationale = EXCLUDED.rationale,
                    evidence = EXCLUDED.evidence,
                    model = EXCLUDED.model,
                    prompt_version = EXCLUDED.prompt_version,
                    cached = EXCLUDED.cached,
                    needs_review = EXCLUDED.needs_review,
                    flags = EXCLUDED.flags,
                    definition_version = EXCLUDED.definition_version,
                    decided_by = EXCLUDED.decided_by,
                    note = EXCLUDED.note
                RETURNING *, (xmax = 0) AS inserted
                """,
                (
                    decision.cache_key,
                    decision.normalised_name,
                    decision.raw_name,
                    decision.institution_type,
                    decision.country,
                    decision.decision,
                    decision.tier,
                    decision.confidence,
                    decision.rationale,
                    _json(decision.evidence),
                    decision.model,
                    decision.prompt_version,
                    decision.cached,
                    decision.needs_review,
                    _json(decision.flags),
                    decision.definition_version,
                    decision.decided_by,
                    decision.note,
                ),
            ).fetchone()
        assert row is not None
        created = bool(row.pop("inserted", False))
        return _scope_decision(row), created

    def scope_decisions_needing_review(self, limit: int = 100) -> list[ProgrammeScopeDecision]:
        """The `uncertain` decisions a human must settle (the review queue)."""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM programme_scope_decisions
                WHERE needs_review
                ORDER BY id
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [_scope_decision(row) for row in rows]

    def scope_counts(self) -> dict[str, int]:
        """Scoping totals for summaries: in/out of scope, tiers, queue size."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT count(*) FILTER (WHERE in_scope) AS in_scope,
                       count(*) FILTER (WHERE NOT in_scope) AS out_of_scope,
                       count(*) FILTER (WHERE scope_tier = 'rule') AS tier_rule,
                       count(*) FILTER (WHERE scope_tier = 'semantic') AS tier_semantic,
                       count(*) FILTER (WHERE scope_tier = 'manual') AS tier_manual,
                       (SELECT count(*) FROM programme_scope_decisions WHERE needs_review)
                           AS review_queue
                FROM programmes
                """
            ).fetchone()
        return {key: int(value or 0) for key, value in dict(row).items()}

    def merge_into(self, source_id: int, target_id: int) -> dict[str, int]:
        """Fold duplicate ``source`` row into ``target`` and delete it.

        Programmes rows are materialised views of claims, so the duplicate row
        itself is deleted — but nothing it references is lost: ``cost_items``
        are re-pointed at the surviving row first and the counts are returned
        for the audit trail. Claims are untouched (append-only evidence).
        """
        with self.db.connection() as conn:
            cost_moved = conn.execute(
                "UPDATE cost_items SET programme_id = %s WHERE programme_id = %s",
                (target_id, source_id),
            ).rowcount
            deleted = conn.execute("DELETE FROM programmes WHERE id = %s", (source_id,)).rowcount
        return {"cost_items_moved": int(cost_moved or 0), "rows_deleted": int(deleted or 0)}


def _fallback_normalised_name(name: str) -> str:
    """Stable grouping key for callers that predate migration 0006.

    The materialisation stage supplies the real ER-normalised key; this is only
    a deterministic fallback so a legacy ``create()`` cannot blow up on the
    ``normalised_name NOT NULL`` constraint.
    """
    return " ".join(str(name or "").casefold().split())


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
    if filters.in_scope is not None:
        conditions.append("p.in_scope = %s")
        params.append(filters.in_scope)
    return (" WHERE " + " AND ".join(conditions) if conditions else ""), params


def _scope_decision(row: dict[str, Any]) -> ProgrammeScopeDecision:
    return ProgrammeScopeDecision.model_validate(row)


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

    def invalidated_score_ids(self, entity_id: int) -> set[int]:
        """Score ids a merge/unmerge recorded as invalidated for ``entity_id``.

        Invalidation lives in the append-only ``er_audit`` row rather than a new
        mutable column, so the merge decision and the scores it invalidated stay
        in one audit record.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT features_json FROM er_audit
                WHERE (subject_entity_id = %s OR candidate_entity_id = %s)
                  AND features_json ? 'scores_invalidated'
                """,
                (entity_id, entity_id),
            ).fetchall()
        invalidated: set[int] = set()
        for row in rows:
            block = (row["features_json"] or {}).get("scores_invalidated") or {}
            for key in ("source", "target"):
                invalidated.update(int(value) for value in block.get(key) or [])
        return invalidated

    def latest_for_entity(
        self, entity_id: int, *, include_stale: bool = False
    ) -> ScoreRecord | None:
        invalidated = self.invalidated_score_ids(entity_id)
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scores WHERE entity_id = %s ORDER BY computed_at DESC, id DESC",
                (entity_id,),
            ).fetchall()
        for row in rows:
            record = _score(row)
            if record.id in invalidated:
                if not include_stale:
                    continue
                record = record.model_copy(update={"stale": True})
            return record
        return None

    def list_for_entity(self, entity_id: int, *, include_stale: bool = True) -> list[ScoreRecord]:
        invalidated = self.invalidated_score_ids(entity_id)
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scores WHERE entity_id = %s ORDER BY computed_at DESC, id DESC",
                (entity_id,),
            ).fetchall()
        records: list[ScoreRecord] = []
        for row in rows:
            record = _score(row)
            stale = record.id in invalidated
            if stale:
                if not include_stale:
                    continue
                record = record.model_copy(update={"stale": True})
            records.append(record)
        return records


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
                "SELECT * FROM llm_calls WHERE purpose = %s "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
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


class PipelineRunsRepo:
    """The SQL run ledger: one row per stage execution (migration 0003)."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def record(self, run: PipelineRunCreate) -> PipelineRunRecord:
        # Idempotent on run_id: re-recording the same run updates its terminal
        # state instead of duplicating a row (a worker may retry a ledger write).
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO pipeline_runs
                    (run_id, stage, started_at, finished_at, counts_json, errors_json, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    stage = EXCLUDED.stage,
                    started_at = EXCLUDED.started_at,
                    finished_at = EXCLUDED.finished_at,
                    counts_json = EXCLUDED.counts_json,
                    errors_json = EXCLUDED.errors_json,
                    status = EXCLUDED.status
                RETURNING *
                """,
                (
                    run.run_id,
                    run.stage,
                    run.started_at,
                    run.finished_at,
                    _json(run.counts),
                    _json(run.errors),
                    run.status,
                ),
            ).fetchone()
        assert row is not None
        return _pipeline_run(row)

    def get(self, run_id: str) -> PipelineRunRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM pipeline_runs WHERE run_id = %s", (run_id,)
            ).fetchone()
        return _pipeline_run(row) if row else None

    def list(
        self,
        *,
        stage: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[PipelineRunRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if stage is not None:
            conditions.append("stage = %s")
            params.append(stage)
        if status is not None:
            conditions.append("status = %s")
            params.append(status)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM pipeline_runs{where} "  # noqa: S608
                "ORDER BY started_at DESC, run_id LIMIT %s",
                tuple(params),
            ).fetchall()
        return [_pipeline_run(row) for row in rows]


class DocumentParsesRepo:
    """The parse-cache lookup table (migration 0011, W10 batched acquire).

    Rows point at content-addressed payload objects in the snapshot store; the
    write order is always object first, row second, so an interrupted write
    leaves an unreachable orphan object (harmless, content-addressed) and never
    a row that reads back as a hit.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    def put(self, parse: DocumentParsePut) -> DocumentParseRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO document_parses
                    (key_digest, snapshot_sha256, tier, policy_digest, schema_version,
                     object_sha256, pages, parser, parser_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (key_digest) DO UPDATE SET
                    object_sha256 = EXCLUDED.object_sha256,
                    pages = EXCLUDED.pages,
                    parser = EXCLUDED.parser,
                    parser_version = EXCLUDED.parser_version
                RETURNING *
                """,
                (
                    parse.key_digest,
                    parse.snapshot_sha256,
                    parse.tier,
                    parse.policy_digest,
                    parse.schema_version,
                    parse.object_sha256,
                    parse.pages,
                    parse.parser,
                    parse.parser_version,
                ),
            ).fetchone()
        assert row is not None
        return _document_parse(row)

    def get(self, key_digest: str) -> DocumentParseRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM document_parses WHERE key_digest = %s", (key_digest,)
            ).fetchone()
        return _document_parse(row) if row else None

    def count(self) -> int:
        with self.db.connection() as conn:
            row = conn.execute("SELECT count(*) AS n FROM document_parses").fetchone()
        return int(row["n"]) if row else 0


class CrawlFrontierRepo:
    def __init__(self, database: Database) -> None:
        self.db = database

    def enqueue(self, item: FrontierUpsert) -> FrontierRecord:
        # Insert/dedupe only. ``LEAST`` keeps the earliest schedule for a URL
        # discovered twice; a retry must use ``set_next_allowed_at`` instead so
        # backoff can move the URL later.
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO crawl_frontier
                    (url, domain, priority, next_allowed_at, state, http_method, http_body,
                     request_headers, source_id, extraction_family, entity_id)
                VALUES (%s, %s, %s, COALESCE(%s, now()), %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    domain = EXCLUDED.domain,
                    priority = EXCLUDED.priority,
                    next_allowed_at = LEAST(crawl_frontier.next_allowed_at,
                                            EXCLUDED.next_allowed_at),
                    state = CASE WHEN crawl_frontier.state = 'done'
                                 THEN crawl_frontier.state ELSE EXCLUDED.state END,
                    http_method = CASE
                        WHEN EXCLUDED.http_method = 'GET' AND crawl_frontier.http_method = 'POST'
                             AND EXCLUDED.http_body IS NULL
                        THEN crawl_frontier.http_method ELSE EXCLUDED.http_method END,
                    http_body = COALESCE(EXCLUDED.http_body, crawl_frontier.http_body),
                    request_headers = COALESCE(EXCLUDED.request_headers,
                                               crawl_frontier.request_headers),
                    source_id = COALESCE(EXCLUDED.source_id, crawl_frontier.source_id),
                    extraction_family = COALESCE(EXCLUDED.extraction_family,
                                                 crawl_frontier.extraction_family),
                    entity_id = COALESCE(EXCLUDED.entity_id, crawl_frontier.entity_id),
                    updated_at = now()
                RETURNING *
                """,
                (
                    item.url,
                    item.domain,
                    item.priority,
                    item.next_allowed_at,
                    item.state,
                    item.http_method,
                    _json(item.http_body),
                    _json(item.request_headers),
                    item.source_id,
                    item.extraction_family,
                    item.entity_id,
                ),
            ).fetchone()
        assert row is not None
        return FrontierRecord.model_validate(row)

    def set_next_allowed_at(self, url: str, when: datetime) -> FrontierRecord | None:
        """Unconditional schedule; the only way to push a retry later."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE crawl_frontier SET next_allowed_at = %s, updated_at = now()
                WHERE url = %s
                RETURNING *
                """,
                (when, url),
            ).fetchone()
        return FrontierRecord.model_validate(row) if row else None

    def bump_attempts(self, url: str, by: int = 1) -> FrontierRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE crawl_frontier SET attempts = attempts + %s, updated_at = now()
                WHERE url = %s
                RETURNING *
                """,
                (by, url),
            ).fetchone()
        return FrontierRecord.model_validate(row) if row else None

    def get(self, url: str) -> FrontierRecord | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM crawl_frontier WHERE url = %s", (url,)).fetchone()
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

    def count_for_domain_since(self, domain: str, since: datetime) -> int:
        """Pages completed for a domain since ``since`` (daily budget denominator)."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT count(*) AS n FROM crawl_frontier
                WHERE domain = %s AND state = 'done' AND updated_at >= %s
                """,
                (domain, since),
            ).fetchone()
        return int(row["n"]) if row else 0

    def update_yield(self, domain: str, pages: int, yield_score: float) -> None:
        """Record the yield of the page just fetched for ``domain``.

        The schema keeps yield per frontier row; the most recently touched row
        is the page that produced it. ``pages`` is the denominator the ratio was
        computed from and is ignored when it is zero.
        """
        if pages <= 0:
            return
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE crawl_frontier SET yield_score = %s
                WHERE url = (
                    SELECT url FROM crawl_frontier WHERE domain = %s
                    ORDER BY updated_at DESC, url DESC LIMIT 1
                )
                """,
                (max(0.0, min(1.0, yield_score)), domain),
            )

    def yield_for_domain(self, domain: str) -> tuple[int, float]:
        """``(pages_fetched, summed_yield)`` for the whole crawl of a domain."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT count(*) AS n, COALESCE(sum(yield_score), 0.0) AS valuable
                FROM crawl_frontier
                WHERE domain = %s AND state = 'done'
                """,
                (domain,),
            ).fetchone()
        if not row:
            return 0, 0.0
        return int(row["n"]), float(row["valuable"])

    def record_tier(self, domain: str, tier: str) -> None:
        """Attribute the tier that answered to the domain's latest frontier row."""
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE crawl_frontier SET tier_used = %s
                WHERE url = (
                    SELECT url FROM crawl_frontier WHERE domain = %s
                    ORDER BY updated_at DESC, url DESC LIMIT 1
                )
                """,
                (tier, domain),
            )

    def tier_stats(self, domain: str) -> dict[str, int]:
        """How often each fetch tier answered for a domain (ban-risk metric)."""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(tier_used, 'none') AS tier, count(*) AS n
                FROM crawl_frontier
                WHERE domain = %s
                GROUP BY COALESCE(tier_used, 'none')
                ORDER BY tier
                """,
                (domain,),
            ).fetchall()
        return {row["tier"]: int(row["n"]) for row in rows}

    def quarantine(self, domain: str, until: datetime) -> None:
        """Set/extend a domain quarantine; an existing later deadline wins."""
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO domain_politeness (domain, quarantined_until)
                VALUES (%s, %s)
                ON CONFLICT (domain) DO UPDATE SET
                    quarantined_until = GREATEST(domain_politeness.quarantined_until,
                                                 EXCLUDED.quarantined_until)
                """,
                (domain, until),
            )

    def is_quarantined(self, domain: str, at: datetime | None = None) -> bool:
        at = at or datetime.now(UTC)
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT quarantined_until > %s AS active
                FROM domain_politeness WHERE domain = %s
                """,
                (at, domain),
            ).fetchone()
        return bool(row["active"]) if row else False

    def get_politeness(self, domain: str) -> DomainPolitenessRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM domain_politeness WHERE domain = %s", (domain,)
            ).fetchone()
        return _politeness(row) if row else None

    def upsert_politeness(self, state: DomainPolitenessRecord) -> DomainPolitenessRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO domain_politeness
                    (domain, rps, crawl_delay, robots_json, robots_fetched_at,
                     quarantined_until)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (domain) DO UPDATE SET
                    rps = EXCLUDED.rps,
                    crawl_delay = EXCLUDED.crawl_delay,
                    robots_json = EXCLUDED.robots_json,
                    robots_fetched_at = EXCLUDED.robots_fetched_at,
                    quarantined_until = EXCLUDED.quarantined_until
                RETURNING *
                """,
                (
                    state.domain,
                    state.rps,
                    state.crawl_delay,
                    _json(state.robots),
                    state.robots_fetched_at,
                    state.quarantined_until,
                ),
            ).fetchone()
        assert row is not None
        return _politeness(row)


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


def _institution_domain(row: dict[str, Any]) -> InstitutionDomainRecord:
    return InstitutionDomainRecord.model_validate(row)


def _document(row: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord.model_validate(row)


def _politeness(row: dict[str, Any]) -> DomainPolitenessRecord:
    return DomainPolitenessRecord(
        domain=row["domain"],
        rps=row["rps"],
        crawl_delay=row["crawl_delay"],
        robots=row["robots_json"],
        robots_fetched_at=row["robots_fetched_at"],
        quarantined_until=row["quarantined_until"],
    )


def _snapshot(row: dict[str, Any]) -> SnapshotRecord:
    return SnapshotRecord(
        document_id=row["document_id"],
        sha256=row["sha256"],
        bytes_path=row["bytes_path"],
        fetched_at=row["fetched_at"],
        fetcher_tier=row["fetcher_tier"],
        content_type=row["content_type"],
        headers=row["headers_json"],
        text_sha256=row.get("text_sha256"),
        dedupe_basis=row.get("dedupe_basis"),
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


def _section_node(row: dict[str, Any]) -> SectionNodeRecord:
    bbox = row.get("bbox_json")
    bbox = list(bbox) if bbox is not None else None
    return SectionNodeRecord(
        id=row["id"],
        document_id=row["document_id"],
        parent_id=row["parent_id"],
        node_key=row["node_key"],
        level=row["level"],
        title=row["title"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        heading_char_start=row["heading_char_start"],
        heading_char_end=row["heading_char_end"],
        page_no=row["page_no"],
        bbox=bbox,
    )


def _er_audit(row: dict[str, Any]) -> ERAuditRecord:
    return ERAuditRecord(
        subject_entity_id=row["subject_entity_id"],
        candidate_entity_id=row["candidate_entity_id"],
        band=row["band"],
        score=row["score"],
        features=row["features_json"] or {},
        thresholds_version=row["thresholds_version"],
        model=row["model"],
        model_verdict=row["model_verdict"],
        model_reason=row["model_reason"],
        decided_by=row["decided_by"],
        claims_hash=row["claims_hash"],
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
        verification_status=row.get("verification_status") or "unverified",
        verified_by=row.get("verified_by"),
        verified_at=row.get("verified_at"),
        verification_notes=row.get("verification_notes"),
    )


def _pipeline_run(row: dict[str, Any]) -> PipelineRunRecord:
    return PipelineRunRecord(
        run_id=row["run_id"],
        stage=row["stage"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        counts=row["counts_json"] or {},
        errors=row["errors_json"] or [],
        status=row["status"],
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


def _document_parse(row: dict[str, Any]) -> DocumentParseRecord:
    return DocumentParseRecord.model_validate(row)
