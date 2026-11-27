from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from fmd.db import (
    AliasCreate,
    AliasesRepo,
    ClaimCreate,
    ClaimsRepo,
    CostItemCreate,
    CostItemsRepo,
    CountriesRepo,
    CountrySeed,
    CrawlFrontierRepo,
    DocumentsRepo,
    DocumentUpsert,
    EntitiesRepo,
    EntityCreate,
    ExamSeed,
    ExamsRepo,
    ExtractionCreate,
    ExtractionsRepo,
    FinancingOptionCreate,
    FinancingOptionsRepo,
    FrontierUpsert,
    LLMCallLog,
    LLMCallsRepo,
    ProgrammeCreate,
    ProgrammeFilter,
    ProgrammesRepo,
    ScoreCreate,
    ScoresRepo,
    SnapshotCreate,
    SnapshotsRepo,
    SourceExclusionSeed,
    SourceExclusionsRepo,
    SourceSeed,
    SourcesRepo,
    applied_migrations,
    discover_migrations,
    run_migrations,
)
from fmd.db.repositories import DomainPolitenessRecord
from fmd.fetch.types import FetchRequest, http_date

pytestmark = pytest.mark.usefixtures("pg_database")


def _suffix() -> str:
    return uuid.uuid4().hex[:12]


def _country(database) -> None:
    CountriesRepo(database).upsert(
        CountrySeed(
            code="IN",
            name="India",
            enabled=True,
            currency="INR",
            tier="A",
            languages=["en", "hi"],
            capabilities={"programme_inventory": True},
        )
    )


def test_migrations_are_idempotent_and_match_the_files_on_disk(pg_database) -> None:
    assert run_migrations(pg_database) == []
    applied = applied_migrations(pg_database)
    on_disk = {migration.version: migration.checksum for migration in discover_migrations()}
    # Never assert a migration *count*: every new forward-only file would fail it
    # and the quickest fix would be to stop checking the schema at all.
    assert applied == on_disk


def test_evidence_and_claim_supersede_roundtrip(pg_database) -> None:
    _country(pg_database)
    suffix = _suffix()
    entity = EntitiesRepo(pg_database).create(
        EntityCreate(
            kind="institution",
            canonical_name=f"Test Institute {suffix}",
            country="IN",
            state="MH",
            city="Pune",
        )
    )
    alias = AliasesRepo(pg_database).create(
        AliasCreate(entity_id=entity.id, alias=f"TI{suffix}", alias_type="acronym")
    )
    assert alias.entity_id == entity.id

    url = f"https://example.ac.in/placement/{suffix}.pdf"
    document = DocumentsRepo(pg_database).upsert(
        DocumentUpsert(url=url, domain="example.ac.in", kind="placement_report")
    )
    data = b"%PDF-1.4 fake evidence"
    digest = hashlib.sha256(data).hexdigest()
    snapshot = SnapshotsRepo(pg_database).create(
        SnapshotCreate(
            document_id=document.id,
            sha256=digest,
            bytes_path=f"objects/{digest[:2]}/{digest}",
            fetcher_tier="httpx",
            content_type="application/pdf",
        )
    )
    text = "Internship-to-offer conversion was 60% (n=120) in 2025."
    extraction = ExtractionsRepo(pg_database).create(
        ExtractionCreate(
            snapshot_id=snapshot.id,
            tool="test",
            text=text,
            prompt_version="v1",
        )
    )
    start = text.index("60%")
    claims = ClaimsRepo(pg_database)
    original = claims.create(
        ClaimCreate(
            extraction_id=extraction.id,
            entity_id=entity.id,
            predicate="internship_to_offer_conversion_12m",
            value={"rate": 0.6},
            unit="ratio",
            qualifier={"denominator": 120},
            quote=text[start : start + 3],
            char_start=start,
            char_end=start + 3,
            confidence=0.8,
        )
    )
    replacement = claims.create(
        ClaimCreate(
            extraction_id=extraction.id,
            entity_id=entity.id,
            predicate="internship_to_offer_conversion_12m",
            value={"rate": 0.6},
            unit="ratio",
            qualifier={"denominator": 120},
            quote=text[start : start + 3],
            char_start=start,
            char_end=start + 3,
            confidence=0.9,
            method="derived",
        )
    )
    superseded = claims.supersede(original.id, replacement.id)
    assert superseded.superseded_by == replacement.id
    assert superseded.quote == original.quote  # never mutated

    current = claims.list_for_entity(entity.id)
    assert [row.id for row in current] == [replacement.id]
    assert claims.count_for_entity(entity.id) == 1

    page = claims.evidence_page(entity.id)
    assert page.total == 1
    assert page.items[0].url == url
    assert page.items[0].sha256 == digest
    assert page.items[0].snapshot_id == snapshot.id


def test_programme_scores_costs_financing_exams(pg_database) -> None:
    _country(pg_database)
    suffix = _suffix()
    institution = EntitiesRepo(pg_database).create(
        EntityCreate(
            kind="institution",
            canonical_name=f"Institute {suffix}",
            country="IN",
            state="KA",
        )
    )
    university = EntitiesRepo(pg_database).create(
        EntityCreate(kind="institution", canonical_name=f"University {suffix}", country="IN")
    )
    programme = ProgrammesRepo(pg_database).create(
        ProgrammeCreate(
            institution_id=institution.id,
            name=f"BSc MLT {suffix}",
            credential_level="degree",
            duration_months=48,
            internship_months=12,
            affiliated_university_id=university.id,
            recognition_status="recognised",
            seats=60,
            medium="English",
        )
    )
    listed = ProgrammesRepo(pg_database).list(
        ProgrammeFilter(state="KA", credential_level="degree", query="BSc MLT")
    )
    assert any(row.id == programme.id for row in listed)
    assert ProgrammesRepo(pg_database).count(ProgrammeFilter(institution_id=institution.id)) == 1
    facets = ProgrammesRepo(pg_database).facets(ProgrammeFilter(state="KA"))
    assert facets["credential_level"]["degree"] >= 1

    cost = CostItemsRepo(pg_database).create(
        CostItemCreate(
            programme_id=programme.id,
            category="tuition",
            label="Year 1 tuition",
            amount=100000,
            currency="INR",
            period="per_year",
        )
    )
    assert cost.amount == 100000
    assert len(CostItemsRepo(pg_database).list_for_programme(programme.id)) == 1

    financing = FinancingOptionsRepo(pg_database).create(
        FinancingOptionCreate(
            scope="domestic",
            lender="Example Bank",
            scheme="iba_model_scheme",
            rate_min=8.5,
            rate_max=10.5,
            tenure_months=180,
        )
    )
    assert financing.id
    assert FinancingOptionsRepo(pg_database).list("domestic")

    score = ScoresRepo(pg_database).create(
        ScoreCreate(
            entity_id=institution.id,
            scorer_version="0.1.0",
            weights_version="test",
            claims_hash="a" * 64,
            sub_scores={"outcomes": {"point": 50, "lo": 20, "hi": 80}},
            composite=50,
            composite_lo=20,
            composite_hi=80,
            evidence_tier="thin",
        )
    )
    latest = ScoresRepo(pg_database).latest_for_entity(institution.id)
    assert latest is not None and latest.id == score.id

    exam = ExamsRepo(pg_database).upsert(
        ExamSeed(
            id=f"exam_{suffix}",
            name="Test Entrance",
            country="IN",
            scope="state",
            provider="Test Board",
            requires_travel=True,
        )
    )
    assert exam.id == f"exam_{suffix}"
    assert ExamsRepo(pg_database).list(country="IN")


def test_source_registry_and_llm_audit_and_frontier(pg_database) -> None:
    _country(pg_database)
    suffix = _suffix()
    sources = SourcesRepo(pg_database)
    source_id = f"source_{suffix}"
    source = sources.upsert(
        SourceSeed(
            id=source_id,
            name="Test source",
            base_url="https://example.ac.in",
            kind="inventory",
            format="html",
            cadence_days=30,
            trust="high",
            adapter="ugc",
            enabled=True,
            notes="fixture",
            config={"custom": 1},
        )
    )
    assert source.id == source_id
    assert sources.get(source_id) is not None
    SourceExclusionsRepo(pg_database).append(
        SourceExclusionSeed(id=f"ex_{suffix}", name="Login wall", reason="auth required")
    )

    llm = LLMCallsRepo(pg_database)
    call_id = llm.record(
        LLMCallLog(
            purpose="bulk_extract",
            prompt_sha256="b" * 64,
            prompt_path="prompts/extract/v1.md",
            model="mock",
            provider="mock",
            params={"guided_json": True},
            raw_response='{"value": 1}',
            tokens_in=10,
            tokens_out=5,
            latency_ms=12,
            cost_usd=0.0,
            cache_hit=False,
        )
    )
    assert call_id > 0
    assert llm.list_for_purpose("bulk_extract")
    assert llm.spend_since(datetime(2000, 1, 1, tzinfo=UTC)) == 0

    frontier = CrawlFrontierRepo(pg_database)
    url = f"https://example.ac.in/{suffix}"
    frontier.enqueue(FrontierUpsert(url=url, domain="example.ac.in", priority=5))
    due = frontier.list_due(datetime.now(UTC))
    assert any(row.url == url for row in due)
    updated = frontier.set_state(url, "done", tier_used="httpx", attempts=1)
    assert updated is not None and updated.state == "done"
    assert frontier.count_by_state().get("done", 0) >= 1


def test_frontier_repo_backoff_and_domain_accounting(pg_database) -> None:
    domain = f"page-{_suffix()}.ac.in"
    url = f"https://{domain}/report.pdf"
    repo = CrawlFrontierRepo(pg_database)
    start = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    repo.enqueue(FrontierUpsert(url=url, domain=domain, next_allowed_at=start))
    retry_at = start + timedelta(minutes=5)
    moved = repo.set_next_allowed_at(url, retry_at)
    assert moved is not None and moved.next_allowed_at == retry_at

    # Dedupe picks the earlier of the two times; it never pushes a URL later.
    duplicate = repo.enqueue(FrontierUpsert(url=url, domain=domain, next_allowed_at=start))
    assert duplicate.next_allowed_at == start

    bumped = repo.bump_attempts(url)
    assert bumped is not None and bumped.attempts == 1

    repo.set_state(url, "done", tier_used="curl_cffi")
    assert repo.count_for_domain_since(domain, start) == 1
    repo.update_yield(domain, pages=4, yield_score=0.25)
    assert repo.yield_for_domain(domain) == (1, 0.25)
    repo.record_tier(domain, "invisible_playwright_mcp")
    assert repo.tier_stats(domain) == {"patchright": 1}

    until = datetime.now(UTC) + timedelta(hours=6)
    assert repo.is_quarantined(domain) is False
    repo.quarantine(domain, until)
    assert repo.is_quarantined(domain) is True
    record = repo.get_politeness(domain)
    assert record is not None and record.quarantined_until == until
    repo.upsert_politeness(
        DomainPolitenessRecord(
            domain=domain,
            rps=2.0,
            crawl_delay=1.5,
            robots={"text": "User-agent: *", "status": 200},
            robots_fetched_at=start,
            quarantined_until=None,
        )
    )
    saved = repo.get_politeness(domain)
    assert saved is not None
    assert (saved.rps, saved.crawl_delay) == (2.0, 1.5)
    assert saved.robots is not None and saved.robots["status"] == 200


def test_conditional_get_roundtrip_preserves_validators_and_bumps_last_seen(pg_database) -> None:
    documents = DocumentsRepo(pg_database)
    snapshots = SnapshotsRepo(pg_database)
    url = f"https://example.ac.in/conditional/{_suffix()}.pdf"
    first_seen = datetime(2020, 1, 1, tzinfo=UTC)

    stored = documents.upsert(
        DocumentUpsert(
            url=url,
            domain="example.ac.in",
            etag='"v1"',
            last_modified=first_seen,
            last_seen=first_seen,
        )
    )
    assert stored.etag == '"v1"'

    request = FetchRequest.revalidate(
        stored.url, etag=stored.etag, last_modified=stored.last_modified
    )
    assert request.etag == '"v1"'
    assert request.last_modified == http_date(first_seen)

    # A 304 writes no snapshot; the caller records the revalidation instead.
    revalidated = documents.upsert(DocumentUpsert(url=url, domain="example.ac.in", http_status=304))
    assert revalidated.id == stored.id
    assert revalidated.etag == '"v1"'
    assert revalidated.http_status == 304
    assert revalidated.last_seen is not None
    assert revalidated.last_seen > first_seen
    assert snapshots.list_for_document(revalidated.id) == []


def test_seed_projects_registry(pg_database, registry) -> None:
    from fmd.seed import seed_database

    counts = seed_database(pg_database, registry)
    assert counts["countries"] == len(registry.countries.countries)
    assert counts["sources"] == len(registry.sources.sources)
    assert counts["exams"] == len(registry.exams.exams)
    assert CountriesRepo(pg_database).get("IN") is not None
    assert SourcesRepo(pg_database).get(registry.sources.sources[0].id) is not None
    assert ExamsRepo(pg_database).get(registry.exams.exams[0].id) is not None


def test_post_request_shape_is_replayable_on_documents_and_frontier(pg_database) -> None:
    suffix = _suffix()
    documents = DocumentsRepo(pg_database)
    url = f"https://nabh.example/admin-ajax.php?t={suffix}"
    body = {"action": "hospitals", "page": 2}
    stored = documents.upsert(
        DocumentUpsert(
            url=url,
            domain="nabh.example",
            kind="directory",
            http_method="POST",
            http_body=body,
            request_headers={"content-type": "application/x-www-form-urlencoded"},
        )
    )
    assert stored.http_method == "POST"
    assert stored.http_body == body
    assert stored.request_headers is not None
    # A later conditional revalidation written as a bare GET must not erase the
    # POST shape that makes the source replayable.
    revalidated = documents.upsert(DocumentUpsert(url=url, domain="nabh.example", http_status=304))
    assert revalidated.http_method == "POST"
    assert revalidated.http_body == body

    frontier = CrawlFrontierRepo(pg_database)
    record = frontier.enqueue(
        FrontierUpsert(
            url=f"https://nabh.example/page/{suffix}",
            domain="nabh.example",
            http_method="POST",
            http_body=body,
        )
    )
    assert record.http_method == "POST"
    assert record.http_body == body
    again = frontier.enqueue(FrontierUpsert(url=record.url, domain="nabh.example"))
    assert again.http_method == "POST"
    assert again.http_body == body


def test_document_upsert_can_be_built_from_a_document_ref(pg_database) -> None:
    from fmd.discover.base import DocumentRef

    suffix = _suffix()
    ref = DocumentRef(
        url=f"https://nabh.example/admin-ajax.php?d={suffix}",
        metadata={"request_method": "POST", "request_body": {"action": "states"}},
    )
    upsert = DocumentUpsert.from_document_ref(ref)
    assert upsert.domain == "nabh.example"
    assert upsert.http_method == "POST"
    assert upsert.http_body == {"action": "states"}
    saved = DocumentsRepo(pg_database).upsert(upsert)
    assert saved.http_body == {"action": "states"}

    frontier_upsert = FrontierUpsert.from_document_ref(ref)
    assert frontier_upsert.domain == "nabh.example"
    assert frontier_upsert.http_method == "POST"
    frontier_record = CrawlFrontierRepo(pg_database).enqueue(frontier_upsert)
    assert frontier_record.http_body == {"action": "states"}


def test_alias_accepts_manual_and_evidence_span(pg_database) -> None:
    suffix = _suffix()
    entity = EntitiesRepo(pg_database).create(
        EntityCreate(kind="institution", canonical_name=f"Span Institute {suffix}")
    )
    manual = AliasesRepo(pg_database).create(
        AliasCreate(
            entity_id=entity.id,
            alias=f"Span Vishwavidyalaya {suffix}",
            alias_type="manual",
            span_start=4,
            span_end=31,
            quote=f"Span Vishwavidyalaya {suffix}",
            confirmed_by="reviewer@example.org",
            confirmed_at=datetime.now(UTC),
        )
    )
    assert manual.alias_type == "manual"
    assert (manual.span_start, manual.span_end) == (4, 31)
    assert manual.confirmed_by == "reviewer@example.org"
    listed = AliasesRepo(pg_database).list_manual_for_entity(entity.id)
    assert [row.id for row in listed] == [manual.id]


# --------------------------------------------------------------- merge integrity

CLAIM_CONTENT_COLUMNS = (
    "extraction_id",
    "predicate",
    "value_json",
    "unit",
    "qualifier_json",
    "quote",
    "char_start",
    "char_end",
    "confidence",
    "observed_at",
    "method",
    "superseded_by",
)


def _extraction(database, suffix: str) -> tuple[int, str]:
    url = f"https://example.ac.in/merge/{suffix}.pdf"
    document = DocumentsRepo(database).upsert(
        DocumentUpsert(url=url, domain="example.ac.in", kind="fee_schedule")
    )
    text = f"Tuition for {suffix} is INR 100000 per year."
    data = text.encode()
    digest = hashlib.sha256(data).hexdigest()
    snapshot = SnapshotsRepo(database).create(
        SnapshotCreate(document_id=document.id, sha256=digest, bytes_path=f"o/{digest}")
    )
    extraction = ExtractionsRepo(database).create(
        ExtractionCreate(snapshot_id=snapshot.id, tool="test", text=text)
    )
    return extraction.id, text


def _add_claim(database, entity_id: int, extraction_id: int, text: str, confidence: float = 0.9):
    return ClaimsRepo(database).create(
        ClaimCreate(
            extraction_id=extraction_id,
            entity_id=entity_id,
            predicate="tuition_per_year",
            value={"amount": 100000, "currency": "INR"},
            unit="INR",
            qualifier={"period": "per_year"},
            quote=text,
            char_start=0,
            char_end=len(text),
            confidence=confidence,
        )
    )


def _claim_rows(database, entity_id: int) -> dict[int, dict]:
    with database.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE entity_id = %s ORDER BY id", (entity_id,)
        ).fetchall()
    return {int(row["id"]): dict(row) for row in rows}


def _score(database, entity_id: int, claims_hash: str) -> int:
    return (
        ScoresRepo(database)
        .create(
            ScoreCreate(
                entity_id=entity_id,
                scorer_version="0.1.0",
                weights_version="test",
                claims_hash=claims_hash,
                sub_scores={"fees": {"point": 1, "lo": 1, "hi": 1}},
                evidence_tier="thin",
            )
        )
        .id
    )


def test_merge_reparents_evidence_and_unmerge_restores_exactly(pg_database) -> None:
    suffix = _suffix()
    entities = EntitiesRepo(pg_database)
    source = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Source Institute {suffix}", country="IN")
    )
    target = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Target Institute {suffix}", country="IN")
    )
    extraction_id, text = _extraction(pg_database, suffix)
    source_claim = _add_claim(pg_database, source.id, extraction_id, text)
    target_claim = _add_claim(pg_database, target.id, extraction_id, text, confidence=0.7)

    aliases = AliasesRepo(pg_database)
    manual = aliases.create(
        AliasCreate(
            entity_id=source.id,
            alias=f"Source Vishwavidyalaya {suffix}",
            alias_type="manual",
            confirmed_by="reviewer@example.org",
            confirmed_at=datetime.now(UTC),
        )
    )
    acronym = aliases.create(
        AliasCreate(entity_id=source.id, alias=f"SI{suffix}", alias_type="acronym")
    )

    programme = ProgrammesRepo(pg_database).create(
        ProgrammeCreate(
            institution_id=source.id,
            name=f"BSc MLT {suffix}",
            credential_level="degree",
        )
    )
    cost = CostItemsRepo(pg_database).create(
        CostItemCreate(
            programme_id=programme.id,
            category="tuition",
            label="Year 1",
            amount=100000,
            currency="INR",
            period="per_year",
        )
    )
    financing = FinancingOptionsRepo(pg_database).create(
        FinancingOptionCreate(
            scope="domestic",
            lender="Example Bank",
            scheme=f"scheme-{suffix}",
            source_claim_id=source_claim.id,
        )
    )

    before = _claim_rows(pg_database, source.id)
    assert set(before) == {source_claim.id}

    _, audit = entities.merge(source.id, target.id, score=0.9, thresholds_version="er-v1")
    evidence = audit.features["evidence"]
    assert evidence["claims"] == [source_claim.id]
    assert set(evidence["aliases"]) == {manual.id, acronym.id}
    assert evidence["programmes"] == [programme.id]
    assert evidence["cost_items"] == [cost.id]
    assert evidence["financing_options"] == [financing.id]
    assert evidence["target_claims_before"] == [target_claim.id]

    # Evidence moved, content untouched.
    assert set(_claim_rows(pg_database, source.id)) == set()
    moved_rows = _claim_rows(pg_database, target.id)
    assert set(moved_rows) == {source_claim.id, target_claim.id}
    for column in CLAIM_CONTENT_COLUMNS:
        assert moved_rows[source_claim.id][column] == before[source_claim.id][column]
    assert ProgrammesRepo(pg_database).get(programme.id).institution_id == target.id

    moved_manual = AliasesRepo(pg_database).list_manual_for_entity(target.id)
    assert [(row.id, row.alias_type) for row in moved_manual] == [(manual.id, "manual")]

    result = entities.unmerge(source.id)
    assert result.ok and not result.conflict
    assert result.moved_back["claims"] == [source_claim.id]
    assert result.moved_back["programmes"] == [programme.id]

    restored = _claim_rows(pg_database, source.id)
    assert set(restored) == {source_claim.id}
    for column in CLAIM_CONTENT_COLUMNS:
        assert restored[source_claim.id][column] == before[source_claim.id][column]
    assert set(_claim_rows(pg_database, target.id)) == {target_claim.id}
    assert ProgrammesRepo(pg_database).get(programme.id).institution_id == source.id
    restored_manual = AliasesRepo(pg_database).list_manual_for_entity(source.id)
    assert [(row.id, row.alias_type) for row in restored_manual] == [(manual.id, "manual")]
    assert aliases.list_for_entity(target.id) == []
    history = entities.audit_for_pair(source.id, target.id)
    assert [row.band for row in history] == ["reject", "accept"]


def test_merge_never_alters_claim_content_and_hash_follows_membership(pg_database) -> None:
    suffix = _suffix()
    entities = EntitiesRepo(pg_database)
    source = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Hash Source {suffix}", country="IN")
    )
    target = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Hash Target {suffix}", country="IN")
    )
    blank = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Blank {suffix}", country="IN")
    )
    extraction_id, text = _extraction(pg_database, suffix)
    claims = ClaimsRepo(pg_database)
    source_claim = _add_claim(pg_database, source.id, extraction_id, text)
    target_claim = _add_claim(pg_database, target.id, extraction_id, text, confidence=0.55)

    source_hash_before = claims.claims_hash(source.id)
    target_hash_before = claims.claims_hash(target.id)
    content_before = {
        claim_id: {column: row[column] for column in CLAIM_CONTENT_COLUMNS}
        for claim_id, row in _claim_rows(pg_database, target.id).items()
    }
    content_before.update(
        {
            claim_id: {column: row[column] for column in CLAIM_CONTENT_COLUMNS}
            for claim_id, row in _claim_rows(pg_database, source.id).items()
        }
    )

    entities.merge(source.id, target.id, score=0.8, thresholds_version="er-v1")

    content_after = {
        claim_id: {column: row[column] for column in CLAIM_CONTENT_COLUMNS}
        for claim_id, row in _claim_rows(pg_database, target.id).items()
    }
    assert content_after == content_before

    # The source's hash changed only because its claim membership became empty;
    # the target's only because it gained the source's rows.
    assert claims.claims_hash(source.id) == claims.claims_hash(blank.id)
    assert claims.claims_hash(source.id) != source_hash_before
    assert claims.claims_hash(target.id) != target_hash_before
    assert set(content_after) == {source_claim.id, target_claim.id}


def test_merge_marks_scores_stale_without_deleting_them(pg_database) -> None:
    suffix = _suffix()
    entities = EntitiesRepo(pg_database)
    source = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Score Source {suffix}", country="IN")
    )
    target = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Score Target {suffix}", country="IN")
    )
    scores = ScoresRepo(pg_database)
    source_score = _score(pg_database, source.id, "a" * 64)
    target_score = _score(pg_database, target.id, "b" * 64)
    assert scores.latest_for_entity(source.id).id == source_score

    _, audit = entities.merge(source.id, target.id, score=0.95, thresholds_version="er-v1")
    assert audit.features["scores_invalidated"] == {
        "source": [source_score],
        "target": [target_score],
    }

    for entity_id, score_id in ((source.id, source_score), (target.id, target_score)):
        rows = scores.list_for_entity(entity_id)
        assert [(row.id, row.stale) for row in rows] == [(score_id, True)]
        # The stale row survives; it is never deleted.
        assert scores.latest_for_entity(entity_id) is None
        assert scores.latest_for_entity(entity_id, include_stale=True).id == score_id

    fresh = _score(pg_database, target.id, "c" * 64)
    latest = scores.latest_for_entity(target.id)
    assert latest is not None and latest.id == fresh and latest.stale is False


def test_unmerge_returns_a_conflict_when_the_evidence_has_drifted(pg_database) -> None:
    suffix = _suffix()
    entities = EntitiesRepo(pg_database)
    source = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Drift Source {suffix}", country="IN")
    )
    target = entities.create(
        EntityCreate(kind="institution", canonical_name=f"Drift Target {suffix}", country="IN")
    )
    extraction_id, text = _extraction(pg_database, suffix)
    source_claim = _add_claim(pg_database, source.id, extraction_id, text)
    entities.merge(source.id, target.id, score=0.9, thresholds_version="er-v1")

    intruder = _add_claim(pg_database, target.id, extraction_id, text, confidence=0.4)

    result = entities.unmerge(source.id)
    assert result.conflict and not result.ok
    assert result.audit is None
    assert any(str(intruder.id) in item for item in result.drift)

    # Nothing moved, the merge link is intact, and no decision was recorded.
    assert EntitiesRepo(pg_database).get(source.id).merged_into_entity_id == target.id
    assert set(_claim_rows(pg_database, target.id)) == {source_claim.id, intruder.id}
    assert [row.band for row in entities.audit_for_pair(source.id, target.id)] == ["accept"]


def test_frontier_keeps_post_shape_and_conditional_validators_across_the_queue(
    pg_database,
) -> None:
    from fmd.discover.base import DocumentRef
    from fmd.frontier.queue import Frontier

    suffix = _suffix()
    ref = DocumentRef(
        url=f"https://nabh.example/admin-ajax.php?t={suffix}",
        etag='"abc123"',
        last_modified="Tue, 13 Oct 2026 10:00:00 GMT",
        metadata={
            "request_method": "POST",
            "request_body": {"action": "hospitals"},
            "request_headers": {"X-Requested-With": "XMLHttpRequest"},
        },
    )
    frontier = Frontier.from_database(pg_database)
    queued = frontier.enqueue_ref(ref)
    assert queued.http_method == "POST"
    assert queued.http_body == {"action": "hospitals"}

    stored = CrawlFrontierRepo(pg_database).get(ref.url)
    assert stored is not None
    assert stored.request_headers is not None
    assert stored.request_headers["If-None-Match"] == '"abc123"'
    assert stored.request_headers["If-Modified-Since"] == "Tue, 13 Oct 2026 10:00:00 GMT"
    assert stored.request_headers["X-Requested-With"] == "XMLHttpRequest"


# ----------------------------------------------------------------- sections (W8)


def test_sections_repo_round_trip_and_breadcrumb(pg_database) -> None:
    suffix = _suffix()
    url = f"https://example.ac.in/outline/{suffix}.pdf"
    document = DocumentsRepo(pg_database).upsert(
        DocumentUpsert(url=url, domain="example.ac.in", kind="accreditation")
    )
    from fmd.db import SectionNodeCreate, SectionsRepo, SectionTreeWrite

    repo = SectionsRepo(pg_database)
    tree = SectionTreeWrite(
        document_id=document.id,
        extractor="heuristic",
        extractor_version="pdftext:1.2.3",
        nodes=[
            SectionNodeCreate(
                node_key=1,
                level=1,
                title="4. Fees",
                char_start=100,
                char_end=400,
                heading_char_start=100,
                heading_char_end=108,
                page_no=2,
                bbox=[10.0, 20.0, 300.0, 40.0],
            ),
            SectionNodeCreate(
                node_key=2,
                level=2,
                title="4.2 Hostel",
                char_start=140,
                char_end=400,
                heading_char_start=140,
                heading_char_end=150,
                page_no=2,
                parent_key=1,
            ),
        ],
    )
    assert repo.replace_for_document(tree) == 2

    rows = repo.list_for_document(document.id)
    assert [(row.node_key, row.title, row.level) for row in rows] == [
        (1, "4. Fees", 1),
        (2, "4.2 Hostel", 2),
    ]
    assert rows[1].parent_id == rows[0].id
    assert rows[0].bbox == [10.0, 20.0, 300.0, 40.0]

    # The breadcrumb lookup resolves the deepest section containing the span.
    found = repo.breadcrumb(document.id, 150, 160)
    assert found is not None
    assert found["node"].title == "4.2 Hostel"
    assert found["breadcrumb"] == ["4. Fees", "4.2 Hostel"]
    # A span before the first heading has no section.
    assert repo.breadcrumb(document.id, 0, 10) is None

    # A whole-tree write replaces (derived structure, never appended).
    smaller = SectionTreeWrite(
        document_id=document.id,
        extractor="docling",
        nodes=tree.nodes[:1],
    )
    repo.replace_for_document(smaller)
    assert [row.title for row in repo.list_for_document(document.id)] == ["4. Fees"]
    repo.delete_for_document(document.id)
    assert repo.list_for_document(document.id) == []


def test_migration_0009_is_registered_and_idempotent(pg_database) -> None:
    applied = applied_migrations(pg_database)
    assert "0009_document_sections" in applied
    # A re-apply is a no-op with the same checksum (never an error).
    versions = run_migrations(pg_database)
    assert versions == []
