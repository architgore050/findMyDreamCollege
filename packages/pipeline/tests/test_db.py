from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

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
    run_migrations,
)

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


def test_migrations_are_idempotent(pg_database) -> None:
    assert run_migrations(pg_database) == []
    applied = applied_migrations(pg_database)
    assert "0001_init" in applied
    assert len(applied) == 1


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
    SourceExclusionsRepo(pg_database).upsert(
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


def test_seed_projects_registry(pg_database, registry) -> None:
    from fmd.seed import seed_database

    counts = seed_database(pg_database, registry)
    assert counts["countries"] == len(registry.countries.countries)
    assert counts["sources"] == len(registry.sources.sources)
    assert counts["exams"] == len(registry.exams.exams)
    assert CountriesRepo(pg_database).get("IN") is not None
    assert SourcesRepo(pg_database).get(registry.sources.sources[0].id) is not None
    assert ExamsRepo(pg_database).get(registry.exams.exams[0].id) is not None
