"""Project the validated config registry into Postgres (``fmd seed``).

Idempotent: every write is an upsert keyed by the config identifier. The config
files remain the source of truth; this table set exists so the API and crawler
can join against the registry without re-reading YAML mid-request.
"""

from __future__ import annotations

from typing import Any

from fmd.config import Registry
from fmd.db.connection import Database
from fmd.db.repositories import (
    CountriesRepo,
    CountrySeed,
    ExamSeed,
    ExamsRepo,
    SourceExclusionSeed,
    SourceExclusionsRepo,
    SourceSeed,
    SourcesRepo,
)


class SeedError(RuntimeError):
    pass


def _capabilities(country: Any) -> dict[str, Any] | None:
    if country.capabilities is None:
        return None
    return country.capabilities.model_dump(mode="json", exclude_none=True)


def _source_extras(source: Any) -> dict[str, Any]:
    known = set(SourceSeed.model_fields)
    return {
        key: value
        for key, value in source.model_dump(mode="json").items()
        if key not in known
    }


def seed_database(database: Database, registry: Registry) -> dict[str, int]:
    countries_repo = CountriesRepo(database)
    sources_repo = SourcesRepo(database)
    exclusions_repo = SourceExclusionsRepo(database)
    exams_repo = ExamsRepo(database)

    counts = {"countries": 0, "sources": 0, "source_exclusions": 0, "exams": 0}

    for code, country in registry.countries.countries.items():
        countries_repo.upsert(
            CountrySeed(
                code=code,
                name=country.name,
                enabled=country.enabled,
                currency=country.currency,
                tier=country.tier,
                languages=country.languages,
                capabilities=_capabilities(country),
            )
        )
        counts["countries"] += 1

    for source in registry.sources.sources:
        if not source.adapter:
            # Registry validation already rejects this; keep the guard local.
            raise SeedError(f"source {source.id!r} has no adapter")
        sources_repo.upsert(
            SourceSeed(
                id=source.id,
                name=source.name,
                base_url=source.base_url,
                kind=source.kind,
                format=source.format,
                cadence_days=source.cadence_days,
                trust=source.trust,
                adapter=source.adapter,
                enabled=source.enabled,
                notes=source.notes,
                requires_manual_review=source.requires_manual_review,
                seeds=source.seeds,
                config=_source_extras(source),
            )
        )
        counts["sources"] += 1

    for exclusion in registry.sources.exclusions:
        exclusions_repo.upsert(
            SourceExclusionSeed(
                id=exclusion.id, name=exclusion.name, reason=exclusion.reason
            )
        )
        counts["source_exclusions"] += 1

    for exam in registry.exams.exams:
        exams_repo.upsert(
            ExamSeed(
                id=exam.id,
                name=exam.name,
                country=exam.country,
                scope=exam.scope,
                provider=exam.provider,
                registration_opens=None,
                registration_closes=None,
                earliest_sitting=None,
                latest_sitting=None,
                requires_travel=exam.requires_travel,
                remote_options=exam.remote_options,
                fee_amount=exam.fee_amount,
                currency=exam.currency,
            )
        )
        counts["exams"] += 1

    return counts
