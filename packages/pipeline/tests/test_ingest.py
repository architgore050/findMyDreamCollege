"""Inventory parsing, targeting and idempotent ingest (offline, no DB, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_orchestrate_support import MemoryDocuments, MemoryEntities, MemorySnapshots

from fmd.db.repositories import DocumentUpsert, SnapshotCreate
from fmd.ingest import (
    InventoryFieldMap,
    TargetWeights,
    domain_from_website,
    priority_for,
    records_from_payload,
    select_targets,
)
from fmd.ingest.inventory import InventoryPayloadError
from fmd.orchestrate.ingest import InventoryStage
from fmd.orchestrate.ledger import RunSummary, utcnow

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"
UGC_PAYLOAD = (FIXTURES / "ugc_university_details.json").read_text(encoding="utf-8")

FIELD_MAP = InventoryFieldMap.from_params(
    {
        "list_key": "List",
        "country": "IN",
        "fields": {
            "name": "uni_name",
            "state": "state",
            "type": "uni_type",
            "website": "url",
            "status": "status",
        },
    }
)


class FakeSource:
    def __init__(self, source_id: str = "fixture_inventory", adapter: str = "ugc") -> None:
        self.id = source_id
        self.adapter = adapter
        self.enabled = True
        self.params: dict[str, Any] = (
            {
                "inventory": {
                    "list_key": "List",
                    "country": "IN",
                    "fields": {
                        "name": "uni_name",
                        "state": "state",
                        "type": "uni_type",
                        "website": "url",
                        "status": "status",
                    },
                }
            }
            if adapter == "ugc"
            else {}
        )


class TargetSource(FakeSource):
    def __init__(self) -> None:
        super().__init__(source_id="fixture_institution_sites", adapter="institution_site")
        self.params = {
            "targeting": {
                "max_domains_per_run": 2,
                "known_website_bonus": 15,
                "type_weights": {"central": 40, "state": 20, "deemed": 30},
            }
        }


class FakeDomain:
    def __init__(self, **fields: Any) -> None:
        self.domain = fields["domain"]
        self.seed_url = fields["seed_url"]
        self.entity_id = fields.get("entity_id")
        self.priority = fields.get("priority", 0)
        self.state = fields.get("state", "pending")
        self.institution_type = None


class MemoryDomains:
    def __init__(self) -> None:
        self.rows: dict[str, FakeDomain] = {}

    def upsert(self, seed) -> FakeDomain:
        existing = self.rows.get(seed.domain)
        if existing is None:
            existing = FakeDomain(
                domain=seed.domain,
                seed_url=seed.seed_url,
                entity_id=seed.entity_id,
                priority=seed.priority,
            )
            self.rows[seed.domain] = existing
        return existing

    def list_candidates(self, limit: int = 500) -> list[FakeDomain]:
        rows = [row for row in self.rows.values() if row.state in {"pending", "queued"}]
        return sorted(rows, key=lambda row: (-row.priority, row.domain))[:limit]

    def list_by_state(self, state: str, limit: int = 500) -> list[FakeDomain]:
        return [row for row in self.rows.values() if row.state == state][:limit]

    def mark(self, domains, state: str, *, reason: str | None = None) -> int:
        marked = 0
        for domain in domains:
            row = self.rows.get(domain)
            if row is None:
                continue
            row.state = state
            marked += 1
        return marked

    def mark_seed_urls(self, urls, state: str) -> int:
        wanted = set(urls)
        marked = 0
        for row in self.rows.values():
            if row.seed_url in wanted:
                row.state = state
                marked += 1
        return marked

    def counts_by_state(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows.values():
            counts[row.state] = counts.get(row.state, 0) + 1
        return counts


class MemoryAliases:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def create_if_missing(self, create):
        key = (create.entity_id, create.alias, create.alias_type)
        if any((row.entity_id, row.alias, row.alias_type) == key for row in self.rows):
            return None
        self.rows.append(create)
        return create


class FakeStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get(self, sha256: str) -> bytes:
        return self.objects[sha256]


class FakeRegistry:
    def __init__(self, sources: list[Any]) -> None:
        self.sources = type("Sources", (), {"sources": sources})()

    def by_id(self, source_id: str):
        return next((s for s in self.sources.sources if s.id == source_id), None)


def _seed_document(documents, snapshots, store_objects, *, url: str, source_id: str) -> None:
    document = documents.upsert(
        DocumentUpsert(url=url, domain="ugc.example.gov.in", source_id=source_id, role="inventory")
    )
    payload = UGC_PAYLOAD.encode("utf-8")
    digest = __import__("hashlib").sha256(payload).hexdigest()
    store_objects[digest] = payload
    snapshots.create(SnapshotCreate(document_id=document.id, sha256=digest, bytes_path="objects/x"))


def _stage(objects, documents, snapshots, entities, aliases, domains, source) -> InventoryStage:
    return InventoryStage(
        documents=documents,
        snapshots=snapshots,
        store=FakeStore(objects),
        entities=entities,
        aliases=aliases,
        domains=domains,
        sources=[source],
        registry=FakeRegistry([source, TargetSource()]),
    )


# ------------------------------------------------------------------- parsing


def test_payload_yields_records_and_domains() -> None:
    records, dropped = records_from_payload(UGC_PAYLOAD, field_map=FIELD_MAP)

    assert [record.name for record in records] == [
        "Alpha Institute of Medical Sciences",
        "Beta Health University",
        "Gamma Paramedical University",
    ]
    # The nameless row is dropped with a reason, never invented.
    assert [entry["reason"] for entry in dropped] == ["missing_name"]
    assert records[0].state == "Delhi"  # whitespace collapsed
    assert records[0].institution_type == "Central"
    assert records[0].country == "IN"
    assert records[0].domain == "www.alpha-med.example.ac.in"
    assert records[1].domain == "beta-health.example.ac.in"  # bare host + path
    assert records[2].domain is None  # mailto is not a website
    assert records[1].acronyms  # derived acronym aliases exist


def test_html_error_shell_is_not_an_empty_inventory() -> None:
    with pytest.raises(InventoryPayloadError):
        records_from_payload("<html><body>Not found</body></html>", field_map=FIELD_MAP)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://www.example.ac.in", "www.example.ac.in"),
        ("example.ac.in/some/page", "example.ac.in"),
        ("mailto:registrar@example.ac.in", None),
        ("javascript:void(0)", None),
        ("not-a-host", None),
    ],
)
def test_domain_from_website(value, expected) -> None:
    assert domain_from_website(value) == expected


# ----------------------------------------------------------------- targeting


class _Target:
    def __init__(self, domain, priority=0, institution_type=None, seed_url=None):
        self.domain = domain
        self.seed_url = seed_url or f"https://{domain}/"
        self.priority = priority
        self.institution_type = institution_type


def test_targeting_prefers_type_and_known_website_then_caps() -> None:
    weights = TargetWeights.from_params(
        {
            "targeting": {
                "max_domains_per_run": 2,
                "known_website_bonus": 15,
                "type_weights": {"central": 40, "state": 20},
            }
        }
    )
    rows = [
        _Target("state.example.ac.in", institution_type="State"),
        _Target("central.example.ac.in", institution_type="Central"),
        _Target("private.example.ac.in", institution_type="Private"),
    ]

    chosen = select_targets(rows, weights=weights)

    assert [row.domain for row in chosen] == ["central.example.ac.in", "state.example.ac.in"]
    central = priority_for(institution_type="Central", has_website=True, weights=weights)
    assert central.priority == 55


# ------------------------------------------------------------ idempotent ingest


def test_ingest_is_idempotent_and_emits_bounded_domains() -> None:
    documents = MemoryDocuments()
    entities = MemoryEntities()
    snapshots = MemorySnapshots(documents=documents)
    aliases = MemoryAliases()
    domains = MemoryDomains()
    objects: dict[str, bytes] = {}
    source = FakeSource()
    _seed_document(
        documents,
        snapshots,
        objects,
        url="https://www.ugc.gov.in/universitydetails/Getuniversity_details?unitypeID=1",
        source_id=source.id,
    )
    stage = _stage(objects, documents, snapshots, entities, aliases, domains, source)

    first = RunSummary(run_id="run-1", stage="ingest", started_at=utcnow())
    selected = stage.run(summary=first)

    assert first.counts["inventory_documents"] == 1
    assert first.counts["records"] == 3
    assert first.counts["entities_created"] == 3
    assert first.counts["domains_discovered"] == 2  # mailto row yields no domain
    assert len(entities.rows) == 3
    assert len(aliases.rows) >= 1
    # max_domains_per_run=2 from the institution_sites targeting block.
    assert len(selected) == 2
    assert all(row.state == "queued" for row in domains.rows.values())
    assert first.counts["domains_queued"] == 2

    alias_count = len(aliases.rows)
    second = RunSummary(run_id="run-2", stage="ingest", started_at=utcnow())
    stage.run(summary=second)

    assert second.counts.get("entities_created", 0) == 0
    assert second.counts["entities_matched"] == 3
    assert len(entities.rows) == 3
    assert len(aliases.rows) == alias_count  # no duplicate aliases written
    assert len(domains.rows) == 2


def test_ingest_skips_a_source_without_a_field_map() -> None:
    documents = MemoryDocuments()
    entities = MemoryEntities()
    snapshots = MemorySnapshots(documents=documents)
    domains = MemoryDomains()
    objects: dict[str, bytes] = {}
    source = type("NoMap", (), {"id": "no_map", "adapter": "ugc", "params": {}})()
    _seed_document(
        documents, snapshots, objects, url="https://x.gov.in/list.json", source_id="no_map"
    )
    stage = _stage(objects, documents, snapshots, entities, MemoryAliases(), domains, source)

    summary = RunSummary(run_id="run-3", stage="ingest", started_at=utcnow())
    stage.run(summary=summary)

    assert summary.counts["skipped"] == 1
    assert entities.rows == []
    assert domains.rows == {}


def test_pii_columns_never_enter_an_inventory_record() -> None:
    # PII columns exist in the payload but ingest reads only mapped fields.
    raw = json.loads(UGC_PAYLOAD)["List"][0]
    records, _ = records_from_payload(UGC_PAYLOAD, field_map=FIELD_MAP)
    serialised = json.dumps(records[0].__dict__, default=str)
    assert "must-never-enter-ingest" not in serialised
    assert raw["user_password"] == "must-never-enter-ingest"
