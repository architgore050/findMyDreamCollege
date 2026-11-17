"""Ingest stage: inventory documents -> entities -> institution-site crawl seeds.

This is the link the first live crawl was missing. Inventory payloads (UGC JSON)
used to be stored, sent to the planner, and abstained on as ``family_undetected``
forever. Here they are read once, turned into ``entities`` (institution, state,
type, website) and a bounded, prioritised ``institution_domains`` work queue.

The stage is idempotent: entities match on the ER-normalised name, aliases are
inserted only when absent, and a domain that is already queued or planned is not
reset by a re-ingest. Nothing here fetches: the payload bytes come from the
snapshot store, so a re-run over the same evidence produces the same rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fmd.db.repositories import (
    AliasCreate,
    EntityCreate,
    InstitutionDomainSeed,
)
from fmd.er.normalise import blocking_key
from fmd.ingest.inventory import (
    InventoryFieldMap,
    InventoryPayloadError,
    domain_from_website,
    records_from_payload,
    seed_url,
)
from fmd.ingest.targeting import TargetWeights, priority_for, select_targets
from fmd.orchestrate.ledger import RunSummary

logger = logging.getLogger(__name__)

DEFAULT_INVENTORY_LIMIT = 50
#: Registry source whose adapter consumes the discovered domains.
INSTITUTION_ADAPTER = "institution_site"


@dataclass(frozen=True)
class _Target:
    """One domain plus the inventory facts targeting is allowed to see."""

    domain: str
    seed_url: str
    entity_id: int | None
    institution_type: str | None
    priority: int


class InventoryStage:
    def __init__(
        self,
        *,
        documents: Any,
        snapshots: Any,
        store: Any,
        entities: Any,
        aliases: Any,
        domains: Any,
        sources: Any | None = None,
        registry: Any | None = None,
        naac_domains: Any | None = None,
    ) -> None:
        self.documents = documents
        self.snapshots = snapshots
        self.store = store
        self.entities = entities
        self.aliases = aliases
        self.domains = domains
        self.sources = {str(getattr(source, "id", "")): source for source in (sources or ())}
        self.registry = registry
        self.naac_domains = naac_domains

    # ------------------------------------------------------------------ helpers
    def _source(self, source_id: str | None) -> Any | None:
        if not source_id:
            return None
        if source_id in self.sources:
            return self.sources[source_id]
        if self.registry is not None:
            try:
                return self.registry.sources.by_id(source_id)
            except Exception:  # pragma: no cover - registry shape is stable
                return None
        return None

    def _institution_source(self) -> Any | None:
        injected = next(
            (
                source
                for source in self.sources.values()
                if getattr(source, "adapter", None) == INSTITUTION_ADAPTER
            ),
            None,
        )
        if injected is not None:
            return injected
        if self.registry is None:
            return None
        return next(
            (
                source
                for source in self.registry.sources.sources
                if getattr(source, "adapter", None) == INSTITUTION_ADAPTER
            ),
            None,
        )

    def _text(self, document: Any) -> str | None:
        snapshot = self.snapshots.latest_for_document(document.id)
        if snapshot is None:
            return None
        try:
            payload = self.store.get(snapshot.sha256)
        except Exception as exc:  # missing/corrupt object: surface, never guess
            logger.warning("inventory snapshot %s unreadable: %s", snapshot.id, exc)
            return None
        if isinstance(payload, bytes):
            return payload.decode("utf-8", "replace")
        return str(payload)

    # --------------------------------------------------------------------- run
    def _structured_source(self, source_id: str | None) -> Any | None:
        """The registry source that owns a structured inventory payload, if any."""
        source = self._source(source_id)
        if source is None or getattr(source, "kind", None) != "inventory":
            return None
        try:
            InventoryFieldMap.from_source(source)
        except InventoryPayloadError:
            return None
        return source

    def _inventory_documents(self, limit: int) -> list[Any]:
        """Role=inventory documents, plus stale-roled structured payloads.

        Migrations 0003/0004 introduced ``documents.role`` after the first UGC
        payload was fetched, so an old row can still be marked ``evidence``.
        Re-planning it is out of this stage's control (the crawler skips URLs
        already in the frontier), but the bytes are authoritative: a source that
        declares ``params.inventory`` is an inventory payload whatever the row
        says. The union stays deterministic (oldest first) and capped.
        """
        selected = list(self.documents.list_by_role("inventory", limit=limit))
        seen = {getattr(document, "id", None) for document in selected}
        lister = getattr(self.documents, "list_by_domain", None)
        if lister is None:
            return selected[:limit]
        for source in self._all_sources():
            if getattr(source, "kind", None) != "inventory":
                continue
            try:
                InventoryFieldMap.from_source(source)
            except InventoryPayloadError:
                continue
            base_url = getattr(source, "base_url", None)
            if not base_url:
                continue
            from urllib.parse import urlsplit

            # documents.domain is the registrable domain, so the ``www.`` host of
            # a registry base_url would select nothing.
            domain = (urlsplit(str(base_url)).hostname or "").lower().removeprefix("www.")
            if not domain:
                continue
            source_id = str(getattr(source, "id", ""))
            for document in lister(domain, limit):
                if getattr(document, "source_id", None) != source_id:
                    continue
                if getattr(document, "id", None) in seen:
                    continue
                if getattr(document, "role", None) == "inventory":
                    continue
                seen.add(getattr(document, "id", None))
                selected.append(document)
        selected.sort(key=lambda document: getattr(document, "id", 0) or 0)
        return selected[:limit]

    def _all_sources(self) -> list[Any]:
        if self.sources:
            return list(self.sources.values())
        if self.registry is None:
            return []
        return list(self.registry.sources.sources)

    def run(self, *, summary: RunSummary, limit: int | None = None) -> list[str]:
        """Ingest inventory documents; return the seed URLs selected to crawl."""
        if self.domains is None or self.entities is None:
            summary.bump("skipped")
            summary.metadata["ingest"] = {"reason": "no_inventory_repository"}
            return []

        documents = self._inventory_documents(limit or DEFAULT_INVENTORY_LIMIT)
        summary.bump("inventory_documents", len(documents))
        for document in documents:
            self._ingest_document(document, summary=summary)

        selected = self._select_targets(summary=summary)
        self.domains.mark(
            [target.domain for target in selected], "queued", reason="selected_for_crawl"
        )
        summary.bump("domains_queued", len(selected))
        summary.metadata["domains"] = [target.seed_url for target in selected]
        summary.metadata["institution_domains"] = _safe_counts(self.domains)
        return [target.seed_url for target in selected]

    def _ingest_document(self, document: Any, *, summary: RunSummary) -> None:
        source = self._source(getattr(document, "source_id", None))
        try:
            field_map = InventoryFieldMap.from_source(source)
        except InventoryPayloadError as exc:
            summary.bump("skipped")
            summary.outcomes.append(
                {
                    "document_id": getattr(document, "id", None),
                    "url": getattr(document, "url", None),
                    "outcome": "skipped",
                    "reason": str(exc),
                }
            )
            return

        text = self._text(document)
        if text is None:
            summary.bump("skipped")
            summary.outcomes.append(
                {
                    "document_id": getattr(document, "id", None),
                    "url": getattr(document, "url", None),
                    "outcome": "skipped",
                    "reason": "no_snapshot",
                }
            )
            return

        try:
            records, dropped = records_from_payload(text, field_map=field_map)
        except InventoryPayloadError as exc:
            summary.bump("rejected")
            summary.outcomes.append(
                {
                    "document_id": getattr(document, "id", None),
                    "url": getattr(document, "url", None),
                    "outcome": "rejected",
                    "reason": str(exc),
                }
            )
            return

        summary.bump("records", len(records))
        summary.bump("records_dropped", len(dropped))
        source_id = getattr(document, "source_id", None)
        for record in records:
            entity, created = self._upsert_entity(record, source_id=source_id)
            if entity is None:
                summary.bump("records_dropped")
                continue
            summary.bump("entities_created" if created else "entities_matched")
            summary.bump("aliases", self._put_aliases(entity.id, record, source_id=source_id))
            if self._put_domain(entity, record, source_id=source_id):
                summary.bump("domains_discovered")

    def _upsert_entity(self, record: Any, *, source_id: str | None) -> tuple[Any | None, bool]:
        normalised = record.normalised_name
        if not normalised:
            return None, False
        existing = None
        finder = getattr(self.entities, "find_normalised", None)
        if finder is not None:
            existing = finder("institution", normalised)
        if existing is None:
            existing = self.entities.find("institution", record.name)
        if existing is not None:
            update = getattr(self.entities, "update_details", None)
            if update is not None:
                existing = (
                    update(
                        existing.id,
                        state=record.state,
                        institution_type=record.institution_type,
                        website_url=record.website,
                    )
                    or existing
                )
            return existing, False

        keys = blocking_key(record.name, record.state)
        created = self.entities.create(
            EntityCreate(
                kind="institution",
                canonical_name=record.name,
                country=record.country,
                state=record.state,
                status="active",
                normalised_name=normalised,
                blocking_key=keys[0] if keys else None,
                website_url=record.website,
                institution_type=record.institution_type,
            )
        )
        return created, True

    def _put_aliases(self, entity_id: int, record: Any, *, source_id: str | None) -> int:
        writer = getattr(self.aliases, "create_if_missing", None)
        if writer is None:
            return 0
        written = 0
        for acronym in record.acronyms:
            row = writer(
                AliasCreate(
                    entity_id=entity_id,
                    alias=acronym,
                    alias_type="acronym",
                    source_id=source_id,
                    confidence=0.8,
                )
            )
            if row is not None:
                written += 1
        return written

    def _put_domain(self, entity: Any, record: Any, *, source_id: str | None) -> bool:
        domain = domain_from_website(record.website)
        url = seed_url(record.website)
        if domain is None or url is None:
            return False
        score = priority_for(
            institution_type=record.institution_type,
            has_website=True,
            weights=self._weights(),
        )
        self.domains.upsert(
            InstitutionDomainSeed(
                domain=domain,
                seed_url=url,
                entity_id=getattr(entity, "id", None),
                discovered_by=source_id,
                priority=score.priority,
                reason=",".join(score.reasons) or None,
            )
        )
        return True

    def _weights(self) -> TargetWeights:
        source = self._institution_source()
        if source is None:
            return TargetWeights()
        try:
            return TargetWeights.from_source(source)
        except Exception:  # pragma: no cover - defensive registry read
            return TargetWeights()

    def _select_targets(self, *, summary: RunSummary) -> list[_Target]:
        weights = self._weights()
        candidates = self.domains.list_candidates(limit=1000)
        naac = list(self.naac_domains() if callable(self.naac_domains) else ())
        targets: list[_Target] = []
        for row in candidates:
            entity = None
            if getattr(row, "entity_id", None) is not None:
                getter = getattr(self.entities, "get", None)
                entity = getter(row.entity_id) if getter is not None else None
            targets.append(
                _Target(
                    domain=row.domain,
                    seed_url=row.seed_url,
                    entity_id=row.entity_id,
                    institution_type=getattr(entity, "institution_type", None),
                    priority=int(getattr(row, "priority", 0) or 0),
                )
            )
        return select_targets(targets, weights=weights, naac_domains=naac)


def _safe_counts(repo: Any) -> dict[str, int]:
    counter = getattr(repo, "counts_by_state", None)
    if counter is None:
        return {}
    try:
        return dict(counter())
    except Exception:  # pragma: no cover - observability only
        return {}


__all__ = ["DEFAULT_INVENTORY_LIMIT", "INSTITUTION_ADAPTER", "InventoryStage"]
