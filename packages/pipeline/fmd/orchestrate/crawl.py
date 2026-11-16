"""Crawl orchestration: adapters -> frontier -> fetch ladder -> snapshots.

This module owns sequencing only. It does not decide politeness (the frontier and
fetch ladder do), does not parse documents (adapters do), and does not claim to
have succeeded when a fetch failed: every non-2xx outcome becomes a recorded
rejection, abstention or failure with a machine-readable reason.

Planning is a column, not a header: ``documents.source_id``,
``documents.extraction_family`` and ``documents.entity_id`` (migration 0003) are
written here, so ``fmd extract`` and SQL can both see why a document is or is not
extractable. Discovery may *reference* a family/entity in
``DocumentRef.metadata``; when it does not, :class:`fmd.plan.DocumentPlanner`
detects the family from the registry + heuristics and links the entity from the
fetched text.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from fmd.db.repositories import (
    DocumentRecord,
    DocumentsRepo,
    DocumentUpsert,
    FrontierUpsert,
    SnapshotCreate,
    SnapshotRecord,
    SnapshotsRepo,
)
from fmd.discover.base import dedupe_refs
from fmd.fetch.dedupe import near_duplicate, simhash64
from fmd.fetch.politeness import registrable_domain
from fmd.fetch.types import FetchRequest, FetchResult, TierUnavailable
from fmd.frontier.queue import Frontier
from fmd.orchestrate.adapters import call_fetch
from fmd.orchestrate.ledger import RunSummary
from fmd.plan import persist_plan
from fmd.plan.planner import declared_family
from fmd.plan.roles import RoleConfigError, role_for
from fmd.plan.rules import DEFAULT_RULES, detect_family

logger = logging.getLogger(__name__)

# A source that cannot answer this many times in one run is stopped, not retried
# forever. A block/quarantine is handled per URL by the frontier as well.
DEFAULT_SOURCE_FAILURE_BUDGET = 5
SIMHASH_MAX_DISTANCE = 3


def utc() -> datetime:
    return datetime.now(UTC)


def domain_for(url: str) -> str:
    """Registrable domain, or the hostname when the crawl extra is absent.

    ``tldextract`` is the correct eTLD+1 implementation; the fallback keeps the
    queue usable (and every fetch honestly abstained) in a base install instead
    of aborting the whole run on an import error.
    """
    try:
        return registrable_domain(url)
    except TierUnavailable:
        from urllib.parse import urlsplit

        return (urlsplit(url).hostname or "").lower()


def frontier_upsert(ref: Any, *, source_id: str | None) -> FrontierUpsert:
    """Enqueue a ref with its registry source persisted as a column."""
    return FrontierUpsert.from_document_ref(
        ref,
        domain=domain_for(ref.url),
        state="queued",
        next_allowed_at=utc(),
        source_id=source_id,
    )


def _hostname(url: str) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def _match_host(host: str, by_host: dict[str, int]) -> int | None:
    """Exact host, or a subdomain of a seed host (``files.alpha.ac.in``)."""
    if not host:
        return None
    if host in by_host:
        return by_host[host]
    for seed_host, entity_id in by_host.items():
        if host.endswith("." + seed_host):
            return entity_id
    return None


def _header(result: FetchResult, name: str) -> str | None:
    value = result.headers.get(name)
    return str(value) if value else None


def _http_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _is_textual(content_type: str | None) -> bool:
    if not content_type:
        return False
    lowered = content_type.lower()
    return any(token in lowered for token in ("text/", "json", "xml", "html"))


class Crawler:
    def __init__(
        self,
        *,
        frontier: Frontier,
        fetcher: Any,
        store: Any,
        documents: DocumentsRepo | Any,
        snapshots: SnapshotsRepo | Any,
        adapter_factory: Any,
        denylisted_domains: Iterable[str] | None = None,
        max_source_failures: int = DEFAULT_SOURCE_FAILURE_BUDGET,
        planner: Any | None = None,
        sources: Iterable[Any] | None = None,
    ) -> None:
        self.frontier = frontier
        self.fetcher = fetcher
        self.store = store
        self.documents = documents
        self.snapshots = snapshots
        self.adapter_factory = adapter_factory
        self.denylisted_domains = tuple(denylisted_domains or ())
        self.planner = planner
        self.sources_by_id = {str(getattr(source, "id", "")): source for source in sources or ()}
        self.max_source_failures = max_source_failures
        self._seen_sha: set[str] = set()
        self._fingerprints: list[int] = []
        self._source_failures: dict[str, int] = {}
        self._stopped_sources: set[str] = set()

    # ------------------------------------------------------------------- planning
    def plan(
        self,
        sources: Iterable[Any],
        *,
        summary: RunSummary,
        limit: int | None = None,
        dry_run: bool = False,
        domains: dict[str, list[str]] | None = None,
        entity_hints: dict[str, int] | None = None,
    ) -> None:
        remaining = limit
        for source in sources:
            if remaining is not None and remaining <= 0:
                break
            source_id = str(getattr(source, "id", ""))
            outcome: dict[str, Any] = {
                "source_id": source_id,
                "adapter": getattr(source, "adapter", None),
                "status": "ok",
                "planned": 0,
                "enqueued": 0,
            }
            try:
                adapter = self.adapter_factory(source)
                seeds = list(getattr(source, "seeds", None) or ())
                discovered = list((domains or {}).get(source_id) or ())
                # Discovered domains are appended after registry seeds; both are
                # bounded by the adapter's own page budget.
                refs = call_fetch(adapter, domains=[*seeds, *discovered])
                if discovered:
                    outcome["domains"] = discovered
            except Exception as exc:  # one bad source never aborts the run
                reason = getattr(exc, "reason", None) or type(exc).__name__
                detail = str(exc)
                outcome.update({"status": "error", "reason": reason, "detail": detail})
                summary.sources.append(outcome)
                summary.fail("crawl.plan", str(reason), source_id=outcome["source_id"])
                logger.warning("source %s failed to plan: %s", outcome["source_id"], detail)
                continue

            unique = dedupe_refs(list(refs))
            unique = self._with_family_hints(unique)
            unique = self._with_entity_hints(unique, entity_hints or {})
            if remaining is not None:
                unique = unique[:remaining]
            outcome["planned"] = len(unique)
            summary.bump("planned", len(unique))
            for ref in unique:
                if dry_run:
                    summary.planned_urls.append(ref.url)
                    continue
                self.frontier.repo.enqueue(frontier_upsert(ref, source_id=source_id))
                outcome["enqueued"] += 1
                summary.bump("enqueued")
            summary.sources.append(outcome)
            if remaining is not None:
                remaining -= len(unique)

    def _with_family_hints(self, refs: list[Any]) -> list[Any]:
        """Carry the discovery anchor's family signal through the frontier.

        A PDF named ``2024-25.pdf`` is evidence when the link said "Annual
        Report" and impossible to classify from the URL alone. The frontier has
        no anchor column, but it already persists ``DocumentRef.metadata["family"]``
        as ``extraction_family`` (migration 0003), so the hint is attached here —
        detection, not invention: the same rule set the planner uses decides it.
        """
        for ref in refs:
            metadata = getattr(ref, "metadata", None)
            if not isinstance(metadata, dict) or metadata.get("family"):
                continue
            anchor = metadata.get("anchor_text")
            if not anchor:
                continue
            family, _score, _signals, _reason = detect_family(
                url=str(getattr(ref, "url", "")),
                anchor_text=str(anchor),
                rules=DEFAULT_RULES,
            )
            if family:
                metadata["family"] = family
        return refs

    def _with_entity_hints(self, refs: list[Any], hints: dict[str, int]) -> list[Any]:
        """Link a discovered document to the institution that owns the seed domain.

        Inventory ingest establishes ``domain -> entity`` from a published
        website. A placement PDF on that domain belongs to that institution; a
        PDF tier cannot tell the planner this, because its text is only readable
        after extraction. The link is *reference* provenance (registry/inventory),
        never an ER guess, and it is only applied when the registrable domain
        matches exactly.
        """
        if not hints:
            return refs
        by_host: dict[str, int] = {}
        for seed, entity_id in hints.items():
            host = _hostname(seed)
            if host:
                by_host[host] = entity_id
        for ref in refs:
            metadata = getattr(ref, "metadata", None)
            if not isinstance(metadata, dict) or metadata.get("entity_id") is not None:
                continue
            entity_id = _match_host(_hostname(str(getattr(ref, "url", ""))), by_host)
            if entity_id is not None:
                metadata["entity_id"] = entity_id
        return refs

    # --------------------------------------------------------------------- drain
    def drain(self, *, summary: RunSummary, limit: int | None = None) -> None:
        processed = 0
        while limit is None or processed < limit:
            due = self.frontier.repo.list_due(utc(), 25)
            if not due:
                break
            progressed = False
            for record in due:
                if limit is not None and processed >= limit:
                    break
                source_id = str(getattr(record, "source_id", None) or "")
                if source_id in self._stopped_sources:
                    continue
                if self.frontier._quarantined(record.domain):
                    summary.bump("quarantined")
                    continue
                if self.frontier.budget_remaining(record.domain) <= 0:
                    summary.bump("budget_exhausted")
                    continue
                if self.frontier.should_stop(record.domain):
                    summary.bump("yield_stopped")
                    continue
                progressed = True
                self._process(record, summary=summary, source_id=source_id)
                processed += 1
            if not progressed:
                break

    def _process(self, record: Any, *, summary: RunSummary, source_id: str) -> None:
        self.frontier.repo.set_state(record.url, "in_progress")
        try:
            result = self._fetch(record)
        except Exception as exc:  # transport/trap: an abstention with a reason
            reason = getattr(exc, "reason", None) or type(exc).__name__
            summary.bump("abstained")
            summary.outcomes.append(
                {"url": record.url, "outcome": "abstained", "reason": str(reason)}
            )
            self.frontier.fail(record.url)
            return

        if result.not_modified:
            self._handle_not_modified(record, result, summary=summary)
            return
        if result.ok:
            self._handle_ok(record, result, summary=summary, source_id=source_id)
            return
        if result.terminal:
            role = self._role(
                record,
                content_type=_header(result, "content-type"),
                source_id=source_id,
                summary=summary,
            )
            self._upsert_document(
                record,
                result,
                status=result.status,
                summary=summary,
                source_id=source_id,
                role=role,
            )
            self.frontier.complete(record.url, valuable_pages=0, pages=1, tier=result.fetcher_tier)
            summary.bump("rejected")
            summary.outcomes.append(
                {"url": record.url, "outcome": "rejected", "reason": f"http_{result.status}"}
            )
            return

        reason = result.error or f"http_{result.status}"
        if result.status == 0 and reason == "robots_disallowed":
            summary.bump("abstained")
            summary.outcomes.append({"url": record.url, "outcome": "abstained", "reason": reason})
            self.frontier.fail(record.url)
            return
        self.frontier.fail(record.url, blocked=result.blocked or result.quarantined)
        summary.bump("failed")
        summary.outcomes.append({"url": record.url, "outcome": "failed", "reason": reason})
        summary.fail("crawl.fetch", reason, url=record.url, source_id=source_id or None)
        self._note_source_failure(source_id)

    def _fetch(self, record: Any) -> FetchResult:
        document = self.documents.get_by_url(record.url)
        request = FetchRequest.revalidate(
            record.url,
            etag=getattr(document, "etag", None) if document else None,
            last_modified=getattr(document, "last_modified", None) if document else None,
        )
        return self.fetcher.fetch(request)

    def _note_source_failure(self, source_id: str) -> None:
        if not source_id:
            return
        count = self._source_failures.get(source_id, 0) + 1
        self._source_failures[source_id] = count
        if count >= self.max_source_failures:
            self._stopped_sources.add(source_id)
            logger.warning(
                "source %s stopped after %d failures (budget %d)",
                source_id,
                count,
                self.max_source_failures,
            )

    # ------------------------------------------------------------------ outcomes
    def _handle_not_modified(
        self, record: Any, result: FetchResult, *, summary: RunSummary
    ) -> None:
        self._upsert_document(record, result, status=304, summary=summary, last_seen=utc())
        self.frontier.complete(record.url, valuable_pages=0, pages=0, tier=result.fetcher_tier)
        summary.bump("not_modified")

    def _handle_ok(
        self, record: Any, result: FetchResult, *, summary: RunSummary, source_id: str
    ) -> None:
        summary.bump("fetched")
        content_type = _header(result, "content-type")
        body = result.body or b""
        sha = self.store.put(body)
        role = self._role(record, content_type=content_type, source_id=source_id, summary=summary)

        document = self._upsert_document(
            record,
            result,
            status=result.status,
            summary=summary,
            last_seen=utc(),
            source_id=source_id,
            role=role,
        )
        latest = self._latest(document)
        summary.bump(f"role.{role}")
        fingerprint = simhash64(result.text) if _is_textual(content_type) else None
        duplicate = False
        reason = ""
        if latest is not None and latest.sha256 == sha:
            duplicate = True
            reason = "unchanged_content"
        elif sha in self._seen_sha:
            duplicate = True
            reason = "duplicate_sha256"
        elif fingerprint is not None and any(
            near_duplicate(fingerprint, other, max_distance=SIMHASH_MAX_DISTANCE)
            for other in self._fingerprints
        ):
            duplicate = True
            reason = "near_duplicate_simhash"

        if duplicate:
            summary.bump("deduped")
            summary.outcomes.append(
                {"url": record.url, "outcome": "deduped", "reason": reason, "sha256": sha}
            )
        else:
            self.snapshots.create(
                SnapshotCreate(
                    document_id=document.id,
                    sha256=sha,
                    bytes_path=str(self.store.path_for(sha)),
                    fetched_at=utc(),
                    fetcher_tier=result.fetcher_tier,
                    content_type=content_type,
                    headers=dict(result.headers),
                )
            )
            summary.bump("stored")
            self._seen_sha.add(sha)
            if fingerprint is not None:
                self._fingerprints.append(fingerprint)

        self.frontier.complete(
            record.url,
            valuable_pages=0 if duplicate else 1,
            pages=1 if result.costs_budget else 0,
            tier=result.fetcher_tier,
        )
        document = self._plan(
            document, source_id=source_id, text_sample=result.text, summary=summary
        )
        if source_id:
            self._source_failures[source_id] = 0

    def _role(
        self,
        record: Any,
        *,
        content_type: str | None,
        source_id: str,
        summary: RunSummary,
    ) -> str:
        """Classify the fetched document (migration 0004).

        A registry typo in ``params.document_role`` is surfaced as a recorded
        outcome, never as a crashed run; the conservative fallback is evidence
        so the planner still gets a chance to abstain honestly.
        """
        try:
            source = self.sources_by_id.get(source_id)
            return role_for(
                url=str(getattr(record, "url", "")),
                content_type=content_type,
                source=source,
                family=getattr(record, "extraction_family", None) or declared_family(source),
            )
        except RoleConfigError as exc:
            summary.outcomes.append(
                {
                    "url": getattr(record, "url", None),
                    "outcome": "role_config_error",
                    "reason": str(exc),
                }
            )
            logger.warning("role classification fell back to evidence: %s", exc)
            return "evidence"

    def _plan(
        self,
        document: DocumentRecord,
        *,
        source_id: str,
        text_sample: str | None,
        summary: RunSummary | None = None,
    ) -> DocumentRecord:
        """Fill family/entity columns for a document the planner can see.

        Runs after the snapshot is stored because PDF/HTML signals live in the
        fetched text. A plan that cannot link an entity leaves ``entity_id`` NULL
        and records ``needs_review``; extraction then refuses to guess.
        """
        if self.planner is None:
            return document
        plan = self.planner.plan(
            document,
            source=self.sources_by_id.get(source_id),
            family_hint=getattr(document, "extraction_family", None),
            entity_id_hint=getattr(document, "entity_id", None),
            text_sample=text_sample,
            role=getattr(document, "role", None),
        )
        updated = persist_plan(self.documents, document, plan)
        if summary is not None:
            summary.bump(f"plan.{plan.state}")
            if plan.abstain_reason:
                summary.bump(f"plan.reason.{plan.abstain_reason}")
        return updated if updated is not None else document

    def _upsert_document(
        self,
        record: Any,
        result: FetchResult,
        *,
        status: int,
        summary: RunSummary,
        last_seen: datetime | None = None,
        source_id: str | None = None,
        role: str | None = None,
    ) -> DocumentRecord:
        robots = result.robots
        source_id = source_id or getattr(record, "source_id", None)
        document = DocumentUpsert(
            url=record.url,
            domain=record.domain,
            kind=None,
            last_seen=last_seen,
            etag=_header(result, "etag"),
            last_modified=_http_datetime(_header(result, "last-modified")),
            robots_decision=getattr(robots, "reason", None),
            robots_checked_at=utc() if robots is not None else None,
            http_status=status,
            http_method=record.http_method,
            http_body=record.http_body,
            request_headers=record.request_headers,
            source_id=source_id,
            extraction_family=getattr(record, "extraction_family", None),
            entity_id=getattr(record, "entity_id", None),
            role=role,
        )
        return self.documents.upsert(document)

    def _latest(self, document: DocumentRecord) -> SnapshotRecord | None:
        return self.snapshots.latest_for_document(document.id)
