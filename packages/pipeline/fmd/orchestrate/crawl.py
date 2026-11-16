"""Crawl orchestration: adapters -> frontier -> fetch ladder -> snapshots.

This module owns sequencing only. It does not decide politeness (the frontier and
fetch ladder do), does not parse documents (adapters do), and does not claim to
have succeeded when a fetch failed: every non-2xx outcome becomes a recorded
rejection, abstention or failure with a machine-readable reason.

Replay metadata is carried in ``request_headers["fmd_plan"]`` on the frontier and
document rows. That is the only replay-safe JSON column available without a
schema change, and it survives a restart so ``fmd extract`` can recover the
source id, extraction family and entity link for each document.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Mapping
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

logger = logging.getLogger(__name__)

PLAN_KEY = "fmd_plan"
# A source that cannot answer this many times in one run is stopped, not retried
# forever. A block/quarantine is handled per URL by the frontier as well.
DEFAULT_SOURCE_FAILURE_BUDGET = 5
SIMHASH_MAX_DISTANCE = 3


def utc() -> datetime:
    return datetime.now(UTC)


def plan_for_ref(ref: Any, source: Any) -> dict[str, Any]:
    """Build the replayable plan for one discovered document.

    Extraction family and entity link are *claims made by discovery*, not
    inferred here. A ref that omits them yields a document that extraction will
    abstain on with ``missing_family``/``unresolved_entity`` rather than guess.
    """
    metadata = getattr(ref, "metadata", None) or {}
    plan: dict[str, Any] = {"source_id": str(source.id)}
    if getattr(source, "kind", None):
        plan["source_kind"] = str(source.kind)
    family = metadata.get("family") or metadata.get("document_family")
    if family:
        plan["family"] = str(family)
    if metadata.get("entity_id") is not None:
        with contextlib.suppress(TypeError, ValueError):
            plan["entity_id"] = int(metadata["entity_id"])
    if metadata.get("institution"):
        plan["institution"] = str(metadata["institution"])
    return plan


def plan_from_headers(headers: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(headers, Mapping):
        return {}
    value = headers.get(PLAN_KEY)
    return dict(value) if isinstance(value, Mapping) else {}


def merge_headers(headers: Mapping[str, Any] | None, plan: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(headers or {})
    merged[PLAN_KEY] = dict(plan)
    return merged


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


def frontier_upsert(ref: Any, plan: Mapping[str, Any]) -> FrontierUpsert:
    """Use the contract's ref-aware enqueue, then add the plan to replay headers.

    ``from_document_ref`` gives us the POST body/header shape adapters emit;
    a conflicting ``request_headers`` argument would duplicate that keyword, so
    the plan is merged afterwards.
    """
    item = FrontierUpsert.from_document_ref(
        ref,
        domain=domain_for(ref.url),
        state="queued",
        next_allowed_at=utc(),
    )
    return item.model_copy(update={"request_headers": merge_headers(item.request_headers, plan)})


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
    ) -> None:
        self.frontier = frontier
        self.fetcher = fetcher
        self.store = store
        self.documents = documents
        self.snapshots = snapshots
        self.adapter_factory = adapter_factory
        self.denylisted_domains = tuple(denylisted_domains or ())
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
    ) -> None:
        remaining = limit
        for source in sources:
            if remaining is not None and remaining <= 0:
                break
            outcome: dict[str, Any] = {
                "source_id": str(getattr(source, "id", "")),
                "adapter": getattr(source, "adapter", None),
                "status": "ok",
                "planned": 0,
                "enqueued": 0,
            }
            try:
                adapter = self.adapter_factory(source)
                domains = list(getattr(source, "seeds", None) or ())
                refs = call_fetch(adapter, domains=domains)
            except Exception as exc:  # one bad source never aborts the run
                reason = getattr(exc, "reason", None) or type(exc).__name__
                detail = str(exc)
                outcome.update({"status": "error", "reason": reason, "detail": detail})
                summary.sources.append(outcome)
                summary.fail("crawl.plan", str(reason), source_id=outcome["source_id"])
                logger.warning("source %s failed to plan: %s", outcome["source_id"], detail)
                continue

            unique = dedupe_refs(list(refs))
            if remaining is not None:
                unique = unique[:remaining]
            outcome["planned"] = len(unique)
            summary.bump("planned", len(unique))
            for ref in unique:
                plan = plan_for_ref(ref, source)
                if dry_run:
                    summary.planned_urls.append(ref.url)
                    continue
                self.frontier.repo.enqueue(frontier_upsert(ref, plan))
                outcome["enqueued"] += 1
                summary.bump("enqueued")
            summary.sources.append(outcome)
            if remaining is not None:
                remaining -= len(unique)

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
                source_id = str(plan_from_headers(record.request_headers).get("source_id") or "")
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
            self._upsert_document(record, result, status=result.status, summary=summary)
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

        document = self._upsert_document(
            record,
            result,
            status=result.status,
            summary=summary,
            last_seen=utc(),
        )
        latest = self._latest(document)
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
        if source_id:
            self._source_failures[source_id] = 0

    def _upsert_document(
        self,
        record: Any,
        result: FetchResult,
        *,
        status: int,
        summary: RunSummary,
        last_seen: datetime | None = None,
    ) -> DocumentRecord:
        robots = result.robots
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
        )
        return self.documents.upsert(document)

    def _latest(self, document: DocumentRecord) -> SnapshotRecord | None:
        return self.snapshots.latest_for_document(document.id)
