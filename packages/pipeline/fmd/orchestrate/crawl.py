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
from dataclasses import dataclass, field
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
from fmd.fetch.dedupe import (
    DEFAULT_SIMHASH_THRESHOLD,
    DedupeSettings,
    dedupe_settings,
    effective_mode,
    hamming,
    is_pdf_payload,
    pdf_text_sample,
    row_identities,
    simhash64,
    similarity,
    text_sha256,
)
from fmd.fetch.politeness import registrable_domain
from fmd.fetch.types import FetchRequest, FetchResult, TierUnavailable, http_date
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
#: Backwards-compatible alias; the effective value is read per source.
SIMHASH_MAX_DISTANCE = DEFAULT_SIMHASH_THRESHOLD
PAGINATION_REPEAT = "pagination_repeat"


@dataclass(frozen=True)
class ReplayRequest(FetchRequest):
    """A fetch request carrying the persisted replay shape.

    ``fmd.fetch.types.FetchRequest`` is declared as a conditional GET and the
    ladder tiers only read ``url``/validators. The frontier, however, persists
    ``http_method``/``http_body``/``request_headers`` per document (migration
    0001) because some sources answer only a POST body. The crawler builds the
    complete request here instead of dropping that shape on the floor.

    Interface gap (reported, not worked around): ``FetchLadder.fetch`` and the
    tier fetchers ignore ``method``/``body`` and always send GET. Until the
    fetch layer accepts these fields, a paginated POST source still serves page
    1; the crawler surfaces that as a ``pagination_repeat`` warning.
    """

    method: str = "GET"
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)


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
        # sha256 -> (url, document_id); identities are scoped per source so a
        # mirrored listing on another host cannot suppress a real page.
        self._seen_sha: dict[str, tuple[str, int]] = {}
        # text-sha256 -> (url, document_id): identical extracted text under a
        # different byte representation (recompressed/mirrored PDF).
        self._seen_text_sha: dict[str, tuple[str, int]] = {}
        self._fingerprints: list[tuple[int, str, int, str]] = []
        self._identities: dict[str, set[str]] = {}
        self._identity_owner: dict[str, tuple[str, int, str]] = {}
        self._source_failures: dict[str, int] = {}
        self._stopped_sources: set[str] = set()
        # Consecutive successful pages per source, for pagination diagnostics.
        self._last_ok_sha: dict[str, str] = {}
        self._last_ok_url: dict[str, str] = {}

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
            result = self._fetch(record, summary=summary)
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

    def _fetch(self, record: Any, *, summary: RunSummary | None = None) -> FetchResult:
        """Fetch with the request shape persisted on the frontier row.

        A method-blind GET is what broke NABH's ``admin-ajax.php`` (HTTP 400) and
        silently repeated page 1 for every paginated POST source. The persisted
        shape wins; the stored ``documents`` row is the fallback so a re-crawl
        cannot downgrade a POST because a later discovery ref omitted the body.
        """
        request = self._request_for(record)
        if summary is not None:
            methods = summary.metadata.setdefault("request_methods", {})
            methods[request.method] = methods.get(request.method, 0) + 1
        return self.fetcher.fetch(request)

    def _request_for(self, record: Any) -> ReplayRequest:
        document = self.documents.get_by_url(record.url)
        document_method = (
            str(getattr(document, "http_method", None) or "GET").upper() if document else "GET"
        )
        method = str(getattr(record, "http_method", None) or document_method).upper()
        body = getattr(record, "http_body", None)
        if body is None and document is not None:
            body = getattr(document, "http_body", None)
        # A frontier row that lost its POST shape must not override a document
        # that still records one (the failure mode this guard prevents).
        if method == "GET" and document_method == "POST" and body is not None:
            method = "POST"
        if method not in {"GET", "POST"}:
            raise ValueError(f"unsupported persisted request method {method!r} (GET or POST)")

        raw_headers = getattr(record, "request_headers", None)
        if not raw_headers and document is not None:
            raw_headers = getattr(document, "request_headers", None)
        headers = {str(key): str(value) for key, value in dict(raw_headers or {}).items()}

        # Conditional re-crawl: the stored validator is the newest one; a header
        # folded in at discovery time is the fallback. Overwrite (not setdefault)
        # so a stale discovery ETag cannot mask the last fetched one.
        etag = (getattr(document, "etag", None) if document else None) or headers.get(
            "If-None-Match"
        )
        last_modified = (
            http_date(getattr(document, "last_modified", None)) if document else None
        ) or headers.get("If-Modified-Since")
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        referer = headers.get("Referer") or headers.get("referer")
        return ReplayRequest(
            url=record.url,
            etag=etag,
            last_modified=last_modified,
            referer=referer,
            method=method,
            body=dict(body) if isinstance(body, dict) else None,
            headers=headers,
        )

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
        self._note_page_content(record, sha, summary=summary, source_id=source_id)
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
        decision = self._dedupe_decision(
            record,
            result,
            sha=sha,
            content_type=content_type,
            source_id=source_id,
            latest=latest,
        )
        mode = decision["mode"]
        duplicate = decision["duplicate"]
        reason = decision["reason"]

        if duplicate:
            summary.bump("deduped")
            entry = {
                "url": record.url,
                "outcome": "deduped",
                "reason": reason,
                "sha256": sha,
                "mode": mode,
                "basis": decision["basis"],
                "similarity": decision["similarity"],
                "compared_document_id": decision["compared_document_id"],
                "compared_url": decision["compared_url"],
                "text_sha256": decision["text_sha256"],
                "payload_kind": decision["payload_kind"],
                "text_extraction": decision["text_extraction"],
            }
            if decision["hamming_distance"] is not None:
                entry["hamming_distance"] = decision["hamming_distance"]
            if source_id:
                entry["source_id"] = source_id
            summary.outcomes.append(entry)
            # A near-duplicate must never look like "sparse data" in a run
            # summary: keep the threshold, the compared document, the basis and
            # the text hash so a review can tell boilerplate from a real mirror.
            summary.metadata.setdefault("dedupe", []).append(dict(entry))
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
                    text_sha256=decision["text_sha256"],
                    dedupe_basis=decision["basis"],
                )
            )
            summary.bump("stored")
            self._remember(
                sha=sha,
                text_sha=decision["text_sha256"],
                url=record.url,
                document_id=document.id,
                fingerprint=decision["fingerprint"],
                identities=decision["identities"],
                source_id=source_id,
            )

        # Persist the decision on the document row itself. A dropped document
        # has no snapshot, so this is the only durable answer to "why".
        document = self._record_decision(document, decision)

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

    def _dedupe_decision(
        self,
        record: Any,
        result: FetchResult,
        *,
        sha: str,
        content_type: str | None,
        source_id: str,
        latest: SnapshotRecord | None,
    ) -> dict[str, Any]:
        """Decide whether this fetch is a duplicate, and say exactly why.

        The policy is registry-driven. Structured payloads default to exact
        matching because simhash on a paginated JSON wrapper compares
        boilerplate, not rows. ``identity`` mode additionally compares row
        identities so two pages of one listing stay distinct.

        A PDF is never fingerprinted from ``result.text``: that is the container
        bytes decoded as UTF-8. Its fingerprint is a bounded extracted-text
        sample. If that sample is unavailable the decision degrades to exact
        sha256 (``basis='raw_bytes'``) and the document is stored. A PDF is only
        dropped on *identical* extracted text: two editions of one template are
        a few tokens apart and must both survive.
        """
        settings: DedupeSettings = dedupe_settings(self.sources_by_id.get(source_id))
        payload_kind = "pdf" if is_pdf_payload(content_type, result.body) else "text"
        text: str
        text_digest: str | None
        if payload_kind == "pdf":
            sample = pdf_text_sample(
                result.body,
                max_pages=settings.pdf_text_max_pages,
                max_chars=settings.pdf_text_max_chars,
            )
            if sample is None:
                text = ""
                text_digest = None
                text_extraction = "unavailable"
            else:
                text = sample
                text_digest = text_sha256(sample)
                text_extraction = "extracted"
        else:
            text = result.text
            text_digest = text_sha256(text) if text.strip() else None
            text_extraction = "decoded" if text_digest else "unavailable"

        mode = effective_mode(settings, content_type=content_type, text=text)
        # No text basis: the only defensible comparison is the exact bytes.
        if mode == "simhash" and not text:
            mode = "exact"
        fingerprint = simhash64(text) if mode == "simhash" else None
        identities = (
            row_identities(text, settings.identity_fields) if mode == "identity" else frozenset()
        )
        if mode == "identity":
            basis = "identity"
        elif mode == "simhash":
            basis = "extracted_text"
        else:
            basis = "raw_bytes"

        decision: dict[str, Any] = {
            "mode": mode,
            "basis": basis,
            "fingerprint": fingerprint,
            "simhash_hex": f"{fingerprint:016x}" if fingerprint is not None else None,
            "identities": identities,
            "text_sha256": text_digest,
            "text_extraction": text_extraction,
            "payload_kind": payload_kind,
            "duplicate": False,
            "reason": "",
            "similarity": None,
            "hamming_distance": None,
            "compared_document_id": None,
            "compared_url": None,
        }

        if latest is not None and latest.sha256 == sha:
            decision.update(
                {
                    "duplicate": True,
                    "reason": "unchanged_content",
                    "basis": "raw_bytes",
                    "similarity": 1.0,
                    "compared_document_id": latest.document_id,
                    "compared_url": record.url,
                }
            )
            return decision

        owner = self._seen_sha.get(sha)
        if owner is not None:
            decision.update(
                {
                    "duplicate": True,
                    "reason": "duplicate_sha256",
                    "basis": "raw_bytes",
                    "similarity": 1.0,
                    "compared_document_id": owner[1],
                    "compared_url": owner[0],
                }
            )
            return decision

        # Same text under different bytes: a recompressed or mirrored PDF. Only
        # meaningful when the mode actually uses text.
        if mode == "simhash" and text_digest is not None:
            text_owner = self._seen_text_sha.get(text_digest)
            if text_owner is not None:
                decision.update(
                    {
                        "duplicate": True,
                        "reason": "duplicate_text_sha256",
                        "similarity": 1.0,
                        "compared_document_id": text_owner[1],
                        "compared_url": text_owner[0],
                    }
                )
                return decision

        # A PDF is only dropped when its *full* extracted text matches. Simhash
        # is still computed and recorded, but a different edition of the same
        # template differs in a handful of tokens (a year, a seat count), which
        # sits at distance 0 and is nonetheless a different document. Approximate
        # similarity may never authorise dropping document evidence.
        if mode == "simhash" and fingerprint is not None and payload_kind != "pdf":
            closest = min(
                self._fingerprints,
                key=lambda row: hamming(fingerprint, row[0]),
                default=None,
            )
            if closest is not None:
                distance = hamming(fingerprint, closest[0])
                if distance <= settings.simhash_threshold:
                    decision.update(
                        {
                            "duplicate": True,
                            "reason": "near_duplicate_simhash",
                            "similarity": similarity(fingerprint, closest[0]),
                            "hamming_distance": distance,
                            "compared_document_id": closest[2],
                            "compared_url": closest[1],
                        }
                    )
            return decision

        if mode == "identity" and identities:
            key = source_id or ""
            seen = self._identities.get(key, set())
            if seen and identities <= seen:
                sample = next(identity for identity in identities if identity in seen)
                owner = self._identity_owner.get(f"{key}\x00{sample}")
                decision.update(
                    {
                        "duplicate": True,
                        "reason": "near_duplicate_identity",
                        "similarity": len(identities & seen) / len(identities),
                        "compared_document_id": owner[1] if owner else None,
                        "compared_url": owner[0] if owner else None,
                    }
                )
            return decision

        return decision

    def _record_decision(
        self, document: DocumentRecord, decision: dict[str, Any]
    ) -> DocumentRecord:
        """Persist the dedupe decision on the document row (migration 0007)."""
        payload = DocumentUpsert(
            **document.model_copy(
                update={
                    "text_sha256": decision["text_sha256"],
                    "dedupe_mode": decision["mode"],
                    "dedupe_basis": decision["basis"],
                    "dedupe_similarity": decision["similarity"],
                    "dedupe_reason": decision["reason"] or None,
                    "dedupe_compared_document_id": decision["compared_document_id"],
                    "dedupe_simhash": decision["simhash_hex"],
                }
            ).model_dump()
        )
        return self.documents.upsert(payload)

    def _remember(
        self,
        *,
        sha: str,
        text_sha: str | None,
        url: str,
        document_id: int,
        fingerprint: int | None,
        identities: frozenset[str],
        source_id: str,
    ) -> None:
        """Record what a stored snapshot represents, for future comparisons."""
        self._seen_sha.setdefault(sha, (url, document_id))
        if text_sha is not None:
            self._seen_text_sha.setdefault(text_sha, (url, document_id))
        if fingerprint is not None:
            self._fingerprints.append((fingerprint, url, document_id, source_id))
        if identities:
            key = source_id or ""
            self._identities.setdefault(key, set()).update(identities)
            for identity in identities:
                self._identity_owner.setdefault(
                    f"{key}\x00{identity}", (url, document_id, source_id)
                )

    def _note_page_content(
        self, record: Any, sha: str, *, summary: RunSummary, source_id: str
    ) -> None:
        """Flag consecutive pages of one source that are byte-identical.

        A paginated POST source whose page parameter is dropped answers page 1
        forever with HTTP 200. Counts look plausible and the data looks sparse;
        this warning makes the repeat explicit in the run summary instead. It is
        a warning, not a failure: a mirror that legitimately serves one body
        under two URLs must not fail the run.
        """
        if not source_id:
            return
        previous_sha = self._last_ok_sha.get(source_id)
        previous_url = self._last_ok_url.get(source_id)
        if previous_sha == sha and previous_url != str(getattr(record, "url", "")):
            entry = {
                "url": getattr(record, "url", None),
                "previous_url": previous_url,
                "outcome": "warning",
                "reason": PAGINATION_REPEAT,
                "source_id": source_id,
                "sha256": sha,
            }
            summary.bump(PAGINATION_REPEAT)
            summary.outcomes.append(entry)
            summary.metadata.setdefault("warnings", []).append(entry)
        self._last_ok_sha[source_id] = sha
        self._last_ok_url[source_id] = str(getattr(record, "url", ""))

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
        # HTTP text is readable immediately (HTML/plain). A PDF's HTTP body is
        # not its text: that document stays deferred until the acquisition stage
        # can read it, instead of being abstained on from the URL alone.
        phase = "b" if text_sample and text_sample.strip() else "a"
        plan = self.planner.plan(
            document,
            source=self.sources_by_id.get(source_id),
            family_hint=getattr(document, "extraction_family", None),
            entity_id_hint=getattr(document, "entity_id", None),
            text_sample=text_sample if phase == "b" else None,
            role=getattr(document, "role", None),
            phase=phase,
        )
        updated = persist_plan(self.documents, document, plan)
        if summary is not None:
            summary.bump(f"plan.{plan.state}")
            if plan.deferred:
                summary.bump("plan.deferred")
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
        try:
            shape = self._request_for(record)
            http_method, http_body, request_headers = (
                shape.method,
                shape.body,
                shape.headers,
            )
        except ValueError:  # malformed persisted method: never rewrite the row
            http_method = getattr(record, "http_method", "GET")
            http_body = getattr(record, "http_body", None)
            request_headers = getattr(record, "request_headers", None)
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
            http_method=http_method,
            http_body=http_body,
            request_headers=request_headers,
            source_id=source_id,
            extraction_family=getattr(record, "extraction_family", None),
            entity_id=getattr(record, "entity_id", None),
            role=role,
        )
        return self.documents.upsert(document)

    def _latest(self, document: DocumentRecord) -> SnapshotRecord | None:
        return self.snapshots.latest_for_document(document.id)
