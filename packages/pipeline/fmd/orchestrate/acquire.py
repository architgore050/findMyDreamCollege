"""Acquisition stage (W10): convert documents into cached parses — zero LLM calls.

``fmd acquire`` is the conversion half of the phase split: it runs the tier
dispatcher for every planned snapshot and writes complete T1 parses to the
persistent parse cache (migration 0011 + object store). It never touches the
model router — callers may hand one over (the orchestrator's) and it stays
untouched, which is exactly what the zero-LLM-call test asserts.

Selection is the extractor's own queue, in the extractor's own order:
``snapshots.list_extractable(tool_version, limit)`` resolved through
``target_for_document`` — the same priority the extractor uses today, so a
parse written here is the parse the next ``fmd extract`` run consumes.

A conversion that fails writes nothing (a cache entry represents a complete
document only), the document is recorded with a stable reason and the run
continues with the next one. ``AcquisitionFailed`` (device/budget aborts)
additionally parks the document ``needs_review`` and fails the run summary —
the runbook contract, not a new decision.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from fmd.acquire.errors import ACQUISITION_FAILED_PREFIX, AcquisitionFailed
from fmd.acquire.parse_cache import ParseCache, acquire_document, guard_conversion_error
from fmd.acquire.tiers import AcquirePolicy
from fmd.extract.pipeline import target_for_document
from fmd.orchestrate.extract import DEFAULT_TOOL_VERSION
from fmd.orchestrate.ledger import RunSummary
from fmd.store import SnapshotNotFound, SnapshotStore

logger = logging.getLogger(__name__)

#: Env var honoured by ``docling_device``; the stage sets it around the run so
#: ``--device`` and the batched runner can pin CUDA for the acquisition phase
#: without editing config. The previous value is always restored.
DEVICE_ENV = "FMD_DOCLING_DEVICE"

#: Stable reason recorded when the stored snapshot bytes are unreadable.
SNAPSHOT_MISSING = "snapshot_bytes_missing"


@dataclass(frozen=True)
class AcquisitionItem:
    """One selected snapshot with what the parse cache already knows."""

    snapshot_id: int
    document_id: int
    url: str | None
    sha256: str
    content_type: str | None
    #: Cached-parse status: ``True`` ⇒ this document is a guaranteed cache hit
    #: and would convert zero pages; ``False`` ⇒ live conversion needed, with
    #: ``pages`` unknown until it completes (never guessed, never truncated).
    cache_hit: bool
    pages: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "document_id": self.document_id,
            "url": self.url,
            "sha256": self.sha256,
            "cache_hit": self.cache_hit,
            "pages": self.pages,
        }


class AcquisitionStage:
    """Convert planned snapshots into cached parses. No router, no LLM."""

    def __init__(
        self,
        *,
        snapshots: Any,
        documents: Any,
        store: SnapshotStore | Any,
        parse_cache: ParseCache | Any | None,
        policy: AcquirePolicy | None = None,
        tool_version: str = DEFAULT_TOOL_VERSION,
        t0: Any = None,
        t1: Any = None,
        device: str | None = None,
        t2_parse_cache: ParseCache | Any | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.documents = documents
        self.store = store
        self.parse_cache = parse_cache
        self.t2_parse_cache = t2_parse_cache
        self.policy = policy
        self.tool_version = tool_version
        self.t0 = t0
        self.t1 = t1
        self.device = device

    # ------------------------------------------------------------------ selection
    def plan(
        self, *, limit: int | None = None, selected: list[Any] | None = None
    ) -> list[AcquisitionItem]:
        """The would-convert queue with cached-parse status, touching nothing.

        Same selection and order as the extractor. Page counts are reported
        only where the cache already knows them; uncached pages stay ``None``
        (the honest state — they become known exactly when conversion does).
        """
        pending = (
            list(selected)
            if selected is not None
            else self.snapshots.list_extractable(
                self.tool_version, limit if limit is not None else 500
            )
        )
        items: list[AcquisitionItem] = []
        for snapshot in pending:
            document = self.documents.get(snapshot.document_id)
            target, _reason = self._target(snapshot, document)
            if target is None:
                continue
            hit, pages = self._cache_status(snapshot.sha256)
            items.append(
                AcquisitionItem(
                    snapshot_id=snapshot.id,
                    document_id=snapshot.document_id,
                    url=getattr(document, "url", None),
                    sha256=snapshot.sha256,
                    content_type=snapshot.content_type,
                    cache_hit=hit,
                    pages=pages,
                )
            )
        return items

    def _cache_status(self, sha256: str) -> tuple[bool, int | None]:
        """Cached-parse status across the tier caches (T1 first, then T2).

        A T2 row means these exact bytes were read completely by the pinned
        OCR engine under the current policy — the snapshot converts zero pages
        in this stage, exactly like a T1 hit.
        """
        for cache in (self.parse_cache, self.t2_parse_cache):
            if cache is None:
                continue
            try:
                if cache.has_parse(sha256, self.policy):
                    return True, cache.pages_for(sha256, self.policy)
            except Exception as exc:  # a plan must not fail on cache trouble
                logger.warning("parse-cache peek failed for snapshot %s: %s", sha256, exc)
        return False, None

    # ------------------------------------------------------------------ execution
    def run(
        self,
        *,
        summary: RunSummary,
        limit: int | None = None,
        pages_budget: int | None = None,
        selected: list[Any] | None = None,
        dry_run: bool = False,
        data_for: Any = None,
    ) -> None:
        restore = _set_device_env(self.device)
        try:
            self._run(
                summary=summary,
                limit=limit,
                pages_budget=pages_budget,
                selected=selected,
                dry_run=dry_run,
                data_for=data_for,
            )
        finally:
            restore()

    def _run(
        self,
        *,
        summary: RunSummary,
        limit: int | None,
        pages_budget: int | None,
        selected: list[Any] | None,
        dry_run: bool,
        data_for: Any = None,
    ) -> None:
        items = self.plan(limit=limit, selected=selected)
        summary.bump("selected", len(items))
        converted_pages = 0
        parsed = 0
        for item in items:
            # Page budgets are evaluated only at document boundaries: a budget
            # already reached stops *taking new documents*; the document that
            # crossed it was finished in full before this check runs again.
            if pages_budget is not None and converted_pages >= pages_budget:
                summary.bump("deferred")
                continue
            if dry_run:
                self._record_plan(summary, item)
                continue
            outcome = self._acquire_one(item, summary, data_for=data_for)
            if outcome is None:
                continue
            parsed += 1
            pages = int(outcome.get("pages") or 0)
            if pages:
                summary.bump("pages", pages)
            if not item.cache_hit:
                converted_pages += pages
        summary.bump("parsed", parsed)
        summary.metadata["device"] = self.device or _current_device()
        summary.metadata["pages_budget"] = pages_budget
        summary.metadata["zero_llm_calls"] = True

    def _record_plan(self, summary: RunSummary, item: AcquisitionItem) -> None:
        if item.cache_hit:
            summary.bump("cache_hits")
            summary.bump("parsed")
        else:
            summary.bump("to_convert")
        summary.outcomes.append(
            {
                **item.as_dict(),
                "outcome": "cache_hit" if item.cache_hit else "would_convert",
            }
        )

    def _acquire_one(
        self,
        item: AcquisitionItem,
        summary: RunSummary,
        *,
        data_for: Any = None,
    ) -> dict[str, Any] | None:
        """Convert (or cache-hit) one snapshot; append exactly one outcome row.

        ``data_for(sha256)`` returns prefetched snapshot bytes (or None — the
        caller then reads live); it is consulted only on the convert path.
        """
        if item.cache_hit:
            # Lookup happens before conversion; a verified hit reconstructs
            # the parse and invokes no converter at all. T1 is consulted
            # first, then T2 (the plan reported a hit in either cache).
            first_miss: dict[str, Any] | None = None
            for cache in (self.parse_cache, self.t2_parse_cache):
                if cache is None:
                    continue
                doc, provenance = cache.lookup(item.sha256, self.policy)
                if doc is not None:
                    hit_info = provenance.as_dict()
                    summary.bump("cache_hits")
                    summary.outcomes.append(
                        {
                            **item.as_dict(),
                            "outcome": "cache_hit",
                            "pages": hit_info.get("pages"),
                            "key_digest": hit_info.get("key_digest"),
                            "tier": cache.tier,
                        }
                    )
                    return {"pages": hit_info.get("pages") or 0}
                if first_miss is None:
                    first_miss = provenance.as_dict()
            # The rows said hit but the payloads failed verification
            # (corrupt/truncated/moved): reconvert, record the reason.
            summary.bump("converted")
            summary.bump("cache_misses")
            return self._convert(
                item,
                summary,
                force_reason=(first_miss or {}).get("reason"),
                data=self._prefetched(data_for, item.sha256),
            )
        return self._convert(item, summary, data=self._prefetched(data_for, item.sha256))

    @staticmethod
    def _prefetched(data_for: Any, sha256: str) -> bytes | None:
        """Prefetched bytes for a snapshot, or None (read live in _convert)."""
        if data_for is None:
            return None
        try:
            return data_for(sha256)
        except Exception:  # a broken prefetch must never fail the document
            return None

    def _convert(
        self,
        item: AcquisitionItem,
        summary: RunSummary,
        *,
        force_reason: str | None = None,
        data: bytes | None = None,
    ) -> dict[str, Any] | None:
        try:
            # ``data`` arrives prefetched (double-buffered runner): the bytes
            # were read off-thread while the previous cycle's extraction ran.
            # None ⇒ read live, exactly as before this seam existed.
            data = data if data is not None else self.store.get(item.sha256)
        except SnapshotNotFound as exc:
            summary.bump("abstained")
            summary.outcomes.append(
                {
                    **item.as_dict(),
                    "outcome": "abstained",
                    "reason": SNAPSHOT_MISSING,
                    "detail": str(exc),
                }
            )
            return None
        except Exception as exc:  # store trouble is recorded, never a crash
            summary.bump("abstained")
            summary.outcomes.append(
                {**item.as_dict(), "outcome": "abstained", "reason": type(exc).__name__}
            )
            return None
        document = self.documents.get(item.document_id)
        try:
            doc, provenance = acquire_document(
                data,
                content_type=item.content_type,
                policy=self.policy,
                cache=self.parse_cache,
                t2_cache=self.t2_parse_cache,
                snapshot_sha256=item.sha256,
                t0=self.t0,
                t1=self.t1,
            )
        except AcquisitionFailed as exc:
            # Device/budget abort: park the document with the stable reason and
            # fail the run summary (runbook §6 contract). Nothing was cached.
            reason = str(getattr(exc, "reason", None) or guard_conversion_error(exc))
            self._mark_needs_review(document, reason)
            summary.bump("failed")
            summary.fail(
                "acquire",
                reason,
                snapshot_id=item.snapshot_id,
                document_id=item.document_id,
                url=item.url,
                detail=str(exc),
            )
            summary.outcomes.append(
                {
                    **item.as_dict(),
                    "outcome": "acquisition_failed",
                    "reason": reason,
                    "detail": str(exc),
                }
            )
            return None
        except Exception as exc:
            # Any other conversion failure writes nothing and continues with
            # the next document; the reason is recorded, never guessed away.
            reason = guard_conversion_error(exc)
            summary.bump("abstained")
            summary.outcomes.append(
                {
                    **item.as_dict(),
                    "outcome": "abstained",
                    "reason": reason,
                    "detail": str(exc)[:500],
                }
            )
            return None
        info = provenance.as_dict()
        pages = int((getattr(provenance, "info", None) or {}).get("pages") or 0)
        if info.get("mode") == "cache":
            # A document that converted on a previous run of this same stage.
            summary.bump("cache_hits")
        else:
            summary.bump("converted")
            if info.get("stored"):
                summary.bump("cache_stores")
            else:
                summary.bump("cache_store_skipped")
        summary.outcomes.append(
            {
                **item.as_dict(),
                "outcome": "cache_hit" if info.get("mode") == "cache" else "converted",
                "pages": pages,
                "tier": getattr(doc, "tier", None),
                "reason": force_reason or info.get("reason"),
                "stored": bool(info.get("stored")) or info.get("mode") == "cache",
            }
        )
        return {"pages": pages}

    def _target(self, snapshot: Any, document: Any) -> tuple[Any | None, str | None]:
        if document is None:
            return None, "missing_document"
        return target_for_document(document, snapshot.id)

    def _mark_needs_review(self, document: Any, reason: str) -> None:
        setter = getattr(self.documents, "set_plan", None)
        if setter is None or document is None:
            return
        try:
            setter(
                int(document.id),
                extraction_family=getattr(document, "extraction_family", None),
                entity_id=getattr(document, "entity_id", None),
                plan_state="needs_review",
                plan_reason=reason,
                role=getattr(document, "role", None),
            )
        except Exception as exc:  # observability must not mask the failure
            logger.warning("could not persist acquisition failure on document: %s", exc)


def _set_device_env(device: str | None):
    """Pin ``FMD_DOCLING_DEVICE`` for the duration of the run; restore after."""
    if device is None:
        return lambda: None
    previous = os.environ.get(DEVICE_ENV)

    def restore() -> None:
        if previous is None:
            os.environ.pop(DEVICE_ENV, None)
        else:
            os.environ[DEVICE_ENV] = previous

    os.environ[DEVICE_ENV] = device
    return restore


def _current_device() -> str | None:
    return os.environ.get(DEVICE_ENV)


__all__ = [
    "ACQUISITION_FAILED_PREFIX",
    "AcquisitionItem",
    "AcquisitionStage",
    "DEVICE_ENV",
    "SNAPSHOT_MISSING",
]
