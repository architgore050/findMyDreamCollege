"""Extract orchestration: pick planned snapshots, run the extract pipeline.

The pipeline itself (WS-D) owns PDF tiering, guided JSON and span validation.
This module owns the queue: which snapshot to process, which entity and family
it belongs to, and what to do when those are not resolvable. A document that
cannot be extracted is recorded as an abstention with a reason code; it is never
written as a silent empty success.

Since migration 0003 the queue is SQL, not JSON-in-a-header: selection reads
``documents.extraction_family`` / ``documents.entity_id``, and
:func:`fmd.extract.pipeline.target_for_document` turns those columns into an
``ExtractTarget``. Documents that were never planned are counted separately
(``unplanned``) so the run ledger shows exactly how much evidence the crawl
stage left behind.

Resumability: selection is ``snapshots.list_extractable(tool_version)`` and the
pipeline skips a snapshot already extracted at the same
``tool_version + prompt_version``, so re-running after an interruption resumes
instead of duplicating evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from fmd.acquire.errors import ACQUISITION_FAILED_PREFIX
from fmd.db.repositories import ClaimsRepo, ExtractionsRepo, SectionsRepo, SnapshotsRepo
from fmd.extract.chunking import CHUNK_BUDGET_EXHAUSTED, CHUNK_FAILED_PREFIX
from fmd.extract.pipeline import (
    ExtractPipeline,
    ExtractTarget,
    build_router,
    target_for_document,
)
from fmd.extract.validate import Rejection
from fmd.orchestrate.ledger import RunSummary

logger = logging.getLogger(__name__)

DEFAULT_TOOL_VERSION = "1.0.0"


class ForcedExtractPipeline(ExtractPipeline):
    """``extract --force`` re-runs a snapshot even at the same tool version.

    Only the skip predicate changes; validation, spans, cost and audit are the
    pipeline's own code, never duplicated here.
    """

    def _already_extracted(self, snapshot_id: int, prompt_ref: str) -> bool:
        return False


class CountingRouter:
    """Delegating router wrapper that counts LLM calls for the run summary.

    ``llm_calls`` in the extract summary counts every routed call of the run —
    extraction, chunk windows and the verification pass alike. Everything else
    (``models``, budgets, providers) delegates untouched, so the pipeline and
    verifier cannot tell the difference.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def models(self) -> Any:
        return self._inner.models

    def call(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._inner.call(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class Extractor:
    def __init__(
        self,
        *,
        snapshots: SnapshotsRepo | Any,
        documents: Any,
        extractions: ExtractionsRepo | Any,
        claims: ClaimsRepo | Any,
        entities: Any,
        store: Any,
        router: Any,
        tool_version: str = DEFAULT_TOOL_VERSION,
        pipeline_factory: Any = None,
        verifier: Any | None = None,
        verifier_factory: Any = None,
        verify: bool = True,
        predicates: Any | None = None,
        sections: SectionsRepo | Any | None = None,
        parse_cache: Any | None = None,
        t0: Any = None,
        t1: Any | None = None,
        allow_convert: bool = True,
    ) -> None:
        self.snapshots = snapshots
        self.documents = documents
        self.extractions = extractions
        self.claims = claims
        self.entities = entities
        self.store = store
        self.router = router
        self.tool_version = tool_version
        self.pipeline_factory = pipeline_factory
        self.verifier = verifier
        self.verifier_factory = verifier_factory
        self.verify = verify
        self.predicates = predicates
        #: Section-tree repo (W8); ``None`` keeps the outline audit-only.
        self.sections = sections
        #: Parse cache (W10). ``None`` keeps the fused inline-conversion path.
        self.parse_cache = parse_cache
        #: Injected T0/T1 tier callables (test seams), passed to the pipeline.
        self.t0 = t0
        self.t1 = t1
        #: ``False`` for ``--from-cache`` / ``--no-acquire``: a missing cached
        #: parse is recorded (``missing_parse_cache``), never converted inline.
        self.allow_convert = allow_convert
        self._counting_router: CountingRouter | None = None
        self._verifier_ready = False
        self._verifier_cost = 0.0
        self._verifier_exhausted = False
        self._verifier_error: str | None = None

    def _pipeline(self, *, force: bool) -> ExtractPipeline:
        cls = ForcedExtractPipeline if force else ExtractPipeline
        if self.pipeline_factory is not None:
            return self.pipeline_factory(cls)
        router = self._counting_router if self._counting_router is not None else self.router
        return cls(
            router,
            snapshots=self.snapshots,
            extractions=self.extractions,
            claims=self.claims,
            store=self.store,
            tool_version=self.tool_version,
            sections=self.sections,
            parse_cache=self.parse_cache,
            t0=self.t0,
            t1=self.t1,
            allow_convert=self.allow_convert,
        )

    def run(
        self,
        *,
        summary: RunSummary,
        limit: int | None = None,
        force: bool = False,
        selected: list[Any] | None = None,
    ) -> None:
        self._count_unplanned(summary)
        if selected is not None:
            # Batched mode (W10): the runner hands over exactly the snapshots
            # of one page-batch; the queue is not re-read here.
            pending = list(selected)
        else:
            pending = self.snapshots.list_extractable(
                None if force else self.tool_version,
                limit if limit is not None else 500,
            )
        # One counter for every routed call of this run (extraction, chunk
        # windows, verification): the summary's ``llm_calls``.
        self._counting_router = (
            CountingRouter(self.router) if self.router is not None else None
        )
        pipeline = self._pipeline(force=force)
        verifier = self._get_verifier()
        if verifier is not None:
            verifier.start_run()
        for snapshot in pending:
            document = self.documents.get(snapshot.document_id)
            target, reason = self._target(snapshot, document)
            if target is None:
                summary.bump("abstained")
                summary.outcomes.append(
                    {
                        "snapshot_id": snapshot.id,
                        "document_id": snapshot.document_id,
                        "url": getattr(document, "url", None),
                        "outcome": "abstained",
                        "reason": reason,
                    }
                )
                continue
            try:
                result = pipeline.run_batch([target])[0]
            except Exception as exc:  # per-document isolation
                reason = getattr(exc, "reason", None) or type(exc).__name__
                if str(reason).startswith(ACQUISITION_FAILED_PREFIX):
                    # A starved GPU is a failed run, not an empty success. The
                    # document is parked for review with the stable reason and
                    # the summary is marked failed so the CLI exits non-zero.
                    self._mark_acquisition_failed(document, str(reason))
                    summary.bump("acquisition_failed")
                    summary.fail(
                        "extract",
                        str(reason),
                        snapshot_id=snapshot.id,
                        document_id=snapshot.document_id,
                        url=getattr(document, "url", None),
                        detail=str(exc),
                    )
                    summary.outcomes.append(
                        {
                            "snapshot_id": snapshot.id,
                            "document_id": snapshot.document_id,
                            "url": getattr(document, "url", None),
                            "outcome": "acquisition_failed",
                            "reason": str(reason),
                            "detail": str(exc),
                        }
                    )
                    logger.error(
                        "acquisition failed for snapshot %s: %s", snapshot.id, exc
                    )
                    continue
                logger.warning("extraction abstained for snapshot %s: %s", snapshot.id, exc)
                summary.bump("abstained")
                summary.outcomes.append(
                    {
                        "snapshot_id": snapshot.id,
                        "document_id": snapshot.document_id,
                        "url": getattr(document, "url", None),
                        "outcome": "abstained",
                        "reason": str(reason),
                        "detail": str(exc),
                    }
                )
                continue

            if result.skipped:
                acquire = getattr(result, "acquire", None)
                reason = acquire.get("reason") if isinstance(acquire, dict) else None
                if reason == "missing_parse_cache":
                    # --from-cache / --no-acquire with no cached parse: the
                    # document is recorded and the run continues. The exit rule
                    # is unchanged — non-zero only when nothing at all was
                    # extractable (abstained feeds that rule; extracted or
                    # already-extracted documents clear it).
                    summary.bump("missing_parse_cache")
                    summary.bump("abstained")
                    summary.outcomes.append(
                        {
                            "snapshot_id": snapshot.id,
                            "document_id": snapshot.document_id,
                            "url": getattr(document, "url", None),
                            "outcome": "abstained",
                            "reason": "missing_parse_cache",
                        }
                    )
                    continue
                summary.bump("skipped")
                summary.outcomes.append(
                    {
                        "snapshot_id": snapshot.id,
                        "document_id": snapshot.document_id,
                        "outcome": "skipped",
                        "reason": "already_extracted",
                    }
                )
                continue

            rejected = len(result.rejections)
            verification = self._verify(result, verifier)
            summary.bump("extracted")
            summary.bump("claims", len(result.claims))
            summary.bump("rejected_claims", rejected)
            acquire = self._record_acquire(summary, result)
            entry: dict[str, Any] = {
                "snapshot_id": snapshot.id,
                "document_id": snapshot.document_id,
                "url": getattr(document, "url", None),
                "outcome": "extracted",
                "extraction_id": result.extraction_id,
                "family": target.family,
                "entity_id": target.entity_id,
                "claims": len(result.claims),
                "rejected": rejected,
            }
            if isinstance(acquire, dict):
                entry["acquire"] = acquire
            chunking = getattr(result, "chunking", None)
            if isinstance(chunking, dict) and chunking.get("mode") == "chunked":
                # W4: a long document was windowed. Surface the chunk outcome so
                # a partial pass (some windows failed) or a capped pass (the
                # window budget stopped early) never reads as a clean pass.
                summary.bump("chunked_documents")
                summary.bump("chunk_windows", int(chunking.get("window_count") or 0))
                failed_windows = chunking.get("failed_windows") or []
                if failed_windows:
                    summary.bump("chunk_failed_windows", len(failed_windows))
                    entry["chunk_failed_windows"] = list(failed_windows)
                    entry["reason"] = f"{CHUNK_FAILED_PREFIX}{len(failed_windows)}"
                if chunking.get("truncated"):
                    summary.bump("chunk_budget_exhausted")
                    entry["reason"] = entry.get("reason") or CHUNK_BUDGET_EXHAUSTED
                entry["chunking"] = {
                    "window_count": chunking.get("window_count"),
                    "pages_covered": chunking.get("pages_covered"),
                    "pages_total": chunking.get("pages_total"),
                    "truncated": chunking.get("truncated"),
                    "failed_windows": failed_windows,
                    "deduped_claims": chunking.get("deduped_claims"),
                    "resumed": chunking.get("resumed"),
                }
            if verification is not None:
                counts = verification.counts()
                summary.bump("verified", counts["supported"])
                summary.bump("verification_rejected", counts["rejected"])
                summary.bump("unverified", counts["unverified"])
                summary.bump("verifier_calls", verification.calls)
                summary.bump("verifier_repairs", verification.repairs)
                self._verifier_cost += verification.estimated_cost_usd
                self._verifier_exhausted = (
                    self._verifier_exhausted or verification.budget_exhausted
                )
                self._verifier_error = self._verifier_error or verification.error
                entry["verified"] = counts["supported"]
                entry["verification_rejected"] = counts["rejected"]
                entry["unverified"] = counts["unverified"]
                entry["verifier_calls"] = verification.calls
            elif verifier is not None and result.claims:
                # A verifier that raised leaves every claim unverified; say so
                # rather than letting a missing count look like a clean pass.
                summary.bump("unverified", len(result.claims))
                self._verifier_error = self._verifier_error or "verification_failed"
                entry["unverified"] = len(result.claims)
            if rejected:
                entry["reasons"] = [_rejection_reason(item) for item in result.rejections]
            summary.outcomes.append(entry)

        if verifier is not None:
            summary.metadata["verifier"] = {
                "model": getattr(verifier, "model", None),
                "prompt_version": getattr(getattr(verifier, "prompt", None), "ref", None),
                "estimated_cost_usd": round(self._verifier_cost, 6),
                "budget_exhausted": self._verifier_exhausted,
                "error": self._verifier_error,
            }
        if self._counting_router is not None:
            summary.bump("llm_calls", self._counting_router.calls)
            self._counting_router = None

    # ------------------------------------------------------------------ provenance
    @staticmethod
    def _record_acquire(summary: RunSummary, result: Any) -> dict[str, Any] | None:
        """Fold one document's parse provenance into the run summary (W10).

        ``parsed`` counts every document that produced a usable parse this run
        (cache hit or live conversion); ``cache_hits`` counts the lookups the
        cache satisfied — zero converter invocations; ``pages_converted`` counts
        pages actually converted live this run. Cached pages were counted when
        their cycle converted them and never again.
        """
        info = getattr(result, "acquire", None)
        if not isinstance(info, dict):
            return None
        if info.get("reason") == "missing_parse_cache":
            return info
        if info.get("cache_hit"):
            summary.bump("cache_hits")
            summary.bump("parsed")
            return info
        if info.get("mode") == "live":
            summary.bump("parsed")
            pages = info.get("pages")
            if isinstance(pages, int) and pages > 0:
                summary.bump("pages_converted", pages)
        return info

    # ------------------------------------------------------------------ verification
    def _get_verifier(self) -> Any | None:
        """Build the second pass once per run, or return ``None`` if unavailable.

        Verification is optional by design: a missing profile or prompt must not
        fail extraction. Every claim simply stays ``unverified``, which is the
        honest state.
        """
        if not self.verify:
            return None
        if self._verifier_ready:
            return self.verifier
        self._verifier_ready = True
        if self.verifier is not None:
            return self.verifier
        try:
            router = self._counting_router if self._counting_router is not None else self.router
            if self.verifier_factory is not None:
                self.verifier = self.verifier_factory(router)
            else:
                models = getattr(router, "models", None)
                if models is None or "verify" not in models.profiles:
                    return None
                from fmd.verify import Verifier

                self.verifier = Verifier(
                    router,
                    claims=self.claims,
                    predicates=self.predicates,
                )
        except Exception as exc:  # config/prompt absence degrades, never crashes
            logger.warning("verification disabled for this run: %s", exc)
            self.verifier = None
        return self.verifier

    def _verify(self, result: Any, verifier: Any | None) -> Any | None:
        if verifier is None or not result.claims:
            return None
        text = result.doc.text if result.doc is not None else ""
        try:
            return verifier.verify_claims(result.claims, text=text)
        except Exception as exc:  # one bad page must not lose the extracted claims
            logger.warning("verification failed for extraction %s: %s", result.extraction_id, exc)
            return None

    # ------------------------------------------------------------------ failures
    def _mark_acquisition_failed(self, document: Any, reason: str) -> None:
        """Persist ``acquisition_failed:<reason>`` on the document row.

        ``plan_state`` stays inside migration 0003's closed set, so the reason
        lives in ``plan_reason`` and the state is ``needs_review``: the retry
        queue stops silently re-running a document that cannot be acquired on
        this host until a human acts.
        """
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
        except Exception as exc:  # observability must not mask the original failure
            logger.warning("could not persist acquisition failure on document: %s", exc)

    # ------------------------------------------------------------------ linkage
    def _target(self, snapshot: Any, document: Any) -> tuple[ExtractTarget | None, str | None]:
        if document is None:
            return None, "missing_document"
        return target_for_document(document, snapshot.id)

    def _count_unplanned(self, summary: RunSummary) -> None:
        """Surface documents extraction will skip, with an honest reason per bucket.

        ``not_applicable`` (inventory / navigation, migration 0004) is counted
        separately and never as an extraction failure: those documents were
        never candidates. Only the remaining buckets are extraction debt.
        """
        counter = getattr(self.documents, "count_unplanned", None)
        if counter is None:
            return
        try:
            reasons = dict(counter())
        except Exception as exc:  # observability must not fail a run
            logger.warning("could not count unplanned documents: %s", exc)
            return
        not_applicable = int(reasons.pop("not_applicable", 0) or 0)
        if not_applicable:
            summary.bump("not_applicable", not_applicable)
        total = sum(reasons.values())
        if total:
            summary.bump("unplanned", total)
            summary.metadata["unplanned"] = dict(sorted(reasons.items()))


def build_extract_router(
    registry: Any,
    *,
    mock: bool = False,
    fixtures_dir: str | None = None,
) -> Any:
    """Router factory used by the orchestrator (offline replay when ``mock``)."""
    return build_router(registry.models, mock=mock, fixtures_dir=fixtures_dir)


def summarise_rejections(items: Iterable[Rejection]) -> list[str]:
    return [_rejection_reason(item) for item in items]


def _rejection_reason(item: Any) -> str:
    reason = getattr(item, "reason", None)
    if reason:
        return str(reason)
    if hasattr(item, "as_dict"):
        payload = item.as_dict()
        return str(payload.get("reason") or payload.get("code") or "rejected")
    return "rejected"
