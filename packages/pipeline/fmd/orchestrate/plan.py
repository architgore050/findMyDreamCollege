"""Phase B planning: acquire a bounded text sample, then re-plan (brief §9).

Phase A (crawl time) only sees URL/anchor/extension. When those signals are
ambiguous it writes ``defer_for_text`` instead of tossing the document. This
stage is the bridge: it finds evidence snapshots whose plan is deferred, reads a
bounded text sample from the already-stored bytes, and lets
:class:`fmd.plan.DocumentPlanner` classify the family and link the entity from
the document's own text.

Design constraints:

* **One acquisition per document per run.** Samples are cached by snapshot
  sha256, and a snapshot whose plan is no longer deferred is skipped. A re-run
  after an interruption resumes; it does not re-read a document already decided.
* **No model calls.** Acquisition is the deterministic PDF/HTML tier ladder;
  family detection is lexical; entity linking is :mod:`fmd.er` restricted to the
  registry candidate universe.
* **Honest cost.** Every acquired page is counted, including pages spent on a
  document that ultimately abstains (``pages_wasted``).

The stage never decides a family itself. It reads bytes, calls the planner with
``phase="b"``, and persists the resulting plan through the same seam as crawl.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from fmd.plan import DocumentPlan, persist_plan
from fmd.plan.planner import DEFER_CODE
from fmd.plan.roles import ROLE_EVIDENCE

logger = logging.getLogger(__name__)


def _pages(doc: Any) -> int:
    pages = getattr(doc, "page_map", None) or getattr(doc, "pages", None) or ()
    return max(1, len(pages))


def _cheap_text(
    data: bytes,
    *,
    content_type: str | None = None,
    policy: Any | None = None,
) -> Any:
    """Read only the born-digital text layer (T0) for a Phase B sample.

    Planning must not pay for OCR. A scanned document with no text layer simply
    abstains here; extraction can still escalate it later if the URL was
    confident enough to plan without text. This is the cost guard made explicit.
    """
    from fmd.acquire.tiers import t0_extract

    if (content_type or "").lower().find("pdf") >= 0 or data.lstrip()[:5] == b"%PDF-":
        return t0_extract(data, policy=policy)
    from fmd.extract.pdf import extract_document

    return extract_document(data, content_type=content_type, policy=policy)


class PlanStage:
    """Resolve Phase A deferrals with a bounded sample of acquired text."""

    def __init__(
        self,
        *,
        planner: Any,
        documents: Any,
        snapshots: Any,
        store: Any,
        sources: Iterable[Any] | None = None,
        acquire: Callable[..., Any] | None = None,
        policy: Any | None = None,
        text_sample_chars: int = 4000,
    ) -> None:
        self.planner = planner
        self.documents = documents
        self.snapshots = snapshots
        self.store = store
        self.sources_by_id = {str(getattr(source, "id", "")): source for source in sources or ()}
        self.policy = policy
        self.text_sample_chars = int(text_sample_chars)
        # The resume scan is wider than the per-run planning cap: deferred rows
        # may sit behind already-extracted snapshots in id order.
        self.scan_limit = 5000
        if acquire is None:
            acquire = _cheap_text
        self.acquire = acquire
        # sha256 -> (sample, pages). An interrupted run must not re-buy text.
        self._cache: dict[str, tuple[str, int]] = {}
        self._done: set[int] = set()

    # ------------------------------------------------------------------- selection
    def _is_deferred(self, document: Any) -> bool:
        return (
            (getattr(document, "role", None) or ROLE_EVIDENCE) == ROLE_EVIDENCE
            and getattr(document, "plan_state", None) == "planned"
            and getattr(document, "plan_reason", None) == DEFER_CODE
        )

    def pending(self, limit: int | None = None) -> list[tuple[Any, Any]]:
        """Snapshots whose document still waits for text, in stable order."""
        fetch = getattr(self.snapshots, "list_without_extraction", None)
        if fetch is None:  # a minimal fake: no queue support, nothing to resume
            return []
        rows = list(fetch(None, self.scan_limit))
        pending: list[tuple[Any, Any]] = []
        seen_documents: set[int] = set()
        for snapshot in rows:
            document_id = getattr(snapshot, "document_id", None)
            if document_id in seen_documents or document_id in self._done:
                continue
            document = self.documents.get(document_id)
            if document is None or not self._is_deferred(document):
                continue
            seen_documents.add(document_id)
            pending.append((document, snapshot))
            if limit is not None and len(pending) >= limit:
                break
        return pending

    # ------------------------------------------------------------------------ run
    def run(self, *, summary: Any, limit: int | None = None) -> int:
        """Resolve deferred plans; return how many documents were re-planned."""
        pending = self.pending(limit=limit)
        metrics: dict[str, Any] = {
            "deferred": len(pending),
            "acquired": 0,
            "resolved_family": 0,
            "resolved_entity": 0,
            "abstained": 0,
            "acquire_failed": 0,
            "pages_acquired": 0,
            "pages_wasted": 0,
            "family_counts": {},
            "abstain_reasons": {},
        }
        if pending:
            summary.bump("plan.deferred", len(pending))

        for document, snapshot in pending:
            sha = str(getattr(snapshot, "sha256", ""))
            cached = self._cache.get(sha)
            try:
                if cached is None:
                    data = self.store.get(sha)
                    doc = self.acquire(
                        data,
                        content_type=getattr(snapshot, "content_type", None),
                        policy=self.policy,
                    )
                    sample = str(getattr(doc, "text", "") or "")[: self.text_sample_chars]
                    pages = _pages(doc)
                    self._cache[sha] = (sample, pages)
                    cached = self._cache[sha]
                    metrics["acquired"] += 1
                    metrics["pages_acquired"] += pages
            except Exception as exc:  # unimplemented tier / missing bytes: retry later
                metrics["acquire_failed"] += 1
                logger.warning("phase B could not acquire snapshot %s: %s", sha[:12], exc)
                summary.bump("plan.phase_b.acquire_failed")
                continue

            sample, pages = cached
            plan = self.planner.plan(
                document,
                source=self.sources_by_id.get(str(getattr(document, "source_id", "") or "")),
                family_hint=getattr(document, "extraction_family", None),
                entity_id_hint=getattr(document, "entity_id", None),
                text_sample=sample,
                anchor_text=None,
                role=getattr(document, "role", None) or ROLE_EVIDENCE,
                phase="b",
            )
            updated = persist_plan(self.documents, document, plan)
            resolved = updated if updated is not None else document
            # Keep the loop's view current: a second snapshot of the same
            # document must see the decision, not the stale row.
            self._done.add(int(getattr(document, "id", 0) or 0))
            self._record(plan, metrics)
            summary.bump(f"plan.{plan.state}")
            if plan.family:
                summary.bump("plan.phase_b.resolved_family")
                summary.bump(f"plan.family.{plan.family}")
            if plan.entity_id is not None:
                summary.bump("plan.phase_b.resolved_entity")
            if not plan.extractable:
                summary.bump("plan.phase_b.still_abstained")
                if plan.abstain_reason:
                    summary.bump(f"plan.reason.{plan.abstain_reason}")
                metrics["pages_wasted"] += pages
            if resolved is not document:
                document = resolved

        summary.bump("plan.phase_b.pages_acquired", metrics["pages_acquired"])
        summary.bump("plan.phase_b.pages_wasted", metrics["pages_wasted"])
        if metrics["deferred"] or metrics["acquired"]:
            summary.metadata["phase_b"] = metrics
        return metrics["deferred"]

    @staticmethod
    def _record(plan: DocumentPlan, metrics: dict[str, Any]) -> None:
        if plan.family:
            counts = metrics["family_counts"]
            counts[plan.family] = counts.get(plan.family, 0) + 1
            metrics["resolved_family"] += 1
        if plan.entity_id is not None:
            metrics["resolved_entity"] += 1
        if not plan.extractable:
            metrics["abstained"] += 1
            reason = plan.abstain_reason or plan.defer_reason or "unknown"
            reasons = metrics["abstain_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1


__all__ = ["PlanStage"]
