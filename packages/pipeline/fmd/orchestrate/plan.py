"""Planning stages over stored documents (brief §9, §10).

Two stages live here:

* :class:`PlanStage` — the extract-time bridge. It resolves the Phase A
  deferrals crawl wrote (``plan_state='planned'``, ``plan_reason='defer_for_text'``)
  with a bounded sample of the already-stored bytes.
* :class:`PlanningPass` — the standalone planning pass behind ``fmd plan``. It
  plans every document that *should* be extractable but is not: documents that
  arrived through snapshot recovery, backfills or re-crawls and were never
  planned (``plan_state IS NULL``), deferrals still awaiting text, and — with
  ``--force`` — decided documents whose plan predates the current planner
  version. Without it, a harvest run after a recovery reports dozens of
  ``unplanned`` snapshots and produces zero claims, because extraction only
  reads planned documents (:func:`fmd.db.repositories.SnapshotsRepo.list_extractable`).

Both stages share the same rules:

* **One planning brain.** Neither stage classifies anything itself; both call
  :class:`fmd.plan.DocumentPlanner` — the same functions crawl calls — with the
  fixed precedence registry-declared family > Phase B text evidence > Phase A
  URL evidence.
* **T0 only, no OCR, no model.** A text sample is read from the stored
  snapshot's born-digital text layer. A scanned document is not escalated here;
  acquisition (T1/T2) belongs to the existing pipeline.
* **Never plan from nothing.** An evidence document with no snapshot and no
  stored reference hints is reported as ``awaiting_text``
  (``plan_state='planned'``, ``plan_reason='defer_for_text'``) — never linked
  from its URL alone. Non-evidence roles stay ``not_applicable``.
* **Human decisions are sacrosanct.** Reasons written by review actions are
  never replanned, at any planner version.
* **Resumable and idempotent.** Candidates are decided oldest-first; a
  decision identical to the stored row is not written again.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from fmd.plan import DocumentPlan, persist_plan
from fmd.plan.planner import DEFER_CODE
from fmd.plan.roles import ROLE_EVIDENCE, is_extractable_role
from fmd.plan.versioning import (
    is_current_version,
    is_replannable_reason,
    planner_version,
    stamp_reason,
)

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


#: The planner version a pass stamps when its caller supplies none. Production
#: runs get ``config/plan.yaml → version`` via :func:`build_planning_pass`; the
#: fallback keeps direct constructions (tests, scripts) internally consistent.
PLANNER_VERSION = planner_version(None)


class PlanningPass:
    """Plan documents that should be extractable but are not (the 84-unplanned fix).

    A harvest run after snapshot recovery reported
    ``{"unplanned": 84, "claims": 0}``: the recovery drain fetched and stored
    snapshots but was built without a planner, so every recovered document kept
    ``plan_state IS NULL`` and extraction never selected it. This pass is the
    durable answer: it walks the unplanned queue, decides each document with the
    same :class:`fmd.plan.DocumentPlanner` crawl uses, and persists the plan.

    Per document (oldest first, resumable via ``--limit``):

    * a non-evidence role (inventory / navigation) is ``not_applicable``;
    * evidence with a stored snapshot is read at T0 only (born-digital text, no
      OCR, no model) and planned at Phase B — registry family first, then the
      document's own text, with Phase A URL evidence as the weaker fallback;
    * evidence with no snapshot, or a snapshot with no T0 text layer, is
      ``awaiting_text`` — persisted as ``planned`` / ``defer_for_text`` so the
      existing acquisition pipeline picks it up — unless reference hints
      (discovery family / inventory entity) already exist, in which case the
      planner re-affirms them at Phase A exactly as crawl would;
    * a document whose T0 read fails is counted ``acquisition_failed`` and left
      untouched for the next run;
    * with ``--force`` the queue widens to decided documents that are stale at
      the current planner version (an unstamped or older-versioned plan reason).
      Decisions written by human review are never replanned.

    Counters land in the run summary and, through the ledger, in
    ``pipeline_runs``. ``pipeline_runs.stage`` is a closed set without a
    ``plan`` value (migrations are out of scope), so the row is recorded under
    ``stage='crawl'`` — where planning already lives — with all counters
    plan-shaped and ``metadata['stage_detail']='planning_pass'``.
    """

    #: Upper bound on the candidate scan when no explicit limit is given; a run
    #: is resumable, so a bounded scan cannot strand documents permanently.
    SCAN_LIMIT = 5000

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
        version: str | None = None,
    ) -> None:
        self.planner = planner
        self.documents = documents
        self.snapshots = snapshots
        self.store = store
        self.sources_by_id = {str(getattr(source, "id", "")): source for source in sources or ()}
        self.policy = policy
        self.text_sample_chars = int(text_sample_chars)
        self.version = version or PLANNER_VERSION
        if acquire is None:
            acquire = _cheap_text
        self.acquire = acquire
        # sha256 -> sample: one bounded read per snapshot per run.
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------ candidates
    def _candidate_ids(self, *, force: bool, limit: int) -> list[Any]:
        fetch = getattr(self.documents, "list_plan_candidates", None)
        if fetch is None:  # a minimal repo without queue support: nothing to plan
            return []
        return list(fetch(limit=limit, force=force, planner_version=self.version))

    # ------------------------------------------------------------------------ text
    def _sample_text(self, snapshot: Any) -> str:
        sha = str(getattr(snapshot, "sha256", ""))
        cached = self._cache.get(sha)
        if cached is None:
            data = self.store.get(sha)
            doc = self.acquire(
                data,
                content_type=getattr(snapshot, "content_type", None),
                policy=self.policy,
            )
            cached = str(getattr(doc, "text", "") or "")[: self.text_sample_chars]
            self._cache[sha] = cached
        return cached

    # --------------------------------------------------------------------- outcome
    def _decide(self, document: Any) -> tuple[DocumentPlan | None, str | None]:
        """One document's decision, or the reason no planner call was made.

        Returns ``(plan, None)`` normally; ``(None, 'awaiting_text')`` when the
        document must wait for text (never planned from nothing); ``(None,
        None)`` after an acquisition failure (counter only, row untouched).
        """
        role = getattr(document, "role", None) or ROLE_EVIDENCE
        source = self.sources_by_id.get(str(getattr(document, "source_id", "") or ""))
        if not is_extractable_role(role):
            return (
                self.planner.plan(
                    document,
                    source=source,
                    role=role,
                    phase="a",
                ),
                None,
            )

        snapshot = None
        fetch_latest = getattr(self.snapshots, "latest_for_document", None)
        if fetch_latest is not None:
            snapshot = fetch_latest(int(document.id))
        text: str | None = None
        if snapshot is not None:
            try:
                text = self._sample_text(snapshot)
            except Exception as exc:  # missing bytes / unimplemented tier: retry later
                logger.warning(
                    "planning pass could not read snapshot for document %s: %s",
                    getattr(document, "id", None),
                    exc,
                )
                return None, None

        family_hint = getattr(document, "extraction_family", None)
        entity_hint = getattr(document, "entity_id", None)
        if text is None or not text.strip():
            # No stored bytes, or bytes with no born-digital text layer. The
            # pass must not decide a family from the URL alone: without
            # reference hints the honest answer is "awaiting text".
            if family_hint is None and entity_hint is None:
                return None, "awaiting_text"
            return (
                self.planner.plan(
                    document,
                    source=source,
                    family_hint=family_hint,
                    entity_id_hint=entity_hint,
                    text_sample=None,
                    role=role,
                    phase="a",
                ),
                None,
            )
        return (
            self.planner.plan(
                document,
                source=source,
                family_hint=family_hint,
                entity_id_hint=entity_hint,
                text_sample=text,
                role=role,
                phase="b",
            ),
            None,
        )

    def _persist(self, document: Any, plan: DocumentPlan | None, *, dry_run: bool) -> str:
        """Write the decision when it differs from the stored row.

        A deferred plan always keeps the exact ``defer_for_text`` code (the
        phase-B resolvers match on it); every other decision carries the
        planner-version stamp. Returns ``"written"`` or ``"unchanged"``.
        """
        role = getattr(document, "role", None) or ROLE_EVIDENCE
        if plan is None:
            # awaiting_text: plan_state must stay inside migration 0003's set,
            # so the wait is persisted exactly like a Phase A deferral. Stored
            # reference hints (family / entity) are preserved untouched.
            target = {
                "family": getattr(document, "extraction_family", None),
                "entity": getattr(document, "entity_id", None),
                "state": "planned",
                "confidence": getattr(document, "plan_confidence", None),
                "reason": DEFER_CODE,
                "role": role,
            }
        else:
            reason = (
                DEFER_CODE if plan.deferred else stamp_reason(plan.persisted_reason, self.version)
            )
            target = {
                "family": plan.family,
                "entity": plan.entity_id,
                "state": plan.persisted_state,
                "confidence": plan.confidence,
                "reason": reason,
                "role": role,
            }
        stored = {
            "family": getattr(document, "extraction_family", None),
            "entity": getattr(document, "entity_id", None),
            "state": getattr(document, "plan_state", None),
            "confidence": getattr(document, "plan_confidence", None),
            "reason": getattr(document, "plan_reason", None),
            "role": getattr(document, "role", None),
        }
        if stored == target:
            return "unchanged"
        if dry_run:
            return "written"
        # One explicit UPDATE through the same seam persist_plan uses
        # (``documents.set_plan``); the reason carries the planner stamp.
        self.documents.set_plan(
            int(document.id),
            source_id=getattr(document, "source_id", None),
            extraction_family=target["family"],
            entity_id=target["entity"],
            plan_state=target["state"],
            plan_confidence=target["confidence"],
            plan_reason=target["reason"],
            role=role,
        )
        return "written"

    # ------------------------------------------------------------------------ run
    def run(
        self,
        *,
        summary: Any,
        limit: int | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Decide the unplanned queue; return the run's counter dict."""
        scan = self.SCAN_LIMIT if limit is None else max(1, int(limit))
        candidates = self._candidate_ids(force=force, limit=scan)
        metrics: dict[str, Any] = {
            "candidates": len(candidates),
            "planned": 0,
            "linked": 0,
            "needs_review": 0,
            "still_abstained": 0,
            "skipped_not_applicable": 0,
            "awaiting_text": 0,
            "acquisition_failed": 0,
            "skipped_current": 0,
            "unchanged": 0,
            "reasons": {},
        }
        summary.bump("plan.candidates", len(candidates))

        for document in candidates:
            if (
                force
                and getattr(document, "plan_state", None) not in (None, "planned")
                and not is_replannable_reason(getattr(document, "plan_reason", None))
            ):
                # A decision this pass does not recognise (human-review prose)
                # is never replanned, whatever --force says.
                metrics["skipped_current"] += 1
                summary.bump("plan.skipped_current")
                continue
            if (
                force
                and getattr(document, "plan_state", None) not in (None, "planned")
                and is_current_version(getattr(document, "plan_reason", None), self.version)
            ):
                # Already decided at this planner version: stale versions only.
                metrics["skipped_current"] += 1
                summary.bump("plan.skipped_current")
                continue

            plan, marker = self._decide(document)
            if plan is None and marker is None:
                # Same per-document isolation as the extract-time stage: one
                # unreadable snapshot is counted and warned, never a failed run.
                metrics["acquisition_failed"] += 1
                summary.bump("acquisition_failed")
                continue

            state = plan.state if plan is not None else "awaiting_text"
            if plan is not None and plan.state == "defer_for_text":
                state = "awaiting_text"
            outcome = self._persist(document, plan, dry_run=dry_run)
            metrics["planned"] += 1
            summary.bump("planned")
            if state == "awaiting_text":
                metrics["awaiting_text"] += 1
                summary.bump("awaiting_text")
            elif state == "linked":
                metrics["linked"] += 1
                summary.bump("linked")
            elif state == "needs_review":
                metrics["needs_review"] += 1
                summary.bump("needs_review")
            elif state == "not_applicable":
                metrics["skipped_not_applicable"] += 1
                summary.bump("skipped_not_applicable")
            else:
                metrics["still_abstained"] += 1
                summary.bump("still_abstained")
            reason = getattr(plan, "abstain_reason", None) if plan is not None else None
            if reason:
                # One breakdown for both buckets: an ambiguous entity lands in
                # needs_review, an undetected family in still_abstained, but the
                # reason taxonomy is the same stable code set.
                metrics["reasons"][reason] = metrics["reasons"].get(reason, 0) + 1
                summary.bump(f"plan.reason.{reason}")
            if outcome == "unchanged":
                metrics["unchanged"] += 1
                summary.bump("unchanged")

        summary.metadata["stage_detail"] = "planning_pass"
        summary.metadata["planner_version"] = self.version
        summary.metadata["force"] = force
        summary.metadata["reasons"] = dict(sorted(metrics["reasons"].items()))
        return metrics


def build_planning_pass(
    registry: Any,
    database: Any,
    *,
    store: Any | None = None,
    planner: Any | None = None,
    ledger: Any | None = None,
    sources: Iterable[Any] | None = None,
) -> tuple[PlanningPass, Any]:
    """Wire the production planning pass from a registry and a database.

    Mirrors :class:`fmd.orchestrate.Orchestrator`'s crawl wiring: registry
    sources drive ``declared_family``, the ER candidate universe comes from the
    entities/aliases repositories, and the snapshot store serves T0 reads.
    Returns ``(pass, ledger)`` so the caller owns run recording.
    """
    from fmd.db.repositories import AliasesRepo, DocumentsRepo, EntitiesRepo, SnapshotsRepo
    from fmd.plan import DocumentPlanner
    from fmd.store import SnapshotStore

    if store is None:
        from fmd.paths import resolve_snapshot_dir

        store = SnapshotStore(resolve_snapshot_dir())
    entities = EntitiesRepo(database) if database is not None else None
    aliases = AliasesRepo(database) if database is not None else None
    planner = planner or DocumentPlanner(
        entities=entities,
        aliases=aliases,
        er_config=getattr(registry, "er", None),
        plan_config=getattr(registry, "plan", None),
    )
    selected = list(sources) if sources is not None else list(registry.enabled_sources())
    version = planner_version(getattr(registry, "plan", None))
    phase_b = getattr(registry.plan, "phase_b", None)
    stage = PlanningPass(
        planner=planner,
        documents=DocumentsRepo(database),
        snapshots=SnapshotsRepo(database),
        store=store,
        sources=selected,
        text_sample_chars=getattr(phase_b, "text_sample_chars", 4000),
        version=version,
    )
    return stage, ledger


def run_planning_pass(
    registry: Any,
    database: Any,
    *,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    run_id: str | None = None,
    store: Any | None = None,
    planner: Any | None = None,
    ledger: Any | None = None,
) -> Any:
    """One ``fmd plan`` run end to end: decide, record, return the summary.

    ``pipeline_runs.stage`` is a closed set (crawl/extract/score/worker/ingest)
    and this workstream must not migrate, so the run is recorded under
    ``stage='crawl'`` — planning is the crawl stage's responsibility, and its
    counters are already ``plan.*``-shaped there. The summary keeps
    ``metadata['stage_detail']='planning_pass'`` and a ``plan-`` prefixed run id
    so the ledger row is identifiable in SQL.
    """
    import uuid

    from fmd.orchestrate.ledger import RunSummary, utcnow

    stage, ledger = build_planning_pass(
        registry, database, store=store, planner=planner, ledger=ledger
    )
    summary = RunSummary(
        run_id=run_id or f"plan-{uuid.uuid4().hex}",
        stage="crawl",
        started_at=utcnow(),
        dry_run=dry_run,
    )
    stage.run(summary=summary, limit=limit, force=force, dry_run=dry_run)
    summary.finish()
    if ledger is None:
        from fmd.orchestrate.ledger import SqlLedger

        ledger = SqlLedger(database) if database is not None else None
    if ledger is not None:
        # A dry run is recorded with status='dry_run' (SqlLedger) the same way
        # crawl records its dry runs: knowing it happened is observability.
        try:
            ledger.record(summary)
        except Exception as exc:  # observability must not fail a successful run
            logger.warning("could not persist planning run %s: %s", summary.run_id, exc)
    return summary


__all__ = ["PlanStage", "PlanningPass", "build_planning_pass", "run_planning_pass"]
