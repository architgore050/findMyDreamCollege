"""Extract orchestration: pick pending snapshots, run the extract pipeline.

The pipeline itself (WS-D) owns PDF tiering, guided JSON and span validation.
This module owns the queue: which snapshot to process, which entity and family
it belongs to, and what to do when those are not resolvable. A document that
cannot be extracted is recorded as an abstention with a reason; it is never
written as a silent empty success.

Resumability: selection is ``snapshots.list_without_extraction(tool_version)``
and the pipeline skips a snapshot already extracted at the same
``tool_version + prompt_version``, so re-running after an interruption resumes
instead of duplicating evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from fmd.db.repositories import ClaimsRepo, ExtractionsRepo, SnapshotsRepo
from fmd.extract.pipeline import ExtractPipeline, ExtractTarget, build_router
from fmd.extract.schemas import UnknownFamily, schema_for
from fmd.extract.validate import Rejection
from fmd.orchestrate.crawl import plan_from_headers
from fmd.orchestrate.ledger import RunSummary

logger = logging.getLogger(__name__)

DEFAULT_TOOL_VERSION = "1.0.0"

# Source taxonomy -> extraction family where the mapping is unambiguous. Anything
# else must declare ``family`` in DocumentRef.metadata; guessing a family would
# produce plausible-but-wrong claims, which the brief forbids (§9).
FAMILY_BY_SOURCE_KIND: dict[str, str] = {
    "fees": "fee_structure",
    "admissions": "seat_matrix",
}


class ForcedExtractPipeline(ExtractPipeline):
    """``extract --force`` re-runs a snapshot even at the same tool version.

    Only the skip predicate changes; validation, spans, cost and audit are the
    pipeline's own code, never duplicated here.
    """

    def _already_extracted(self, snapshot_id: int, prompt_ref: str) -> bool:
        return False


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

    def _pipeline(self, *, force: bool) -> ExtractPipeline:
        cls = ForcedExtractPipeline if force else ExtractPipeline
        if self.pipeline_factory is not None:
            return self.pipeline_factory(cls)
        return cls(
            self.router,
            snapshots=self.snapshots,
            extractions=self.extractions,
            claims=self.claims,
            store=self.store,
            tool_version=self.tool_version,
        )

    def run(
        self,
        *,
        summary: RunSummary,
        limit: int | None = None,
        force: bool = False,
    ) -> None:
        pending = self.snapshots.list_without_extraction(
            None if force else self.tool_version,
            limit if limit is not None else 500,
        )
        pipeline = self._pipeline(force=force)
        for snapshot in pending:
            document = self.documents.get(snapshot.document_id)
            plan = plan_from_headers(getattr(document, "request_headers", None))
            target, reason = self._target(snapshot, document, plan)
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
            summary.bump("extracted")
            summary.bump("claims", len(result.claims))
            summary.bump("rejected_claims", rejected)
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
            if rejected:
                entry["reasons"] = [_rejection_reason(item) for item in result.rejections]
            summary.outcomes.append(entry)

    # ------------------------------------------------------------------ linkage
    def _target(
        self, snapshot: Any, document: Any, plan: dict[str, Any]
    ) -> tuple[ExtractTarget | None, str | None]:
        family = plan.get("family") or getattr(document, "kind", None)
        if not family:
            family = FAMILY_BY_SOURCE_KIND.get(str(plan.get("source_kind") or ""))
        if not family:
            return None, "missing_family"
        try:
            schema_for(str(family))
        except UnknownFamily:
            return None, f"unsupported_family:{family}"

        entity_id = plan.get("entity_id")
        if entity_id is None:
            institution = plan.get("institution")
            if institution and self.entities is not None:
                found = self.entities.find("institution", str(institution))
                entity_id = getattr(found, "id", None)
        if entity_id is None:
            return None, "unresolved_entity"
        return (
            ExtractTarget(snapshot_id=snapshot.id, entity_id=int(entity_id), family=str(family)),
            None,
        )


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
