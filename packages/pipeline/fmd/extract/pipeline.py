"""Snapshot -> text -> guided JSON -> validation -> claims (brief §9).

The pipeline is deliberately thin: it never parses document prose itself and it
never computes anything. Every number it publishes has already passed
:mod:`fmd.extract.validate`, and the model call is audited through the router.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fmd.config import Registry, get_registry
from fmd.config.models import ModelsConfig, ProviderConfig
from fmd.db.repositories import (
    ClaimCreate,
    ClaimsRepo,
    ExtractionCreate,
    ExtractionsRepo,
    SnapshotRecord,
    SnapshotsRepo,
)
from fmd.extract.pdf import (
    ExtractedDoc,
    TierPolicy,
    TierUnavailable,
    VLMAdjudicator,
    extract_document,
)
from fmd.extract.prompts import load_prompt, user_message
from fmd.extract.schemas import FamilyDocument, UnknownFamily, schema_for
from fmd.extract.validate import Bands, Rejection, SpanRewrite, validate_document
from fmd.llm import ModelRouter
from fmd.store import SnapshotStore

logger = logging.getLogger(__name__)


class ExtractError(RuntimeError):
    """The snapshot cannot be extracted in this configuration."""


@dataclass(frozen=True)
class ExtractTarget:
    snapshot_id: int
    entity_id: int
    family: str


def target_for_document(document: Any, snapshot_id: int) -> tuple[ExtractTarget | None, str | None]:
    """Build a target from a document's plan columns (``documents.*``, 0003).

    This is the only seam that knows how a document becomes an extraction
    target. Returns ``(target, None)`` or ``(None, reason_code)``; a reason code
    is a stable string (``missing_family``, ``unresolved_entity``,
    ``needs_review``, ``unsupported_family:<name>``) so abstentions are countable.
    """
    family = getattr(document, "extraction_family", None)
    if not family:
        return None, "missing_family"
    entity_id = getattr(document, "entity_id", None)
    if entity_id is None:
        state = getattr(document, "plan_state", None)
        return None, "needs_review" if state == "needs_review" else "unresolved_entity"
    try:
        schema_for(str(family))
    except UnknownFamily:
        return None, f"unsupported_family:{family}"
    return (
        ExtractTarget(snapshot_id=snapshot_id, entity_id=int(entity_id), family=str(family)),
        None,
    )


@dataclass
class ExtractResult:
    target: ExtractTarget
    extraction_id: int | None = None
    doc: ExtractedDoc | None = None
    claims: list[Any] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    rewrites: list[SpanRewrite] = field(default_factory=list)
    adjudications: list[dict[str, Any]] = field(default_factory=list)
    skipped: bool = False


class ExtractPipeline:
    def __init__(
        self,
        router: ModelRouter,
        *,
        snapshots: SnapshotsRepo,
        extractions: ExtractionsRepo,
        claims: ClaimsRepo,
        store: SnapshotStore,
        tool: str = "fmd.extract",
        tool_version: str = "1.0.0",
        policy: TierPolicy | None = None,
        bands: Bands | None = None,
        observed_at: datetime | None = None,
        default_confidence: float = 0.5,
        adjudicate: bool = False,
        adjudicator: VLMAdjudicator | None = None,
    ) -> None:
        self.router = router
        self.snapshots = snapshots
        self.extractions = extractions
        self.claims = claims
        self.store = store
        self.tool = tool
        self.tool_version = tool_version
        self.policy = policy or TierPolicy.load()
        self.bands = bands or Bands.load()
        self.observed_at = observed_at
        self.default_confidence = default_confidence
        self.adjudicator = adjudicator
        if adjudicate and adjudicator is None:
            self.adjudicator = VLMAdjudicator(router)

    # ------------------------------------------------------------------ public API

    def run(self, target: ExtractTarget) -> ExtractResult:
        snapshot = self.snapshots.get(target.snapshot_id)
        if snapshot is None:
            raise ExtractError(f"snapshot {target.snapshot_id} does not exist")
        prompt = load_prompt(target.family)
        if self._already_extracted(snapshot.id, prompt.ref):
            logger.info(
                "skipping snapshot %s (%s): already extracted at tool_version %s",
                snapshot.id,
                prompt.ref,
                self.tool_version,
            )
            return ExtractResult(target=target, skipped=True)

        data = self.store.get(snapshot.sha256)
        doc = extract_document(data, content_type=snapshot.content_type, policy=self.policy)
        if doc.is_empty:
            raise TierUnavailable(
                f"snapshot {snapshot.id}: every available tier produced empty text"
            )

        schema = schema_for(target.family)
        purpose = self.policy.profile_for(doc.tier)
        model = self._model_for(purpose)
        observed_at = self.observed_at or snapshot.fetched_at

        payload = self.router.call(
            purpose,
            schema,
            prompt.text,
            user_message(doc.text),
            input_ref=f"{snapshot.sha256}:{target.family}:{prompt.version}",
        )
        if not isinstance(payload, FamilyDocument):
            raise ExtractError(
                f"router returned {type(payload).__name__}, expected a "
                f"{schema.__name__} family document"
            )

        validation = validate_document(
            payload,
            doc.text,
            extraction_id=0,
            entity_id=target.entity_id,
            bands=self.bands,
            observed_at=observed_at,
            default_confidence=self.default_confidence,
        )
        adjudications = self._adjudicate(validation.claims, doc.text)

        extraction = self.extractions.create(
            ExtractionCreate(
                snapshot_id=snapshot.id,
                tool=self.tool,
                tool_version=self.tool_version,
                model=model,
                prompt_version=prompt.ref,
                page_range=_page_range(doc),
                text=doc.text,
                structure={
                    **doc.to_structure(),
                    "family": target.family,
                    "prompt_sha256": prompt.sha256,
                    "rejections": [item.as_dict() for item in validation.rejections],
                    "rewrites": [item.as_dict() for item in validation.rewrites],
                    "adjudications": adjudications,
                },
            )
        )

        stored: list[Any] = []
        for claim in validation.claims:
            adjusted = _apply_adjudications(claim, adjudications)
            stored.append(
                self.claims.create(adjusted.model_copy(update={"extraction_id": extraction.id}))
            )

        return ExtractResult(
            target=target,
            extraction_id=extraction.id,
            doc=doc,
            claims=stored,
            rejections=validation.rejections,
            rewrites=validation.rewrites,
            adjudications=adjudications,
        )

    def run_batch(self, targets: list[ExtractTarget]) -> list[ExtractResult]:
        """Deterministic, resumable batch: same input order and same skips."""
        ordered = sorted(targets, key=lambda item: (item.snapshot_id, item.family))
        return [self.run(target) for target in ordered]

    # ------------------------------------------------------------------ internals

    def _already_extracted(self, snapshot_id: int, prompt_ref: str) -> bool:
        for existing in self.extractions.list_for_snapshot(snapshot_id):
            if existing.tool_version == self.tool_version and existing.prompt_version == prompt_ref:
                return True
        return False

    def _model_for(self, purpose: str) -> str | None:
        profile = self.router.models.profiles.get(purpose)
        return profile.model if profile else None

    def _adjudicate(self, claims: list[ClaimCreate], text: str) -> list[dict[str, Any]]:
        if self.adjudicator is None or not claims:
            return []
        items = [
            {
                "key": f"{claim.predicate}:{claim.char_start}:{claim.char_end}",
                "field": claim.predicate,
                "value": claim.value,
                "quote": claim.quote,
            }
            for claim in claims
        ]
        decisions = self.adjudicator.adjudicate(items, text=text)
        return [
            {"field": item.get("field"), "verdict": verdict.model_dump()}
            for item, verdict in decisions
        ]


def _apply_adjudications(claim: ClaimCreate, adjudications: list[dict[str, Any]]) -> ClaimCreate:
    key = f"{claim.predicate}:{claim.char_start}:{claim.char_end}"
    for decision in adjudications:
        verdict = decision.get("verdict") or {}
        if decision.get("field") != claim.predicate:
            continue
        corrected_value = verdict.get("corrected_value") or {}
        corrected_quote = (verdict.get("corrected_quote") or {}).get("value")
        if corrected_value.get("value") is not None:
            if corrected_quote and corrected_quote != claim.quote:
                continue  # a new quote needs re-validation; never splice silently
            return claim.model_copy(update={"value": corrected_value["value"]})
        supported = verdict.get("supported") or {}
        if supported.get("value") is False:
            logger.warning("T3 adjudication rejected claim %s; halving confidence", key)
            return claim.model_copy(update={"confidence": claim.confidence * 0.5})
    return claim


def _page_range(doc: ExtractedDoc) -> str | None:
    if not doc.page_map:
        return None
    pages = [span.page for span in doc.page_map]
    return f"{min(pages)}-{max(pages)}"


# --------------------------------------------------------------------------- router


def build_router(
    models: ModelsConfig | None = None,
    *,
    mock: bool = False,
    fixtures_dir: str | Path | None = None,
) -> ModelRouter:
    """Router factory. ``mock=True`` is the offline/CI path (recorded fixtures)."""
    if models is None:
        models = get_registry().models
    if not mock:
        return ModelRouter(models)
    return ModelRouter(mock_models(models, fixtures_dir=fixtures_dir))


def mock_models(models: ModelsConfig, *, fixtures_dir: str | Path | None = None) -> ModelsConfig:
    """Re-point every profile at the replay provider, keeping prompt semantics."""
    directory = fixtures_dir
    if directory is None:
        provider = models.providers.get("mock")
        directory = provider.fixtures_dir if provider else None
    if not directory:
        raise ExtractError(
            "no replay fixtures directory configured: pass fixtures_dir or declare "
            "providers.mock.fixtures_dir in config/models.yaml"
        )
    payload = models.model_dump()
    payload["providers"]["mock"] = ProviderConfig(
        kind="replay", fixtures_dir=str(directory)
    ).model_dump()
    for profile in payload["profiles"].values():
        profile["provider"] = "mock"
        if not profile.get("model"):
            profile["model"] = "mock-model"
    return ModelsConfig.model_validate(payload)


def run_extract(
    targets: list[ExtractTarget],
    *,
    database: Any,
    store: SnapshotStore,
    registry: Registry | None = None,
    router: ModelRouter | None = None,
    mock: bool = False,
    fixtures_dir: str | Path | None = None,
    **pipeline_kwargs: Any,
) -> list[ExtractResult]:
    """Convenience entrypoint for ``make extract`` / the future CLI command."""
    registry = registry or get_registry()
    router = router or build_router(registry.models, mock=mock, fixtures_dir=fixtures_dir)
    pipeline = ExtractPipeline(
        router,
        snapshots=SnapshotsRepo(database),
        extractions=ExtractionsRepo(database),
        claims=ClaimsRepo(database),
        store=store,
        **pipeline_kwargs,
    )
    return pipeline.run_batch(targets)


def snapshot_for_test(snapshot_id: int, entity_id: int, family: str) -> ExtractTarget:
    """Small helper kept for scripts/tests that build targets from config."""
    return ExtractTarget(snapshot_id=snapshot_id, entity_id=entity_id, family=family)


def describe_snapshot(snapshot: SnapshotRecord) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "document_id": snapshot.document_id,
        "sha256": snapshot.sha256,
        "content_type": snapshot.content_type,
        "fetched_at": snapshot.fetched_at.isoformat() if snapshot.fetched_at else None,
    }
