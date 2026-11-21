"""Snapshot -> text -> guided JSON -> validation -> claims (brief §9).

The pipeline is deliberately thin: it never parses document prose itself and it
never computes anything. Every number it publishes has already passed
:mod:`fmd.extract.validate`, and the model call is audited through the router.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from fmd.acquire.canonical import outline_to_sections
from fmd.acquire.outline import OutlineNode, SectionIndex, outline_summary
from fmd.acquire.parse_cache import acquire_document
from fmd.acquire.types import PageSpan
from fmd.config import Registry, get_registry
from fmd.config.models import ModelsConfig, ProviderConfig
from fmd.db.repositories import (
    ClaimCreate,
    ClaimsRepo,
    ExtractionCreate,
    ExtractionsRepo,
    SectionNodeCreate,
    SectionsRepo,
    SectionTreeWrite,
    SnapshotRecord,
    SnapshotsRepo,
)
from fmd.extract.chunking import (
    ChunkedExtractError,
    ChunkingConfig,
    ChunkResume,
    ContextExceeded,
    TextWindow,
    chunk_failed_code,
    claim_keys_record,
    classify_failure,
    dedupe_claims,
    estimate_input_tokens,
    filter_known_claims,
    is_context_overflow,
    known_claim_keys,
    plan_windows,
    to_absolute_offsets,
)
from fmd.extract.pdf import (
    ExtractedDoc,
    TierPolicy,
    TierUnavailable,
    VLMAdjudicator,
)
from fmd.extract.prompts import load_prompt, user_message
from fmd.extract.schemas import FamilyDocument, UnknownFamily, schema_for
from fmd.extract.validate import Bands, FieldOutcome, Rejection, SpanRewrite, validate_document
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
    outcomes: list[FieldOutcome] = field(default_factory=list)
    adjudications: list[dict[str, Any]] = field(default_factory=list)
    skipped: bool = False
    needs_review: bool = False
    review: dict[str, Any] | None = None
    #: Chunk-path summary (``None`` on the single-pass path, which is unchanged).
    #: Shape documented in ``fmd/extract/chunking.py`` and runbook §9.
    chunking: dict[str, Any] | None = None
    #: Parse provenance (W10 parse cache): ``mode`` (``cache``|``live``|
    #: ``cache_only``), ``cache_hit``, ``pages``, and for misses the stable
    #: ``reason`` (``no_cache_entry``, ``missing_parse_cache``, ...) plus
    #: ``stored`` when the fresh parse was written to the cache.
    acquire: dict[str, Any] | None = None


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
        sections: SectionsRepo | Any | None = None,
        parse_cache: Any | None = None,
        t0: Any = None,
        t1: Any = None,
        t2: Any = None,
        t2b: Any = None,
        allow_convert: bool = True,
    ) -> None:
        self.router = router
        self.snapshots = snapshots
        self.extractions = extractions
        self.claims = claims
        self.store = store
        #: Parse cache (W10). ``None`` keeps the legacy fused behaviour: every
        #: run converts inline and nothing is cached.
        self.parse_cache = parse_cache
        #: Injected tier callables (test seams, passed straight through to the
        #: tier dispatcher).
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t2b = t2b
        #: ``False`` for ``extract --from-cache`` / ``--no-acquire``: a missing
        #: cached parse is reported (``missing_parse_cache``), never converted.
        self.allow_convert = allow_convert
        #: The section-tree repo (W8). ``None`` keeps the outline audit-only:
        #: it still rides in ``structure_json`` but is not persisted as rows.
        self.sections = sections
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
        already = self._already_extracted(snapshot.id, prompt.ref)
        resume = self._chunk_resume(snapshot.id, prompt.ref) if already else None
        if already and (resume is None or not resume.has_failures):
            logger.info(
                "skipping snapshot %s (%s): already extracted at tool_version %s",
                snapshot.id,
                prompt.ref,
                self.tool_version,
            )
            return ExtractResult(target=target, skipped=True)

        data = self.store.get(snapshot.sha256)
        doc, provenance = acquire_document(
            data,
            content_type=snapshot.content_type,
            policy=self.policy,
            cache=self.parse_cache,
            snapshot_sha256=snapshot.sha256,
            t0=self.t0,
            t1=self.t1,
            t2=self.t2,
            t2b=self.t2b,
            allow_convert=self.allow_convert,
        )
        acquire_info = _acquire_info(provenance)
        if doc is None:
            # Cache-only mode with no cached parse (extract --from-cache /
            # --no-acquire): record the stable reason and continue with the
            # next document — Docling is never invoked on this path.
            logger.info(
                "snapshot %s has no cached parse (mode=%s); recorded %s",
                snapshot.id,
                acquire_info.get("mode"),
                acquire_info.get("reason"),
            )
            return ExtractResult(target=target, skipped=True, acquire=acquire_info)
        if doc.is_empty:
            engine = doc.engine_meta or {}
            raise TierUnavailable(
                f"snapshot {snapshot.id}: every available tier produced empty text "
                f"(tier={doc.tier} tool={doc.tool} pages={len(doc.pages or doc.page_map)} "
                f"engine={engine.get('engine') or 'n/a'} "
                f"notes={'; '.join(doc.notes) or 'none'})"
            )
        if doc.ocr_anomaly or doc.needs_review:
            # A decoder loop or a failed script self-check is not evidence. Route
            # to the human queue and emit no claims from this document.
            reasons = ", ".join(doc.anomaly_reasons) or "tier_escalation"
            logger.warning("snapshot %s routed to T3 (%s); no claims emitted", snapshot.id, reasons)
            return ExtractResult(
                target=target,
                doc=doc,
                skipped=True,
                needs_review=True,
                review=doc.as_review_item(),
                acquire=acquire_info,
            )

        schema = schema_for(target.family)
        purpose = self.policy.profile_for(doc.tier)
        model = self._model_for(purpose)
        observed_at = self.observed_at or snapshot.fetched_at
        config = ChunkingConfig.from_models(
            self.router.models,
            (getattr(self.router.models, "profiles", None) or {}).get(purpose),
        )
        if not config.fits(doc.text):
            # W4: the canonical text cannot fit one call. Chunk it when enabled;
            # otherwise fail with the honest sizing code before wasting a call.
            if not config.enabled:
                estimated = estimate_input_tokens(
                    prompt.text, user_message(doc.text), chars_per_token=config.chars_per_token
                )
                raise ContextExceeded(
                    f"snapshot {snapshot.id}: canonical text is {len(doc.text)} chars "
                    f"(~{estimated} tokens with the system prompt), over the context budget "
                    f"of {config.context_tokens} tokens ({config.output_tokens} reserved for "
                    f"output, {config.prompt_headroom_tokens} prompt headroom); enable "
                    "chunking in config/models.yaml:chunking"
                )
            return self._run_chunked(
                target, snapshot, doc, prompt, schema, purpose, model, observed_at, config, resume,
                acquire_info=acquire_info,
            )

        try:
            section_bounds, section_source = _doc_section_bounds(doc)
            payload = self.router.call(
                purpose,
                schema,
                prompt.text,
                user_message(doc.text, document_sections=_document_sections_metadata(doc)),
                input_ref=f"{snapshot.sha256}:{target.family}:{prompt.version}",
            )
        except ContextExceeded:
            raise
        except Exception as exc:
            # The provider answered 400 "request (N tokens) exceeds the available
            # context size" — a sizing fault the estimate under-counted (config
            # budget above the served ceiling). Record it as context_exceeded,
            # never http_error; an operator must fix the budget, not retry.
            if is_context_overflow(str(exc)):
                raise ContextExceeded(
                    f"snapshot {snapshot.id}: provider reported the input did not fit the "
                    f"served context ({exc})"
                ) from exc
            raise
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
            tables=doc.tables,
            pages=doc.page_map or None,
            sections=section_bounds,
            section_source=section_source,
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
                    "outcomes": [item.as_dict() for item in validation.outcomes],
                    "outcome_counts": validation.outcome_counts(),
                    "adjudications": adjudications,
                    "sections": _outline_structure(doc),
                },
            )
        )
        self._persist_sections(snapshot, doc)

        stored: list[Any] = []
        for claim in validation.claims:
            adjusted = _apply_adjudications(claim, adjudications)
            adjusted = _attach_bbox_provenance(adjusted, doc)
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
            outcomes=validation.outcomes,
            adjudications=adjudications,
            acquire=acquire_info,
        )

    def run_batch(self, targets: list[ExtractTarget]) -> list[ExtractResult]:
        """Deterministic, resumable batch: same input order and same skips."""
        ordered = sorted(targets, key=lambda item: (item.snapshot_id, item.family))
        return [self.run(target) for target in ordered]

    # ------------------------------------------------------------------ chunked path

    def _run_chunked(
        self,
        target: ExtractTarget,
        snapshot: SnapshotRecord,
        doc: ExtractedDoc,
        prompt: Any,
        schema: type,
        purpose: str,
        model: str | None,
        observed_at: datetime | None,
        config: ChunkingConfig,
        resume: ChunkResume | None,
        *,
        acquire_info: dict[str, Any] | None = None,
    ) -> ExtractResult:
        """The W4 chunk loop: one call per window, one extraction row per run.

        Each window is a verbatim slice of the canonical text; validation
        resolves the model's quotes against the window exactly as in the
        single-pass path, then :func:`to_absolute_offsets` shifts every span to
        the window's absolute start, so stored ``char_start``/``char_end`` are
        global and bbox/page projection works unchanged.

        Failure isolation: a window that raises is recorded with
        ``chunk_failed:<n>`` (plus the classified provider reason) and the
        remaining windows still run. Partial success is a real extraction row:
        claims from the windows that resolved are persisted with per-window
        outcomes, and a rerun retries only the failed windows — windows the
        prior run completed are skipped from its structure and their claims are
        excluded from this run by the same dedupe keys, so overlap facts are
        never written twice.

        The window cap bounds one run's cost. When the cap stops the plan with
        text uncovered, the extraction records ``chunk_budget_exhausted`` with
        pages covered vs total; covering the tail needs a higher
        ``max_windows_per_document`` (and a rerun, which will skip the windows
        already done). When every window attempted in this pass fails, nothing
        is persisted and :class:`ChunkedExtractError` is raised so the run
        ledger records an abstention, not an empty success.
        """
        outline_index = (
            SectionIndex(doc.canonical.outline) if doc.canonical is not None else None
        )
        plan = plan_windows(
            doc.text,
            window_chars=config.window_chars,
            overlap_chars=config.overlap_chars,
            max_windows=config.max_windows_per_document,
            min_window_chars=config.min_window_chars,
            page_map=doc.page_map,
            section_starts=_outline_starts(doc),
        )
        logger.info(
            "snapshot %s: chunking %s chars into %s window(s) (~%s chars each, overlap %s)",
            snapshot.id,
            plan.total_chars,
            plan.window_count,
            plan.window_chars,
            plan.overlap_chars,
        )
        rejections: list[Rejection] = []
        rewrites: list[SpanRewrite] = []
        outcomes: list[FieldOutcome] = []
        structure_rejections: list[dict[str, Any]] = []
        structure_rewrites: list[dict[str, Any]] = []
        structure_outcomes: list[dict[str, Any]] = []
        adjudications: list[dict[str, Any]] = []
        window_rows: list[dict[str, Any]] = []
        collected: list[ClaimCreate] = []
        for window in plan.windows:
            prior = resume.completed_row(window) if resume is not None else None
            row = {
                "index": window.index,
                "char_start": window.char_start,
                "char_end": window.char_end,
                "core_start": window.core_start,
                "core_end": window.core_end,
                "pages": list(window.pages),
            }
            window_rows.append(row)
            if prior is not None:
                row.update(
                    status="completed_prior_run",
                    code=None,
                    reason=None,
                    claims=prior.get("claims", 0),
                    claims_key=prior.get("claims_key", []),
                    prior_extraction_id=resume.prior_extraction_id,
                )
                continue
            try:
                payload = self.router.call(
                    purpose,
                    schema,
                    prompt.text,
                    user_message(
                        window.text,
                        document_sections=_window_sections_metadata(outline_index, window),
                    ),
                    input_ref=(
                        f"{snapshot.sha256}:{target.family}:{prompt.version}:w{window.index}"
                    ),
                )
            except Exception as exc:
                reason = classify_failure(exc)
                code = chunk_failed_code(window.index)
                logger.warning(
                    "window %s/%s of snapshot %s failed (%s): %s",
                    window.index,
                    plan.window_count,
                    snapshot.id,
                    reason,
                    exc,
                )
                row.update(
                    status="failed",
                    code=code,
                    reason=reason,
                    detail=str(exc)[:500],
                    claims=0,
                    claims_key=[],
                )
                continue
            window_pages, window_sections = _window_scope_bounds(doc, window)
            validation = validate_document(
                payload,
                window.text,
                extraction_id=0,
                entity_id=target.entity_id,
                bands=self.bands,
                observed_at=observed_at,
                default_confidence=self.default_confidence,
                tables=doc.tables,
                pages=window_pages,
                sections=window_sections,
                section_source=_section_source(doc),
            )
            window_adjudications = self._adjudicate(validation.claims, window.text)
            adjudications.extend({**item, "window": window.index} for item in window_adjudications)
            window_claims: list[ClaimCreate] = []
            for claim in validation.claims:
                adjusted = _apply_adjudications(claim, window_adjudications)
                window_claims.append(to_absolute_offsets(adjusted, window.char_start))
            for item in validation.rejections:
                rejections.append(item)
                structure_rejections.append({**item.as_dict(), "window": window.index})
            for item in validation.rewrites:
                shifted = replace(
                    item,
                    after_start=item.after_start + window.char_start,
                    after_end=item.after_end + window.char_start,
                )
                rewrites.append(shifted)
                structure_rewrites.append({**shifted.as_dict(), "window": window.index})
            for item in validation.outcomes:
                absolute = replace(
                    item,
                    char_start=(
                        item.char_start + window.char_start
                        if item.char_start is not None
                        else None
                    ),
                    char_end=(
                        item.char_end + window.char_start if item.char_end is not None else None
                    ),
                )
                outcomes.append(absolute)
                structure_outcomes.append({**absolute.as_dict(), "window": window.index})
            row.update(
                status="ok",
                code=None,
                reason=None,
                claims=len(window_claims),
                rejections=len(validation.rejections),
                claims_key=[claim_keys_record(claim) for claim in window_claims],
            )
            collected.extend(window_claims)

        deduped, dropped_in_run = dedupe_claims(collected)
        if resume is not None:
            kept, dropped_prior = filter_known_claims(deduped, resume.known)
        else:
            kept, dropped_prior = deduped, 0

        failed_rows = [row for row in window_rows if row.get("status") == "failed"]
        attempted = [row for row in window_rows if row.get("status") in {"ok", "failed"}]
        if attempted and all(row["status"] == "failed" for row in attempted):
            codes = ", ".join(str(row["code"]) for row in failed_rows)
            raise ChunkedExtractError(
                len(failed_rows),
                f"snapshot {snapshot.id}: all {len(failed_rows)} window(s) failed ({codes}); "
                "no extraction recorded this pass",
            )

        chunk_summary = {
            "mode": "chunked",
            "config": {
                "context_tokens": config.context_tokens,
                "output_tokens": config.output_tokens,
                "prompt_headroom_tokens": config.prompt_headroom_tokens,
                "chars_per_token": config.chars_per_token,
                "window_chars": config.window_chars,
                "overlap_chars": config.overlap_chars,
                "min_window_chars": config.min_window_chars,
                "max_windows_per_document": config.max_windows_per_document,
            },
            "estimated_single_pass_tokens": estimate_input_tokens(
                prompt.text, user_message(doc.text), chars_per_token=config.chars_per_token
            ),
            "window_count": plan.window_count,
            "total_chars": plan.total_chars,
            "truncated": plan.truncated,
            "budget_exhausted": plan.truncated,
            "pages_covered": plan.pages_covered,
            "pages_total": plan.pages_total,
            "windows": window_rows,
            "failed_windows": [row["index"] for row in failed_rows],
            "deduped_claims": dropped_in_run + dropped_prior,
            "deduped_within_run": dropped_in_run,
            "deduped_against_prior": dropped_prior,
            "resumed": resume is not None,
            "prior_extraction_id": resume.prior_extraction_id if resume else None,
        }

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
                    "rejections": structure_rejections,
                    "rewrites": structure_rewrites,
                    "outcomes": structure_outcomes,
                    "outcome_counts": _outcome_counts(structure_outcomes),
                    "adjudications": adjudications,
                    "chunking": chunk_summary,
                    "sections": _outline_structure(doc),
                },
            )
        )
        self._persist_sections(snapshot, doc)

        stored: list[Any] = []
        for claim in kept:
            adjusted = _attach_bbox_provenance(claim, doc)
            stored.append(
                self.claims.create(adjusted.model_copy(update={"extraction_id": extraction.id}))
            )

        return ExtractResult(
            target=target,
            extraction_id=extraction.id,
            doc=doc,
            claims=stored,
            rejections=rejections,
            rewrites=rewrites,
            outcomes=outcomes,
            adjudications=adjudications,
            chunking=chunk_summary,
            acquire=acquire_info,
        )

    # ------------------------------------------------------------------ internals

    def _already_extracted(self, snapshot_id: int, prompt_ref: str) -> bool:
        for existing in self.extractions.list_for_snapshot(snapshot_id):
            if existing.tool_version == self.tool_version and existing.prompt_version == prompt_ref:
                return True
        return False

    def _chunk_resume(self, snapshot_id: int, prompt_ref: str) -> ChunkResume | None:
        """Resume state from the latest chunked extraction at this tool version.

        Returns ``None`` when the snapshot has no chunked extraction, so plain
        single-pass records keep the legacy skip behaviour untouched. Completed
        windows are keyed by their absolute span — never by index — so a config
        change between runs cannot mistake one window for another. Windows the
        prior run recorded as ``failed`` are what a rerun retries.
        """
        for record in reversed(self.extractions.list_for_snapshot(snapshot_id)):
            if record.tool_version != self.tool_version or record.prompt_version != prompt_ref:
                continue
            chunking = (record.structure or {}).get("chunking")
            if not isinstance(chunking, Mapping) or chunking.get("mode") != "chunked":
                return None
            completed: dict[tuple[int, int], dict[str, Any]] = {}
            keys: list[Mapping[str, Any]] = []
            has_failures = False
            for row in chunking.get("windows") or []:
                if not isinstance(row, Mapping):
                    continue
                status = row.get("status")
                start, end = row.get("char_start"), row.get("char_end")
                if status == "failed":
                    has_failures = True
                if (
                    status in {"ok", "completed_prior_run"}
                    and isinstance(start, int)
                    and isinstance(end, int)
                ):
                    completed[(start, end)] = dict(row)
                for item in row.get("claims_key") or []:
                    if isinstance(item, Mapping):
                        keys.append(item)
            return ChunkResume(
                prior_extraction_id=record.id,
                completed=completed,
                known=known_claim_keys(keys),
                has_failures=has_failures,
            )
        return None

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

    def _persist_sections(self, snapshot: SnapshotRecord, doc: ExtractedDoc) -> None:
        """Persist the outline tree for the snapshot's document (W8).

        Derived structure, written wholesale. Only a **non-empty** tree is
        written: a no-headings document must never clear a tree a prior
        snapshot produced, and an empty write would look like data loss in the
        review UI. ``self.sections is None`` keeps the outline audit-only (it
        still rides in ``structure_json``).
        """
        repo = self.sections
        canonical = doc.canonical
        if repo is None or canonical is None or not canonical.outline:
            return
        nodes = [
            SectionNodeCreate(
                node_key=node.node_key,
                level=node.level,
                title=node.title,
                char_start=node.char_start,
                char_end=node.char_end,
                heading_char_start=node.heading_char_start,
                heading_char_end=node.heading_char_end,
                page_no=node.page_no,
                bbox=list(node.bbox) if node.bbox is not None else None,
                parent_key=node.parent_key,
            )
            for node in canonical.outline
        ]
        repo.replace_for_document(
            SectionTreeWrite(
                document_id=snapshot.document_id,
                extractor=canonical.outline_extractor or "heuristic",
                extractor_version=f"{doc.tool}:{doc.tool_version or 'unknown'}",
                nodes=nodes,
            )
        )


def _doc_section_bounds(doc: ExtractedDoc) -> tuple[tuple[Any, ...] | None, str | None]:
    """The document's section bounds for the resolver, plus their extractor.

    W8 prefers the outline tree (``SectionGeometry`` bounds carrying level and
    a root→leaf breadcrumb); a document with no tree keeps the flat W7
    heuristic bounds, recorded as ``section_source: heuristic`` so a degraded
    run stays visible. Both are ``None`` when no heading was found.
    """
    canonical = doc.canonical
    if canonical is None:
        return None, None
    if canonical.outline:
        return outline_to_sections(canonical.outline), "outline"
    if canonical.sections:
        return canonical.sections, "heuristic"
    return None, None


def _section_source(doc: ExtractedDoc) -> str | None:
    """Which extractor produced the bounds used for the section rung."""
    _bounds, source = _doc_section_bounds(doc)
    return source


def _outline_structure(doc: ExtractedDoc) -> dict[str, Any] | None:
    """Audit summary of the outline in ``extractions.structure_json``."""
    canonical = doc.canonical
    if canonical is None:
        return None
    if not canonical.outline and not canonical.outline_diagnostic:
        return None
    return outline_summary(
        canonical.outline,
        extractor=canonical.outline_extractor or "heuristic",
        diagnostic=canonical.outline_diagnostic,
    )


def _outline_starts(doc: ExtractedDoc) -> tuple[int, ...]:
    """Heading offsets for chunk cuts at section boundaries."""
    canonical = doc.canonical
    if canonical is None or not canonical.outline:
        return ()
    return tuple(sorted({node.char_start for node in canonical.outline}))


def _render_outline_lines(nodes: Sequence[OutlineNode], limit: int = 200) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        if len(lines) >= limit:
            lines.append(f"... ({len(nodes) - limit} more headings omitted)")
            break
        indent = "  " * (node.level - 1)
        page = f" (page {node.page_no})" if node.page_no else ""
        lines.append(f"{indent}- {node.title}{page}")
    return lines


def _document_sections_metadata(doc: ExtractedDoc) -> str | None:
    """The outline as prompt metadata (never quotable document text)."""
    canonical = doc.canonical
    if canonical is None or not canonical.outline:
        return None
    lines = _render_outline_lines(canonical.outline)
    return "\n".join(lines)


def _window_sections_metadata(index: SectionIndex | None, window: TextWindow) -> str | None:
    """Window breadcrumb + the headings the window actually contains.

    The breadcrumb is the section chain in effect at the window's owned start;
    the listed headings are those fully inside the window, so the model sees
    where it is without the metadata ever being quotable text (validation
    resolves quotes against the window body only).
    """
    if index is None or not len(index):
        return None
    breadcrumb = index.breadcrumb(window.core_start)
    lines: list[str] = []
    if breadcrumb:
        lines.append(f"window breadcrumb: {' > '.join(breadcrumb)}")
    inner = index.inside(window.char_start, window.char_end)
    if not inner and not lines:
        return None
    lines.extend(_render_outline_lines(inner))
    return "\n".join(lines)


def _window_scope_bounds(
    doc: ExtractedDoc, window: TextWindow
) -> tuple[list[PageSpan] | None, list[Any] | None]:
    """Window-relative page/section bounds for the chunked validation path.

    Validation resolves against the window's verbatim slice, so every bound is
    clipped to the window and shifted by its start; the claims' spans are
    window-relative until ``to_absolute_offsets`` shifts them (the qualifier's
    ``scope_char_start``/``scope_char_end`` shift with it). A section counts
    only when its printed heading lies fully inside the window: a heading the
    model never saw cannot scope a quote, so a section that started in an
    earlier window is dropped, never clipped at the heading. W8 outline bounds
    (level + breadcrumb) are preferred; the flat W7 heuristic bounds are the
    fallback and keep their shape.
    """
    pages: list[PageSpan] = []
    for span in doc.page_map:
        start = max(span.char_start, window.char_start)
        end = min(span.char_end, window.char_end)
        if end > start:
            pages.append(
                PageSpan(
                    page=span.page,
                    char_start=start - window.char_start,
                    char_end=end - window.char_start,
                )
            )
    sections: list[Any] = []
    if doc.canonical is not None:
        bounds = (
            outline_to_sections(doc.canonical.outline)
            if doc.canonical.outline
            else doc.canonical.sections
        )
        for section in bounds:
            if (
                window.char_start <= section.heading_char_start
                and section.heading_char_end <= window.char_end
            ):
                sections.append(
                    replace(
                        section,
                        heading_char_start=section.heading_char_start - window.char_start,
                        heading_char_end=section.heading_char_end - window.char_start,
                        char_start=section.char_start - window.char_start,
                        char_end=min(section.char_end, window.char_end) - window.char_start,
                    )
                )
    return (pages or None), (sections or None)


def _attach_bbox_provenance(claim: ClaimCreate, doc: ExtractedDoc) -> ClaimCreate:
    """Attach resolved page bboxes to the claim qualifier.

    The quote already resolved against the canonical text in ``validate``; this
    projects the same offsets onto the source geometry. A multi-span quote keeps
    every box; it is never collapsed into one box that covers the gap.

    A row-scoped claim also carries the **row** span (label cell through value
    cell), so a reviewer can see that this number belongs to this row.
    """
    if doc.canonical is None:
        return _attach_engine_provenance(claim, doc)
    qualifier = dict(claim.qualifier or {})
    if claim.char_start is not None and claim.char_end is not None:
        hits = doc.canonical.bboxes(claim.char_start, claim.char_end)
        if hits:
            qualifier["bboxes"] = [hit.as_dict() for hit in hits]
            qualifier["bbox"] = list(hits[0].bbox)
            qualifier["page_no"] = hits[0].page_no
            granularities = {hit.granularity for hit in hits}
            qualifier["bbox_granularity"] = (
                hits[0].granularity if len(granularities) == 1 else doc.bbox_granularity
            )
            qualifier["bbox_multi_span"] = len(hits) > 1
    row_start = qualifier.get("row_char_start")
    row_end = qualifier.get("row_char_end")
    if isinstance(row_start, int) and isinstance(row_end, int) and row_end > row_start:
        row_hits = doc.canonical.bboxes(row_start, row_end)
        if row_hits:
            qualifier["row_bboxes"] = [hit.as_dict() for hit in row_hits]
            qualifier["row_bbox"] = list(row_hits[0].bbox)
            pages = {hit.page_no for hit in row_hits}
            if len(pages) == 1:
                qualifier["row_page_no"] = row_hits[0].page_no
    if doc.tables:
        qualifier["table_cells_uncertain"] = any(
            table.column_header_uncertain for table in doc.tables
        )
    return _attach_engine_provenance(claim.model_copy(update={"qualifier": qualifier}), doc)


def _attach_engine_provenance(claim: ClaimCreate, doc: ExtractedDoc) -> ClaimCreate:
    """Record the OCR engine that read the claim's page (W7, D18 visibility).

    Additive only, keyed on the acquire-owned ``ExtractedDoc.engine_meta``:
    born-digital tiers carry ``None`` and are untouched. An OCR claim records
    engine + engine_version + model_ids + device in its qualifier, and the
    page's measured flags (conjunct risk, dropped low-confidence boxes,
    normalised Devanagari digits) — a mixed-engine corpus stays auditable.
    """
    engine_meta = doc.engine_meta
    if not engine_meta:
        return claim
    qualifier = dict(claim.qualifier or {})
    qualifier["ocr_engine"] = engine_meta.get("engine")
    qualifier["ocr_engine_version"] = engine_meta.get("engine_version")
    qualifier["ocr_model_ids"] = engine_meta.get("model_ids")
    qualifier["ocr_device"] = engine_meta.get("device")
    qualifier["ocr_device_reason"] = engine_meta.get("device_reason")
    qualifier["ocr_granularity"] = engine_meta.get("granularity")
    page_no = qualifier.get("page_no")
    for row in engine_meta.get("pages") or []:
        if isinstance(row, dict) and row.get("page") == page_no:
            qualifier["conjunct_risk"] = row.get("conjunct_risk")
            qualifier["ocr_dropped_low_confidence"] = row.get("dropped_low_confidence")
            qualifier["devanagari_digits_normalized"] = row.get(
                "devanagari_digits_normalized"
            )
            break
    return claim.model_copy(update={"qualifier": qualifier})


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


def _acquire_info(provenance: Any) -> dict[str, Any]:
    """Normalised parse provenance for run summaries (W10 parse cache)."""
    info = dict(provenance.as_dict())
    reason = getattr(provenance, "reason", None)
    if reason is not None:
        info.setdefault("reason", reason)
    if info.get("reason") == "missing_parse_cache":
        info["mode"] = "cache_only"
    info["cache_hit"] = info.get("mode") == "cache"
    pages = getattr(provenance, "pages", None)
    if pages is None:
        pages = (getattr(provenance, "info", None) or {}).get("pages")
    if pages is not None:
        info["pages"] = int(pages)
    return info


def _page_range(doc: ExtractedDoc) -> str | None:
    if not doc.page_map:
        return None
    pages = [span.page for span in doc.page_map]
    return f"{min(pages)}-{max(pages)}"


def _outcome_counts(structure_outcomes: list[dict[str, Any]]) -> dict[str, int]:
    """Per-outcome counts over every window's merged per-field results."""
    counts: dict[str, int] = {}
    for outcome in structure_outcomes:
        key = str(outcome.get("outcome"))
        counts[key] = counts.get(key, 0) + 1
    return counts


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
        sections=SectionsRepo(database),
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
