"""Stage 6 — resolution service (brief §10).

Orchestrates normalise → block → score → decide and returns one :class:`Resolution`
per mention with an :class:`AuditRecord` explaining the outcome. The service is
pure: the caller supplies the candidate universe (e.g.
``EntitiesRepo.list(kind="institution")``), the router, and the config. It never
writes to the database.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

from fmd.er.blocking import block
from fmd.er.config import ERConfig, Lexicon
from fmd.er.decide import (
    AdjudicationCache,
    AdjudicationOutcome,
    AdjudicationPair,
    adjudicate,
    band_for,
    pair_hash,
)
from fmd.er.normalise import Transliterator, normalise_name
from fmd.er.similarity import (
    EmbeddingScorer,
    candidate_sort_key,
    score_pairs,
)
from fmd.er.types import (
    AuditDecision,
    AuditRecord,
    BlockingResult,
    EntityCandidate,
    Mention,
    ModelVerdict,
    NormalisedName,
    Resolution,
    ResolutionBatch,
    ScoredCandidate,
)

LOGGER = logging.getLogger(__name__)


def _manual_match(mention: NormalisedName, candidate: EntityCandidate, lexicon: Lexicon) -> bool:
    """True only when the mention is exactly one of the entity's manual aliases."""
    for alias in candidate.manual_aliases:
        if normalise_name(alias.alias, lexicon).text == mention.text:
            return True
    return False


AuditSink = Callable[[AuditDecision], object]


def _manual_matches(
    mention: NormalisedName, scored: Sequence[ScoredCandidate], lexicon: Lexicon
) -> list[ScoredCandidate]:
    """Every blocked candidate this mention is an exact manual alias of."""
    return [item for item in scored if _manual_match(mention, item.candidate, lexicon)]


def _adjudication_pair(
    mention: Mention,
    mention_norm: NormalisedName,
    scored: ScoredCandidate,
    config: ERConfig,
) -> AdjudicationPair:
    candidate = scored.candidate
    pair_id = pair_hash(
        mention_norm.text,
        candidate.canonical_name,
        salt=config.version,
        state=candidate.state or mention.state,
        pincode=candidate.pincode or mention.pincode,
    )
    return AdjudicationPair(
        pair_id=pair_id,
        mention_name=mention.text,
        candidate_name=candidate.canonical_name,
        score=scored.score,
        mention_state=mention.state,
        mention_city=mention.city,
        mention_pincode=mention.pincode,
        candidate_state=candidate.state,
        candidate_city=candidate.city,
        candidate_pincode=candidate.pincode,
        features=scored.features.model_dump(),
    )


def resolve_with_audit(
    mentions: Sequence[Mention],
    candidates: Sequence[EntityCandidate],
    *,
    config: ERConfig | None = None,
    router: object | None = None,
    cache: AdjudicationCache | None = None,
    embedder: EmbeddingScorer | None = None,
    logger: logging.Logger | None = None,
    transliterate: Transliterator | None = None,
    audit_sink: AuditSink | None = None,
    claims_hash_by_entity: Mapping[int, str] | None = None,
) -> ResolutionBatch:
    """Resolve mentions against a candidate universe, keeping the audit trail.

    ``audit_sink`` receives a persistable :class:`AuditDecision` for every
    mention that carries an ``entity_id`` and has a candidate. Pairs without an
    entity id (a bare string mention) cannot be written to ``er_audit`` without
    inventing a subject, so they stay in memory with an explicit note.
    """
    cfg = config or ERConfig()
    log = logger or LOGGER
    lexicon = cfg.lexicon
    mention_norms = [
        normalise_name(mention.text, lexicon, transliterate=transliterate) for mention in mentions
    ]
    ordered_candidates = sorted(candidates, key=candidate_sort_key)
    blocking = block(mentions, ordered_candidates, config=cfg.blocking, lexicon=lexicon, logger=log)

    active_embedder = embedder
    if active_embedder is None and cfg.similarity.use_embeddings:
        active_embedder = EmbeddingScorer(model_name=cfg.similarity.embedding_model)

    scored_by_mention: dict[int, list[ScoredCandidate]] = {}
    pairs: list[AdjudicationPair] = []
    pair_owner: dict[str, int] = {}
    for index, mention in enumerate(mentions):
        indices = blocking.pairs.get(str(index), [])
        chosen_candidates = [ordered_candidates[position] for position in indices]
        scored = score_pairs(
            mention,
            mention_norms[index],
            chosen_candidates,
            config=cfg,
            embedder=active_embedder,
        )
        scored = [
            item.model_copy(update={"band": band_for(item.score, cfg.thresholds)})
            for item in scored
        ]
        for item in scored:
            if item.band == "adjudicate":
                pair = _adjudication_pair(mention, mention_norms[index], item, cfg)
                pairs.append(pair)
                pair_owner[pair.pair_id] = index
        scored_by_mention[index] = scored

    outcome = adjudicate(pairs, config=cfg, router=router, cache=cache, logger=log)

    resolutions: list[Resolution] = []
    for index, mention in enumerate(mentions):
        resolutions.append(
            _resolution_for(
                mention=mention,
                mention_norm=mention_norms[index],
                scored=scored_by_mention.get(index, []),
                blocking=blocking,
                outcome=outcome,
                config=cfg,
                pair_owner=pair_owner,
                mention_index=index,
                embedder=active_embedder,
            )
        )
    batch = ResolutionBatch(
        resolutions=resolutions,
        blocking=blocking,
        adjudication=outcome.verdicts,
    )
    if audit_sink is not None:
        for decision in audit_decisions(
            batch,
            thresholds_version=cfg.version,
            claims_hash_by_entity=claims_hash_by_entity,
        ):
            audit_sink(decision)
    return batch


def audit_decisions(
    batch: ResolutionBatch,
    *,
    thresholds_version: str,
    claims_hash_by_entity: Mapping[int, str] | None = None,
) -> list[AuditDecision]:
    """Turn resolutions into persistable ``er_audit`` rows.

    Only resolutions whose mention carries an ``entity_id`` and whose top
    candidate is a stored entity produce a row: the audit table references two
    entities, so anything else would need a fabricated id.
    """
    hashes = claims_hash_by_entity or {}
    decisions: list[AuditDecision] = []
    for resolution in batch.resolutions:
        subject = resolution.mention.entity_id
        top = resolution.candidates[0] if resolution.candidates else None
        if subject is None or top is None or top.candidate.entity_id is None:
            continue
        verdict = resolution.audit.model_verdict
        decided_by = {
            "score": "auto",
            "model": "model",
            "manual": "human",
            "none": "auto",
        }[resolution.audit.decided_by]
        decisions.append(
            AuditDecision(
                subject_entity_id=subject,
                candidate_entity_id=top.candidate.entity_id,
                band=resolution.audit.band,
                score=resolution.audit.score,
                features=resolution.audit.features or {"band": resolution.audit.band},
                thresholds_version=thresholds_version,
                model_verdict=verdict.verdict if verdict is not None else None,
                model_reason=verdict.reason if verdict is not None else None,
                decided_by=decided_by,
                claims_hash=hashes.get(subject),
            )
        )
    return decisions



def resolve(
    mentions: Sequence[Mention],
    candidates: Sequence[EntityCandidate],
    *,
    config: ERConfig | None = None,
    router: object | None = None,
    cache: AdjudicationCache | None = None,
    embedder: EmbeddingScorer | None = None,
    logger: logging.Logger | None = None,
    transliterate: Transliterator | None = None,
    audit_sink: AuditSink | None = None,
    claims_hash_by_entity: Mapping[int, str] | None = None,
) -> list[Resolution]:
    """Resolve mentions to entities; see :func:`resolve_with_audit` for the batch."""
    batch = resolve_with_audit(
        mentions,
        candidates,
        config=config,
        router=router,
        cache=cache,
        embedder=embedder,
        logger=logger,
        transliterate=transliterate,
        audit_sink=audit_sink,
        claims_hash_by_entity=claims_hash_by_entity,
    )
    return batch.resolutions


def _resolution_for(
    *,
    mention: Mention,
    mention_norm: NormalisedName,
    scored: list[ScoredCandidate],
    blocking: BlockingResult,
    outcome: AdjudicationOutcome,
    config: ERConfig,
    pair_owner: dict[str, int],
    mention_index: int,
    embedder: EmbeddingScorer | None,
) -> Resolution:
    cfg = config
    lexicon = cfg.lexicon
    notes: list[str] = list(mention_norm.notes)
    if not scored:
        notes.append("no_blocked_candidates")
    if blocking.used_fallback:
        notes.append("blocking_fallback")
    if embedder is not None and not embedder.available and embedder.reason:
        notes.append(f"embedding_unavailable: {embedder.reason}")

    top = scored[0] if scored else None
    band = top.band if top is not None else "adjudicate"
    top_score = top.score if top is not None else 0.0
    decided_by = "none"
    verdict: ModelVerdict | None = None
    chosen: EntityCandidate | None = None
    manual_matches = _manual_matches(mention_norm, scored, lexicon)

    if top is None:
        decided_by = "none"
        notes.append("no candidates available for this mention")
    elif len(manual_matches) == 1:
        # A human confirmation outranks any heuristic or model score, even when
        # a different candidate scored higher.
        top = manual_matches[0]
        band = "accept"
        top_score = top.score
        decided_by = "manual"
        chosen = top.candidate
        notes.append("manual_alias_authoritative")
    elif len(manual_matches) > 1:
        # Two human confirmations disagree: never guess.
        band = "adjudicate"
        decided_by = "none"
        chosen = None
        notes.append("manual_alias_conflict")
    elif band == "accept":
        decided_by = "score"
        chosen = top.candidate
        if top.candidate.manual_lock:
            band = "adjudicate"
            chosen = None
            decided_by = "none"
            notes.append("manual_canonical_lock")
    elif band == "reject":
        decided_by = "score"
    else:
        pair_id = pair_hash(
            mention_norm.text,
            top.candidate.canonical_name,
            salt=cfg.version,
            state=top.candidate.state or mention.state,
            pincode=top.candidate.pincode or mention.pincode,
        )
        verdict = outcome.verdicts.get(pair_id)
        if verdict is None:
            decided_by = "none"
            notes.append("awaiting_adjudication")
        elif verdict.verdict == "same":
            if top.candidate.manual_lock:
                decided_by = "none"
                notes.append("manual_canonical_lock: model accept declined")
            else:
                band = "accept"
                decided_by = "model"
                chosen = top.candidate
        elif verdict.verdict == "different":
            band = "reject"
            decided_by = "model"
        else:
            decided_by = "none"
            notes.append("model_unsure")

    mention_pairs = tuple(
        pair_id for pair_id, owner in pair_owner.items() if owner == mention_index
    )
    audit = AuditRecord(
        mention_text=mention.text,
        normalised=mention_norm.text,
        blocking_keys=tuple(blocking.keys.get(str(mention_index), [])),
        blocking_backend=blocking.backend,
        blocking_reason=blocking.reason,
        thresholds={"high": cfg.thresholds.high, "low": cfg.thresholds.low},
        config_version=cfg.version,
        band=band,
        score=round(top_score, 6),
        decided_by=decided_by,
        model_verdict=verdict,
        adjudication_pairs=mention_pairs,
        notes=tuple(notes),
        features=top.features.model_dump() if top is not None else {},
    )
    return Resolution(
        mention=mention,
        normalised=mention_norm,
        band=band,
        chosen=chosen,
        score=round(top_score, 6),
        candidates=scored,
        audit=audit,
    )


__all__ = ["AuditSink", "audit_decisions", "resolve", "resolve_with_audit"]
