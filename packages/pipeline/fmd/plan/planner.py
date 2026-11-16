"""Document planner: family + entity for one fetched document (brief §6, §9, §10).

The planner is a pure-ish decision stage over an existing document row. It never
fetches, never writes (except through :func:`persist_plan`) and never invents an
id: an entity link is either produced by :func:`fmd.er.resolve` in its accept
band (or by an authoritative manual alias) or left NULL for review.

Registry first: if the source entry declares ``extraction_family`` (under
``params``), that family is used as-is. Heuristics only run for sources that do
not declare a family; their signals are the data in :mod:`fmd.plan.rules`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fmd.discover.base import params_from_source
from fmd.er.config import ERConfig
from fmd.er.service import resolve
from fmd.er.types import Alias, EntityCandidate, Mention, Resolution
from fmd.extract.schemas import FAMILIES
from fmd.plan.mentions import collect_mentions
from fmd.plan.rules import DEFAULT_RULES, PlanRules, detect_family, normalise

PlanState = Literal["planned", "linked", "needs_review", "abstained"]
FamilySource = Literal["registry", "reference", "heuristic", "none"]

#: Reasons are stable codes, never prose: the run ledger and review queue filter them.
ABSTAIN_CODES = (
    "family_undetected",
    "ambiguous_family",
    "entity_undetected",
    "entity_unresolved",
    "entity_ambiguous",
)


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EntityCandidatePlan(_PlanModel):
    """One candidate the ER service considered, kept for the review queue."""

    entity_id: int | None = None
    canonical_name: str
    score: float = 0.0
    band: str = "adjudicate"
    decided_by: str = "none"


class DocumentPlan(_PlanModel):
    """What extraction is allowed to do with a document, and why."""

    family: str | None = None
    family_source: FamilySource = "none"
    entity_id: int | None = None
    entity_candidates: list[EntityCandidatePlan] = Field(default_factory=list)
    confidence: float = 0.0
    abstain_reason: str | None = None
    state: PlanState = "abstained"
    signals: dict[str, Any] = Field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return self.state == "needs_review"

    @property
    def extractable(self) -> bool:
        return self.family is not None and self.entity_id is not None and self.state == "linked"

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def abstention_reason(plan: DocumentPlan) -> str | None:
    return plan.abstain_reason


def declared_family(source: Any) -> str | None:
    """The family a registry entry declares, or ``None``.

    Accepts a pydantic ``Source``, a seeded ``SourceRecord`` (params nested under
    ``config``) or a raw mapping, via :func:`params_from_source`.
    """
    if source is None:
        return None
    params = params_from_source(source)
    value = params.get("extraction_family")
    if value is None:
        value = getattr(source, "extraction_family", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class DocumentPlanner:
    def __init__(
        self,
        *,
        entities: Any | None = None,
        aliases: Any | None = None,
        er_config: ERConfig | None = None,
        rules: PlanRules = DEFAULT_RULES,
        router: Any | None = None,
        candidate_limit: int = 500,
        max_mentions: int = 8,
        use_embeddings: bool = False,
        candidates_provider: Callable[[], Sequence[EntityCandidate]] | None = None,
    ) -> None:
        self.entities = entities
        self.aliases = aliases
        self.rules = rules
        self.router = router
        self.candidate_limit = candidate_limit
        self.max_mentions = max_mentions
        self.candidates_provider = candidates_provider
        base = er_config or ERConfig()
        # Planning runs inside crawl/extract: it must be reproducible offline, so
        # embeddings are opt-in even when the registry enables them for ER merges.
        if not use_embeddings and base.similarity.use_embeddings:
            base = base.model_copy(
                update={"similarity": base.similarity.model_copy(update={"use_embeddings": False})}
            )
        self.er_config = base

    # ------------------------------------------------------------------ planning
    def plan(
        self,
        document: Any,
        *,
        source: Any | None = None,
        family_hint: str | None = None,
        entity_id_hint: int | None = None,
        text_sample: str | None = None,
        anchor_text: str | None = None,
    ) -> DocumentPlan:
        url = str(getattr(document, "url", "") or "")
        family, family_source, family_conf, signals, family_reason = self._family(
            url=url,
            source=source,
            family_hint=family_hint,
            text_sample=text_sample,
            anchor_text=anchor_text,
        )

        candidates = self._candidate_universe()
        candidate_plans: list[EntityCandidatePlan] = []
        entity_id: int | None = None
        entity_score = 0.0
        entity_reason: str | None = None

        if entity_id_hint is not None:
            entity_id = int(entity_id_hint)
            entity_score = 1.0
            signals["entity_source"] = "reference"
        else:
            entity_id, entity_score, candidate_plans, entity_reason = self._resolve_entity(
                document, candidates, text_sample=text_sample, anchor_text=anchor_text
            )

        reason = family_reason or entity_reason
        if family is None:
            state: PlanState = (
                "needs_review" if family_reason == "ambiguous_family" else "abstained"
            )
            confidence = 0.0
        elif entity_id is not None:
            state = "linked"
            confidence = round(min(family_conf, entity_score), 6)
        elif entity_reason == "entity_ambiguous":
            state = "needs_review"
            confidence = round(family_conf * 0.5, 6)
        else:
            state = "abstained"
            confidence = round(family_conf * 0.5, 6)

        return DocumentPlan(
            family=family,
            family_source=family_source,
            entity_id=entity_id,
            entity_candidates=candidate_plans,
            confidence=confidence,
            abstain_reason=reason,
            state=state,
            signals=signals,
        )

    # ------------------------------------------------------------------ internals
    def _family(
        self,
        *,
        url: str,
        source: Any | None,
        family_hint: str | None,
        text_sample: str | None,
        anchor_text: str | None,
    ) -> tuple[str | None, FamilySource, float, dict[str, Any], str | None]:
        declared = declared_family(source)
        if declared is None and family_hint:
            declared = str(family_hint).strip() or None
            family_source: FamilySource = "reference"
        else:
            family_source = "registry" if declared else "none"

        if declared:
            if declared not in FAMILIES:
                # A registry/ref typo must fail loudly: extracting against an
                # unknown schema would either crash or fabricate a family.
                return None, family_source, 0.0, {}, f"unsupported_family:{declared}"
            return declared, family_source, 1.0, {"declared": declared}, None

        family, score, signals, reason = detect_family(
            url=url,
            anchor_text=anchor_text,
            text_sample=text_sample,
            rules=self.rules,
        )
        return family, "heuristic" if family else "none", score, signals, reason

    def _candidate_universe(self) -> list[EntityCandidate]:
        if self.candidates_provider is not None:
            return list(self.candidates_provider())
        if self.entities is None or not hasattr(self.entities, "list"):
            return []
        try:
            rows = self.entities.list(kind="institution", limit=self.candidate_limit)
        except TypeError:  # a minimal repo without the keyword-only shape
            rows = self.entities.list("institution", self.candidate_limit)
        candidates: list[EntityCandidate] = []
        for row in rows:
            if getattr(row, "merged_into_entity_id", None) is not None:
                continue
            candidates.append(
                EntityCandidate(
                    entity_id=getattr(row, "id", None),
                    canonical_name=row.canonical_name,
                    state=getattr(row, "state", None),
                    city=getattr(row, "city", None),
                    aliases=self._aliases(getattr(row, "id", None)),
                )
            )
        return candidates

    def _aliases(self, entity_id: int | None) -> list[Alias]:
        if entity_id is None or self.aliases is None:
            return []
        aliases: list[Alias] = []
        for row in self.aliases.list_for_entity(entity_id):
            aliases.append(
                Alias(
                    alias=row.alias,
                    alias_type=row.alias_type,
                    confidence=row.confidence if row.confidence is not None else 1.0,
                    source_id=getattr(row, "source_id", None),
                )
            )
        return aliases

    def _resolve_entity(
        self,
        document: Any,
        candidates: Sequence[EntityCandidate],
        *,
        text_sample: str | None,
        anchor_text: str | None,
    ) -> tuple[int | None, float, list[EntityCandidatePlan], str | None]:
        mentions = collect_mentions(
            document,
            text_sample=text_sample,
            anchor_text=anchor_text,
            lexicon=self.er_config.lexicon,
            max_mentions=self.max_mentions,
            normalise=lambda value: normalise(value),
        )
        if not mentions:
            return None, 0.0, [], "entity_undetected"
        if not candidates:
            return None, 0.0, [], "entity_unresolved"

        resolutions = resolve(
            [Mention(text=name, doc_id=_doc_id(document)) for name in mentions],
            candidates,
            config=self.er_config,
            router=self.router,
        )
        return _summarise(resolutions)


def _doc_id(document: Any) -> str | None:
    value = getattr(document, "id", None)
    return None if value is None else str(value)


def _summarise(
    resolutions: Sequence[Resolution],
) -> tuple[int | None, float, list[EntityCandidatePlan], str | None]:
    """Decide the link from ER output without ever overriding a manual decision."""
    seen: dict[tuple[int | None, str], EntityCandidatePlan] = {}
    accepted: dict[int, float] = {}
    manual: dict[int, float] = {}
    any_adjudicate = False

    for resolution in resolutions:
        for scored in resolution.candidates:
            candidate = scored.candidate
            key = (candidate.entity_id, candidate.canonical_name)
            plan = EntityCandidatePlan(
                entity_id=candidate.entity_id,
                canonical_name=candidate.canonical_name,
                score=scored.score,
                band=scored.band,
                decided_by=resolution.audit.decided_by,
            )
            if key not in seen or plan.score > seen[key].score:
                seen[key] = plan
            if scored.band == "adjudicate":
                any_adjudicate = True
            if scored.band == "accept" and candidate.entity_id is not None:
                accepted[candidate.entity_id] = max(
                    accepted.get(candidate.entity_id, 0.0), scored.score
                )
        chosen = resolution.chosen
        if chosen is not None and chosen.entity_id is not None:
            accepted[chosen.entity_id] = max(accepted.get(chosen.entity_id, 0.0), resolution.score)
            if resolution.audit.decided_by == "manual":
                manual[chosen.entity_id] = max(manual.get(chosen.entity_id, 0.0), resolution.score)

    candidates = sorted(
        seen.values(), key=lambda item: (-item.score, item.canonical_name, item.entity_id or 0)
    )

    # A human confirmation outranks any score (brief §10), even when a second
    # candidate also lands in the accept band.
    if len(manual) == 1:
        entity_id, score = next(iter(manual.items()))
        return entity_id, score, candidates, None
    if len(manual) > 1:
        return None, 0.0, candidates, "entity_ambiguous"
    if len(accepted) == 1:
        entity_id, score = next(iter(accepted.items()))
        return entity_id, score, candidates, None
    if len(accepted) > 1:
        # Two candidates above the auto-accept band: never pick between them.
        return None, 0.0, candidates, "entity_ambiguous"
    if any_adjudicate:
        # The 0.40–0.85 band needs a model or a human; without one it is review.
        return None, 0.0, candidates, "entity_ambiguous"
    return None, 0.0, candidates, "entity_unresolved"


def persist_plan(documents: Any, document: Any, plan: DocumentPlan) -> Any | None:
    """Write a plan onto ``documents``; returns the updated row when supported."""
    setter = getattr(documents, "set_plan", None)
    if setter is None:
        return None
    document_id = getattr(document, "id", None)
    if document_id is None:
        return None
    return setter(
        document_id,
        source_id=getattr(document, "source_id", None),
        extraction_family=plan.family,
        entity_id=plan.entity_id,
        plan_state=plan.state,
        plan_confidence=plan.confidence,
        plan_reason=plan.abstain_reason,
    )
