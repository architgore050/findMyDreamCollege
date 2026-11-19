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
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fmd.discover.base import params_from_source
from fmd.er.config import ERConfig
from fmd.er.service import resolve
from fmd.er.types import Alias, EntityCandidate, Mention, Resolution
from fmd.extract.schemas import FAMILIES
from fmd.plan.mentions import collect_mentions
from fmd.plan.roles import (
    ROLE_EVIDENCE,
    family_hints_from_structure,
    is_extractable_role,
    role_for,
)
from fmd.plan.rules import DEFAULT_RULES, PlanRules, detect_family, normalise, score_families

PlanState = Literal[
    "planned", "linked", "needs_review", "abstained", "not_applicable", "defer_for_text"
]
FamilySource = Literal["registry", "reference", "heuristic", "none"]
Phase = Literal["a", "b"]

#: Migration 0003's ``documents.plan_state`` is a closed set that predates the
#: two-phase planner (``defer_for_text``). Migrations are out of this
#: workstream's scope, so the precise state is carried in ``plan_reason`` and the
#: row keeps the DB-safe value. Phase B overwrites both when text arrives.
PERSISTED_PLAN_STATES = {"defer_for_text": "planned"}

#: Reasons are stable codes, never prose: the run ledger and review queue filter them.
ABSTAIN_CODES = (
    "family_undetected",
    "ambiguous_family",
    "entity_undetected",
    "entity_unresolved",
    "entity_ambiguous",
)

DEFER_CODE = "defer_for_text"


@dataclass(frozen=True)
class PhaseThresholds:
    """The configured cutoffs for one planning run (values from ``plan.yaml``)."""

    phase_a_min_score: float
    phase_a_margin: float
    acquire_floor: float
    phase_b_min_score: float
    phase_b_margin: float
    text_sample_chars: int
    max_headings: int
    #: Recurrence count before a recurring token counts (rules.py W5).
    recurring_min: int
    #: Table rows needed before a prospectus resolves to fee/seat (rules.py W5).
    table_min_rows: int

    @classmethod
    def from_config(cls, config: Any | None, rules: PlanRules) -> PhaseThresholds:
        weights = rules.weights
        if config is None:
            return cls(
                phase_a_min_score=weights.min_score,
                phase_a_margin=weights.margin,
                acquire_floor=weights.min_score / 3,
                phase_b_min_score=weights.min_score,
                phase_b_margin=weights.margin,
                text_sample_chars=4000,
                max_headings=8,
                recurring_min=weights.recurring_min,
                table_min_rows=weights.table_min_rows,
            )
        phase_a = config.phase_a
        phase_b = config.phase_b
        return cls(
            phase_a_min_score=float(phase_a.min_score),
            phase_a_margin=float(phase_a.margin),
            acquire_floor=float(phase_a.acquire_floor),
            phase_b_min_score=float(phase_b.min_score),
            phase_b_margin=float(phase_b.margin),
            text_sample_chars=int(phase_b.text_sample_chars),
            max_headings=int(phase_b.max_headings),
            # New W5 keys are optional in plan.yaml (the typed config allows
            # extras); a missing key falls back to the rule weights.
            recurring_min=int(getattr(phase_b, "recurring_min", 0) or weights.recurring_min),
            table_min_rows=int(getattr(phase_b, "table_min_rows", 0) or weights.table_min_rows),
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
    #: Why a Phase A document needs text before it can be accepted. ``None`` on
    #: every non-deferred state.
    defer_reason: str | None = None
    state: PlanState = "abstained"
    #: Migration 0004: a non-evidence role is ``not_applicable``, not a failure.
    role: str = ROLE_EVIDENCE
    signals: dict[str, Any] = Field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return self.state == "needs_review"

    @property
    def deferred(self) -> bool:
        return self.state == "defer_for_text"

    @property
    def extractable(self) -> bool:
        return self.family is not None and self.entity_id is not None and self.state == "linked"

    @property
    def persisted_state(self) -> str:
        """The value safe for ``documents.plan_state`` (migration 0003)."""
        return PERSISTED_PLAN_STATES.get(self.state, self.state)

    @property
    def persisted_reason(self) -> str | None:
        if self.deferred:
            return DEFER_CODE
        return self.abstain_reason or self.defer_reason

    @property
    def not_applicable(self) -> bool:
        return self.state == "not_applicable"

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
        plan_config: Any | None = None,
    ) -> None:
        self.entities = entities
        self.aliases = aliases
        self.rules = rules
        self.plan_config = plan_config
        self.thresholds = PhaseThresholds.from_config(plan_config, rules)
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
        role: str | None = None,
        phase: Phase | None = None,
    ) -> DocumentPlan:
        url = str(getattr(document, "url", "") or "")
        resolved_phase: Phase = phase or ("b" if text_sample is not None else "a")
        resolved_role = role or role_for(
            url=url,
            source=source,
            family=family_hint or declared_family(source),
        )
        if not is_extractable_role(resolved_role):
            # Inventory and navigation were never extraction candidates. This is
            # an answer, not an abstention: it must never be bucketed as
            # ``family_undetected``.
            return DocumentPlan(
                role=resolved_role,
                state="not_applicable",
                confidence=0.0,
                signals={"role": resolved_role},
            )

        sample = None
        if resolved_phase == "b":
            sample = (text_sample or "")[: self.thresholds.text_sample_chars]

        family, family_source, family_conf, signals, family_reason = self._family(
            url=url,
            source=source,
            family_hint=family_hint,
            text_sample=sample,
            anchor_text=anchor_text,
            phase=resolved_phase,
            content_type=getattr(document, "content_type", None),
        )
        signals["phase"] = resolved_phase

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
                document,
                candidates,
                text_sample=sample if resolved_phase == "b" else None,
                anchor_text=anchor_text,
                max_mentions=self.thresholds.max_headings
                if resolved_phase == "b"
                else self.max_mentions,
            )

        best_score = float(signals.get("best_score", family_conf))
        if resolved_phase == "a":
            if family is None:
                # A plausible-but-unclear URL is exactly what text acquisition
                # is for. Below the floor it is noise: abstain and spend nothing.
                if best_score > 0.0 and best_score >= self.thresholds.acquire_floor:
                    state: PlanState = "defer_for_text"
                    defer_reason = family_reason or "family_undetected"
                    abstain_reason = defer_reason
                    confidence = round(best_score, 6)
                else:
                    state = (
                        "needs_review" if family_reason == "ambiguous_family" else "abstained"
                    )
                    defer_reason = None
                    abstain_reason = family_reason
                    confidence = round(best_score, 6)
            elif entity_id is not None:
                state = "linked"
                defer_reason = None
                abstain_reason = None
                confidence = round(min(family_conf, entity_score), 6)
            else:
                # Family known from registry/URL, entity only readable in text.
                state = "defer_for_text"
                defer_reason = entity_reason or "entity_undetected"
                abstain_reason = defer_reason
                confidence = round(family_conf, 6)
                if family_source not in ("registry", "reference"):
                    # A heuristic family must not be replayed as if discovery had
                    # declared it: Phase B text is allowed to outrank it.
                    signals["tentative_family"] = family
                    family = None
                    family_source = "none"
        else:
            if family is None:
                state = (
                    "needs_review" if family_reason == "ambiguous_family" else "abstained"
                )
                defer_reason = None
                abstain_reason = family_reason
                confidence = round(best_score, 6)
            elif entity_id is not None:
                state = "linked"
                defer_reason = None
                abstain_reason = None
                confidence = round(min(family_conf, entity_score), 6)
            elif entity_reason == "entity_ambiguous":
                state = "needs_review"
                defer_reason = None
                abstain_reason = entity_reason
                confidence = round(family_conf * 0.5, 6)
            else:
                state = "abstained"
                defer_reason = None
                abstain_reason = entity_reason or "entity_undetected"
                confidence = round(family_conf * 0.5, 6)

        return DocumentPlan(
            family=family,
            family_source=family_source,
            entity_id=entity_id,
            entity_candidates=candidate_plans,
            confidence=confidence,
            abstain_reason=abstain_reason,
            defer_reason=defer_reason,
            state=state,
            role=resolved_role,
            signals=signals,
        )

    # ------------------------------------------------------------------ internals
    def _structure_hints(
        self, *, url: str, content_type: str | None, source: Any | None
    ) -> tuple[str, ...]:
        """Filename-shape family hints (roles.py); document structure only."""
        return family_hints_from_structure(
            url=url, content_type=content_type, source=source
        )

    def _family(
        self,
        *,
        url: str,
        source: Any | None,
        family_hint: str | None,
        text_sample: str | None,
        anchor_text: str | None,
        phase: Phase,
        content_type: str | None = None,
    ) -> tuple[str | None, FamilySource, float, dict[str, Any], str | None]:
        declared = declared_family(source)
        if declared is None and family_hint:
            declared = str(family_hint).strip() or None
            family_source: FamilySource = "reference"
        else:
            family_source = "registry" if declared else "none"

        # Document-structure hints are the weakest tier: recorded on every
        # decision, used only when registry, text and URL token evidence are
        # all silent. A declared family is never reconsidered by them.
        hints = self._structure_hints(url=url, content_type=content_type, source=source)
        signals: dict[str, Any] = {"structure_hints": list(hints)}

        if declared:
            if declared not in FAMILIES:
                # A registry/ref typo must fail loudly: extracting against an
                # unknown schema would either crash or fabricate a family.
                return None, family_source, 0.0, signals, f"unsupported_family:{declared}"
            # Precedence invariant: a declared family is never reconsidered,
            # downgraded or replaced by later text evidence.
            signals["declared"] = declared
            return declared, family_source, 1.0, signals, None

        if phase == "b":
            # Phase B reads text only. The URL was Phase A's evidence; letting it
            # vote again would let a stale URL token outrank the document text.
            family, score, detected, reason = detect_family(
                url="",
                anchor_text=None,
                text_sample=text_sample,
                rules=self.rules,
                min_score=self.thresholds.phase_b_min_score,
                margin=self.thresholds.phase_b_margin,
                recurring_min=self.thresholds.recurring_min,
                table_min_rows=self.thresholds.table_min_rows,
            )
            signals.update(detected)
            signals["phase_b_score"] = score
            if family is not None:
                return family, "heuristic", score, signals, None
            # Text found nothing: Phase A URL evidence is the weaker fallback.
            url_family, url_score, url_signals, url_reason = detect_family(
                url=url,
                anchor_text=anchor_text,
                rules=self.rules,
                min_score=self.thresholds.phase_a_min_score,
                margin=self.thresholds.phase_a_margin,
            )
            signals.update({f"phase_a_{k}": v for k, v in url_signals.items()})
            if url_family is not None:
                signals["fallback"] = "phase_a_url"
                return url_family, "heuristic", url_score, signals, None
            # URL token evidence is silent too (or below its floor): the
            # filename shape is the last, weakest tier. One hinted family is an
            # answer; two are a refusal, never a guess.
            if len(hints) == 1 and (url_reason or "family_undetected") == "family_undetected":
                signals["fallback"] = "structure_hint"
                signals["structure_family"] = hints[0]
                return hints[0], "heuristic", self.rules.weights.structure, signals, None
            if len(hints) > 1:
                signals["structure_hints_ambiguous"] = 1
            signals["best_score"] = score
            return None, "none", score, signals, reason or url_reason

        family, score, detected, reason = detect_family(
            url=url,
            anchor_text=anchor_text,
            rules=self.rules,
            min_score=self.thresholds.phase_a_min_score,
            margin=self.thresholds.phase_a_margin,
        )
        signals.update(detected)
        # The raw score survives even when detection abstains, because the
        # acquisition floor is a different question from acceptance.
        best = score_families(
            url=url, anchor_text=anchor_text, rules=self.rules
        )
        best_score = best[0].score if best else 0.0
        # A single filename hint is worth acquiring text for: it lifts the
        # plausible-but-unconfident score above the acquire floor so Phase A
        # defers instead of abstaining. It never declares the family itself.
        if family is None and (reason or "family_undetected") == "family_undetected":
            if len(hints) == 1:
                signals["structure_hint"] = hints[0]
                best_score = max(best_score, self.rules.weights.structure)
            elif len(hints) > 1:
                signals["structure_hints_ambiguous"] = 1
        signals["best_score"] = best_score
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
        max_mentions: int | None = None,
    ) -> tuple[int | None, float, list[EntityCandidatePlan], str | None]:
        mentions = collect_mentions(
            document,
            text_sample=text_sample,
            anchor_text=anchor_text,
            lexicon=self.er_config.lexicon,
            max_mentions=max_mentions or self.max_mentions,
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
        plan_state=plan.persisted_state,
        plan_confidence=plan.confidence,
        plan_reason=plan.persisted_reason,
        role=plan.role,
    )
