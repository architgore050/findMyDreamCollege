"""Reusable programme materialisation: *existing* verified claims -> rows.

``ProgrammeStage`` ran only inside the crawl/ingest path, so claims discovered
after a crawl — or re-verified after a prompt/scope fix — never became
programme rows and the tool could not rank anything despite holding seat data.
This module owns the single implementation of the write path; both
``ProgrammeStage`` (crawl path) and ``fmd programmes materialise`` (on demand)
call it, so the two can never drift.

Contract (unchanged from the pure stage, now with an explicit plan):

* input is ``programmes.materialisation_claims()`` — only positively verified,
  unsuperseded, non-modelled claims from ``role='evidence'`` documents whose
  entity is an institution;
* one row per ``(institution, ER-normalised programme name)``; every set
  attribute records the claim id that produced it in ``field_sources``;
* an attribute with no claim stays ``NULL`` — nothing is invented, nothing is
  derived (semesters are not months);
* ``seats.total`` claims are refused with ``AGGREGATE_SEATS_REFUSED``: an
  aggregate is never a per-programme intake (the claim itself stays in
  ``claims``; the refusal is reported so the discard is auditable);
* idempotent and resumable: an unchanged row (same values, same claim ids in
  ``field_sources``) is *not* rewritten unless ``force=True``; a changed claim
  is reported per field via :class:`FieldChange` (old claim id -> new).

Nothing here reads the network or a model. ``dry_run=True`` builds and reports
the whole plan and writes nothing.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fmd.db.repositories import ProgrammeCreate
from fmd.ingest.programmes import (
    AGGREGATE_SEATS_REFUSED,
    ATTRIBUTE_FIELDS,
    ProgrammeDraft,
    ProgrammeMapping,
    field_for_predicate_map,
    programme_drafts,
)
from fmd.orchestrate.ledger import RunSummary


@dataclass(frozen=True)
class FieldChange:
    """One attribute whose proving claim changed (old claim id -> new)."""

    field: str
    old_claim_id: str | None
    new_claim_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "old_claim_id": self.old_claim_id,
            "new_claim_id": self.new_claim_id,
        }


@dataclass(frozen=True)
class AggregateRefusal:
    """An aggregate ``seats.total`` claim seen and refused (never an intake)."""

    claim_id: str
    institution_id: int | None
    value: Any
    reason: str = AGGREGATE_SEATS_REFUSED

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "institution_id": self.institution_id,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProgrammePlanEntry:
    """One (institution, normalised name) row: the plan and what was written."""

    institution_id: int
    normalised_name: str
    name: str
    action: str  # create | update | unchanged
    changed_fields: tuple[FieldChange, ...]
    field_sources: Mapping[str, str]
    seats: int | None
    needs_review: bool
    review_reasons: tuple[str, ...]
    existing_id: int | None = None
    record_id: int | None = None
    written: bool = False
    forced: bool = False
    #: The repo object returned by ``upsert()`` (a ProgrammeRecord for the real
    #: repo). Kept out of ``as_dict`` — it is for the caller's loop, not JSON.
    record: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "institution_id": self.institution_id,
            "normalised_name": self.normalised_name,
            "name": self.name,
            "action": self.action,
            "changed_fields": [change.as_dict() for change in self.changed_fields],
            "field_sources": dict(self.field_sources),
            "seats": self.seats,
            "needs_review": self.needs_review,
            "review_reasons": list(self.review_reasons),
            "existing_id": self.existing_id,
            "record_id": self.record_id,
            "written": self.written,
            "forced": self.forced,
        }


@dataclass(frozen=True)
class EvidenceCompleteness:
    """How much of a programme's evidence base is actually present.

    ``backed_fields`` counts the six materialisation attribute fields (name,
    credential level, sanctioned intake, aggregate, duration, internship
    months) that at least one eligible claim proves — the aggregate counts as
    *evidence present* even though it is never stored on the row.
    ``components_backed`` counts the scoring components (from the predicate
    registry) that at least one eligible claim feeds.
    """

    fields_backed: int
    fields_total: int
    backed_fields: tuple[str, ...]
    components_backed: tuple[str, ...]
    components_total: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields_backed": self.fields_backed,
            "fields_total": self.fields_total,
            "backed_fields": list(self.backed_fields),
            "components_backed": list(self.components_backed),
            "components_total": self.components_total,
        }


@dataclass(frozen=True)
class MaterialiseResult:
    """The full plan (and, without ``dry_run``, the written outcome)."""

    claims: int
    entries: tuple[ProgrammePlanEntry, ...]
    created: int
    updated: int
    unchanged: int
    written: int
    dry_run: bool
    rejected: tuple[dict[str, Any], ...] = ()
    aggregates_refused: tuple[AggregateRefusal, ...] = ()
    drafts: tuple[ProgrammeDraft, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return True  # materialisation refuses rather than fails; a run always plans

    def as_dict(self) -> dict[str, Any]:
        return {
            "claims": self.claims,
            "created": self.created,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "written": self.written,
            "dry_run": self.dry_run,
            "entries": [entry.as_dict() for entry in self.entries],
            "rejected": list(self.rejected),
            "aggregates_refused": [refusal.as_dict() for refusal in self.aggregates_refused],
        }


# ------------------------------------------------------------------ helpers


def programme_create_from_draft(draft: ProgrammeDraft) -> ProgrammeCreate:
    """The persisted payload for one draft (moved verbatim from the stage)."""
    return ProgrammeCreate(
        institution_id=draft.institution_id,
        name=draft.name,
        credential_level=draft.credential_level,
        duration_months=draft.duration_months,
        internship_months=draft.internship_months,
        affiliated_university_id=draft.affiliated_university_id,
        recognition_status=draft.recognition_status,
        seats=draft.seats,
        medium=draft.medium,
        normalised_name=draft.normalised_name,
        field_sources=dict(draft.field_sources),
        needs_review=draft.needs_review,
        review_reasons=list(draft.review_reasons),
        source_family=draft.source_family,
        in_scope=draft.in_scope,
        scope_reason=draft.scope_reason,
        scope_tier=draft.scope_tier,  # type: ignore[arg-type]
        scope_evidence=list(draft.scope_evidence),
        scope_model=draft.scope_model,
        scope_prompt_version=draft.scope_prompt_version,
        scope_cached=draft.scope_cached,
    )


def _resolve_institution_fn(entities: Any | None):
    """Entity resolver over a duck-typed EntitiesRepo (``get`` + ``kind``)."""
    if entities is None:
        return None
    getter = getattr(entities, "get", None)

    def resolve(entity_id: int) -> Any | None:
        entity = getter(entity_id) if getter is not None else None
        if entity is None or getattr(entity, "kind", None) != "institution":
            return None
        return entity

    return resolve


def _resolve_university_fn(entities: Any | None):
    """Affiliating-university resolver: exact name first, then ER-normalised."""
    if entities is None:
        return lambda name: None

    def resolve(name: str | None) -> int | None:
        if not name:
            return None
        entity = None
        finder = getattr(entities, "find", None)
        if finder is not None:
            entity = finder("institution", name)
        if entity is None:
            from fmd.er.normalise import normalise_name

            normalised_finder = getattr(entities, "find_normalised", None)
            if normalised_finder is not None:
                entity = normalised_finder("institution", normalise_name(name).text)
        return getattr(entity, "id", None) if entity is not None else None

    return resolve


def _existing_row(programmes: Any, institution_id: int, normalised_name: str) -> Any | None:
    """The stored row for one (institution, normalised name) key, if any.

    Repos without a key lookup (older fakes) fall back to a scan; a repo with
    neither reports "unknown", which is treated as changed so a write is never
    silently skipped on the strength of a missing lookup.
    """
    finder = getattr(programmes, "get_by_key", None)
    if callable(finder):
        return finder(institution_id, normalised_name)
    lister = getattr(programmes, "list_for_institution", None)
    if callable(lister):
        for row in lister(institution_id):
            if getattr(row, "normalised_name", None) == normalised_name:
                return row
    return None


def _scope_content(payload: ProgrammeCreate) -> tuple[Any, ...]:
    """The scoping decision's *content*, without its timestamp."""
    return (
        payload.in_scope,
        payload.scope_reason,
        payload.scope_tier,
        tuple(payload.scope_evidence or ()),
        payload.scope_model,
        payload.scope_prompt_version,
        payload.scope_cached,
    )


def _with_decided_at(payload: ProgrammeCreate, *, existing: Any | None) -> ProgrammeCreate:
    """Stamp ``scope_decided_at`` only when the decision itself is new.

    A fresh decision (new row, or content that differs from the stored row)
    gets ``now``; an unchanged decision keeps the stored timestamp so a rerun
    of an unchanged corpus writes nothing.
    """
    if payload.scope_tier is None:
        return payload
    if existing is not None and _scope_content(payload) == _scope_content(existing):
        return payload.model_copy(update={"scope_decided_at": existing.scope_decided_at})
    return payload.model_copy(update={"scope_decided_at": datetime.now(UTC)})


_AFFILIATION_COLUMN = "affiliated_university"

#: Metadata, not content: when a row is otherwise unchanged, its
#: ``scope_decided_at`` must not tick on every run (that would break
#: idempotence), so the value-comparison safety net skips it.
_METADATA_COLUMNS = frozenset({"field_sources", "scope_decided_at"})


def _changed_fields(
    existing: Any,
    payload: ProgrammeCreate,
    field_for_predicate: Mapping[str, str],
) -> tuple[FieldChange, ...]:
    """Fields whose proving claim (or value) differs from the stored row.

    Primary signal: the predicate-keyed ``field_sources`` claim ids. Safety
    net: a direct column comparison catches value changes that keep the same
    claim id (an alias-table rename, a re-resolved affiliation).
    """
    changes: list[FieldChange] = []
    old_sources = dict(getattr(existing, "field_sources", {}) or {})
    new_sources = dict(payload.field_sources or {})
    for predicate in sorted(set(old_sources) | set(new_sources)):
        old_id = old_sources.get(predicate)
        new_id = new_sources.get(predicate)
        if old_id != new_id:
            changes.append(
                FieldChange(field_for_predicate.get(predicate, predicate), old_id, new_id)
            )
    old_view = existing.model_dump(exclude={"id", "created_at"})
    new_view = payload.model_dump()
    for column in sorted(set(old_view) | set(new_view)):
        if column in _METADATA_COLUMNS or column not in old_view or column not in new_view:
            continue
        if old_view[column] != new_view[column]:
            name = _AFFILIATION_COLUMN if column == "affiliated_university_id" else column
            if not any(change.field == name for change in changes):
                changes.append(FieldChange(name, None, None))
    return tuple(changes)


def component_index(predicates: Any) -> dict[str, str]:
    """Predicate -> scoring component, from the committed registry.

    Accepts the typed ``PredicatesConfig`` (or anything with ``.predicates`` /
    ``.aliases`` mappings). Declared aliases resolve to their canonical
    predicate's component. Unknown predicates contribute nothing — the same
    "reported, never guessed" rule the scorer applies.
    """
    if predicates is None:
        return {}
    rules = getattr(predicates, "predicates", None) or {}
    aliases = getattr(predicates, "aliases", None) or {}
    index: dict[str, str] = {}
    for name, rule in rules.items():
        component = getattr(rule, "component", None)
        if component:
            index[str(name)] = str(component)
    for alias, canonical in dict(aliases).items():
        if str(canonical) in index:
            index[str(alias)] = index[str(canonical)]
    return index


def evidence_completeness(
    claims: Sequence[Any],
    mapping: ProgrammeMapping,
    predicates: Any = None,
) -> EvidenceCompleteness:
    """Evidence completeness for one institution's eligible claims.

    ``claims`` are the materialisation-eligible claims *of one institution*
    (``programmes.materialisation_claims()`` rows already carry the filters).
    """
    bridge = field_for_predicate_map(mapping)
    backed: dict[str, str] = {}
    components: set[str] = set()
    index = component_index(predicates)
    for claim in claims:
        predicate = str(_get(claim, "predicate", "") or "")
        claim_id = str(_get(claim, "id", "") or "")
        attribute = bridge.get(predicate)
        if attribute is not None and attribute not in backed:
            backed[attribute] = claim_id
        component = index.get(predicate)
        if component:
            components.add(component)
    # credential_level is classified from the name claim, never claimed on its
    # own: it is backed exactly when the name is.
    if "name" in backed:
        backed.setdefault("credential_level", backed["name"])
    ordered = tuple(name for name in ATTRIBUTE_FIELDS if name in backed)
    return EvidenceCompleteness(
        fields_backed=len(ordered),
        fields_total=len(ATTRIBUTE_FIELDS),
        backed_fields=ordered,
        components_backed=tuple(sorted(components)),
        components_total=len(set(index.values())),
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ----------------------------------------------------------------- entrypoint


def materialise_programmes(
    programmes: Any,
    entities: Any | None = None,
    mapping: ProgrammeMapping | None = None,
    *,
    scoper: Any | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> MaterialiseResult:
    """Materialise programme rows from the repo's existing verified claims.

    ``programmes`` needs ``materialisation_claims()`` and ``upsert()`` (the
    :class:`~fmd.db.repositories.ProgrammesRepo` interface; test fakes use the
    same two methods). ``entities`` is an optional duck-typed EntitiesRepo for
    institution resolution and affiliation linking. ``scoper`` is the optional
    W9 three-tier scoping service: when present every draft is scoped
    (rule → semantic → manual) before it is written; when absent drafts keep
    the pre-W9 defaults. ``dry_run`` writes nothing; ``force`` rewrites rows
    that would otherwise be left untouched.
    """
    fetch = getattr(programmes, "materialisation_claims", None)
    claims: list[Any] = list(fetch() or []) if callable(fetch) else []
    mapping = mapping or ProgrammeMapping.from_params(None)
    field_for_predicate = field_for_predicate_map(mapping)

    drafts, rejected = programme_drafts(
        claims,
        mapping,
        resolve_institution=_resolve_institution_fn(entities),
        scoper=scoper,
    )
    aggregates_refused = tuple(
        AggregateRefusal(
            claim_id=str(entry.get("claim_id", "")),
            institution_id=entry.get("institution_id"),
            value=entry.get("value"),
        )
        for entry in rejected
        if entry.get("reason") == AGGREGATE_SEATS_REFUSED
    )

    resolve_university = _resolve_university_fn(entities)
    entries: list[ProgrammePlanEntry] = []
    created = updated = unchanged = written = 0
    #: A repo without any key lookup (legacy fakes, pre-0006 callers) cannot
    #: know whether a row exists before writing; those always write and derive
    #: create-vs-update from the upsert's own flag — the crawl-path semantics.
    lookup_available = callable(getattr(programmes, "get_by_key", None)) or callable(
        getattr(programmes, "list_for_institution", None)
    )
    for draft in drafts:
        resolved = draft.with_affiliation(resolve_university(draft.affiliated_university_name))
        payload = programme_create_from_draft(resolved)
        payload = _with_decided_at(payload, existing=None)
        existing = (
            _existing_row(programmes, resolved.institution_id, resolved.normalised_name)
            if lookup_available
            else None
        )
        if existing is not None:
            payload = _with_decided_at(payload, existing=existing)
        existing_id = getattr(existing, "id", None) if existing is not None else None
        changes = (
            _changed_fields(existing, payload, field_for_predicate) if existing is not None else ()
        )
        action = "create" if existing is None else ("update" if changes else "unchanged")
        forced = force and action == "unchanged"
        will_write = action != "unchanged" or forced
        record: Any | None = None
        record_id = None
        if not will_write:
            unchanged += 1
        elif dry_run:
            # Plan only: a create or a rewrite (changed or forced), no write.
            written += 1
            if action == "create":
                created += 1
            else:
                updated += 1
        else:
            record, upsert_created = programmes.upsert(payload)
            record_id = getattr(record, "id", None)
            written += 1
            if not lookup_available:
                # No lookup: trust the upsert's own create/update flag.
                action = "create" if upsert_created else "update"
            if action == "create":
                created += 1
            else:
                # A changed row, or a forced rewrite of an unchanged one.
                updated += 1
        entries.append(
            ProgrammePlanEntry(
                institution_id=resolved.institution_id,
                normalised_name=resolved.normalised_name,
                name=resolved.name,
                action=action,
                changed_fields=changes,
                field_sources=dict(resolved.field_sources),
                seats=resolved.seats,
                needs_review=resolved.needs_review,
                review_reasons=resolved.review_reasons,
                existing_id=existing_id,
                record_id=record_id,
                written=will_write and not dry_run,
                forced=forced,
                record=record,
            )
        )

    return MaterialiseResult(
        claims=len(claims),
        entries=tuple(entries),
        created=created,
        updated=updated,
        unchanged=unchanged,
        written=written,
        dry_run=dry_run,
        rejected=tuple(rejected),
        aggregates_refused=aggregates_refused,
        drafts=tuple(drafts),
    )


def materialise_from_database(
    database: Any,
    registry: Any | None = None,
    *,
    scoper: Any | None = None,
    router: Any | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> MaterialiseResult:
    """One-shot materialisation against a database, without crawling.

    The mapping comes from the registry's institution-site source
    (``params.programme_mapping``) when a registry is supplied, otherwise from
    the committed defaults — the same resolution the crawl-path stage uses.
    When no ``scoper`` is supplied one is built from the registry's
    ``config/programmes.yaml`` taxonomy (Tier 1 + cached manual decisions);
    ``router`` (the provider-agnostic model router) additionally enables the
    Tier-2 semantic classifier. Without a router, names no rule decides stay
    ``uncertain`` → ``needs_review`` — conservative by construction.
    """
    from fmd.orchestrate.ingest import ProgrammeStage

    stage = ProgrammeStage.from_database(database, registry)
    mapping = stage.mapping()
    if scoper is None:
        scoper = _scoper_for(database, registry, mapping, router=router)
    return materialise_programmes(
        stage.programmes,
        stage.entities,
        mapping,
        scoper=scoper,
        dry_run=dry_run,
        force=force,
    )


def _scoper_for(
    database: Any, registry: Any | None, mapping: ProgrammeMapping, *, router: Any | None
) -> Any | None:
    """A scoper over the registry taxonomy, or ``None`` without one.

    The decision store is the DB repo when available (the cache and the
    manual-decision learning loop persist); test fakes without the scope
    methods simply run store-less.
    """
    programmes_config = getattr(registry, "programmes", None)
    if programmes_config is None:
        return None
    from fmd.programmes.service import RepoDecisionStore, build_scoper

    store = None
    if callable(getattr(database, "connection", None)):
        from fmd.db.repositories import ProgrammesRepo

        store = RepoDecisionStore(ProgrammesRepo(database))
    return build_scoper(registry, mapping, store=store, router=router)


def materialise_stage_run(stage: Any, *, summary: RunSummary) -> list[Any]:
    """Drive the crawl-path stage through the same implementation.

    ``force=True`` keeps the stage's historical semantics (every draft is
    rewritten on every run, so a vanished claim cannot linger); the on-demand
    CLI defaults to writing only changed rows. The stage is scoped with the
    registry taxonomy (Tier 1 rules + stored manual decisions, no router —
    names no rule decides stay ``uncertain`` → ``needs_review`` and are queued
    for `fmd programmes requalify`, which runs the full semantic pass).
    """
    mapping = stage.mapping()
    scoper = None
    programmes_config = getattr(stage, "registry", None)
    if programmes_config is not None and getattr(programmes_config, "programmes", None):
        from fmd.programmes.service import RepoDecisionStore, build_scoper

        store = None
        if hasattr(stage.programmes, "scope_decision_get"):
            store = RepoDecisionStore(stage.programmes)
        scoper = build_scoper(stage.registry, mapping, store=store, router=None)
    result = materialise_programmes(
        stage.programmes,
        stage.entities,
        mapping,
        scoper=scoper,
        dry_run=False,
        force=True,
    )
    summary.bump("programme_claims", result.claims)
    summary.bump("programmes_rejected", len(result.rejected))
    if result.rejected:
        summary.metadata["programme_rejections"] = list(result.rejected)[:50]
    for entry in result.entries:
        if entry.written:
            summary.bump("programmes_created" if entry.action == "create" else "programmes_updated")
            if entry.needs_review:
                summary.bump("programmes_needs_review")
    summary.bump("programmes", result.written)
    summary.metadata["programme_aggregates_refused"] = [
        refusal.as_dict() for refusal in result.aggregates_refused
    ][:50]
    counter = getattr(stage.programmes, "counts", None)
    if counter is not None:
        with contextlib.suppress(Exception):  # observability only
            summary.metadata["programmes"] = dict(counter())
    return [entry.record for entry in result.entries if entry.written]


__all__ = [
    "AggregateRefusal",
    "EvidenceCompleteness",
    "FieldChange",
    "MaterialiseResult",
    "ProgrammePlanEntry",
    "component_index",
    "evidence_completeness",
    "materialise_from_database",
    "materialise_programmes",
    "materialise_stage_run",
    "programme_create_from_draft",
]
