"""Three-tier programme scoping service: rules → semantic → manual.

Tier 1 (:mod:`fmd.programmes.taxonomy`) decides the obvious names from
``config/programmes.yaml``. Tier 2 (:mod:`fmd.programmes.classifier`) asks the
provider-agnostic router about the rest, citing evidence, and degrades to
``uncertain`` — never a guessed ``yes``. Tier 3 is the learning loop: a human
``fmd programmes decide`` verdict is stored as a ``manual`` decision that
Tier 1 consumes on every later run (manual > exclude > include > semantic >
uncertain).

Precedence is enforced by evaluation order here: manual lookup first, rules
second, classifier last. Out-of-scope rows are *kept* (``in_scope=false`` +
``scope_reason``) so the audit trail survives; scoring reads ``in_scope`` and
can never rank an MBBS.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fmd.db.repositories import ProgrammeScopeDecisionCreate
from fmd.programmes.classifier import (
    DecisionStore,
    EquivalenceCandidate,
    classify,
    decision_cache_key,
)
from fmd.programmes.taxonomy import (
    SCOPE_UNCERTAIN_REASON,
    ScopeDecision,
    Taxonomy,
)

#: Claim predicate families whose quotes are useful classifier evidence.
EVIDENCE_PREDICATE_FAMILIES = ("curriculum.", "accreditation.", "recognition.", "regulator.")

#: Cap per candidate so the prompt stays bounded.
MAX_EVIDENCE_PER_CANDIDATE = 8


@dataclass(frozen=True)
class ScopeInput:
    """One programme awaiting a scope decision, with its citable context."""

    institution_id: int
    raw_name: str
    normalised_name: str
    institution_name: str | None = None
    institution_kind: str | None = None
    country: str | None = None
    department: str | None = None
    claims: tuple[str, ...] = ()

    def institution_type(self) -> str:
        """The institution-type half of the semantic cache key."""
        return f"{self.institution_kind or 'institution'}:{self.country or ''}"

    def context_text(self) -> str:
        return " ".join(
            part
            for part in (
                self.raw_name,
                self.institution_name or "",
                self.department or "",
                " ".join(self.claims),
            )
            if part
        )


def evidence_from_claims(claims: Iterable[Any]) -> tuple[str, ...]:
    """Render the claims the classifier may cite (claim id, predicate, quote).

    Only curriculum/accreditation/recognition/regulator claims are rendered:
    a seat count says nothing about what a programme teaches. Deterministic
    (claim-id order), capped, and every entry names its claim id so a
    `semantic` admission resolves back to the append-only evidence.
    """
    rendered: list[str] = []
    for claim in sorted(claims, key=lambda c: int(_get(c, "id", 0) or 0)):
        predicate = str(_get(claim, "predicate", "") or "")
        if not predicate.startswith(EVIDENCE_PREDICATE_FAMILIES):
            continue
        value = _get(claim, "value")
        quote = _get(claim, "quote")
        text = f"claim {_get(claim, 'id')} [{predicate}] {value}"
        if quote:
            text += f' — "{quote}"'
        rendered.append(text)
        if len(rendered) >= MAX_EVIDENCE_PER_CANDIDATE:
            break
    return tuple(rendered)


class ProgrammeScoper:
    """Applies the three tiers in precedence order to a batch of programmes."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        config: Any,
        *,
        store: DecisionStore | None = None,
        router: Any | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.config = config
        self.store = store
        self.router = router

    def cache_key(self, inp: ScopeInput, normalised_name: str | None = None) -> str:
        return decision_cache_key(
            normalised_name or inp.normalised_name,
            inp.institution_type(),
            self.config.definition.version,
        )

    def scope_all(
        self, inputs: Sequence[ScopeInput]
    ) -> list[tuple[ScopeInput, ScopeDecision]]:
        """Decide every input; one decision per input, order preserved.

        Manual decisions win, rules decide the obvious, and everything left
        goes to the classifier in ONE batched pass (config-sized bundles,
        cached by name hash).
        """
        results: list[tuple[ScopeInput, ScopeDecision]] = []
        pending: list[tuple[ScopeInput, EquivalenceCandidate]] = []
        countries: dict[str, str | None] = {}
        for inp in inputs:
            normalised, matched_alias = self.taxonomy.resolve_alias(
                inp.raw_name, inp.country
            )
            key = self.cache_key(inp, normalised)
            record = self.store.get(key) if self.store is not None else None
            if record is not None and record.tier == "manual":
                results.append((inp, _manual_from_record(record)))
                continue
            verdict = self.taxonomy.apply(raw_name=inp.raw_name, country=inp.country)
            if verdict.decided:
                results.append((inp, _rule_decision(verdict)))
                continue
            candidate = EquivalenceCandidate(
                key=key,
                normalised_name=normalised,
                raw_name=inp.raw_name,
                institution=inp.institution_name,
                institution_type=inp.institution_type(),
                country=inp.country,
                department=inp.department,
                evidence=inp.claims,
            )
            countries[key] = inp.country
            pending.append((inp, candidate))

        if pending:
            semantic, notes = classify(
                [candidate for _inp, candidate in pending],
                taxonomy=self.taxonomy,
                config=self.config,
                router=self.router,
                store=self.store,
                countries=countries,
            )
            for note in notes:
                _log_note(note)
            for inp, candidate in pending:
                results.append(
                    (
                        inp,
                        semantic.get(candidate.key)
                        or ScopeDecision(
                            in_scope=False,
                            tier="semantic",
                            reason="semantic_unavailable_uncertain",
                            needs_review=True,
                            prompt_version=self.config.classifier.prompt_version,
                        ),
                    )
                )
        return results


def _rule_decision(verdict: Any) -> ScopeDecision:
    return ScopeDecision(
        in_scope=verdict.in_scope,
        tier="rule",
        reason=verdict.reason,
        needs_review=False,
        evidence=tuple(verdict.matched),
        flags=tuple(verdict.flags),
    )


def _manual_from_record(record: Any) -> ScopeDecision:
    return ScopeDecision(
        in_scope=record.decision == "in_scope",
        tier="manual",
        reason=f"manual_{record.decision}",
        needs_review=False,
        evidence=tuple(record.evidence or ()),
        cached=True,
        flags=tuple(record.flags or ()),
    )


def _log_note(note: str) -> None:
    logging.getLogger(__name__).info("programme scoping: %s", note)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ------------------------------------------------------------------- store


class RepoDecisionStore:
    """Adapts :class:`~fmd.db.repositories.ProgrammesRepo` to DecisionStore."""

    def __init__(self, repo: Any) -> None:
        self.repo = repo

    def get(self, cache_key: str) -> Any | None:
        return self.repo.scope_decision_get(cache_key)

    def put(self, payload: Mapping[str, Any]) -> Any | None:
        create = ProgrammeScopeDecisionCreate.model_validate(dict(payload))
        record, _created = self.repo.scope_decision_put(create)
        return record


def build_scoper(
    registry: Any,
    mapping: Any | None = None,
    *,
    store: DecisionStore | None = None,
    router: Any | None = None,
) -> ProgrammeScoper:
    """A scoper over the registry's taxonomy (``config/programmes.yaml``).

    ``mapping`` (the materialisation mapping) contributes its alias table as
    extra Tier-1 aliases, so the grouping vocabulary and the scoping
    vocabulary can never disagree. ``store`` persists semantic decisions and
    manual verdicts; ``router`` enables the Tier-2 classifier — without one,
    undecided names stay ``uncertain`` → ``needs_review``.
    """
    programmes_config = registry.programmes
    extra_aliases = dict(getattr(mapping, "aliases", {}) or {}) if mapping is not None else {}
    taxonomy = Taxonomy(programmes_config, extra_aliases=extra_aliases)
    return ProgrammeScoper(taxonomy, programmes_config, store=store, router=router)


def scoper_for_database(
    database: Any,
    registry: Any,
    mapping: Any | None = None,
    *,
    router: Any | None = None,
) -> ProgrammeScoper:
    """A scoper whose decision store is the database's (cache + loop persist)."""
    from fmd.db.repositories import ProgrammesRepo

    return build_scoper(
        registry,
        mapping,
        store=RepoDecisionStore(ProgrammesRepo(database)),
        router=router,
    )


# ------------------------------------------------------------ manual (tier 3)


@dataclass(frozen=True)
class ManualDecisionOutcome:
    """What ``fmd programmes decide`` did, for the CLI and the audit trail."""

    programme_id: int
    name: str
    equivalent: bool
    note: str
    cache_key: str
    previous: dict[str, Any] | None
    row_updated: bool
    in_scope: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "programme_id": self.programme_id,
            "name": self.name,
            "equivalent": self.equivalent,
            "note": self.note,
            "cache_key": self.cache_key,
            "previous": self.previous,
            "row_updated": self.row_updated,
            "in_scope": self.in_scope,
        }


def record_manual_decision(
    programmes: Any,
    entities: Any | None,
    taxonomy: Taxonomy,
    config: Any,
    programme_id: int,
    *,
    equivalent: bool,
    note: str,
    decided_by: str = "human",
) -> ManualDecisionOutcome:
    """Write a human equivalence decision and apply it to the row.

    The decision is stored under the same cache key the classifier uses, with
    ``tier='manual'`` and ``decided_by='human'``, so every later materialise/
    requalify run consumes it *before* rules and the classifier — the system
    accumulates knowledge instead of re-asking. The programme row is updated
    immediately; the scope-uncertain review reason is cleared while any other
    review reasons (name collisions, unresolved affiliation) are preserved.
    """
    row = programmes.get(programme_id)
    if row is None:
        raise ValueError(f"no programme with id {programme_id}")
    country = None
    entity_kind = None
    if entities is not None:
        entity = entities.get(row.institution_id)
        if entity is not None:
            country = getattr(entity, "country", None)
            entity_kind = getattr(entity, "kind", None)
    normalised, _matched = taxonomy.resolve_alias(row.name, country)
    institution_type = f"{entity_kind or 'institution'}:{country or ''}"
    key = decision_cache_key(
        normalised, institution_type, config.definition.version
    )
    store = RepoDecisionStore(programmes) if hasattr(programmes, "scope_decision_put") else None
    previous = None
    if store is not None:
        existing = store.get(key)
        if existing is not None:
            previous = {
                "decision": existing.decision,
                "tier": existing.tier,
                "decided_by": existing.decided_by,
                "rationale": existing.rationale,
                "evidence": list(existing.evidence or ()),
                "model": existing.model,
            }
        store.put(
            {
                "cache_key": key,
                "normalised_name": normalised,
                "raw_name": row.name,
                "institution_type": institution_type,
                "country": country,
                "decision": "in_scope" if equivalent else "out_of_scope",
                "tier": "manual",
                "confidence": None,
                "rationale": note or None,
                "evidence": [note] if note else [],
                "model": None,
                "prompt_version": None,
                "cached": False,
                "needs_review": False,
                "flags": [],
                "definition_version": config.definition.version,
                "decided_by": decided_by,
                "note": note or None,
            }
        )

    reasons = [
        reason
        for reason in (row.review_reasons or [])
        if reason != SCOPE_UNCERTAIN_REASON
    ]
    updated = None
    if hasattr(programmes, "update_scope"):
        updated = programmes.update_scope(
            programme_id,
            in_scope=equivalent,
            scope_reason=f"manual_{'equivalent' if equivalent else 'not_equivalent'}",
            scope_tier="manual",
            scope_evidence=[note] if note else [],
            scope_model=None,
            scope_prompt_version=None,
            scope_cached=False,
            needs_review=bool(reasons),
            review_reasons=reasons,
            scope_decided_at=datetime.now(UTC),
        )
    return ManualDecisionOutcome(
        programme_id=programme_id,
        name=row.name,
        equivalent=equivalent,
        note=note,
        cache_key=key,
        previous=previous,
        row_updated=updated is not None,
        in_scope=equivalent,
    )


# ---------------------------------------------------------------- requalify


@dataclass(frozen=True)
class RequalifyEntry:
    """One existing row and what requalification did to it."""

    programme_id: int
    institution_id: int
    name: str
    action: str  # rescope | merge | unchanged
    in_scope: bool | None = None
    scope_tier: str | None = None
    scope_reason: str | None = None
    needs_review: bool | None = None
    flags: tuple[str, ...] = ()
    merged_into: int | None = None
    merged_rows: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "programme_id": self.programme_id,
            "institution_id": self.institution_id,
            "name": self.name,
            "action": self.action,
            "in_scope": self.in_scope,
            "scope_tier": self.scope_tier,
            "scope_reason": self.scope_reason,
            "needs_review": self.needs_review,
            "flags": list(self.flags),
            "merged_into": self.merged_into,
            "merged_rows": dict(self.merged_rows),
        }


@dataclass(frozen=True)
class RequalifyResult:
    """The requalification plan (and outcome when not ``dry_run``)."""

    rows: int
    rescoped: int
    merged: int
    unchanged: int
    uncertain: int
    dry_run: bool
    entries: tuple[RequalifyEntry, ...] = ()
    notes: tuple[str, ...] = ()
    would_admit: int = 0
    would_reject: int = 0
    would_defer: int = 0
    would_classify: int = 0

    @property
    def ok(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "rescoped": self.rescoped,
            "merged": self.merged,
            "unchanged": self.unchanged,
            "uncertain": self.uncertain,
            "dry_run": self.dry_run,
            "entries": [entry.as_dict() for entry in self.entries],
            "notes": list(self.notes),
            "would_admit": self.would_admit,
            "would_reject": self.would_reject,
            "would_defer": self.would_defer,
            "would_classify": self.would_classify,
        }


def requalify_programmes(
    programmes: Any,
    entities: Any | None,
    mapping: Any,
    scoper: ProgrammeScoper | None,
    *,
    dry_run: bool = False,
) -> RequalifyResult:
    """Re-scope existing rows under the current taxonomy and decisions.

    Idempotent: a row whose stored scope already matches the fresh decision is
    left untouched, a merged duplicate cannot reappear (the grouping
    normalisation no longer produces its old key), and ``--dry-run`` plans
    everything without writing. The duplicate merge folds the *derived* row
    only — ``cost_items`` are re-pointed at the survivor and claims are never
    touched.
    """
    rows = _all_rows(programmes)
    if not rows:
        return RequalifyResult(
            rows=0, rescoped=0, merged=0, unchanged=0, uncertain=0, dry_run=dry_run
        )

    institution_names: dict[int, str | None] = {}
    countries: dict[int, str | None] = {}
    kinds: dict[int, str | None] = {}
    claims_by_entity: dict[int, list[Any]] = {}
    fetch_claims = getattr(programmes, "materialisation_claims", None)
    if callable(fetch_claims):
        for claim in fetch_claims() or []:
            claims_by_entity.setdefault(int(_get(claim, "entity_id", 0) or 0), []).append(claim)
    if entities is not None:
        for row in rows:
            if row.institution_id not in institution_names:
                entity = entities.get(row.institution_id)
                institution_names[row.institution_id] = getattr(entity, "canonical_name", None)
                countries[row.institution_id] = getattr(entity, "country", None)
                kinds[row.institution_id] = getattr(entity, "kind", None)

    # --- step 1: duplicates and stale keys from the improved normalisation.
    # A row grouped before the numbering-strip fix (or before an alias landed
    # in the registry) may hold a stale (institution, normalised_name) key:
    # rows sharing a recomputed key fold into one survivor, a singleton with a
    # stale key is rekeyed. Merges execute BEFORE any rekey so the unique
    # (institution, normalised_name) index can never collide.
    groups: dict[tuple[int, str], list[Any]] = {}
    keys: dict[int, str] = {}
    labels: dict[int, str] = {}
    for row in rows:
        key, label = _normalise_with_mapping(row.name, mapping)
        keys[int(row.id)] = key
        labels[int(row.id)] = label
        groups.setdefault((int(row.institution_id), key), []).append(row)
    merges: dict[int, tuple[int, dict[str, int]]] = {}
    survivors: set[int] = set()
    merge_entries: list[RequalifyEntry] = []
    for (institution_id, _key), members in sorted(groups.items()):
        if len(members) == 1:
            survivors.add(int(members[0].id))
            continue
        ordered = sorted(members, key=lambda row: int(row.id))
        # Prefer the row whose stored key is already the canonical one; else
        # the lowest id. Deterministic and stable across reruns.
        survivor = next(
            (row for row in ordered if getattr(row, "normalised_name", None) == _key),
            ordered[0],
        )
        survivors.add(int(survivor.id))
        for duplicate in members:
            if int(duplicate.id) == int(survivor.id):
                continue
            merges[int(duplicate.id)] = (int(survivor.id), {})
            merge_entries.append(
                RequalifyEntry(
                    programme_id=int(duplicate.id),
                    institution_id=institution_id,
                    name=duplicate.name,
                    action="merge",
                    merged_into=int(survivor.id),
                )
            )

    # --- step 2: fold the duplicates (derived rows only; claims untouched,
    # cost_items are re-pointed at the survivor first).
    merged = 0
    if not dry_run:
        for duplicate_id, (survivor_id, stats) in merges.items():
            result = programmes.merge_into(duplicate_id, survivor_id)
            stats.update(result)
            merged += 1
            for entry in merge_entries:
                if entry.programme_id == duplicate_id:
                    entry.merged_rows.update(result)
    else:
        merged = len(merges)

    # --- step 3: re-scope (and rekey) every surviving row.
    inputs: list[ScopeInput] = []
    ordered_survivors = sorted(survivors)
    row_by_id: dict[int, Any] = {int(row.id): row for row in rows}
    for row_id in ordered_survivors:
        row = row_by_id[row_id]
        if scoper is None:
            continue
        inputs.append(
            ScopeInput(
                institution_id=int(row.institution_id),
                raw_name=row.name,
                normalised_name=keys[int(row.id)],
                institution_name=institution_names.get(int(row.institution_id)),
                institution_kind=kinds.get(int(row.institution_id)),
                country=countries.get(int(row.institution_id)),
                department=None,
                claims=evidence_from_claims(claims_by_entity.get(int(row.institution_id), ())),
            )
        )
    decisions = scoper.scope_all(inputs) if scoper and inputs else []

    entries: list[RequalifyEntry] = list(merge_entries)
    rescoped = uncertain = unchanged = 0
    # One institution can hold several programmes; key by (institution, name).
    decision_by_input: dict[tuple[int, str], ScopeDecision] = {
        (inp.institution_id, inp.raw_name): decision for inp, decision in decisions
    }
    now = datetime.now(UTC)
    for row_id in ordered_survivors:
        row = row_by_id[row_id]
        inp_key = (int(row.institution_id), row.name)
        decision = decision_by_input.get(inp_key)
        # Rekey when the recomputed grouping key (or its alias-canonical label)
        # differs from what the pre-W9 materialisation stored.
        rekey = keys[int(row.id)] != getattr(row, "normalised_name", None) or labels[
            int(row.id)
        ] != getattr(row, "name", None)
        if decision is None and not rekey:
            entries.append(
                RequalifyEntry(
                    programme_id=int(row.id),
                    institution_id=int(row.institution_id),
                    name=row.name,
                    action="unchanged",
                )
            )
            unchanged += 1
            continue
        desired_reasons = [
            reason
            for reason in (row.review_reasons or [])
            if reason != SCOPE_UNCERTAIN_REASON
        ]
        stored_evidence = list(getattr(row, "scope_evidence", None) or [])
        desired_evidence = list(decision.evidence) if decision else stored_evidence
        if decision is not None and decision.needs_review:
            desired_reasons.append(SCOPE_UNCERTAIN_REASON)
        scope_changed = decision is not None and (
            bool(getattr(row, "in_scope", True)) != decision.in_scope
            or (getattr(row, "scope_tier", None) or None) != decision.tier
            or (getattr(row, "scope_reason", None) or None) != decision.reason
            or list(getattr(row, "scope_evidence", None) or []) != desired_evidence
            or bool(getattr(row, "needs_review", False)) != bool(desired_reasons)
            or list(row.review_reasons or []) != desired_reasons
        )
        if decision is not None and not scope_changed and not rekey:
            entries.append(
                RequalifyEntry(
                    programme_id=int(row.id),
                    institution_id=int(row.institution_id),
                    name=row.name,
                    action="unchanged",
                    in_scope=decision.in_scope,
                    scope_tier=decision.tier,
                    scope_reason=decision.reason,
                    needs_review=decision.needs_review,
                    flags=decision.flags,
                )
            )
            unchanged += 1
            continue
        rescoped += 1
        if decision is not None and decision.needs_review:
            uncertain += 1
        if not dry_run and hasattr(programmes, "update_scope"):
            update: dict[str, Any] = {
                "in_scope": decision.in_scope if decision else bool(getattr(row, "in_scope", True)),
                "scope_reason": decision.reason if decision else getattr(row, "scope_reason", None),
                "scope_tier": decision.tier if decision else getattr(row, "scope_tier", None),
                "scope_evidence": desired_evidence,
                "scope_model": decision.model if decision else getattr(row, "scope_model", None),
                "scope_prompt_version": (
                    decision.prompt_version
                    if decision
                    else getattr(row, "scope_prompt_version", None)
                ),
                "scope_cached": (
                    decision.cached if decision else bool(getattr(row, "scope_cached", False))
                ),
                "needs_review": bool(desired_reasons),
                "review_reasons": desired_reasons,
                "scope_decided_at": now if scope_changed or decision is None else None,
            }
            if rekey:
                # Only passed when actually rekeying: an absent kwarg means
                # "keep the stored key", never an accidental NULL.
                update["normalised_name"] = keys[int(row.id)]
                update["name"] = labels[int(row.id)]
            programmes.update_scope(int(row.id), **update)
        entries.append(
            RequalifyEntry(
                programme_id=int(row.id),
                institution_id=int(row.institution_id),
                name=row.name,
                action="rescope" if scope_changed or decision is not None else "rekey",
                in_scope=decision.in_scope if decision else bool(getattr(row, "in_scope", True)),
                scope_tier=decision.tier if decision else getattr(row, "scope_tier", None),
                scope_reason=decision.reason if decision else getattr(row, "scope_reason", None),
                needs_review=bool(desired_reasons),
                flags=decision.flags if decision else (),
            )
        )

    notes: list[str] = []
    if merges:
        notes.append(
            f"{len(merges)} duplicate row(s) folded (row-numbering normalisation)"
        )
    tally = _tally_decisions(decisions)
    return RequalifyResult(
        rows=len(rows),
        rescoped=rescoped,
        merged=merged,
        unchanged=unchanged,
        uncertain=uncertain,
        dry_run=dry_run,
        entries=tuple(entries),
        notes=tuple(notes),
        **tally,
    )


def _tally_decisions(
    decisions: Sequence[tuple[ScopeInput, ScopeDecision]],
) -> dict[str, int]:
    """Count scope outcomes for the dry-run report.

    A semantic decision whose reason is ``semantic_unavailable_uncertain`` is
    a name the classifier would have been asked about: it is both
    ``would_classify`` and ``would_defer`` (nothing is admitted without a
    decision). Rules and manual verdicts count as admit/reject, and any other
    uncertain outcome counts as deferred.
    """
    tally = {"would_admit": 0, "would_reject": 0, "would_defer": 0, "would_classify": 0}
    for _inp, decision in decisions:
        if (
            decision.tier == "semantic"
            and decision.reason == "semantic_unavailable_uncertain"
        ):
            tally["would_classify"] += 1
            tally["would_defer"] += 1
        elif decision.in_scope:
            tally["would_admit"] += 1
        elif decision.needs_review:
            tally["would_defer"] += 1
        else:
            tally["would_reject"] += 1
    return tally


def _all_rows(programmes: Any) -> list[Any]:
    lister = getattr(programmes, "list_all", None)
    if callable(lister):
        return list(lister() or [])
    filters = getattr(programmes, "list", None)
    if callable(filters):
        from fmd.db.repositories import ProgrammeFilter

        return list(filters(ProgrammeFilter(limit=10**6, offset=0)) or [])
    return []


def _normalise_with_mapping(name: str, mapping: Any) -> tuple[str, str]:
    """The (grouping key, alias-canonical label) a row's name would get today.

    Numbering stripping included: this is what makes the row-numbering
    duplicates fold and stale pre-W9 keys rekey.
    """
    from fmd.ingest.programmes import normalise_programme_name

    match = normalise_programme_name(name, mapping)
    return match.key, match.label


__all__ = [
    "MAX_EVIDENCE_PER_CANDIDATE",
    "ManualDecisionOutcome",
    "ProgrammeScoper",
    "RequalifyEntry",
    "RequalifyResult",
    "RepoDecisionStore",
    "SCOPE_UNCERTAIN_REASON",
    "ScopeInput",
    "build_scoper",
    "evidence_from_claims",
    "record_manual_decision",
    "requalify_programmes",
    "scoper_for_database",
]
