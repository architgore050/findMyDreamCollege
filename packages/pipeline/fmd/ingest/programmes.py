"""Materialise programme rows from verified claims (pure, deterministic).

This is the link scoring was missing: claims existed but nothing owned them, so
``programmes`` stayed empty and no row could be ranked. The rules here are the
whole contract:

* only claims that are **verified** (``verification_status == 'supported'``),
  not superseded, and not ``method='modelled'`` become programme attributes;
* a claim is attached to the programme named by the *same institution's*
  ``*.programme_name`` claim — programme name is the grouping key;
* every attribute records the claim id that produced it (``field_sources``);
* no claim ⇒ the column stays ``NULL`` and the map has no entry. Seats and
  duration are never backfilled from a default;
* names are collapsed through ER normalisation plus a registry-declared alias
  table (``params.programme_mapping.name_aliases``). Two names that are merely
  *similar* stay separate and are flagged ``needs_review`` — collapsing them
  would merge two real programmes.

Nothing here reads the network, the database or a clock. The orchestration
stage in :mod:`fmd.orchestrate.ingest` owns persistence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from fmd.er.normalise import normalise_name

#: Predicate vocabulary is data. The defaults mirror the committed registry so
#: offline tests and one-off scripts work without a config dir; production
#: always reads ``params.programme_mapping`` from the institution-site source.
DEFAULT_PROGRAMME_MAPPING: dict[str, Any] = {
    "name_predicates": [
        "curriculum.programme_name",
        "seats.programme_name",
        "accreditation.programme_name",
    ],
    "duration_month_predicates": ["curriculum.duration_months"],
    "internship_month_predicates": ["curriculum.internship_months"],
    "seats_predicates": ["seats.total"],
    "medium_predicates": ["curriculum.medium", "programme.medium"],
    "affiliated_university_predicates": [
        "curriculum.affiliated_university",
        "accreditation.affiliated_university",
    ],
    "recognition_predicates": ["accreditation.recognition_status", "recognition.status"],
    "credential_default": "degree",
    "credential_rules": [
        {"level": "certificate", "terms": ["certificate"]},
        {"level": "diploma", "terms": ["diploma", "dmlt"]},
        {"level": "lateral_entry", "terms": ["lateral entry", "lateral"]},
        {
            "level": "degree",
            "terms": [
                "bachelor",
                "b sc",
                "bsc",
                "b mls",
                "bmls",
                "b mlt",
                "bmlt",
                "degree",
            ],
        },
    ],
    "recognition_values": {
        "recognised": ["recognised", "recognized"],
        "conditional": ["conditional", "provisional"],
        "unrecognised": [
            "unrecognised",
            "unrecognized",
            "not recognised",
            "not recognized",
            "not approved",
        ],
    },
    # MLT/MLS is a domain vocabulary, not a country/field branch. Aliases are
    # intentionally conservative: MLT and MLS share several words but are kept
    # as different canonical programmes (uncertain collapse ⇒ review, not merge).
    "name_aliases": [
        {
            "key": "bsc mlt",
            "label": "B.Sc. Medical Laboratory Technology",
            "aliases": [
                "B.Sc. MLT",
                "BSc MLT",
                "B.Sc. Medical Laboratory Technology",
                "BSc Medical Laboratory Technology",
                "Bachelor of Medical Laboratory Technology",
                "Bachelor of Medical Laboratory Technology (BMLT)",
                "BMLT",
            ],
        },
        {
            "key": "bsc mls",
            "label": "Bachelor of Medical Laboratory Sciences",
            "aliases": [
                "Bachelor of Medical Laboratory Sciences",
                "Bachelor of Medical Laboratory Sciences (B.MLS)",
                "Bachelor of Medical Laboratory Sciences(B.MLS)",
                "B.Sc. Medical Laboratory Science",
                "BSc Medical Laboratory Science",
                "B.MLS",
                "BMLS",
            ],
        },
        {
            "key": "dmlt",
            "label": "Diploma in Medical Laboratory Technology",
            "aliases": [
                "DMLT",
                "Diploma in Medical Laboratory Technology",
                "Diploma in Medical Laboratory Technician",
            ],
        },
    ],
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clean(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass(frozen=True)
class NameMatch:
    """How one programme name normalises and whether an alias proved it."""

    raw: str
    key: str
    label: str
    matched_alias: bool


@dataclass(frozen=True)
class ProgrammeMapping:
    name_predicates: tuple[str, ...]
    duration_month_predicates: tuple[str, ...]
    internship_month_predicates: tuple[str, ...]
    seats_predicates: tuple[str, ...]
    medium_predicates: tuple[str, ...]
    affiliated_university_predicates: tuple[str, ...]
    recognition_predicates: tuple[str, ...]
    credential_default: str
    credential_rules: tuple[tuple[str, tuple[str, ...]], ...]
    recognition_values: Mapping[str, tuple[str, ...]]
    aliases: Mapping[str, tuple[str, str]]  # normalised alias -> (key, label)

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> ProgrammeMapping:
        block = (params or {}).get("programme_mapping")
        raw = dict(DEFAULT_PROGRAMME_MAPPING)
        if isinstance(block, Mapping):
            for key, value in block.items():
                if (
                    key in {"name_aliases", "credential_rules", "recognition_values"}
                    or value is not None
                ):
                    raw[key] = value
        aliases: dict[str, tuple[str, str]] = {}
        for entry in raw.get("name_aliases") or ():
            if not isinstance(entry, Mapping):
                continue
            key = _text(entry.get("key")).casefold()
            label = _text(entry.get("label")) or key
            if not key:
                continue
            names = [entry.get("label"), *(entry.get("aliases") or ())]
            for name in names:
                normalised = normalise_name(_text(name)).text if _text(name) else ""
                if normalised:
                    aliases.setdefault(normalised, (key, label))
        rules: list[tuple[str, tuple[str, ...]]] = []
        for rule in raw.get("credential_rules") or ():
            if not isinstance(rule, Mapping):
                continue
            level = _text(rule.get("level"))
            terms = tuple(
                _text(term).casefold() for term in (rule.get("terms") or ()) if _text(term)
            )
            if level and terms:
                rules.append((level, terms))
        recognition: dict[str, tuple[str, ...]] = {}
        for status, values in (raw.get("recognition_values") or {}).items():
            recognition[str(status)] = tuple(_text(value).casefold() for value in values or ())
        return cls(
            name_predicates=_tuple(raw.get("name_predicates")),
            duration_month_predicates=_tuple(raw.get("duration_month_predicates")),
            internship_month_predicates=_tuple(raw.get("internship_month_predicates")),
            seats_predicates=_tuple(raw.get("seats_predicates")),
            medium_predicates=_tuple(raw.get("medium_predicates")),
            affiliated_university_predicates=_tuple(raw.get("affiliated_university_predicates")),
            recognition_predicates=_tuple(raw.get("recognition_predicates")),
            credential_default=_text(raw.get("credential_default")) or "degree",
            credential_rules=tuple(rules),
            recognition_values=recognition,
            aliases=aliases,
        )

    @classmethod
    def from_source(cls, source: Any) -> ProgrammeMapping:
        from fmd.discover.base import params_from_source

        return cls.from_params(params_from_source(source))


def _tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in (value or ()) if str(item).strip())


# --------------------------------------------------------------------- naming


def normalise_programme_name(name: str, mapping: ProgrammeMapping) -> NameMatch:
    """ER-normalise a programme name, then resolve it through the alias table."""
    raw = _text(name)
    normalised = normalise_name(raw).text
    alias = mapping.aliases.get(normalised)
    if alias is not None:
        key, label = alias
        return NameMatch(raw=raw, key=key, label=label, matched_alias=True)
    return NameMatch(raw=raw, key=normalised, label=raw, matched_alias=False)


def _tokens(key: str) -> frozenset[str]:
    return frozenset(token for token in key.split() if token)


def names_are_similar(a: str, b: str) -> bool:
    """True when two *different* keys might name the same programme.

    Deliberately conservative: it never merges, it only raises ``needs_review``.
    Equal keys are handled before this is called.
    """
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return False
    shared = left & right
    if len(shared) >= 3:
        return True
    union = left | right
    if union and len(shared) / len(union) >= 0.6:
        return True
    # Short canonical keys ("bsc mlt" / "bsc mls") differ by one character while
    # naming adjacent qualifications: that is uncertainty, so flag, never merge.
    return SequenceMatcher(None, a, b).ratio() >= 0.8


def credential_level_for(name: str, mapping: ProgrammeMapping) -> str:
    """Classify the credential from the programme name using configured terms.

    Only word-boundary term matches count; when nothing matches the declared
    default is used and the caller records ``credential_level_defaulted``.
    """
    key = normalise_name(name).text
    padded = f" {key} "
    for level, terms in mapping.credential_rules:
        for term in terms:
            if f" {term} " in padded:
                return level
    return mapping.credential_default


def credential_was_defaulted(name: str, mapping: ProgrammeMapping) -> bool:
    key = normalise_name(name).text
    padded = f" {key} "
    return not any(
        f" {term} " in padded for _level, terms in mapping.credential_rules for term in terms
    )


# ------------------------------------------------------------------- values


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value.is_integer() else None
    text = _text(value).replace(",", "")
    match = re.match(r"^(-?\d+)(?:\.0+)?$", text)
    if match is None:
        return None
    number = int(match.group(1))
    return number if number >= 0 else None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _text(value).replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _unit(claim: Any) -> str:
    return (str(_get(claim, "unit", "") or "")).strip().casefold()


@dataclass(frozen=True)
class ProgrammeDraft:
    """One programme row plus the claim id behind every set attribute."""

    institution_id: int
    name: str
    normalised_name: str
    credential_level: str
    source_family: str
    duration_months: int | None = None
    internship_months: float | None = None
    seats: int | None = None
    medium: str | None = None
    affiliated_university_name: str | None = None
    affiliated_university_id: int | None = None
    recognition_status: str = "unknown"
    field_sources: Mapping[str, str] = field(default_factory=dict)
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()

    def with_affiliation(
        self, entity_id: int | None, *, review: str | None = None
    ) -> ProgrammeDraft:
        """Attach the resolved affiliating-university entity id.

        ``field_sources`` keeps the *claim id* that named the university; the
        resolved entity id is a separate fact and is never invented. An
        unresolvable name leaves both NULL and raises ``needs_review``.
        """
        reason = review
        if entity_id is None and self.affiliated_university_name:
            reason = "affiliated_university_unresolved"
        reasons = tuple(dict.fromkeys([*self.review_reasons, *([reason] if reason else [])]))
        return ProgrammeDraft(
            institution_id=self.institution_id,
            name=self.name,
            normalised_name=self.normalised_name,
            credential_level=self.credential_level,
            source_family=self.source_family,
            duration_months=self.duration_months,
            internship_months=self.internship_months,
            seats=self.seats,
            medium=self.medium,
            affiliated_university_name=self.affiliated_university_name,
            affiliated_university_id=entity_id,
            recognition_status=self.recognition_status,
            field_sources=dict(self.field_sources),
            needs_review=self.needs_review or reason is not None,
            review_reasons=reasons,
        )


def _eligible(claim: Any) -> bool:
    if str(_get(claim, "verification_status", "") or "") != "supported":
        return False
    if _get(claim, "superseded_by", None) is not None:
        return False
    if str(_get(claim, "method", "") or "") == "modelled":
        return False
    role = _get(claim, "document_role", None)
    return role in (None, "evidence")


def _sort_key(claim: Any) -> tuple[float, int]:
    """Highest confidence wins; claim id breaks ties so the choice is stable."""
    confidence = _get(claim, "confidence", 0.0) or 0.0
    return (float(confidence), int(_get(claim, "id", 0) or 0))


def _latest(claims: Sequence[Any], predicates: Iterable[str]) -> Any | None:
    """Newest supported claim for the first predicate that has one."""
    for predicate in predicates:
        matches = [claim for claim in claims if _get(claim, "predicate") == predicate]
        if matches:
            return max(matches, key=_sort_key)
    return None


def _normalise_recognition(value: Any, mapping: ProgrammeMapping) -> str:
    text = _text(value).casefold()
    if not text:
        return "unknown"
    for status, values in mapping.recognition_values.items():
        if any(value_text == text for value_text in values):
            return status
    return "unknown"


def programme_drafts(
    claims: Iterable[Any],
    mapping: ProgrammeMapping,
    *,
    resolve_institution: Any | None = None,
) -> tuple[list[ProgrammeDraft], list[dict[str, Any]]]:
    """Turn supported claims into drafts plus per-group rejections.

    ``claims`` may be claim records or mappings. ``document_role`` is optional;
    when present, anything other than ``evidence`` is refused so a navigation
    or inventory document can never create a programme.
    """
    eligible = [claim for claim in claims if _eligible(claim)]
    grouped: dict[int, list[Any]] = {}
    rejected: list[dict[str, Any]] = []
    for claim in eligible:
        entity_id = _get(claim, "entity_id")
        if entity_id is None:
            rejected.append({"claim_id": _get(claim, "id"), "reason": "no_entity"})
            continue
        if resolve_institution is not None and resolve_institution(int(entity_id)) is None:
            rejected.append({"claim_id": _get(claim, "id"), "reason": "entity_unresolvable"})
            continue
        grouped.setdefault(int(entity_id), []).append(claim)

    drafts: list[ProgrammeDraft] = []
    for entity_id in sorted(grouped):
        entity_claims = grouped[entity_id]
        by_key: dict[str, Any] = {}
        for name_claim in sorted(
            (
                claim
                for claim in entity_claims
                if _get(claim, "predicate") in mapping.name_predicates
                and _clean(_get(claim, "value"))
            ),
            key=_sort_key,
        ):
            raw_name = _clean(_get(name_claim, "value"))
            assert raw_name is not None
            match = normalise_programme_name(raw_name, mapping)
            by_key[match.key] = name_claim
        if not by_key:
            reason = (
                "no_programme_name_claim"
                if not any(_get(c, "predicate") in mapping.name_predicates for c in entity_claims)
                else "blank_programme_name"
            )
            rejected.append({"entity_id": entity_id, "reason": reason})
            continue
        multiple = len(by_key) > 1
        for key, name_claim in sorted(by_key.items()):
            drafts.append(
                _draft_for_name(
                    entity_id,
                    key,
                    name_claim,
                    entity_claims,
                    mapping,
                    multiple=multiple,
                )
            )

    return _flag_similar_names(drafts), rejected


def _qualifier_programme_key(claim: Any, mapping: ProgrammeMapping) -> str | None:
    qualifier = _get(claim, "qualifier", None)
    if not isinstance(qualifier, Mapping):
        return None
    name = _clean(qualifier.get("programme_name") or qualifier.get("programme"))
    if not name:
        return None
    return normalise_programme_name(name, mapping).key


def _attributable(
    claims: Sequence[Any], key: str, mapping: ProgrammeMapping, *, multiple: bool
) -> list[Any]:
    """Claims that may describe *this* programme.

    With one programme name at an institution every claim is attributable. With
    several, only a claim whose qualifier names the programme can be attached;
    the rest stay NULL rather than being assigned to a guessed programme.
    """
    if not multiple:
        return list(claims)
    return [claim for claim in claims if _qualifier_programme_key(claim, mapping) == key]


def _draft_for_name(
    entity_id: int,
    key: str,
    name_claim: Any,
    entity_claims: Sequence[Any],
    mapping: ProgrammeMapping,
    *,
    multiple: bool,
) -> ProgrammeDraft:
    raw_name = _clean(_get(name_claim, "value")) or ""
    match = normalise_programme_name(raw_name, mapping)
    sources: dict[str, str] = {str(_get(name_claim, "predicate")): str(_get(name_claim, "id"))}
    reasons: list[str] = []

    if credential_was_defaulted(raw_name, mapping):
        reasons.append("credential_level_defaulted")
    credential = credential_level_for(raw_name, mapping)

    candidates = _attributable(entity_claims, key, mapping, multiple=multiple)
    if multiple:
        reasons.append("attribute_programme_ambiguous")
    duration = _latest(candidates, mapping.duration_month_predicates)
    duration_months = None
    if duration is not None and _unit(duration) in {"month", "months", ""}:
        duration_months = as_int(_get(duration, "value"))
        if duration_months is not None and duration_months > 0:
            sources[str(_get(duration, "predicate"))] = str(_get(duration, "id"))
    internship = _latest(candidates, mapping.internship_month_predicates)
    internship_months = None
    if internship is not None and _unit(internship) in {"month", "months"}:
        internship_months = as_float(_get(internship, "value"))
        if internship_months is not None and internship_months >= 0:
            sources[str(_get(internship, "predicate"))] = str(_get(internship, "id"))
    seats_claim = _latest(candidates, mapping.seats_predicates)
    seats = None
    if seats_claim is not None and _unit(seats_claim) in {"count", "seats", ""}:
        seats = as_int(_get(seats_claim, "value"))
        if seats is not None:
            sources[str(_get(seats_claim, "predicate"))] = str(_get(seats_claim, "id"))
    medium_claim = _latest(candidates, mapping.medium_predicates)
    medium = _clean(_get(medium_claim, "value")) if medium_claim is not None else None
    if medium:
        sources[str(_get(medium_claim, "predicate"))] = str(_get(medium_claim, "id"))
    affiliation_claim = _latest(candidates, mapping.affiliated_university_predicates)
    affiliation = (
        _clean(_get(affiliation_claim, "value")) if affiliation_claim is not None else None
    )
    if affiliation:
        sources[str(_get(affiliation_claim, "predicate"))] = str(_get(affiliation_claim, "id"))
    recognition_claim = _latest(candidates, mapping.recognition_predicates)
    recognition = "unknown"
    if recognition_claim is not None:
        recognition = _normalise_recognition(_get(recognition_claim, "value"), mapping)
        if recognition == "unknown":
            reasons.append("recognition_value_ambiguous")
        else:
            sources[str(_get(recognition_claim, "predicate"))] = str(_get(recognition_claim, "id"))

    return ProgrammeDraft(
        institution_id=entity_id,
        name=match.label,
        normalised_name=match.key,
        credential_level=credential,
        source_family=str(_get(name_claim, "predicate", "")).split(".", 1)[0],
        duration_months=duration_months,
        internship_months=internship_months,
        seats=seats,
        medium=medium,
        affiliated_university_name=affiliation,
        recognition_status=recognition,
        field_sources={source_key: value for source_key, value in sources.items() if value},
        review_reasons=tuple(dict.fromkeys(reasons)),
        needs_review=bool(reasons),
    )


def _flag_similar_names(drafts: list[ProgrammeDraft]) -> list[ProgrammeDraft]:
    """Mark (never merge) same-institution names that may be one programme."""
    by_institution: dict[int, list[ProgrammeDraft]] = {}
    for draft in drafts:
        by_institution.setdefault(draft.institution_id, []).append(draft)
    flagged: list[ProgrammeDraft] = []
    for draft in drafts:
        reason = None
        for other in by_institution.get(draft.institution_id, ()):
            if other is draft or other.normalised_name == draft.normalised_name:
                continue
            if names_are_similar(draft.normalised_name, other.normalised_name):
                reason = f"possible_duplicate_programme_name:{other.name}"
                break
        if reason is None:
            flagged.append(draft)
            continue
        reasons = tuple(dict.fromkeys([*draft.review_reasons, reason]))
        flagged.append(
            ProgrammeDraft(
                institution_id=draft.institution_id,
                name=draft.name,
                normalised_name=draft.normalised_name,
                credential_level=draft.credential_level,
                source_family=draft.source_family,
                duration_months=draft.duration_months,
                internship_months=draft.internship_months,
                seats=draft.seats,
                medium=draft.medium,
                affiliated_university_name=draft.affiliated_university_name,
                affiliated_university_id=draft.affiliated_university_id,
                recognition_status=draft.recognition_status,
                field_sources=draft.field_sources,
                needs_review=True,
                review_reasons=reasons,
            )
        )
    return flagged


__all__ = [
    "DEFAULT_PROGRAMME_MAPPING",
    "NameMatch",
    "ProgrammeDraft",
    "ProgrammeMapping",
    "as_float",
    "as_int",
    "credential_level_for",
    "credential_was_defaulted",
    "names_are_similar",
    "normalise_programme_name",
    "programme_drafts",
]
