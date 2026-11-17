"""Registry-backed predicate resolution for scoring.

Extraction emits namespaced predicates (``seats.total``,
``curriculum.internship_months``, ...). Scoring never guesses what one means:
:class:`PredicateRegistry` reads ``config/predicates.yaml`` and answers three
questions deterministically:

* which sub-score component a predicate feeds (or ``None`` for evidence-only
  inputs such as fees, exam dates and programme names);
* what unit the claim must carry, and whether a denominator / printed period is
  mandatory (a mismatch is *rejected*, never coerced);
* which configured predicates are still unknown to the registry so the caller
  can report them instead of silently dropping evidence.

Aliases are declared legacy spellings only; they resolve to a canonical row and
carry ``alias_of`` so callers can surface the deprecation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from fmd.config.models import PredicatesConfig
from fmd.score.evidence import ClaimView, claim_view

# Unit spellings the extraction / derived-claim layers actually emit.
_UNIT_ALIASES: dict[str, frozenset[str]] = {
    "count": frozenset(
        {"count", "number", "integer", "int", "students", "seats", "persons", "people", "units"}
    ),
    "money": frozenset(
        {
            "money",
            "currency",
            "inr",
            "rs",
            "usd",
            "gbp",
            "eur",
            "aud",
            "cad",
            "₹",
            "$",
        }
    ),
    "hours": frozenset({"hour", "hours", "hr", "hrs"}),
    "months": frozenset({"month", "months", "mo"}),
    "years": frozenset({"year", "years", "yr", "yrs"}),
    "percent": frozenset(
        {"%", "percent", "pct", "percentage", "rate", "fraction", "ratio", "share"}
    ),
    "ratio": frozenset({"ratio", "fraction", "unitless", "index", "multiple", "x"}),
    "grade": frozenset({"grade", "gpa", "letter"}),
    "boolean": frozenset({"boolean", "bool", "yes_no", "flag"}),
    "date": frozenset({"date", "datetime", "iso8601"}),
    "text": frozenset({"text", "string", "name", "label"}),
}

# Free-form kinds carry no machine unit in extraction; a stray unit is not a
# contradiction and must not cause rejection.
_UNIT_AGNOSTIC = frozenset({"text", "date"})

_PERIOD_KEYS = ("period", "fee_period", "salary_period", "pay_period")


def normalise_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    text = str(unit).strip().lower()
    return text or None


def unit_matches(expected: str | None, unit: Any) -> bool:
    """True when ``unit`` can carry a value of ``expected`` kind.

    Absent units are accepted (extraction may omit them for booleans and
    text); a *present* contradicting unit is a hard rejection.
    """

    if expected is None or expected in _UNIT_AGNOSTIC:
        return True
    actual = normalise_unit(unit)
    if actual is None:
        return True
    if actual == expected:
        return True
    return actual in _UNIT_ALIASES.get(expected, frozenset())


def period_present(claim: ClaimView) -> bool:
    for key in _PERIOD_KEYS:
        value = claim.qualifier.get(key)
        if value is not None and str(value).strip():
            return True
    return False


@dataclass(frozen=True)
class PredicateRuleView:
    """Resolved, read-only view of one predicate row."""

    predicate: str
    component: str | None
    label: str
    unit_kind: str | None
    requires_denominator: bool
    requires_period: bool
    kind: str | None
    transform: str | None
    scale: float
    factor: float
    alias_of: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "component": self.component,
            "unit_kind": self.unit_kind,
            "requires_denominator": self.requires_denominator,
            "requires_period": self.requires_period,
            "scoring": None
            if self.kind is None
            else {
                "kind": self.kind,
                "transform": self.transform,
                "scale": self.scale,
                "factor": self.factor,
            },
            "alias_of": self.alias_of,
        }


class PredicateRegistryError(ValueError):
    """The predicate registry is missing or internally inconsistent."""


class PredicateRegistry:
    """Config-backed predicate map used by every sub-score."""

    __slots__ = ("_rules", "_aliases", "version")

    def __init__(
        self,
        rules: Mapping[str, PredicateRuleView],
        aliases: Mapping[str, str],
        version: str,
    ):
        self._rules = dict(rules)
        self._aliases = dict(aliases)
        self.version = version
        for alias, target in self._aliases.items():
            if target not in self._rules:
                raise PredicateRegistryError(
                    f"alias {alias!r} points at unknown predicate {target!r}"
                )

    @classmethod
    def from_config(cls, value: Any) -> PredicateRegistry:
        if value is None:
            raise PredicateRegistryError(
                "no predicate registry supplied; config/predicates.yaml is required"
            )
        if isinstance(value, PredicateRegistry):
            return value
        if isinstance(value, PredicatesConfig):
            raw = value.model_dump()
        elif isinstance(value, Mapping):
            raw = value
        elif hasattr(value, "model_dump"):
            raw = value.model_dump()
        else:
            raise PredicateRegistryError(
                f"unsupported predicate registry type {type(value).__name__}"
            )

        rows = raw.get("predicates")
        if not isinstance(rows, Mapping) or not rows:
            raise PredicateRegistryError(
                "predicate registry has no `predicates` map; scoring cannot map claims"
            )

        rules: dict[str, PredicateRuleView] = {}
        for predicate, spec in rows.items():
            if not isinstance(spec, Mapping):
                raise PredicateRegistryError(f"predicate {predicate!r} must be a mapping")
            scoring = spec.get("scoring") or {}
            rules[str(predicate)] = PredicateRuleView(
                predicate=str(predicate),
                component=spec.get("component"),
                label=str(spec.get("label") or predicate),
                unit_kind=spec.get("unit_kind"),
                requires_denominator=bool(spec.get("requires_denominator", False)),
                requires_period=bool(spec.get("requires_period", False)),
                kind=scoring.get("kind") if scoring else None,
                transform=scoring.get("transform") if scoring else None,
                scale=float(scoring.get("scale", 1.0) or 1.0) if scoring else 1.0,
                factor=float(scoring.get("factor", 1.0) or 1.0) if scoring else 1.0,
            )
        aliases_raw = raw.get("aliases") or {}
        if not isinstance(aliases_raw, Mapping):
            raise PredicateRegistryError("predicate registry `aliases` must be a mapping")
        aliases = {str(k): str(v) for k, v in aliases_raw.items()}
        return cls(rules, aliases, str(raw.get("version", "unversioned")))

    # ------------------------------------------------------------------ lookup

    def canonical_for(self, predicate: str) -> str | None:
        name = str(predicate)
        if name in self._rules:
            return name
        target = self._aliases.get(name)
        return target if target in self._rules else None

    def resolve(self, predicate: str) -> PredicateRuleView | None:
        target = self.canonical_for(predicate)
        if target is None:
            return None
        rule = self._rules[target]
        if target == str(predicate):
            return rule
        return PredicateRuleView(
            predicate=rule.predicate,
            component=rule.component,
            label=rule.label,
            unit_kind=rule.unit_kind,
            requires_denominator=rule.requires_denominator,
            requires_period=rule.requires_period,
            kind=rule.kind,
            transform=rule.transform,
            scale=rule.scale,
            factor=rule.factor,
            alias_of=str(predicate),
        )

    def rules_for(self, component: str) -> tuple[PredicateRuleView, ...]:
        """Deterministic predicate rows feeding ``component`` (sorted by name)."""

        return tuple(
            sorted(
                (
                    rule
                    for rule in self._rules.values()
                    if rule.component == component and rule.kind is not None
                ),
                key=lambda rule: rule.predicate,
            )
        )

    def aliases_for(self, predicate: str) -> tuple[str, ...]:
        """Declared legacy spellings that resolve to ``predicate``."""

        return tuple(
            sorted(alias for alias, target in self._aliases.items() if target == predicate)
        )

    def component_predicates(self) -> dict[str, tuple[str, ...]]:
        components: dict[str, list[str]] = {}
        for rule in self._rules.values():
            if rule.component is not None:
                components.setdefault(rule.component, []).append(rule.predicate)
        return {key: tuple(sorted(values)) for key, values in sorted(components.items())}

    # ------------------------------------------------------------- validation

    def rejects(self, claim: ClaimView, rule: PredicateRuleView) -> str | None:
        """Why a claim cannot feed ``rule``'s component, or ``None``."""

        if not unit_matches(rule.unit_kind, claim.unit):
            return (
                f"unit {claim.unit!r} does not match {rule.unit_kind!r} "
                f"for {rule.predicate}"
            )
        if rule.requires_period and not period_present(claim):
            return f"period required for {rule.predicate}"
        if rule.requires_denominator:
            from fmd.score.evidence import denominator, is_rate_claim

            if is_rate_claim(claim) and denominator(claim) is None:
                return f"denominator required for {rule.predicate}"
        return None

    def unknown_predicates(self, claims: Iterable[Any]) -> tuple[str, ...]:
        """Predicates no registry row (canonical or alias) can explain."""

        unknown: set[str] = set()
        for raw in claims:
            view = claim_view(raw)
            if self.canonical_for(view.predicate) is None:
                unknown.add(view.predicate)
        return tuple(sorted(unknown))

    def alias_predicates(self, claims: Iterable[Any]) -> tuple[str, ...]:
        aliases: set[str] = set()
        for raw in claims:
            view = claim_view(raw)
            if view.predicate in self._aliases:
                aliases.add(view.predicate)
        return tuple(sorted(aliases))

    def all_predicates(self) -> tuple[str, ...]:
        return tuple(sorted(self._rules))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "predicates": {
                key: self._rules[key].as_dict() for key in sorted(self._rules)
            },
            "aliases": dict(sorted(self._aliases.items())),
        }
