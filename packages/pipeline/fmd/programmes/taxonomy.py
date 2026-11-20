"""Tier 1 — deterministic scoping rules from ``config/programmes.yaml``.

Fast, high-precision, no model. A country's taxonomy carries include patterns,
exclude patterns and aliases; ``exclude`` beats ``include``. A name that
matches no rule is *not* decided here — it falls through to the semantic tier
(:mod:`fmd.programmes.classifier`), which is the whole point (14 §B2): rules
decide the obvious, evidence decides the rest.

Declared-but-empty country blocks (AU, US, ...) are valid stubs: they defer
every name to the semantic tier and never admit anything on a guessed pattern.
A country missing from the map falls back to ``default`` (also empty) and the
decision carries the ``country_taxonomy_missing`` flag.

Aliases canonicalise the name *before* matching (and before the semantic cache
key is computed), so ``"B.Sc. MLT"`` and its spelled-out form reach the same
rules and the same decision. The registry aliases live in
``config/programmes.yaml``; callers may inject extra aliases (the
materialisation mapping's committed vocabulary, manual decisions) which merge
over them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from fmd.config.models import CountryTaxonomy, ProgrammesConfig
from fmd.programmes.normalise import normalise_programme_label

#: Flag recorded when a country has no declared taxonomy (or the institution's
#: country is unknown) so the fallback to ``default`` is visible in audits.
COUNTRY_TAXONOMY_MISSING = "country_taxonomy_missing"

#: Flag for a declared-but-empty taxonomy: the country exists, the rules
#: intentionally do not, so every name goes to the semantic tier.
COUNTRY_TAXONOMY_STUB = "country_taxonomy_stub"

#: Review reason appended to a row the scoping tiers could not decide:
#: `uncertain` is excluded from the ranking and queued for a human.
SCOPE_UNCERTAIN_REASON = "scope_uncertain"


@dataclass(frozen=True)
class ScopeDecision:
    """The outcome of one scoping decision, whatever tier made it.

    ``in_scope`` is the gate scoring reads. ``uncertain`` rows are
    ``in_scope=False`` AND ``needs_review=True``: never ranked until a human
    decides, and never silently dropped either.
    """

    in_scope: bool
    tier: str  # rule | semantic | manual
    reason: str
    needs_review: bool = False
    evidence: tuple[str, ...] = ()
    confidence: float | None = None
    model: str | None = None
    prompt_version: str | None = None
    cached: bool = False
    flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "in_scope": self.in_scope,
            "tier": self.tier,
            "reason": self.reason,
            "needs_review": self.needs_review,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "cached": self.cached,
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class RuleVerdict:
    """What one country's rule block said about one normalised name."""

    decided: bool
    in_scope: bool = False
    reason: str = ""
    matched: tuple[str, ...] = ()
    normalised_name: str = ""
    matched_alias: bool = False
    flags: tuple[str, ...] = ()


class Taxonomy:
    """Tier-1 matcher over the loaded ``config/programmes.yaml``.

    Patterns are pre-compiled once; matching is pure regex search on the
    normalised label. ``extra_aliases`` lets callers merge runtime alias
    tables (the materialisation mapping, manual decisions) over the registry
    block, so Tier 1 consumes every alias source there is.
    """

    def __init__(
        self,
        config: ProgrammesConfig,
        *,
        extra_aliases: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self.config = config
        self._includes: dict[str, tuple[re.Pattern[str], ...]] = {}
        self._excludes: dict[str, tuple[re.Pattern[str], ...]] = {}
        self._aliases: dict[str, dict[str, tuple[str, str]]] = {}
        for key in (*config.countries.keys(), "default"):
            taxonomy = config.default if key == "default" else config.countries[key]
            self._includes[key] = tuple(
                re.compile(pattern, re.IGNORECASE) for pattern in taxonomy.include_patterns
            )
            self._excludes[key] = tuple(
                re.compile(pattern, re.IGNORECASE) for pattern in taxonomy.exclude_patterns
            )
            merged = _alias_map(taxonomy)
            for normalised, entry in (extra_aliases or {}).items():
                merged.setdefault(normalised, entry)
            self._aliases[key] = merged
        self._stub_keys = frozenset(
            key
            for key, taxonomy in config.countries.items()
            if not taxonomy.include_patterns and not taxonomy.exclude_patterns
        )

    def country_key(self, country: str | None) -> tuple[str, bool]:
        """ ``(taxonomy key, fell_back_to_default)`` for a country code."""
        _, missing = self.config.taxonomy_for(country)
        return ("default" if missing else (country or "").strip().upper()), missing

    def resolve_alias(self, name: str, country: str | None) -> tuple[str, bool]:
        """Canonicalise one raw programme name through the country's aliases.

        Returns ``(normalised_label, matched_alias)``: the numbering-stripped,
        ER-normalised text, or — when a registry alias matched — the alias's
        canonical label in the same normalised form, so ``"B.Sc. MLT"`` and
        its spelled-out source produce identical rule input and identical
        semantic cache keys.
        """
        label = normalise_programme_label(name)
        key, _missing = self.country_key(country)
        alias = self._aliases.get(key, {}).get(label)
        if alias is not None:
            return normalise_programme_label(alias[1]), True
        return label, False

    def alias_map_for(self, country: str | None) -> dict[str, tuple[str, str]]:
        """The merged alias map for one country (normalised alias -> key, label)."""
        key, _missing = self.country_key(country)
        return dict(self._aliases.get(key, {}))

    def apply(
        self,
        *,
        raw_name: str,
        country: str | None,
    ) -> RuleVerdict:
        """Tier-1 decision for one raw name. Undecided names: ``decided=False``.

        The name is numbering-stripped, ER-normalised and alias-canonicalised
        here, so callers never pre-normalise and the alias canonical form and
        its spelled-out source cannot disagree.
        """
        normalised, matched_alias = self.resolve_alias(raw_name, country)
        key, missing = self.country_key(country)
        flags: list[str] = []
        if missing:
            flags.append(COUNTRY_TAXONOMY_MISSING)
        elif key in self._stub_keys:
            flags.append(COUNTRY_TAXONOMY_STUB)
        for pattern in self._excludes.get(key, ()):
            if pattern.search(normalised):
                return RuleVerdict(
                    decided=True,
                    in_scope=False,
                    reason=f"rule_exclude:{pattern.pattern}",
                    matched=(pattern.pattern,),
                    normalised_name=normalised,
                    matched_alias=matched_alias,
                    flags=tuple(flags),
                )
        for pattern in self._includes.get(key, ()):
            if pattern.search(normalised):
                return RuleVerdict(
                    decided=True,
                    in_scope=True,
                    reason=f"rule_include:{pattern.pattern}",
                    matched=(pattern.pattern,),
                    normalised_name=normalised,
                    matched_alias=matched_alias,
                    flags=tuple(flags),
                )
        return RuleVerdict(
            decided=False,
            normalised_name=normalised,
            matched_alias=matched_alias,
            flags=tuple(flags),
        )

    def definition_text(self) -> str:
        """The conceptual definition the classifier receives verbatim."""
        definition = self.config.definition
        domains = "; ".join(definition.domains)
        return f"{definition.statement.strip()} Working domains: {domains}."


def _alias_map(taxonomy: CountryTaxonomy) -> dict[str, tuple[str, str]]:
    aliases: dict[str, tuple[str, str]] = {}
    for entry in taxonomy.aliases:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "").strip().casefold()
        label = str(entry.get("label") or "").strip() or key
        if not key:
            continue
        names = [entry.get("label"), *(entry.get("aliases") or [])]
        for name in names:
            text = str(name or "").strip()
            if not text:
                continue
            aliases.setdefault(normalise_programme_label(text), (key, label))
    return aliases


__all__ = [
    "COUNTRY_TAXONOMY_MISSING",
    "COUNTRY_TAXONOMY_STUB",
    "SCOPE_UNCERTAIN_REASON",
    "RuleVerdict",
    "ScopeDecision",
    "Taxonomy",
]
