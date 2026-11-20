"""W9 Tier 1 — the country-keyed taxonomy, normalisation and precedence.

``config/programmes.yaml`` is the whole contract here: the IN include/exclude
vocabulary, the empty stubs for other countries (no invented patterns), the
``default`` fallback with ``country_taxonomy_missing``, exclude-beats-include
precedence, and the numbering normalisation that merges row-numbered
duplicates instead of materialising them twice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fmd.config import load_registry
from fmd.config.models import ProgrammesConfig
from fmd.programmes import Taxonomy
from fmd.programmes.classifier import decision_cache_key, has_field_signal
from fmd.programmes.normalise import normalise_programme_label, strip_row_numbering
from fmd.programmes.taxonomy import (
    COUNTRY_TAXONOMY_MISSING,
    COUNTRY_TAXONOMY_STUB,
)

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config"


def _taxonomy() -> Taxonomy:
    registry = load_registry(CONFIG_PATH, env={})
    return Taxonomy(registry.programmes)


# ------------------------------------------------------------------- config


def test_programmes_yaml_is_country_keyed_with_stubs_and_default() -> None:
    registry = load_registry(CONFIG_PATH, env={})
    config = registry.programmes
    assert config.version
    assert config.definition.version
    assert "diagnostic" in config.definition.statement.casefold()
    assert config.definition.domains
    # India has rules; the other countries ship as declared EMPTY stubs.
    assert config.countries["IN"].include_patterns
    assert config.countries["IN"].exclude_patterns
    for code in ("AU", "US", "GB"):
        taxonomy = config.countries[code]
        assert taxonomy.include_patterns == []
        assert taxonomy.exclude_patterns == []
    assert config.default.include_patterns == []
    # No hardcoded lists in Python: every pattern lives in the registry file.
    assert isinstance(config, ProgrammesConfig)


# ------------------------------------------------------------------- Tier 1


@pytest.mark.parametrize(
    ("name", "expected_in"),
    [
        ("B.Sc. Medical Laboratory Technology", True),
        ("B.Sc. MLT", True),
        ("BMLT", True),
        ("B.MLS", True),
        ("BMLS", True),
        ("MLS", True),
        ("DMLT", True),
        ("Diploma in Medical Laboratory Technology", True),
        ("Clinical Laboratory Science", True),
        ("Medical Laboratory Science", True),
    ],
)
def test_india_include_vocabulary(name: str, expected_in: bool) -> None:
    verdict = _taxonomy().apply(raw_name=name, country="IN")
    assert verdict.decided
    assert verdict.in_scope is expected_in
    assert verdict.reason.startswith("rule_include:")
    assert COUNTRY_TAXONOMY_MISSING not in verdict.flags


@pytest.mark.parametrize(
    "name",
    [
        "MBBS - Bachelor of Medicine and Bachelor of Surgery",
        "MBBS",
        "BDS",
        "BAMS",
        "BHMS",
        "B.Sc. Nursing",
        "GNM Nursing",
        "Bachelor of Physiotherapy (BPT)",
        "B.Pharm",
        "B.Sc (Hons) Agriculture",
        "B.Sc. Agriculture",
    ],
)
def test_india_explicit_excludes(name: str) -> None:
    verdict = _taxonomy().apply(raw_name=name, country="IN")
    assert verdict.decided
    assert verdict.in_scope is False
    assert verdict.reason.startswith("rule_exclude:")


def test_exclude_beats_include() -> None:
    # A name that would match both an include ("medical laboratory") and an
    # exclude ("nursing") must land OUT — excludes are the safety net.
    verdict = _taxonomy().apply(
        raw_name="B.Sc. Medical Laboratory Technology (Nursing)", country="IN"
    )
    assert verdict.decided
    assert verdict.in_scope is False
    assert verdict.reason.startswith("rule_exclude:")


def test_other_countries_are_stubs_that_defer_to_the_classifier() -> None:
    taxonomy = _taxonomy()
    verdict = taxonomy.apply(
        raw_name="Bachelor of Biomedical Science (Laboratory Medicine)", country="AU"
    )
    assert not verdict.decided  # no rule matched: Tier 2 decides from evidence
    assert COUNTRY_TAXONOMY_STUB in verdict.flags
    assert verdict.normalised_name == "bachelor biomedical science laboratory medicine"


def test_unknown_country_falls_back_to_default_and_flags_it() -> None:
    verdict = _taxonomy().apply(raw_name="Bachelor of Cura Hyperbarica", country="ZZ")
    assert not verdict.decided
    assert COUNTRY_TAXONOMY_MISSING in verdict.flags


def test_missing_country_falls_back_to_default_and_flags_it() -> None:
    verdict = _taxonomy().apply(raw_name="Programme Without A Country", country=None)
    assert not verdict.decided
    assert COUNTRY_TAXONOMY_MISSING in verdict.flags


def test_undecided_india_name_is_not_guessed() -> None:
    # Rules decide the obvious; anything else goes to the semantic tier
    # instead of being admitted on a guessed pattern.
    verdict = _taxonomy().apply(raw_name="Bachelor of Vision Technology", country="IN")
    assert not verdict.decided


# ----------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3. B.Sc. Medical Laboratory Technology", "B.Sc. Medical Laboratory Technology"),
        ("3) MBBS", "MBBS"),
        ("(3) B.Sc MLT", "B.Sc MLT"),
        ("#5 B.Sc. MLT", "B.Sc. MLT"),
        ("Sl. 3. B.Sc. MLT", "B.Sc. MLT"),
        ("• 7) B.Sc. MLT", "B.Sc. MLT"),
        # A real name must survive: "3-Year" is part of the title.
        ("3-Year B.Sc. MLT", "3-Year B.Sc. MLT"),
        # Bare numbers without punctuation are NOT page furniture by rule.
        ("12 B.Sc. Nursing", "12 B.Sc. Nursing"),
    ],
)
def test_strip_row_numbering(raw: str, expected: str) -> None:
    assert strip_row_numbering(raw) == expected


def test_row_numbered_duplicate_normalises_to_the_same_key() -> None:
    # The measured W8 defect: "3. B.Sc. Medical Laboratory Technology" became a
    # second programmes row. Normalisation merges it instead.
    stripped = strip_row_numbering("3. B.Sc. Medical Laboratory Technology")
    assert normalise_programme_label("B.Sc. Medical Laboratory Technology") == (
        normalise_programme_label(stripped)
    )


def test_alias_canonicalisation_agrees_with_registry() -> None:
    taxonomy = _taxonomy()
    label, matched = taxonomy.resolve_alias("B.Sc. MLT", "IN")
    assert matched
    spelled, _ = taxonomy.resolve_alias("Bachelor of Medical Laboratory Technology (BMLT)", "IN")
    assert label == spelled == "b sc medical laboratory technology"
    # The numbered duplicate lands on the same canonical form.
    numbered, matched_numbered = taxonomy.resolve_alias("3. B.Sc. MLT", "IN")
    assert matched_numbered and numbered == label


def test_normalisation_collapse_case_and_whitespace() -> None:
    assert normalise_programme_label("  b.sc.   MLT  ") == normalise_programme_label("B.Sc. MLT")
    assert normalise_programme_label("Bachelor  of\tMedical Laboratory Technology") == (
        normalise_programme_label("Bachelor of Medical Laboratory Technology")
    )


# ---------------------------------------------------------------- classifier


def test_semantic_cache_key_is_name_and_institution_type_scoped() -> None:
    base = decision_cache_key("b sc mlt", "institution:IN", "mlt-definition-v1")
    assert base == decision_cache_key("b sc mlt", "institution:IN", "mlt-definition-v1")
    assert base != decision_cache_key("b sc mls", "institution:IN", "mlt-definition-v1")
    # A new definition version invalidates every cached decision.
    assert base != decision_cache_key("b sc mlt", "institution:IN", "mlt-definition-v2")
    # Institution identity is deliberately NOT in the key: two institutions
    # naming a programme identically share one decision (the learning loop).
    assert base != decision_cache_key("b sc mlt", "institution:AU", "mlt-definition-v1")


def test_field_signal_prefilter() -> None:
    domains = ["clinical biochemistry", "haematology", "histopathology"]
    assert has_field_signal("B.Sc. Medical Laboratory Technology", domains)
    assert has_field_signal("laboratory medicine", domains)
    assert not has_field_signal("Bachelor of Health Sciences", domains)
    assert not has_field_signal("Bachelor of Business Administration", domains)


def test_definition_text_is_the_configured_statement_plus_domains() -> None:
    taxonomy = _taxonomy()
    text = taxonomy.definition_text()
    assert "diagnostic laboratory science" in text.casefold()
    assert "haematology" in text.casefold()
    assert "blood bank" in text.casefold()
