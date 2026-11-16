"""Stage 5 tests: alias mining from a fixture document, dedupe, span resolution."""

from __future__ import annotations

from er_support import lexicon, load_text

from fmd.er.aliases import mine_aliases, to_alias_creates

CANONICAL = "King George's Medical University"
DEVANAGARI = "किंग जॉर्ज मेडिकल यूनिवर्सिटी"


def _transliterate(value: str) -> str:
    if value == DEVANAGARI:
        return "King George Medical University"
    return value


def test_mines_acronym_former_name_and_local_name() -> None:
    document = load_text("document.txt")
    aliases = mine_aliases(
        document,
        canonical_name=CANONICAL,
        lexicon=lexicon(),
        transliterate=_transliterate,
    )
    by_type: dict[str, set[str]] = {}
    for alias in aliases:
        by_type.setdefault(alias.alias_type, set()).add(alias.alias)

    assert "KGMU" in by_type.get("acronym", set())
    assert "King George Medical College" in by_type.get("former_name", set())
    assert DEVANAGARI in by_type.get("local_name", set())
    assert "King George Medical University" in by_type.get("transliteration", set())
    assert all(0.0 < alias.confidence <= 1.0 for alias in aliases)


def test_aliases_are_sorted_by_confidence_then_name() -> None:
    aliases = mine_aliases(load_text("document.txt"), canonical_name=CANONICAL, lexicon=lexicon())
    ordered = sorted(aliases, key=lambda item: (-item.confidence, item.alias))
    assert aliases == ordered
    assert len({alias.alias for alias in aliases}) == len(aliases)


def test_canonical_name_is_never_emitted_as_its_own_alias() -> None:
    aliases = mine_aliases(load_text("document.txt"), canonical_name=CANONICAL, lexicon=lexicon())
    assert CANONICAL not in {alias.alias for alias in aliases}


def test_known_aliases_are_deduped() -> None:
    aliases = mine_aliases(
        load_text("document.txt"),
        canonical_name=CANONICAL,
        known_aliases=["KGMU", "King George Medical College"],
        lexicon=lexicon(),
    )
    names = {alias.alias for alias in aliases}
    assert "KGMU" not in names
    assert "King George Medical College" not in names


def test_marker_spans_resolve_to_the_quoted_text() -> None:
    document = load_text("document.txt")
    aliases = mine_aliases(document, canonical_name=CANONICAL, lexicon=lexicon())
    marker_aliases = [alias for alias in aliases if alias.marker]
    assert marker_aliases
    for alias in marker_aliases:
        assert document[alias.char_start : alias.char_end] == alias.alias
        assert alias.quote


def test_misspelling_is_retyped_with_low_confidence() -> None:
    document = "The college, also known as King George Medicl University, was renamed."
    aliases = mine_aliases(document, canonical_name=CANONICAL, lexicon=lexicon())
    misspellings = [alias for alias in aliases if alias.alias_type == "misspelling"]
    assert misspellings
    assert "King George Medicl University" in {alias.alias for alias in misspellings}


def test_local_name_is_kept_without_a_transliterator() -> None:
    aliases = mine_aliases(f"Notice\n{DEVANAGARI}\n", canonical_name=CANONICAL, lexicon=lexicon())
    assert DEVANAGARI in {alias.alias for alias in aliases}
    assert not any(alias.alias_type == "transliteration" for alias in aliases)


def test_alias_rows_match_the_schema_types() -> None:
    aliases = mine_aliases(load_text("document.txt"), canonical_name=CANONICAL, lexicon=lexicon())
    rows = to_alias_creates(7, aliases, source_id="ugc")
    assert rows
    allowed = {"acronym", "former_name", "transliteration", "misspelling", "local_name"}
    assert {row.alias_type for row in rows} <= allowed
    assert {row.entity_id for row in rows} == {7}
    assert {row.source_id for row in rows} == {"ugc"}
