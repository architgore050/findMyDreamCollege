"""Stage 1 tests: normalisation and blocking keys (offline, data-driven)."""

from __future__ import annotations

from er_support import lexicon

from fmd.er.config import BlockingConfig, Lexicon
from fmd.er.normalise import (
    blocking_key,
    detect_script,
    normalise_name,
    normalise_text,
    soundex,
)


def test_nfkc_casefold_and_punctuation() -> None:
    normalised = normalise_name("  Universit\u00e9  of  Paris!! ", lexicon())
    assert normalised.tokens == ("universite", "paris")
    assert normalised.text == "universite paris"
    assert normalised.script == "latin"


def test_legal_suffixes_and_abbreviations_are_stripped_from_data() -> None:
    lex = lexicon()
    abbreviation = normalise_name("Govt. Medical Instt.", lex)
    full = normalise_name("Government Medical College", lex)
    assert abbreviation.tokens == full.tokens == ("government", "medical")
    # Without the lexicon the abbreviation stays literal: nothing is hardcoded.
    bare = normalise_name(
        "Govt. Medical Instt.",
        Lexicon(abbreviations={}, acronyms={}, legal_phrases=()),
    )
    assert "govt" in bare.tokens


def test_possessive_and_stopwords() -> None:
    normalised = normalise_name("King George's Medical University", lexicon())
    assert normalised.tokens == ("king", "george", "medical")
    assert "s" not in normalised.tokens


def test_acronym_expansion_comes_from_injected_lexicon() -> None:
    lex = lexicon()
    acronym = normalise_name("KGMU", lex)
    full = normalise_name("King George's Medical University", lex)
    assert acronym.tokens == full.tokens
    assert "kgmu" in acronym.acronyms


def test_custom_acronym_lexicon_is_data_not_code() -> None:
    custom = Lexicon.from_mapping(
        {"acronyms": {"xyz": "alpha beta"}, "legal_phrases": [], "stopwords": []}
    )
    assert normalise_name("XYZ", custom).tokens == ("alpha", "beta")
    assert normalise_name("IIT", lexicon()).tokens == ("indian", "technology")


def test_devanagari_hook_without_dependency_is_recorded_not_guessed() -> None:
    text = "किंग जॉर्ज मेडिकल यूनिवर्सिटी"
    assert detect_script(text) == "devanagari"
    normalised = normalise_name(text, lexicon())
    assert normalised.script == "devanagari"
    assert "devanagari_untransliterated" in normalised.notes
    assert normalised.transliterated is False


def test_devanagari_hook_applies_injected_transliterator() -> None:
    text = "किंग जॉर्ज मेडिकल यूनिवर्सिटी"

    def transliterate(value: str) -> str:
        assert value == text
        return "King George Medical University"

    normalised = normalise_name(text, lexicon(), transliterate=transliterate)
    assert normalised.transliterated is True
    assert normalised.tokens == ("king", "george", "medical")


def test_empty_name_after_stripping_is_flagged() -> None:
    normalised = normalise_name("University of", lexicon())
    assert "empty_after_normalisation" in normalised.notes
    assert normalised.tokens


def test_blocking_key_is_deterministic_and_uses_state_token() -> None:
    first = blocking_key(
        "King George's Medical University", "Uttar Pradesh", "226003", lexicon=lexicon()
    )
    second = blocking_key(
        "KING GEORGE MEDICAL UNIVERSITY", "uttar pradesh", "226003", lexicon=lexicon()
    )
    assert first == second
    assert any(key.startswith("state_token:uttar pradesh|king") for key in first)
    assert any(key.startswith("pincode:226003") for key in first)
    assert any(key.startswith("soundex:") for key in first)
    other_state = blocking_key(
        "King George's Medical University", "Bihar", "226003", lexicon=lexicon()
    )
    assert first != other_state


def test_blocking_key_without_state_still_produces_name_keys() -> None:
    keys = blocking_key("Delhi University", None, None, lexicon=lexicon())
    assert keys == tuple(dict.fromkeys(keys))
    assert any(key.startswith("token:delhi") for key in keys)


def test_soundex_is_ascii_only_and_stable() -> None:
    assert soundex("King George") == soundex("king george")
    assert soundex("किंग") == ""
    assert len(soundex("Delhi", 4)) == 4


def test_normalise_text_keeps_digits_and_scripts() -> None:
    assert normalise_text("  AIIMS, New Delhi-110029 ", lexicon()) == "aiims new delhi 110029"


def test_pincode_is_not_required() -> None:
    keys = blocking_key(
        "Some Institute", config=BlockingConfig(use_pincode_soundex=False), lexicon=lexicon()
    )
    assert not any(key.startswith("pincode") for key in keys)
