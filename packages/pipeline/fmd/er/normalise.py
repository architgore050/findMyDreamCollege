"""Stage 1 — normalisation (brief §10).

Unicode NFKC → casefold → punctuation/possessive removal → abbreviation and
acronym expansion (from an injected lexicon) → stopword and legal-phrase
stripping → canonical token tuple. Devanagari transliteration is a pluggable
hook: without a transliterator the text is kept verbatim and the omission is
recorded, never guessed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

from fmd.er.config import BlockingConfig, Lexicon
from fmd.er.types import BlockingKeys, NormalisedName, Script

Transliterator = Callable[[str], str]

_POSSESSIVE = re.compile(r"['\u2019]s\b")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_MULTISPACE = re.compile(r"\s+")
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_LATIN = re.compile(r"[A-Za-z]")
_SOUNDEX_CODES = {
    **dict.fromkeys("BFPV", "1"),
    **dict.fromkeys("CGJKQSXZ", "2"),
    **dict.fromkeys("DT", "3"),
    "L": "4",
    **dict.fromkeys("MN", "5"),
    "R": "6",
}


def transliterate_devanagari(text: str) -> str:
    """Default Devanagari transliteration hook.

    TODO(WS-E): wire an offline transliteration model (e.g. a local
    ``indic-transliteration`` / ICU rule set). The dependency is not part of the
    ``er`` extra yet, so this hook deliberately refuses rather than producing a
    lossy ad-hoc mapping. Callers may pass their own callable to
    :func:`normalise_name` instead.
    """
    raise NotImplementedError(
        "no Devanagari transliterator is configured; pass transliterate=<callable> "
        "to normalise_name (see TODO in fmd/er/normalise.py)"
    )


def detect_script(text: str) -> Script:
    has_devanagari = bool(_DEVANAGARI.search(text))
    has_latin = bool(_LATIN.search(text))
    if has_devanagari and has_latin:
        return "mixed"
    if has_devanagari:
        return "devanagari"
    if has_latin:
        return "latin"
    return "other"


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _fold_latin_accents(text: str) -> str:
    """Drop combining marks on Latin letters only (never on Devanagari matras)."""
    folded: list[str] = []
    for char in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(char) and folded and _LATIN.fullmatch(folded[-1]):
            continue
        folded.append(char)
    return "".join(folded)


def _tokenize(text: str) -> list[str]:
    cleaned = _NON_WORD.sub(" ", text)
    # ``\w`` keeps letters/digits of every script and drops punctuation.
    return [token for token in _MULTISPACE.split(cleaned) if token]


def _expand_tokens(
    tokens: Iterable[str], lexicon: Lexicon
) -> tuple[list[str], list[str], list[str]]:
    """Expand abbreviations/acronyms; also return the acronym forms seen/derived."""
    expanded: list[str] = []
    acronyms: list[str] = []
    seen: list[str] = []
    for token in tokens:
        expansion = lexicon.acronyms.get(token)
        if expansion:
            seen.append(token)
            acronyms.append(token)
            expanded.extend(_tokenize(normalise_text(expansion, lexicon)))
            continue
        abbreviation = lexicon.abbreviations.get(token)
        expanded.append(abbreviation or token)
    return expanded, acronyms, seen


def _strip_phrases(tokens: list[str], phrases: Iterable[str]) -> list[str]:
    """Remove configured legal phrases wherever they occur as token subsequences."""
    patterns: list[tuple[str, ...]] = []
    for phrase in phrases:
        parts = tuple(_tokenize(phrase.casefold()))
        if parts:
            patterns.append(parts)
    patterns.sort(key=len, reverse=True)
    result = list(tokens)
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            size = len(pattern)
            for start in range(0, len(result) - size + 1):
                if tuple(result[start : start + size]) == pattern:
                    del result[start : start + size]
                    changed = True
                    break
            if changed:
                break
    return result


def _derived_acronyms(tokens: Iterable[str]) -> list[str]:
    initials = "".join(token[0] for token in tokens if token)
    return [initials] if len(initials) >= 2 else []


def soundex(value: str, length: int = 4) -> str:
    """Deterministic ASCII Soundex. Non-Latin input yields an empty key."""
    letters = [ch for ch in _strip_accents(value).upper() if "A" <= ch <= "Z"]
    if not letters:
        return ""
    first = letters[0]
    encoded = [first]
    previous = _SOUNDEX_CODES.get(first, "")
    for char in letters[1:]:
        code = _SOUNDEX_CODES.get(char, "")
        if code and code != previous:
            encoded.append(code)
        if char not in "HW":
            previous = code
        if len(encoded) >= length:
            break
    return "".join(encoded).ljust(length, "0")[:length]


def normalise_text(text: str, lexicon: Lexicon) -> str:
    """NFKC + casefold + punctuation cleanup, before token-level work."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = _fold_latin_accents(folded)
    folded = _POSSESSIVE.sub("", folded)
    return _MULTISPACE.sub(" ", _NON_WORD.sub(" ", folded)).strip()


def normalise_name(
    text: str,
    lexicon: Lexicon | None = None,
    *,
    transliterate: Transliterator | None = None,
) -> NormalisedName:
    """Normalise one institution name into comparable canonical tokens."""
    if lexicon is None:
        from fmd.er.config import DEFAULT_LEXICON

        lexicon = DEFAULT_LEXICON

    raw = text
    script = detect_script(raw)
    notes: list[str] = []
    working = raw
    transliterated = False

    if script in {"devanagari", "mixed"} and transliterate is not None:
        working = transliterate(working)
        transliterated = True
        notes.append("transliterated")
        script = detect_script(working)
    elif script in {"devanagari", "mixed"}:
        notes.append("devanagari_untransliterated")

    folded = normalise_text(working, lexicon)
    tokens = _tokenize(folded)
    tokens, acronyms, _ = _expand_tokens(tokens, lexicon)
    raw_derived = _derived_acronyms([token for token in tokens if token not in lexicon.stopwords])
    tokens = _strip_phrases(tokens, lexicon.legal_phrases)
    tokens = [token for token in tokens if token not in lexicon.stopwords]
    # Collapse immediate repeats ("medical medical college").
    deduped: list[str] = []
    for token in tokens:
        if not deduped or deduped[-1] != token:
            deduped.append(token)
    tokens = deduped
    if not tokens:
        notes.append("empty_after_normalisation")
        tokens = _tokenize(folded)
    derived = _derived_acronyms(tokens)
    all_acronyms = tuple(dict.fromkeys([*acronyms, *raw_derived, *derived]))
    return NormalisedName(
        raw=raw,
        text=" ".join(tokens),
        tokens=tuple(tokens),
        acronyms=all_acronyms,
        script=script,
        transliterated=transliterated,
        notes=tuple(notes),
    )


def blocking_keys(
    normalised: NormalisedName,
    *,
    state: str | None = None,
    pincode: str | None = None,
    config: BlockingConfig | None = None,
    lexicon: Lexicon | None = None,
) -> BlockingKeys:
    """Blocking keys for one normalised name (see :func:`blocking_key`)."""
    config = config or BlockingConfig()
    lexicon = lexicon or Lexicon.default()
    tokens = [token for token in normalised.tokens if token not in lexicon.stopwords]
    first = tokens[0] if tokens else ""
    sx = soundex(normalised.text, config.soundex_length)
    digits = "".join(ch for ch in (pincode or "") if ch.isdigit())
    keys: list[str] = []
    state_key = None
    pincode_key = None

    if state:
        state_key = normalise_text(state, lexicon)
        if state_key:
            keys.append(f"state:{state_key}")
            if first and config.use_state_token:
                keys.append(f"state_token:{state_key}|{first}")
    if first:
        keys.append(f"token:{first}")
    if sx and config.use_name_soundex:
        keys.append(f"soundex:{sx}")
    if digits:
        pincode_key = digits
        keys.append(f"pincode:{digits}")
        if sx and config.use_pincode_soundex:
            keys.append(f"pincode_soundex:{digits}|{sx}")
    return BlockingKeys(
        keys=tuple(dict.fromkeys(keys)),
        first_token=first,
        soundex=sx,
        state_key=state_key,
        pincode_key=pincode_key,
    )


def blocking_key(
    name: str,
    state: str | None = None,
    pincode: str | None = None,
    *,
    lexicon: Lexicon | None = None,
    config: BlockingConfig | None = None,
    transliterate: Transliterator | None = None,
) -> tuple[str, ...]:
    """Convenience helper: ``(name, state, pincode) -> blocking keys``."""
    normalised = normalise_name(name, lexicon, transliterate=transliterate)
    return blocking_keys(normalised, state=state, pincode=pincode, config=config).keys


def normalise_many(
    texts: Iterable[str],
    lexicon: Lexicon | None = None,
    *,
    transliterate: Transliterator | None = None,
) -> list[NormalisedName]:
    return [normalise_name(text, lexicon, transliterate=transliterate) for text in texts]
