"""Alias mining from documents (brief §10).

Headers/footers are high-value: a placement PDF may say only "KGMU" while the
full name appears once. This stage extracts acronyms, former names, local names,
transliterations and likely misspellings, with a span and confidence for each,
and dedupes against aliases already known for the entity.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Literal

from fmd.er.config import AliasMiningConfig, Lexicon
from fmd.er.normalise import normalise_name, normalise_text
from fmd.er.types import Alias, ErModel, SchemaAliasType

EvidenceKind = Literal["header", "body", "footer"]

_BRACKET_ACRONYM = re.compile(r"\(\s*([A-Z][A-Z0-9&.\-]{1,15})\s*\)")
_CAPS_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9&]{2,}(?![A-Za-z0-9])")
_DEVANAGARI_RUN = re.compile(r"[\u0900-\u097f]+(?:\s+[\u0900-\u097f]+)*")
_ACRONYM_SHAPE = re.compile(r"[A-Z][A-Z0-9&.\-]{1,15}")


class AliasCandidate(ErModel):
    alias: str
    alias_type: SchemaAliasType
    confidence: float
    evidence_kind: EvidenceKind = "body"
    quote: str = ""
    char_start: int = 0
    char_end: int = 0
    marker: str | None = None


def _initials(text: str, lexicon: Lexicon) -> str:
    normalised = normalise_name(text, lexicon)
    return "".join(token[0] for token in normalised.tokens if token)


def _line_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        spans.append((stripped, offset, offset + len(stripped)))
        offset += len(line)
    if not spans:
        spans.append(("", 0, 0))
    return spans


def _evidence_kind(index: int, total: int, config: AliasMiningConfig) -> EvidenceKind:
    if index < config.header_lines:
        return "header"
    if index >= max(0, total - config.footer_lines):
        return "footer"
    return "body"


def _looks_like_acronym(value: str) -> bool:
    return bool(_ACRONYM_SHAPE.fullmatch(value.strip()))


def _normalised_key(value: str, lexicon: Lexicon) -> str:
    return normalise_name(value, lexicon).text


def _trim_alias(value: str) -> str:
    alias = value.strip(" \t.,;:|\u2013-\u2014")
    words = alias.split()
    while words and words[0].casefold() in {"the", "a", "an"}:
        words.pop(0)
    return " ".join(words).strip(" \t.,;:|\u2013-\u2014")


def _marker_candidates(
    spans: Sequence[tuple[str, int, int]],
    *,
    lexicon: Lexicon,
    config: AliasMiningConfig,
) -> list[AliasCandidate]:
    found: list[AliasCandidate] = []
    markers = sorted(lexicon.alias_markers, key=lambda item: len(item.marker), reverse=True)
    for index, (line, line_start, _) in enumerate(spans):
        consumed: list[tuple[int, int]] = []
        lowered = line.casefold()
        for marker in markers:
            needle = marker.marker.casefold()
            start = lowered.find(needle)
            while start >= 0:
                end = start + len(needle)
                if any(lo <= start < hi for lo, hi in consumed):
                    start = lowered.find(needle, start + 1)
                    continue
                tail = line[end:]
                match = re.search(r"[A-Za-z\u0900-\u097f]", tail)
                if match is None:
                    break
                tail = tail[match.start() :]
                stop = re.search(r"[;,|()\[\]\n]", tail)
                raw = tail[: stop.start()] if stop else tail
                alias = _trim_alias(" ".join(raw.split()[: config.max_alias_tokens]))
                if len(alias) < 2:
                    start = lowered.find(needle, start + 1)
                    continue
                alias_start = line.find(alias, end)
                if alias_start < 0:
                    alias_start = end + match.start()
                consumed.append((start, alias_start + len(alias)))
                alias_type: SchemaAliasType = marker.alias_type
                confidence = marker.confidence
                if _looks_like_acronym(alias):
                    alias_type = "acronym"
                    confidence = max(confidence, config.bracket_acronym_confidence)
                found.append(
                    AliasCandidate(
                        alias=alias,
                        alias_type=alias_type,
                        confidence=round(min(confidence, 1.0), 3),
                        evidence_kind=_evidence_kind(index, len(spans), config),
                        quote=line.strip(),
                        char_start=line_start + alias_start,
                        char_end=line_start + alias_start + len(alias),
                        marker=marker.marker,
                    )
                )
                start = lowered.find(needle, start + 1)
    return found


def _bracket_candidates(
    spans: Sequence[tuple[str, int, int]],
    *,
    lexicon: Lexicon,
    config: AliasMiningConfig,
) -> list[AliasCandidate]:
    found: list[AliasCandidate] = []
    for index, (line, line_start, _) in enumerate(spans):
        for match in _BRACKET_ACRONYM.finditer(line):
            alias = _trim_alias(match.group(1))
            if not alias:
                continue
            found.append(
                AliasCandidate(
                    alias=alias,
                    alias_type="acronym",
                    confidence=config.bracket_acronym_confidence,
                    evidence_kind=_evidence_kind(index, len(spans), config),
                    quote=line.strip(),
                    char_start=line_start + match.start(1),
                    char_end=line_start + match.end(1),
                )
            )
    return found


def _caps_candidates(
    spans: Sequence[tuple[str, int, int]],
    *,
    canonical_name: str,
    lexicon: Lexicon,
    config: AliasMiningConfig,
) -> list[AliasCandidate]:
    canonical = normalise_name(canonical_name, lexicon)
    canonical_key = canonical.text
    signature = set(canonical.tokens)
    name_lines = {
        index
        for index, (line, _, _) in enumerate(spans)
        if signature and signature.issubset(set(normalise_name(line, lexicon).tokens))
    }
    derived = set(canonical.acronyms)
    found: list[AliasCandidate] = []
    for index, (line, line_start, _) in enumerate(spans):
        kind = _evidence_kind(index, len(spans), config)
        for match in _CAPS_TOKEN.finditer(line):
            token = match.group(0)
            key = token.casefold()
            if key == canonical_key or key in derived:
                confidence = 0.95
            elif index in name_lines or kind in {"header", "footer"}:
                confidence = config.caps_acronym_confidence
            else:
                continue
            found.append(
                AliasCandidate(
                    alias=token,
                    alias_type="acronym",
                    confidence=confidence,
                    evidence_kind=kind,
                    quote=line.strip(),
                    char_start=line_start + match.start(),
                    char_end=line_start + match.end(),
                )
            )
    return found


def _local_name_candidates(
    text: str,
    *,
    canonical_name: str,
    lexicon: Lexicon,
    config: AliasMiningConfig,
    transliterate,
) -> list[AliasCandidate]:
    canonical = normalise_name(canonical_name, lexicon)
    if canonical.script in {"devanagari", "mixed"}:
        return []  # the canonical name is already in the local script
    found: list[AliasCandidate] = []
    for match in _DEVANAGARI_RUN.finditer(text):
        alias = " ".join(match.group(0).split()[: config.max_alias_tokens])
        if len(alias) < 2:
            continue
        found.append(
            AliasCandidate(
                alias=alias,
                alias_type="local_name",
                confidence=config.local_name_confidence,
                quote=alias,
                char_start=match.start(),
                char_end=match.start() + len(alias),
            )
        )
        if transliterate is not None:
            try:
                latin = transliterate(alias)
            except Exception:  # noqa: BLE001 - a hook failure must not stop mining
                latin = ""
            if latin and latin != alias:
                found.append(
                    AliasCandidate(
                        alias=latin,
                        alias_type="transliteration",
                        confidence=config.transliteration_confidence,
                        quote=alias,
                        char_start=match.start(),
                        char_end=match.start() + len(alias),
                    )
                )
    return found


def _retype_misspellings(
    candidates: list[AliasCandidate],
    *,
    canonical_name: str,
    lexicon: Lexicon,
    config: AliasMiningConfig,
) -> list[AliasCandidate]:
    from fmd.er.similarity import char_ratio

    canonical = normalise_name(canonical_name, lexicon)
    canonical_key = canonical.text
    retyped: list[AliasCandidate] = []
    for candidate in candidates:
        key = _normalised_key(candidate.alias, lexicon)
        looks_like_misspelling = (
            candidate.alias_type not in {"misspelling", "transliteration", "local_name", "acronym"}
            and key != canonical_key
            and key
            and canonical_key
            and char_ratio(key, canonical_key) >= config.misspelling_similarity
        )
        if looks_like_misspelling:
            candidate = candidate.model_copy(update={"alias_type": "misspelling"})
        retyped.append(candidate)
    return retyped


def _dedupe(
    candidates: Iterable[AliasCandidate],
    *,
    canonical_name: str,
    known: Iterable[str],
    lexicon: Lexicon,
    config: AliasMiningConfig,
) -> list[AliasCandidate]:
    excluded = {_normalised_key(canonical_name, lexicon)}
    excluded.update(_normalised_key(value, lexicon) for value in known if value)
    best: dict[str, AliasCandidate] = {}
    for candidate in candidates:
        key = _normalised_key(candidate.alias, lexicon)
        if not key or key in excluded:
            continue
        if len(candidate.alias.split()) > config.max_alias_tokens:
            continue
        previous = best.get(key)
        if previous is None or candidate.confidence > previous.confidence:
            best[key] = candidate
    ordered = sorted(best.values(), key=lambda item: (-item.confidence, item.alias))
    return ordered[: config.max_aliases]


def mine_aliases(
    text: str,
    *,
    canonical_name: str,
    known_aliases: Iterable[str | Alias] = (),
    lexicon: Lexicon | None = None,
    config: AliasMiningConfig | None = None,
    transliterate=None,
) -> list[AliasCandidate]:
    """Extract candidate aliases from a document, deduped and ranked."""
    lex = lexicon or Lexicon.default()
    cfg = config or AliasMiningConfig()
    known = [
        item.alias if isinstance(item, Alias) else str(item) for item in known_aliases
    ]
    spans = _line_spans(text)
    found = [
        *_marker_candidates(spans, lexicon=lex, config=cfg),
        *_bracket_candidates(spans, lexicon=lex, config=cfg),
        *_caps_candidates(
            spans, canonical_name=canonical_name, lexicon=lex, config=cfg
        ),
        *_local_name_candidates(
            text,
            canonical_name=canonical_name,
            lexicon=lex,
            config=cfg,
            transliterate=transliterate,
        ),
    ]
    found = _retype_misspellings(
        found, canonical_name=canonical_name, lexicon=lex, config=cfg
    )
    return _dedupe(
        found,
        canonical_name=canonical_name,
        known=known,
        lexicon=lex,
        config=cfg,
    )


def to_alias_creates(
    entity_id: int,
    candidates: Iterable[AliasCandidate],
    *,
    source_id: str | None = None,
):
    """Convert mined candidates to ``entity_aliases`` rows (5 schema-safe types)."""
    from fmd.db.repositories import AliasCreate

    return [
        AliasCreate(
            entity_id=entity_id,
            alias=candidate.alias,
            alias_type=candidate.alias_type,
            source_id=source_id,
            confidence=candidate.confidence,
        )
        for candidate in candidates
    ]


__all__ = [
    "AliasCandidate",
    "mine_aliases",
    "to_alias_creates",
]
