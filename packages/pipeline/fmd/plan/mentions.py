"""Name mentions for entity linking, mined from a document's own surface.

Brief §10: aliases discovered in headers/footers are high value. This module
finds *candidate strings*, never entity ids. It is deliberately conservative:

* an HTML ``<title>`` / heading is always a candidate;
* a plain-text line is a candidate only when it contains one of the generic
  institution markers the ER lexicon already publishes (``config/er.yaml``),
  so no institution or state vocabulary is hardcoded here;
* a line with an implausible shape (too short, too long, too many words) is
  dropped rather than truncated, because a truncated name is a wrong mention.

Only the first ``max_chars`` of text and the first few headings are read: entity
linking needs the cover page, not the whole PDF.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from fmd.er.config import Lexicon

_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"<(title|h1|h2|h3)\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_SPACE = re.compile(r"\s+")
_PUNCT = " \t,;:|-–—·"

DEFAULT_MAX_CHARS = 2000
DEFAULT_MAX_MENTIONS = 8
MAX_MENTION_WORDS = 14
MAX_MENTION_CHARS = 160


def _clean(value: str) -> str:
    text = _TAG.sub(" ", value)
    text = text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#39;", "'")
    return _SPACE.sub(" ", text).strip(_PUNCT)


def _markers(lexicon: Lexicon) -> tuple[str, ...]:
    return tuple(phrase.lower() for phrase in lexicon.legal_phrases if phrase.strip())


def _plausible(text: str, markers: tuple[str, ...]) -> bool:
    if not text or len(text) > MAX_MENTION_CHARS:
        return False
    words = text.split()
    if not 2 <= len(words) <= MAX_MENTION_WORDS:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _lines(text: str | None) -> list[str]:
    if not text:
        return []
    return [line for line in text.splitlines() if line.strip()]


def collect_mentions(
    document: object,
    *,
    text_sample: str | None = None,
    anchor_text: str | None = None,
    lexicon: Lexicon | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_mentions: int = DEFAULT_MAX_MENTIONS,
    normalise: Callable[[str], str] | None = None,
) -> list[str]:
    """Ordered, de-duplicated name mentions found on a document's surface."""
    lex = lexicon or Lexicon.default()
    markers = _markers(lex)
    sample = (text_sample or "")[:max_chars]
    key_of = normalise or (lambda value: value.lower())

    candidates: list[str] = []
    if anchor_text:
        candidates.extend(_clean(part) for part in anchor_text.splitlines())

    for match in _HEADING.finditer(sample):
        candidates.append(_clean(match.group(2)))
    for line in _lines(_TAG.sub("\n", sample)):
        candidates.append(_clean(line))

    seen: set[str] = set()
    mentions: list[str] = []
    for candidate in candidates:
        if not _plausible(candidate, markers):
            continue
        key = key_of(candidate)
        if key in seen:
            continue
        seen.add(key)
        mentions.append(candidate)
        if len(mentions) >= max_mentions:
            break
    return mentions
