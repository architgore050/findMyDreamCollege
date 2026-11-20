"""Programme-name normalisation shared by grouping and scoping.

The duplicate-row defect this fixes, measured on the operator DB: a seat-matrix
row numbered ``3.`` produced a claim whose value was ``3. B.Sc. Medical
Laboratory Technology``, and ER normalisation kept the leading ``3`` as a token
— so the same programme became a second (institution, normalised name) row.
Stripping leading row numbering *before* ER normalisation merges it back.

Order of operations (each step feeds the next):

1. strip decoration bullets and leading row numbering (``"3. "``, ``"3) "``,
   ``"(3) "``, ``"#3 "``, ``"Sl. 3. "``-style prefixes), repeatedly;
2. collapse whitespace;
3. ER normalisation (NFKC, casefold, punctuation folding, tokenisation) via
   :func:`fmd.er.normalise.normalise_name`;
4. registry alias canonicalisation (the caller owns the alias table).

Nothing here reads config, the database or a model.
"""

from __future__ import annotations

import re

from fmd.er.normalise import normalise_name

#: Bullets and list markers that are page furniture, not programme names.
_BULLET = re.compile(r"^\s*[#•*·‣◦▪\-–—]+\s*")

#: Bullet glued to a row number: ``#5 B.Sc.``, ``*3. MBBS``.
_BULLET_NUMBERING = re.compile(r"^\s*[#•*·‣◦▪]\s*\d{1,3}\s*[.)\]:\-]?\s+")

#: Leading row numbering: ``3.``, ``3)``, ``3 -``, ``03.`` — digits plus a
#: separator, only stripped when a space separates it from the name so a real
#: name like ``3-Year B.Sc.`` (no space after the hyphen) survives.
_NUMBERING = re.compile(r"^\s*\d{1,3}\s*[.)\]:\-–—]\s+")

#: Parenthesised row numbering: ``(3) B.Sc. MLT``.
_PAREN_NUMBERING = re.compile(r"^\s*\(\s*\d{1,3}\s*\)\s*")

#: Serial-label numbering: ``Sl. 3. B.Sc.``, ``Sr.No. 12 B.Sc.`` — the label is
#: furniture, the digits are the row number.
_SERIAL_LABEL = re.compile(
    r"^\s*(?:sl\.?|s\.?\s?no\.?|sr\.?\s?no\.?|s\.?\s?serial|no\.?)\s*[:.\-]?\s*\d{0,3}\s*[.)\]:\-]?\s+",
    re.IGNORECASE,
)

_MAX_STRIPS = 4


def strip_row_numbering(name: str) -> str:
    """Remove leading row-number prefixes (page furniture), keeping the name.

    Applied repeatedly because documents mix markers (``"#3. B.Sc."``). A
    prefix is only furniture when something follows it; a name that *is* a
    number never reaches this module as a programme name.
    """
    text = " ".join(str(name or "").split())
    for _ in range(_MAX_STRIPS):
        stripped = _SERIAL_LABEL.sub("", text, count=1)
        stripped = _PAREN_NUMBERING.sub("", stripped, count=1)
        stripped = _BULLET_NUMBERING.sub("", stripped, count=1)
        stripped = _NUMBERING.sub("", stripped, count=1)
        stripped = _BULLET.sub("", stripped, count=1)
        stripped = " ".join(stripped.split())
        if stripped == text:
            return text
        text = stripped
    return text


def normalise_programme_label(name: str) -> str:
    """Numbering-stripped, whitespace-collapsed, ER-normalised label."""
    return normalise_name(strip_row_numbering(name)).text


__all__ = ["normalise_programme_label", "strip_row_numbering"]
