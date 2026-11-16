"""Snapshot normalisation and near-duplicate detection (brief §7).

``sha256`` of normalised bytes is the exact-dedupe key. A 64-bit simhash over
normalised text catches the same PDF mirrored across portals without loading a
model.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_MASK_64 = (1 << 64) - 1


def normalise_bytes(body: bytes) -> bytes:
    """Canonicalise line endings and trailing whitespace for hashing.

    Byte-level normalisation is intentionally conservative: we do not strip
    binary content, only make text snapshots stable across CRLF/LF mirrors.
    """
    body = body.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return b"\n".join(line.rstrip() for line in body.split(b"\n"))


def sha256_hex(body: bytes) -> str:
    """Content hash of *normalised* bytes, as the snapshot store expects."""
    return hashlib.sha256(normalise_bytes(body)).hexdigest()


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _WHITESPACE.sub(" ", text).strip()


def _tokens(text: str) -> Iterable[str]:
    return _TOKEN.findall(normalise_text(text))


def simhash64(text: str) -> int:
    """64-bit Charikar simhash over word unigrams.

    Deterministic across processes (``blake2b`` instead of Python's salted
    ``hash``), which matters because the value is persisted.
    """
    vector = [0] * 64
    seen = False
    for token in _tokens(text):
        seen = True
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    if not seen:
        return 0
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result & _MASK_64


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicate(a: int, b: int, *, max_distance: int = 3) -> bool:
    return hamming(a, b) <= max_distance


def text_fingerprint(text: str) -> dict[str, object]:
    """Small JSON-safe record for persisting alongside a snapshot."""
    return {"simhash": f"{simhash64(text):016x}", "algo": "simhash64/blake2b-unigram"}
