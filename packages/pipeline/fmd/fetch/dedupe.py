"""Snapshot normalisation and near-duplicate detection (brief §7).

``sha256`` of normalised bytes is the exact-dedupe key. A 64-bit simhash over
normalised text catches the same PDF mirrored across portals without loading a
model.

Text simhash is only meaningful for *document-like* content (PDF text, an HTML
article body). A structured listing (JSON/XML results, a paginated directory)
has a shingle set dominated by its wrapper/boilerplate, so two distinct pages of
one listing look near-identical to simhash. Those payloads therefore dedupe on
exact bytes only, unless a source opts into ``identity`` mode: row identities
extracted from configured fields let disjoint pages of the same listing remain
distinct even at maximum structural similarity.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_MASK_64 = (1 << 64) - 1

#: Closed set of dedupe policies (registry ``dedupe.mode``).
DEDUPE_MODES: tuple[str, ...] = ("exact", "simhash", "identity")
DEFAULT_SIMHASH_THRESHOLD = 3

#: Content types that carry a machine-structured listing, not prose.
_STRUCTURED_CONTENT_MARKERS = (
    "json",
    "xml",
    "csv",
    "excel",
    "spreadsheet",
    "x-www-form-urlencoded",
)


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


def similarity(a: int, b: int) -> float:
    """Normalised similarity in ``[0, 1]`` derived from 64-bit Hamming distance."""
    return 1.0 - hamming(a, b) / 64.0


@dataclass(frozen=True)
class DedupeSettings:
    """Per-source dedupe policy, read from the registry (never source names).

    ``mode`` is ``None`` for an undeclared source: content type then decides
    (structured payloads get ``exact``, documents get ``simhash``).
    """

    mode: str | None = None
    simhash_threshold: int = DEFAULT_SIMHASH_THRESHOLD
    identity_fields: tuple[str, ...] = ()


def _params_of(source: Any) -> Mapping[str, Any]:
    params = getattr(source, "params", None)
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return params
    dump = getattr(params, "model_dump", None)
    return dict(dump()) if callable(dump) else {}


def dedupe_settings(source: Any) -> DedupeSettings:
    """Read ``params.dedupe`` for *source*; an absent block means auto mode.

    A malformed mode is a registry error and raises, never a silent fallback:
    a typo that disabled near-duplicate detection would hide exactly the defect
    this policy exists to prevent.
    """
    block = _params_of(source).get("dedupe")
    if not isinstance(block, Mapping):
        block = {}
    mode = block.get("mode")
    if mode is not None:
        mode = str(mode).strip().lower()
        if mode not in DEDUPE_MODES:
            raise ValueError(
                f"unknown dedupe.mode {mode!r}; expected one of {', '.join(DEDUPE_MODES)}"
            )
    threshold = block.get("simhash_threshold", DEFAULT_SIMHASH_THRESHOLD)
    try:
        threshold_int = int(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dedupe.simhash_threshold {threshold!r} is not an integer") from exc
    if not 0 <= threshold_int <= 64:
        raise ValueError(f"dedupe.simhash_threshold {threshold_int} must be between 0 and 64")
    fields = block.get("identity_fields") or ()
    if isinstance(fields, str):
        fields = (fields,)
    clean_fields = tuple(str(field).strip() for field in fields if str(field).strip())
    return DedupeSettings(
        mode=mode,
        simhash_threshold=threshold_int,
        identity_fields=clean_fields,
    )


def is_structured_payload(content_type: str | None, text: str) -> bool:
    """True when *text* is a machine-structured listing rather than prose.

    Content type wins; a mislabelled JSON body is still recognised by parsing.
    HTML is deliberately not structured: its article text is document-like.
    """
    lowered = (content_type or "").lower()
    if any(marker in lowered for marker in _STRUCTURED_CONTENT_MARKERS):
        return True
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except (TypeError, ValueError):
        return False
    return True


def effective_mode(settings: DedupeSettings, *, content_type: str | None, text: str) -> str:
    """Resolve the declared mode, or pick one from the payload shape."""
    if settings.mode is not None:
        return settings.mode
    if is_structured_payload(content_type, text):
        return "exact"
    return "simhash"


def row_identities(text: str, fields: Iterable[str]) -> frozenset[str]:
    """Row identities from a structured payload, keyed by configured fields.

    The walk is schema-agnostic: any object whose keys include a configured
    identity field contributes one ``field=value`` token, and nested JSON
    strings are parsed recursively. A payload with no matching field (or that is
    not JSON) yields an empty set, which callers must treat as "unknown", never
    as "duplicate".
    """
    wanted = tuple(field for field in fields if field)
    if not wanted:
        return frozenset()
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for field in wanted:
                value = node.get(field)
                if isinstance(value, (str, int, float)) and str(value).strip():
                    found.add(f"{field}={normalise_text(str(value))}")
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, str):
            stripped = node.strip()
            if stripped[:1] in "{[":
                try:
                    walk(json.loads(stripped))
                except (TypeError, ValueError):
                    return

    try:
        walk(json.loads(text))
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(found)
