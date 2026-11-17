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

**Dedupe basis.** A decision is only defensible if the bytes hashed are the
document's text. ``FetchResult.text`` is the raw body decoded as UTF-8, which
for a PDF is ``%PDF`` headers, object streams and font boilerplate. Hashing that
collapsed 73 distinct PDFs in the 2026-11-18 crawl. PDFs therefore hash a
bounded *extracted text* sample (the T0 ``pdftext`` path, first
``text_max_pages`` pages, at most ``text_max_chars`` characters). HTML/plain
bodies keep hashing their decoded text. When extraction is unavailable the
policy fails safe: the comparison degrades to exact ``sha256`` and the document
is stored, never dropped on a hash of container bytes. Every decision records
its ``basis`` so a later review can see what was compared:

``raw_bytes``      the decision used the sha256 of the fetched bytes;
``extracted_text`` the decision used text extracted/decoded from the document;
``identity``       the decision used row identities from a structured payload.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_MASK_64 = (1 << 64) - 1

#: Closed set of dedupe policies (registry ``dedupe.mode``).
DEDUPE_MODES: tuple[str, ...] = ("exact", "simhash", "identity")
#: Closed set of justifications for a dedupe decision (persisted with it).
DEDUPE_BASES: tuple[str, ...] = ("raw_bytes", "extracted_text", "identity")
DEFAULT_SIMHASH_THRESHOLD = 3
#: A PDF's fingerprint is bounded to the first pages/characters: enough prose to
#: be discriminating, cheap enough to sit inside the fetch path.
DEFAULT_PDF_TEXT_MAX_PAGES = 3
DEFAULT_PDF_TEXT_MAX_CHARS = 20_000

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


def text_sha256(text: str) -> str:
    """Content hash of normalised *document text* (the text-level dedupe key).

    Distinct from :func:`sha256_hex`, which hashes fetched bytes: a PDF mirrored
    with different object-stream compression has different bytes but the same
    text, and this is the key that recognises it.
    """
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def is_pdf_payload(content_type: str | None, body: bytes) -> bool:
    """True when the payload is a PDF, by declared type or by magic bytes.

    Content type wins when it says PDF. The magic-byte probe covers the common
    ``application/octet-stream`` answer from misconfigured file handlers, which
    is exactly where the raw-bytes defect hid.
    """
    if "pdf" in (content_type or "").lower():
        return True
    return body[:1024].lstrip()[:5] == b"%PDF-"


def pdf_text_sample(
    body: bytes,
    *,
    max_pages: int = DEFAULT_PDF_TEXT_MAX_PAGES,
    max_chars: int = DEFAULT_PDF_TEXT_MAX_CHARS,
) -> str | None:
    """Bounded text layer of *body* for fingerprinting, or ``None``.

    Uses the same T0 library path as :func:`fmd.acquire.tiers.t0_extract`
    (``pdftext`` over pdfium) but reads only the first ``max_pages`` and keeps
    at most ``max_chars`` characters, because dedupe must stay inside the fetch
    path. Any failure — missing extra, malformed/encrypted PDF, empty text layer
    (a scan) — returns ``None``. Callers must treat that as "no text basis" and
    fall back to exact bytes; it is never a reason to drop a document.
    """
    if not body or max_pages <= 0 or max_chars <= 0:
        return None
    path: Path | None = None
    try:
        import pdftext.extraction as extraction  # lazy: an extract-extra import
        import pypdfium2 as pdfium

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(body)
            handle.flush()
            path = Path(handle.name)

        document = pdfium.PdfDocument(str(path))
        try:
            page_count = len(document)
        finally:
            document.close()
        if page_count <= 0:
            return None

        # pdftext errors on an out-of-range page index, so the count is bounded
        # here rather than relying on the reader to clip the range.
        text = extraction.plain_text_output(
            str(path),
            sort=True,
            page_range=range(min(page_count, max_pages)),
            flatten_pdf=False,
        )
    except Exception:  # any reader failure means "no text basis", not a drop
        return None
    finally:
        if path is not None:
            path.unlink(missing_ok=True)

    text = text.strip()
    return text[:max_chars] if text else None


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
    pdf_text_max_pages: int = DEFAULT_PDF_TEXT_MAX_PAGES
    pdf_text_max_chars: int = DEFAULT_PDF_TEXT_MAX_CHARS


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
        pdf_text_max_pages=_bounded_int(block, "text_max_pages", DEFAULT_PDF_TEXT_MAX_PAGES),
        pdf_text_max_chars=_bounded_int(block, "text_max_chars", DEFAULT_PDF_TEXT_MAX_CHARS),
    )


def _bounded_int(block: Mapping[str, Any], key: str, default: int) -> int:
    value = block.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dedupe.{key} {value!r} is not an integer") from exc
    if parsed < 1:
        raise ValueError(f"dedupe.{key} {parsed} must be a positive integer")
    return parsed


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
