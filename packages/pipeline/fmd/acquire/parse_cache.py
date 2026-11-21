"""Persistent parse cache for the T1 (Docling) conversion (W10 batched acquire).

Why this exists (runbook §6, measured 2026-11-20): extraction was one fused
loop — Docling T1 on CPU, then the LLM call — so the GPU idled during every
conversion and re-running extraction re-converted every document. This module
persists the T1 parse as versioned JSON in the content-addressed object store,
with a lookup row in ``document_parses`` (migration 0011). Acquisition and
extraction become two stages: ``fmd acquire`` converts and caches with zero LLM
calls; ``fmd extract`` consumes cached parses while the model server is up.

Cache key = sha256 over a canonical JSON of

    (snapshot_sha256, tier, policy_digest, schema_version)

* ``tier`` is the tier whose parse is stored — ``"T1"`` only in this patch
  (T0 pdftext is cheap to redo; T2/OCR output is a known gap, never cached);
* ``policy_digest`` covers the device-independent, parse-shaping subset of
  ``config/acquire.yaml`` — a policy or schema change invalidates;
* **the Docling device is deliberately not part of the key**: a CUDA parse is
  reusable by a CPU run and vice versa. The Docling library version is
  recorded in the payload for provenance but does not invalidate (known gap).

Guarantees:

* lookup happens before conversion; a hit invokes no converter at all;
* corrupt, truncated or schema-mismatched payloads are reported as misses with
  a stable reason — the document is reconverted, never guessed, never a crash;
* writes are atomic: payload object first (temp → fsync → rename, same scheme
  as evidence bytes), lookup row second. An interrupted write is never read
  back;
* an entry represents a **complete document only**. A failed conversion writes
  nothing, so there is no partial parse to resume or to extract from;
* the reconstruction path rebuilds exactly the object the extractor consumes
  (canonical text, anchors, page map, tables with cell associations, outline),
  so live-converted and cached parses produce identical extraction output.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from fmd.acquire.canonical import PageGeometry, TextBlock, build_canonical
from fmd.acquire.errors import AcquisitionFailed, TierUnavailable
from fmd.acquire.outline import OutlinePolicy
from fmd.acquire.tables import CellHeaderLink
from fmd.acquire.types import ExtractedDoc, PageSpan, Table
from fmd.db.repositories import DocumentParsePut

logger = logging.getLogger(__name__)

#: Bump to invalidate every stored parse (also mirrored in
#: ``config/acquire.yaml:cache.schema_version``, which must agree).
PARSE_SCHEMA_VERSION = 1

#: The only tier cached in this patch.
CACHED_TIERS = ("T1",)

# Stable miss/reconvert reasons (never prose).
MISS_NO_ENTRY = "no_cache_entry"
MISS_CORRUPT = "corrupt_cache"
MISS_SCHEMA_MISMATCH = "schema_mismatch"
MISS_OBJECT_MISSING = "cache_object_missing"
MISS_NOT_PDF = "not_pdf_cached_tier"
MISS_DISABLED = "cache_unavailable"

#: Parse-shaping fields of ``config/acquire.yaml`` that feed the policy digest.
#: Deliberately excludes every device/artifact/endpoint setting: those change
#: where and on what hardware a parse runs, never what the parse says.
_PARSE_POLICY_FIELDS = ("tiers", "outline", "anomaly")
_T1_PARSE_FIELDS = ("tool", "model", "granularity")


def parse_policy_digest(acquire: Any) -> str:
    """Device-independent digest of the acquire registry's parse-shaping keys.

    ``acquire`` is the validated ``AcquireConfig``. Extra keys allowed by the
    model (``extra="allow"``) are included only when they are one of the named
    parse-shaping blocks, so adding an unrelated knob to the registry never
    invalidates the cache wholesale.
    """
    projection: dict[str, Any] = {}
    for name in _PARSE_POLICY_FIELDS:
        value = getattr(acquire, name, None)
        if value is None:
            continue
        projection[name] = _plain(value)
    t1 = (getattr(acquire, "backends", None) or {}).get("T1")
    if t1 is not None:
        projection["backends.T1"] = {
            key: _plain(getattr(t1, key, None)) for key in _T1_PARSE_FIELDS
        }
    payload = json.dumps(projection, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    """JSON-stable projection of a pydantic model / dataclass / mapping."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _plain(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _plain(model_dump())
        except TypeError:
            pass
    if hasattr(value, "__dataclass_fields__"):
        return {key: _plain(getattr(value, key)) for key in sorted(value.__dataclass_fields__)}
    return str(value)


def parse_key(snapshot_sha256: str, tier: str, policy_digest: str) -> str:
    """The cache key: sha256 over the canonical component tuple."""
    canonical = json.dumps(
        {
            "snapshot_sha256": str(snapshot_sha256).lower(),
            "tier": str(tier),
            "policy_digest": str(policy_digest),
            "schema_version": PARSE_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ serialise


def serialize_parse(doc: ExtractedDoc) -> bytes:
    """Versioned JSON payload sufficient to reconstruct the extractor's object.

    The payload keeps the *inputs* of the canonical build (blocks with text,
    label, bbox and granularity; page geometry; tables with cell associations)
    rather than the derived canonical string: ``build_canonical`` is
    deterministic, so storing its inputs is enough and keeps the payload small.
    """
    payload = {
        "schema_version": PARSE_SCHEMA_VERSION,
        "tier": doc.tier,
        "tool": doc.tool,
        "tool_version": doc.tool_version,
        "fallback_from": doc.fallback_from,
        "notes": list(doc.notes),
        "bbox_granularity": doc.bbox_granularity,
        "two_column": doc.two_column,
        "table_confidence": doc.table_confidence,
        "engine_meta": doc.engine_meta,
        "pages": [
            {
                "page_no": page.page_no,
                "width": page.width,
                "height": page.height,
                "bbox": list(page.bbox),
            }
            for page in (doc.pages or ())
        ],
        "blocks": [_block_payload(block) for block in (doc.blocks or ())],
        "tables": [table.as_dict() for table in (doc.tables or ())],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _block_payload(block: TextBlock) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "page_no": block.page_no,
        "text": block.text,
        "bbox": list(block.bbox) if block.bbox is not None else None,
        "granularity": block.granularity,
        "block_id": block.block_id,
        "line": block.line,
        "outline_label": block.outline_label,
    }
    if block.char_boxes is not None:
        payload["char_boxes"] = [list(box) for box in block.char_boxes]
    if block.char_indexes is not None:
        payload["char_indexes"] = list(block.char_indexes)
    return payload


def reconstruct_parse(payload: bytes | str | dict[str, Any], policy: Any) -> ExtractedDoc:
    """Rebuild the :class:`ExtractedDoc` the extractor consumes from a payload.

    Mirrors ``fmd.acquire.tiers._docling_doc``'s construction step for step
    (same ``build_canonical`` inputs, same table-confidence aggregation), so
    the cached object is indistinguishable from a live conversion of the same
    document. Raises :class:`ParseCacheError` on any structural mismatch.
    """
    data = json.loads(payload) if isinstance(payload, (bytes, str)) else dict(payload)
    if not isinstance(data, dict):
        raise ParseCacheError(MISS_CORRUPT, "payload is not a JSON object")
    version = data.get("schema_version")
    if version != PARSE_SCHEMA_VERSION:
        raise ParseCacheError(
            MISS_SCHEMA_MISMATCH,
            f"payload schema_version {version!r} != {PARSE_SCHEMA_VERSION}",
        )
    tier = data.get("tier")
    if tier not in CACHED_TIERS:
        raise ParseCacheError(
            MISS_SCHEMA_MISMATCH, f"payload tier {tier!r} is not one of {CACHED_TIERS}"
        )
    try:
        blocks = tuple(_block_from_payload(raw) for raw in data.get("blocks") or [])
        geoms = tuple(_page_from_payload(raw) for raw in data.get("pages") or [])
        tables = [_table_from_payload(raw) for raw in data.get("tables") or []]
    except ParseCacheError:
        raise
    except Exception as exc:  # malformed field shapes of any kind
        raise ParseCacheError(MISS_CORRUPT, f"malformed payload field: {exc}") from exc

    canonical = build_canonical(blocks, geoms, outline_policy=_outline_policy(policy))
    confidences = [table.confidence for table in tables if table.confidence is not None]
    return ExtractedDoc(
        text=canonical.text,
        page_map=[
            PageSpan(page=page.page_no, char_start=page.char_start, char_end=page.char_end)
            for page in canonical.pages
        ],
        tables=tables,
        tier=str(tier),
        tool=str(data.get("tool") or "docling"),
        tool_version=data.get("tool_version"),
        fallback_from=data.get("fallback_from"),
        notes=tuple(str(note) for note in data.get("notes") or ()),
        canonical=canonical,
        pages=canonical.pages,
        blocks=blocks,
        bbox_granularity=str(data.get("bbox_granularity") or "span"),
        two_column=bool(data.get("two_column")),
        table_confidence=min(confidences) if confidences else None,
        engine_meta=data.get("engine_meta"),
    )


def _outline_policy(policy: Any) -> OutlinePolicy:
    acquire = getattr(policy, "acquire", None) if policy is not None else None
    return OutlinePolicy.from_acquire(acquire)


def _block_from_payload(raw: Any) -> TextBlock:
    if not isinstance(raw, dict):
        raise ParseCacheError(MISS_CORRUPT, "block entry is not an object")
    bbox = raw.get("bbox")
    char_boxes = raw.get("char_boxes")
    return TextBlock(
        page_no=int(raw.get("page_no") or 0),
        text=str(raw.get("text") or ""),
        bbox=tuple(float(value) for value in bbox) if bbox is not None else None,
        granularity=str(raw.get("granularity") or "span"),
        char_boxes=(
            tuple(tuple(float(value) for value in box) for box in char_boxes)
            if char_boxes is not None
            else None
        ),
        char_indexes=(
            tuple(int(index) for index in raw.get("char_indexes"))
            if raw.get("char_indexes") is not None
            else None
        ),
        block_id=(int(raw["block_id"]) if raw.get("block_id") is not None else None),
        line=(int(raw["line"]) if raw.get("line") is not None else None),
        outline_label=raw.get("outline_label"),
    )


def _page_from_payload(raw: Any) -> PageGeometry:
    if not isinstance(raw, dict):
        raise ParseCacheError(MISS_CORRUPT, "page entry is not an object")
    bbox = raw.get("bbox") or (
        0.0,
        0.0,
        float(raw.get("width") or 0.0),
        float(raw.get("height") or 0.0),
    )
    return PageGeometry(
        page_no=int(raw.get("page_no") or 1),
        width=float(raw.get("width") or 0.0),
        height=float(raw.get("height") or 0.0),
        bbox=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
        char_start=0,
        char_end=0,
    )


def _table_from_payload(raw: Any) -> Table:
    if not isinstance(raw, dict):
        raise ParseCacheError(MISS_CORRUPT, "table entry is not an object")
    cells = tuple(
        CellHeaderLink(
            row=int(cell.get("row") or 0),
            col=int(cell.get("col") or 0),
            text=str(cell.get("text") or ""),
            header_text=cell.get("header_text"),
            header_row=(int(cell["header_row"]) if cell.get("header_row") is not None else None),
            ambiguous=bool(cell.get("column_header_uncertain")),
            reason=cell.get("reason"),
        )
        for cell in raw.get("cells") or []
        if isinstance(cell, dict)
    )
    confidence = raw.get("confidence")
    return Table(
        rows=[
            [str(value) for value in row] for row in raw.get("rows") or [] if isinstance(row, list)
        ],
        page=(int(raw["page"]) if raw.get("page") is not None else None),
        caption=raw.get("caption"),
        cells=cells,
        column_header_uncertain=bool(raw.get("column_header_uncertain")),
        confidence=(float(confidence) if confidence is not None else None),
    )


class ParseCacheError(Exception):
    """A stored parse could not be trusted; the document must be reconverted."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


# ------------------------------------------------------------------- the cache


@dataclass
class CacheHit:
    """Provenance of one cache hit, surfaced in run summaries."""

    key_digest: str
    object_sha256: str
    pages: int
    parser: str
    parser_version: str | None
    created_at: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "cache",
            "key_digest": self.key_digest,
            "object_sha256": self.object_sha256,
            "pages": self.pages,
            "parser": self.parser,
            "parser_version": self.parser_version,
        }


@dataclass
class CacheMiss:
    """Why a document needs conversion, with the stable reason code."""

    key_digest: str | None
    reason: str
    detail: str | None = None
    #: Set when a live conversion completed and was written to the cache.
    stored: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {"mode": "live", "reason": self.reason}
        if self.key_digest:
            payload["key_digest"] = self.key_digest
        if self.detail:
            payload["detail"] = self.detail[:500]
        if self.stored:
            payload["stored"] = True
        return payload


class ParseCache:
    """Lookup/store façade over the object store and the ``document_parses`` repo.

    Both operations are failure-isolated: a cache problem (corrupt payload,
    missing object, unusable repo) degrades to a miss with a recorded reason —
    the caller converts live — and never fails a run by itself.
    """

    def __init__(self, repo: Any, store: Any, *, tier: str = "T1") -> None:
        self.repo = repo
        self.store = store
        self.tier = tier
        #: Counters for run summaries (bumped in place; never reset here).
        self.hits = 0
        self.stores = 0

    # ------------------------------------------------------------------ keying
    def key_for(self, snapshot_sha256: str, policy: Any) -> str:
        return parse_key(snapshot_sha256, self.tier, parse_policy_digest(_acquire_of(policy)))

    def pages_for(self, snapshot_sha256: str, policy: Any) -> int | None:
        """Known page count for batch sizing, without loading the payload."""
        row = self._row(snapshot_sha256, policy)
        return int(row.pages) if row is not None else None

    def has_parse(self, snapshot_sha256: str, policy: Any) -> bool:
        return self._row(snapshot_sha256, policy) is not None

    def _row(self, snapshot_sha256: str, policy: Any) -> Any | None:
        try:
            key = self.key_for(snapshot_sha256, policy)
            row = self.repo.get(key)
        except Exception as exc:  # repo unavailability is a miss, never a crash
            logger.warning("parse-cache lookup unavailable: %s", exc)
            return None
        return row

    # ------------------------------------------------------------------ lookup
    def lookup(
        self, snapshot_sha256: str, policy: Any
    ) -> tuple[ExtractedDoc | None, CacheHit | CacheMiss]:
        """Return ``(doc, hit)`` on a verified hit, ``(None, miss)`` otherwise.

        Verification order: row present → row components match the requested
        key → payload object exists and its sha256 verifies → payload parses
        and its schema matches. Any failure is a miss with a stable reason.
        """
        acquire = _acquire_of(policy)
        try:
            key = self.key_for(snapshot_sha256, policy)
        except Exception as exc:
            return None, CacheMiss(None, MISS_DISABLED, f"key derivation failed: {exc}")
        try:
            row = self.repo.get(key)
        except Exception as exc:
            return None, CacheMiss(key, MISS_DISABLED, f"parse repo unavailable: {exc}")
        if row is None:
            return None, CacheMiss(key, MISS_NO_ENTRY)
        if int(getattr(row, "schema_version", 0)) != PARSE_SCHEMA_VERSION:
            return None, CacheMiss(key, MISS_SCHEMA_MISMATCH, "row schema_version differs")
        if str(getattr(row, "tier", "")) != self.tier:
            return None, CacheMiss(key, MISS_SCHEMA_MISMATCH, "row tier differs")
        if str(getattr(row, "policy_digest", "")) != parse_policy_digest(acquire):
            return None, CacheMiss(key, MISS_SCHEMA_MISMATCH, "policy digest differs")
        try:
            payload = self.store.get(str(row.object_sha256))
        except Exception as exc:
            reason = MISS_CORRUPT if "integrity" in str(exc).lower() else MISS_OBJECT_MISSING
            return None, CacheMiss(key, reason, str(exc))
        try:
            doc = reconstruct_parse(payload, policy)
        except ParseCacheError as exc:
            return None, CacheMiss(key, exc.reason, exc.detail)
        except Exception as exc:
            return None, CacheMiss(key, MISS_CORRUPT, str(exc))
        hit = CacheHit(
            key_digest=key,
            object_sha256=str(row.object_sha256),
            pages=int(row.pages),
            parser=str(row.parser),
            parser_version=getattr(row, "parser_version", None),
            created_at=getattr(row, "created_at", None),
        )
        self.hits += 1
        return doc, hit

    # ------------------------------------------------------------------- store
    def store_parse(
        self, snapshot_sha256: str, doc: ExtractedDoc, policy: Any
    ) -> CacheMiss | CacheHit:
        """Write one *complete* parse; returns provenance, never raises.

        Order: payload bytes → object store (atomic) → lookup row. Only a
        complete, cacheable parse (T1, born-digital, no review flag, non-empty)
        is accepted; anything else is reported with a reason and nothing is
        written.
        """
        acquire = _acquire_of(policy)
        try:
            key = self.key_for(snapshot_sha256, policy)
        except Exception as exc:
            return CacheMiss(None, MISS_DISABLED, f"key derivation failed: {exc}")
        if doc.tier not in CACHED_TIERS or doc.fallback_from is not None:
            return CacheMiss(key, MISS_NOT_PDF, f"tier {doc.tier!r} is not cacheable")
        if doc.ocr_anomaly or doc.needs_review or doc.is_empty:
            return CacheMiss(key, MISS_NOT_PDF, "incomplete or refused parse is never cached")
        payload = serialize_parse(doc)
        try:
            object_sha = self.store.put(payload)
        except Exception as exc:
            return CacheMiss(key, MISS_DISABLED, f"object store unavailable: {exc}")
        try:
            self.repo.put(
                DocumentParsePut(
                    key_digest=key,
                    snapshot_sha256=str(snapshot_sha256).lower(),
                    tier=self.tier,
                    policy_digest=parse_policy_digest(acquire),
                    schema_version=PARSE_SCHEMA_VERSION,
                    object_sha256=object_sha,
                    pages=_page_count(doc),
                    parser=str(doc.tool),
                    parser_version=doc.tool_version,
                )
            )
        except Exception as exc:
            # Object written, row refused: the entry stays unreachable (a miss
            # on every later run) until the row lands. Never a partial read.
            return CacheMiss(key, MISS_DISABLED, f"parse repo write failed: {exc}")
        self.stores += 1
        return CacheHit(
            key_digest=key,
            object_sha256=object_sha,
            pages=_page_count(doc),
            parser=str(doc.tool),
            parser_version=doc.tool_version,
        )


def _acquire_of(policy: Any) -> Any:
    return getattr(policy, "acquire", policy)


def _page_count(doc: ExtractedDoc) -> int:
    pages = doc.pages or doc.page_map
    return len(pages) if pages is not None else 0


# ------------------------------------------------------------------- dispatch


def cacheable(doc: ExtractedDoc) -> bool:
    """Is this acquisition result a complete T1 parse worth caching?"""
    return (
        doc.tier in CACHED_TIERS
        and doc.fallback_from is None
        and not doc.ocr_anomaly
        and not doc.needs_review
        and not doc.is_empty
    )


def acquire_document(
    data: bytes,
    *,
    content_type: str | None = None,
    policy: Any = None,
    cache: ParseCache | None = None,
    snapshot_sha256: str | None = None,
    t0: Any = None,
    t1: Any = None,
    t2: Any = None,
    t2b: Any = None,
    allow_convert: bool = True,
) -> tuple[ExtractedDoc | None, CacheHit | CacheMiss]:
    """The cached seam around :func:`fmd.extract.pdf.extract_document`.

    Lookup before conversion: a verified hit reconstructs the parse and invokes
    no tier function at all. A miss converts live via the ordinary tier
    dispatch and stores the result when it is a complete T1 parse. With
    ``allow_convert=False`` (``extract --from-cache`` / ``--no-acquire``) a
    miss returns ``(None, miss)`` instead of converting — Docling is never
    invoked on that path.

    Returns ``(doc, provenance)``; ``doc`` is ``None`` only for the
    cache-only miss case.
    """
    from fmd.extract.pdf import _is_pdf, extract_document

    if cache is None or snapshot_sha256 is None or not _is_pdf(data, content_type):
        if not allow_convert:
            # extract --from-cache / --no-acquire with nothing behind the run:
            # the document has no cached parse, and it will not be converted.
            return None, CacheMiss(None, "missing_parse_cache", "no cache available")

        policy = policy or _default_policy()
        doc = extract_document(
            data, content_type=content_type, policy=policy, t0=t0, t1=t1, t2=t2, t2b=t2b
        )
        miss = CacheMiss(None, MISS_DISABLED, "no cache configured for this document")
        miss.info["pages"] = _page_count(doc)
        return doc, miss

    doc, provenance = cache.lookup(snapshot_sha256, policy)
    if doc is not None:
        return doc, provenance

    miss = provenance
    if not allow_convert:
        return None, CacheMiss(miss.key_digest, "missing_parse_cache", miss.reason)

    policy = policy or _default_policy()
    doc = extract_document(
        data, content_type=content_type, policy=policy, t0=t0, t1=t1, t2=t2, t2b=t2b
    )
    if cacheable(doc):
        stored = cache.store_parse(snapshot_sha256, doc, policy)
        miss.stored = isinstance(stored, CacheHit)
        miss.info["stored"] = miss.stored
        miss.info["pages"] = _page_count(doc)
    else:
        miss.info["pages"] = _page_count(doc)
        miss.info["stored"] = False
        miss.info["not_cacheable_tier"] = doc.tier
    return doc, miss


def _default_policy() -> Any:

    from fmd.acquire.tiers import AcquirePolicy

    return AcquirePolicy.load()


def guard_conversion_error(exc: BaseException) -> str:
    """Stable reason for a conversion failure (used by stage summaries)."""
    if isinstance(exc, AcquisitionFailed):
        return str(getattr(exc, "reason", None) or "acquisition_failed")
    if isinstance(exc, TierUnavailable):
        return "tier_unavailable"
    return type(exc).__name__


__all__ = [
    "CACHED_TIERS",
    "CacheHit",
    "CacheMiss",
    "MISS_CORRUPT",
    "MISS_DISABLED",
    "MISS_NO_ENTRY",
    "MISS_NOT_PDF",
    "MISS_OBJECT_MISSING",
    "MISS_SCHEMA_MISMATCH",
    "PARSE_SCHEMA_VERSION",
    "ParseCache",
    "ParseCacheError",
    "acquire_document",
    "cacheable",
    "guard_conversion_error",
    "parse_key",
    "parse_policy_digest",
    "reconstruct_parse",
    "serialize_parse",
]
