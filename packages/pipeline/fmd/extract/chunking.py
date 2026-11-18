"""Token-budgeted context windows for long documents (W4, brief §9).

Why this exists
---------------
The served context is finite (``config/models.yaml:profiles.bulk_extract.
context_tokens`` — 65536 on the host server). An annual report is 150–300
pages, far past it; before this module the whole canonical text went into one
``user_message`` and the provider answered HTTP 400 with ``request (195492
tokens) exceeds the available context size (65536)``, which was classified
``http_error`` — a network shape — so a sizing fault looked like an outage.
This module sizes windows from the profile's real budget, plans deterministic
overlapping windows over the canonical text, classifies provider failures
honestly, dedupes the claims overlapping windows agree on, and caps how many
windows one document may cost per run.

Contract with the pipeline
--------------------------
* A window is a **verbatim slice** of the canonical text built by
  ``fmd/acquire/canonical.py`` — never a rewrite. The model quotes from the
  window; validation resolves the quote against the window text exactly as it
  would in the single-pass path; the resolved offsets are shifted by the
  window's absolute start (:func:`to_absolute_offsets`) so every stored
  ``char_start``/``char_end`` is **global** in the original canonical text and
  every downstream consumer (bbox projection, page mapping, verifier windows)
  works unchanged.
* Cuts snap back to page boundaries (blank-line separated blocks), then line
  boundaries, and every window after the first starts ``overlap_chars`` before
  the previous cut, so a quote that straddles a cut is whole in at least one of
  the two neighbouring windows.
* The same fact recovered from two overlapping windows must become one claim:
  :func:`dedupe_claims` merges by ``(predicate, span)`` or by
  ``(predicate, normalised value, row context)``.
* Window planning is pure and deterministic — no clock, no network, no model —
  because resume depends on reproducing the same windows across runs.

Failure codes owned here: ``context_exceeded`` (a sizing fault, before or at
call time), ``chunk_failed:<n>`` (window ``n`` failed; ``n`` is the 1-based
window ordinal) and ``chunk_budget_exhausted`` (the per-document window cap
stopped the plan with pages still uncovered).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fmd.extract.validate import canonicalise

CONTEXT_EXCEEDED = "context_exceeded"
CHUNK_FAILED_PREFIX = "chunk_failed:"
CHUNK_BUDGET_EXHAUSTED = "chunk_budget_exhausted"

#: Lowercase signatures a provider prints when the request cannot fit the
#: served context. The first is the measured llama.cpp answer (runbook §2.7)
#: that used to be recorded as ``http_error``; the rest cover vLLM, OpenAI,
#: Anthropic and Gemini phrasings. Matched against exception text only — never
#: against a status code, which every provider shapes differently.
OVERFLOW_SIGNATURES = (
    "exceeds the available context size",
    "exceeds the context length",
    "exceeds the maximum number of tokens",
    "maximum context length",
    "context length exceeded",
    "context_length_exceeded",
    "input length exceeds",
    "too many input tokens",
    "prompt is too long",
)


class ContextExceeded(RuntimeError):
    """The text cannot fit the served context; a sizing fault, not a network fault.

    Raised before a call when the single-pass input is over budget, and re-raised
    over a provider error whose text matches :data:`OVERFLOW_SIGNATURES`, so the
    recorded reason is ``context_exceeded`` instead of ``http_error``.
    """

    reason = CONTEXT_EXCEEDED


class ChunkedExtractError(RuntimeError):
    """Every window attempted in this pass failed; nothing was persisted.

    ``reason`` is ``chunk_failed:<n>`` with ``n`` = the number of failed
    windows; the per-window outcome rows carry the precise ``chunk_failed:<n>``
    code and underlying provider reason for each ordinal.
    """

    def __init__(self, failed: int, detail: str) -> None:
        self.reason = chunk_failed_code(failed)
        super().__init__(detail)


def chunk_failed_code(window_index: int) -> str:
    """The stable failure code for window ``window_index`` (1-based ordinal)."""
    return f"{CHUNK_FAILED_PREFIX}{window_index}"


def is_context_overflow(detail: str | None) -> bool:
    """Does this provider error text say the request did not fit the context?"""
    if not detail:
        return False
    lowered = detail.lower()
    return any(signature in lowered for signature in OVERFLOW_SIGNATURES)


def classify_failure(exc: BaseException) -> str:
    """One stable reason code for a failed model call (window or single pass).

    ``ProviderError.reason`` when present; ``context_exceeded`` when the error
    text matches an overflow signature regardless of the provider's reason;
    the exception class name as the last resort. Never ``http_error`` for a
    sizing fault.
    """
    if isinstance(exc, ContextExceeded):
        return CONTEXT_EXCEEDED
    reason = getattr(exc, "reason", None)
    if reason == CONTEXT_EXCEEDED or is_context_overflow(str(exc)):
        return CONTEXT_EXCEEDED
    return str(reason) if reason else type(exc).__name__


def estimate_input_tokens(*texts: str, chars_per_token: float) -> int:
    """Conservative token estimate: characters divided by ``chars_per_token``.

    Deliberately crude — it gates the single-pass decision, and an estimate
    that is slightly high only moves a document to the chunk path a few pages
    earlier. A precise tokenizer would add a model dependency to a module that
    must stay offline.
    """
    body = sum(len(text or "") for text in texts)
    if chars_per_token <= 0:
        return body
    return math.ceil(body / chars_per_token)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk budget parameters (``config/models.yaml:chunking``).

    The sizing rule: of the served context, the output reservation
    (``output_tokens``, default the profile's ``max_tokens``) and the prompt
    headroom (``prompt_headroom_tokens``: system prompt, delimiters and
    guided-JSON instructions) are subtracted first; the remainder is converted
    to characters with the conservative ``chars_per_token``. That product is
    the maximum body a single model call may carry, whether in one pass or as
    one window.
    """

    enabled: bool = True
    context_tokens: int = 65536
    output_tokens: int = 4096
    prompt_headroom_tokens: int = 2048
    chars_per_token: float = 4.0
    overlap_tokens: int = 1024
    min_window_chars: int = 2000
    max_windows_per_document: int = 24

    @classmethod
    def from_models(cls, models: Any, profile: Any = None) -> ChunkingConfig:
        """Read ``models.chunking`` (registry extra) with profile-aware defaults.

        ``context_tokens`` prefers the chunking block (an operator may budget
        below the served ceiling), falls back to the profile's declared
        ceiling, then to 65536. ``output_tokens`` prefers the chunking block,
        then the profile's ``max_tokens``.
        """
        raw = getattr(models, "chunking", None)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        context = raw.get("context_tokens") or getattr(profile, "context_tokens", None) or 65536
        output = raw.get("output_tokens") or getattr(profile, "max_tokens", None) or 4096
        return cls(
            enabled=bool(raw.get("enabled", True)),
            context_tokens=int(context),
            output_tokens=int(output),
            prompt_headroom_tokens=int(raw.get("prompt_headroom_tokens", 2048)),
            chars_per_token=float(raw.get("chars_per_token", 4.0)),
            overlap_tokens=int(raw.get("overlap_tokens", 1024)),
            min_window_chars=int(raw.get("min_window_chars", 2000)),
            max_windows_per_document=int(raw.get("max_windows_per_document", 24)),
        )

    @property
    def usable_tokens(self) -> int:
        """Context minus output reservation minus prompt headroom."""
        return max(1, self.context_tokens - self.output_tokens - self.prompt_headroom_tokens)

    @property
    def window_chars(self) -> int:
        """Maximum canonical-text body one model call may carry."""
        return max(1, int(self.usable_tokens * self.chars_per_token))

    @property
    def overlap_chars(self) -> int:
        """Characters duplicated between consecutive windows (~1 page)."""
        return max(0, round(self.overlap_tokens * self.chars_per_token))

    def fits(self, text: str) -> bool:
        """Can this text be extracted in the existing single pass?"""
        return len(text) <= self.window_chars


# --------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class TextWindow:
    """One verbatim slice of the canonical text, with its absolute offsets.

    ``char_start``/``char_end`` are global canonical offsets (end exclusive);
    ``core_start``/``core_end`` bound the region this window alone owns — the
    overlap that precedes the core is shared with the previous window. ``pages``
    lists every page the slice touches (including overlap), ``core_pages`` only
    the pages of the owned region, so page accounting never double-counts.
    """

    index: int
    char_start: int
    char_end: int
    core_start: int
    core_end: int
    text: str
    pages: tuple[int, ...] = ()
    core_pages: tuple[int, ...] = ()

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True)
class WindowPlan:
    """A deterministic window plan over one canonical text."""

    windows: tuple[TextWindow, ...]
    window_chars: int
    overlap_chars: int
    total_chars: int
    pages_total: int
    pages_covered: int
    truncated: bool

    @property
    def window_count(self) -> int:
        return len(self.windows)


@dataclass(frozen=True)
class ChunkResume:
    """What a previous chunked run left unfinished, from its structure JSON.

    ``completed`` is keyed by ``(char_start, char_end)`` — never by window
    index — so a config change between runs cannot mistake one window for
    another. ``known`` holds the dedupe keys of every claim the prior run
    stored, so a retried window that re-quotes an overlap fact does not emit
    the claim a second time. ``has_failures`` is what makes a rerun a resume
    instead of a skip: without a failed window there is nothing to retry.
    """

    prior_extraction_id: int
    completed: Mapping[tuple[int, int], Mapping[str, Any]]
    known: tuple[set, set]
    has_failures: bool = False

    def completed_row(self, window: TextWindow) -> Mapping[str, Any] | None:
        return self.completed.get((window.char_start, window.char_end))


def _pages_for(page_map: Sequence[Any], start: int, end: int) -> tuple[int, ...]:
    """Page numbers whose char span intersects ``[start, end)``."""
    pages: list[int] = []
    for span in page_map or ():
        span_start = getattr(span, "char_start", None)
        span_end = getattr(span, "char_end", None)
        page = getattr(span, "page", None)
        if span_start is None or span_end is None or page is None:
            continue
        if span_start < end and span_end > start:
            pages.append(int(page))
    return tuple(pages)


def _cut_before(text: str, target: int, *, page_zone: int, line_zone: int) -> int:
    """Move a character-budget cut back to the latest structural boundary.

    Page boundary first (a blank-line separated block), then a single newline,
    then the raw budget cut. A cut inside a sentence would let the model quote
    a half sentence; a page-aligned cut keeps table rows and their labels
    together and makes the overlap zone a real safety margin.
    """
    page = text.rfind("\n\n", max(0, target - page_zone), target)
    if page != -1:
        return page + 2
    line = text.rfind("\n", max(0, target - line_zone), target)
    if line != -1:
        return line + 1
    return target


def _start_at_line(text: str, start: int, limit: int) -> int:
    """Move a window start forward to a line start, staying inside the overlap."""
    if start <= 0 or start >= limit:
        return start
    if text[start - 1] == "\n":
        return start
    line = text.find("\n", start, limit)
    if line != -1 and line + 1 <= limit:
        return line + 1
    return start


def plan_windows(
    text: str,
    *,
    window_chars: int,
    overlap_chars: int,
    max_windows: int,
    min_window_chars: int = 1,
    page_map: Sequence[Any] = (),
) -> WindowPlan:
    """Split canonical text into overlapping, boundary-aligned windows.

    Deterministic: the same text and parameters produce the same plan, which is
    what resume (skipping windows a prior run completed) relies on. Windows are
    capped at ``max_windows``; when the cap stops the plan with text still
    uncovered, ``truncated`` is ``True`` and ``pages_covered`` < ``pages_total``
    — the caller must report ``chunk_budget_exhausted`` rather than continue
    silently.

    A tail shorter than ``min_window_chars`` is merged into the previous window
    instead of becoming a stub call, and the overlap is clamped to half a
    window so a misconfigured overlap can never make a window larger than the
    budget it was sized from.
    """
    total = len(text)
    limit = max(1, max_windows)
    window_chars = max(1, window_chars)
    zone = min(max(0, overlap_chars), window_chars // 2)
    line_zone = max(1, min(window_chars, 2000))
    windows: list[TextWindow] = []
    prev_cut = 0
    truncated = False
    while True:
        target = prev_cut + window_chars
        if target >= total:
            start = 0 if not windows else max(0, prev_cut - zone)
            windows.append(
                _window(
                    text, len(windows) + 1, start=start, end=total, core_start=prev_cut,
                    page_map=page_map,
                )
            )
            break
        cut = _cut_before(text, target, page_zone=max(1, zone), line_zone=line_zone)
        if total - cut < max(1, min_window_chars):
            start = 0 if not windows else max(0, prev_cut - zone)
            windows.append(
                _window(
                    text, len(windows) + 1, start=start, end=total, core_start=prev_cut,
                    page_map=page_map,
                )
            )
            break
        start = 0 if not windows else _start_at_line(text, max(0, prev_cut - zone), prev_cut)
        windows.append(
            _window(text, len(windows) + 1, start=start, end=cut, core_start=prev_cut,
                    page_map=page_map)
        )
        prev_cut = cut
        if len(windows) >= limit:
            truncated = prev_cut < total
            break
    core_pages: set[int] = set()
    for window in windows:
        core_pages.update(window.core_pages)
    return WindowPlan(
        windows=tuple(windows),
        window_chars=window_chars,
        overlap_chars=zone,
        total_chars=total,
        pages_total=len(_pages_for(page_map, 0, max(total, 1))),
        pages_covered=len(core_pages),
        truncated=truncated,
    )


def _window(
    text: str,
    index: int,
    *,
    start: int,
    end: int,
    core_start: int,
    page_map: Sequence[Any],
) -> TextWindow:
    return TextWindow(
        index=index,
        char_start=start,
        char_end=end,
        core_start=core_start,
        core_end=end,
        text=text[start:end],
        pages=_pages_for(page_map, start, end),
        core_pages=_pages_for(page_map, core_start, end),
    )


# --------------------------------------------------------------------------- offsets


_OFFSET_QUALIFIER_KEYS = (
    "context_char_start",
    "context_char_end",
    "row_char_start",
    "row_char_end",
)


def to_absolute_offsets(claim: Any, offset: int) -> Any:
    """Shift window-relative spans by the window's absolute start.

    The window is a verbatim slice of the canonical text, so adding
    ``offset`` (= ``window.char_start``) lands the span on exactly the
    characters the model quoted; bbox and page projection
    (``fmd/acquire/canonical.py``) then consume the shifted span unchanged.
    Row and context spans in the qualifier shift with it — they point into the
    same canonical text.
    """
    if offset <= 0:
        return claim
    updates: dict[str, Any] = {
        "char_start": int(claim.char_start) + offset,
        "char_end": int(claim.char_end) + offset,
    }
    qualifier = getattr(claim, "qualifier", None)
    if isinstance(qualifier, Mapping):
        patched = dict(qualifier)
        for key in _OFFSET_QUALIFIER_KEYS:
            value = patched.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                patched[key] = value + offset
        updates["qualifier"] = patched
    return claim.model_copy(update=updates)


def shift_outcome(outcome: Mapping[str, Any], offset: int) -> dict[str, Any]:
    """Shift one per-field outcome dict's spans to absolute offsets."""
    shifted = dict(outcome)
    if offset <= 0:
        return shifted
    for key in ("char_start", "char_end"):
        value = shifted.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            shifted[key] = value + offset
    return shifted


# --------------------------------------------------------------------------- dedupe


def _normalised_value(value: Any) -> str:
    """Canonical string for value comparison (NFKC, grouping commas, quotes)."""
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, int | float):
        return f"{float(value):g}"
    normalised, _ = canonicalise(str(value))
    return normalised or str(value)


def _row_context(qualifier: Any) -> tuple[str, ...] | None:
    """The row a claim was attributed to, strongest identity first.

    ``(table_index, row_index)`` when the resolver located the table row, the
    row label when only the label is known, else the model's ``context`` quote.
    Two claims with the same value and the same row context are the same fact;
    identical values in different rows are not.
    """
    if not isinstance(qualifier, Mapping):
        return None
    row = qualifier.get("row")
    if isinstance(row, Mapping):
        table = row.get("table_index")
        index = row.get("row_index")
        if isinstance(table, int) and isinstance(index, int):
            return ("row", str(table), str(index))
        label = row.get("row_label")
        if label:
            return ("label", _normalised_value(label))
    context = qualifier.get("context_quote")
    if context:
        return ("context", _normalised_value(context))
    return None


def claim_dedupe_keys(claim: Any) -> tuple[tuple[Any, ...], tuple[Any, ...] | None]:
    """The dedupe identity of one claim.

    Primary key: ``(predicate, char_start, char_end)`` — the same span
    recovered from two overlapping windows. Secondary key:
    ``(predicate, normalised value, row context)`` — the same fact re-quoted
    with slightly different spans across windows. ``None`` when the claim
    carries no row context, in which case only the span key applies.
    """
    span = (str(claim.predicate), int(claim.char_start), int(claim.char_end))
    context = _row_context(getattr(claim, "qualifier", None))
    if context is None:
        return span, None
    value = _normalised_value(getattr(claim, "value", None))
    return span, (str(claim.predicate), value, *context)


def claim_keys_record(claim: Any) -> dict[str, Any]:
    """Serialisable dedupe keys, stored per window for resume-time exclusion."""
    span_key, context_key = claim_dedupe_keys(claim)
    return {
        "predicate": str(claim.predicate),
        "value": claim.value,
        "quote": getattr(claim, "quote", None),
        "char_start": int(claim.char_start),
        "char_end": int(claim.char_end),
        "span_key": list(span_key),
        "context_key": list(context_key) if context_key is not None else None,
    }


def dedupe_claims(claims: Sequence[Any]) -> tuple[list[Any], int]:
    """Merge duplicate claims from overlapping windows; keep the strongest copy.

    A later duplicate never inserts a new row: it either replaces the incumbent
    (strictly higher confidence — a window with more context resolved the same
    fact better) or is dropped. Order is first appearance, so output is
    deterministic for the same input. Returns ``(claims, dropped_count)``.
    """
    kept: list[Any] = []
    by_span: dict[tuple[Any, ...], int] = {}
    by_context: dict[tuple[Any, ...], int] = {}
    dropped = 0
    for claim in claims:
        span_key, context_key = claim_dedupe_keys(claim)
        match = by_span.get(span_key)
        if match is None and context_key is not None:
            match = by_context.get(context_key)
        if match is not None:
            dropped += 1
            incumbent = kept[match]
            if float(getattr(claim, "confidence", 0.0)) > float(
                getattr(incumbent, "confidence", 0.0)
            ):
                kept[match] = claim
            continue
        kept.append(claim)
        by_span[span_key] = len(kept) - 1
        if context_key is not None:
            by_context.setdefault(context_key, len(kept) - 1)
    return kept, dropped


def known_claim_keys(records: Iterable[Mapping[str, Any]]) -> tuple[set, set]:
    """Span/context keys from a prior run's stored ``claims_key`` records."""
    spans: set = set()
    contexts: set = set()
    for record in records or ():
        span = record.get("span_key")
        if span:
            spans.add(tuple(span))
        context = record.get("context_key")
        if context:
            contexts.add(tuple(context))
    return spans, contexts


def filter_known_claims(
    claims: Sequence[Any], known: tuple[set, set]
) -> tuple[list[Any], int]:
    """Drop claims a prior run already stored (resume must not double-emit).

    A retried window overlaps its completed neighbours, so the model may return
    the same fact again. Facts already recorded under a completed window's
    extraction are excluded here instead of being written a second time.
    """
    spans, contexts = known
    kept: list[Any] = []
    dropped = 0
    for claim in claims:
        span_key, context_key = claim_dedupe_keys(claim)
        if span_key in spans or (context_key is not None and context_key in contexts):
            dropped += 1
            continue
        kept.append(claim)
    return kept, dropped


__all__ = [
    "CHUNK_BUDGET_EXHAUSTED",
    "CHUNK_FAILED_PREFIX",
    "CONTEXT_EXCEEDED",
    "ChunkResume",
    "ChunkedExtractError",
    "ChunkingConfig",
    "ContextExceeded",
    "OVERFLOW_SIGNATURES",
    "TextWindow",
    "WindowPlan",
    "claim_dedupe_keys",
    "claim_keys_record",
    "chunk_failed_code",
    "classify_failure",
    "dedupe_claims",
    "estimate_input_tokens",
    "filter_known_claims",
    "is_context_overflow",
    "known_claim_keys",
    "plan_windows",
    "shift_outcome",
    "to_absolute_offsets",
]
