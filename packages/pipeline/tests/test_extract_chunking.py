"""W4 context chunking: window planning, dedupe, failure codes, offset shifting.

Every case is offline and deterministic — no model, no network, no clock. The
planning tests use small budgets (a few hundred characters) so the window
arithmetic is readable; the sizing rule itself is proven by
``test_chunking_config_sizing_rule`` against the production numbers.
"""

from __future__ import annotations

from fmd.acquire.types import PageSpan
from fmd.db.repositories import ClaimCreate
from fmd.extract.chunking import (
    CONTEXT_EXCEEDED,
    ChunkedExtractError,
    ChunkingConfig,
    ContextExceeded,
    chunk_failed_code,
    claim_dedupe_keys,
    classify_failure,
    dedupe_claims,
    estimate_input_tokens,
    filter_known_claims,
    is_context_overflow,
    known_claim_keys,
    plan_windows,
    to_absolute_offsets,
)
from fmd.llm.providers import ProviderError

#: The measured llama.cpp answer that used to be recorded as ``http_error``.
MEASURED_OVERFLOW = "request (195492 tokens) exceeds the available context size (65536)"

PLAN_PARAMS = {"window_chars": 900, "overlap_chars": 80, "max_windows": 24, "min_window_chars": 200}


def _report(pages: int = 40) -> str:
    return "".join(
        f"Annual report page {number}: audit lines with numbers {number}00.\n\n"
        for number in range(1, pages + 1)
    )


def _page_map(text: str) -> list[PageSpan]:
    """The real page geometry of ``_report`` output: one page per blank-line block."""
    spans: list[PageSpan] = []
    start = 0
    page = 1
    while start < len(text):
        end = text.find("\n\n", start)
        end = len(text) if end == -1 else end + 2
        spans.append(PageSpan(page=page, char_start=start, char_end=end))
        start = end
        page += 1
    return spans


def _claim(
    predicate: str = "placement.eligible_count",
    *,
    char_start: int = 0,
    char_end: int = 5,
    value: object = 120,
    qualifier: dict | None = None,
    confidence: float = 0.9,
    extraction_id: int = 0,
) -> ClaimCreate:
    return ClaimCreate(
        extraction_id=extraction_id,
        entity_id=1,
        predicate=predicate,
        value=value,
        unit="count",
        qualifier=qualifier,
        quote=f"quote-{char_start}",
        char_start=char_start,
        char_end=char_end,
        confidence=confidence,
        method="extracted",
    )


# --------------------------------------------------------------------------- planning


def test_window_plan_covers_every_character_and_cuts_on_boundaries() -> None:
    text = _report()
    plan = plan_windows(text, **PLAN_PARAMS)

    assert plan.total_chars == len(text)
    assert plan.pages_total == 0  # no page map supplied
    # Cores tile the text exactly: every character is owned by one window,
    # overlap only duplicates, never gaps.
    assert sum(w.core_end - w.core_start for w in plan.windows) == len(text)
    previous_core = 0
    for window in plan.windows:
        assert text[window.char_start : window.char_end] == window.text
        assert window.core_start == previous_core
        previous_core = window.core_end
        # A cut never lands mid-line: the window body ends at a page or line
        # boundary (or at the raw budget cut when no boundary exists nearby).
        if window.char_end < len(text):
            assert text[window.char_end - 2 : window.char_end] == "\n\n" or text[
                window.char_end - 1
            ] == "\n"
    assert previous_core == len(text)


def test_window_plan_overlaps_consecutive_windows() -> None:
    text = _report()
    plan = plan_windows(text, **PLAN_PARAMS)

    assert plan.window_count >= 2
    for first, second in zip(plan.windows[:-1], plan.windows[1:], strict=True):
        # The second window starts before the first one ends: the boundary text
        # is duplicated, so a quote straddling the cut is whole in one of them.
        assert second.char_start < first.char_end
        assert first.char_end - second.char_start >= 1
        # and never beyond the clamp (half a window)
        assert first.char_end - second.char_start <= PLAN_PARAMS["window_chars"] // 2
        assert text[second.char_start : first.char_end] in text[first.char_start : first.char_end]


def test_window_plan_cap_stops_and_reports_uncovered_pages() -> None:
    text = _report(pages=80)
    page_map = _page_map(text)
    plan = plan_windows(
        text,
        window_chars=900,
        overlap_chars=80,
        max_windows=3,
        min_window_chars=200,
        page_map=page_map,
    )

    assert plan.window_count == 3
    assert plan.truncated is True
    assert plan.pages_covered < plan.pages_total
    assert plan.pages_covered >= 3  # the cores still cover real pages
    assert plan.pages_total == 80


def test_window_plan_merges_a_small_tail_instead_of_a_stub_window() -> None:
    text = _report(pages=6)  # ~540 chars: barely two windows at these params
    plan = plan_windows(
        text, window_chars=300, overlap_chars=40, max_windows=24, min_window_chars=200
    )

    # The tail after the first cut is smaller than min_window_chars, so the
    # last window swallows it: the plan ends on the document end, not a stub.
    assert plan.windows[-1].char_end == len(text)
    assert all(w.length >= 200 for w in plan.windows) or plan.window_count == 1


def test_window_plan_single_window_when_text_fits() -> None:
    text = _report(pages=3)
    plan = plan_windows(text, **PLAN_PARAMS)

    assert plan.window_count == 1
    assert plan.truncated is False
    assert plan.windows[0].char_start == 0
    assert plan.windows[0].char_end == len(text)


def test_window_plan_pages_use_the_page_map() -> None:
    text = _report(pages=20)
    page_map = _page_map(text)
    plan = plan_windows(
        text,
        window_chars=950,
        overlap_chars=95,
        max_windows=24,
        min_window_chars=200,
        page_map=page_map,
    )

    first = plan.windows[0]
    assert first.core_pages[0] == 1
    assert plan.pages_total == 20
    # Window cores cover consecutive page runs without double-counting: the
    # sum of per-window core pages equals the pages actually covered.
    assert sum(len(w.core_pages) for w in plan.windows) == plan.pages_covered
    covered: set[int] = set()
    for window in plan.windows:
        assert not (covered & set(window.core_pages))
        covered.update(window.core_pages)
    assert covered == set(range(1, 21))


# --------------------------------------------------------------------------- sizing


def test_chunking_config_sizing_rule() -> None:
    class Profile:
        context_tokens = 65536
        max_tokens = 4096

    class Models:
        chunking = {
            "enabled": True,
            "chars_per_token": 4.0,
            "prompt_headroom_tokens": 2048,
            "overlap_tokens": 1024,
        }
        profiles = {"bulk_extract": Profile()}

    config = ChunkingConfig.from_models(Models(), Profile())
    # usable = context 65536 - output 4096 - headroom 2048 = 59392 tokens;
    # window_chars = 59392 * 4 chars/token; overlap = 1024 * 4 chars (~1 page).
    assert config.usable_tokens == 59392
    assert config.window_chars == 237568
    assert config.overlap_chars == 4096
    assert config.fits("x" * 237568)
    assert not config.fits("x" * 237569)


def test_chunking_config_prefers_the_chunking_block_over_the_profile() -> None:
    class Profile:
        context_tokens = 65536
        max_tokens = 4096

    class Models:
        chunking = {"context_tokens": 32000, "output_tokens": 1024, "enabled": False}
        profiles = {"bulk_extract": Profile()}

    config = ChunkingConfig.from_models(Models(), Profile())
    assert config.context_tokens == 32000
    assert config.output_tokens == 1024
    assert config.enabled is False
    assert config.usable_tokens == 32000 - 1024 - 2048


def test_chunking_config_defaults_without_the_block() -> None:
    config = ChunkingConfig.from_models(object(), None)
    assert config.context_tokens == 65536
    assert config.output_tokens == 4096
    assert config.enabled is True


def test_estimate_input_tokens_is_conservative() -> None:
    assert estimate_input_tokens("a" * 4000, "b" * 400, chars_per_token=4.0) == 1100
    assert estimate_input_tokens("", chars_per_token=4.0) == 0


# --------------------------------------------------------------------------- failure codes


def test_overflow_signature_matches_the_measured_error() -> None:
    assert is_context_overflow(MEASURED_OVERFLOW)
    assert is_context_overflow("This model's maximum context length is 65536 tokens")
    assert is_context_overflow("prompt is too long: 70000 tokens > 65536 maximum")
    assert not is_context_overflow("connection reset by peer")
    assert not is_context_overflow(None)


def test_classify_failure_maps_overflow_to_context_exceeded() -> None:
    overflow = ProviderError("http_error", MEASURED_OVERFLOW)
    assert classify_failure(overflow) == CONTEXT_EXCEEDED
    timeout = ProviderError("timeout", "local: request timed out")
    assert classify_failure(timeout) == "timeout"
    assert classify_failure(ValueError("boom")) == "ValueError"
    assert classify_failure(ContextExceeded("over budget")) == CONTEXT_EXCEEDED


def test_chunk_failed_code_is_the_stable_per_window_shape() -> None:
    assert chunk_failed_code(3) == "chunk_failed:3"
    failed = ChunkedExtractError(2, "all 2 window(s) failed")
    assert failed.reason == "chunk_failed:2"


# --------------------------------------------------------------------------- dedupe


def test_dedupe_merges_the_same_span_from_overlapping_windows() -> None:
    first = _claim(char_start=100, char_end=105, confidence=0.9)
    second = _claim(char_start=100, char_end=105, confidence=0.8)
    kept, dropped = dedupe_claims([first, second])

    assert dropped == 1
    assert len(kept) == 1
    assert kept[0].confidence == 0.9  # the stronger copy wins


def test_dedupe_merges_same_value_in_the_same_row_with_different_spans() -> None:
    row = {"table_index": 2, "row_index": 5, "row_label": "Eligible students"}
    first = _claim(char_start=100, char_end=110, value=120, qualifier={"row": row})
    second = _claim(char_start=400, char_end=415, value=120, qualifier={"row": row})
    # Same fact recovered a third time, back at the second span: the dropped
    # duplicate's span was never registered, the row context still merges it.
    third = _claim(char_start=400, char_end=415, value=120, qualifier={"row": row})

    kept, dropped = dedupe_claims([first, second, third])
    assert dropped == 2
    assert len(kept) == 1
    assert kept[0].char_start == 100


def test_dedupe_keeps_distinct_rows_and_predicates() -> None:
    first = _claim(
        char_start=100, char_end=110, value=120,
        qualifier={"row": {"table_index": 1, "row_index": 1}},
    )
    second = _claim(
        char_start=400, char_end=410, value=120,
        qualifier={"row": {"table_index": 1, "row_index": 2}},
    )
    other = _claim(
        predicate="placement.placed_count",
        char_start=100,
        char_end=110,
        value=120,
        qualifier={"row": {"table_index": 1, "row_index": 1}},
    )

    kept, dropped = dedupe_claims([first, second, other])
    assert dropped == 0
    assert len(kept) == 3
    # context-less claims never merge on value alone: no row context, no merge
    anonymous_a = _claim(char_start=10, char_end=20, value=120)
    anonymous_b = _claim(char_start=900, char_end=910, value=120)
    kept_two, dropped_two = dedupe_claims([anonymous_a, anonymous_b])
    assert dropped_two == 0
    assert len(kept_two) == 2


def test_dedupe_uses_the_context_quote_when_no_row_is_known() -> None:
    # canonicalise collapses runs of whitespace, so the two contexts compare equal
    first = _claim(
        char_start=10, char_end=20, value=120, qualifier={"context_quote": "eligible students"}
    )
    second = _claim(
        char_start=910, char_end=925, value=120, qualifier={"context_quote": "eligible  students"}
    )

    kept, dropped = dedupe_claims([first, second])
    assert dropped == 1
    assert len(kept) == 1


def test_dedupe_keys_shape() -> None:
    claim = _claim(qualifier={"row": {"table_index": 1, "row_index": 2}})
    span_key, context_key = claim_dedupe_keys(claim)
    assert span_key == ("placement.eligible_count", 0, 5)
    assert context_key == ("placement.eligible_count", "120", "row", "1", "2")


# --------------------------------------------------------------------------- resume


def test_filter_known_claims_drops_what_a_prior_run_stored() -> None:
    prior = [_claim(char_start=100, char_end=110, value=120)]
    known = known_claim_keys(
        [{"span_key": list(claim_dedupe_keys(prior[0])[0]), "context_key": None}]
    )
    retried_overlap = _claim(char_start=100, char_end=110, value=120)
    fresh = _claim(char_start=800, char_end=810, value=96)

    kept, dropped = filter_known_claims([retried_overlap, fresh], known)
    assert dropped == 1
    assert [c.char_start for c in kept] == [800]


# --------------------------------------------------------------------------- offsets


def test_to_absolute_offsets_shifts_spans_and_qualifier_spans() -> None:
    claim = _claim(
        char_start=10,
        char_end=20,
        qualifier={
            "context_char_start": 5,
            "context_char_end": 9,
            "row_char_start": 2,
            "row_char_end": 25,
            "row": {"table_index": 1, "row_index": 0},
        },
    )
    absolute = to_absolute_offsets(claim, 4000)

    assert absolute.char_start == 4010
    assert absolute.char_end == 4020
    assert absolute.qualifier["context_char_start"] == 4005
    assert absolute.qualifier["context_char_end"] == 4009
    assert absolute.qualifier["row_char_start"] == 4002
    assert absolute.qualifier["row_char_end"] == 4025
    assert absolute.qualifier["row"] == {"table_index": 1, "row_index": 0}
    # the original claim is untouched (append-only evidence discipline)
    assert claim.char_start == 10


def test_to_absolute_offsets_is_identity_at_window_one() -> None:
    claim = _claim(char_start=10, char_end=20)
    assert to_absolute_offsets(claim, 0) is claim
