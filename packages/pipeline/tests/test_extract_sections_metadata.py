"""W8 prompt integration: the outline rides as metadata, never as text.

The breadcrumb block is appended after the delimited document text and labelled
metadata. Validation resolves quotes against the extraction text only, so a
heading string that exists only in the metadata can never resolve as a claim
quote — pinned here at both the message and the validation level.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fmd.acquire.canonical import TextBlock, build_canonical, outline_to_sections
from fmd.acquire.outline import SectionIndex
from fmd.extract.chunking import TextWindow
from fmd.extract.pipeline import (
    _document_sections_metadata,
    _window_sections_metadata,
)
from fmd.extract.prompts import load_prompt, user_message
from fmd.extract.schemas import ExtractedField, SeatMatrix
from fmd.extract.validate import validate_document

OBSERVED = datetime(2026, 11, 19, 9, 0, tzinfo=UTC)


def _block(text: str, label: str | None = None) -> TextBlock:
    return TextBlock(page_no=1, text=text, bbox=(10.0, 10.0, 200.0, 20.0), outline_label=label)


def test_user_message_appends_sections_metadata_outside_the_delimiters() -> None:
    text = "Fee schedule body.\n"
    metadata = "- 4. Fees (page 1)\n  - 4.2 Hostel (page 1)"
    message = user_message(text, document_sections=metadata)

    assert message.startswith("DOCUMENT TEXT (verbatim")
    assert "<<<\n" in message and ">>>" in message
    body_end = message.index(">>>")
    metadata_start = message.index("DOCUMENT SECTIONS (metadata only")
    # Metadata sits strictly after the delimited document text.
    assert metadata_start > body_end
    assert "4. Fees" in message[metadata_start:]
    assert metadata in message


def test_user_message_without_sections_is_unchanged() -> None:
    text = "Fee schedule body.\n"
    assert user_message(text) == user_message(text, document_sections=None)
    assert "DOCUMENT SECTIONS" not in user_message(text)


def test_heading_string_that_lives_only_in_metadata_cannot_resolve() -> None:
    # The tree title names a section the body never prints (e.g. a Docling
    # label that differs from the body): quoting it must abstain, never
    # resolve against the metadata block.
    text = "Tuition fee Rs. 85,000 per year.\n"
    canonical = build_canonical(
        [_block("4. Fees", label="section_header"), _block(text.rstrip())],
    )
    assert canonical.outline
    bounds = outline_to_sections(canonical.outline)

    document = SeatMatrix(programme_name=ExtractedField(quote="4. Fees", confidence=0.9))
    result = validate_document(
        document,
        text,
        extraction_id=1,
        entity_id=2,
        observed_at=OBSERVED,
        sections=bounds,
        section_source="outline",
    )
    rejection = result.rejections[0]
    assert rejection.field == "programme_name"
    assert rejection.reason == "unresolved_span"


def test_window_metadata_carries_breadcrumb_and_inner_headings() -> None:
    blocks = [
        _block("4. Fees", label="section_header"),
        _block("Fee schedule intro."),
        _block("4.2 Hostel", label="section_header"),
        _block("Hostel rent is charged per semester."),
        _block("5. Placements", label="section_header"),
        _block("Placement reporting is separate."),
    ]
    canonical = build_canonical(blocks)
    index = SectionIndex(canonical.outline)
    window = TextWindow(
        index=2,
        char_start=canonical.text.index("4.2 Hostel") - 10,
        char_end=canonical.text.index("5. Placements") + 30,
        core_start=canonical.text.index("4.2 Hostel"),
        core_end=canonical.text.index("5. Placements") + 30,
        text=canonical.text[
            canonical.text.index("4.2 Hostel") - 10 : canonical.text.index("5. Placements") + 30
        ],
    )
    metadata = _window_sections_metadata(index, window)

    assert metadata is not None
    assert metadata.splitlines()[0] == "window breadcrumb: 4. Fees > 4.2 Hostel"
    assert "- 4.2 Hostel (page 1)" in metadata
    assert "- 5. Placements (page 1)" in metadata
    # A document with no outline (empty index) has no metadata to add.
    assert _window_sections_metadata(SectionIndex(()), window) is None


def test_document_metadata_renders_levels_and_pages() -> None:
    blocks = [
        _block("4. Fees", label="section_header"),
        _block("Fee schedule intro."),
        _block("4.2 Hostel", label="section_header"),
        _block("Hostel rent is charged per semester."),
    ]
    canonical = build_canonical(blocks)
    metadata = _document_sections_metadata(
        type("Doc", (), {"canonical": canonical})()  # ExtractedDoc-like duck type
    )
    assert metadata == "- 4. Fees (page 1)\n  - 4.2 Hostel (page 1)"


def test_prompt_files_are_untouched_by_the_metadata_block() -> None:
    # The breadcrumb rides in the user message, not the system prompt, so no
    # prompt version bump is warranted (acceptance 5) and recorded replays
    # stay valid.
    prompt = load_prompt("fee_structure")
    assert "DOCUMENT SECTIONS" not in prompt.text
    assert prompt.version == "v4"
