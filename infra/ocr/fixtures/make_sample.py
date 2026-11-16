#!/usr/bin/env python3
"""Regenerate ``infra/ocr/fixtures/scan_sample.png`` (committed, deterministic).

The fixture stands in for a scanned Indian college notice: a grey paper tone and
three short lines in mixed Latin/Devanagari, 600x800 px at 200 dpi. It is used by
``packages/pipeline/tests/test_ocr_service.py`` for the block-contract tests and
by the golden test that only runs against a live service.

    python infra/ocr/fixtures/make_sample.py

No network, no external font download: DejaVuSans is the Linux default and it
carries Devanagari, so the same bytes are produced on any host that has it.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEVANAGARI = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
LINES = (
    ("Total fee: INR 50000 per year", LATIN),
    ("Sanctioned intake: 60 seats", LATIN),
    ("शुल्क और पात्रता की जानकारी", DEVANAGARI),
)


def main() -> None:
    image = Image.new("RGB", (600, 800), (247, 245, 241))
    draw = ImageDraw.Draw(image)
    small = ImageFont.truetype(LATIN, 18)
    draw.text((48, 60), "Prospectus 2026-27", font=small, fill=(70, 70, 70))
    y = 150
    for line, path in LINES:
        draw.text((48, y), line, font=ImageFont.truetype(path, 26), fill=(20, 20, 20))
        y += 70
    target = HERE / "scan_sample.png"
    image.save(target, format="PNG", optimize=True)
    print(f"wrote {target} ({target.stat().st_size} bytes, {image.width}x{image.height})")


if __name__ == "__main__":
    main()
