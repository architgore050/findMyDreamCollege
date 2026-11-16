#!/usr/bin/env python3
"""Post one page or PDF to the OCR stage and print the blocks it returns.

    python infra/ocr/probe.py infra/ocr/fixtures/scan_sample.png
    python infra/ocr/probe.py /path/to/scan.pdf --url http://localhost:8110

URL: ``--url``, else ``$FMD_T2_URL``, else ``http://localhost:8110``. Standard
library only, so it also works outside the pipeline venv. Exit code is 1 when the
service answers non-200 (the reason is printed verbatim: it is never swallowed).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="image (.png/.jpg) or PDF to send")
    parser.add_argument(
        "--url",
        default=os.environ.get("FMD_T2_URL") or "http://localhost:8110",
        help="OCR service base URL",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    data = base64.b64encode(args.path.read_bytes()).decode("ascii")
    field = "pdf_base64" if args.path.suffix.lower() == ".pdf" else "image_base64"
    request = urllib.request.Request(
        args.url.rstrip("/") + "/extract",
        data=json.dumps({field: data}).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    blocks = payload.get("blocks") or []
    print(f"\n{len(blocks)} block(s) on {len(payload.get('pages') or [])} page(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
