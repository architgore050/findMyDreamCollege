"""Page-scoped PDF text extraction for golden curation (``fmd golden add``).

A curated golden row may only quote what a committed fixture contains verbatim,
so the tool extracts the quote from the source document itself instead of
accepting typed text. This module is the pdftext (T0) front-end for that
extraction:

* the source is a local PDF path or an already-published URL (fetched once,
  with a truthful UA; :func:`fmd.eval.curator.check` never touches network);
* exactly one page is turned into a small deterministic text fixture: one line
  per pdftext line, in pdftext ``sort=True`` order, with a blank line inserted
  before every *record header* — a serial-numbered line that starts at the
  page's left margin (:data:`RECORD_MARGIN_TOLERANCE` points around the
  smallest line x0). On government seat-matrix/fee-table PDFs those are the
  ``1 <institution>`` / ``2 <institution>`` rows, and course cells are indented,
  so the blank lines are exactly the record boundaries that
  :func:`fmd.eval.rowctx.row_block` needs for ``context_row`` proofs. A page
  without that shape generates no boundaries; ambiguous quotes then fail the
  proof and fall back to ``unverified`` (never a silent promotion);
* the fixture bytes are pure text (UTF-8) and byte-identical for identical PDF
  bytes, which is what ``fmd golden check`` re-hashes.

The import of pdftext/pdfium is lazy: ``fmd golden check`` and ``fmd golden
list`` work without the extraction extra.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from fmd.eval.curator import CuratorError

RECORD_MARGIN_TOLERANCE = 12.0
_SERIAL_HEADER = re.compile(r"^\d{1,3}(?:\.|\s|$)")
MAX_PDF_BYTES = 64 * 1024 * 1024
USER_AGENT = (
    "findMyDreamCollege-golden-curator/0.1 "
    "(golden-set curation; https://github.com/architgore050/findMyDreamCollege)"
)


@dataclass(frozen=True)
class PdfMeta:
    """Provenance of the PDF the fixture was generated from."""

    sha256: str
    page_count: int
    source: str
    tool: str = "pdftext"
    tool_version: str | None = None


@dataclass(frozen=True)
class PageText:
    """One extracted page as deterministic fixture-ready text."""

    page: int  # 1-based
    lines: tuple[str, ...]
    text: str
    line_spans: tuple[tuple[int, int], ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_pdf(url: str) -> bytes:
    """Download one PDF over HTTPS/HTTP with a truthful UA and a size cap.

    Politeness (AGENTS.md invariant 9): robots.txt is consulted first; a host
    that disallows this path is not fetched — the curator downloads the file
    manually and passes the local path instead. An unreachable or non-2xx
    robots.txt is treated as "no policy" and the single curation fetch
    proceeds (this is one human-driven fetch, never a crawl).
    """
    import urllib.robotparser
    from urllib.parse import urlsplit

    import httpx

    parts = urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        robots = httpx.get(
            robots_url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=15.0
        )
    except httpx.HTTPError:
        robots = None
    if robots is not None and robots.status_code < 400:
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(robots.text.splitlines())
        if not parser.can_fetch(USER_AGENT, url):
            raise _error(
                f"robots.txt disallows {url} for this agent; download the PDF manually "
                "and pass the local path to --source"
            )

    try:
        response = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise _error(f"could not download {url}: {exc}") from exc
    data = response.content
    if len(data) > MAX_PDF_BYTES:
        raise _error(f"{url} is larger than the {MAX_PDF_BYTES} byte curation cap")
    if not data.startswith(b"%PDF-"):
        raise _error(f"{url} did not return a PDF (missing %PDF- header)")
    return data


def _load_pdf(data: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(data)
        handle.flush()
        return Path(handle.name)


def extract_page(source: str, page: int) -> tuple[PageText, PdfMeta]:
    """Extract ``page`` (1-based) of ``source`` (local path or URL) as text."""
    if page < 1:
        raise _error(f"page is 1-based; got {page}")
    if source.startswith(("http://", "https://")):
        data = fetch_pdf(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise _error(f"source PDF does not exist: {source}")
        data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise _error(f"source is not a PDF (missing %PDF- header): {source}")

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise _error(_missing_extra()) from exc

    pdf_path = _load_pdf(data)
    try:
        document = pdfium.PdfDocument(str(pdf_path))
        try:
            page_count = len(document)
        finally:
            document.close()
        if page > page_count:
            raise _error(f"page {page} is out of range; the PDF has {page_count} page(s)")
        try:
            import pdftext.extraction as extraction
        except ImportError as exc:
            raise _error(_missing_extra()) from exc
        pages = extraction.dictionary_output(
            str(pdf_path),
            sort=True,
            keep_chars=True,
            flatten_pdf=False,
            disable_links=True,
            page_range=[page - 1],
        )
    finally:
        pdf_path.unlink(missing_ok=True)

    if not pages:
        raise _error(f"pdftext returned no page for page {page}")
    return _page_text(page, pages[0]), PdfMeta(
        sha256=sha256_bytes(data),
        page_count=page_count,
        source=source,
        tool_version=_pdftext_version(),
    )


def _page_text(page: int, raw_page: dict) -> PageText:
    """Deterministic fixture text for one pdftext page dict."""
    lines: list[tuple[float, str]] = []
    for block in raw_page.get("blocks") or []:
        for line in block.get("lines") or []:
            text = "".join(str(span.get("text") or "") for span in line.get("spans") or [])
            text = text.rstrip("\n")
            if not text.strip():
                continue
            bbox = line.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            lines.append((float(bbox[0]), text))
    if not lines:
        raise _error(f"page {page} has no extractable text (a scan needs a different workflow)")

    min_x = min(x for x, _ in lines)
    ordered: list[str] = []
    for index, (x, text) in enumerate(lines):
        if index and _SERIAL_HEADER.match(text) and x <= min_x + RECORD_MARGIN_TOLERANCE:
            ordered.append("")  # record boundary for rowctx.row_block
        ordered.append(text)

    spans: list[tuple[int, int]] = []
    chunks: list[str] = []
    cursor = 0
    for index, line in enumerate(ordered):
        if index:
            chunks.append("\n")
            cursor += 1
        spans.append((cursor, cursor + len(line)))
        chunks.append(line)
        cursor += len(line)
    text = "".join(chunks) + "\n"
    return PageText(page=page, lines=tuple(ordered), text=text, line_spans=tuple(spans))


def _pdftext_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("pdftext")
    except Exception:  # pragma: no cover - metadata is optional
        return None


def _missing_extra() -> CuratorError:  # noqa: F821
    return _error(
        "pdftext/pypdfium2 are not installed; run `uv sync --all-extras` "
        "(or `uv sync --extra extract`) before `fmd golden add`"
    )


def _error(message: str) -> Exception:
    from fmd.eval.curator import CuratorError

    return CuratorError(message)


__all__ = [
    "PdfMeta",
    "PageText",
    "RECORD_MARGIN_TOLERANCE",
    "extract_page",
    "fetch_pdf",
    "sha256_bytes",
]
