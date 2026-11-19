#!/usr/bin/env python3
"""OCR engine validation — measure, never assume (W7, D18).

Runs the T2 engines over the SAME real scans and reports CER, chrF++ and
pages/min per engine, per document and aggregate. This is the deliverable that
decides the committed routing rule (runbook §3.8):

    PP-OCRv5 within ~5 chrF++ of Qwen3-VL on Devanagari
        → keep PP-OCRv5 everywhere;
    PP-OCRv5 loses by more
        → route conjunct-heavy / Devanagari pages to Qwen3-VL, Latin stays on
          PP-OCRv5, per-page engine recorded.

Ground truth is the documents' own REAL Unicode text layer (extracted with
pypdfium2 from the same file the engines read) — transcripts are never
invented. The committed seed is doc 61 (gsv.ac.in Hindi annual report,
pages 3/5/7), which carries a real Unicode text layer.

Rules (enforced here, not by goodwill):

* fewer than 3 documents with ground truth ⇒ exit non-zero;
* more than 10 documents ⇒ refused without ``--i-know`` (small batches first);
* every engine failure is a recorded reason in the report, never a silent skip;
* the measured baseline (CER 10.9 %, 25.8 pages/min ≈ 2.33 s/page on CPU,
  2026-11-19) is printed with every run so regressions are visible.

Usage
-----
    # one-time prefetch of the pinned Devanagari models (ModelScope, sha256)
    uv run python scripts/ocr_validate.py --prefetch

    # the measurement run (doc 61 pages 3/5/7 by default; DB or --pdf)
    DATABASE_URL=postgresql://fmd:fmd@localhost:5433/fmd \
        uv run python scripts/ocr_validate.py --ground-truth --engine all

    # explicit PDFs instead of the DB, capped to 3 documents
    uv run python scripts/ocr_validate.py --ground-truth \
        --documents /path/a.pdf /path/b.pdf /path/c.pdf --limit 3

Requires the runner extra on the host (rapidocr>=3.10, onnxruntime>=1.32);
Qwen3-VL additionally needs ``FMD_T2_URL`` pointing at the /extract service.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "packages" / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packages" / "pipeline"))

#: Measured on the reference host 2026-11-19 (RTX 5080 Laptop, CPU path,
#: llama-server resident). Printed with every run as the regression baseline.
BASELINE = {
    "measured_at": "2026-11-19",
    "engine": "rapidocr-ppocrv5",
    "device": "cpu",
    "cer": 0.109,
    "pages_per_min": 25.8,
    "seconds_per_page": 2.33,
    "note": "200 pp ≈ 8 min on CPU; cuda opt-in requires ~1.5–2 GiB free VRAM",
}

#: The committed ground-truth seed. ``id`` is a document id in the corpus DB;
#: ``pages`` are the pages whose real Unicode text layer is the reference.
GROUND_TRUTH_DOCS: tuple[dict[str, Any], ...] = (
    {"id": 61, "name": "gsv.ac.in Hindi annual report (doc 61)", "pages": (3, 5, 7)},
)

MAX_DOCS_WITHOUT_I_KNOW = 10
MIN_DOCS_WITH_GROUND_TRUTH = 3
ENGINES = ("rapidocr", "qwen3vl")

os.environ.setdefault("FMD_OFFLINE", "0")


# --------------------------------------------------------------------------- metrics


def _levenshtein(ref: str, hyp: str) -> int:
    """Character edit distance (two-row dynamic programming, O(len·len))."""
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current = [i]
        for j, hyp_char in enumerate(hyp, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate: edit distance / reference length (1.0 if empty)."""
    reference = " ".join(ref.split())
    hypothesis = " ".join(hyp.split())
    if not reference:
        return 1.0 if hypothesis else 0.0
    return _levenshtein(reference, hypothesis) / len(reference)


def _char_ngrams(text: str, n: int) -> Counter:
    return Counter(tuple(text[i : i + n]) for i in range(len(text) - n + 1))


def _word_ngrams(text: str, n: int) -> Counter:
    words = text.split()
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def chrf_plus_plus(ref: str, hyp: str, *, beta: float = 2.0) -> float:
    """chrF++ (Popović 2017): F-score over character n-grams (1–6) plus word
    n-grams (1–2), β=2 (recall-weighted). Statistics are summed across orders
    before a single F-score, matching sacrebleu's behaviour.

    Deterministic and dependency-free so the pipeline can test it offline.
    """
    reference = " ".join(ref.split())
    hypothesis = " ".join(hyp.split())
    if not reference and not hypothesis:
        return 1.0
    ref_total = hyp_total = overlap = 0
    for n in range(1, 7):
        ref_n, hyp_n = _char_ngrams(reference, n), _char_ngrams(hypothesis, n)
        overlap += sum((ref_n & hyp_n).values())
        ref_total += sum(ref_n.values())
        hyp_total += sum(hyp_n.values())
    for n in range(1, 3):
        ref_n, hyp_n = _word_ngrams(reference, n), _word_ngrams(hypothesis, n)
        overlap += sum((ref_n & hyp_n).values())
        ref_total += sum(ref_n.values())
        hyp_total += sum(hyp_n.values())
    if hyp_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / hyp_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


# --------------------------------------------------------------------------- documents


@dataclass
class GroundTruthDoc:
    """One document + the pages whose real text layer is the reference."""

    doc_id: int | None
    name: str
    path: Path | None = None
    pages: tuple[int, ...] = ()
    data: bytes = b""
    issues: tuple[str, ...] = field(default_factory=tuple)


def _document_pages(data: bytes) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    count = len(pdf)
    pdf.close()
    return count


def _page_text_layer(data: bytes, page_no: int) -> str:
    """The page's REAL Unicode text layer (born-digital or embedded OCR text).

    This is the reference the engines are scored against; it is read from the
    file itself and never synthesised.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        page = pdf[page_no - 1]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded() or ""
        textpage.close()
        page.close()
        return text
    finally:
        pdf.close()


def load_ground_truth_docs(
    *,
    pdfs: Sequence[Path] | None = None,
    doc_ids: Sequence[int] | None = None,
    pages: Sequence[int] | None = None,
    bytes_root: Path | None = None,
) -> list[GroundTruthDoc]:
    """Resolve documents to bytes + reference pages, recording what failed."""
    docs: list[GroundTruthDoc] = []
    if pdfs:
        for path in pdfs:
            data = path.read_bytes()
            count = _document_pages(data)
            page_list = tuple(range(1, count + 1))
            docs.append(
                GroundTruthDoc(
                    doc_id=None,
                    name=path.name,
                    path=path,
                    pages=page_list,
                    data=data,
                )
            )
        return docs

    ids = list(doc_ids or [entry["id"] for entry in GROUND_TRUTH_DOCS])
    for doc_id in ids:
        entry = next((e for e in GROUND_TRUTH_DOCS if e["id"] == doc_id), None)
        default_pages = tuple(entry["pages"]) if entry else ()
        try:
            data = _document_bytes(doc_id, bytes_root)
            count = _document_pages(data)
            page_list = tuple(pages) if pages else (default_pages or tuple(range(1, count + 1)))
            docs.append(
                GroundTruthDoc(
                    doc_id=doc_id,
                    name=f"{entry['name']} (doc {doc_id})" if entry else f"doc {doc_id}",
                    pages=page_list,
                    data=data,
                )
            )
        except Exception as exc:  # a missing document is recorded, not silent
            docs.append(
                GroundTruthDoc(
                    doc_id=doc_id,
                    name=f"doc {doc_id}",
                    issues=(f"unavailable: {type(exc).__name__}: {exc}",),
                )
            )
    return docs


def _document_bytes(doc_id: int, bytes_root: Path | None) -> bytes:
    """Snapshot bytes for a corpus document (DB ``snapshots.bytes_path``)."""
    from fmd.config.loader import get_settings
    from fmd.db.connection import Database
    from fmd.db.repositories import SnapshotRepository

    settings = get_settings()
    database_url = settings.database_url
    if not database_url:
        raise RuntimeError(
            f"doc {doc_id} needs DATABASE_URL (or pass --documents <pdf> with a local copy)"
        )
    db = Database(database_url)
    record = SnapshotRepository(db).latest_for_document(doc_id)
    if record is None:
        raise RuntimeError(f"doc {doc_id} has no snapshot in the corpus DB")
    path = Path(record.bytes_path)
    if not path.is_absolute() and bytes_root:
        path = bytes_root / path
    if not path.is_file():
        raise RuntimeError(f"snapshot bytes missing at {path}")
    return path.read_bytes()


# --------------------------------------------------------------------------- engines


@dataclass
class PageResult:
    page_no: int
    text: str
    seconds: float
    error: str | None = None


class RapidOCREngine:
    """In-process PP-OCR via the pipeline's own backend (same params as T2)."""

    name = "rapidocr"

    def __init__(self) -> None:
        from fmd.acquire.rapidocr_backend import RapidOCRBackend, RapidOCRSettings
        from fmd.acquire.tiers import AcquirePolicy

        self.settings = RapidOCRSettings.from_spec(
            AcquirePolicy.load().backend("T2"), env=dict(os.environ)
        )
        self.backend = RapidOCRBackend(self.settings)

    def run(self, data: bytes, pages: Sequence[int]) -> list[PageResult]:
        import fmd.acquire.rapidocr_backend as backend_module

        rendered = backend_module._render_pdf(
            data, dpi=self.settings.dpi, max_side_px=self.settings.max_side_px
        )
        models = self.backend.resolve_models()
        runner = self.backend._runner(models)
        results: list[PageResult] = []
        for page_no in pages:
            image = rendered[page_no - 1]
            started = time.monotonic()
            try:
                output = runner(image)
                seconds = time.monotonic() - started
                # Same confidence floor as the tier: below-floor boxes never
                # reach the transcript (and are never scored against it).
                kept = [
                    str(text)
                    for text, score in zip(output.txts or (), output.scores or (), strict=False)
                    if score >= self.settings.min_box_confidence
                ]
                results.append(PageResult(page_no=page_no, text="\n".join(kept), seconds=seconds))
            except Exception as exc:
                results.append(
                    PageResult(
                        page_no=page_no,
                        text="",
                        seconds=time.monotonic() - started,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results


class Qwen3VLEngine:
    """The Qwen3-VL /extract service (FMD_T2_URL) — the fallback engine."""

    name = "qwen3vl"

    def __init__(self) -> None:
        self.url = (os.environ.get("FMD_T2_URL") or "").strip().rstrip("/")

    def available(self) -> str | None:
        return None if self.url else "FMD_T2_URL is not set (vision service not wired)"

    def run(self, data: bytes, pages: Sequence[int]) -> list[PageResult]:
        import io

        import httpx
        import numpy as np
        from PIL import Image

        import fmd.acquire.rapidocr_backend as backend_module

        rendered = backend_module._render_pdf(data, dpi=200, max_side_px=2400)
        results: list[PageResult] = []
        for page_no in pages:
            image = rendered[page_no - 1]
            png = Image.fromarray(np.asarray(image)[:, :, ::-1])
            buffer = io.BytesIO()
            png.save(buffer, format="PNG")
            started = time.monotonic()
            try:
                response = httpx.post(
                    self.url + "/extract",
                    json={"image_base64": base64.b64encode(buffer.getvalue()).decode("ascii")},
                    timeout=300.0,
                )
                response.raise_for_status()
                payload = response.json()
                text = "\n".join(
                    str(block.get("block_content") or block.get("text") or "")
                    for block in payload.get("blocks") or []
                    if str(block.get("block_content") or block.get("text") or "").strip()
                )
                results.append(
                    PageResult(page_no=page_no, text=text, seconds=time.monotonic() - started)
                )
            except Exception as exc:
                results.append(
                    PageResult(
                        page_no=page_no,
                        text="",
                        seconds=time.monotonic() - started,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results


def build_engines(requested: Sequence[str]) -> tuple[dict[str, Any], dict[str, str]]:
    """Instantiate the requested engines; unavailable ones are recorded."""
    engines: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for name in requested:
        try:
            if name == "rapidocr":
                engines[name] = RapidOCREngine()
            elif name == "qwen3vl":
                engine = Qwen3VLEngine()
                reason = engine.available()
                if reason:
                    unavailable[name] = reason
                else:
                    engines[name] = engine
            else:  # pragma: no cover — argparse restricts choices
                unavailable[name] = f"unknown engine {name!r}"
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"
    return engines, unavailable


# --------------------------------------------------------------------------- report


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    requested = list(ENGINES) if args.engine == "all" else [args.engine]
    docs = load_ground_truth_docs(
        pdfs=[Path(p) for p in args.documents] if args.documents else None,
        doc_ids=[args.doc] if args.doc else None,
        pages=[int(p) for p in args.pages.split(",")] if args.pages else None,
        bytes_root=Path(args.bytes_root) if args.bytes_root else None,
    )
    if args.limit is not None:
        docs = docs[: args.limit]

    scored_docs = [d for d in docs if d.data and d.pages]
    if len(scored_docs) > MAX_DOCS_WITHOUT_I_KNOW and not args.i_know:
        raise SystemExit(
            f"refusing {len(scored_docs)} documents without --i-know "
            f"(batch cap {MAX_DOCS_WITHOUT_I_KNOW}: small batches first, D18)"
        )

    engines, unavailable = build_engines(requested)
    report: dict[str, Any] = {
        "baseline": BASELINE,
        "ground_truth": [
            {
                "id": d.doc_id,
                "name": d.name,
                "pages": list(d.pages),
                "issues": list(d.issues),
            }
            for d in docs
        ],
        "engines_unavailable": unavailable,
        "documents": [],
        "aggregate": {},
    }

    for doc in scored_docs:
        reference = {page_no: _page_text_layer(doc.data, page_no) for page_no in doc.pages}
        doc_row: dict[str, Any] = {
            "id": doc.doc_id,
            "name": doc.name,
            "pages": list(doc.pages),
            "reference_chars": {p: len(t) for p, t in reference.items()},
            "engines": {},
        }
        for engine_name, engine in engines.items():
            started = time.monotonic()
            page_results = engine.run(doc.data, doc.pages)
            rows = []
            cer_values: list[float] = []
            chrf_values: list[float] = []
            seconds_total = 0.0
            for result in page_results:
                ref_text = reference.get(result.page_no, "")
                row = {
                    "page": result.page_no,
                    "error": result.error,
                    "seconds": round(result.seconds, 3),
                    "ref_chars": len(" ".join(ref_text.split())),
                    "hyp_chars": len(" ".join(result.text.split())),
                }
                if result.error is None:
                    row["cer"] = round(cer(ref_text, result.text), 4)
                    row["chrf_pp"] = round(chrf_plus_plus(ref_text, result.text), 4)
                    cer_values.append(row["cer"])
                    chrf_values.append(row["chrf_pp"])
                seconds_total += result.seconds
                rows.append(row)
            elapsed = time.monotonic() - started
            doc_row["engines"][engine_name] = {
                "pages": rows,
                "cer_mean": round(sum(cer_values) / len(cer_values), 4) if cer_values else None,
                "chrf_pp_mean": round(sum(chrf_values) / len(chrf_values), 4)
                if chrf_values
                else None,
                "pages_per_min": round(len(page_results) / (seconds_total / 60.0), 2)
                if seconds_total > 0
                else None,
                "wall_seconds": round(elapsed, 2),
                "errors": [r["error"] for r in rows if r["error"]],
            }
        report["documents"].append(doc_row)

    # Aggregate per engine across documents.
    aggregate: dict[str, Any] = {}
    for engine_name in engines:
        cers: list[float] = []
        chrfs: list[float] = []
        rates: list[float] = []
        errors = 0
        for doc_row in report["documents"]:
            stats = doc_row["engines"].get(engine_name) or {}
            if stats.get("cer_mean") is not None:
                cers.append(stats["cer_mean"])
                chrfs.append(stats["chrf_pp_mean"] or 0.0)
            if stats.get("pages_per_min") is not None:
                rates.append(stats["pages_per_min"])
            errors += len(stats.get("errors") or [])
        aggregate[engine_name] = {
            "cer_mean": round(sum(cers) / len(cers), 4) if cers else None,
            "chrf_pp_mean": round(sum(chrfs) / len(chrfs), 4) if chrfs else None,
            "pages_per_min_mean": round(sum(rates) / len(rates), 2) if rates else None,
            "documents_scored": len(cers),
            "page_errors": errors,
        }
    report["aggregate"] = aggregate

    if "rapidocr" in aggregate and "qwen3vl" in aggregate:
        rr, qq = aggregate["rapidocr"], aggregate["qwen3vl"]
        if rr.get("chrf_pp_mean") is not None and qq.get("chrf_pp_mean") is not None:
            report["decision_rule"] = {
                "chrf_pp_gap_rapidocr_minus_qwen3vl": round(
                    rr["chrf_pp_mean"] - qq["chrf_pp_mean"], 4
                ),
                "rule": "gap >= -0.05 (within ~5 chrF++ of Qwen) -> keep PP-OCRv5 "
                "everywhere; a larger loss -> route conjunct-heavy/Devanagari pages "
                "to Qwen3-VL, Latin stays on PP-OCRv5, per-page engine recorded",
            }
    report["documents_with_ground_truth"] = len(scored_docs)
    return report


# --------------------------------------------------------------------------- prefetch


def prefetch(models_root: Path) -> dict[str, Any]:
    """One-time fetch of the pinned models via the runner's own resolver.

    No URLs are guessed here: ``rapidocr`` resolves the ModelScope
    ``RapidAI/RapidOCR@v3.10.0`` paths and verifies sha256 from its own
    manifest. We additionally verify the Devanagari rec hash ourselves.
    """
    from rapidocr import RapidOCR

    models_root.mkdir(parents=True, exist_ok=True)
    params: dict[str, Any] = {
        "Global.model_root_dir": str(models_root),
        "Global.log_level": "warning",
        "Det.ocr_version": "PP-OCRv6",
        "Det.model_type": "small",
        "Rec.ocr_version": "PP-OCRv5",
        "Rec.model_type": "mobile",
        "Rec.lang_type": "devanagari",
        "EngineConfig.onnxruntime.use_cuda": False,
    }
    from fmd.acquire.rapidocr_backend import _enum_params

    RapidOCR(params=_enum_params(params))  # resolves + downloads every missing pinned artifact

    pinned = {
        "devanagari_PP-OCRv5_rec_mobile.onnx": (
            "d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e"
        ),
        "ppocrv5_devanagari_dict.txt": None,
    }
    verified: dict[str, Any] = {}
    for name, expected in pinned.items():
        path = models_root / name
        if not path.is_file():
            verified[name] = "MISSING — see rapidocr download errors above"
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        verified[name] = {
            "sha256": digest,
            "matches_pin": expected is None or digest == expected,
        }
    return {"models_root": str(models_root), "verified": verified}


# --------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="measure CER/chrF++/pages-min against the real text layer",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="one-time ModelScope fetch of the pinned Devanagari models",
    )
    parser.add_argument("--engine", choices=(*ENGINES, "all"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="cap documents")
    parser.add_argument(
        "--doc",
        type=int,
        default=None,
        help="corpus document id (default: the committed seed, doc 61)",
    )
    parser.add_argument("--pages", default=None, help="comma-separated page numbers")
    parser.add_argument(
        "--documents", nargs="*", default=None, help="explicit PDF paths instead of the corpus DB"
    )
    parser.add_argument(
        "--bytes-root", default=None, help="root to join relative snapshots.bytes_path against"
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help="prefetch target / artifact root (default FMD_RAPIDOCR_MODELS)",
    )
    parser.add_argument(
        "--i-know", action="store_true", help="allow more than 10 documents in one run"
    )
    parser.add_argument("--json", action="store_true", help="JSON-only output")
    args = parser.parse_args(argv)

    if args.prefetch:
        root = Path(
            args.models_root
            or os.environ.get("FMD_RAPIDOCR_MODELS")
            or REPO_ROOT / "data" / "rapidocr-models"
        )
        result = prefetch(root)
        print(json.dumps(result, indent=2))
        return (
            0
            if all(
                isinstance(v, dict) and v.get("matches_pin", True)
                for v in result["verified"].values()
            )
            else 1
        )

    if not args.ground_truth:
        parser.print_help()
        return 2

    report = run_measurement(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    scored = report["documents_with_ground_truth"]
    if scored < MIN_DOCS_WITH_GROUND_TRUTH:
        print(
            f"\nERROR: only {scored} document(s) have ground truth; "
            f"need >= {MIN_DOCS_WITH_GROUND_TRUTH} (fix the missing documents above)",
            file=sys.stderr,
        )
        return 1
    if not report["aggregate"] or all(
        stats.get("chrf_pp_mean") is None for stats in report["aggregate"].values()
    ):
        print("\nERROR: no engine produced a scored page; nothing was measured", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


def _enum_params(params: dict) -> dict:
    """rapidocr 3.10 requires Enum values, not strings (TypeError otherwise).

    Discovered the hard way: the runner was mocked in tests, so this only surfaced
    on the first real prefetch. Maps the string config to the package's enums when
    they are importable, and leaves already-Enum values untouched.
    """
    from contextlib import suppress

    try:
        from rapidocr.utils.typings import LangRec, ModelType, OCRVersion
    except Exception:  # older runner without these enums
        return params
    mapping = {
        "Det.ocr_version": OCRVersion,
        "Rec.ocr_version": OCRVersion,
        "Det.model_type": ModelType,
        "Rec.model_type": ModelType,
        "Rec.lang_type": LangRec,
    }
    out = dict(params)
    for key, enum_cls in mapping.items():
        value = out.get(key)
        if isinstance(value, str):
            # leave invalid values alone; the runner reports a clear error
            with suppress(ValueError):
                out[key] = enum_cls(value)
    return out
