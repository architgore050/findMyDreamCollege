#!/usr/bin/env python3
"""Offline model bake-off harness for extraction (WS-MODELDOC).

The harness answers one question: *on our documents, which candidate model
publishes the most span-exact claims per unit cost?* It deliberately does not
ask a model to grade itself, and it does not re-implement metrics: every
span/value/unit judgement comes from :mod:`fmd.eval.metrics`.

Pipeline per candidate
----------------------
``PDF/text -> fmd.extract.pdf.extract_document -> family prompt + guided JSON
-> fmd.extract.validate.validate_document -> claims -> fmd.eval.metrics``

The validator is the production one. Ground truth is the same YAML schema as
``data/golden``; rows marked ``status: unverified`` are counted and excluded
from every accuracy metric (the loader enforces this, and the harness refuses to
run when there is no verified ground truth at all).

Document/ground-truth pairing
-----------------------------
Each golden entry is matched to a document by, in order:

1. an explicit mapping in ``--map id=path``;
2. ``<pdfs>/<entry.id>.pdf`` (also ``.txt``, ``.html``, ``.htm``, case-insensitive);
3. the entry's ``fixture`` path, relative to ``--root``.

Ground truth for a real PDF should use ``source_url`` + ``retrieved_at`` rather
than ``fixture``: the golden loader only verifies quotes against UTF-8 text, and
a PDF fixture would be rejected. The quote is still checked against the
extracted text at scoring time.

Offline use
-----------
``--mock`` points every candidate at the replay provider. Tests inject a fake
provider through :func:`run_bakeoff`, so no test needs a network or a real model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import yaml

from fmd.config.models import ModelsConfig, Pricing
from fmd.eval.golden import REPO_ROOT, GoldenEntry, load_golden
from fmd.eval.metrics import (
    ExtractionMetrics,
    FieldResult,
    GoldenRow,
    Prediction,
    score_field,
    summarise_fields,
)
from fmd.extract.pdf import TierPolicy, extract_document
from fmd.extract.prompts import load_prompt, user_message
from fmd.extract.schemas import schema_for
from fmd.extract.validate import Bands, validate_document
from fmd.llm.router import ModelRouter

DOCUMENT_SUFFIXES = (".pdf", ".txt", ".html", ".htm")


class BakeoffError(RuntimeError):
    """The bake-off cannot run with the given inputs."""


class Validator(Protocol):
    def __call__(
        self,
        document: Any,
        text: str,
        *,
        entity_id: int,
        bands: Bands,
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class Gates:
    """Hard gates from the inference decision document."""

    min_span_exact: float = 0.98
    max_catastrophic: float = 0.005
    min_null_precision: float = 0.95


@dataclass(frozen=True)
class Candidate:
    """One provider+model pair with optional, caller-supplied price metadata."""

    name: str
    provider: str
    model: str
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    currency: str = "USD"
    price_as_of: str | None = None
    price_source: str | None = None

    @property
    def priced(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    def pricing(self) -> Pricing | None:
        if not self.priced:
            return None
        return Pricing(
            input_per_mtok=float(self.input_per_mtok or 0.0),
            output_per_mtok=float(self.output_per_mtok or 0.0),
        )


@dataclass
class CandidateReport:
    candidate: Candidate
    pages: int = 0
    entries: int = 0
    attempted: int = 0
    schema_valid: int = 0
    predictions: int = 0
    unverified_excluded: int = 0
    metrics: ExtractionMetrics = field(default_factory=ExtractionMetrics)
    fuzzy_precision: float | None = None
    catastrophic_rate: float | None = None
    null_precision: float | None = None
    null_recall: float | None = None
    schema_valid_rate: float | None = None
    cost_usd: float | None = None
    cost_per_page: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    gates: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def span_exact_precision(self) -> float | None:
        return self.metrics.precision

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(self.gates.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": asdict(self.candidate),
            "pages": self.pages,
            "entries": self.entries,
            "attempted": self.attempted,
            "schema_valid": self.schema_valid,
            "unverified_excluded": self.unverified_excluded,
            "metrics": self.metrics.as_dict(),
            "span_exact_precision": self.span_exact_precision,
            "fuzzy_precision": self.fuzzy_precision,
            "catastrophic_rate": self.catastrophic_rate,
            "null_precision": self.null_precision,
            "null_recall": self.null_recall,
            "schema_valid_rate": self.schema_valid_rate,
            "cost_usd": self.cost_usd,
            "cost_per_page": self.cost_per_page,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "gates": self.gates,
            "passed": self.passed,
            "errors": list(self.errors),
        }


class _CostSink:
    """Collect router audit rows so cost/page uses observed token counts."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def record(self, call: Any) -> None:
        self.calls.append(call)


# --------------------------------------------------------------------------- inputs


def _models_from_file(
    path: Path | None = None, *, env: Mapping[str, str] | None = None
) -> ModelsConfig:
    import os

    from fmd.config.loader import interpolate_env

    source = path or REPO_ROOT / "config" / "models.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    environment = os.environ if env is None else env
    return ModelsConfig.model_validate(interpolate_env(raw, environment))


def _candidate_models(base: ModelsConfig, candidate: Candidate) -> ModelsConfig:
    payload = base.model_dump()
    profile = payload["profiles"]["bulk_extract"]
    profile["provider"] = candidate.provider
    profile["model"] = candidate.model
    if candidate.priced:
        profile["pricing"] = candidate.pricing().model_dump()
    return ModelsConfig.model_validate(payload)


def _load_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    if args.candidates_file:
        raw = yaml.safe_load(args.candidates_file.read_text(encoding="utf-8")) or {}
        entries = raw.get("candidates") if isinstance(raw, Mapping) else raw
        if not isinstance(entries, list):
            raise BakeoffError(f"{args.candidates_file}: expected a list or a `candidates:` list")
        for item in entries:
            if not isinstance(item, Mapping):
                raise BakeoffError(f"{args.candidates_file}: every candidate must be a mapping")
            price = item.get("price") or {}
            candidates.append(
                Candidate(
                    name=str(item["name"]),
                    provider=str(item["provider"]),
                    model=str(item["model"]),
                    input_per_mtok=_float_or_none(price.get("input_per_mtok")),
                    output_per_mtok=_float_or_none(price.get("output_per_mtok")),
                    currency=str(price.get("currency", "USD")),
                    price_as_of=price.get("as_of"),
                    price_source=price.get("source"),
                )
            )
    for spec in args.candidate or []:
        candidates.append(_parse_candidate(spec))

    prices = {spec.split("=", 1)[0]: spec.split("=", 1)[1] for spec in args.price or []}
    if prices:
        updated: list[Candidate] = []
        for candidate in candidates:
            if candidate.name not in prices:
                updated.append(candidate)
                continue
            price_spec, _, as_of = prices[candidate.name].partition("@")
            input_text, _, output_text = price_spec.partition(":")
            updated.append(
                replace(
                    candidate,
                    input_per_mtok=float(input_text),
                    output_per_mtok=float(output_text),
                    price_as_of=as_of or None,
                )
            )
        candidates = updated

    if not candidates:
        raise BakeoffError(
            "no candidates: pass --candidate NAME=PROVIDER:MODEL "
            "and/or --candidates-file"
        )
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise BakeoffError(f"duplicate candidate names: {names}")
    for candidate in candidates:
        if not candidate.provider.strip() or not candidate.model.strip():
            raise BakeoffError(
                f"candidate {candidate.name!r}: provider and model must be non-empty"
            )
    return candidates


def _parse_candidate(spec: str) -> Candidate:
    name, separator, rest = spec.partition("=")
    if not separator or ":" not in rest:
        raise BakeoffError(
            f"invalid --candidate {spec!r}; expected NAME=PROVIDER:MODEL "
            "(for example qwen3.5-9b=local:Qwen3.5-9B-AWQ)"
        )
    provider, _, model = rest.partition(":")
    return Candidate(name=name, provider=provider, model=model)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _load_documents(
    pdf_dir: Path,
    golden_entries: Sequence[GoldenEntry],
    *,
    source_map: Mapping[str, str | Path],
    root: Path,
    policy: TierPolicy,
) -> tuple[dict[str, Any], list[str]]:
    documents: dict[str, Any] = {}
    errors: list[str] = []
    for entry in golden_entries:
        try:
            path = _resolve_source(entry, pdf_dir, source_map, root)
        except BakeoffError as exc:
            errors.append(f"{entry.id}: {exc}")
            continue
        try:
            document = extract_document(
                path.read_bytes(),
                content_type=_content_type(path),
                policy=policy,
            )
        except Exception as exc:  # TierUnavailable and PDF-specific failures
            errors.append(f"{entry.id}: {path.name}: {type(exc).__name__}: {exc}")
            continue
        if document.is_empty:
            errors.append(f"{entry.id}: {path.name}: every available tier produced empty text")
            continue
        documents[entry.id] = document
    return documents, errors


def _resolve_source(
    entry: GoldenEntry,
    pdf_dir: Path,
    source_map: Mapping[str, str | Path],
    root: Path,
) -> Path:
    if entry.id in source_map:
        candidate = Path(source_map[entry.id])
        return candidate if candidate.is_absolute() else root / candidate
    for suffix in DOCUMENT_SUFFIXES:
        for name in (f"{entry.id}{suffix}", f"{entry.id}{suffix.upper()}"):
            candidate = pdf_dir / name
            if candidate.is_file():
                return candidate
    if entry.fixture:
        candidate = Path(entry.fixture)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_file() and candidate.suffix.lower() in DOCUMENT_SUFFIXES:
            return candidate
    raise BakeoffError(
        f"no document found; expected {pdf_dir / (entry.id + '.pdf')} or an explicit --map"
    )


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".html", ".htm"}:
        return "text/html"
    return "text/plain"


def _load_ground_truth(path: str | Path, root: Path) -> Any:
    """Load a golden directory, or a single YAML file, with the production loader."""
    source = Path(path)
    if not source.exists():
        raise BakeoffError(f"ground truth does not exist: {source}")
    if source.is_dir():
        return load_golden(source, root=root)
    with tempfile.TemporaryDirectory(prefix="fmd-bakeoff-golden-") as directory:
        shutil.copy2(source, Path(directory) / source.name)
        return load_golden(directory, root=root)


def _rows(entry: GoldenEntry) -> list[GoldenRow]:
    return [
        GoldenRow(
            field=name,
            value=row.value,
            unit=row.unit,
            quote=row.quote,
            status=row.status,
            expectation=row.expectation,
            printed_value=row.printed_value,
            printed_unit=row.printed_unit,
            reason=row.reason,
        )
        for name, row in entry.expected.items()
    ]


def _default_validator(
    document: Any,
    text: str,
    *,
    entity_id: int,
    bands: Bands,
) -> Sequence[Any]:
    return validate_document(
        document,
        text,
        extraction_id=0,
        entity_id=entity_id,
        bands=bands,
    ).claims


def _prediction(claim: Any) -> Prediction:
    return Prediction(
        field=str(claim.predicate),
        value=getattr(claim, "value", None),
        unit=getattr(claim, "unit", None),
        quote=getattr(claim, "quote", None),
        char_start=getattr(claim, "char_start", None),
        char_end=getattr(claim, "char_end", None),
        method=str(getattr(claim, "method", "extracted")),
    )


# --------------------------------------------------------------------------- scoring


def _summarise(
    report: CandidateReport,
    *,
    exact_results: Sequence[FieldResult],
    all_results: Sequence[FieldResult],
    sink: _CostSink,
    latencies: Sequence[float],
    gates: Gates,
) -> None:
    report.metrics = summarise_fields(
        exact_results, unverified_excluded=report.unverified_excluded
    )
    predictions = sum(1 for result in all_results if result.has_prediction)
    report.predictions = predictions

    fuzzy_hits = sum(
        1 for result in all_results if result.has_prediction and result.value_ok and result.unit_ok
    )
    report.fuzzy_precision = None if predictions == 0 else fuzzy_hits / predictions

    catastrophic = sum(1 for result in all_results if result.has_prediction and not result.value_ok)
    report.catastrophic_rate = None if predictions == 0 else catastrophic / predictions

    abstentions = sum(1 for result in all_results if result.abstained)
    correct = sum(1 for result in all_results if result.abstention_correct)
    expected_null = sum(1 for result in all_results if result.expectation == "abstain")
    report.null_precision = None if abstentions == 0 else correct / abstentions
    report.null_recall = None if expected_null == 0 else correct / expected_null

    report.schema_valid_rate = (
        None if report.attempted == 0 else report.schema_valid / report.attempted
    )

    if latencies:
        report.p50_latency_ms = _percentile(latencies, 0.50)
        report.p95_latency_ms = _percentile(latencies, 0.95)

    report.cost_usd, report.cost_per_page = _cost(sink, report.candidate, report.pages)

    span = report.metrics.precision
    cat = report.catastrophic_rate
    null = report.null_precision
    report.gates = {
        "span_exact": span is not None and span >= gates.min_span_exact,
        "catastrophic": cat is not None and cat <= gates.max_catastrophic,
        "null_precision": null is not None and null >= gates.min_null_precision,
    }


def _cost(
    sink: _CostSink, candidate: Candidate, pages: int
) -> tuple[float | None, float | None]:
    """Cost is measured only when caller-supplied prices *and* token usage exist."""
    if not candidate.priced:
        return None, None
    if not any(
        getattr(call, "tokens_in", None) is not None
        or getattr(call, "tokens_out", None) is not None
        for call in sink.calls
    ):
        return None, None
    total = sum(float(getattr(call, "cost_usd", 0.0) or 0.0) for call in sink.calls)
    return total, (None if pages == 0 else total / pages)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


# --------------------------------------------------------------------------- runner


def run_bakeoff(
    pdf_dir: str | Path,
    ground_truth: str | Path,
    candidates: Sequence[Candidate],
    *,
    models: ModelsConfig | None = None,
    policy: TierPolicy | None = None,
    bands: Bands | None = None,
    providers: Mapping[str, Any] | None = None,
    source_map: Mapping[str, str | Path] | None = None,
    root: str | Path | None = None,
    gates: Gates | None = None,
    validator: Validator | None = None,
) -> list[CandidateReport]:
    """Run every candidate over the same documents and return one report each."""
    if not candidates:
        raise BakeoffError("no candidates supplied")
    base = Path(root) if root is not None else REPO_ROOT
    docs_dir = Path(pdf_dir)
    if not docs_dir.is_dir():
        raise BakeoffError(f"document directory does not exist: {docs_dir}")

    golden = _load_ground_truth(ground_truth, base)
    if golden.verified_rows == 0:
        raise BakeoffError(
            "ground truth has zero verified rows. Refusing to invent ground truth; "
            "promote rows only when a committed fixture or a cited source proves them."
        )

    registry_models = models or _models_from_file()
    tier_policy = policy or TierPolicy.load()
    band_set = bands or Bands.load()
    active_gates = gates or Gates()
    validate = validator or _default_validator
    mapping = dict(source_map or {})

    documents, document_errors = _load_documents(
        docs_dir,
        golden.entries,
        source_map=mapping,
        root=base,
        policy=tier_policy,
    )
    if not documents:
        raise BakeoffError(
            "no documents could be read: " + "; ".join(document_errors or ["no entries"])
        )

    reports: list[CandidateReport] = []
    for candidate in candidates:
        report = CandidateReport(candidate=candidate, entries=len(golden.entries))
        report.errors.extend(document_errors)
        sink = _CostSink()
        router = ModelRouter(
            _candidate_models(registry_models, candidate),
            sink=sink,
            providers=dict(providers) if providers else None,
        )
        latencies: list[float] = []
        exact_results: list[FieldResult] = []
        all_results: list[FieldResult] = []

        for index, entry in enumerate(golden.entries, start=1):
            document = documents.get(entry.id)
            if document is None:
                continue
            report.unverified_excluded += sum(
                1 for row in entry.expected.values() if row.status != "verified"
            )
            if not entry.family:
                report.errors.append(f"{entry.id}: no extraction family")
                continue
            try:
                prompt = load_prompt(entry.family)
                schema = schema_for(entry.family)
            except Exception as exc:
                report.errors.append(f"{entry.id}: {type(exc).__name__}: {exc}")
                continue

            report.attempted += 1
            started = time.perf_counter()
            try:
                payload = router.call(
                    "bulk_extract",
                    schema,
                    prompt.text,
                    user_message(document.text),
                    input_ref=f"bakeoff:{entry.id}:{prompt.version}",
                )
            except Exception as exc:
                reason = getattr(exc, "reason", "")
                prefix = f"{type(exc).__name__}" + (f"[{reason}]" if reason else "")
                report.errors.append(f"{entry.id}: {prefix}: {exc}")
                continue
            latencies.append(_elapsed_ms(started))
            report.schema_valid += 1

            try:
                claims = validate(
                    payload, document.text, entity_id=index, bands=band_set
                )
            except Exception as exc:
                report.errors.append(f"{entry.id}: validator: {type(exc).__name__}: {exc}")
                continue

            predictions = {prediction.field: prediction for prediction in map(_prediction, claims)}
            rows = _rows(entry)
            for row in rows:
                if row.status != "verified":
                    continue
                result = score_field(row, predictions.get(row.field), document.text)
                all_results.append(result)
                if row.expectation in {"value", "abstain"}:
                    exact_results.append(result)

            report.pages += max(len(document.page_map), 1)

        _summarise(
            report,
            exact_results=exact_results,
            all_results=all_results,
            sink=sink,
            latencies=latencies,
            gates=active_gates,
        )
        reports.append(report)
    return reports


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- output


def render_table(reports: Sequence[CandidateReport], gates: Gates | None = None) -> str:
    active = gates or Gates()
    headers = [
        "candidate",
        "pages",
        "span-exact",
        "fuzzy",
        "catast.",
        "null P",
        "null R",
        "schema",
        "$/page",
        "p50 ms",
        "p95 ms",
        "gates",
        "verdict",
    ]
    rows: list[list[str]] = []
    for report in reports:
        flag = "".join(
            [
                "S+" if report.gates.get("span_exact") else "S-",
                "C+" if report.gates.get("catastrophic") else "C-",
                "N+" if report.gates.get("null_precision") else "N-",
            ]
        )
        rows.append(
            [
                report.candidate.name,
                str(report.pages),
                _pct(report.span_exact_precision),
                _pct(report.fuzzy_precision),
                _pct(report.catastrophic_rate),
                _pct(report.null_precision),
                _pct(report.null_recall),
                _pct(report.schema_valid_rate),
                _cost_per_page(report),
                _ms(report.p50_latency_ms),
                _ms(report.p95_latency_ms),
                flag,
                "PASS" if report.passed else "FAIL",
            ]
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows
    )
    lines.append("")
    lines.append(
        f"gates: span-exact >= {active.min_span_exact:.1%}  "
        f"catastrophic <= {active.max_catastrophic:.2%}  "
        f"null precision >= {active.min_null_precision:.1%}"
    )
    lines.append(
        "span-exact/fuzzy are precision over verified golden rows; unverified rows are "
        "counted but never scored. $/page is n/a unless prices and token usage were supplied."
    )
    errors = sum(len(report.errors) for report in reports)
    if errors:
        lines.append(f"warnings: {errors} per-entry errors (see --json-out for detail)")
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _cost_per_page(report: CandidateReport) -> str:
    if report.candidate.priced is False or report.cost_per_page is None:
        return "n/a"
    currency = report.candidate.currency or "USD"
    return f"{currency} {report.cost_per_page:.6f}"


def write_report(reports: Sequence[CandidateReport], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "candidates": [report.as_dict() for report in reports],
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline extraction bake-off: candidates x documents x golden rows.",
        epilog=(
            "Example: uv run python scripts/bakeoff.py --pdfs /data/bakeoff/docs "
            "--ground-truth /data/bakeoff/golden --root /data/bakeoff "
            "--candidate qwen3.5-9b=local:Qwen3.5-9B-AWQ --mock"
        ),
    )
    parser.add_argument(
        "--pdfs", required=True, type=Path, help="directory of PDF/text documents"
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="golden YAML (same schema as data/golden)",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PROVIDER:MODEL",
        help="repeatable candidate specification",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        help="YAML list of candidates (may carry price metadata)",
    )
    parser.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="NAME=IN:OUT[@YYYY-MM-DD]",
        help="attach caller-supplied USD per-million-token prices; never invented",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="explicit entry id -> document",
    )
    parser.add_argument("--mock", action="store_true", help="force the replay provider (offline)")
    parser.add_argument("--mock-fixtures", type=Path, help="replay fixtures directory for --mock")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="root for relative paths")
    parser.add_argument("--json-out", type=Path, help="write the full machine-readable report")
    parser.add_argument("--min-span-exact", type=float, default=0.98)
    parser.add_argument("--max-catastrophic", type=float, default=0.005)
    parser.add_argument("--min-null-precision", type=float, default=0.95)
    parser.add_argument(
        "--allow-no-pass",
        action="store_true",
        help="exit 0 even when every candidate fails a hard gate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidates = _load_candidates(args)
        models = _models_from_file()
        if args.mock:
            from fmd.extract.pipeline import mock_models

            models = mock_models(models, fixtures_dir=args.mock_fixtures)
            candidates = [replace(candidate, provider="mock") for candidate in candidates]
        source_map = {}
        for item in args.map:
            entry_id, separator, path = item.partition("=")
            if not separator or not entry_id or not path:
                raise BakeoffError(f"invalid --map {item!r}; expected ID=PATH")
            source_map[entry_id] = path
        reports = run_bakeoff(
            args.pdfs,
            args.ground_truth,
            candidates,
            models=models,
            source_map=source_map,
            root=args.root,
            gates=Gates(
                min_span_exact=args.min_span_exact,
                max_catastrophic=args.max_catastrophic,
                min_null_precision=args.min_null_precision,
            ),
        )
    except BakeoffError as exc:
        print(f"bakeoff: {exc}", file=sys.stderr)
        return 2

    gates = Gates(
        min_span_exact=args.min_span_exact,
        max_catastrophic=args.max_catastrophic,
        min_null_precision=args.min_null_precision,
    )
    print(render_table(reports, gates))
    if args.json_out:
        print(f"\nreport: {write_report(reports, args.json_out)}")
    if any(report.passed for report in reports) or args.allow_no_pass:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
