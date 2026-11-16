"""Committed fabrication fixtures for the verifier (tests and ``fmd eval``).

A case is a JSON manifest plus the page text it was extracted from:

    tests/fixtures/verify/fabrication_v1/
      case.json          # claims + expected labels, quotes only
      page.txt           # the source window
      llm/verify/*.json  # recorded verdicts, keyed by prompt hash

The manifest stores quotes, not offsets: offsets are resolved deterministically
against the page and must be unique, which is the same rule extraction uses. A
claim whose quote cannot be resolved is a broken fixture, not a verifier result.

The expected labels are the *ground truth about the extraction output*:
``correct`` (the page supports it), ``fabricated`` (plausible field the page
never states) and ``wrong_field`` (a real fact in the wrong column). This proves
the plumbing: verifier call -> forced verdict -> selective repair -> persistence
-> metrics. It does not prove the model generalises.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fmd.verify.metrics import VerificationMetrics, verification_metrics
from fmd.verify.types import ClaimView, VerificationRun
from fmd.verify.verifier import Verifier


class CaseError(RuntimeError):
    """The fixture is malformed or its quote does not resolve."""


@dataclass(frozen=True)
class CaseClaim:
    view: ClaimView
    expected: str
    unit: str | None = None


@dataclass(frozen=True)
class VerificationCase:
    case_id: str
    directory: Path
    text: str
    claims: list[CaseClaim]

    @property
    def fixtures_dir(self) -> Path:
        return self.directory / "llm"

    @property
    def expected(self) -> dict[str, str]:
        return {claim.view.key: claim.expected for claim in self.claims}


def load_case(directory: str | Path) -> VerificationCase:
    root = Path(directory)
    manifest_path = root / "case.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseError(f"cannot read {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CaseError(f"{manifest_path}: manifest must be an object")
    text_name = str(manifest.get("text_file") or "page.txt")
    try:
        text = (root / text_name).read_text(encoding="utf-8")
    except OSError as exc:
        raise CaseError(f"cannot read {root / text_name}: {exc}") from exc

    claims: list[CaseClaim] = []
    for raw in manifest.get("claims") or []:
        if not isinstance(raw, dict):
            raise CaseError(f"{manifest_path}: each claim must be an object")
        key = str(raw.get("key") or "").strip()
        quote = str(raw.get("quote") or "")
        predicate = str(raw.get("predicate") or "").strip()
        expected = str(raw.get("expected") or "correct")
        if not key or not predicate or not quote:
            raise CaseError(f"{manifest_path}: claim needs key, predicate and quote")
        if expected not in {"correct", "fabricated", "wrong_field"}:
            raise CaseError(f"{manifest_path}: {key}: unknown expected label {expected!r}")
        start = text.find(quote)
        if start < 0:
            raise CaseError(f"{manifest_path}: {key}: quote does not occur in {text_name}")
        if text.find(quote, start + 1) >= 0:
            raise CaseError(
                f"{manifest_path}: {key}: quote occurs more than once in {text_name}; "
                "use a longer verbatim quote"
            )
        claims.append(
            CaseClaim(
                view=ClaimView(
                    key=key,
                    predicate=predicate,
                    quote=quote,
                    char_start=start,
                    char_end=start + len(quote),
                    value=raw.get("value"),
                ),
                expected=expected,
                unit=raw.get("unit"),
            )
        )
    return VerificationCase(
        case_id=str(manifest.get("case_id") or root.name),
        directory=root,
        text=text,
        claims=claims,
    )


def run_case(
    case: VerificationCase, verifier: Verifier
) -> tuple[VerificationRun, VerificationMetrics]:
    verifier.start_run()
    run = verifier.verify_claims(
        [claim.view for claim in case.claims],
        text=case.text,
        persist=False,
    )
    metrics = verification_metrics(run.decisions, case.expected)
    return run, metrics


def evaluate_fixture_dir(directory: str | Path, registry: Any) -> dict[str, Any]:
    """Run one committed case through the offline replay provider (``fmd eval``)."""
    from fmd.extract.pipeline import build_router

    case = load_case(directory)
    router = build_router(
        registry.models, mock=True, fixtures_dir=str(case.fixtures_dir)
    )
    verifier = Verifier(router)
    run, metrics = run_case(case, verifier)
    payload = metrics.as_dict()
    payload.update(
        {
            "case": case.case_id,
            "calls": run.calls,
            "repairs": run.repairs,
            "budget_exhausted": run.budget_exhausted,
            "error": run.error,
        }
    )
    return payload


__all__ = [
    "CaseClaim",
    "CaseError",
    "VerificationCase",
    "evaluate_fixture_dir",
    "load_case",
    "run_case",
]
