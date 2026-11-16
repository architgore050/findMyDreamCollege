"""The pending-claim verification queue (``fmd verify``).

Claims are grouped by ``extraction_id`` so the verifier can reuse the source
text of the page they came from. Ordering is deterministic (extraction id, then
claim id); a run that hits its budget cap leaves the remainder ``unverified`` and
``budget_exhausted`` is surfaced in the run summary instead of being hidden.
"""

from __future__ import annotations

import logging
from typing import Any

from fmd.verify.types import VerificationRun
from fmd.verify.verifier import Verifier

logger = logging.getLogger(__name__)


def verify_pending(
    *,
    claims: Any,
    extractions: Any,
    verifier: Verifier,
    limit: int = 500,
) -> VerificationRun:
    pending = claims.list_unverified(limit=limit)
    combined = VerificationRun()
    verifier.start_run()

    grouped: dict[int, list[Any]] = {}
    for claim in pending:
        grouped.setdefault(int(claim.extraction_id), []).append(claim)

    for position, extraction_id in enumerate(sorted(grouped)):
        group = sorted(grouped[extraction_id], key=lambda claim: int(claim.id))
        extraction = extractions.get(extraction_id)
        if extraction is None:
            for claim in group:
                combined.decisions.append(
                    _unverified(claim, "missing_extraction_text", verifier)
                )
            continue
        run = verifier.verify_claims(group, text=extraction.text)
        combined.decisions.extend(run.decisions)
        combined.calls += run.calls
        combined.repairs += run.repairs
        combined.estimated_cost_usd += run.estimated_cost_usd
        combined.budget_exhausted = combined.budget_exhausted or run.budget_exhausted
        combined.error = combined.error or run.error
        if run.budget_exhausted:
            # The cap is per run: stop before reading more pages and say so for
            # every claim that was never looked at.
            remaining = [
                claim
                for later in sorted(grouped)[position + 1 :]
                for claim in sorted(grouped[later], key=lambda item: int(item.id))
            ]
            for claim in remaining:
                combined.decisions.append(
                    _unverified(claim, "budget_exhausted", verifier)
                )
            logger.warning(
                "verifier budget exhausted after %s claims; %s claims stay unverified",
                len(combined.decisions) - len(remaining),
                len(remaining),
            )
            break

    return combined


def record_summary(summary: Any, run: VerificationRun) -> None:
    """Attach verified/unverified counts to a :class:`RunSummary`."""
    counts = run.counts()
    summary.bump("considered", counts["attempted"])
    summary.bump("verified", counts["supported"])
    summary.bump("verification_rejected", counts["rejected"])
    summary.bump("unverified", counts["unverified"])
    summary.bump("verifier_calls", run.calls)
    summary.bump("verifier_repairs", run.repairs)
    summary.metadata["verifier"] = {
        "estimated_cost_usd": round(run.estimated_cost_usd, 6),
        "budget_exhausted": run.budget_exhausted,
        "error": run.error,
    }
    for decision in run.decisions:
        if decision.status == "rejected":
            summary.outcomes.append(decision.as_dict())


def _unverified(claim: Any, reason: str, verifier: Verifier) -> Any:
    from fmd.verify.types import VerificationDecision

    return VerificationDecision(
        key=f"claim-{claim.id}",
        claim_id=int(claim.id),
        status="unverified",
        reason=reason,
        prompt_version=verifier.prompt.ref,
        verified_by=verifier.model,
    )


__all__ = ["record_summary", "verify_pending"]
