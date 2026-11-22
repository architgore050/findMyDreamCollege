"""Wave v1-w5: the semantic-tier admission contract.

The deterministic rules (``fmd.plan.rules``) classify first; whatever they
refuse may be proposed by a model — but a proposal is not an admission. This
module validates model-proposed family admissions and applies only those that
survive the contract the brief fixes:

1. **Cited evidence, verbatim.** Every admission carries a quote; the quote
   must appear in the document's fetched text sample byte-for-byte. A quote
   the document does not contain is a fabrication and is refused.
2. **Known family only.** A family outside ``FAMILIES`` is refused: there is
   no schema to extract against.
3. **Uncertain is never admitted.** If two admissions propose *different*
   families for the same document, all of that document's admissions are
   refused — an ambiguous model is an abstaining model.

The validator computes nothing semantic itself: it accepts *already produced*
admissions (the model run that produces them is LOCAL-QUEUE LQ-7 — the cloud
has no model). Refusals are counted with stable reason codes so the operator's
run can be audited without re-running the model.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from fmd.extract.schemas import FAMILIES, UnknownFamily, schema_for
from fmd.plan.planner import DocumentPlan, persist_plan

#: Refusal codes (stable strings; counted in summaries and LQ-7 output).
REF_UNKNOWN_FAMILY = "unknown_family"
REF_EMPTY_QUOTE = "empty_quote"
REF_QUOTE_NOT_VERBATIM = "quote_not_verbatim"
REF_AMBIGUOUS = "ambiguous_document"
REF_NO_SAMPLE = "no_text_sample"


@dataclass(frozen=True)
class SemanticAdmission:
    """One model-proposed family decision, with its cited evidence."""

    document_url: str
    family: str
    quote: str
    #: The model's own confidence, if it emitted one. Recorded for the ledger;
    #: never used to admit or refuse — the quote is the evidence.
    score: float | None = None


@dataclass(frozen=True)
class ValidatedAdmission:
    """An admission that passed the contract; carries the evidence digest."""

    document_url: str
    family: str
    quote: str
    quote_sha256: str
    score: float | None = None


@dataclass(frozen=True)
class RefusedAdmission:
    document_url: str
    family: str
    reason: str
    detail: str | None = None


def _sample_index(samples: Mapping[str, str]) -> dict[str, str]:
    return {str(url): str(text or "") for url, text in samples.items()}


def validate_admissions(
    admissions: Iterable[SemanticAdmission],
    samples_by_url: Mapping[str, str],
) -> tuple[list[ValidatedAdmission], list[RefusedAdmission]]:
    """Apply the contract; return ``(validated, refused)`` in input order.

    ``samples_by_url`` maps document URL → fetched text sample (the same
    bounded sample the planner reads). A document with no sample cannot cite
    evidence: every admission for it is refused ``no_text_sample``.
    """
    samples = _sample_index(samples_by_url)
    by_url: dict[str, list[SemanticAdmission]] = defaultdict(list)
    ordered_urls: list[str] = []
    for admission in admissions:
        if admission.document_url not in by_url:
            ordered_urls.append(admission.document_url)
        by_url[admission.document_url].append(admission)

    validated: list[ValidatedAdmission] = []
    refused: list[RefusedAdmission] = []
    for url in ordered_urls:
        group = by_url[url]
        sample = samples.get(url)
        if sample is None:
            for admission in group:
                refused.append(
                    RefusedAdmission(
                        document_url=url,
                        family=admission.family,
                        reason=REF_NO_SAMPLE,
                        detail="no text sample was fetched for this document",
                    )
                )
            continue
        passed: list[SemanticAdmission] = []
        for admission in group:
            try:
                schema_for(admission.family)
            except UnknownFamily:
                refused.append(
                    RefusedAdmission(
                        document_url=url,
                        family=admission.family,
                        reason=REF_UNKNOWN_FAMILY,
                        detail=f"not one of {sorted(FAMILIES)}",
                    )
                )
                continue
            quote = admission.quote or ""
            if not quote.strip():
                refused.append(
                    RefusedAdmission(
                        document_url=url,
                        family=admission.family,
                        reason=REF_EMPTY_QUOTE,
                    )
                )
                continue
            if quote not in sample:
                refused.append(
                    RefusedAdmission(
                        document_url=url,
                        family=admission.family,
                        reason=REF_QUOTE_NOT_VERBATIM,
                        detail="the cited quote does not appear in the document sample",
                    )
                )
                continue
            passed.append(admission)
        families = {admission.family for admission in passed}
        if len(families) > 1:
            for admission in passed:
                refused.append(
                    RefusedAdmission(
                        document_url=url,
                        family=admission.family,
                        reason=REF_AMBIGUOUS,
                        detail="multiple families cited for one document",
                    )
                )
            continue
        for admission in passed:
            validated.append(
                ValidatedAdmission(
                    document_url=url,
                    family=admission.family,
                    quote=admission.quote,
                    quote_sha256=hashlib.sha256(admission.quote.encode("utf-8")).hexdigest(),
                    score=admission.score,
                )
            )
    return validated, refused


def apply_admissions(
    validated: Iterable[ValidatedAdmission],
    *,
    documents: Any,
    sample_for: Any,
) -> dict[str, Any]:
    """Persist validated admissions through the existing plan surface.

    The plan carries the family with ``family_source="semantic"`` and the
    quote digest in its signals; the entity is left unresolved (the document
    returns to the text queue as ``defer_for_text``) — admission is a family
    decision, never an entity claim. Unknown URLs are skipped, counted. Note
    (verifier wave v1-w5): ``signals`` live on the in-memory plan only — the
    durable audit trail is the CLI's ``admitted_detail`` JSON.
    """
    applied = 0
    missing_document = 0
    applied_urls: list[str] = []
    for admission in validated:
        document = documents.get_by_url(admission.document_url)
        if document is None:
            missing_document += 1
            continue
        sample = sample_for(admission.document_url) or ""
        plan = DocumentPlan(
            family=admission.family,
            family_source="semantic",
            entity_id=None,
            # ``confidence`` is a ledger float: an unscored admission records 0.0
            confidence=0.0 if admission.score is None else round(admission.score, 6),
            abstain_reason=None,
            defer_reason="entity_undetected",
            state="defer_for_text",
            signals={
                "semantic_admitted": True,
                "quote_sha256": admission.quote_sha256,
                "quote": admission.quote[:160],
                "sample_chars": len(sample),
            },
        )
        persist_plan(documents, document, plan)
        applied += 1
        applied_urls.append(admission.document_url)
    return {
        "applied": applied,
        "missing_document": missing_document,
        "applied_urls": applied_urls[:200],
    }
