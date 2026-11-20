"""Tier 2 — semantic equivalence classification (14 §B2 point 2).

For programme names no rule decided: given the name, the institution, the
department/affiliation and any curriculum/regulator claims, decide
``equivalent: yes | no | uncertain`` against the *conceptual* definition in
``config/programmes.yaml`` — never against a name list.

Conservative by construction:

* no evidence and no field signal anywhere in the context ⇒ ``uncertain``
  **without a model call** (nothing to cite, so nothing to admit);
* a model ``yes`` that cites no evidence is downgraded to ``uncertain``;
* every other failure of the model path (profile unresolved, provider error,
  outage, missing fixture) degrades to ``uncertain`` — the row becomes
  ``needs_review`` and never enters the ranking;
* decisions are batched and cached by sha256 over (normalised name,
  institution type, definition version): one call per new name, not per row.
  Model, prompt version and cache flag are recorded per decision for audit.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from fmd.programmes.taxonomy import ScopeDecision

LOGGER = logging.getLogger(__name__)

#: Prompt file paired with ``classifier.prompt_version`` (AGENTS.md: bump the
#: version, keep the old file). Loaded once and interpolated with the
#: conceptual definition.
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "equivalence" / "v1.md"

#: Words whose presence anywhere in a candidate's context means the semantic
#: tier may have something to cite. This is a *pre-filter for asking the
#: model*, never an admission rule; derived domain tokens join these.
_NAME_SIGNALS: tuple[str, ...] = (
    "laborator",
    "lab",
    "medical",
    "clinical",
    "diagnostic",
    "diagnos",
    "patholog",
)

#: Generic words removed from the definition's domain tokens before they
#: become signals (they appear in every programme name).
_SIGNAL_STOPWORDS = frozenset({"clinical", "science", "sciences", "medicine", "working", "domains"})


class EquivalenceDecision(BaseModel):
    """One forced-choice answer (the guided-JSON schema per item)."""

    key: str
    equivalent: Literal["yes", "no", "uncertain"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)


class EquivalenceResponse(BaseModel):
    """The only response shape accepted from the model."""

    decisions: list[EquivalenceDecision] = Field(default_factory=list)


class EquivalenceItem(BaseModel):
    """One candidate as the prompt renders it (deterministic field order)."""

    key: str
    programme_name: str
    institution: str | None = None
    country: str | None = None
    department: str | None = None
    evidence: list[str] = Field(default_factory=list)


class DecisionRecordProtocol(Protocol):
    """What the cache store returns (duck-typed; the DB repo or a fake)."""

    cache_key: str
    decision: str
    tier: str
    confidence: float | None
    rationale: str | None
    evidence: list[str]
    model: str | None
    prompt_version: str | None
    cached: bool
    needs_review: bool
    flags: list[str]


class DecisionStore(Protocol):
    """Durable decision cache + learning loop (duck-typed for tests)."""

    def get(self, cache_key: str) -> DecisionRecordProtocol | None: ...

    def put(self, decision: Any) -> DecisionRecordProtocol | None: ...


@dataclass(frozen=True)
class EquivalenceCandidate:
    """One name awaiting a semantic decision."""

    key: str
    normalised_name: str
    raw_name: str
    institution: str | None = None
    institution_type: str | None = None
    country: str | None = None
    department: str | None = None
    evidence: tuple[str, ...] = ()


def decision_cache_key(
    normalised_name: str,
    institution_type: str | None,
    definition_version: str,
) -> str:
    """Stable semantic cache key (one call per new *name*, not per row).

    ``institution_type`` folds the entity kind and the country in
    (``"institution:IN"``): the same name may be scoped differently per
    country taxonomy, and the institution *identity* is deliberately not part
    of the key — two institutions naming a programme identically share one
    decision, which is what makes the cache a learning loop instead of a
    per-row memo.
    """
    payload = "\x00".join([normalised_name, institution_type or "", definition_version])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt(
    items: list[EquivalenceItem],
    definition_text: str,
    prompt_version: str = "v1",
) -> tuple[str, str]:
    """Deterministic (system, user) prompt for one bundle.

    The system prompt is the versioned template interpolated with the
    conceptual definition; the user payload is one line block per candidate,
    sorted by key so the same bundle always hashes the same.
    """
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    system = template.replace("{{definition}}", definition_text.strip())
    lines = [
        "Decide each item. Output JSON with a `decisions` array; each decision "
        "needs `key`, `equivalent`, `confidence`, `rationale`, `evidence`.",
        "",
    ]
    for item in sorted(items, key=lambda entry: entry.key):
        parts = [f"key={item.key}", f"programme={item.programme_name!r}"]
        if item.institution:
            parts.append(f"institution={item.institution!r}")
        if item.country:
            parts.append(f"country={item.country}")
        if item.department:
            parts.append(f"department={item.department!r}")
        lines.append("- " + " | ".join(parts))
        for entry in item.evidence:
            lines.append(f"    evidence: {entry}")
        if not item.evidence:
            lines.append("    evidence: (none)")
    if prompt_version:
        lines.append("")
        lines.append(f"(prompt_version={prompt_version})")
    return system, "\n".join(lines)


def has_field_signal(text: str, domains: list[str] | tuple[str, ...]) -> bool:
    """True when the context plausibly names the field.

    A *pre-filter*, not a decision: a candidate with no signal anywhere (name,
    institution, department, claims) has nothing the classifier could cite, so
    it is refused an admission up front and left ``uncertain``.
    """
    haystack = text.casefold()
    for signal in _NAME_SIGNALS:
        if signal in haystack:
            return True
    for domain in domains:
        for token in domain.replace("/", " ").split():
            token = token.strip().casefold()
            if len(token) >= 5 and token not in _SIGNAL_STOPWORDS and token in haystack:
                return True
    return False


def classify(
    candidates: list[EquivalenceCandidate],
    *,
    taxonomy: Any,
    config: Any,
    router: Any | None,
    store: DecisionStore | None = None,
    countries: dict[str, str | None] | None = None,
) -> tuple[dict[str, ScopeDecision], list[str]]:
    """Decide a batch of candidates; returns (decisions by key, notes).

    ``taxonomy`` is the Tier-1 :class:`~fmd.programmes.taxonomy.Taxonomy`
    (used for the definition text and country keys); ``config`` is the loaded
    ``ProgrammesConfig``; ``router`` is the provider-agnostic model router
    (may be ``None`` — every candidate then stays ``uncertain``);
    ``countries`` maps candidate key -> country for the decision record.
    """
    classifier = config.classifier
    definition = config.definition
    decisions: dict[str, ScopeDecision] = {}
    notes: list[str] = []
    if not candidates:
        return decisions, notes

    pending: list[EquivalenceCandidate] = []
    for candidate in candidates:
        record = store.get(candidate.key) if store is not None else None
        if record is not None:
            if record.tier == "manual":
                decisions[candidate.key] = _manual_decision(candidate.key, record)
                continue
            if record.decision != "uncertain" or record.cached:
                # A stored semantic decision (fresh or replayed) is reused
                # with cached=True so audits can tell calls from lookups.
                decisions[candidate.key] = _cached_decision(candidate.key, record)
                continue
        if not candidate.evidence and not has_field_signal(
            " ".join(
                part
                for part in (
                    candidate.raw_name,
                    candidate.institution or "",
                    candidate.department or "",
                )
                if part
            ),
            definition.domains,
        ):
            decisions[candidate.key] = ScopeDecision(
                in_scope=False,
                tier="semantic",
                reason="semantic_no_evidence_uncertain",
                needs_review=True,
                evidence=(),
                model=None,
                prompt_version=classifier.prompt_version,
                flags=("no_citable_evidence",),
            )
            continue
        pending.append(candidate)

    if router is None:
        if pending:
            notes.append("no model router: names left uncertain for review")
        for candidate in pending:
            decisions.setdefault(
                candidate.key,
                ScopeDecision(
                    in_scope=False,
                    tier="semantic",
                    reason="semantic_unavailable_uncertain",
                    needs_review=True,
                    prompt_version=classifier.prompt_version,
                ),
            )
        return decisions, notes

    bundles = 0
    size = max(1, classifier.bundle_size)
    model_id = _configured_model(router, classifier.purpose)
    # One call per NEW NAME, not per row: candidates that share a cache key
    # (the same normalised name at any institution of the same type) collapse
    # to one prompt line and one decision.
    unique: dict[str, EquivalenceCandidate] = {}
    for candidate in pending:
        unique.setdefault(candidate.key, candidate)
    ordered = sorted(unique.values(), key=lambda candidate: candidate.key)
    for start in range(0, len(ordered), size):
        bundle = ordered[start : start + size]
        bundles += 1
        response = _call_router(
            bundle,
            purpose=classifier.purpose,
            prompt_version=classifier.prompt_version,
            definition_text=taxonomy.definition_text(),
            router=router,
        )
        if response is None:
            notes.append(f"{classifier.purpose}: model unavailable; bundle left uncertain")
            for candidate in bundle:
                decisions.setdefault(
                    candidate.key,
                    ScopeDecision(
                        in_scope=False,
                        tier="semantic",
                        reason="semantic_unavailable_uncertain",
                        needs_review=True,
                        prompt_version=classifier.prompt_version,
                    ),
                )
            continue
        by_key = {decision.key: decision for decision in response.decisions}
        for candidate in bundle:
            raw = by_key.get(candidate.key)
            if raw is None:
                verdict = EquivalenceDecision(
                    key=candidate.key,
                    equivalent="uncertain",
                    rationale="model omitted this item",
                )
                notes.append("model omitted an item; marked uncertain")
            else:
                verdict = _validated(candidate, raw)
                if verdict is not raw:
                    notes.append(f"{candidate.key}: yes without evidence downgraded to uncertain")
            decision = _to_scope_decision(
                verdict, classifier.prompt_version, model=model_id
            )
            decisions[candidate.key] = decision
            if store is not None and classifier.cache:
                store.put(
                    {
                        "cache_key": candidate.key,
                        "normalised_name": candidate.normalised_name,
                        "raw_name": candidate.raw_name,
                        "institution_type": candidate.institution_type,
                        "country": (countries or {}).get(candidate.key),
                        "decision": _decision_word(decision),
                        "tier": "semantic",
                        "confidence": decision.confidence,
                        "rationale": decision.reason,
                        "evidence": list(decision.evidence),
                        "model": decision.model,
                        "prompt_version": decision.prompt_version,
                        "cached": False,
                        "needs_review": decision.needs_review,
                        "flags": list(decision.flags),
                        "definition_version": definition.version,
                        "decided_by": "classifier",
                    }
                )
    if bundles:
        LOGGER.info(
            "programme equivalence: %d bundle(s), %d name(s)", bundles, len(ordered)
        )
    return decisions, notes


def _validated(candidate: EquivalenceCandidate, raw: EquivalenceDecision) -> EquivalenceDecision:
    """Apply the conservative guards to one model answer.

    * a ``yes`` that cites nothing is downgraded to ``uncertain``
      (every semantic admission cites evidence — 14 §B2 guardrails);
    * confidence is clamped into [0, 1] by the schema itself.
    """
    if raw.equivalent == "yes" and not any(entry.strip() for entry in raw.evidence):
        return raw.model_copy(
            update={
                "equivalent": "uncertain",
                "rationale": (
                    raw.rationale + " — yes without citable evidence downgraded"
                ).strip(" —"),
            }
        )
    return raw


def _call_router(
    bundle: list[EquivalenceItem],
    *,
    purpose: str,
    prompt_version: str,
    definition_text: str,
    router: Any,
) -> EquivalenceResponse | None:
    """Call the router through the configured purpose; degrade to ``None``.

    Never raises: an unavailable model is a designed outcome (``uncertain``),
    not a run failure. The router itself handles provider fallbacks.
    """
    from fmd.llm import LLMError

    items = [
        EquivalenceItem(
            key=candidate.key,
            programme_name=candidate.raw_name,
            institution=candidate.institution,
            country=candidate.country,
            department=candidate.department,
            evidence=list(candidate.evidence),
        )
        for candidate in bundle
    ]
    system, user = build_prompt(items, definition_text, prompt_version)
    input_ref = ",".join(item.key for item in sorted(items, key=lambda entry: entry.key))
    try:
        result = router.call(  # type: ignore[attr-defined]
            purpose=purpose,
            schema=EquivalenceResponse,
            system=system,
            user=user,
            input_ref=input_ref,
        )
    except LLMError as exc:
        LOGGER.warning("programme equivalence: %s failed: %s", purpose, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - degrade, never abort materialisation
        LOGGER.warning("programme equivalence: %s failed: %s", purpose, exc)
        return None
    if isinstance(result, EquivalenceResponse):
        return result
    if hasattr(result, "model_dump"):
        return EquivalenceResponse.model_validate(result.model_dump())
    # A raw provider result (test doubles that bypass the router's validation):
    # its content is the parsed JSON object the router would have validated.
    return EquivalenceResponse.model_validate(result.content)


def _configured_model(router: Any, purpose: str) -> str | None:
    """The profile's configured model id for ``purpose`` (audit provenance).

    The router resolves the *served* model at call time (a provider may answer
    with a different id), but it does not expose that id to the caller. Record
    the configured profile string — the value the operator can reproduce — and
    keep the resolved-id limitation in the known gaps. A test double without a
    ``models`` block yields ``None`` rather than a guessed id.
    """
    models = getattr(router, "models", None)
    profiles = getattr(models, "profiles", None)
    if not isinstance(profiles, dict):
        return None
    profile = profiles.get(purpose)
    if profile is None:
        for rule in getattr(models, "routing_rules", []) or []:
            condition = rule.get("if") if isinstance(rule, dict) else None
            target = rule.get("use") if isinstance(rule, dict) else None
            if (
                isinstance(condition, dict)
                and condition.get("purpose") == purpose
                and all(key == "purpose" for key in condition)
                and target in profiles
            ):
                profile = profiles[target]
                break
    model = getattr(profile, "model", None)
    return str(model) if model else None


def _to_scope_decision(
    verdict: EquivalenceDecision,
    prompt_version: str,
    *,
    model: str | None,
) -> ScopeDecision:
    if verdict.equivalent == "yes":
        return ScopeDecision(
            in_scope=True,
            tier="semantic",
            reason="semantic_equivalent",
            evidence=tuple(verdict.evidence),
            confidence=verdict.confidence,
            model=model,
            prompt_version=prompt_version,
        )
    if verdict.equivalent == "no":
        return ScopeDecision(
            in_scope=False,
            tier="semantic",
            reason="semantic_not_equivalent",
            evidence=tuple(verdict.evidence),
            confidence=verdict.confidence,
            model=model,
            prompt_version=prompt_version,
        )
    return ScopeDecision(
        in_scope=False,
        tier="semantic",
        reason="semantic_uncertain",
        needs_review=True,
        evidence=tuple(verdict.evidence),
        confidence=verdict.confidence,
        model=model,
        prompt_version=prompt_version,
    )


def _decision_word(decision: ScopeDecision) -> str:
    if decision.in_scope:
        return "in_scope"
    return "uncertain" if decision.needs_review else "out_of_scope"


def _manual_decision(key: str, record: DecisionRecordProtocol) -> ScopeDecision:
    """A human decision always wins; it is replayed verbatim as tier=manual."""
    return ScopeDecision(
        in_scope=record.decision == "in_scope",
        tier="manual",
        reason=f"manual_{record.decision}",
        needs_review=False,
        evidence=tuple(record.evidence or ()),
        confidence=None,
        model=None,
        prompt_version=None,
        cached=True,
        flags=tuple(record.flags or ()),
    )


def _cached_decision(key: str, record: DecisionRecordProtocol) -> ScopeDecision:
    return ScopeDecision(
        in_scope=record.decision == "in_scope",
        tier="semantic",
        reason=record.rationale or f"semantic_{record.decision}",
        needs_review=bool(record.needs_review),
        evidence=tuple(record.evidence or ()),
        confidence=record.confidence,
        model=record.model,
        prompt_version=record.prompt_version,
        cached=True,
        flags=tuple(record.flags or ()),
    )


__all__ = [
    "DecisionStore",
    "EquivalenceCandidate",
    "EquivalenceDecision",
    "EquivalenceItem",
    "EquivalenceResponse",
    "build_prompt",
    "classify",
    "decision_cache_key",
    "has_field_signal",
]
