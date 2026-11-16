"""Stage 4 — decision bands and LLM adjudication (brief §10).

``accept`` at/above ``threshold_high``, ``reject`` at/below ``threshold_low`` and
``adjudicate`` in between. Thresholds come from :class:`ERConfig`, never module
constants. Adjudication is the only place a model is consulted, it is forced to a
same/different/unsure choice, bundles are config-sized and results are cached by
pair hash.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

from pydantic import BaseModel, Field

from fmd.er.config import AdjudicationConfig, ERConfig, Lexicon, Thresholds
from fmd.er.normalise import normalise_text
from fmd.er.types import Band, ErModel, ModelVerdict

LOGGER = logging.getLogger(__name__)


class PairDecision(BaseModel):
    """Forced-choice answer for one pair (the guided-JSON schema)."""

    pair_id: str
    verdict: Literal["same", "different", "unsure"]
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)


class AdjudicationResponse(BaseModel):
    """The only response shape accepted from the model."""

    decisions: list[PairDecision] = Field(default_factory=list)


class AdjudicationPair(ErModel):
    pair_id: str
    mention_name: str
    candidate_name: str
    score: float
    mention_state: str | None = None
    mention_city: str | None = None
    mention_pincode: str | None = None
    candidate_state: str | None = None
    candidate_city: str | None = None
    candidate_pincode: str | None = None
    features: dict[str, float | None] = Field(default_factory=dict)


class AdjudicationOutcome(ErModel):
    verdicts: dict[str, ModelVerdict] = Field(default_factory=dict)
    bundles: int = 0
    cache_hits: int = 0
    notes: tuple[str, ...] = ()


class AdjudicationCache:
    """In-memory verdict cache keyed by pair hash + config version."""

    def __init__(self, initial: dict[str, ModelVerdict] | None = None) -> None:
        self._data: dict[str, ModelVerdict] = dict(initial or {})

    def get(self, key: str) -> ModelVerdict | None:
        return self._data.get(key)

    def put(self, key: str, verdict: ModelVerdict) -> None:
        self._data[key] = verdict

    def __len__(self) -> int:
        return len(self._data)


def band_for(score: float, thresholds: Thresholds) -> Band:
    if score >= thresholds.high:
        return "accept"
    if score <= thresholds.low:
        return "reject"
    return "adjudicate"


def _sorted_form(name: str) -> str:
    return " ".join(sorted(normalise_text(name, Lexicon.default()).split()))


def pair_hash(
    mention_name: str,
    candidate_name: str,
    *,
    salt: str = "",
    state: str | None = None,
    pincode: str | None = None,
) -> str:
    """Stable, order-insensitive hash of a name pair plus disambiguating context."""
    ordered = sorted((_sorted_form(mention_name), _sorted_form(candidate_name)))
    payload = "\x00".join([*ordered, (state or "").casefold(), pincode or "", salt])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt(
    pairs: list[AdjudicationPair],
    config: AdjudicationConfig | None = None,
) -> tuple[str, str]:
    """Deterministic (system, user) prompt for a bundle of pairs."""
    cfg = config or AdjudicationConfig()
    system = f"{cfg.system_prompt}\n\n{cfg.instructions}"
    lines = [
        "Decide each pair. Use the names and any place context. Output JSON with a "
        "`decisions` array; each decision needs `pair_id`, `verdict`, `reason`, "
        "`evidence`.",
        "",
    ]
    for pair in sorted(pairs, key=lambda item: item.pair_id):
        context = ", ".join(
            part
            for part in (
                pair.mention_city or pair.candidate_city,
                pair.mention_state or pair.candidate_state,
                pair.mention_pincode or pair.candidate_pincode,
            )
            if part
        )
        lines.append(
            f"- pair_id={pair.pair_id} | mention={pair.mention_name!r} | "
            f"candidate={pair.candidate_name!r} | similarity={pair.score:.4f} | "
            f"context={context or 'none'}"
        )
    user = "\n".join(lines)
    return system, user


def adjudicate(
    pairs: list[AdjudicationPair],
    *,
    config: ERConfig | None = None,
    router: object | None = None,
    cache: AdjudicationCache | None = None,
    logger: logging.Logger | None = None,
) -> AdjudicationOutcome:
    """Ask the model about the ambiguous pairs; degrade to ``unsure`` on failure."""
    cfg = config or ERConfig()
    adjudication = cfg.adjudication
    log = logger or LOGGER
    if not pairs:
        return AdjudicationOutcome()
    if not adjudication.enabled:
        return AdjudicationOutcome(notes=("adjudication disabled by config",))
    if router is None:
        return AdjudicationOutcome(notes=("no model router: pairs left for manual review",))

    verdicts: dict[str, ModelVerdict] = {}
    notes: list[str] = []
    cache_hits = 0
    unique = {pair.pair_id: pair for pair in pairs}
    pending: list[AdjudicationPair] = []
    for pair_id, pair in sorted(unique.items()):
        cached = cache.get(pair_id) if cache is not None and adjudication.cache else None
        if cached is not None:
            verdicts[pair_id] = cached.model_copy(update={"cached": True})
            cache_hits += 1
        else:
            pending.append(pair)

    bundles = 0
    size = max(1, adjudication.bundle_size)
    for start in range(0, len(pending), size):
        bundle = pending[start : start + size]
        bundles += 1
        response, purpose = _call_router(bundle, adjudication, router)
        if response is None:
            notes.append(f"{purpose}: model unavailable; bundle left for manual review")
            continue
        by_id = {decision.pair_id: decision for decision in response.decisions}
        for pair in bundle:
            decision = by_id.get(pair.pair_id)
            if decision is None:
                verdict = ModelVerdict(
                    pair_id=pair.pair_id,
                    verdict="unsure",
                    reason="model omitted this pair",
                )
                notes.append("model omitted a pair; marked unsure")
            else:
                verdict = ModelVerdict(
                    pair_id=decision.pair_id,
                    verdict=decision.verdict,
                    reason=decision.reason,
                    evidence=tuple(decision.evidence),
                )
            verdicts[pair.pair_id] = verdict
            if cache is not None and adjudication.cache:
                cache.put(pair.pair_id, verdict)
    if bundles:
        log.info("ER adjudication: %d bundle(s), %d pair(s)", bundles, len(pending))
    return AdjudicationOutcome(
        verdicts=verdicts,
        bundles=bundles,
        cache_hits=cache_hits,
        notes=tuple(dict.fromkeys(notes)),
    )


def _call_router(
    bundle: list[AdjudicationPair],
    config: AdjudicationConfig,
    router: object,
) -> tuple[AdjudicationResponse | None, str]:
    """Call the router with the configured purpose, then its declared fallbacks."""
    from fmd.llm import LLMError, ProfileUnresolved

    system, user = build_prompt(bundle, config)
    input_ref = ",".join(pair.pair_id for pair in bundle)
    purposes = [config.purpose, *config.fallback_purposes]
    for purpose in purposes:
        try:
            result = router.call(  # type: ignore[attr-defined]
                purpose=purpose,
                schema=AdjudicationResponse,
                system=system,
                user=user,
                input_ref=input_ref,
            )
        except ProfileUnresolved:
            continue
        except LLMError as exc:
            LOGGER.warning("ER adjudication: %s failed: %s", purpose, exc)
            return None, purpose
        if isinstance(result, AdjudicationResponse):
            return result, purpose
        return AdjudicationResponse.model_validate(result.model_dump()), purpose
    return None, config.purpose


__all__ = [
    "AdjudicationCache",
    "AdjudicationOutcome",
    "AdjudicationPair",
    "AdjudicationResponse",
    "PairDecision",
    "adjudicate",
    "band_for",
    "build_prompt",
    "pair_hash",
]
