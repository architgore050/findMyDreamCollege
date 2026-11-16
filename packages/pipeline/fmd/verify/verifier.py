"""The second-pass verifier (brief §9, §15).

Contract:

* it sees a **bounded source window** around the claim span (``window_chars``
  from config) plus the existing ``quote`` — never the claimed value;
* it returns one forced verdict per field: ``supported | contradicted |
  absent | wrong_field``, optionally with a verbatim ``corrected_quote``;
* several fields share a call only when their windows are *identical*, so the
  model never has to reason across pages;
* ``contradicted`` / ``wrong_field`` get **one** single-field repair call; a
  second bad verdict rejects the claim. ``absent`` rejects immediately (the
  field should have been null);
* a per-run claim and spend cap from config stops work honestly: remaining
  claims are marked ``unverified``, not ``supported``.

The verifier never emits a value. Claim content is immutable; a decision is
persisted through :class:`fmd.db.repositories.ClaimVerification` only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fmd.llm import LLMError
from fmd.verify.policy import VerifyPolicy
from fmd.verify.prompts import Prompt, load_prompt
from fmd.verify.schemas import FieldJudgement, VerdictBundle
from fmd.verify.types import (
    VERDICTS,
    ClaimView,
    Verdict,
    VerificationDecision,
    VerificationRun,
)

logger = logging.getLogger(__name__)

PURPOSE = "verify"
WORD_BREAK = set(" \t\r\n")


class VerifierUnavailable(RuntimeError):
    """The verifier profile is missing, so claims must stay unverified."""


@dataclass
class Bundle:
    start: int
    end: int
    window: str
    views: list[ClaimView] = field(default_factory=list)


# --------------------------------------------------------------------------- windowing


def bounded_window(
    text: str,
    start: int,
    end: int,
    *,
    chars: int,
) -> tuple[int, int, str]:
    """Return ``(start, end, text)`` for a window bounded to about ``chars``.

    The span itself is always contained and never truncated. Windows are
    expanded symmetrically, then outward to the nearest whitespace so a quote is
    not cut mid-word; the result can exceed ``chars`` by at most one word at each
    edge. Empty/negative bounds are clamped, never raised.
    """
    if chars <= 0:
        raise ValueError("window_chars must be > 0")
    length = len(text)
    if chars >= length:
        return 0, length, text
    start = max(0, min(int(start), length))
    end = max(start, min(int(end), length))
    span = end - start
    if span >= chars:
        return start, end, text[start:end]

    extra = chars - span
    left = extra // 2
    right = extra - left
    lo = max(0, start - left)
    hi = min(length, end + right)
    # Rebalance when clamped at one edge so the window keeps its size.
    if lo == 0:
        hi = min(length, hi + (start - lo))
    if hi == length:
        lo = max(0, lo - (length - hi))
    # Snap outward to whitespace, at most one word per edge.
    while lo > 0 and text[lo - 1] not in WORD_BREAK:
        lo -= 1
    while hi < length and text[hi] not in WORD_BREAK:
        hi += 1
    return lo, hi, text[lo:hi]


def bundle_claims(
    views: list[ClaimView], text: str, *, chars: int, max_fields: int
) -> list[Bundle]:
    """Group claims deterministically, only when their windows are identical."""
    ordered = sorted(
        views,
        key=lambda view: (view.char_start, view.char_end, view.predicate, view.key),
    )
    bundles: list[Bundle] = []
    for view in ordered:
        start, end, window = bounded_window(
            text, view.char_start, view.char_end, chars=chars
        )
        last = bundles[-1] if bundles else None
        if last is not None and (last.start, last.end) == (start, end) and (
            len(last.views) < max_fields
        ):
            last.views.append(view)
        else:
            bundles.append(Bundle(start=start, end=end, window=window, views=[view]))
    return bundles


# --------------------------------------------------------------------------- messages


def bundle_message(bundle: Bundle) -> str:
    payload = {
        "fields": [
            {"key": view.key, "field": view.predicate, "quote": view.quote}
            for view in bundle.views
        ]
    }
    return (
        "SOURCE WINDOW (verbatim; the delimiters are not content):\n"
        "<<<\n"
        f"{bundle.window}\n"
        ">>>\n\n"
        "FIELDS TO JUDGE (JSON; the claimed value is intentionally withheld):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def repair_message(view: ClaimView, window: str, previous: str) -> str:
    payload = {"key": view.key, "field": view.predicate, "quote": view.quote}
    return (
        "SOURCE WINDOW (verbatim; the delimiters are not content):\n"
        "<<<\n"
        f"{window}\n"
        ">>>\n\n"
        "FIELD (single; the claimed value is intentionally withheld):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + f"\n\nPREVIOUS VERDICT: {previous}"
    )


def _input_ref(bundle: Bundle) -> str:
    digest = hashlib.sha256(bundle.window.encode("utf-8")).hexdigest()[:16]
    keys = ",".join(view.key for view in bundle.views)
    return f"{digest}:{bundle.start}:{bundle.end}:{keys}"


# --------------------------------------------------------------------------- verifier


class Verifier:
    def __init__(
        self,
        router: Any,
        *,
        claims: Any | None = None,
        policy: VerifyPolicy | None = None,
    ) -> None:
        self.router = router
        models = getattr(router, "models", None)
        resolved = policy if policy is not None else VerifyPolicy.load(models)
        if resolved is None:
            raise VerifierUnavailable(
                "no `verify` profile in config/models.yaml; claims cannot be verified"
            )
        self.policy = resolved
        self.prompt: Prompt = load_prompt(resolved.prompt_name, resolved.prompt_version)
        self.repair_prompt: Prompt = load_prompt("verify_repair", resolved.prompt_version)
        self.claims = claims
        profile = models.profiles.get("verify") if models is not None else None
        self.model = getattr(profile, "model", None)
        self._spent = 0.0
        self._claimed = 0
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    def start_run(self) -> None:
        """Reset the per-run claim and spend caps. Called once per stage run."""
        self._spent = 0.0
        self._claimed = 0
        self._started = True

    # --------------------------------------------------------------------- public
    def verify_claims(
        self,
        claims: list[Any],
        *,
        text: str,
        persist: bool = True,
    ) -> VerificationRun:
        if not self._started:
            self.start_run()
        run = VerificationRun()
        views = [
            claim
            if isinstance(claim, ClaimView)
            else ClaimView.from_claim(claim, key=_key_for(claim, index))
            for index, claim in enumerate(claims)
        ]
        if not views:
            return run
        if not text:
            for view in views:
                run.decisions.append(
                    self._decision(view, status="unverified", reason="missing_source_text")
                )
            self._persist(run, persist=persist)
            return run

        bundles = bundle_claims(
            views,
            text,
            chars=self.policy.window_chars,
            max_fields=self.policy.bundle_fields,
        )
        for index, bundle in enumerate(bundles):
            if not self._can_verify(len(bundle.views)):
                run.budget_exhausted = True
                for remaining in bundles[index:]:
                    for view in remaining.views:
                        run.decisions.append(
                            self._decision(
                                view, status="unverified", reason="budget_exhausted"
                            )
                        )
                break
            judgements = self._ask_bundle(bundle, run)
            if judgements is None:
                for remaining in bundles[index:]:
                    for view in remaining.views:
                        run.decisions.append(
                            self._decision(
                                view,
                                status="unverified",
                                reason=f"verifier_unavailable:{run.error or 'error'}",
                            )
                        )
                break
            for view in bundle.views:
                decision = self._decide(view, judgements.get(view.key), bundle.window, run)
                run.decisions.append(decision)

        self._persist(run, persist=persist)
        return run

    # ------------------------------------------------------------------- internals
    def _can_verify(self, fields: int) -> bool:
        if (
            self.policy.max_claims_per_run > 0
            and self._claimed + fields > self.policy.max_claims_per_run
        ):
            return False
        return not (
            self.policy.max_usd_per_run > 0
            and self._spent + self.policy.cost_per_call > self.policy.max_usd_per_run + 1e-12
        )
    def _ask_bundle(
        self, bundle: Bundle, run: VerificationRun
    ) -> dict[str, FieldJudgement] | None:
        try:
            payload = self.router.call(
                PURPOSE,
                VerdictBundle,
                self.prompt.text,
                bundle_message(bundle),
                input_ref=_input_ref(bundle),
            )
        except LLMError as exc:
            run.error = _error_reason(exc)
            logger.warning("verifier call failed: %s", exc)
            return None
        run.calls += 1
        self._claimed += len(bundle.views)
        self._spent += self.policy.cost_per_call
        run.estimated_cost_usd += self.policy.cost_per_call
        return _judgement_map(payload)

    def _ask_repair(
        self, view: ClaimView, window: str, previous: Verdict, run: VerificationRun
    ) -> FieldJudgement | None:
        if not self._can_verify(1):
            run.budget_exhausted = True
            return None
        try:
            payload = self.router.call(
                PURPOSE,
                VerdictBundle,
                self.repair_prompt.text,
                repair_message(view, window, previous),
                input_ref=f"repair:{view.key}:{previous}",
            )
        except LLMError as exc:
            run.error = _error_reason(exc)
            logger.warning("verifier repair failed: %s", exc)
            return None
        run.calls += 1
        run.repairs += 1
        self._claimed += 1
        self._spent += self.policy.cost_per_call
        run.estimated_cost_usd += self.policy.cost_per_call
        judgements = _judgement_map(payload)
        return judgements.get(view.key)

    def _decide(
        self,
        view: ClaimView,
        judgement: FieldJudgement | None,
        window: str,
        run: VerificationRun,
    ) -> VerificationDecision:
        if judgement is None:
            return self._decision(view, status="unverified", reason="no_verdict")
        verdict = judgement.verdict
        if verdict == "supported":
            return self._decision(
                view,
                status="supported",
                verdict=verdict,
                reason=judgement.reason,
                corrected_quote=judgement.corrected_quote,
            )
        if verdict == "absent":
            return self._decision(
                view,
                status="rejected",
                verdict=verdict,
                reason=judgement.reason or "verifier_absent",
            )
        # contradicted / wrong_field: exactly one single-field repair attempt.
        repair = self._ask_repair(view, window, verdict, run)
        if repair is not None and repair.verdict == "supported":
            return self._decision(
                view,
                status="supported",
                verdict=verdict,
                reason="supported_after_repair",
                repair_attempted=True,
                corrected_quote=repair.corrected_quote or judgement.corrected_quote,
            )
        if repair is None:
            return self._decision(
                view,
                status="unverified",
                verdict=verdict,
                reason="budget_exhausted_before_repair"
                if run.budget_exhausted
                else f"verifier_unavailable:{run.error or 'error'}",
                repair_attempted=True,
                corrected_quote=judgement.corrected_quote,
            )
        return self._decision(
            view,
            status="rejected",
            verdict=repair.verdict,
            reason=f"verifier_{verdict}_after_repair",
            repair_attempted=True,
            corrected_quote=repair.corrected_quote or judgement.corrected_quote,
        )

    def _decision(
        self,
        view: ClaimView,
        *,
        status: str,
        verdict: Verdict | None = None,
        reason: str | None = None,
        repair_attempted: bool = False,
        corrected_quote: str | None = None,
    ) -> VerificationDecision:
        return VerificationDecision(
            key=view.key,
            claim_id=view.claim_id,
            status=status,  # type: ignore[arg-type]
            verdict=verdict,
            reason=reason,
            repair_attempted=repair_attempted,
            corrected_quote=corrected_quote,
            prompt_version=self.prompt.ref,
            verified_by=self.model,
        )

    def _persist(self, run: VerificationRun, *, persist: bool) -> None:
        if not persist or self.claims is None:
            return
        writer = getattr(self.claims, "record_verification", None)
        if writer is None:
            return
        from fmd.db.repositories import ClaimVerification

        now = datetime.now(UTC)
        for decision in run.decisions:
            if decision.claim_id is None:
                continue
            writer(
                decision.claim_id,
                ClaimVerification(
                    status=decision.status,
                    verified_by=decision.verified_by,
                    verified_at=now,
                    notes={
                        "verdict": decision.verdict,
                        "reason": decision.reason,
                        "repair_attempted": decision.repair_attempted,
                        "corrected_quote": decision.corrected_quote,
                        "prompt_version": decision.prompt_version,
                        "policy": {
                            "window_chars": self.policy.window_chars,
                            "bundle_fields_per_call": self.policy.bundle_fields,
                        },
                    },
                ),
            )


# --------------------------------------------------------------------------- helpers


def _key_for(claim: Any, index: int) -> str:
    if isinstance(claim, dict):
        if claim.get("key"):
            return str(claim["key"])
        claim_id = claim.get("id")
    else:
        key = getattr(claim, "key", None)
        if key:
            return str(key)
        claim_id = getattr(claim, "id", None)
    return f"claim-{claim_id}" if claim_id is not None else f"claim-{index + 1}"


def _judgement_map(payload: Any) -> dict[str, FieldJudgement]:
    if isinstance(payload, VerdictBundle):
        entries = payload.judgements
    elif isinstance(payload, dict):
        entries = VerdictBundle.model_validate(payload).judgements
    else:  # pragma: no cover - router validates the schema
        return {}
    result: dict[str, FieldJudgement] = {}
    for entry in entries:
        result.setdefault(entry.key, entry)
    return result


def _error_reason(exc: Exception) -> str:
    reason = getattr(exc, "reason", None)
    if reason:
        return str(reason)
    name = type(exc).__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


__all__ = [
    "VERDICTS",
    "Bundle",
    "Verifier",
    "VerifierUnavailable",
    "bounded_window",
    "bundle_claims",
    "bundle_message",
    "repair_message",
]
