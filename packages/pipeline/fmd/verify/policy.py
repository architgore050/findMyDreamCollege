"""Verifier policy, read from ``config/models.yaml:profiles.verify``.

No model id, window size, bundle size or price is hardcoded here. Missing
optional knobs fall back to conservative values so an older config still runs,
but a missing ``verify`` profile disables verification rather than inventing a
provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from fmd.config.models import ModelsConfig

PROFILE = "verify"
DEFAULT_WINDOW_CHARS = 1200
DEFAULT_BUNDLE_FIELDS = 8
DEFAULT_MAX_CLAIMS = 500


@dataclass(frozen=True)
class VerifyPolicy:
    window_chars: int = DEFAULT_WINDOW_CHARS
    bundle_fields: int = DEFAULT_BUNDLE_FIELDS
    max_claims_per_run: int = DEFAULT_MAX_CLAIMS
    max_usd_per_run: float = 0.0
    input_tokens_per_call: int = 0
    output_tokens_per_call: int = 0
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    prompt_name: str = "verify"
    prompt_version: str | None = None

    @classmethod
    def load(cls, models: ModelsConfig, profile: str = PROFILE) -> VerifyPolicy | None:
        """Return the policy, or ``None`` when verification is not configured."""
        configured = models.profiles.get(profile)
        if configured is None:
            return None
        data: dict[str, Any] = configured.model_dump()
        cost = data.get("cost_model") or {}
        pricing = data.get("pricing") or {}
        return cls(
            window_chars=int(data.get("window_chars") or DEFAULT_WINDOW_CHARS),
            bundle_fields=max(1, int(data.get("bundle_fields_per_call") or DEFAULT_BUNDLE_FIELDS)),
            max_claims_per_run=max(0, int(data.get("max_claims_per_run") or DEFAULT_MAX_CLAIMS)),
            max_usd_per_run=max(0.0, float(data.get("max_usd_per_run") or 0.0)),
            input_tokens_per_call=int(
                cost.get("input_tokens_per_call") or 0
            ),
            output_tokens_per_call=int(
                cost.get("output_tokens_per_call") or 0
            ),
            input_per_mtok=float(pricing.get("input_per_mtok") or 0.0),
            output_per_mtok=float(pricing.get("output_per_mtok") or 0.0),
            prompt_name=str(data.get("prompt_name") or "verify"),
            prompt_version=data.get("prompt_version"),
        )

    @property
    def cost_per_call(self) -> float:
        """Configured estimate for one bundled verifier call (USD)."""
        return (
            self.input_tokens_per_call / 1_000_000 * self.input_per_mtok
            + self.output_tokens_per_call / 1_000_000 * self.output_per_mtok
        )

    def estimated_cost(self, claims: int) -> float:
        """Configured estimate for ``claims`` claims, bundle-aware."""
        if claims <= 0:
            return 0.0
        return round(self.calls_for(claims) * self.cost_per_call, 8)

    def calls_for(self, claims: int) -> int:
        return math.ceil(claims / self.bundle_fields) if claims > 0 else 0


def cost_per_1k_claims(policy: VerifyPolicy) -> dict[str, Any]:
    """The numbers the run report quotes, all derived from config."""
    calls = policy.calls_for(1000)
    return {
        "claims": 1000,
        "bundle_fields_per_call": policy.bundle_fields,
        "calls": calls,
        "input_tokens": calls * policy.input_tokens_per_call,
        "output_tokens": calls * policy.output_tokens_per_call,
        "usd": round(calls * policy.cost_per_call, 8),
        "input_per_mtok": policy.input_per_mtok,
        "output_per_mtok": policy.output_per_mtok,
        "note": (
            "configured estimate; repairs add at most one call per flagged claim and "
            "are charged against the same run cap"
        ),
    }
