"""Second-pass claim verification (brief §9, §15, §19).

Extraction asks a model what a page says. Verification asks a *structurally
different* model call whether an existing claim is supported by a bounded source
window. It sees the window and the existing quote, never the claimed value, and
returns a judgement only.

Layout:

* :mod:`fmd.verify.policy`   — config-driven window, bundling and budget limits
* :mod:`fmd.verify.verifier` — bundling, forced verdicts, selective repair
* :mod:`fmd.verify.runner`   — the pending-claim queue (`fmd verify`)
* :mod:`fmd.verify.metrics`  — verified_rate / fabrication_caught / correct_rejected
* :mod:`fmd.verify.casefile` — committed fabrication fixtures used by tests/eval
"""

from fmd.verify.metrics import VerificationMetrics, verification_metrics
from fmd.verify.policy import VerifyPolicy, cost_per_1k_claims
from fmd.verify.types import ClaimView, VerificationDecision, VerificationRun
from fmd.verify.verifier import VERDICTS, Verifier, bounded_window

__all__ = [
    "VERDICTS",
    "ClaimView",
    "VerificationDecision",
    "VerificationMetrics",
    "VerificationRun",
    "VerifyPolicy",
    "Verifier",
    "bounded_window",
    "cost_per_1k_claims",
    "verification_metrics",
]
