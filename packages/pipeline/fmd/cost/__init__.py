"""Cost + financing model — workstream F (brief §13).

TCOA, net price, family out-of-pocket, capitalised moratorium interest, EMI,
DSR25 bands, affordability gap, two non-additive living scenarios, capped
student-work offset and an FX sensitivity band. All arithmetic is pure; the
snapshot carries methods, intervals and pre-rendered display values.
"""

from __future__ import annotations

from fmd.cost.model import (
    AffordabilityResult,
    CostInputs,
    CostItem,
    CostModelError,
    CostRules,
    CostSnapshot,
    DSRResult,
    LivingScenarioCost,
    LoanTerms,
    MissingFXRate,
    MoneyBand,
    StudentWork,
    affordability_ratio,
    capitalised_debt,
    compute,
    dsr_band,
    emi,
    living_annual,
    loan_terms_from_config,
    student_work_from_config,
)

__all__ = [
    "AffordabilityResult",
    "CostInputs",
    "CostItem",
    "CostModelError",
    "CostRules",
    "CostSnapshot",
    "DSRResult",
    "LivingScenarioCost",
    "LoanTerms",
    "MissingFXRate",
    "MoneyBand",
    "StudentWork",
    "affordability_ratio",
    "capitalised_debt",
    "compute",
    "dsr_band",
    "emi",
    "living_annual",
    "loan_terms_from_config",
    "student_work_from_config",
]

# Backwards-compatible name used by the WS-A stub.
total_cost = compute
