"""Planning: turn a fetched document into an extraction target (brief §6, §9).

The crawl layer knows *what* it fetched; extraction needs to know *which family*
the text belongs to and *which entity* it describes. That decision lives here,
not in an HTTP header and not in adapter code:

* the source registry may declare ``extraction_family`` — registry wins;
* otherwise :mod:`fmd.plan.rules` scores generic, data-declared signals
  (URL tokens, file extension, anchor text, first-page text);
* entity candidates come from :mod:`fmd.er` run on names found in the title,
  headers and first page, so a link is auditable rather than guessed.

A plan that cannot reach the auto-accept band stays ``needs_review`` with a
NULL entity. Nothing here writes to the database; :func:`persist_plan` is the
thin persistence seam.
"""

from __future__ import annotations

from fmd.plan.mentions import collect_mentions
from fmd.plan.planner import (
    DocumentPlan,
    DocumentPlanner,
    EntityCandidatePlan,
    abstention_reason,
    persist_plan,
)
from fmd.plan.rules import (
    DEFAULT_RULES,
    FamilyRule,
    PlanRules,
    SignalWeights,
    detect_family,
)

__all__ = [
    "DEFAULT_RULES",
    "DocumentPlan",
    "DocumentPlanner",
    "EntityCandidatePlan",
    "FamilyRule",
    "PlanRules",
    "SignalWeights",
    "abstention_reason",
    "collect_mentions",
    "detect_family",
    "persist_plan",
]
