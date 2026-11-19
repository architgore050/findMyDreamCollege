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
    DEFER_CODE,
    PERSISTED_PLAN_STATES,
    DocumentPlan,
    DocumentPlanner,
    EntityCandidatePlan,
    PhaseThresholds,
    abstention_reason,
    persist_plan,
)
from fmd.plan.roles import (
    ROLE_EVIDENCE,
    ROLE_INVENTORY,
    ROLE_NAVIGATION,
    ROLES,
    RoleRules,
    classify_content,
    family_hints_from_structure,
    role_for,
)
from fmd.plan.rules import (
    DEFAULT_RULES,
    FamilyRule,
    PlanRules,
    SignalWeights,
    detect_family,
)
from fmd.plan.versioning import (
    is_current_version,
    is_replannable_reason,
    planner_version,
    stamp_reason,
    stamped_version,
    unstamped_reason,
)

__all__ = [
    "DEFAULT_RULES",
    "DEFER_CODE",
    "DocumentPlan",
    "DocumentPlanner",
    "EntityCandidatePlan",
    "FamilyRule",
    "PERSISTED_PLAN_STATES",
    "PhaseThresholds",
    "PlanRules",
    "ROLE_EVIDENCE",
    "ROLE_INVENTORY",
    "ROLE_NAVIGATION",
    "ROLES",
    "RoleRules",
    "SignalWeights",
    "abstention_reason",
    "classify_content",
    "collect_mentions",
    "detect_family",
    "family_hints_from_structure",
    "is_current_version",
    "is_replannable_reason",
    "persist_plan",
    "planner_version",
    "role_for",
    "stamp_reason",
    "stamped_version",
    "unstamped_reason",
]
