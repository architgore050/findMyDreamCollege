"""Planner-version stamping for documents the planning pass decides.

``config/plan.yaml`` declares a ``version``, but ``documents`` has no planner
version column and this workstream must not migrate the schema. The stamp is
the only durable, per-document record of *which planner produced a decision*:

* a reason written by this module looks like ``<code>|plan_v<version>`` (an
  abstention code with its planner) or ``plan_v<version>`` (a decision with no
  code — ``linked`` / ``not_applicable``);
* ``defer_for_text`` is **never** stamped: :class:`fmd.orchestrate.plan.PlanStage`
  and the phase-B resolution path match that code exactly, and a deferred
  document is not a decision — it is a request for text.

Staleness is then a pure function of the row: a decided document
(``linked`` / ``needs_review`` / ``abstained`` / ``not_applicable``) whose
reason carries no stamp or a stamp below the current planner version is stale,
and ``fmd plan --force`` may re-decide it. Decisions the pass does not
recognise (free prose written by human review, e.g. ``resolved by human review
(link_entity)``) are never replannable: a human decision outranks any planner
version.
"""

from __future__ import annotations

from typing import Any

#: Version assumed when ``config/plan.yaml`` does not declare one.
PLANNER_VERSION_FALLBACK = "0"

#: Prefix of the stamp token written into ``documents.plan_reason``.
STAMP_PREFIX = "plan_v"

#: Separator between an abstention code and its planner stamp.
STAMP_SEPARATOR = "|"

#: Reasons the planner itself writes (``fmd.plan.planner``), which a planning
#: pass may re-decide. Anything else on a decided row — human-review prose,
#: future codes this pass does not know — is left alone.
PLANNER_REASON_CODES = frozenset(
    {
        "family_undetected",
        "ambiguous_family",
        "entity_undetected",
        "entity_unresolved",
        "entity_ambiguous",
        "not_applicable",
    }
)

#: Prefix of the one planner reason that carries a family suffix
#: (``unsupported_family:<name>``) instead of a bare code.
UNSUPPORTED_FAMILY_PREFIX = "unsupported_family:"


def planner_version(plan_config: Any | None) -> str:
    """The declared planner version (``config/plan.yaml → version``), or the fallback."""
    version = getattr(plan_config, "version", None)
    if not isinstance(version, str) or not version.strip():
        return PLANNER_VERSION_FALLBACK
    cleaned = version.strip().lower().lstrip("v")
    return cleaned or PLANNER_VERSION_FALLBACK


def stamp_token(version: str) -> str:
    return f"{STAMP_PREFIX}{version}"


def stamp_reason(reason: str | None, version: str) -> str:
    """Attach the planner stamp to a decision's reason code.

    ``None`` becomes the bare stamp (``plan_v1``); a code becomes
    ``<code>|plan_v1``. Deferred rows never reach this function.
    """
    token = stamp_token(version)
    code = (reason or "").strip()
    return f"{code}{STAMP_SEPARATOR}{token}" if code else token


def unstamped_reason(reason: str | None) -> str | None:
    """The decision code without its version stamp (``None`` stays ``None``)."""
    if reason is None:
        return None
    value = reason.strip()
    token = stamp_token("")  # "plan_v"
    separator = STAMP_SEPARATOR
    if separator in value:
        code, _, suffix = value.rpartition(separator)
        if suffix.startswith(token) and len(suffix) > len(token):
            return code or None
        return value
    if value.startswith(token) and len(value) > len(token):
        return None
    return value or None


def stamped_version(reason: str | None) -> str | None:
    """The planner version a reason was written at, or ``None`` when unstamped."""
    if reason is None:
        return None
    value = reason.strip()
    token = stamp_token("")
    if STAMP_SEPARATOR in value:
        _, _, suffix = value.rpartition(STAMP_SEPARATOR)
    else:
        suffix = value
    if suffix.startswith(token) and len(suffix) > len(token):
        return suffix[len(token) :]
    return None


def is_current_version(reason: str | None, version: str) -> bool:
    """True when the reason was written by this planner version."""
    return stamped_version(reason) == version


def is_replannable_reason(reason: str | None) -> bool:
    """True when ``--force`` may re-decide a row carrying this reason.

    Only planner-shaped reasons qualify: unstamped planner codes (written by
    crawl, i.e. an older planner), stamped codes from any pass version, or no
    reason at all. Human-review prose and unknown codes are never replanned.
    """
    if reason is None:
        return True
    value = reason.strip()
    if not value:
        return True
    if stamped_version(value) is not None:
        return True
    if value.startswith(UNSUPPORTED_FAMILY_PREFIX):
        return True
    return value in PLANNER_REASON_CODES


__all__ = [
    "PLANNER_REASON_CODES",
    "PLANNER_VERSION_FALLBACK",
    "STAMP_PREFIX",
    "STAMP_SEPARATOR",
    "UNSUPPORTED_FAMILY_PREFIX",
    "is_current_version",
    "is_replannable_reason",
    "planner_version",
    "stamp_reason",
    "stamp_token",
    "stamped_version",
    "unstamped_reason",
]
