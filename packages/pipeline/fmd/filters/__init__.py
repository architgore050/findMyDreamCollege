"""Hard-filter registry (workstream F owns implementations, brief §12).

Filters are binary cutoffs, never weighted penalties. ``config/weights.yaml``
lists the filters that must exist; registry loading fails if a configured filter
has no implementation name. WS-A only declares the names and signatures so the
contract cannot silently drift.

Each implementation returns ``pass | fail | unknown`` plus the deciding claim.
"""

from __future__ import annotations

from typing import Any, Literal

FilterResult = Literal["pass", "fail", "unknown"]


def credential_level(programme: Any, profile: Any) -> FilterResult:
    raise NotImplementedError("credential_level filter is workstream F (see brief §12.1)")


def recognition_status(programme: Any, profile: Any) -> FilterResult:
    raise NotImplementedError("recognition_status filter is workstream F (see brief §12.2)")


def eligibility(programme: Any, profile: Any) -> FilterResult:
    raise NotImplementedError("eligibility filter is workstream F (see brief §12.3)")


def language(programme: Any, profile: Any) -> FilterResult:
    raise NotImplementedError("language filter is workstream F (see brief §12.4)")


def exam_registration_open(programme: Any, profile: Any) -> FilterResult:
    raise NotImplementedError("exam_registration_open is workstream F (see brief §12.5)")


def exam_travel(programme: Any, profile: Any) -> FilterResult:
    raise NotImplementedError("exam_travel filter is workstream F (see brief §12.6)")


def intake_year_available(programme: Any, profile: Any) -> FilterResult:
    raise NotImplementedError("intake_year_available is workstream F (see brief §12.7)")


def budget_ceiling(programme: Any, profile: Any) -> FilterResult:
    raise NotImplementedError("budget_ceiling filter is workstream F (see brief §12.8)")


# filter id in config/weights.yaml -> callable name in this module
IMPLEMENTATIONS: dict[str, str] = {
    "credential_level": "credential_level",
    "recognition_status": "recognition_status",
    "eligibility": "eligibility",
    "language": "language",
    "exam_registration_open": "exam_registration_open",
    "exam_travel": "exam_travel",
    "intake_year_available": "intake_year_available",
    "budget_ceiling": "budget_ceiling",
}
