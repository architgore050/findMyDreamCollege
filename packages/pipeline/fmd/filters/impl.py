"""Hard-filter implementations (brief §12).

Each filter is a pure function ``(programme, profile, *, context) -> FilterDecision``.
``unknown`` is a first-class outcome: a missing claim is never a pass, and a
stale exam window renders "unverified", never "open".

``programme`` may be a ``ProgrammeRecord``, a mapping, or any object exposing
``credential_level``, ``recognition_status`` plus a ``claims`` sequence. Claims
carry the deciding evidence and its ``observed_at``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from fmd.score.evidence import (
    CREDENTIAL_LEVEL,
    ELIGIBILITY_MAX_AGE,
    ELIGIBILITY_MIN_AGGREGATE_PCT,
    ELIGIBILITY_MIN_PCB_PCT,
    ELIGIBILITY_REQUIRES_SUBJECTS,
    EXAM_REGISTRATION_CLOSES,
    EXAM_REQUIRED,
    EXAM_REQUIRES_TRAVEL,
    INTAKE_CYCLES,
    LANGUAGE_MIN_SCORE,
    LANGUAGE_TEST_REQUIRED,
    MEDIUM_OF_INSTRUCTION,
    RECOGNITION_STATUS,
    TCOA_LO,
    ClaimIndex,
    ClaimView,
    numeric,
)

FilterResult = Literal["pass", "fail", "unknown"]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _profile(profile: Any, path: str, default: Any = None) -> Any:
    node = profile
    for part in path.split("."):
        if node is None:
            return default
        node = _get(node, part, None)
    return default if node is None else node


@dataclass(frozen=True)
class FilterContext:
    """Registry-derived constants plus the evaluation clock."""

    stale_exam_window_ttl_days: int = 3
    budget_uses: str = "tcoa_lo"
    currency: str | None = None
    exams: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    now: datetime | None = None
    unknown_blocks_ranking: bool = False

    @classmethod
    def from_registry(cls, registry: Any, *, now: datetime | None = None) -> FilterContext:
        policy = registry.weights.filter_policy
        exams = {
            exam.id: {
                "requires_travel": bool(exam.requires_travel),
                "requires_exam": bool(exam.requires_exam),
                "remote_options": bool(exam.remote_options),
            }
            for exam in registry.exams.exams
        }
        return cls(
            stale_exam_window_ttl_days=int(policy.stale_exam_window_ttl_days),
            budget_uses=str(policy.budget_uses),
            currency=str(registry.cost_rules.currency.base),
            exams=exams,
            now=now,
            unknown_blocks_ranking=bool(policy.unknown_blocks_ranking),
        )

    def clock(self) -> datetime:
        return self.now if self.now is not None else datetime.now(UTC)


def default_context(now: datetime | None = None) -> FilterContext:
    """Context from the validated registry (cached config load)."""

    from fmd.config import get_registry

    return FilterContext.from_registry(get_registry(), now=now)


@dataclass(frozen=True)
class FilterDecision:
    filter_id: str
    result: FilterResult
    claim_id: int | str | None = None
    reason: str = ""
    observed_at: datetime | None = None
    stale: bool = False
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "filter_id": self.filter_id,
            "result": self.result,
            "claim_id": self.claim_id,
            "reason": self.reason,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "stale": self.stale,
            "detail": dict(self.detail),
        }


def _claims(programme: Any) -> ClaimIndex:
    index = _get(programme, "claim_index", None)
    if isinstance(index, ClaimIndex):
        return index
    return ClaimIndex.build(_get(programme, "claims", ()) or ())


def _decide(
    filter_id: str,
    result: FilterResult,
    reason: str,
    claim: ClaimView | None = None,
    *,
    stale: bool = False,
    **detail: Any,
) -> FilterDecision:
    return FilterDecision(
        filter_id=filter_id,
        result=result,
        claim_id=claim.claim_id if claim is not None else None,
        reason=reason,
        observed_at=claim.observed_at if claim is not None else None,
        stale=stale,
        detail=detail,
    )


def _norm(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _age_days(now: datetime, observed: datetime) -> float:
    """Age in days, tolerating one naive and one aware timestamp."""

    if (now.tzinfo is None) != (observed.tzinfo is None):
        if now.tzinfo is None:
            observed = observed.astimezone(UTC).replace(tzinfo=None)
        else:
            now = now.astimezone(UTC).replace(tzinfo=None)
    return (now - observed).total_seconds() / 86_400.0


# --------------------------------------------------------------------------- 1


def credential_level(
    programme: Any, profile: Any, *, context: FilterContext | None = None
) -> FilterDecision:
    claim = None
    level = _get(programme, "credential_level", None)
    if level is None:
        claim = _claims(programme).latest(CREDENTIAL_LEVEL)
        level = claim.value if claim is not None else None
    if level is None:
        return _decide("credential_level", "unknown", "credential level not established", claim)
    allowed = _profile(profile, "credential_levels", None)
    if not allowed:
        return _decide(
            "credential_level",
            "unknown",
            "profile does not state accepted credential levels",
            claim,
            programme_level=level,
        )
    allowed_normalised = {_norm(item) for item in allowed}
    if _norm(level) in allowed_normalised:
        return _decide("credential_level", "pass", f"credential level {level} is accepted", claim)
    return _decide(
        "credential_level",
        "fail",
        f"credential level {level} is not in the profile's accepted levels",
        claim,
        accepted=sorted(allowed_normalised),
    )


# --------------------------------------------------------------------------- 2


def recognition_status(
    programme: Any, profile: Any, *, context: FilterContext | None = None
) -> FilterDecision:
    claim = None
    status = _get(programme, "recognition_status", None)
    if status is None or _norm(status) == "unknown":
        claim = _claims(programme).latest(RECOGNITION_STATUS)
        status = claim.value if claim is not None else status
    if status is None or _norm(status) == "unknown":
        return _decide(
            "recognition_status",
            "unknown",
            "recognition chain not resolvable from current evidence",
            claim,
        )
    chain = claim.qualifier.get("chain_resolved") if claim is not None else None
    if chain is False:
        return _decide(
            "recognition_status",
            "unknown",
            "NCAHP/state-council/university recognition chain is incomplete",
            claim,
        )
    if _norm(status) == "recognised":
        return _decide("recognition_status", "pass", "institute is recognised", claim)
    if _norm(status) == "unrecognised":
        return _decide("recognition_status", "fail", "institute is not recognised", claim)
    return _decide(
        "recognition_status",
        "unknown",
        f"recognition status {status!r} is not a full recognition",
        claim,
    )


# --------------------------------------------------------------------------- 3


def eligibility(
    programme: Any, profile: Any, *, context: FilterContext | None = None
) -> FilterDecision:
    index = _claims(programme)
    checks: list[tuple[str, FilterResult, ClaimView | None, str]] = []

    min_aggregate = index.latest(ELIGIBILITY_MIN_AGGREGATE_PCT)
    aggregate = numeric(_profile(profile, "marks.class12_aggregate_pct"))
    if min_aggregate is not None:
        required = min_aggregate.number
        if required is None:
            checks.append(("aggregate", "unknown", min_aggregate, "threshold unparseable"))
        elif aggregate is None:
            checks.append(("aggregate", "unknown", min_aggregate, "profile marks not supplied"))
        else:
            verdict: FilterResult = "pass" if aggregate >= required else "fail"
            checks.append(
                (
                    "aggregate",
                    verdict,
                    min_aggregate,
                    f"aggregate {aggregate:g} vs minimum {required:g}",
                )
            )

    min_pcb = index.latest(ELIGIBILITY_MIN_PCB_PCT)
    pcb = numeric(_profile(profile, "marks.pcb_pct"))
    if min_pcb is not None:
        required = min_pcb.number
        if required is None:
            checks.append(("pcb", "unknown", min_pcb, "threshold unparseable"))
        elif pcb is None:
            checks.append(("pcb", "unknown", min_pcb, "profile PCB marks not supplied"))
        else:
            verdict = "pass" if pcb >= required else "fail"
            checks.append(("pcb", verdict, min_pcb, f"PCB {pcb:g} vs minimum {required:g}"))

    max_age = index.latest(ELIGIBILITY_MAX_AGE)
    age = numeric(_profile(profile, "age"))
    if max_age is not None:
        limit = max_age.number
        if limit is None:
            checks.append(("age", "unknown", max_age, "age limit unparseable"))
        elif age is None:
            checks.append(("age", "unknown", max_age, "profile age not supplied"))
        else:
            verdict = "pass" if age <= limit else "fail"
            checks.append(("age", verdict, max_age, f"age {age:g} vs maximum {limit:g}"))

    subjects = index.latest(ELIGIBILITY_REQUIRES_SUBJECTS)
    if subjects is not None:
        required_subjects = {_norm(item) for item in _as_list(subjects.value)}
        if "pcb" in required_subjects:
            if pcb is None:
                checks.append(
                    ("subjects", "unknown", subjects, "PCB subject requirement, marks absent")
                )
            else:
                checks.append(("subjects", "pass", subjects, "10+2 PCB requirement met"))

    if not checks:
        return _decide(
            "eligibility",
            "unknown",
            "no eligibility thresholds extracted and no profile comparison possible",
        )
    if any(verdict == "fail" for _, verdict, _, _ in checks):
        failed = next(item for item in checks if item[1] == "fail")
        return _decide("eligibility", "fail", failed[3], failed[2], checks=_checks(checks))
    if any(verdict == "unknown" for _, verdict, _, _ in checks):
        unknown = next(item for item in checks if item[1] == "unknown")
        return _decide("eligibility", "unknown", unknown[3], unknown[2], checks=_checks(checks))
    deciding = checks[0]
    return _decide("eligibility", "pass", deciding[3], deciding[2], checks=_checks(checks))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _checks(checks: Sequence[tuple[str, str, ClaimView | None, str]]) -> list[dict[str, Any]]:
    return [
        {
            "check": name,
            "result": verdict,
            "claim_id": claim.claim_id if claim is not None else None,
            "reason": reason,
        }
        for name, verdict, claim, reason in checks
    ]


# --------------------------------------------------------------------------- 4


def language(
    programme: Any, profile: Any, *, context: FilterContext | None = None
) -> FilterDecision:
    index = _claims(programme)
    required = index.latest(LANGUAGE_TEST_REQUIRED)
    medium_claim = index.latest(MEDIUM_OF_INSTRUCTION)
    medium = _get(programme, "medium", None)
    if medium is None and medium_claim is not None:
        medium = medium_claim.value

    profile_mediums = {_norm(item) for item in _as_list(_profile(profile, "medium_of_instruction"))}
    if medium is not None and _norm(medium) in profile_mediums:
        return _decide(
            "language",
            "pass",
            f"medium-of-instruction exemption ({medium})",
            medium_claim,
        )

    if required is None:
        return _decide(
            "language",
            "unknown",
            "no language requirement extracted; cannot infer that none applies",
            medium_claim,
        )

    required_norm = _norm(required.value)
    if required_norm in {"none", "false", "no", "not_required"} or required.value is False:
        return _decide("language", "pass", "programme states no language test", required)

    minimum = index.latest(LANGUAGE_MIN_SCORE)
    min_score = minimum.number if minimum is not None else None
    test = _profile(profile, "english_test", None) or {}
    declared_type = _norm(_get(test, "type", "") or required.qualifier.get("test", ""))
    declared_score = numeric(_get(test, "score"))
    required_test = _norm(required.qualifier.get("test", "")) or required_norm

    if required_test not in {"", "any"} and declared_type and declared_type != required_test:
        return _decide(
            "language",
            "unknown",
            f"programme wants {required_test}, profile records {declared_type}",
            required,
        )
    if min_score is None:
        return _decide(
            "language",
            "unknown",
            "test required but no minimum score extracted",
            required,
        )
    if declared_score is None:
        return _decide(
            "language",
            "unknown",
            "profile has no language-test score",
            required,
            minimum=min_score,
        )
    if declared_score >= min_score:
        return _decide(
            "language",
            "pass",
            f"score {declared_score:g} meets minimum {min_score:g}",
            required,
        )
    return _decide(
        "language",
        "fail",
        f"score {declared_score:g} below minimum {min_score:g}",
        required,
        minimum=min_score,
    )


# --------------------------------------------------------------------------- 5


def _required_exams(programme: Any) -> tuple[ClaimView, ...]:
    index = _claims(programme)
    claims = index.all(EXAM_REQUIRED)
    if claims:
        return claims
    return ()


def exam_registration_open(
    programme: Any, profile: Any, *, context: FilterContext | None = None
) -> FilterDecision:
    ctx = context or default_context()
    index = _claims(programme)
    requirements = _required_exams(programme)
    if not requirements:
        return _decide(
            "exam_registration_open",
            "unknown",
            "no entrance-exam requirement extracted; registration state unverified",
        )

    now = ctx.clock()
    outcomes: list[tuple[str, FilterResult, ClaimView | None, str, bool]] = []
    for requirement in requirements:
        exam_id = str(requirement.qualifier.get("exam_id") or requirement.value)
        closes = index.latest(EXAM_REGISTRATION_CLOSES, exam_id=exam_id)
        if closes is None:
            outcomes.append(
                (
                    exam_id,
                    "unknown",
                    requirement,
                    f"no registration window extracted for {exam_id}",
                    False,
                )
            )
            continue
        if closes.observed_at is None:
            outcomes.append(
                (
                    exam_id,
                    "unknown",
                    closes,
                    f"{exam_id} window has no observed_at",
                    True,
                )
            )
            continue
        age = _age_days(now, closes.observed_at)
        if age > ctx.stale_exam_window_ttl_days:
            outcomes.append(
                (
                    exam_id,
                    "unknown",
                    closes,
                    (
                        f"{exam_id} window observed {age:.1f}d ago exceeds TTL "
                        f"{ctx.stale_exam_window_ttl_days}d"
                    ),
                    True,
                )
            )
            continue
        close_at = _as_datetime(closes.value)
        if close_at is None:
            outcomes.append(
                (exam_id, "unknown", closes, f"{exam_id} closing date unparseable", False)
            )
            continue
        now_aligned, close_at = _align(now, close_at)
        if now_aligned < close_at:
            outcomes.append(
                (
                    exam_id,
                    "pass",
                    closes,
                    f"{exam_id} registration closes {close_at.date().isoformat()}",
                    False,
                )
            )
        else:
            outcomes.append(
                (
                    exam_id,
                    "fail",
                    closes,
                    f"{exam_id} registration closed {close_at.date().isoformat()}",
                    False,
                )
            )

    return _combine("exam_registration_open", outcomes)


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().replace("/", "-")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _align(now: datetime, when: datetime) -> tuple[datetime, datetime]:
    """Bring a naive/aware pair onto one timeline without changing the instant."""

    if now.tzinfo is None and when.tzinfo is not None:
        return now, when.astimezone(UTC).replace(tzinfo=None)
    if now.tzinfo is not None and when.tzinfo is None:
        return now, when.replace(tzinfo=UTC)
    return now, when


def _combine(
    filter_id: str, outcomes: Sequence[tuple[str, FilterResult, ClaimView | None, str, bool]]
) -> FilterDecision:
    if any(item[1] == "fail" for item in outcomes):
        chosen = next(item for item in outcomes if item[1] == "fail")
        result: FilterResult = "fail"
    elif any(item[1] == "unknown" for item in outcomes):
        chosen = next(item for item in outcomes if item[1] == "unknown")
        result = "unknown"
    else:
        chosen = outcomes[0]
        result = "pass"
    return _decide(
        filter_id,
        result,
        chosen[3],
        chosen[2],
        stale=chosen[4],
        exams=[
            {
                "exam_id": exam_id,
                "result": verdict,
                "claim_id": claim.claim_id if claim is not None else None,
                "reason": reason,
                "stale": stale,
            }
            for exam_id, verdict, claim, reason, stale in outcomes
        ],
    )


# --------------------------------------------------------------------------- 6


def exam_travel(
    programme: Any, profile: Any, *, context: FilterContext | None = None
) -> FilterDecision:
    ctx = context or default_context()
    index = _claims(programme)
    requirements = _required_exams(programme)
    preference = _profile(profile, "travel_for_exams", None)

    if not requirements:
        return _decide(
            "exam_travel",
            "unknown",
            "no entrance-exam requirement extracted; travel requirement unverified",
        )
    if preference is None:
        return _decide("exam_travel", "unknown", "profile does not state travel tolerance")

    forbids_travel = _norm(preference) in {"no", "false", "none", "not_allowed", "avoids"}
    outcomes: list[tuple[str, FilterResult, ClaimView | None, str, bool]] = []
    for requirement in requirements:
        exam_id = str(requirement.qualifier.get("exam_id") or requirement.value)
        catalogue = ctx.exams.get(exam_id) or {}
        travel_claim = index.latest(EXAM_REQUIRES_TRAVEL, exam_id=exam_id)
        if catalogue:
            requires_travel = bool(catalogue.get("requires_travel", False))
            remote = bool(catalogue.get("remote_options", False))
            basis = "config/exams.yaml"
        elif travel_claim is not None:
            requires_travel = bool(travel_claim.value)
            remote = bool(travel_claim.qualifier.get("remote_options", False))
            basis = "claim"
        else:
            outcomes.append(
                (
                    exam_id,
                    "unknown",
                    requirement,
                    f"{exam_id} travel requirement unknown (not in registry, no claim)",
                    False,
                )
            )
            continue
        if not requires_travel or remote:
            outcomes.append(
                (
                    exam_id,
                    "pass",
                    travel_claim or requirement,
                    f"{exam_id} does not require physical travel ({basis})",
                    False,
                )
            )
        elif forbids_travel:
            outcomes.append(
                (
                    exam_id,
                    "fail",
                    travel_claim or requirement,
                    f"{exam_id} requires travel and the profile forbids it ({basis})",
                    False,
                )
            )
        else:
            outcomes.append(
                (
                    exam_id,
                    "pass",
                    travel_claim or requirement,
                    f"{exam_id} requires travel but the profile allows it ({basis})",
                    False,
                )
            )
    return _combine("exam_travel", outcomes)


# --------------------------------------------------------------------------- 7


def intake_year_available(
    programme: Any, profile: Any, *, context: FilterContext | None = None
) -> FilterDecision:
    target = _profile(profile, "target_intake_year", None)
    if target is None:
        return _decide(
            "intake_year_available", "unknown", "profile does not state a target intake year"
        )
    claim = _claims(programme).latest(INTAKE_CYCLES)
    if claim is not None:
        cycles = _as_list(claim.value)
    else:
        cycles = _as_list(_get(programme, "intake_cycles", None))
    if not cycles:
        return _decide(
            "intake_year_available",
            "unknown",
            "no intake cycles extracted for this programme",
            claim,
        )
    try:
        years = {int(str(item)[:4]) for item in cycles if str(item)[:4].isdigit()}
    except (TypeError, ValueError):
        years = set()
    if not years:
        return _decide("intake_year_available", "unknown", "intake cycles unparseable", claim)
    try:
        wanted = int(target)
    except (TypeError, ValueError):
        return _decide("intake_year_available", "unknown", "target intake year unparseable", claim)
    detail = {"cycles": sorted(years), "target": wanted}
    if wanted in years:
        return _decide(
            "intake_year_available",
            "pass",
            f"intake {wanted} is offered",
            claim,
            **detail,
        )
    return _decide(
        "intake_year_available",
        "fail",
        f"intake {wanted} is not in the extracted cycles {sorted(years)}",
        claim,
        **detail,
    )


# --------------------------------------------------------------------------- 8


def budget_ceiling(
    programme: Any, profile: Any, *, context: FilterContext | None = None
) -> FilterDecision:
    ctx = context or default_context()
    ceiling = _profile(profile, "budget.total_ceiling", None)
    annual_ceiling = _profile(profile, "budget.annual_ceiling", None)
    currency = _profile(profile, "budget.currency", None)
    if ctx.currency is None:
        return _decide(
            "budget_ceiling",
            "unknown",
            "cost currency is not configured; refusing to assume one",
        )
    if currency is None:
        return _decide(
            "budget_ceiling",
            "unknown",
            "profile does not state a budget currency",
        )
    if ceiling is None:
        if annual_ceiling is not None:
            return _decide(
                "budget_ceiling",
                "unknown",
                "only an annual ceiling is stated; TCOA is a whole-programme figure",
            )
        return _decide("budget_ceiling", "unknown", "profile does not state a budget ceiling")
    if _norm(currency) != _norm(ctx.currency):
        return _decide(
            "budget_ceiling",
            "unknown",
            f"budget currency {currency} differs from cost currency {ctx.currency}; "
            "no FX comparison inside a hard filter",
        )

    index = _claims(programme)
    claim = index.latest(TCOA_LO)
    tcoa_lo = claim.number if claim is not None else None
    if tcoa_lo is None:
        tcoa_lo = numeric(_get(programme, "tcoa_lo", None))
    if tcoa_lo is None:
        return _decide(
            "budget_ceiling",
            "unknown",
            "TCOA lower bound unavailable; cannot compare with the ceiling",
            claim,
        )
    try:
        ceiling_value = float(ceiling)
    except (TypeError, ValueError):
        return _decide("budget_ceiling", "unknown", "budget ceiling unparseable", claim)

    detail = {"tcoa_lo": tcoa_lo, "ceiling": ceiling_value, "uses": "tcoa_lo"}
    if tcoa_lo > ceiling_value:
        return _decide(
            "budget_ceiling",
            "fail",
            f"TCOA lower bound {tcoa_lo:g} exceeds ceiling {ceiling_value:g}",
            claim,
            **detail,
        )
    return _decide(
        "budget_ceiling",
        "pass",
        f"TCOA lower bound {tcoa_lo:g} is within ceiling {ceiling_value:g}",
        claim,
        **detail,
    )
