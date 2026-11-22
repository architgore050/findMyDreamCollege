"""Verdict stream (docs/v1/eval/01-eval-designs.md §3): the audited research-agent eval.

For each profile fixture, a verdict agent

1. retrieves the product's shortlist for that profile (frozen fixture in the
   fast variant — version-pinned, no live call);
2. selects evidence-rich and thin/no-evidence programme rows (honest
   negatives);
3. does citable research (in the fast variant this is the committed research
   record replayed by the mock router — deterministic, offline, no live web);
4. builds a reference top-3 and emits a structured verdict:
   ``correct | partially_correct | incorrect | unknown``.

The harness computes every score (invariant 2): agreement rate of
``verdict == correct``, top-3 overlap between the product shortlist and the
agent's reference, UNKNOWN coverage, and an evidence audit — every pointer
must carry URL + quote + sha256, and the circularity guard requires at least
one source *outside* the product's own ingested evidence per profile. The
model never scores anything; it only emits the verdict object.

Anti-hallucination stance: the fast backend is a replay router
(``MockVerdictRouter``) with the same contract as ``fmd/llm``'s mock provider.
A live backend plugs into the same protocol without touching the scoring code;
the human-audit protocol lives in docs/v1/eval/04-verdict-audit-protocol.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

VERDICT_VALUES = ("correct", "partially_correct", "incorrect", "unknown")
SCHEMA_VERSION = 1


class VerdictError(ValueError):
    """A profile fixture or verdict object violates the verdict contract."""


class EvidencePointer(BaseModel):
    """One citable source the verdict relied on (audit trail entry)."""

    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    quote: str
    sha256: str
    fetched_at: str
    source_kind: Literal["official", "ranking", "institution", "press", "fixture"]
    external: bool = False  # True = outside the product's ingested evidence

    @model_validator(mode="after")
    def _pointer_rules(self) -> EvidencePointer:
        if not self.url.strip() or not self.quote.strip():
            raise ValueError(f"evidence pointer {self.url!r} needs a non-empty url and quote")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256.lower()):
            raise ValueError(f"evidence pointer {self.url!r}: sha256 must be a 64-hex digest")
        return self


class SelectedProgramme(BaseModel):
    """A programme the agent examined, with its evidence tier (honest negatives)."""

    model_config = ConfigDict(extra="forbid")

    institution_name: str
    programme: str
    evidence_tier: Literal["rich", "thin", "none"]


class VerdictObject(BaseModel):
    """The structured verdict the agent emits — the only model output allowed."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    verdict: Literal["correct", "partially_correct", "incorrect", "unknown"]
    rationale: str
    reference_top3: list[str] = Field(default_factory=list)
    selected_programmes: list[SelectedProgramme] = Field(default_factory=list)
    evidence: list[EvidencePointer] = Field(default_factory=list)
    unknown_aspects: list[str] = Field(default_factory=list)
    as_of: str

    @model_validator(mode="after")
    def _verdict_rules(self) -> VerdictObject:
        if not self.rationale.strip():
            raise ValueError(f"{self.profile_id}: verdict without a rationale is not auditable")
        if self.verdict == "unknown" and not self.unknown_aspects:
            raise ValueError(
                f"{self.profile_id}: verdict unknown must name its unknown_aspects "
                "(UNKNOWN is a value with content, invariant 5)"
            )
        if self.verdict != "unknown" and len(self.reference_top3) < 1:
            raise ValueError(f"{self.profile_id}: a scored verdict needs a reference_top3")
        return self


class ProductShortlistRow(BaseModel):
    """The product's shortlist as frozen for the eval (fast variant)."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    institution_name: str
    programme: str
    n_claims: int = 0
    n_sources: int = 0


class ProfileFixture(BaseModel):
    """One eval profile: input profile + frozen product shortlist + scripted research."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profile_id: str
    synthetic: bool = False
    notes: str | None = None
    profile: dict[str, Any]
    product_shortlist: list[ProductShortlistRow]
    ingested_evidence_urls: list[str] = Field(default_factory=list)
    scripted_verdict: dict[str, Any]
    as_of: str

    @model_validator(mode="after")
    def _fixture_rules(self) -> ProfileFixture:
        if not self.synthetic:
            raise ValueError(
                f"{self.profile_id}: profile fixtures must declare synthetic: true "
                "(eval profiles are test inputs, never user data)"
            )
        ranks = [row.rank for row in self.product_shortlist]
        if ranks != sorted(ranks) or (ranks and ranks[0] != 1):
            raise ValueError(f"{self.profile_id}: product_shortlist ranks must start at 1 in order")
        if not self.product_shortlist:
            raise ValueError(f"{self.profile_id}: fixture needs a product shortlist")
        return self


class VerdictBackend(Protocol):
    def respond(self, profile: ProfileFixture) -> VerdictObject: ...


class MockVerdictRouter:
    """The mock/replay router for the verdict stream (fast variant).

    Same semantics as the pipeline's replay provider: it deterministically
    replays the committed research record (``scripted_verdict``) and validates
    it against the verdict schema. No network, no model call. A live/slow
    backend implements the same ``VerdictBackend`` protocol; scoring code is
    unchanged when it arrives.
    """

    name = "mock_verdict_router"

    def respond(self, profile: ProfileFixture) -> VerdictObject:
        try:
            verdict = VerdictObject.model_validate(profile.scripted_verdict)
        except Exception as exc:
            raise VerdictError(
                f"{profile.profile_id}: scripted verdict is not schema-valid: {exc}"
            ) from exc
        if verdict.profile_id != profile.profile_id:
            raise VerdictError(
                f"{profile.profile_id}: scripted verdict names {verdict.profile_id!r}"
            )
        return verdict


# --------------------------------------------------------------------------- harness scoring


def product_top_n(profile: ProfileFixture, n: int = 3) -> list[str]:
    return [
        row.institution_name
        for row in profile.product_shortlist
        if row.rank <= n
    ]


def top3_overlap_with_reference(profile: ProfileFixture, verdict: VerdictObject) -> int:
    return len(set(product_top_n(profile, 3)) & set(verdict.reference_top3))


def audit_verdict(profile: ProfileFixture, verdict: VerdictObject) -> dict[str, Any]:
    """Evidence audit trail (harness-side; no network in the fast variant)."""

    failures: list[str] = []
    for index, pointer in enumerate(verdict.evidence, start=1):
        # Model validators enforce url/quote/sha256 shape; audit re-checks the
        # parts the harness depends on and adds the circularity guard.
        if not pointer.fetched_at.strip():
            failures.append(f"evidence #{index}: missing fetched_at")
    ingested = set(profile.ingested_evidence_urls)
    external = [pointer for pointer in verdict.evidence if pointer.external]
    if not external:
        failures.append(
            "circularity guard: verdict cites no source outside the product's "
            "ingested evidence (needs >= 1 external source)"
        )
    for pointer in external:
        if pointer.url in ingested:
            failures.append(
                f"circularity guard: pointer marked external but URL is in the "
                f"product's ingested evidence: {pointer.url}"
            )
    return {
        "failures": failures,
        "n_evidence": len(verdict.evidence),
        "n_external": len(external),
        "selected_programmes": [s.model_dump() for s in verdict.selected_programmes],
        "evidence": [pointer.model_dump() for pointer in verdict.evidence],
    }


def run_verdict_fast(profiles: list[ProfileFixture]) -> list[dict[str, Any]]:
    """Run the canary end-to-end on the mock router; returns harness-scored cases."""

    backend = MockVerdictRouter()
    cases: list[dict[str, Any]] = []
    for profile in profiles:
        verdict = backend.respond(profile)
        audit = audit_verdict(profile, verdict)
        cases.append(
            {
                "profile_id": profile.profile_id,
                "verdict": verdict.verdict,
                "rationale": verdict.rationale,
                "unknown_aspects": verdict.unknown_aspects,
                "as_of": verdict.as_of,
                "product_top3": product_top_n(profile, 3),
                "reference_top3": verdict.reference_top3,
                "top3_overlap": top3_overlap_with_reference(profile, verdict),
                "audit": audit,
            }
        )
    return cases


def verdict_aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Harness-computed scores for a set of verdict cases (pure, offline)."""

    total = len(cases)
    if not total:
        return {"n_profiles": 0, "agreement_rate": None, "unknown_coverage": None}
    return {
        "n_profiles": total,
        "agreement_rate": round(
            sum(1 for case in cases if case["verdict"] == "correct") / total, 6
        ),
        "partial_rate": round(
            sum(1 for case in cases if case["verdict"] == "partially_correct") / total, 6
        ),
        "incorrect_rate": round(
            sum(1 for case in cases if case["verdict"] == "incorrect") / total, 6
        ),
        "unknown_coverage": round(
            sum(1 for case in cases if case["verdict"] == "unknown") / total, 6
        ),
        "top3_overlap_mean": round(
            sum(case["top3_overlap"] for case in cases) / total, 6
        ),
        "audit_failures": sum(1 for case in cases if case["audit"]["failures"]),
    }


# --------------------------------------------------------------------------- fixtures


def load_profile_fixtures(directory: str | Path) -> list[ProfileFixture]:
    """Load every ``*.json`` profile fixture; files starting with '_' are docs."""

    directory = Path(directory)
    if not directory.is_dir():
        raise VerdictError(f"profile fixture directory not found: {directory}")
    profiles: list[ProfileFixture] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile = ProfileFixture.model_validate(payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise VerdictError(f"{path}: cannot read profile fixture: {exc}") from exc
        except Exception as exc:
            raise VerdictError(f"{path}: profile fixture invalid: {exc}") from exc
        if profile.profile_id in seen:
            raise VerdictError(f"{path}: duplicate profile_id {profile.profile_id}")
        seen.add(profile.profile_id)
        profiles.append(profile)
    if not profiles:
        raise VerdictError(f"{directory}: no profile fixtures found")
    return profiles


def render_verdict_summary(report: dict[str, Any]) -> str:
    aggregate = report.get("aggregate") or {}
    lines = [
        "fmd eval (verdict-fast): {n} profile(s), mock router, offline".format(
            n=aggregate.get("n_profiles", 0)
        ),
        "  agreement_rate (verdict==correct): {v}".format(v=aggregate.get("agreement_rate")),
        "  unknown_coverage: {v}  top3_overlap_mean: {t}".format(
            v=aggregate.get("unknown_coverage"), t=aggregate.get("top3_overlap_mean")
        ),
    ]
    for case in report.get("cases") or []:
        lines.append(
            "  {profile}: {verdict} top3_overlap {overlap} audit_failures {af}".format(
                profile=case.get("profile_id"),
                verdict=case.get("verdict"),
                overlap=case.get("top3_overlap"),
                af=len((case.get("audit") or {}).get("failures") or []),
            )
        )
    lines.append(
        "  note: fixtures are synthetic; evidence pointers are structural audit "
        "fixtures, not live citations (see docs/v1/eval/04-verdict-audit-protocol.md)"
    )
    return "\n".join(lines)


__all__ = [
    "EvidencePointer",
    "MockVerdictRouter",
    "ProfileFixture",
    "ProductShortlistRow",
    "SCHEMA_VERSION",
    "SelectedProgramme",
    "VerdictBackend",
    "VerdictError",
    "VerdictObject",
    "audit_verdict",
    "load_profile_fixtures",
    "product_top_n",
    "render_verdict_summary",
    "run_verdict_fast",
    "top3_overlap_with_reference",
    "verdict_aggregate",
]
