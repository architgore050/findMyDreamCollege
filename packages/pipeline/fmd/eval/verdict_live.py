"""Live verdict backend (docs/v1/eval/04-verdict-audit-protocol.md §4, FMD-CLOUD-024).

The slow, live-research variant of the verdict stream. :class:`LiveVerdictBackend`
implements the same :class:`~fmd.eval.verdict.VerdictBackend` protocol as
:class:`~fmd.eval.verdict.MockVerdictRouter` — the harness scoring code is
untouched (scoring is FROZEN by design). The backend runs a BOUNDED research
loop through the model router:

1. system prompt + profile + the product shortlist rows go to the model;
2. each model turn is ONE action from the tool schema: ``fetch_url`` (routed
   through the fetch layer — robots.txt, per-domain rate limits, truthful UA),
   ``cite_source`` (url + verbatim quote of a page fetched *this run*), or
   ``emit_verdict`` (the structured verdict object);
3. hard limits are enforced in code: at most ``max_turns`` model turns and at
   most ``max_fetches`` fetches per profile. When the loop stops before a
   schema-valid verdict, the backend returns an honest ``unknown`` verdict
   (UNKNOWN is a first-class output; the model is instructed to prefer it over
   guessing).

Anti-hallucination stance:

* the model NEVER supplies hashes or scores. ``sha256`` of every evidence
  pointer is computed by the harness over the exact bytes the fetch layer
  stored (:class:`StoredPage` ledger); ``run_verdict_live`` re-verifies each
  pointer against that ledger, so a fabricated citation or a tampered page is
  caught harness-side;
* the turn schema has ``extra="forbid"``: a model output that tries to emit a
  score (or any undeclared field) is rejected as a schema violation;
* ``as_of`` must equal the fixture's ``as_of`` (doc 04 §3 pins it per verdict);
* the circularity guard (>= 1 source outside the product's ingested evidence)
  stays in :func:`fmd.eval.verdict.audit_verdict`, harness-side, unchanged.

Offline guarantees: :class:`ScriptedModelRouter` and :class:`ScriptedPageFetcher`
replay committed sessions under
``tests/fixtures/eval_verdict/live/sessions/`` — router turns + fake fetch
responses, no network, no real model. Everything proven about this stream is
proven on that replay path; a live-model run is the operator's first run (LQ-9).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fmd.eval.verdict import (
    EvidencePointer,
    ProfileFixture,
    VerdictError,
    VerdictObject,
    audit_verdict,
    product_top_n,
    top3_overlap_with_reference,
    verdict_aggregate,
)
from fmd.fetch.types import FetchResult
from fmd.llm.providers import ProviderError

#: Human-audit acceptance gate (docs/v1/eval/04-verdict-audit-protocol.md §2.3;
#: threshold originates in docs/v1/eval/02-findings-viable.md). REPORTED, never
#: enforced in CI — no operator exists in CI.
AUDIT_GATE = 0.70

#: The audit sample is a random 10-20 % of verdict cases (doc 04 §2.1); the CLI
#: fraction is clamped into this band. Stratification adds every incorrect and
#: partially_correct verdict on top, per the same doc.
AUDIT_FRACTION_MIN = 0.10
AUDIT_FRACTION_MAX = 0.20

STRATIFIED_VERDICTS = ("incorrect", "partially_correct")

SCHEMA_VERSION = 1

VERDICT_LIVE_PURPOSE = "verdict_live"

SYSTEM_PROMPT = """You are the research agent of the findMyDreamCollege verdict eval.
For one synthetic profile you independently check the product's ranked shortlist
against public, citable evidence and emit a structured verdict.

Hard rules:
1. One action per turn, exactly one of:
   - {"action": "fetch_url", "url": "...", "note": "why this page matters"} —
     the page is fetched by the harness (robots.txt and rate limits are its
     business, never yours). Only pages fetched in THIS run can be cited.
   - {"action": "cite_source", "url": "...", "quote": "verbatim snippet",
      "title": "short page title", "source_kind": "official|ranking|institution|press"} —
     the quote must appear verbatim in the stored content of a fetched page;
     anything else is a fabrication and is rejected.
   - {"action": "emit_verdict", "verdict": {...}} — the final structured verdict.
2. NEVER emit scores, weights, hash values, or any field outside the schema.
   The harness computes every number. The verdict object must not contain a
   "score" field, sha256 values, or evidence pointers — cite sources with
   cite_source instead; the harness attaches the audited evidence graph.
3. NEVER invent facts. Every claim you rely on must be backed by a verbatim
   quote from a page you fetched and cited this run.
4. UNKNOWN is a first-class answer. If your research is thin, contradictory,
   or out of budget, emit verdict "unknown" and name the unknown_aspects.
   Prefer unknown over guessing — a guessed label is worse than an honest gap.
5. Always set verdict.profile_id and verdict.as_of to the values given in the
   profile context below (a stale as_of is an audit failure, doc 04 §3).
6. A scored verdict (correct | partially_correct | incorrect) must include a
   reference_top3 built ONLY from your research; "correct" means the product's
   top-3 matches your reference and you found no counter-evidence.
"""


# --------------------------------------------------------------------------- turn schema


class ResearchAction(BaseModel):
    """One model turn — the tool schema of the bounded research loop.

    ``extra="forbid"`` is the anti-score guard: a model output that carries a
    score, a hash, or any undeclared field cannot validate, and the turn is
    rejected as a schema violation.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["fetch_url", "cite_source", "emit_verdict"]
    url: str | None = None
    quote: str | None = None
    title: str | None = None
    note: str | None = None
    source_kind: Literal["official", "ranking", "institution", "press"] | None = None
    verdict: VerdictObject | None = None

    @model_validator(mode="after")
    def _action_rules(self) -> ResearchAction:
        if self.action == "fetch_url" and not (self.url or "").strip():
            raise ValueError("fetch_url needs a url")
        if self.action == "cite_source":
            if not (self.url or "").strip() or not (self.quote or "").strip():
                raise ValueError("cite_source needs url + verbatim quote")
            if not (self.title or "").strip():
                raise ValueError("cite_source needs a title")
            if self.source_kind is None:
                raise ValueError("cite_source needs a source_kind")
        if self.action == "emit_verdict" and self.verdict is None:
            raise ValueError("emit_verdict needs the verdict object")
        return self


# --------------------------------------------------------------------------- model client


class VerdictModelClient(Protocol):
    """The router surface the backend needs (``fmd.llm.router.ModelRouter``)."""

    def call(
        self,
        purpose: str,
        schema: type[BaseModel],
        system: str,
        user: str,
        input_ref: str,
    ) -> BaseModel: ...


# --------------------------------------------------------------------------- scripted replay


class ScriptedPage(BaseModel):
    """One fake fetch response committed in a live-session fixture."""

    model_config = ConfigDict(extra="forbid")

    url: str
    status: int = 200
    content_type: str = "text/html; charset=utf-8"
    body: str


class ScriptedTurn(BaseModel):
    """One scripted router turn: ``content`` is validated only at replay time
    (that is the point — a session turn that tries to emit a score must be
    rejected by the schema, not by the fixture loader)."""

    model_config = ConfigDict(extra="forbid")

    content: dict[str, Any]


class LiveSession(BaseModel):
    """A committed scripted live session: router turns + fake fetch responses."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    profile_id: str
    notes: str | None = None
    pages: list[ScriptedPage] = Field(default_factory=list)
    turns: list[ScriptedTurn] = Field(default_factory=list)


class ScriptedModelRouter:
    """Replay client over one committed live session (offline; no real model).

    Mirrors the real router's contract: it validates each scripted turn against
    the requested schema and raises ``ProviderError("json_schema_violation")``
    on mismatch — so a session turn that tries to emit a score is rejected
    exactly like a live provider response would be.
    """

    name = "scripted_verdict_router"

    def __init__(self, session: LiveSession) -> None:
        self.session = session
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        purpose: str,
        schema: type[BaseModel],
        system: str,
        user: str,
        input_ref: str,
    ) -> BaseModel:
        self.calls.append({"purpose": purpose, "input_ref": input_ref, "turn": self._cursor})
        if self._cursor >= len(self.session.turns):
            raise ProviderError(
                "fixture_missing",
                f"scripted session for {self.session.profile_id!r} exhausted after "
                f"{len(self.session.turns)} turn(s)",
            )
        turn = self.session.turns[self._cursor]
        self._cursor += 1
        try:
            return schema.model_validate(turn.content)
        except ValidationError as exc:
            raise ProviderError(
                "json_schema_violation",
                f"scripted turn {self._cursor} for {self.session.profile_id!r} is not a "
                f"valid {schema.__name__}: {exc}",
            ) from exc


class VerdictPageFetcher(Protocol):
    """The fetch surface the backend needs (``fmd.fetch.ladder.FetchLadder``)."""

    def fetch(self, url: str) -> FetchResult: ...


class ScriptedPageFetcher:
    """Fake fetch layer over the committed pages of the live sessions.

    An unknown URL is an honest 404 (terminal), never invented content; a
    non-200 fixture status is replayed verbatim.
    """

    fetcher_tier = "replay"

    def __init__(self, pages: list[ScriptedPage]) -> None:
        self.pages = {page.url: page for page in pages}

    def fetch(self, url: str) -> FetchResult:
        page = self.pages.get(url)
        if page is None:
            return FetchResult(
                url=url,
                final_url=url,
                status=404,
                fetcher_tier=self.fetcher_tier,
                error="not_in_fixture",
            )
        if page.status != 200:
            return FetchResult(
                url=url,
                final_url=url,
                status=page.status,
                fetcher_tier=self.fetcher_tier,
                error=f"fixture_status_{page.status}",
            )
        return FetchResult(
            url=url,
            final_url=url,
            status=page.status,
            headers={"content-type": page.content_type},
            body=page.body.encode("utf-8"),
            fetcher_tier=self.fetcher_tier,
        )


# --------------------------------------------------------------------------- the backend


class StoredPage:
    """What the fetch layer actually stored for one cited URL this run.

    The sha256 is computed over the exact body bytes returned by the fetch
    layer; ``run_verdict_live`` re-verifies every evidence pointer against it.
    """

    def __init__(self, url: str, final_url: str, status: int, body: bytes, fetched_at: str) -> None:
        self.url = url
        self.final_url = final_url
        self.status = status
        self.body = body
        self.fetched_at = fetched_at
        self.sha256 = hashlib.sha256(body).hexdigest()

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


@dataclass
class Citation:
    """A source the model cited via ``cite_source`` (quote already verified)."""

    url: str
    quote: str
    title: str
    source_kind: str
    note: str = ""


class BackendRunStats:
    """How the bounded loop actually spent its budget for one profile."""

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        self.turns_used = 0
        self.fetches_used = 0
        self.citations = 0
        self.schema_violations = 0
        self.rejected_verdicts = 0
        self.stop_reason = "turn_cap_reached"

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "turns_used": self.turns_used,
            "fetches_used": self.fetches_used,
            "citations": self.citations,
            "schema_violations": self.schema_violations,
            "rejected_verdicts": self.rejected_verdicts,
            "stop_reason": self.stop_reason,
        }


def _excerpt(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + " …[truncated]"


class LiveVerdictBackend:
    """The bounded live-research verdict backend (``VerdictBackend`` protocol).

    ``model_client`` is the provider-agnostic router (``fmd.llm.router``);
    ``page_fetcher`` is the fetch layer (``fmd.fetch.ladder.FetchLadder`` on the
    operator's host, ``ScriptedPageFetcher`` offline). Hard limits:
    ``max_turns`` model turns and ``max_fetches`` fetches per profile, enforced
    in code, never negotiable by the model.
    """

    name = "live_verdict_backend"

    def __init__(
        self,
        model_client: VerdictModelClient,
        page_fetcher: VerdictPageFetcher,
        *,
        purpose: str = VERDICT_LIVE_PURPOSE,
        max_turns: int = 8,
        max_fetches: int = 6,
        excerpt_chars: int = 480,
        clock: Any | None = None,
    ) -> None:
        if max_turns < 1 or max_fetches < 0:
            raise VerdictError(
                "verdict-live limits must be positive (max_turns>=1, max_fetches>=0)"
            )
        self.model_client = model_client
        self.page_fetcher = page_fetcher
        self.purpose = purpose
        self.max_turns = max_turns
        self.max_fetches = max_fetches
        self.excerpt_chars = excerpt_chars
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ledgers: dict[str, dict[str, StoredPage]] = {}
        self._last_stats: BackendRunStats | None = None

    # ----------------------------------------------------------------- protocol
    def respond(self, profile: ProfileFixture) -> VerdictObject:
        stats = BackendRunStats(profile.profile_id)
        ledger: dict[str, StoredPage] = {}
        citations: list[Citation] = []
        journal: list[str] = []
        self._ledgers[profile.profile_id] = ledger
        self._last_stats = stats

        for _turn in range(self.max_turns):
            stats.turns_used += 1
            try:
                action = self.model_client.call(
                    self.purpose,
                    ResearchAction,
                    SYSTEM_PROMPT,
                    self._build_user_prompt(profile, journal, citations),
                    f"verdict-live:{profile.profile_id}:turn-{stats.turns_used}",
                )
            except ProviderError as exc:
                reason = getattr(exc, "reason", "unknown")
                stats.schema_violations += 1
                stats.stop_reason = f"model_error:{reason}"
                if reason == "json_schema_violation" and stats.turns_used < self.max_turns:
                    journal.append(
                        f"REJECTED OUTPUT: {exc}. Emit exactly one schema-valid action; "
                        "scores, hashes and undeclared fields are forbidden."
                    )
                    continue
                break

            assert isinstance(action, ResearchAction)  # narrows BaseModel for mypy/ruff

            if action.action == "fetch_url":
                journal.append(self._handle_fetch(str(action.url), stats, ledger))
                continue

            if action.action == "cite_source":
                journal.append(
                    self._handle_cite(action, profile, ledger, citations, stats)
                )
                continue

            verdict, problems = self._finalize(action.verdict, profile, ledger, citations)
            if not problems:
                stats.stop_reason = "emitted_verdict"
                return verdict
            stats.rejected_verdicts += 1
            journal.append(
                "VERDICT REJECTED: " + "; ".join(problems) + " Fix the issues and re-emit."
            )
        else:
            # Loop exhausted without a schema-valid verdict: the honest outcome
            # is UNKNOWN, never a guessed label (invariant 5).
            stats.stop_reason = "turn_cap_reached"
        return self._degrade_to_unknown(profile, citations, ledger, stats.stop_reason)

    # ----------------------------------------------------------------- inspection
    def ledger_for(self, profile_id: str) -> dict[str, StoredPage]:
        """What the fetch layer stored for that profile (sha256 re-verification)."""
        return self._ledgers.get(profile_id, {})

    @property
    def last_run(self) -> dict[str, Any]:
        if self._last_stats is None:  # pragma: no cover - respond() always sets it
            raise VerdictError("respond() has not run yet")
        return self._last_stats.as_dict()

    # ----------------------------------------------------------------- internals
    def _now(self) -> str:
        return self._clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _handle_fetch(
        self, url: str, stats: BackendRunStats, ledger: dict[str, StoredPage]
    ) -> str:
        url = url.strip()
        if stats.fetches_used >= self.max_fetches:
            return (
                f"FETCH BUDGET EXHAUSTED ({self.max_fetches} fetches used). Continue with "
                "the pages you have, or emit_verdict / emit unknown."
            )
        if not url.startswith(("http://", "https://")):
            return f"FETCH REJECTED: {url!r} — only http(s) URLs; no auth or CAPTCHA bypass."
        stats.fetches_used += 1
        result = self.page_fetcher.fetch(url)
        if result.ok:
            page = StoredPage(
                url=url,
                final_url=result.final_url,
                status=result.status,
                body=result.body,
                fetched_at=self._now(),
            )
            ledger[url] = page
            return (
                f"FETCH OK: {url} status {result.status} sha256 {page.sha256}\n"
                f"EXCERPT: {_excerpt(page.text, self.excerpt_chars)}"
            )
        return (
            f"FETCH FAILED: {url} status {result.status} "
            f"({result.error or 'transport error'}). It cannot be cited."
        )

    def _handle_cite(
        self,
        action: ResearchAction,
        profile: ProfileFixture,
        ledger: dict[str, StoredPage],
        citations: list[Citation],
        stats: BackendRunStats,
    ) -> str:
        url = (action.url or "").strip()
        quote = (action.quote or "").strip()
        page = ledger.get(url)
        if page is None:
            return (
                f"CITE REJECTED: {url} was not fetched in this run. Cite only pages the "
                "harness fetched for you (anti-hallucination)."
            )
        if quote not in page.text:
            return (
                f"CITE REJECTED: quote not found verbatim in the stored content of {url}. "
                "Quote exactly; do not paraphrase or invent."
            )
        citations.append(
            Citation(
                url=url,
                quote=quote,
                title=(action.title or "").strip(),
                source_kind=str(action.source_kind),
                note=(action.note or "").strip(),
            )
        )
        stats.citations = len(citations)
        external = url not in profile.ingested_evidence_urls
        return (
            f"CITED: {url} ({action.source_kind}, "
            f"{'external' if external else 'product-ingested'})"
        )

    def _finalize(
        self,
        verdict: VerdictObject | None,
        profile: ProfileFixture,
        ledger: dict[str, StoredPage],
        citations: list[Citation],
    ) -> tuple[VerdictObject | None, list[str]]:
        """Validate the emitted verdict and attach the audited evidence graph."""
        if verdict is None:  # pragma: no cover - schema guarantees presence
            return None, ["emit_verdict without a verdict object"]
        problems: list[str] = []
        if verdict.profile_id != profile.profile_id:
            problems.append(
                f"profile_id is {verdict.profile_id!r}, expected {profile.profile_id!r}"
            )
        if verdict.as_of != profile.as_of:
            problems.append(
                f"as_of is {verdict.as_of!r}, expected the fixture as_of {profile.as_of!r}"
            )
        if verdict.evidence:
            problems.append(
                "verdict must not carry evidence pointers; cite sources with cite_source "
                "and let the harness attach the audited graph (sha256 is harness-computed)"
            )
        if problems:
            return None, problems
        pointers = [
            EvidencePointer(
                url=citation.url,
                title=citation.title,
                quote=citation.quote,
                sha256=ledger[citation.url].sha256,
                fetched_at=ledger[citation.url].fetched_at,
                source_kind=citation.source_kind,  # type: ignore[arg-type]
                external=citation.url not in profile.ingested_evidence_urls,
            )
            for citation in citations
            if citation.url in ledger
        ]
        attached = verdict.model_copy(update={"evidence": pointers})
        return attached, []

    def _degrade_to_unknown(
        self,
        profile: ProfileFixture,
        citations: list[Citation],
        ledger: dict[str, StoredPage],
        stop_reason: str,
    ) -> VerdictObject:
        """Honest degradation: the loop stopped without a supported verdict.

        UNKNOWN is a value, never a guessed label (invariant 5). Valid citations
        made so far are still attached — they are verbatim-verified evidence.
        """
        pointers = [
            EvidencePointer(
                url=citation.url,
                title=citation.title,
                quote=citation.quote,
                sha256=ledger[citation.url].sha256,
                fetched_at=ledger[citation.url].fetched_at,
                source_kind=citation.source_kind,  # type: ignore[arg-type]
                external=citation.url not in profile.ingested_evidence_urls,
            )
            for citation in citations
            if citation.url in ledger
        ]
        return VerdictObject(
            profile_id=profile.profile_id,
            verdict="unknown",
            rationale=(
                "Live research stopped before producing a citable verdict "
                f"(stop_reason: {stop_reason}). No score is invented for incomplete "
                "research; the verdict stays unknown per the anti-hallucination rules."
            ),
            reference_top3=[],
            evidence=pointers,
            unknown_aspects=[
                f"stop_reason: {stop_reason}",
                "research incomplete: the reference top-3 was never derived",
            ],
            as_of=profile.as_of,
        )

    def _build_user_prompt(
        self,
        profile: ProfileFixture,
        journal: list[str],
        citations: list[Citation],
    ) -> str:
        lines = [
            "PROFILE CONTEXT (synthetic eval fixture; runtime input, never user data)",
            f"profile_id: {profile.profile_id}",
            f"as_of (must be echoed into the verdict): {profile.as_of}",
            "profile: " + json.dumps(profile.profile, sort_keys=True),
            "product_shortlist (rank | institution | programme | n_claims | n_sources):",
        ]
        for row in profile.product_shortlist:
            lines.append(
                f"- {row.rank} | {row.institution_name} | {row.programme} "
                f"| {row.n_claims} | {row.n_sources}"
            )
        lines.append(
            "ingested_evidence_urls (the product's own sources — your verdict must "
            "also cite at least one source outside this list):"
        )
        for url in profile.ingested_evidence_urls:
            lines.append(f"- {url}")
        lines.append("")
        lines.append(
            f"RESEARCH JOURNAL (turn budget: {self.max_turns} model turns, "
            f"{self.max_fetches} fetches):"
        )
        if journal:
            lines.extend(f"- {entry}" for entry in journal)
        else:
            lines.append("- (empty: this is turn 1; start with fetch_url)")
        lines.append("")
        lines.append("CITATIONS SO FAR:")
        if citations:
            for citation in citations:
                lines.append(f"- {citation.url} | {citation.title} | {citation.source_kind}")
        else:
            lines.append("- (none yet)")
        lines.append("")
        lines.append(
            "NEXT ACTION: respond with exactly one schema-valid action object "
            "(fetch_url | cite_source | emit_verdict). If the evidence is thin, "
            "emit_verdict with verdict \"unknown\"."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- harness scoring


def reverify_evidence(
    backend: LiveVerdictBackend, profile: ProfileFixture, verdict: VerdictObject
) -> list[str]:
    """Harness-side sha256 re-verification (doc 04 §2.2 task 2, mechanised).

    Every pointer must hash to what the fetch layer actually stored this run,
    and its quote must appear verbatim in that stored content. A tampered page
    or a fabricated pointer fails here — the model cannot talk its way past it.
    """
    failures: list[str] = []
    ledger = backend.ledger_for(profile.profile_id)
    for index, pointer in enumerate(verdict.evidence, start=1):
        page = ledger.get(pointer.url)
        if page is None:
            failures.append(
                f"sha256 re-verification: evidence #{index} ({pointer.url}) was never "
                "stored by the fetch layer this run (fabricated pointer)"
            )
        elif hashlib.sha256(page.body).hexdigest() != pointer.sha256:
            failures.append(
                f"sha256 re-verification: evidence #{index} ({pointer.url}) recorded "
                f"{pointer.sha256[:12]}… but hashing the stored bytes does not match "
                "(tampered content)"
            )
        elif pointer.quote not in page.text:
            failures.append(
                f"quote check: evidence #{index} quote not found in the stored content "
                f"of {pointer.url}"
            )
    return failures


def _fetch_ledger(backend: LiveVerdictBackend, profile: ProfileFixture) -> list[dict[str, Any]]:
    """The fetch layer's own record for this profile (operator re-derivation)."""
    ledger = backend.ledger_for(profile.profile_id)
    return [
        {
            "url": page.url,
            "final_url": page.final_url,
            "status": page.status,
            "sha256": page.sha256,
            "fetched_at": page.fetched_at,
            "n_bytes": len(page.body),
        }
        for page in ledger.values()
    ]


def run_verdict_live(
    profiles: list[ProfileFixture], backend: LiveVerdictBackend
) -> list[dict[str, Any]]:
    """Run the profiles through the live backend; harness-scored cases out.

    Scoring is identical to the fast stream (verdict.py) plus the sha256
    re-verification block and the backend's budget accounting.
    """
    cases: list[dict[str, Any]] = []
    for profile in profiles:
        verdict = backend.respond(profile)
        audit = audit_verdict(profile, verdict)
        reverify = reverify_evidence(backend, profile, verdict)
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
                "reverify": {
                    "failures": reverify,
                    "n_evidence": len(verdict.evidence),
                },
                "fetch_ledger": _fetch_ledger(backend, profile),
                "backend_run": backend.last_run,
            }
        )
    return cases


def verdict_live_aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Fast-stream aggregate + budget/re-verification counters (pure, offline)."""
    aggregate = verdict_aggregate(cases)
    if cases:
        aggregate["n_turns_mean"] = round(
            sum(case["backend_run"]["turns_used"] for case in cases) / len(cases), 6
        )
        aggregate["n_fetches_mean"] = round(
            sum(case["backend_run"]["fetches_used"] for case in cases) / len(cases), 6
        )
        aggregate["schema_violations"] = sum(
            case["backend_run"]["schema_violations"] for case in cases
        )
        aggregate["reverify_failures"] = sum(1 for case in cases if case["reverify"]["failures"])
    return aggregate


# --------------------------------------------------------------------------- audit tooling


class OperatorAgreement(BaseModel):
    """The auditor's per-verdict record (doc 04 §2.2 task 3).

    ``citation_ok=False`` is a fabricated-citation finding and fails the whole
    run (doc 04 §2.3).
    """

    model_config = ConfigDict(extra="forbid")

    agrees: bool
    citation_ok: bool = True
    reason: str = ""


class AuditWorksheetEntry(BaseModel):
    """One sampled verdict with its full evidence graph, ready to audit."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    verdict: str
    rationale: str
    reference_top3: list[str] = Field(default_factory=list)
    selected_programmes: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidencePointer] = Field(default_factory=list)
    audit_failures: list[str] = Field(default_factory=list)
    reverify_failures: list[str] = Field(default_factory=list)
    fetch_ledger: list[dict[str, Any]] = Field(default_factory=list)
    unknown_aspects: list[str] = Field(default_factory=list)
    as_of: str
    operator_agreement: OperatorAgreement | None = None


class AuditWorksheet(BaseModel):
    """The auditable worksheet written by ``--audit-sample``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    stream: str
    generated_at: str
    seed: int
    fraction_requested: float
    fraction_used: float
    gate: float
    sampling_rule: str
    n_cases: int
    entries: list[AuditWorksheetEntry]


def clamp_audit_fraction(fraction: float) -> float:
    """The sample is a random 10-20 % of verdict cases (doc 04 §2.1)."""
    return min(AUDIT_FRACTION_MAX, max(AUDIT_FRACTION_MIN, fraction))


def build_audit_worksheet(
    cases: list[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
    stream: str = "verdict-live",
    generated_at: str | None = None,
) -> AuditWorksheet:
    """Select the audit sample with a seeded RNG and write the worksheet.

    Deterministic: the same cases + fraction + seed produce the same worksheet.
    Stratification (doc 04 §2.1): every incorrect and partially_correct verdict
    is always sampled; the seeded RNG fills the rest up to the clamped 10-20 %
    target, so ``n_sampled`` may exceed the random band when the run itself has
    many failures.
    """
    fraction_used = clamp_audit_fraction(fraction)
    rng = random.Random(seed)
    required = [case for case in cases if case["verdict"] in STRATIFIED_VERDICTS]
    others = [case for case in cases if case["verdict"] not in STRATIFIED_VERDICTS]
    rng.shuffle(others)
    target = max(1, round(len(cases) * fraction_used)) if cases else 0
    sampled = required + others[: max(0, target - len(required))]
    sampled.sort(key=lambda case: case["profile_id"])
    entries = [
        AuditWorksheetEntry(
            profile_id=case["profile_id"],
            verdict=case["verdict"],
            rationale=case["rationale"],
            reference_top3=case["reference_top3"],
            selected_programmes=case["audit"]["selected_programmes"],
            evidence=case["audit"]["evidence"],
            audit_failures=case["audit"]["failures"],
            reverify_failures=case["reverify"]["failures"],
            fetch_ledger=case.get("fetch_ledger", []),
            unknown_aspects=case["unknown_aspects"],
            as_of=case["as_of"],
        )
        for case in sampled
    ]
    return AuditWorksheet(
        stream=stream,
        generated_at=generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        seed=seed,
        fraction_requested=fraction,
        fraction_used=fraction_used,
        gate=AUDIT_GATE,
        sampling_rule=(
            "all incorrect/partially_correct verdicts + seeded random fill up to the "
            "clamped 10-20% target (docs/v1/eval/04-verdict-audit-protocol.md §2.1)"
        ),
        n_cases=len(cases),
        entries=entries,
    )


def score_audit_worksheet(worksheet: AuditWorksheet) -> dict[str, Any]:
    """Compute the human-agreement rate and report PASS/FAIL/INSUFFICIENT.

    Harness-computed, never enforced in CI (no operator exists there). The
    gate is the >=70 % threshold from doc 04 §2.3. A fabricated-citation
    finding (``citation_ok: false``) fails the whole run per the same doc.
    """
    completed = [entry for entry in worksheet.entries if entry.operator_agreement is not None]
    disagreements = [
        {
            "profile_id": entry.profile_id,
            "agent_verdict": entry.verdict,
            "reason": entry.operator_agreement.reason if entry.operator_agreement else "",
        }
        for entry in completed
        if entry.operator_agreement is not None and not entry.operator_agreement.agrees
    ]
    fabricated = [
        entry.profile_id
        for entry in completed
        if entry.operator_agreement is not None and not entry.operator_agreement.citation_ok
    ]
    if not worksheet.entries or len(completed) < 2:
        result = "INSUFFICIENT"
        reason = (
            f"only {len(completed)} of {len(worksheet.entries)} sampled verdict(s) carry an "
            "operator_agreement record; the gate needs at least 2 completed audits"
        )
        agreement_rate = None
    else:
        agreement_rate = round(
            sum(
                1
                for entry in completed
                if entry.operator_agreement is not None and entry.operator_agreement.agrees
            )
            / len(completed),
            6,
        )
        if fabricated:
            result = "FAIL"
            reason = (
                "fabricated citation found by the auditor (citation_ok=false) — the verdict "
                "set is quarantined and re-run after the fix (doc 04 §2.3)"
            )
        elif agreement_rate >= worksheet.gate:
            result = "PASS"
            reason = f"agreement {agreement_rate} >= gate {worksheet.gate}"
        else:
            result = "FAIL"
            reason = (
                f"agreement {agreement_rate} < gate {worksheet.gate}; the stream stays "
                "advisory and must not feed a product claim (doc 04 §2.3)"
            )
    return {
        "result": result,
        "reason": reason,
        "gate": worksheet.gate,
        "n_sampled": len(worksheet.entries),
        "n_completed": len(completed),
        "agreement_rate": agreement_rate,
        "disagreements": disagreements,
        "fabricated_citations": fabricated,
    }


# --------------------------------------------------------------------------- fixtures


def load_live_sessions(directory: str | Path) -> dict[str, LiveSession]:
    """Load every ``*.session.json`` live-session fixture, keyed by profile_id."""
    directory = Path(directory)
    if not directory.is_dir():
        raise VerdictError(f"live session directory not found: {directory}")
    sessions: dict[str, LiveSession] = {}
    for path in sorted(directory.glob("*.session.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            session = LiveSession.model_validate(payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise VerdictError(f"{path}: cannot read live session: {exc}") from exc
        except Exception as exc:
            raise VerdictError(f"{path}: live session invalid: {exc}") from exc
        if session.profile_id in sessions:
            raise VerdictError(f"{path}: duplicate session for {session.profile_id}")
        sessions[session.profile_id] = session
    if not sessions:
        raise VerdictError(f"{directory}: no *.session.json live sessions found")
    return sessions


# --------------------------------------------------------------------------- summary


def render_verdict_live_summary(report: dict[str, Any]) -> str:
    aggregate = report.get("aggregate") or {}
    backend_info = report.get("backend") or {}
    mode = backend_info.get("mode", "unknown")
    lines = [
        "fmd eval (verdict-live): {n} profile(s), backend mode {mode}, provider {provider}".format(
            n=aggregate.get("n_profiles", 0),
            mode=mode,
            provider=backend_info.get("provider"),
        ),
        "  limits: max_turns {t} max_fetches {f} (hard, enforced in code)".format(
            t=backend_info.get("max_turns"), f=backend_info.get("max_fetches")
        ),
        "  agreement_rate (verdict==correct): {v}".format(v=aggregate.get("agreement_rate")),
        "  unknown_coverage: {v}  top3_overlap_mean: {t}".format(
            v=aggregate.get("unknown_coverage"), t=aggregate.get("top3_overlap_mean")
        ),
        "  turns_mean: {t}  fetches_mean: {f}  schema_violations: {s}".format(
            t=aggregate.get("n_turns_mean"),
            f=aggregate.get("n_fetches_mean"),
            s=aggregate.get("schema_violations"),
        ),
    ]
    for case in report.get("cases") or []:
        lines.append(
            "  {profile}: {verdict} turns {turns} fetches {fetches} stop {stop} "
            "reverify_failures {rf} audit_failures {af}".format(
                profile=case.get("profile_id"),
                verdict=case.get("verdict"),
                turns=(case.get("backend_run") or {}).get("turns_used"),
                fetches=(case.get("backend_run") or {}).get("fetches_used"),
                stop=(case.get("backend_run") or {}).get("stop_reason"),
                rf=len((case.get("reverify") or {}).get("failures") or []),
                af=len((case.get("audit") or {}).get("failures") or []),
            )
        )
    if mode == "replay":
        lines.append(
            "  note: replay path (scripted sessions, offline) — the live stream has NEVER "
            "run against a real model yet; the first real run is the operator's (LQ-9)"
        )
    return "\n".join(lines)


__all__ = [
    "AUDIT_FRACTION_MAX",
    "AUDIT_FRACTION_MIN",
    "AUDIT_GATE",
    "AuditWorksheet",
    "AuditWorksheetEntry",
    "BackendRunStats",
    "Citation",
    "LiveSession",
    "LiveVerdictBackend",
    "OperatorAgreement",
    "ResearchAction",
    "SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "ScriptedModelRouter",
    "ScriptedPage",
    "ScriptedPageFetcher",
    "ScriptedTurn",
    "StoredPage",
    "VERDICT_LIVE_PURPOSE",
    "VerdictModelClient",
    "VerdictPageFetcher",
    "build_audit_worksheet",
    "clamp_audit_fraction",
    "load_live_sessions",
    "reverify_evidence",
    "render_verdict_live_summary",
    "run_verdict_live",
    "score_audit_worksheet",
    "verdict_live_aggregate",
]
