"""Institution-site evidence discovery (spec §4.9).

Given a college domain, find the PDFs that actually carry outcomes evidence:
NAAC SSR/AQAR, NIRF-DCS mirrors, annual/IQAC reports, placement reports,
prospectuses, seat matrices and fee structures. This is discovery, not
extraction: the adapter yields ``DocumentRef`` values and never reads a
placement table.

The crawl is bounded by registry parameters (``max_depth``,
``max_pages_per_domain``, ``min_yield_percent``) and stays on one registrable
domain. Login walls, ``javascript:`` anchors, archives and images are hard
skips. robots is enforced by the fetch ladder; a robots refusal is an
abstention, never an empty success.

Evidence targeting (2026-11-18 W1): the scarce budget is the per-domain
document cap, and the measured 50-domain run spent it on accreditation PDFs
while fee/seat-matrix documents scored zero. Three registry-driven mechanisms
fix the ordering and the budget:

* ``evidence_types`` / ``evidence_type_weights`` — a data-driven boost table
  (filename + anchor level) for seat/intake/sanctioned, fee/tuition/hostel/mess,
  prospectus/admission brochure, placement/annual report/AQAR/SSR and NIRF
  links, overridable per deployment. An administrative fee URL ("Fee for
  recognition of Institution as Research Centre …") is vetoed by the type's
  ``negative`` patterns: no boost, no slot in the reserved budget, no family.
* ``pdf_first_reserve_percent`` — a share of ``max_documents_per_domain`` is
  reserved for candidate evidence documents, so news/tender/gallery documents
  can only fill the unreserved remainder.
* family hints — a seat/fee document recognised at anchor/filename level is
  carried as ``metadata["family"]``; the planner treats it as a *reference*
  hint and a registry-declared ``extraction_family`` always outranks it.
"""

from __future__ import annotations

import heapq
import logging
import math
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, resolve
from fmd.adapters.links import (
    LinkDecision,
    classify_link,
    rank_decisions,
    require_param,
    same_site,
)
from fmd.discover.base import DocumentRef

logger = logging.getLogger(__name__)

#: Default ordering weights. They exist only to order a crawl queue, and every
#: one of them is overridable from ``params.link_weights`` (invariant 7).
DEFAULT_LINK_WEIGHTS: dict[str, int] = {
    "document": 10,
    "filename_match": 5,
    "anchor_match": 5,
    "navigate": 5,
    "navigate_anchor": 3,
    "navigate_filename": 2,
    "deprioritised": -50,
}

#: Administrative-fee context phrases shared by the seat and fee types. A URL
#: like ``Guidelines-…-Fee-for-recognition-of-Institution-as-Research-Centre…``
#: matches the fee vocabulary but the charge is institution-paid, not a student
#: fee structure: the match is vetoed for that link.
ADMIN_FEE_NEGATIVE_PATTERNS: tuple[str, ...] = (
    r"(?i)fee for recognition",
    r"(?i)recognition of institution",
    r"(?i)research cent(re|er)",
    r"(?i)affiliation fee",
    r"(?i)inspection fee",
    r"(?i)re-?valuation fee",
)

#: Default evidence-type boost table (``params.evidence_types`` overrides or
#: extends it per type). ``filename`` patterns match the link URL, ``anchor``
#: patterns the anchor text, ``hint`` the strong phrases that justify carrying a
#: family hint to the planner, ``negative`` the administrative contexts that
#: veto the type on that link. Boosts order the queue; only a ``family`` value
#: classifies anything.
DEFAULT_EVIDENCE_TYPES: dict[str, dict[str, Any]] = {
    "seat_matrix": {
        "family": "seat_matrix",
        "filename": [
            r"(?i)(seat[-_ ]?matrix|seat[-_ ]?availability)",
            r"(?i)(sanctioned[-_ ]?(intake|strength|seats)|intake[-_ ]?capacity)",
            r"(?i)(\bintake\b|\bseats?\b)",
        ],
        "anchor": [
            r"(?i)(seat matrix|sanctioned intake|intake capacity)",
            r"(?i)(number of seats|seat availability|total seats|\bintake\b)",
        ],
        "hint": [
            r"(?i)(seat[-_ ]?matrix|sanctioned[-_ ]?(intake|strength))",
            r"(?i)(intake[-_ ]?capacity|number of seats|seat availability)",
        ],
        "negative": list(ADMIN_FEE_NEGATIVE_PATTERNS),
    },
    "fee_structure": {
        "family": "fee_structure",
        "filename": [
            r"(?i)(fee[-_ ]?structure|fee[-_ ]?schedule|fee[-_ ]?details)",
            r"(?i)(tuition[-_ ]?fee|hostel[-_ ]?fee|mess[-_ ]?fee)",
            r"(?i)\bfees?\b",
        ],
        "anchor": [
            r"(?i)(fee structure|fee details|fee schedule)",
            r"(?i)(tuition fee|hostel fee|mess fee|fee notification)",
        ],
        "hint": [
            r"(?i)(fee[-_ ]?structure|fee[-_ ]?schedule)",
            r"(?i)(tuition[-_ ]?fee|hostel[-_ ]?fee|mess[-_ ]?fee)",
        ],
        "negative": list(ADMIN_FEE_NEGATIVE_PATTERNS),
    },
    "prospectus": {
        "filename": [r"(?i)(prospectus|admission[-_ ]?brochure)"],
        "anchor": [r"(?i)(prospectus|admission brochure)"],
    },
    "placement": {
        "filename": [r"(?i)(placement|annual[-_ ]?report|aqar|ssr|self[-_ ]?study|iqac)"],
        "anchor": [r"(?i)(placement|annual report|aqar|ssr|self study|iqac)"],
    },
    "nirf": {
        "filename": [r"(?i)(nirf|dcs)"],
        "anchor": [r"(?i)(nirf|dcs)"],
    },
}

#: Default boost per evidence type (``params.evidence_type_weights`` overrides).
DEFAULT_EVIDENCE_TYPE_WEIGHTS: dict[str, int] = {
    "seat_matrix": 40,
    "fee_structure": 40,
    "prospectus": 25,
    "placement": 15,
    "nirf": 12,
}


class InstitutionSiteAdapter(BaseAdapter):
    name = "institution_site"

    def __init__(self, source: Any, *, ladder: Any | None = None) -> None:
        super().__init__(source, ladder=ladder)
        self._evidence_cache: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------ params
    def _weights(self) -> dict[str, int]:
        raw = self.param("link_weights") or {}
        weights = dict(DEFAULT_LINK_WEIGHTS)
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    weights[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        return weights

    def _evidence_table(self) -> dict[str, dict[str, Any]]:
        """Compiled evidence-type table: defaults, extended by ``params.evidence_types``.

        A config entry for an existing type overrides that type's pattern lists
        wholesale; an unknown type name is added. ``family`` values the planner
        does not know are dropped later at hint time, never here.
        """
        if self._evidence_cache is not None:
            return self._evidence_cache
        table: dict[str, dict[str, Any]] = {}
        raw = self.param("evidence_types") or {}
        specs: dict[str, Any] = dict(DEFAULT_EVIDENCE_TYPES)
        if isinstance(raw, dict):
            for name, spec in raw.items():
                merged = dict(specs.get(str(name)) or {})
                if isinstance(spec, dict):
                    merged.update(spec)
                specs[str(name)] = merged
        for name, spec in specs.items():
            if not isinstance(spec, dict):
                continue
            table[name] = {
                "family": str(spec["family"]) if spec.get("family") else None,
                "filename": tuple(re.compile(p) for p in (spec.get("filename") or ()) if p),
                "anchor": tuple(re.compile(p) for p in (spec.get("anchor") or ()) if p),
                "hint": tuple(re.compile(p) for p in (spec.get("hint") or ()) if p),
                "negative": tuple(re.compile(p) for p in (spec.get("negative") or ()) if p),
            }
        self._evidence_cache = table
        return table

    def _evidence_type_weights(self) -> dict[str, int]:
        raw = self.param("evidence_type_weights") or {}
        weights = dict(DEFAULT_EVIDENCE_TYPE_WEIGHTS)
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    weights[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        return weights

    def _evidence_types_for(self, url: str, text: str) -> tuple[str, ...]:
        """Evidence types matching this link, in table order, vetoes applied.

        A type whose ``negative`` patterns match the URL or the anchor is not a
        candidate: no boost, no reserved slot, no family hint. This is what
        keeps an administrative order ("Fee for recognition of Institution as
        Research Centre …") out of the fee_structure lane.
        """
        lowered = (url or "").lower()
        label = text or ""
        # URL slugs separate words with hyphens/underscores; the negative
        # phrases are written in prose ("fee for recognition"), so the veto
        # haystack is separator-normalised like fmd.plan.rules.normalise does.
        veto_haystack = re.sub(r"[-_]+", " ", f"{lowered} {label}")
        matched: list[str] = []
        for name, spec in self._evidence_table().items():
            hit = self._matches(spec["filename"], lowered) or self._matches(spec["anchor"], label)
            if not hit:
                continue
            if self._matches(spec["negative"], veto_haystack):
                continue
            matched.append(name)
        return tuple(matched)

    def _declared_family(self) -> str | None:
        """A registry-declared family is never overridden by a discovery hint."""
        value = self.param("extraction_family")
        if value is None:
            value = getattr(self.source, "extraction_family", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _family_hint(self, url: str, text: str, types: Iterable[str]) -> str | None:
        """Family discovered at anchor/filename level, or ``None``.

        Only evidence types that declare a ``family`` participate, and only via
        their ``hint`` patterns (the strong phrases — "fee structure", "seat
        matrix" — not the bare boost vocabulary). The planner still treats the
        result as a reference hint: a registry-declared family outranks it.
        """
        if self._declared_family():
            return None
        lowered = (url or "").lower()
        label = text or ""
        for name in types:
            spec = self._evidence_table().get(name)
            if not spec or not spec.get("family"):
                continue
            if self._matches(spec["hint"], f"{lowered} {label}"):
                return str(spec["family"])
        return None

    def _reserve_size(self, max_documents: int) -> int:
        """Reserved document slots for candidate evidence, from config percent.

        ``pdf_first_reserve_percent`` (default 0 = off) is a share of
        ``max_documents_per_domain``; the reserve is the rounded-up share. The
        reserve only gates non-evidence documents, so ``0`` documents found
        never blocks a domain from filling its cap with what it has.
        """
        try:
            percent = float(self.param("pdf_first_reserve_percent", 0) or 0)
        except (TypeError, ValueError):
            percent = 0.0
        percent = min(max(percent, 0.0), 100.0)
        if not max_documents or percent <= 0:
            return 0
        return min(max_documents, math.ceil(max_documents * percent / 100.0))

    def _deprioritise_patterns(self) -> list[re.Pattern[str]]:
        values = self.param("deprioritise_patterns") or ()
        if isinstance(values, str):
            values = (values,)
        return [re.compile(value) for value in values if value]

    @staticmethod
    def _matches(patterns: Iterable[re.Pattern[str]], value: str) -> bool:
        return any(pattern.search(value) for pattern in patterns)

    def _score(self, decision: LinkDecision) -> int:
        """Config-weighted score for one classified link.

        The classification gate (document vs navigation) is shared with the
        other adapters; this method only decides *order*. A news/events/gallery
        page is pushed to the back of the queue and can never outrank a
        placement/AQAR/SSR/seat-matrix/fee-structure link.
        """
        weights = self._weights()
        filename_patterns = [
            re.compile(pattern) for pattern in (self.param("document_filename_patterns") or ())
        ]
        anchor_patterns = [re.compile(p) for p in (self.param("anchor_patterns") or ())]
        filename_match = self._matches(filename_patterns, decision.url.lower())
        anchor_match = self._matches(anchor_patterns, decision.text)
        if decision.kind == "document":
            score = weights["document"]
            score += weights["filename_match"] if filename_match else 0
            score += weights["anchor_match"] if anchor_match else 0
        else:
            score = weights["navigate"]
            score += weights["navigate_anchor"] if anchor_match else 0
            score += weights["navigate_filename"] if filename_match else 0
        evidence_weights = self._evidence_type_weights()
        for name in self._evidence_types_for(decision.url, decision.text):
            score += evidence_weights.get(name, 0)
        if self._matches(self._deprioritise_patterns(), f"{decision.url} {decision.text}"):
            score += weights["deprioritised"]
        return score

    def _decisions(self, text: str, *, base_url: str) -> list[LinkDecision]:
        decisions: list[LinkDecision] = []
        seen: set[str] = set()
        for href, label in anchors(text):
            try:
                # urllib raises on malformed authorities (for example an
                # unmatched IPv6 bracket), including inside classify_link.
                parts = urlsplit(href)
                if (parts.scheme and not parts.netloc) or (parts.netloc and not parts.hostname):
                    logger.debug("Skipping institution link without a host: %r", href)
                    continue
                # Accessing port validates values that urlsplit leaves lazy.
                _ = parts.port

                decision = self._classify(href, label)
                if decision is None:
                    continue
                url = resolve(base_url, decision.url)
                parts = urlsplit(url)
                if parts.scheme not in ("http", "https") or not parts.hostname:
                    logger.debug("Skipping institution link without an HTTP host: %r", href)
                    continue
                _ = parts.port
                on_site = same_site(base_url, url)
            except ValueError as exc:
                # A single bad href must not abort discovery for the source.
                logger.debug("Skipping malformed institution link %r: %s", href, exc)
                continue
            if url in seen or not on_site:
                continue
            seen.add(url)
            score = self._score(decision)
            if decision.kind == "navigate" and self._matches(
                self._deprioritise_patterns(), f"{url} {decision.text}"
            ):
                # Do not spend a navigation hop on news/events/gallery/admission
                # brochure pages. A document is never dropped here: a potential
                # evidence PDF is only ranked last.
                continue
            decisions.append(
                LinkDecision(url, decision.text, decision.kind, score, decision.reason)
            )
        return rank_decisions(decisions)

    def _seeds(self, domains: Iterable[str] | None = None) -> list[str]:
        candidates: list[str] = list(domains or [])
        if not candidates:
            candidates = list(self.param("domains") or [])
        if not candidates:
            candidates = list(getattr(self.source, "seeds", None) or [])
        base = getattr(self.source, "base_url", None)
        if not candidates and base:
            candidates = [str(base)]
        seeds: list[str] = []
        for candidate in candidates:
            value = str(candidate).strip()
            if not value:
                continue
            if not value.startswith(("http://", "https://")):
                value = f"https://{value.lstrip('/')}"
            seeds.append(value)
        return seeds

    def _classify(self, href: str, text: str) -> LinkDecision | None:
        # The evidence-type vocabulary participates in *classification* too: a
        # "/admissions/fee-structure" page carries no generic navigation signal,
        # so without its patterns the link would be dropped before scoring.
        types = self._evidence_table()
        document_patterns = tuple(self.param("document_filename_patterns") or ()) + tuple(
            pattern.pattern for spec in types.values() for pattern in spec["filename"]
        )
        anchor_patterns = tuple(self.param("anchor_patterns") or ()) + tuple(
            pattern.pattern for spec in types.values() for pattern in spec["anchor"]
        )
        return classify_link(
            href,
            text,
            document_extensions=tuple(self.param("document_extensions") or (".pdf",)),
            page_extensions=tuple(self.param("page_extensions") or (".html", ".htm", ".php")),
            document_patterns=document_patterns,
            anchor_patterns=anchor_patterns,
            navigate_patterns=tuple(self.param("navigate_patterns") or ()),
            skip_patterns=tuple(self.param("skip_patterns") or ()),
        )

    def _bounded_int(self, key: str, default: int) -> int:
        raw = self.param(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise SourceUnavailable(
                "invalid_registry_param",
                detail=f"{self.source_id} params.{key} must be an integer",
            ) from exc
        return max(0, value)

    # ---------------------------------------------------------------- parsing
    def parse_listing(self, text: str, *, base_url: str | None = None) -> list[DocumentRef]:
        """Return only document refs, best-scoring first (pure, offline-testable).

        Each ref carries its matched evidence types and, when a seat/fee family
        was recognised at anchor/filename level, the ``family`` hint the frontier
        persists as ``extraction_family`` for the planner.
        """
        base = base_url or (self._seeds() or [""])[0]
        refs: list[DocumentRef] = []
        for decision in self._decisions(text, base_url=base):
            if decision.kind != "document":
                continue
            types = self._evidence_types_for(decision.url, decision.text)
            metadata: dict[str, Any] = {
                "anchor_text": decision.text,
                "link_score": decision.score,
                "match_reason": decision.reason,
                "found_on": base,
            }
            if types:
                metadata["evidence_types"] = list(types)
            hint = self._family_hint(decision.url, decision.text, types)
            if hint:
                metadata["family"] = hint
            refs.append(self.ref(decision.url, metadata=metadata, priority=decision.score))
        return refs

    def parse_navigation(self, text: str, *, base_url: str) -> list[str]:
        """Return bounded-crawl candidates on the same registrable domain."""
        return [decision.url for decision in self._navigation_decisions(text, base_url=base_url)]

    def _navigation_decisions(self, text: str, *, base_url: str) -> list[LinkDecision]:
        return [
            decision
            for decision in self._decisions(text, base_url=base_url)
            if decision.kind == "navigate"
        ]

    # ------------------------------------------------------------------ crawl
    def _crawl(self, seed: str) -> tuple[list[DocumentRef], str | None]:
        max_depth = self._bounded_int("max_depth", 2)
        max_pages = self._bounded_int("max_pages_per_domain", 200)
        max_documents = self._bounded_int("max_documents_per_domain", 0)
        yield_check_after = self._bounded_int("yield_check_after", 25)
        try:
            min_yield = float(self.param("min_yield_percent", 2.0) or 2.0)
        except (TypeError, ValueError):
            min_yield = 2.0

        # A priority queue, not FIFO: the best-scoring navigation target is
        # visited first, so a bounded page budget is spent on NAAC/IQAC/NIRF/
        # placement/downloads sections before news or galleries.
        queue: list[tuple[int, int, str]] = [(0, 0, seed)]
        visited: set[str] = {seed}
        docs: dict[str, DocumentRef] = {}
        pages = 0
        first_error: str | None = None
        fetched_any = False

        while queue and pages < max_pages:
            _score, depth, url = heapq.heappop(queue)
            try:
                text = self.fetch_text(url)
            except SourceUnavailable as exc:
                if not fetched_any and first_error is None:
                    first_error = exc.reason
                continue
            fetched_any = True
            pages += 1

            # Collect every document the page links, then keep the best N at the
            # end. Stopping at N *during* the BFS would let a generic homepage
            # PDF end the crawl before the NAAC/IQAC page is ever visited.
            for ref in self.parse_listing(text, base_url=url):
                if ref.url in docs:
                    continue
                docs[ref.url] = ref

            if depth < max_depth:
                for decision in self._navigation_decisions(text, base_url=url):
                    if decision.url in visited:
                        continue
                    visited.add(decision.url)
                    heapq.heappush(queue, (-decision.score, depth + 1, decision.url))

            # Yield stop rule: a domain that returns almost no documents after a
            # fair sample is not worth the remaining budget (spec §4.9).
            if pages >= yield_check_after and min_yield > 0:
                yield_percent = 100.0 * len(docs) / pages
                if yield_percent < min_yield:
                    break

        refs = sorted(docs.values(), key=lambda ref: (-ref.priority, ref.url))
        if max_documents:
            refs = self._apply_pdf_first_reserve(refs, max_documents)
        if not fetched_any:
            return [], first_error or "unreachable"
        return refs, None

    def _apply_pdf_first_reserve(
        self, refs: list[DocumentRef], max_documents: int
    ) -> list[DocumentRef]:
        """Trim to the document cap, reserving a share for candidate evidence.

        ``pdf_first_reserve_percent`` of ``max_documents_per_domain`` may only be
        filled by evidence-typed documents (a link whose seat/fee/prospectus/
        placement/NIRF vocabulary matched, vetoes applied). News, tender,
        gallery and untitled documents can only fill the unreserved remainder —
        they never consume the reserve. Evidence documents keep normal order;
        unused reserved capacity simply goes unfilled by non-evidence documents.
        """
        reserve = self._reserve_size(max_documents)
        if reserve <= 0:
            return refs[:max_documents]
        other_cap = max(0, max_documents - reserve)
        kept: list[DocumentRef] = []
        others_admitted = 0
        for ref in refs:
            if len(kept) >= max_documents:
                break
            if ref.metadata.get("evidence_types"):
                kept.append(ref)
            elif others_admitted < other_cap:
                kept.append(ref)
                others_admitted += 1
        return kept

    def fetch(self, *, domains: Iterable[str] | None = None) -> Iterable[DocumentRef]:
        if getattr(self.source, "abstain", False):
            raise SourceUnavailable("abstained", detail="source is marked abstain in the registry")
        seeds = self._seeds(domains)
        if not seeds:
            raise SourceUnavailable(
                "no_domain",
                detail=(
                    f"{self.source_id} has no institution domain; pass the discovered "
                    "domains at runtime (params.domains or the domains= argument)"
                ),
            )
        for key in ("max_depth", "max_pages_per_domain"):
            require_param(self.param(key), name=key, source_id=self.source_id)

        refs: list[DocumentRef] = []
        seen: set[str] = set()
        first_reason: str | None = None
        for seed in seeds:
            found, reason = self._crawl(seed)
            if reason is not None and first_reason is None:
                first_reason = reason
            for ref in found:
                if ref.url in seen:
                    continue
                seen.add(ref.url)
                refs.append(ref)
        if not refs and first_reason is not None:
            # Nothing was readable: "no documents" would be a lie.
            raise SourceUnavailable(first_reason, detail="no institution page could be read")
        return refs
