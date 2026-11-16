"""Institution-site evidence discovery (spec §4.9).

Given a college domain, find the PDFs that actually carry outcomes evidence:
NAAC SSR/AQAR, NIRF-DCS mirrors, annual/IQAC reports, placement reports and
prospectuses. This is discovery, not extraction: the adapter yields
``DocumentRef`` values and never reads a placement table.

The crawl is bounded by registry parameters (``max_depth``,
``max_pages_per_domain``, ``min_yield_percent``) and stays on one registrable
domain. Login walls, ``javascript:`` anchors, archives and images are hard
skips. robots is enforced by the fetch ladder; a robots refusal is an
abstention, never an empty success.
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Iterable

from fmd.adapters.base import BaseAdapter, SourceUnavailable, anchors, resolve
from fmd.adapters.links import (
    LinkDecision,
    classify_link,
    rank_decisions,
    require_param,
    same_site,
)
from fmd.discover.base import DocumentRef

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


class InstitutionSiteAdapter(BaseAdapter):
    name = "institution_site"

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
        placement/AQAR/SSR/annual-report link.
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
        if self._matches(self._deprioritise_patterns(), f"{decision.url} {decision.text}"):
            score += weights["deprioritised"]
        return score

    def _decisions(self, text: str, *, base_url: str) -> list[LinkDecision]:
        decisions: list[LinkDecision] = []
        seen: set[str] = set()
        for href, label in anchors(text):
            decision = self._classify(href, label)
            if decision is None:
                continue
            url = resolve(base_url, decision.url)
            if url in seen or not same_site(base_url, url):
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
        return classify_link(
            href,
            text,
            document_extensions=tuple(self.param("document_extensions") or (".pdf",)),
            page_extensions=tuple(self.param("page_extensions") or (".html", ".htm", ".php")),
            document_patterns=tuple(self.param("document_filename_patterns") or ()),
            anchor_patterns=tuple(self.param("anchor_patterns") or ()),
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
        """Return only document refs, best-scoring first (pure, offline-testable)."""
        base = base_url or (self._seeds() or [""])[0]
        return [
            self.ref(
                decision.url,
                metadata={
                    "anchor_text": decision.text,
                    "link_score": decision.score,
                    "match_reason": decision.reason,
                    "found_on": base,
                },
                priority=decision.score,
            )
            for decision in self._decisions(text, base_url=base)
            if decision.kind == "document"
        ]

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
            refs = refs[:max_documents]
        if not fetched_any:
            return [], first_error or "unreachable"
        return refs, None

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
