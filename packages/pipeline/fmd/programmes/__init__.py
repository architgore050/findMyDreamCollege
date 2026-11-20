"""Programme scoping (W9): decide whether a materialised programme IS the
product's target field, in three tiers with precedence manual > exclude >
include > semantic > uncertain.

* Tier 1 — :mod:`fmd.programmes.taxonomy`: deterministic include/exclude rules
  from ``config/programmes.yaml`` (country-keyed; other countries are empty
  stubs, never invented patterns).
* Tier 2 — :mod:`fmd.programmes.classifier`: semantic equivalence through the
  provider-agnostic router, evidence-cited, cached, conservative
  (``uncertain`` → ``needs_review``, never ranked).
* Tier 3 — :mod:`fmd.programmes.service`: the learning loop. A human
  ``fmd programmes decide`` verdict becomes a ``manual`` decision that
  Tier 1 consumes on the next run; :func:`requalify_programmes` re-scopes
  existing rows and folds the row-numbering duplicates.
"""

from fmd.programmes.classifier import (
    EquivalenceCandidate,
    EquivalenceDecision,
    EquivalenceItem,
    EquivalenceResponse,
    build_prompt,
    classify,
    decision_cache_key,
    has_field_signal,
)
from fmd.programmes.normalise import normalise_programme_label, strip_row_numbering
from fmd.programmes.taxonomy import (
    COUNTRY_TAXONOMY_MISSING,
    COUNTRY_TAXONOMY_STUB,
    RuleVerdict,
    ScopeDecision,
    Taxonomy,
)

__all__ = [
    "COUNTRY_TAXONOMY_MISSING",
    "COUNTRY_TAXONOMY_STUB",
    "EquivalenceCandidate",
    "EquivalenceDecision",
    "EquivalenceItem",
    "EquivalenceResponse",
    "RuleVerdict",
    "ScopeDecision",
    "Taxonomy",
    "build_prompt",
    "classify",
    "decision_cache_key",
    "has_field_signal",
    "normalise_programme_label",
    "strip_row_numbering",
]
