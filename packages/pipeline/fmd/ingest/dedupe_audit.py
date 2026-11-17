"""Offline audit of simhash-dropped documents (brief §7, task 6).

The crawler records each near-duplicate decision as a JSON outcome
(``url``, ``compared_url``, ``similarity``, ``hamming_distance``, ``mode``,
``source_id``). This module turns those outcomes into a verdict *without*
re-fetching anything: it compares the two URLs' document identity.

A collapse is **boilerplate** when the two URLs name the same document (same
basename, a mirror, a paginated wrapper). It is a **false positive** when two
*distinctly named* documents were collapsed — two annual reports, a prospectus
and a circular, an English and a Hindi edition. The latter is exactly what
happens when simhash is computed over raw PDF bytes decoded as text: the
``%PDF``/object-stream/font boilerplate dominates and every PDF looks alike.

The audit never edits content; it returns numbers and examples so a source
policy (``params.dedupe.mode`` / ``simhash_threshold``) can be justified.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlsplit

_TOKEN = re.compile(r"[a-z0-9]+")
#: Similarity at or above this is "the same document" for a 64-bit simhash.
SAME_DOCUMENT_SIMILARITY = 0.97
#: Hamming distance at or below this is "the same document" at threshold 3.
SAME_DOCUMENT_DISTANCE = 3


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def document_stem(url: str) -> str:
    """Filename stem of a URL, without extension; paginated URLs keep the path.

    Years are kept: ``annual-report-2021-22`` and ``annual-report-2022-23`` are
    different editions of a publication, not the same document.
    """
    path = unquote(urlsplit(str(url or "")).path or "")
    name = path.rstrip("/").rsplit("/", 1)[-1]
    name = name.rsplit(".", 1)[0] if "." in name else name
    tokens = _TOKEN.findall(name.casefold())
    return " ".join(tokens)


def is_same_document(url: str, compared_url: str | None) -> bool:
    """True when two URLs plausibly address the same document (not a false +)."""
    if not compared_url:
        return False
    left, right = urlsplit(str(url)), urlsplit(str(compared_url))
    if left.path.rstrip("/") == right.path.rstrip("/"):
        return True
    left_stem, right_stem = document_stem(url), document_stem(compared_url)
    if left_stem and left_stem == right_stem:
        # Same filename on two hosts is a mirror of one document.
        return True
    # A paginated listing repeats the same path prefix with only a page number.
    left_parts = [part for part in left.path.split("/") if part]
    right_parts = [part for part in right.path.split("/") if part]
    if left_parts[:-1] == right_parts[:-1] and left_parts and right_parts:
        return left_parts[-1].isdigit() and right_parts[-1].isdigit()
    return False


@dataclass(frozen=True)
class DedupeVerdict:
    url: str
    compared_url: str | None
    similarity: float | None
    hamming_distance: int | None
    mode: str
    source_id: str | None
    verdict: str  # boilerplate | false_positive | unknown

    @property
    def dropped(self) -> bool:
        return True


@dataclass(frozen=True)
class DedupeAudit:
    total: int
    boilerplate: int
    false_positives: int
    unknown: int
    by_source: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    examples: tuple[DedupeVerdict, ...] = ()

    @property
    def false_positive_ratio(self) -> float:
        return self.false_positives / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "boilerplate": self.boilerplate,
            "false_positives": self.false_positives,
            "unknown": self.unknown,
            "false_positive_ratio": round(self.false_positive_ratio, 4),
            "by_source": {key: dict(value) for key, value in self.by_source.items()},
            "examples": [
                {
                    "url": item.url,
                    "compared_url": item.compared_url,
                    "similarity": item.similarity,
                    "hamming_distance": item.hamming_distance,
                    "mode": item.mode,
                    "source_id": item.source_id,
                    "verdict": item.verdict,
                }
                for item in self.examples
            ],
        }


def classify(entry: Any) -> DedupeVerdict:
    """Classify one recorded dedupe outcome."""
    url = str(_get(entry, "url", "") or "")
    compared_url = _get(entry, "compared_url", None)
    compared_url = str(compared_url) if compared_url else None
    similarity = _get(entry, "similarity", None)
    distance = _get(entry, "hamming_distance", None)
    try:
        similarity_value = None if similarity is None else float(similarity)
    except (TypeError, ValueError):
        similarity_value = None
    try:
        distance_value = None if distance is None else int(distance)
    except (TypeError, ValueError):
        distance_value = None
    mode = str(_get(entry, "mode", "") or "unknown")
    source_id = _get(entry, "source_id", None)

    if not compared_url:
        verdict = "unknown"
    elif is_same_document(url, compared_url):
        verdict = "boilerplate"
    elif (similarity_value is not None and similarity_value >= SAME_DOCUMENT_SIMILARITY) or (
        distance_value is not None and distance_value <= SAME_DOCUMENT_DISTANCE
    ):
        # Distinct document names, yet treated as identical: a false positive.
        verdict = "false_positive"
    else:
        verdict = "unknown"
    return DedupeVerdict(
        url=url,
        compared_url=compared_url,
        similarity=similarity_value,
        hamming_distance=distance_value,
        mode=mode,
        source_id=str(source_id) if source_id else None,
        verdict=verdict,
    )


def audit(entries: Iterable[Any], *, examples: int = 10) -> DedupeAudit:
    """Aggregate recorded dedupe outcomes into an honest verdict."""
    verdicts = [classify(entry) for entry in entries]
    counts = {"boilerplate": 0, "false_positive": 0, "unknown": 0}
    by_source: dict[str, dict[str, int]] = {}
    for item in verdicts:
        counts[item.verdict] += 1
        source = item.source_id or "unknown"
        bucket = by_source.setdefault(source, {"total": 0, "boilerplate": 0, "false_positive": 0})
        bucket["total"] += 1
        bucket[item.verdict] = bucket.get(item.verdict, 0) + 1
    picked = tuple(item for item in verdicts if item.verdict == "false_positive")[
        :examples
    ] or tuple(verdicts[:examples])
    return DedupeAudit(
        total=len(verdicts),
        boilerplate=counts["boilerplate"],
        false_positives=counts["false_positive"],
        unknown=counts["unknown"],
        by_source=by_source,
        examples=picked,
    )


def audit_from_run(summary: Mapping[str, Any], *, examples: int = 10) -> DedupeAudit:
    """Read the ``metadata.dedupe`` list a crawl run records (never guessed)."""
    metadata = summary.get("metadata") or {}
    entries = metadata.get("dedupe") or []
    if not entries:
        entries = [
            item for item in (summary.get("outcomes") or ()) if _get(item, "outcome") == "deduped"
        ]
    return audit(entries, examples=examples)


__all__ = [
    "DedupeAudit",
    "DedupeVerdict",
    "SAME_DOCUMENT_DISTANCE",
    "SAME_DOCUMENT_SIMILARITY",
    "audit",
    "audit_from_run",
    "classify",
    "document_stem",
    "is_same_document",
]
