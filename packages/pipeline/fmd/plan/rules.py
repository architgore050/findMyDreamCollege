"""Family-detection rules as data (planner invariant 7).

Every signal is a lowercase token matched against a normalised view of one
document attribute. There are no source ids and no institution names here: the
vocabulary describes *document shapes* that the extraction schemas already
model, and a deployment can replace :data:`DEFAULT_RULES` wholesale.

* ``url`` tokens match the URL path and query.
* ``anchor`` tokens match the text of the link that produced the URL.
* ``extension`` matches the URL's file extension.
* ``text`` tokens match the first page(s) of extracted text.

``negative_*`` tokens are the veto channel. A URL like
``Guidelines-…-Fee-for-recognition-of-Institution-as-Research-Centre-for-Ph.D-Course``
contains the token ``fee`` but the fee is an administrative charge paid by the
institution, not a student fee structure. When a negative token matches a
channel, that channel's positive vote is cancelled (and the veto is recorded in
the returned signals), so the family cannot be declared from it alone.

Weights and the accept floor are separate from the rules so a stricter run does
not need a code fork. A rule that scores on three independent tokens is stronger
than one that repeats a single token, because each token contributes once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SignalWeights:
    """Contribution of one independent token match, per signal (0..1 each)."""

    url: float = 0.35
    anchor: float = 0.45
    extension: float = 0.20
    text: float = 0.30
    #: Minimum winning score for a family to be declared.
    min_score: float = 0.45
    #: Minimum distance from the runner-up; below this the document is ambiguous.
    margin: float = 0.10


@dataclass(frozen=True)
class FamilyRule:
    family: str
    url_tokens: tuple[str, ...] = ()
    anchor_tokens: tuple[str, ...] = ()
    text_tokens: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    #: Phrases that mark an administrative sense of an otherwise evidential
    #: word ("fee" in *Fee for recognition of Institution as Research Centre*).
    #: A match cancels the vote of the channel it appears in.
    negative_url_tokens: tuple[str, ...] = ()
    negative_anchor_tokens: tuple[str, ...] = ()
    negative_text_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanRules:
    """A complete rule set; ``min_mentions`` keeps entity mining bounded."""

    families: tuple[FamilyRule, ...] = ()
    weights: SignalWeights = field(default_factory=SignalWeights)


DEFAULT_RULES = PlanRules(
    families=(
        FamilyRule(
            family="placement_report",
            url_tokens=("placement", "placements", "career", "recruit"),
            anchor_tokens=("placement report", "placement", "career", "recruit"),
            text_tokens=("placement rate", "students placed", "median salary", "recruiters"),
            extensions=(),
        ),
        FamilyRule(
            family="fee_structure",
            url_tokens=("fee", "fees", "tuition", "fee-order"),
            anchor_tokens=("fee structure", "tuition fee", "fee order", "fee notification"),
            text_tokens=(
                "tuition fee",
                "fee structure",
                "hostel fee",
                "per annum",
                "fee details",
                "annual fee",
                "fee & other details",
                "fee and other details",
            ),
            extensions=(),
            # Observed 2026-11-17 (RUHS): "Guidelines … Fee for recognition of
            # Institution as Research Centre for Ph.D. Course" — an
            # administrative order whose URL token "fee" is an institution-paid
            # charge, never a student fee structure. Any of these phrases
            # cancels the fee vote of the channel it appears in.
            negative_url_tokens=(
                "fee for recognition",
                "recognition of institution",
                "research centre",
                "research center",
                "affiliation fee",
                "inspection fee",
                "revaluation fee",
                "re-evaluation fee",
            ),
            negative_anchor_tokens=(
                "fee for recognition",
                "recognition of institution",
                "research centre",
                "research center",
                "affiliation fee",
                "inspection fee",
            ),
            negative_text_tokens=(
                "fee for recognition",
                "recognition of institution",
                "research centre",
                "research center",
                "affiliation fee",
                "inspection fee",
                "revaluation fee",
                "re-evaluation fee",
            ),
        ),
        FamilyRule(
            family="seat_matrix",
            url_tokens=("seat", "seats", "matrix", "intake", "cutoff", "cut-off"),
            anchor_tokens=("seat matrix", "sanctioned intake", "intake", "cutoff"),
            text_tokens=("seat matrix", "sanctioned intake", "number of seats", "intake capacity"),
            extensions=(),
        ),
        FamilyRule(
            family="accreditation",
            url_tokens=("naac", "accredit", "aqar", "ssr", "nirf", "iqac"),
            anchor_tokens=(
                "naac",
                "accreditation",
                "grade",
                "cgpa",
                "aqar",
                "ssr",
                "nirf",
                "iqac",
                "annual report",
                "self study",
            ),
            text_tokens=("naac", "accredited", "cgpa", "grade", "aqar", "ssr"),
            extensions=(),
        ),
        FamilyRule(
            family="curriculum",
            url_tokens=("syllabus", "curriculum", "syllabi", "prospectus"),
            anchor_tokens=("syllabus", "curriculum", "prospectus", "course structure"),
            text_tokens=("semester", "course code", "curriculum", "credit"),
            extensions=(),
        ),
        FamilyRule(
            family="exam_notification",
            # A bare "notification" is a generic administrative heading (a fee
            # circular and a Ph.D. affiliation notice both carry it). Only an
            # exam-shaped URL defers/accepted on URL evidence.
            url_tokens=(
                "examination",
                "entrance",
                "exam-notification",
                "admission-notification",
                "exam-schedule",
            ),
            anchor_tokens=("notification", "entrance", "exam", "schedule"),
            text_tokens=("entrance test", "last date", "application form", "examination"),
            extensions=(),
        ),
    ),
)


def normalise(value: str | None) -> str:
    """Lowercase and collapse separators so ``Seat-Matrix`` matches ``seat matrix``."""
    if not value:
        return ""
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def extension_of(url: str) -> str:
    path = urlsplit(url).path.rsplit("/", 1)[-1]
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def _matches(tokens: tuple[str, ...], haystack: str) -> int:
    return sum(1 for token in tokens if token and normalise(token) in haystack)


@dataclass(frozen=True)
class FamilyScore:
    family: str
    score: float
    signals: dict[str, int]


def _rule_negatives(rule: FamilyRule) -> dict[str, tuple[str, ...]]:
    """Negative tokens per signal channel; tokens are normalised per match."""
    return {
        "url": rule.negative_url_tokens,
        "anchor": rule.negative_anchor_tokens,
        "text": rule.negative_text_tokens,
    }


def score_families(
    *,
    url: str = "",
    anchor_text: str | None = None,
    text_sample: str | None = None,
    rules: PlanRules = DEFAULT_RULES,
) -> list[FamilyScore]:
    """Rank every configured family against one document. Deterministic order."""
    weights = rules.weights
    url_key = normalise(f"{urlsplit(url).path} {urlsplit(url).query}")
    anchor_key = normalise(anchor_text)
    text_key = normalise(text_sample)
    extension = extension_of(url)
    haystacks = {"url": url_key, "anchor": anchor_key, "text": text_key}

    scores: list[FamilyScore] = []
    for rule in rules.families:
        signals = {
            "url": _matches(rule.url_tokens, url_key),
            "anchor": _matches(rule.anchor_tokens, anchor_key),
            "extension": int(bool(extension) and extension in rule.extensions),
            "text": _matches(rule.text_tokens, text_key),
        }
        # A negative-context phrase cancels the vote of the channel it appears
        # in; the veto itself stays visible in the signals for the run ledger.
        negatives = _rule_negatives(rule)
        for channel, tokens in negatives.items():
            hits = _matches(tokens, haystacks[channel]) if tokens else 0
            if hits:
                signals[f"{channel}_vetoed"] = hits
                signals[channel] = 0
        score = (
            signals["url"] * weights.url
            + signals["anchor"] * weights.anchor
            + signals["extension"] * weights.extension
            + signals["text"] * weights.text
        )
        scores.append(
            FamilyScore(family=rule.family, score=round(min(1.0, score), 6), signals=signals)
        )
    return sorted(scores, key=lambda item: (-item.score, item.family))


def detect_family(
    *,
    url: str = "",
    anchor_text: str | None = None,
    text_sample: str | None = None,
    rules: PlanRules = DEFAULT_RULES,
    min_score: float | None = None,
    margin: float | None = None,
) -> tuple[str | None, float, dict[str, int], str | None]:
    """Return ``(family, confidence, signals, abstain_reason)``.

    Resolution order is explicit: registry-declared family is handled by the
    planner, so this function only answers the heuristic question. A tie or a
    thin margin is ``ambiguous_family``; no winner is ``family_undetected``.

    ``min_score`` / ``margin`` let :mod:`fmd.plan.planner` apply the registry's
    per-phase thresholds without forking the rule set. They default to the
    rule weights so standalone callers (tests, fixtures) stay deterministic.
    """
    floor = rules.weights.min_score if min_score is None else min_score
    separation = rules.weights.margin if margin is None else margin
    ranked = [
        item
        for item in score_families(
            url=url, anchor_text=anchor_text, text_sample=text_sample, rules=rules
        )
        if item.score > 0
    ]
    if not ranked:
        return None, 0.0, {}, "family_undetected"

    best = ranked[0]
    runner_up = ranked[1].score if len(ranked) > 1 else 0.0
    if best.score < floor:
        return None, best.score, dict(best.signals), "family_undetected"
    if len(ranked) > 1 and best.score - runner_up < separation:
        return None, round(best.score, 6), dict(best.signals), "ambiguous_family"
    return best.family, best.score, dict(best.signals), None
