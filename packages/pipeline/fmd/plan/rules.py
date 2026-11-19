"""Family-detection rules as data (planner invariant 7).

Every signal is a lowercase token matched against a normalised view of one
document attribute. There are no source ids and no institution names here: the
vocabulary describes *document shapes* that the extraction schemas already
model, and a deployment can replace :data:`DEFAULT_RULES` wholesale.

Signal channels (one score contribution each, weights in :class:`SignalWeights`):

* ``url`` tokens match the URL path and query.
* ``anchor`` tokens match the text of the link that produced the URL.
* ``extension`` matches the URL's file extension.
* ``text`` tokens match the first page(s) of extracted text (T0 sample).
* ``heading`` tokens match *heading-shaped* lines of the sample (short,
  non-sentence lines such as ``Annual Report 2023-24`` or ``5. PLACEMENTS``).
* ``title`` tokens match the document's title line — the first heading-shaped
  line. A document that names itself on its cover is decided by its cover: when
  exactly one family's title tokens fire, that family wins outright
  (``title_decisive``) and no body-table noise can tie it.
* ``recurring`` tokens count occurrences; a token that recurs at least
  ``recurring_min`` times contributes once. A fee structure repeats the word
  ``fee`` dozens of times; an office order that mentions a fee once does not.
* ``structure`` is the confidence weight the planner assigns to a filename-shape
  family hint (:func:`fmd.plan.roles.family_hints_from_structure`); it is the
  weakest tier and only fires when registry, text and URL token evidence are
  all silent.

``negative_*`` tokens are the veto channel. A URL like
``Guidelines-…-Fee-for-recognition-of-Institution-as-Research-Centre-for-Ph.D-Course``
contains the token ``fee`` but the fee is an administrative charge paid by the
institution, not a student fee structure. When a negative token matches a
channel, that channel's positive vote is cancelled (and the veto is recorded in
the returned signals), so the family cannot be declared from it alone. A
negative match in the text sample cancels every text-derived channel of that
family (``text``, ``heading``, ``recurring`` and ``title``) — the cover text of
the observed administrative order names ``Fee Details`` and ``Annual fee`` while
never being a student fee structure.

**Prospectus shape chain.** A prospectus / admission brochure is a bundle of
several families (fees, intake, curriculum), so a bare ``prospectus`` vote
cannot pick one. When the prospectus identity appears in the title or heading
lines of the acquired text, the chain resolves the document by its own tables:
fee table rows → ``fee_structure``, else intake table rows → ``seat_matrix``,
else ``curriculum``. The resolution and the row counts are recorded in the
signals; the pseudo-family ``prospectus`` never escapes :func:`detect_family`.

Weights and the accept floor are separate from the rules so a stricter run does
not need a code fork. A rule that scores on three independent tokens is stronger
than one that repeats a single token, because each token contributes once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SignalWeights:
    """Contribution of one independent token match, per signal (0..1 each)."""

    url: float = 0.35
    anchor: float = 0.45
    extension: float = 0.20
    text: float = 0.30
    #: One heading-shaped line naming the family ("Annual Report 2023-24").
    text_heading: float = 0.35
    #: One recurring token (occurs >= recurring_min times in the sample).
    text_recurring: float = 0.20
    #: The document's title line naming the family — cover-page identity.
    title: float = 0.35
    #: Confidence the planner assigns to a filename-shape family hint
    #: (:mod:`fmd.plan.roles`); the weakest acceptance tier.
    structure: float = 0.30
    #: A prospectus resolved by its own fee/intake table rows.
    shape_table: float = 0.35
    #: Prospectus-shape identity itself (cover announces a prospectus).
    shape_identity: float = 0.30
    #: Minimum winning score for a family to be declared.
    min_score: float = 0.45
    #: Minimum distance from the runner-up; below this the document is ambiguous.
    margin: float = 0.10
    #: Occurrences needed before a recurring token counts.
    recurring_min: int = 3
    #: Table rows needed before a prospectus resolves to fee/seat.
    table_min_rows: int = 3


@dataclass(frozen=True)
class FamilyRule:
    family: str
    url_tokens: tuple[str, ...] = ()
    anchor_tokens: tuple[str, ...] = ()
    text_tokens: tuple[str, ...] = ()
    #: Tokens matched against heading-shaped lines ("Annual Report 2023-24").
    heading_tokens: tuple[str, ...] = ()
    #: Tokens counted across the sample; each token with >= ``recurring_min``
    #: occurrences contributes once (independent of the other channels).
    recurring_tokens: tuple[str, ...] = ()
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
            text_tokens=(
                "placement rate",
                "students placed",
                "median salary",
                "recruiters",
                "placement report",
                "campus placement",
                "placement cell",
                "placement statistics",
                "number of students placed",
                "highest salary",
                "average salary",
            ),
            heading_tokens=("placement report", "placement", "placements", "campus placement"),
            recurring_tokens=("placement", "placements"),
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
                "fee schedule",
                "fee circular",
                "fee notification",
                "fee payable",
                "prescribed fee",
                "course fee",
                "programme fee",
                "admission fee",
                "mess fee",
                "library fee",
                "examination fee",
                "caution deposit",
                "fee shall be",
            ),
            heading_tokens=(
                "fee structure",
                "fee schedule",
                "fee circular",
                "fee notification",
                "fee details",
                "fees payable",
                "fee & other details",
                "fee and other details",
            ),
            recurring_tokens=("fee", "fees", "tuition fee", "hostel fee", "mess fee"),
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
            text_tokens=(
                "seat matrix",
                "sanctioned intake",
                "number of seats",
                "intake capacity",
                "seats available",
                "sanctioned strength",
                "seat distribution",
                "distribution of seats",
                "intake statement",
                "intake position",
            ),
            heading_tokens=(
                "seat matrix",
                "sanctioned intake",
                "intake capacity",
                "number of seats",
                "seat distribution",
            ),
            recurring_tokens=("seat", "seats", "intake"),
            extensions=(),
        ),
        FamilyRule(
            family="accreditation",
            url_tokens=(
                "naac",
                "accredit",
                "aqar",
                "ssr",
                "nirf",
                "iqac",
                "annual report",
                "annualreport",
                "self study",
                "selfstudy",
            ),
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
                "self appraisal",
                "annual quality assurance",
            ),
            text_tokens=(
                "naac",
                "accredited",
                "cgpa",
                "grade",
                "aqar",
                "ssr",
                "annual report",
                "national assessment and accreditation council",
                "national institutional ranking framework",
                "self study report",
                "self appraisal",
                "annual quality assurance report",
                "iqac",
                "internal quality assurance",
                "peer team",
            ),
            heading_tokens=(
                "annual report",
                "naac",
                "accreditation",
                "aqar",
                "ssr",
                "nirf",
                "iqac",
                "self study report",
                "annual quality assurance report",
                "national institutional ranking framework",
            ),
            recurring_tokens=("naac", "accreditation", "accredited", "aqar", "ssr", "nirf", "iqac"),
            extensions=(),
        ),
        FamilyRule(
            family="curriculum",
            url_tokens=("syllabus", "curriculum", "syllabi", "prospectus"),
            anchor_tokens=("syllabus", "curriculum", "prospectus", "course structure"),
            text_tokens=(
                "semester",
                "course code",
                "curriculum",
                "credit",
                "eligibility criteria",
                "duration of the course",
                "medium of instruction",
                "course content",
                "scheme of examination",
                "programme outcomes",
                "course objectives",
            ),
            heading_tokens=("curriculum", "syllabus", "course structure", "scheme of examination"),
            recurring_tokens=("semester", "curriculum", "syllabus"),
            extensions=(),
        ),
        FamilyRule(
            family="exam_notification",
            # A bare "notification" is a generic administrative heading (a fee
            # circular and a Ph.D. affiliation notice both carry it). Only an
            # exam-shaped URL defers/accepted on URL evidence. Schedule-shaped
            # headings ("Examination Schedule", "Date Sheet") are deliberately
            # absent: a date sheet is not a registration notification (W5
            # negative guard).
            url_tokens=(
                "examination",
                "entrance",
                "exam-notification",
                "admission-notification",
                "exam-schedule",
            ),
            anchor_tokens=("notification", "entrance", "exam", "schedule"),
            text_tokens=("entrance test", "last date", "application form", "examination"),
            heading_tokens=(
                "examination notification",
                "admission notification",
                "entrance examination",
                "registration notification",
            ),
            recurring_tokens=(),
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


def _recurring_matches(tokens: tuple[str, ...], haystack: str, min_count: int) -> int:
    """Distinct tokens occurring at least ``min_count`` times in the sample.

    Counting is substring-based on the normalised sample, so ``fee`` also
    counts inside ``fees`` and ``fee structure``: a recurring count is evidence
    of repetition, not of independent vocabulary. Each qualifying token
    contributes once.
    """
    if min_count < 2:
        return 0
    return sum(1 for token in tokens if token and haystack.count(normalise(token)) >= min_count)


# --------------------------------------------------------------------- headings
#: Heading-shaped lines are the document's own outline. A sample line counts as
#: a heading when it is short, carries no sentence punctuation, and reads as a
#: label rather than a sentence: ALL-CAPS, numbered (``5. PLACEMENTS``),
#: predominantly capitalised, or a short sentence-case label
#: ("Fee Structure 2025-26"). Sentence connectors disqualify the last shape.
_HEADING_MAX_CHARS = 120
_HEADING_MAX_WORDS = 14
_LABEL_MAX_WORDS = 8
#: Cover-title scan depth: the document-type line sits near the top of the
#: cover; scanning deeper would mistake section headings for identity.
_TITLE_SCAN_HEADING_LINES = 5
_NUMBERED_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[a-z]\)|\(?[ivxlc]+\))\s+\S", re.IGNORECASE)
_SENTENCEISH = re.compile(
    r"\b(is|are|was|were|be|been|has|have|had|shall|will|the|this|that)\b", re.IGNORECASE
)
_TRAILING_PUNCTUATION = (".", "!", "?", ";", ",")


def _capitalised_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    capitalised = sum(1 for word in words if word[:1].isupper() or word[:1].isdigit())
    return capitalised / len(words)


def is_heading_line(line: str) -> bool:
    """True when a raw text line is heading-shaped (document outline, not prose)."""
    text = line.strip()
    if not 2 <= len(text) <= _HEADING_MAX_CHARS:
        return False
    if text.endswith(_TRAILING_PUNCTUATION):
        return False
    words = text.split()
    if not words or len(words) > _HEADING_MAX_WORDS:
        return False
    if text.isupper():
        return True
    if _NUMBERED_HEADING.match(text):
        return True
    if _capitalised_ratio(words) >= 0.6:
        return True
    # A short sentence-case label with no sentence inside ("Fee Structure
    # 2025-26") is still a heading; a wrapped prose fragment is not.
    return (
        len(words) <= _LABEL_MAX_WORDS
        and words[0][:1].isupper()
        and not _SENTENCEISH.search(text)
    )


def heading_lines(sample: str | None) -> list[str]:
    """The heading-shaped lines of a bounded sample, in document order."""
    if not sample:
        return []
    return [line for line in sample.splitlines() if is_heading_line(line)]


def title_candidates(sample: str | None) -> list[str]:
    """The first few heading lines of a sample — the cover-page title area.

    Real covers start with the institution name ("Demo Institute of Health
    Sciences") and name the document on a following line ("Annual Report
    2023-24"). The scan is bounded: at most the first five heading lines.
    """
    lines = heading_lines(sample)[:_TITLE_SCAN_HEADING_LINES]
    if lines:
        return lines
    if sample:
        for line in sample.splitlines():
            text = line.strip()
            if text and len(text) <= _HEADING_MAX_CHARS:
                return [text]
    return []


def title_line(sample: str | None, rules: PlanRules | None = None) -> str:
    """The document's title line: the first cover heading that names a family.

    Without rules, the first heading candidate is returned. With rules, the
    first candidate whose text matches a family's heading tokens is the title —
    the line where the document says what it is.
    """
    candidates = title_candidates(sample)
    if not candidates:
        return ""
    if rules is not None:
        for line in candidates:
            key = normalise(line)
            if any(
                rule.heading_tokens and _matches(rule.heading_tokens, key)
                for rule in rules.families
            ):
                return line
    return candidates[0]


# ------------------------------------------------------------- prospectus shape
#: Identity vocabulary: a prospectus announces itself on its cover. Bare
#: "brochure" is too generic (a placement brochure is a placement_report), so
#: only the admission shapes count.
_PROSPECTUS_IDENTITY_TOKENS = (
    "prospectus",
    "admission brochure",
    "admissions brochure",
    "admission brochure & fees",
)
#: Lines that carry one of these words *and* an amount are fee-table rows.
_FEE_ROW_VOCABULARY = ("fee", "fees", "tuition")
#: Lines that carry one of these words *and* a number are intake-table rows.
_INTAKE_ROW_VOCABULARY = ("seat", "seats", "intake", "sanctioned")
_AMOUNT = re.compile(
    r"(?:rs\.?|inr|₹|rupees)\s*[\d(]|\d{1,3}(?:,\d{3})+|\b\d{3,7}\b", re.IGNORECASE
)
_NUMBER = re.compile(r"\b\d{1,4}\b")


def _table_rows(sample: str, vocabulary: tuple[str, ...], value: re.Pattern[str]) -> int:
    """Count tabular-looking rows: vocabulary word and an amount/number per line."""
    rows = 0
    for line in sample.splitlines():
        lowered = line.lower()
        if any(word in lowered for word in vocabulary) and value.search(line):
            rows += 1
    return rows


@dataclass(frozen=True)
class ShapeResolution:
    """The outcome of the prospectus shape chain (never a bare ``prospectus``)."""

    family: str
    confidence: float
    via: str
    identity: tuple[str, ...]
    fee_rows: int = 0
    intake_rows: int = 0


def resolve_prospectus_shape(
    *,
    url: str = "",
    anchor_text: str | None = None,
    text_sample: str | None = None,
    rules: PlanRules = DEFAULT_RULES,
    table_min_rows: int | None = None,
) -> ShapeResolution | None:
    """Resolve a prospectus / admission brochure into a real extraction family.

    Fires only on cover-page identity (title or heading lines of a non-empty
    text sample): a URL that merely contains "prospectus" is Phase A deferral
    territory, not a classification. The chain is the spec'd precedence:

    fee table rows >= ``table_min_rows`` → ``fee_structure``; else intake table
    rows → ``seat_matrix``; else ``curriculum``. Fee-table evidence is ignored
    when the fee family's text veto fires (the administrative-fee case).
    """
    sample = text_sample or ""
    if not sample.strip():
        return None
    heading_key = normalise(" ".join(heading_lines(sample)))
    # Identity is cover evidence only: the first heading-shaped lines. The
    # prose fallback of :func:`title_candidates` must not apply — a body
    # sentence mentioning "prospectus" is not a prospectus.
    cover_lines = heading_lines(sample)[:_TITLE_SCAN_HEADING_LINES]
    first_heading_key = normalise(cover_lines[0]) if cover_lines else ""
    text_key = normalise(sample)
    weights = rules.weights
    min_rows = weights.table_min_rows if table_min_rows is None else table_min_rows

    identity: list[str] = []
    if any(token in first_heading_key for token in _PROSPECTUS_IDENTITY_TOKENS):
        identity.append("title")
    elif any(token in heading_key for token in _PROSPECTUS_IDENTITY_TOKENS):
        identity.append("heading")
    if not identity:
        return None

    fee_rule = next((rule for rule in rules.families if rule.family == "fee_structure"), None)
    vetoed = False
    if fee_rule is not None:
        vetoed = bool(_matches(fee_rule.negative_text_tokens, text_key))

    fee_rows = 0 if vetoed else _table_rows(sample, _FEE_ROW_VOCABULARY, _AMOUNT)
    intake_rows = _table_rows(sample, _INTAKE_ROW_VOCABULARY, _NUMBER)
    if fee_rows >= min_rows:
        family, via = "fee_structure", "fee_table"
    elif intake_rows >= min_rows:
        family, via = "seat_matrix", "intake_table"
    else:
        family, via = "curriculum", "default"

    confidence = weights.shape_identity + (weights.shape_table if via != "default" else 0.0)
    if "title" in identity:
        confidence += weights.title
    confidence = round(min(1.0, confidence), 6)
    return ShapeResolution(
        family=family,
        confidence=confidence,
        via=via,
        identity=tuple(identity),
        fee_rows=fee_rows,
        intake_rows=intake_rows,
    )


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
    recurring_min: int | None = None,
) -> list[FamilyScore]:
    """Rank every configured family against one document. Deterministic order."""
    weights = rules.weights
    url_key = normalise(f"{urlsplit(url).path} {urlsplit(url).query}")
    anchor_key = normalise(anchor_text)
    text_key = normalise(text_sample)
    extension = extension_of(url)
    sample = text_sample or ""
    title_key = normalise(title_line(sample, rules)) if sample.strip() else ""
    heading_key = normalise(" ".join(heading_lines(sample)))
    recurring_floor = weights.recurring_min if recurring_min is None else recurring_min
    haystacks = {"url": url_key, "anchor": anchor_key, "text": text_key}

    scores: list[FamilyScore] = []
    for rule in rules.families:
        signals = {
            "url": _matches(rule.url_tokens, url_key),
            "anchor": _matches(rule.anchor_tokens, anchor_key),
            "extension": int(bool(extension) and extension in rule.extensions),
            "text": _matches(rule.text_tokens, text_key),
            "heading": _matches(rule.heading_tokens, heading_key),
            "recurring": _recurring_matches(rule.recurring_tokens, text_key, recurring_floor),
            "title": _matches(rule.heading_tokens, title_key),
        }
        # A negative-context phrase cancels the vote of the channel it appears
        # in; the veto itself stays visible in the signals for the run ledger.
        negatives = _rule_negatives(rule)
        for channel, tokens in negatives.items():
            hits = _matches(tokens, haystacks[channel]) if tokens else 0
            if hits:
                signals[f"{channel}_vetoed"] = hits
                signals[channel] = 0
        # A veto in the text sample cancels every text-derived channel: the
        # administrative-fee cover names "Fee Details" and "Annual fee" while
        # never being a student fee structure.
        if signals.get("text_vetoed"):
            for channel in ("heading", "recurring", "title"):
                if signals[channel]:
                    signals[channel] = 0
        score = (
            signals["url"] * weights.url
            + signals["anchor"] * weights.anchor
            + signals["extension"] * weights.extension
            + signals["text"] * weights.text
            + signals["heading"] * weights.text_heading
            + signals["recurring"] * weights.text_recurring
            + signals["title"] * weights.title
        )
        scores.append(
            FamilyScore(family=rule.family, score=round(min(1.0, score), 6), signals=signals)
        )
    return sorted(scores, key=lambda item: (-item.score, item.family))


def _annotate_winner(
    winner: FamilyScore, ranked: list[FamilyScore], signals: dict[str, int]
) -> None:
    """Record the runner-up and the annual-report placement hint on a winner."""
    runner_up = next((item for item in ranked if item.family != winner.family), None)
    if runner_up is not None:
        signals["runner_up_family_score"] = round(runner_up.score, 6)
    if winner.family == "accreditation":
        placement = next((item for item in ranked if item.family == "placement_report"), None)
        if placement is not None and placement.score > 0:
            signals["secondary_hint"] = 1
            signals["secondary_hint_family_score"] = round(placement.score, 6)


def detect_family(
    *,
    url: str = "",
    anchor_text: str | None = None,
    text_sample: str | None = None,
    rules: PlanRules = DEFAULT_RULES,
    min_score: float | None = None,
    margin: float | None = None,
    recurring_min: int | None = None,
    table_min_rows: int | None = None,
) -> tuple[str | None, float, dict[str, int], str | None]:
    """Return ``(family, confidence, signals, abstain_reason)``.

    Resolution order is explicit: registry-declared family is handled by the
    planner, so this function only answers the heuristic question. A prospectus
    cover resolves through its own shape chain; a cover that names exactly one
    family (``Annual Report``, ``Fee Structure``) decides the document outright;
    a tie or a thin margin is ``ambiguous_family``; no winner is
    ``family_undetected``.

    ``min_score`` / ``margin`` / ``recurring_min`` / ``table_min_rows`` let
    :mod:`fmd.plan.planner` apply the registry's per-phase thresholds without
    forking the rule set. They default to the rule weights so standalone callers
    (tests, fixtures) stay deterministic.
    """
    weights = rules.weights
    floor = weights.min_score if min_score is None else min_score
    separation = weights.margin if margin is None else margin
    ranked = [
        item
        for item in score_families(
            url=url,
            anchor_text=anchor_text,
            text_sample=text_sample,
            rules=rules,
            recurring_min=recurring_min,
        )
        if item.score > 0
    ]

    shape = resolve_prospectus_shape(
        url=url,
        anchor_text=anchor_text,
        text_sample=text_sample,
        rules=rules,
        table_min_rows=table_min_rows,
    )
    if shape is not None:
        signals: dict[str, int] = {
            "prospectus_identity": len(shape.identity),
            "prospectus_fee_table_rows": shape.fee_rows,
            "prospectus_intake_table_rows": shape.intake_rows,
        }
        if ranked:
            signals["best_score"] = round(ranked[0].score, 6)
        return shape.family, shape.confidence, signals, None

    if not ranked:
        return None, 0.0, {}, "family_undetected"

    best = ranked[0]
    runner_up = ranked[1].score if len(ranked) > 1 else 0.0
    if best.score < floor:
        return None, best.score, dict(best.signals), "family_undetected"

    # Cover-page identity: exactly one family is named by the title line. The
    # document said what it is; body-table noise cannot tie that decision.
    title_families = [item.family for item in ranked if item.signals.get("title", 0) > 0]
    if len(title_families) == 1 and best.score >= floor:
        signals = dict(best.signals)
        decisive = next(item for item in ranked if item.family == title_families[0])
        signals["title_decisive"] = 1
        _annotate_winner(decisive, ranked, signals)
        return decisive.family, decisive.score, signals, None

    if len(ranked) > 1 and best.score - runner_up < separation:
        signals = dict(best.signals)
        signals["runner_up_family_score"] = round(runner_up, 6)
        return None, round(best.score, 6), signals, "ambiguous_family"

    signals = dict(best.signals)
    _annotate_winner(best, ranked, signals)
    return best.family, best.score, signals, None
