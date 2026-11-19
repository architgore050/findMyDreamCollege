"""Document *role* classification (migration 0004, brief §7, §9).

The first live crawl stored inventory payloads, navigation pages and
administrative PDFs without distinction. The planner then abstained with
``family_undetected`` on all of them, which reads as an extraction failure when
it is really "we fetched the wrong kind of page". A role answers that before
extraction is asked anything:

``inventory``   a structured payload that *describes* institutions (UGC JSON,
                AISHE workbook). It is ingested into entities, never extracted.
``navigation``  a listing or nav page that exists to be followed. Not evidence.
``evidence``    a PDF/report page (placement, annual report, AQAR/SSR, NIRF,
                audit report, prospectus, fee structure) — the only role the
                planner and extractor act on.

Everything here is data. The content class is decided from the URL/extension
and the response content type; the mapping from content class to role is read
from the registry entry's ``params.document_role`` (with a conservative default).
No host, institution or source id is hardcoded (invariant 7).

An explicit family — declared by the registry or carried by discovery — always
wins: if a document was discovered *as* a seat matrix, it is evidence whatever
its URL looks like.

W5 family coverage adds the weakest classification tier here:
:func:`family_hints_from_structure` maps the *file name's shape*
(``Annual-Report-2024.pdf``, ``prospectus-2025.pdf``, ``SeatMatrix.xlsx`` …) to
a family hint. Structure only — no text is read, no institution is named — and
administrative shapes (tender, convocation, exam schedule, news, office order)
veto every hint. The planner records the hints on every decision and uses them
only when registry, text and URL token evidence are all silent, so they can
never override a declared family and never outrank the document's own text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from fmd.discover.base import params_from_source

Role = Literal["inventory", "navigation", "evidence"]
ContentClass = Literal["document", "structured", "page"]

ROLE_INVENTORY = "inventory"
ROLE_NAVIGATION = "navigation"
ROLE_EVIDENCE = "evidence"
ROLES: tuple[str, ...] = (ROLE_INVENTORY, ROLE_NAVIGATION, ROLE_EVIDENCE)

#: Conservative fallback: a file we can read as a report is evidence; a machine
#: payload describes entities; everything else is a page to follow, not cite.
DEFAULT_ROLE_MAP: dict[str, str] = {
    "document": ROLE_EVIDENCE,
    "structured": ROLE_INVENTORY,
    "page": ROLE_NAVIGATION,
}

#: Document *shapes*, not institution names. A deployment can extend these.
DEFAULT_DOCUMENT_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
    ".ppt",
    ".pptx",
)
DEFAULT_STRUCTURED_EXTENSIONS: tuple[str, ...] = (
    ".json",
    ".xml",
    ".csv",
    ".xls",
    ".xlsx",
    ".ods",
)
DEFAULT_STRUCTURED_CONTENT_TYPES: tuple[str, ...] = (
    "json",
    "xml",
    "csv",
    "spreadsheet",
    "vnd.ms-excel",
)
DEFAULT_DOCUMENT_CONTENT_TYPES: tuple[str, ...] = (
    "application/pdf",
    "application/msword",
    "wordprocessingml",
    "presentationml",
    "application/rtf",
)

#: Filename-shape → family hints (W5 family coverage). Structure only: the
#: patterns run against the URL's file name, never against document text, so a
#: hint is URL-tier evidence — weaker than text, never overriding a declared
#: family. Administrative shapes veto every hint: a tender notice, a
#: convocation circular or an exam date sheet must never gain a family from its
#: file name.
DEFAULT_FAMILY_HINT_PATTERNS: dict[str, tuple[str, ...]] = {
    "accreditation": (
        r"annual[-_ ]?report",
        r"self[-_ ]?study",
        r"(?<![a-z])aqar",
        r"(?<![a-z])ssr",
        r"(?<![a-z])nirf",
        r"(?<![a-z])iqac",
    ),
    "placement_report": (r"placement",),
    # The administrative-fee case ("Fee for recognition … Research Centre")
    # carries "fee" but is not a fee structure; the fee family carries its own
    # negative patterns below.
    "fee_structure": (
        r"fee[-_ ]?structure",
        r"fee[-_ ]?schedule",
        r"fee[-_ ]?circular",
        r"fee[-_ ]?notification",
        r"fee[-_ ]?order",
        r"fees?[\s_-]*&[\s_-]*other",
    ),
    "seat_matrix": (
        r"seat[-_ ]?matrix",
        r"sanctioned[-_ ]?intake",
        r"intake[-_ ]?(capacity|statement|position|details)",
    ),
    # A prospectus is a multi-family bundle (fees, intake, curriculum); its
    # tables decide (fmd.plan.rules.resolve_prospectus_shape) once text is
    # read. Until then the shape's default branch is curriculum.
    "curriculum": (r"prospectus", r"admission[-_ ]?brochure", r"syllabus", r"curriculum"),
    # Deliberately none for exam_notification: a schedule or an admit card is
    # not evidence, and Phase A deferral already handles registrations.
}

#: Family-specific hint vetoes (same phrases as fmd.plan.rules).
DEFAULT_FAMILY_HINT_NEGATIVES: dict[str, tuple[str, ...]] = {
    "fee_structure": (
        r"fee[-_ ]?for[-_ ]?recognition",
        r"recognition[-_ ]?of[-_ ]?institution",
        r"research[-_ ]?cent(re|er)",
        r"affiliation[-_ ]?fee",
        r"inspection[-_ ]?fee",
    )
}

#: Administrative document shapes: no family hint may fire for these, whatever
#: else the file name contains. Word-boundary anchored so ``newsletter`` or
#: ``Seventh`` cannot trip them.
GLOBAL_HINT_NEGATIVES: tuple[str, ...] = (
    r"\btender",
    r"\brecruitment",
    r"\bconvocation",
    r"admit[-_ ]?card",
    r"hall[-_ ]?ticket",
    r"date[-_ ]?sheet",
    r"exam[-_ ]?schedule",
    r"examination[-_ ]?schedule",
    r"\btimetable",
    r"office[-_ ]?order",
    r"\bnews\b",
    r"\bevents?\b",
    r"\bgallery\b",
    r"press[-_ ]?release",
)


class RoleConfigError(ValueError):
    """The registry declares a role the schema does not allow."""


def _suffix(url: str) -> str:
    path = urlsplit(url).path.lower()
    tail = path.rsplit("/", 1)[-1]
    return "." + tail.rsplit(".", 1)[-1] if "." in tail else ""


def _compiled(values: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(value) for value in values if value]


def _matches(patterns: list[re.Pattern[str]], value: str) -> bool:
    return any(pattern.search(value) for pattern in patterns)


@dataclass(frozen=True)
class RoleRules:
    """One source's role policy. Missing keys fall back to :data:`DEFAULT_ROLE_MAP`."""

    role_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLE_MAP))
    document_extensions: tuple[str, ...] = DEFAULT_DOCUMENT_EXTENSIONS
    structured_extensions: tuple[str, ...] = DEFAULT_STRUCTURED_EXTENSIONS
    structured_content_types: tuple[str, ...] = DEFAULT_STRUCTURED_CONTENT_TYPES
    document_content_types: tuple[str, ...] = DEFAULT_DOCUMENT_CONTENT_TYPES
    structured_url_patterns: tuple[str, ...] = ()
    evidence_url_patterns: tuple[str, ...] = ()
    #: Filename-shape family hints (W5). Document structure only — never text.
    family_hints: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_FAMILY_HINT_PATTERNS)
    )
    family_hint_negatives: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_FAMILY_HINT_NEGATIVES)
    )
    global_hint_negatives: tuple[str, ...] = GLOBAL_HINT_NEGATIVES

    @classmethod
    def from_source(cls, source: object | None) -> RoleRules:
        params = params_from_source(source) if source is not None else {}
        raw_map = params.get("document_role") or {}
        role_map = dict(DEFAULT_ROLE_MAP)
        if isinstance(raw_map, Mapping):
            for key, value in raw_map.items():
                role = str(value).strip().lower()
                if role not in ROLES:
                    raise RoleConfigError(
                        f"params.document_role.{key} = {value!r} is not one of {ROLES}"
                    )
                role_map[str(key)] = role
        family_hints = dict(DEFAULT_FAMILY_HINT_PATTERNS)
        raw_hints = params.get("family_hints") or {}
        if isinstance(raw_hints, Mapping):
            for family, patterns in raw_hints.items():
                family_hints[str(family)] = tuple(str(value) for value in patterns)
        return cls(
            role_map=role_map,
            structured_url_patterns=tuple(params.get("structured_url_patterns") or ()),
            evidence_url_patterns=tuple(params.get("evidence_url_patterns") or ()),
            family_hints=family_hints,
        )

    @property
    def content_classes(self) -> tuple[str, ...]:
        return ("document", "structured", "page")


def classify_content(
    *,
    url: str,
    content_type: str | None = None,
    rules: RoleRules | None = None,
) -> ContentClass:
    """Return ``document`` | ``structured`` | ``page`` from observable bytes.

    The content type is authoritative when present; the extension is the
    fallback (a mislabelled UGC XHR is still a structured payload). Anything
    unrecognised is a page: a page is safe to follow, never safe to cite.
    """
    rules = rules or RoleRules()
    ctype = (content_type or "").lower()
    suffix = _suffix(url)

    if _matches(_compiled(rules.structured_url_patterns), url):
        return "structured"
    if any(token in ctype for token in rules.structured_content_types):
        return "structured"
    if suffix in rules.structured_extensions:
        return "structured"
    if any(token in ctype for token in rules.document_content_types):
        return "document"
    if suffix in rules.document_extensions:
        return "document"
    return "page"


def role_for(
    *,
    url: str,
    content_type: str | None = None,
    source: object | None = None,
    family: str | None = None,
    rules: RoleRules | None = None,
) -> str:
    """Classify one fetched document. Deterministic; never guesses a family."""
    rules = rules or RoleRules.from_source(source)
    # Discovery/registry already said what this document is: that outranks any
    # URL heuristic, and an extractable family is evidence by definition.
    if family:
        return ROLE_EVIDENCE
    if _matches(_compiled(rules.evidence_url_patterns), url):
        return ROLE_EVIDENCE
    content = classify_content(url=url, content_type=content_type, rules=rules)
    return rules.role_map.get(content, DEFAULT_ROLE_MAP[content])


def is_extractable_role(role: str | None) -> bool:
    """A ``None`` role is legacy evidence (migration default), never a skip."""
    return role in (None, ROLE_EVIDENCE)


def _filename_of(url: str) -> str:
    return urlsplit(url).path.rsplit("/", 1)[-1] if url else ""


def _any_pattern(
    patterns: tuple[str, ...] | Mapping[str, tuple[str, ...]],
    filename: str,
    family: str | None = None,
) -> bool:
    if family is not None:
        if not isinstance(patterns, Mapping):
            return False
        patterns = patterns.get(family, ())
    for pattern in patterns:
        try:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        except re.error:
            # A bad pattern is a registry typo: it must never classify, and it
            # must never crash a planning run. Skip it.
            continue
    return False


def family_hints_from_structure(
    *,
    url: str,
    content_type: str | None = None,
    source: object | None = None,
    rules: RoleRules | None = None,
) -> tuple[str, ...]:
    """Families hinted by the document's *structure*: its file name only.

    This is URL-tier evidence, deliberately weaker than any text signal:

    * it never fires for a non-document (an inventory payload or a nav page is
      not an annual report because a directory is named ``reports``);
    * administrative shapes (tender, convocation, exam schedule, office order,
      news/event) veto every hint;
    * family-specific vetoes apply on top (the known administrative-fee case
      carries ``fee`` in its file name but is not a fee structure);
    * the result is in declaration order and duplicates are impossible.

    The planner records hints on every decision and uses them only when
    registry, text and URL token evidence are all silent.
    """
    rules = rules or RoleRules.from_source(source)
    if classify_content(url=url, content_type=content_type, rules=rules) != "document":
        return ()
    filename = _filename_of(url)
    if not filename:
        return ()
    lowered = filename.lower()
    if _any_pattern(rules.global_hint_negatives, lowered):
        return ()
    hints: list[str] = []
    for family in rules.family_hints:
        if not _any_pattern(rules.family_hints, lowered, family):
            continue
        if _any_pattern(rules.family_hint_negatives, lowered, family):
            continue
        hints.append(family)
    return tuple(hints)


__all__ = [
    "DEFAULT_FAMILY_HINT_NEGATIVES",
    "DEFAULT_FAMILY_HINT_PATTERNS",
    "DEFAULT_ROLE_MAP",
    "GLOBAL_HINT_NEGATIVES",
    "ROLES",
    "ROLE_EVIDENCE",
    "ROLE_INVENTORY",
    "ROLE_NAVIGATION",
    "ContentClass",
    "Role",
    "RoleConfigError",
    "RoleRules",
    "classify_content",
    "family_hints_from_structure",
    "is_extractable_role",
    "role_for",
]
