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
        return cls(
            role_map=role_map,
            structured_url_patterns=tuple(params.get("structured_url_patterns") or ()),
            evidence_url_patterns=tuple(params.get("evidence_url_patterns") or ()),
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


__all__ = [
    "DEFAULT_ROLE_MAP",
    "ROLES",
    "ROLE_EVIDENCE",
    "ROLE_INVENTORY",
    "ROLE_NAVIGATION",
    "ContentClass",
    "Role",
    "RoleConfigError",
    "RoleRules",
    "classify_content",
    "is_extractable_role",
    "role_for",
]
