"""Versioned prompt registry (brief §9).

Prompts are files, never inline strings: ``prompts/<name>/<version>.md``. The
router hashes the exact system+user text into ``llm_calls.prompt_sha256``, so a
prompt edit is auditable even before the version is bumped.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
_VERSION_RE = re.compile(r"^(v\d+)\.md$")


class PromptNotFound(LookupError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str
    path: Path

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def ref(self) -> str:
        """Stable identifier persisted as ``extractions.prompt_version``."""
        return f"{self.name}@{self.version}"


def prompt_versions(name: str) -> list[str]:
    directory = PROMPTS_DIR / name
    if not directory.is_dir():
        return []
    versions = [
        match.group(1) for path in directory.iterdir() if (match := _VERSION_RE.match(path.name))
    ]
    return sorted(versions, key=lambda version: int(version[1:]))


def load_prompt(name: str, version: str | None = None) -> Prompt:
    versions = prompt_versions(name)
    if not versions:
        raise PromptNotFound(
            f"no prompts for {name!r} under {PROMPTS_DIR}. Expected "
            f"{PROMPTS_DIR / name}/<version>.md"
        )
    chosen = version or versions[-1]
    if chosen not in versions:
        raise PromptNotFound(
            f"prompt {name!r} has no version {chosen!r}; available: {', '.join(versions)}"
        )
    path = PROMPTS_DIR / name / f"{chosen}.md"
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PromptNotFound(f"prompt file {path} is empty")
    return Prompt(name=name, version=chosen, text=text, path=path)


def system_prompt(name: str, version: str | None = None) -> str:
    return load_prompt(name, version).text


def user_message(text: str, *, max_chars: int | None = None) -> str:
    """The user turn: verbatim text, delimited so the quote is unambiguous.

    The model returns quotes, not offsets: offsets are computed against the
    un-delimited extraction text by :mod:`fmd.extract.validate`, so the
    delimiter is never part of a span.
    """
    body = text if max_chars is None else text[:max_chars]
    return (
        "DOCUMENT TEXT (verbatim; quote from this block exactly, without the "
        "delimiters):\n"
        "<<<\n"
        f"{body}\n"
        ">>>"
    )
