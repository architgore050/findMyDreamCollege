"""Versioned verifier prompts: ``fmd/verify/prompts/<name>/<version>.md``.

Kept separate from the extractor's prompt registry on purpose: the second pass
is a different task with a different prompt contract, and its version is
recorded in ``claims.verification_notes.prompt_version``.
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
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _versions(name: str) -> list[str]:
    directory = PROMPTS_DIR / name
    if not directory.is_dir():
        return []
    versions = [
        match.group(1) for path in directory.iterdir() if (match := _VERSION_RE.match(path.name))
    ]
    return sorted(versions, key=lambda version: int(version[1:]))


def prompt_versions(name: str) -> list[str]:
    """Every committed version of one verifier prompt, oldest first (v1..vN)."""
    return _versions(name)


def load_prompt(name: str, version: str | None = None) -> Prompt:
    versions = _versions(name)
    if not versions:
        raise PromptNotFound(f"no verifier prompt for {name!r} under {PROMPTS_DIR}")
    chosen = version or versions[-1]
    if chosen not in versions:
        raise PromptNotFound(
            f"verifier prompt {name!r} has no version {chosen!r}; available: "
            + ", ".join(versions)
        )
    path = PROMPTS_DIR / name / f"{chosen}.md"
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PromptNotFound(f"verifier prompt {path} is empty")
    return Prompt(name=name, version=chosen, text=text, path=path)
