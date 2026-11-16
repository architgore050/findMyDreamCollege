"""Golden-set schema and loader (brief §19).

A golden row is a claim about what a *document* says. It is only allowed to
assert a value that a reader can check against a committed fixture or a cited
public source. The loader enforces that mechanically:

* ``status: verified`` + ``expectation: value`` requires a non-null value, a
  verbatim quote, and either a ``fixture`` whose text contains that quote or a
  ``source_url`` together with ``retrieved_at``;
* ``status: unverified`` forbids a value (``expected`` must be null). Unverified
  rows are counted separately and never enter accuracy metrics;
* ``expectation: abstain`` records that the source is silent about a field; the
  loader checks ``absent_terms`` really are absent from the fixture;
* ``expectation: reject`` records that a value must be rejected, with a reason.

Files whose name starts with ``_`` are documentation/templates and are not
loaded. There is deliberately no network access here: a cited source is recorded
as such and reported separately from a fixture-verified quote, never silently
treated as checked.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GOLDEN_DIR = REPO_ROOT / "data" / "golden"

Status = Literal["verified", "unverified"]
Expectation = Literal["value", "abstain", "reject"]
Traceability = Literal["fixture", "citation", "unverified"]


class GoldenError(ValueError):
    """The golden set is malformed or asserts something it cannot prove."""

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = list(problems)
        joined = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__("golden set is invalid. Fix the following and re-run:\n" + joined)


class GoldenValue(BaseModel):
    """One expected field on one golden entry."""

    model_config = ConfigDict(extra="forbid")

    value: Any = None
    unit: str | None = None
    quote: str | None = None
    status: Status
    expectation: Expectation = "value"
    printed_value: str | int | float | None = None
    printed_unit: str | None = None
    reason: str | None = None
    absent_terms: list[str] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def _status_rules(self) -> GoldenValue:
        if self.status == "unverified":
            if self.value is not None:
                raise ValueError(
                    "an unverified row must set value: null; promote it to verified only "
                    "when a committed fixture or cited source proves the value"
                )
            return self
        if self.expectation == "value":
            if self.value is None:
                raise ValueError("a verified value row needs a non-null value")
            if not (self.quote or "").strip():
                raise ValueError("a verified value row needs a verbatim quote")
            if self.unit is not None and not self.unit.strip():
                raise ValueError("unit must be null or a non-empty string")
        elif self.expectation == "abstain":
            if self.value is not None or self.quote is not None:
                raise ValueError("an abstain row must leave value and quote null")
        elif self.expectation == "reject" and not (self.reason or "").strip():
            raise ValueError("a reject row needs a reason")
        return self


class GoldenEntry(BaseModel):
    """One programme/document pair with its expected fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    institution: str
    programme: str
    state: str | None = None
    source_url: str | None = None
    retrieved_at: str | None = None
    fixture: str | None = None
    llm_fixture: str | None = None
    family: str | None = None
    notes: str | None = None
    expected: dict[str, GoldenValue]

    @property
    def verified_rows(self) -> dict[str, GoldenValue]:
        return {name: row for name, row in self.expected.items() if row.status == "verified"}

    @property
    def unverified_rows(self) -> dict[str, GoldenValue]:
        return {name: row for name, row in self.expected.items() if row.status == "unverified"}


@dataclass(frozen=True)
class RowTrace:
    entry_id: str
    field: str
    traceability: Traceability
    detail: str = ""


@dataclass
class GoldenSet:
    entries: list[GoldenEntry]
    traces: list[RowTrace] = field(default_factory=list)
    source_path: str = ""

    @property
    def verified_rows(self) -> int:
        return sum(len(entry.verified_rows) for entry in self.entries)

    @property
    def unverified_rows(self) -> int:
        return sum(len(entry.unverified_rows) for entry in self.entries)

    def rows_for(
        self, expectation: Expectation | None = None
    ) -> list[tuple[str, str, GoldenValue]]:
        rows: list[tuple[str, str, GoldenValue]] = []
        for entry in self.entries:
            for name, row in entry.expected.items():
                if expectation is None or row.expectation == expectation:
                    rows.append((entry.id, name, row))
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": len(self.entries),
            "verified_rows": self.verified_rows,
            "unverified_rows": self.unverified_rows,
            "by_expectation": {
                expectation: len(self.rows_for(expectation))
                for expectation in ("value", "abstain", "reject")
            },
            "by_traceability": {
                kind: sum(1 for trace in self.traces if trace.traceability == kind)
                for kind in ("fixture", "citation", "unverified")
            },
            "ids": [entry.id for entry in self.entries],
        }


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def load_golden(path: str | Path | None = None, *, root: Path | None = None) -> GoldenSet:
    """Load and validate every ``*.yaml`` under ``path/programmes`` (or ``path``)."""

    directory = Path(path) if path is not None else DEFAULT_GOLDEN_DIR
    base = Path(root) if root is not None else REPO_ROOT
    if not directory.exists() and not directory.is_absolute():
        # Convenience for `cd packages/pipeline && fmd eval --golden data/golden`.
        candidate = base / directory
        if candidate.exists():
            directory = candidate
    if not directory.exists():
        raise GoldenError([f"golden directory does not exist: {directory}"])
    files = sorted(directory.rglob("*.yaml")) + sorted(directory.rglob("*.yml"))
    files = [file for file in files if not file.name.startswith("_")]
    if not files:
        raise GoldenError([f"no golden YAML files found under {directory}"])

    problems: list[str] = []
    entries: list[GoldenEntry] = []
    traces: list[RowTrace] = []
    seen: dict[str, Path] = {}

    for file in files:
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            problems.append(f"{file}: cannot read YAML: {exc}")
            continue
        documents = raw if isinstance(raw, list) else [raw]
        for document in documents:
            if not isinstance(document, Mapping):
                problems.append(f"{file}: every golden entry must be a mapping")
                continue
            try:
                entry = GoldenEntry.model_validate(document)
            except Exception as exc:  # pydantic.ValidationError and custom validators
                problems.append(f"{file}: {exc}")
                continue
            if entry.id in seen:
                problems.append(
                    f"{file}: duplicate golden id {entry.id!r} (also in {seen[entry.id]})"
                )
                continue
            seen[entry.id] = file
            fixture_text = None
            if entry.fixture:
                fixture_path = base / entry.fixture
                if not fixture_path.exists():
                    problems.append(f"{file}: fixture {entry.fixture!r} does not exist")
                else:
                    fixture_text = _read_text(fixture_path)
                    if fixture_text is None:
                        problems.append(
                            f"{file}: fixture {entry.fixture!r} is not UTF-8 text; "
                            "only text fixtures can verify quotes offline"
                        )
            for name, row in entry.expected.items():
                problems.extend(_check_row(entry, name, row, fixture_text))
                traces.append(_trace(entry, name, row))
            entries.append(entry)

    if problems:
        raise GoldenError(problems)
    return GoldenSet(entries=entries, traces=traces, source_path=str(directory))


def _check_row(
    entry: GoldenEntry,
    name: str,
    row: GoldenValue,
    fixture_text: str | None,
) -> list[str]:
    prefix = f"{entry.id}.{name}"
    problems: list[str] = []
    if row.status == "unverified":
        return problems
    if row.expectation == "abstain":
        if fixture_text is not None:
            for term in row.absent_terms:
                if term in fixture_text:
                    problems.append(
                        f"{prefix}: abstention claims {term!r} is absent, but the fixture "
                        "contains it"
                    )
        return problems
    if row.expectation == "reject":
        return problems

    if entry.fixture:
        if fixture_text is None:
            problems.append(f"{prefix}: verified row has no readable fixture text")
        elif row.quote not in fixture_text:
            problems.append(f"{prefix}: quote is not verbatim in {entry.fixture}: {row.quote!r}")
    if not entry.fixture:
        if not entry.source_url:
            problems.append(
                f"{prefix}: verified row needs either a committed fixture or a source_url"
            )
        if not entry.retrieved_at:
            problems.append(f"{prefix}: a cited public source needs retrieved_at (YYYY-MM-DD)")
    return problems


def _trace(entry: GoldenEntry, name: str, row: GoldenValue) -> RowTrace:
    if row.status == "unverified":
        return RowTrace(entry.id, name, "unverified", "excluded from accuracy metrics")
    if entry.fixture:
        return RowTrace(entry.id, name, "fixture", entry.fixture)
    return RowTrace(entry.id, name, "citation", entry.source_url or "")


__all__ = [
    "DEFAULT_GOLDEN_DIR",
    "REPO_ROOT",
    "GoldenEntry",
    "GoldenError",
    "GoldenSet",
    "GoldenValue",
    "RowTrace",
    "Traceability",
    "load_golden",
]
