"""Shared guided-JSON field shape and family registry (brief §9).

Extraction is **quote-first**. The model emits only what it can copy: a verbatim
``quote``, a ``confidence`` and (when the field is null) a short
``reason_if_null``. It never emits numbers, units or character offsets: models
reliably identify the relevant text but reliably fail to reproduce exact span
boundaries, so offsets and values are computed deterministically in
:mod:`fmd.extract.validate` from the resolved span. A published number therefore
always has a source span, and the span is never a model guess.

Predicate names live here (not in prompt strings) so the review console, scoring
and the web app can agree on a vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

JsonScalar = str | int | float | bool

UnitKind = Literal[
    "money",
    "count",
    "hours",
    "months",
    "years",
    "percent",
    "date",
    "text",
    "grade",
    "boolean",
]
Period = Literal["one_time", "per_year", "per_month", "per_semester"]


class ExtractedField(BaseModel):
    """One guided-JSON field. ``null`` means "the document did not say".

    ``quote`` is the evidence and the *only* place a value may come from;
    ``char_start``/``char_end`` are deliberately absent so the model cannot
    fabricate a boundary. ``reason_if_null`` is review-console metadata and is
    never treated as evidence.
    """

    model_config = ConfigDict(extra="forbid")

    quote: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason_if_null: str | None = None


@dataclass(frozen=True)
class FieldSpec:
    """Structural facts about a field that validation needs.

    ``unit_kind`` selects the deterministic parser that derives the value and
    unit from the resolved span. ``is_rate`` requires a denominator in the claim
    qualifier; ``denominator_field`` names the sibling field that supplies it.
    ``period_required`` marks a field whose printed period must come from the
    span itself (a salary without "per month"/"per annum" is UNKNOWN).
    ``value_pattern`` optionally narrows a free-text span to the meaningful
    phrase (still a verbatim substring of the resolved span).
    """

    predicate: str
    unit_required: bool = False
    unit_kind: UnitKind | None = None
    is_rate: bool = False
    denominator_field: str | None = None
    period: Period | None = None
    period_required: bool = False
    value_pattern: str | None = None
    context: bool = False


class FamilyDocument(BaseModel):
    """Base class for a document family's guided-JSON response."""

    model_config = ConfigDict(extra="forbid")

    FAMILY: ClassVar[str] = ""
    SPECS: ClassVar[dict[str, FieldSpec]] = {}
    CONTEXT_FIELDS: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore[arg-type]
        if not cls.FAMILY:
            raise ValueError(f"{cls.__name__} must declare FAMILY")
        cls.CONTEXT_FIELDS = tuple(name for name, spec in cls.SPECS.items() if spec.context)

    @classmethod
    def validate_schema(cls) -> None:
        """Cheap integrity check; call once after the model class is built."""
        declared = set(cls.model_fields)
        specified = set(cls.SPECS)
        if declared != specified:
            raise ValueError(
                f"{cls.FAMILY}: SPECS and model fields disagree: "
                f"missing specs={sorted(declared - specified)}, "
                f"stale specs={sorted(specified - declared)}"
            )
        for name, spec in cls.SPECS.items():
            if spec.denominator_field is not None and spec.denominator_field not in cls.SPECS:
                raise ValueError(
                    f"{cls.FAMILY}.{name}: unknown denominator_field {spec.denominator_field!r}"
                )

    def fields(self) -> Iterator[tuple[str, FieldSpec, ExtractedField | None]]:
        for name, spec in self.SPECS.items():
            yield name, spec, getattr(self, name)
