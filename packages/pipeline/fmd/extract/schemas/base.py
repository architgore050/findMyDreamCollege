"""Shared guided-JSON field shape and family registry (brief §9).

Every extracted field is nullable and carries a verbatim quote plus a span into
the extraction text. Predicate names live here (not in prompt strings) so the
review console, scoring and the web app can agree on a vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

JsonScalar = str | int | float | bool

UnitKind = Literal["money", "count", "hours", "months", "years", "percent", "date", "text"]
Period = Literal["one_time", "per_year", "per_month", "per_semester"]


class ExtractedField(BaseModel):
    """One guided-JSON field. ``null`` means "the document did not say"."""

    model_config = ConfigDict(extra="forbid")

    value: JsonScalar | None = None
    unit: str | None = None
    quote: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


@dataclass(frozen=True)
class FieldSpec:
    """Structural facts about a field that validation needs.

    ``unit_required`` / ``unit_kind`` decide whether a missing unit is a
    rejection. ``is_rate`` requires a denominator in the claim qualifier.
    ``denominator_field`` names the sibling field that supplies it.
    """

    predicate: str
    unit_required: bool = False
    unit_kind: UnitKind | None = None
    is_rate: bool = False
    denominator_field: str | None = None
    period: Period | None = None
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

    def context_values(self) -> dict[str, JsonScalar]:
        values: dict[str, JsonScalar] = {}
        for name in self.CONTEXT_FIELDS:
            field = getattr(self, name)
            if field is not None and field.value is not None:
                values[name] = field.value
        return values
