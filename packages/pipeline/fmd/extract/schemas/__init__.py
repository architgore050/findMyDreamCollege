"""Document-family schema registry.

Families are data: adding one means adding a module here, a prompt directory and
a band entry — no changes to the pipeline or scoring code.
"""

from __future__ import annotations

from fmd.extract.schemas.accreditation import Accreditation
from fmd.extract.schemas.adjudication import AdjudicationVerdict
from fmd.extract.schemas.base import (
    ExtractedField,
    FamilyDocument,
    FieldSpec,
    JsonScalar,
    Period,
    UnitKind,
)
from fmd.extract.schemas.cost_of_living import CostOfLiving
from fmd.extract.schemas.curriculum import Curriculum
from fmd.extract.schemas.exam_notification import ExamNotification
from fmd.extract.schemas.fee_structure import FeeStructure
from fmd.extract.schemas.placement_report import PlacementReport
from fmd.extract.schemas.seat_matrix import SeatMatrix

FAMILIES: dict[str, type[FamilyDocument]] = {
    model.FAMILY: model
    for model in (
        PlacementReport,
        FeeStructure,
        SeatMatrix,
        Accreditation,
        Curriculum,
        ExamNotification,
        CostOfLiving,
    )
}

for _family_model in [*FAMILIES.values(), AdjudicationVerdict]:
    _family_model.validate_schema()

# Not a document family: only the T3 adjudication pass uses it.
ADJUDICATION = AdjudicationVerdict


class UnknownFamily(KeyError):
    pass


def schema_for(family: str) -> type[FamilyDocument]:
    try:
        return FAMILIES[family]
    except KeyError as exc:
        raise UnknownFamily(
            f"unknown extraction family {family!r}. Known families: " + ", ".join(sorted(FAMILIES))
        ) from exc


def predicate_specs() -> dict[str, FieldSpec]:
    """Every predicate this package can emit, for config/console cross-checks."""
    specs: dict[str, FieldSpec] = {}
    for name, model in FAMILIES.items():
        for field, spec in model.SPECS.items():
            existing = specs.get(spec.predicate)
            if existing is not None and existing != spec:
                raise ValueError(
                    f"predicate {spec.predicate!r} is declared inconsistently by "
                    f"{name}.{field} and another family"
                )
            specs[spec.predicate] = spec
    return specs


__all__ = [
    "ADJUDICATION",
    "FAMILIES",
    "Accreditation",
    "AdjudicationVerdict",
    "CostOfLiving",
    "Curriculum",
    "ExamNotification",
    "ExtractedField",
    "FamilyDocument",
    "FeeStructure",
    "FieldSpec",
    "JsonScalar",
    "Period",
    "PlacementReport",
    "SeatMatrix",
    "UnitKind",
    "UnknownFamily",
    "predicate_specs",
    "schema_for",
]
