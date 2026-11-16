"""City cost-of-living basket family (brief §13; never summed with hostel)."""

from __future__ import annotations

from fmd.extract.schemas.base import ExtractedField, FamilyDocument, FieldSpec


class CostOfLiving(FamilyDocument):
    FAMILY = "cost_of_living"
    SPECS = {
        "city": FieldSpec("cost.city", unit_kind="text", context=True),
        "basket_period": FieldSpec("cost.basket_period", unit_kind="text", context=True),
        "rent_monthly": FieldSpec(
            "cost.rent_monthly", unit_required=True, unit_kind="money", period="per_month"
        ),
        "food_monthly": FieldSpec(
            "cost.food_monthly", unit_required=True, unit_kind="money", period="per_month"
        ),
        "transport_monthly": FieldSpec(
            "cost.transport_monthly", unit_required=True, unit_kind="money", period="per_month"
        ),
        "utilities_monthly": FieldSpec(
            "cost.utilities_monthly", unit_required=True, unit_kind="money", period="per_month"
        ),
    }

    city: ExtractedField | None = None
    basket_period: ExtractedField | None = None
    rent_monthly: ExtractedField | None = None
    food_monthly: ExtractedField | None = None
    transport_monthly: ExtractedField | None = None
    utilities_monthly: ExtractedField | None = None
