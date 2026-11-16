from __future__ import annotations

import pytest
from pydantic import ValidationError

from fmd.extract.prompts import PROMPTS_DIR, load_prompt, prompt_versions
from fmd.extract.schemas import (
    FAMILIES,
    ExtractedField,
    FamilyDocument,
    UnknownFamily,
    predicate_specs,
    schema_for,
)

ANTI_INFLATION_MARKERS = (
    "Quote verbatim",
    "Never guess",
    "No scores",
)


def test_every_family_has_a_versioned_prompt_with_anti_inflation_rules() -> None:
    for family in FAMILIES:
        assert prompt_versions(family), f"{family} has no prompt version"
        prompt = load_prompt(family)
        assert prompt.version.startswith("v")
        for marker in ANTI_INFLATION_MARKERS:
            assert marker in prompt.text, f"{family}/{prompt.version} is missing {marker!r}"
        assert "<<<" not in prompt.text
        assert prompt.path.is_file()


def test_prompts_are_files_not_inline_strings() -> None:
    prompt = load_prompt("placement_report")
    assert prompt.path.parent.name == "placement_report"
    assert prompt.path.parent.parent == PROMPTS_DIR
    assert prompt.sha256 == load_prompt("placement_report", "v1").sha256


def test_unknown_prompt_version_fails_with_available_versions() -> None:
    with pytest.raises(LookupError) as excinfo:
        load_prompt("placement_report", "v99")
    assert "v1" in str(excinfo.value)


def test_every_schema_field_is_nullable_and_carries_the_full_shape() -> None:
    for family, model in FAMILIES.items():
        document = model()
        annotation_names = set(ExtractedField.model_fields)
        assert annotation_names == {
            "value",
            "unit",
            "quote",
            "char_start",
            "char_end",
            "confidence",
        }
        for name, _spec, extracted in document.fields():
            assert extracted is None, f"{family}.{name} must default to null"
            json_schema = model.model_json_schema()["properties"][name]
            assert any("null" in option.get("type", "") for option in json_schema.get("anyOf", []))


def test_guided_json_schema_does_not_force_a_value() -> None:
    schema = FAMILIES["fee_structure"].model_json_schema()
    tuition = schema["properties"]["tuition_amount"]
    assert tuition["default"] is None
    assert any(option.get("type") == "null" for option in tuition["anyOf"])


def test_extra_fields_are_rejected_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        FAMILIES["fee_structure"](hallucinated_field=ExtractedField(value=1))


def test_predicate_vocabulary_is_registry_compatible() -> None:
    specs = predicate_specs()
    assert specs["placement.eligible_count"].unit_kind == "count"
    assert specs["salary.median"].period == "per_year"
    assert specs["fee.tuition"].unit_kind == "money"
    assert specs["placement.rate"].is_rate is True
    assert specs["placement.rate"].denominator_field == "eligible_count"
    assert all("." in predicate for predicate in specs)


def test_schema_for_reports_known_families() -> None:
    assert schema_for("placement_report") is FAMILIES["placement_report"]
    with pytest.raises(UnknownFamily) as excinfo:
        schema_for("no_such_family")
    assert "placement_report" in str(excinfo.value)


def test_family_models_declare_their_specs_before_use() -> None:
    for family, model in FAMILIES.items():
        assert issubclass(model, FamilyDocument)
        assert FAMILIES[model.FAMILY] is model
        assert family == model.FAMILY
        model.validate_schema()
