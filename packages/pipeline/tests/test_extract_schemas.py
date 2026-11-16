"""Quote-first guided JSON: the model never emits values, units or offsets."""

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
    "Never compute, convert or infer",
    "No scores",
)


def test_every_family_has_a_quote_first_v2_prompt_and_keeps_v1() -> None:
    for family in FAMILIES:
        versions = prompt_versions(family)
        assert {"v1", "v2"}.issubset(set(versions)), f"{family} must keep v1 and add v2"
        prompt = load_prompt(family)
        assert prompt.version == "v2"
        for marker in ANTI_INFLATION_MARKERS:
            assert marker in prompt.text, f"{family}/{prompt.version} is missing {marker!r}"
        assert "<<<" not in prompt.text
        assert prompt.path.is_file()


def test_v1_prompts_still_load() -> None:
    prompt = load_prompt("placement_report", "v1")
    assert prompt.version == "v1"
    assert "char_start" in prompt.text  # retained for replay of historical runs
    assert prompt.ref == "placement_report@v1"


def test_prompts_are_files_not_inline_strings() -> None:
    prompt = load_prompt("placement_report")
    assert prompt.path.parent.name == "placement_report"
    assert prompt.path.parent.parent == PROMPTS_DIR
    assert prompt.sha256 == load_prompt("placement_report", "v2").sha256


def test_unknown_prompt_version_fails_with_available_versions() -> None:
    with pytest.raises(LookupError) as excinfo:
        load_prompt("placement_report", "v99")
    assert "v1" in str(excinfo.value)
    assert "v2" in str(excinfo.value)


def test_every_schema_field_is_nullable_and_carries_only_the_quote_shape() -> None:
    for family, model in FAMILIES.items():
        document = model()
        assert set(ExtractedField.model_fields) == {
            "quote",
            "confidence",
            "reason_if_null",
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


def test_guided_json_schema_forbids_model_supplied_offsets_and_values() -> None:
    field_schema = ExtractedField.model_json_schema()
    assert set(field_schema["properties"]) == {"quote", "confidence", "reason_if_null"}
    assert "char_start" not in field_schema["properties"]
    assert "value" not in field_schema["properties"]
    assert field_schema["additionalProperties"] is False


def test_extra_fields_are_rejected_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        ExtractedField(value=1, quote="120")
    with pytest.raises(ValidationError):
        FAMILIES["fee_structure"](hallucinated_field=ExtractedField(quote="120"))


def test_predicate_vocabulary_is_registry_compatible() -> None:
    specs = predicate_specs()
    assert specs["placement.eligible_count"].unit_kind == "count"
    assert specs["salary.median"].period == "per_year"
    assert specs["salary.median"].period_required is True
    assert specs["fee.tuition"].unit_kind == "money"
    assert specs["placement.rate"].is_rate is True
    assert specs["placement.rate"].denominator_field == "eligible_count"
    assert specs["accreditation.naac_grade"].unit_kind == "grade"
    assert specs["exam.requires_travel"].unit_kind == "boolean"
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
