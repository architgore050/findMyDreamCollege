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


def test_every_family_declares_a_real_prompt_file_history_with_labels() -> None:
    """Prompt completeness, enforced generally.

    For every family: the declared (latest) prompt_version is a real file at
    prompts/<family>/<version>.md, every earlier version still exists, the
    history is contiguous v1..vN with no phantom gaps, no file is empty, and
    the version label in the body matches the filename.
    """
    for family in FAMILIES:
        versions = prompt_versions(family)
        assert versions, f"{family} must ship at least one prompt file"
        contiguous = [f"v{index}" for index in range(1, len(versions) + 1)]
        assert versions == contiguous, (
            f"{family} prompt history must be contiguous v1..vN with every "
            f"earlier version kept; got {versions}"
        )
        for version in versions:
            prompt = load_prompt(family, version)
            assert prompt.path == PROMPTS_DIR / family / f"{version}.md"
            assert prompt.path.is_file(), f"{family}/{version}.md must be a real file"
            assert prompt.text.strip(), f"{family}/{version}.md must not be empty"
            assert prompt.version in prompt.text, (
                f"{family}/{version}.md must carry its version label in the body"
            )
        latest = load_prompt(family)
        assert latest.version == versions[-1]
        assert latest.path == PROMPTS_DIR / family / f"{latest.version}.md"
        assert latest.path.is_file()


def test_every_family_keeps_v1_v2_and_tabular_families_add_their_next_version() -> None:
    # fee_structure/seat_matrix were at v3, so wave 2 ships v4 for them;
    # placement_report was at v2, so its next version is v3. Every other
    # family stays at its current latest, and old files are kept everywhere.
    tabular_v4 = {"seat_matrix", "fee_structure"}
    for family in FAMILIES:
        versions = prompt_versions(family)
        assert {"v1", "v2"}.issubset(set(versions)), f"{family} must keep v1 and v2"
        prompt = load_prompt(family)
        if family in tabular_v4:
            expected = "v4"
        elif family == "placement_report":
            expected = "v3"
        else:
            expected = "v2"
        assert prompt.version == expected, f"{family} latest prompt must be {expected}"
        for marker in ANTI_INFLATION_MARKERS:
            assert marker in prompt.text, f"{family}/{prompt.version} is missing {marker!r}"
        assert "<<<" not in prompt.text
        assert prompt.path.is_file()
    for family in tabular_v4:
        prompt = load_prompt(family, "v3")
        assert "context" in prompt.text
        assert "together they must identify exactly one location" in prompt.text
        wave_prompt = load_prompt(family, "v4")
        assert "context" in wave_prompt.text
        assert "together they must identify exactly one location" in wave_prompt.text


WAVE2_CONTRACT_MARKERS = (
    "Return ONLY `quote`, `context`, `confidence`, `reason_if_null`",
    "the quote must contain the value",
    "the context must contain the row identifier",
    "together they must identify exactly one location",
    "printed currency AND the printed period",
    "year / semester / month",
    "must state its denominator, else it is null",
    "null is a correct answer",
    "Never compute, convert or infer",
    "## Worked null example",
    "## Worked table-row example",
)


def test_wave2_prompts_state_the_full_contract() -> None:
    """The wave-2 prompts say, explicitly, what the validator enforces.

    Matching is whitespace-normalised: a prompt is prose and may wrap a
    required statement across lines.
    """
    for family, version in (
        ("fee_structure", "v4"),
        ("seat_matrix", "v4"),
        ("placement_report", "v3"),
    ):
        prompt = load_prompt(family, version)
        flattened = " ".join(prompt.text.split()).lower()
        for marker in WAVE2_CONTRACT_MARKERS:
            assert " ".join(marker.split()).lower() in flattened, (
                f"{family}@{version} is missing the contract statement {marker!r}"
            )


def test_v2_prompts_still_load_for_the_row_scoped_families() -> None:
    prompt = load_prompt("seat_matrix", "v2")
    assert prompt.ref == "seat_matrix@v2"
    assert "context" not in prompt.text  # v2 predates the row-scoped contract


def test_v1_prompts_still_load() -> None:
    prompt = load_prompt("placement_report", "v1")
    assert prompt.version == "v1"
    assert "char_start" in prompt.text  # retained for replay of historical runs
    assert prompt.ref == "placement_report@v1"


def test_prompts_are_files_not_inline_strings() -> None:
    prompt = load_prompt("placement_report")
    assert prompt.path.parent.name == "placement_report"
    assert prompt.path.parent.parent == PROMPTS_DIR
    assert prompt.sha256 == load_prompt("placement_report", "v3").sha256


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
            "context",
            "confidence",
            "reason_if_null",
        }
        for name, _spec, extracted in document.fields():
            assert extracted is None, f"{family}.{name} must default to null"
            json_schema = model.model_json_schema()["properties"][name]
            assert any("null" in option.get("type", "") for option in json_schema.get("anyOf", []))


def test_context_is_optional_and_never_carries_a_value() -> None:
    field_schema = ExtractedField.model_json_schema()
    assert "context" in field_schema["properties"]
    assert field_schema["properties"]["context"]["default"] is None
    # A recorded v2 payload with no context still validates: the contract is additive.
    assert ExtractedField(quote="20").context is None
    field = ExtractedField(quote="20", context="Stanley Medical College, Chennai")
    assert field.context == "Stanley Medical College, Chennai"
    assert not hasattr(field, "value")


def test_guided_json_schema_does_not_force_a_value() -> None:
    schema = FAMILIES["fee_structure"].model_json_schema()
    tuition = schema["properties"]["tuition_amount"]
    assert tuition["default"] is None
    assert any(option.get("type") == "null" for option in tuition["anyOf"])


def test_guided_json_schema_forbids_model_supplied_offsets_and_values() -> None:
    field_schema = ExtractedField.model_json_schema()
    assert set(field_schema["properties"]) == {
        "quote",
        "context",
        "confidence",
        "reason_if_null",
    }
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
    assert specs["salary.median"].period_source == "salary_period"
    assert specs["fee.tuition"].unit_kind == "money"
    assert specs["fee.tuition"].period_required is True
    assert specs["fee.tuition"].period_source == "fee_period"
    assert specs["fee.mess"].unit_kind == "money"
    assert specs["fee.mess"].period_required is True
    assert specs["fee.other"].unit_kind == "money"
    assert specs["fee.period"].unit_kind == "text"
    assert specs["seats.admitted"].unit_kind == "count"
    assert specs["seats.duration_months"].unit_kind == "months"
    assert specs["placement.salary_period"].unit_kind == "text"
    assert specs["placement.recruiters_count"].unit_kind == "count"
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
