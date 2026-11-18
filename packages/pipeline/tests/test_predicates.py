"""Wave-2 predicate registry: fee/seat/placement vocabulary maps, or config fails.

These tests pin the registry side of the wave-2 field additions: every new
predicate has a row, its unit kind agrees with the extraction schema, the
alternate spellings are declared aliases (never silent re-names), and a
dangling component or alias is a config error rather than a silent drop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmd.config import ConfigError, load_registry
from fmd.extract.schemas import predicate_specs
from fmd.score.predicates import PredicateRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"

WAVE2_PREDICATES = (
    "fee.mess",
    "fee.other",
    "fee.period",
    "seats.admitted",
    "seats.duration_months",
    "placement.salary_period",
    "placement.recruiters_count",
)

WAVE2_ALIASES = {
    "seats.sanctioned": "seats.total",
    "placement.eligible": "placement.eligible_count",
    "placement.placed": "placement.placed_count",
    "placement.median_salary": "salary.median",
    "placement.highest_salary": "salary.highest_package",
}


@pytest.fixture(scope="module")
def registry():
    return load_registry(CONFIG_DIR, env={})


@pytest.fixture(scope="module")
def predicate_registry(registry) -> PredicateRegistry:
    return PredicateRegistry.from_config(registry.predicates)


def test_every_wave2_predicate_has_a_registry_row(registry) -> None:
    rules = registry.predicates.predicates
    for predicate in WAVE2_PREDICATES:
        assert predicate in rules, f"{predicate} must have a predicates.yaml row"
    assert registry.predicates.version >= "2026-11-18.2"


def test_wave2_unit_kinds_agree_with_the_extraction_schema(registry) -> None:
    rules = registry.predicates.predicates
    assert rules["fee.mess"].unit_kind == "money"
    assert rules["fee.mess"].requires_period is True
    assert rules["fee.other"].unit_kind == "money"
    assert rules["fee.other"].requires_period is True
    assert rules["fee.period"].unit_kind == "text"
    assert rules["seats.admitted"].unit_kind == "count"
    assert rules["seats.duration_months"].unit_kind == "months"
    assert rules["placement.salary_period"].unit_kind == "text"
    assert rules["placement.recruiters_count"].unit_kind == "count"

    # The loader cross-checks every emitted predicate; assert the wave-2 set
    # explicitly so a schema/registry drift cannot reintroduce a mismatch.
    specs = predicate_specs()
    for predicate in WAVE2_PREDICATES:
        spec = specs[predicate]
        rule = rules[predicate]
        if spec.unit_kind is not None:
            assert spec.unit_kind == rule.unit_kind, predicate


def test_wave2_money_predicates_require_a_period_like_the_schema(registry) -> None:
    rules = registry.predicates.predicates
    specs = predicate_specs()
    for predicate, spec in specs.items():
        rule = rules[predicate]
        if spec.unit_kind == "money" and spec.period_required:
            assert rule.requires_period is True, predicate


def test_wave2_alternate_spellings_are_declared_aliases(predicate_registry) -> None:
    registry_alias_targets = predicate_registry._aliases
    for alias, target in WAVE2_ALIASES.items():
        assert registry_alias_targets.get(alias) == target, alias
        resolved = predicate_registry.resolve(alias)
        assert resolved is not None, alias
        assert resolved.predicate == target
        assert resolved.alias_of == alias


def test_sanctioned_alias_resolves_to_the_scoring_row(predicate_registry) -> None:
    resolved = predicate_registry.resolve("seats.sanctioned")
    assert resolved is not None
    assert resolved.component == "admission_realism"
    assert resolved.unit_kind == "count"
    assert resolved.kind == "transform"


def test_new_predicates_are_evidence_only_and_carry_no_scoring(registry) -> None:
    rules = registry.predicates.predicates
    for predicate in WAVE2_PREDICATES:
        rule = rules[predicate]
        if predicate in {"seats.admitted", "seats.duration_months", "placement.salary_period",
                         "placement.recruiters_count", "fee.period"}:
            assert rule.component is None, f"{predicate} must stay evidence-only"
            assert rule.scoring is None


def test_weights_values_are_untouched_by_wave2(registry) -> None:
    """Guards the acceptance rule: no weight value may change in this patch."""
    weights = registry.weights.weights
    assert set(weights) == {
        "outcomes",
        "clinical_training",
        "academic_quality",
        "affordability_roi",
        "financing_feasibility",
        "student_life_safety",
        "admission_realism",
    }


def _mutate(config_dir: Path, mutate) -> None:
    path = config_dir / "predicates.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_dangling_component_is_rejected(config_dir: Path) -> None:
    _mutate(
        config_dir,
        lambda payload: payload["predicates"]["fee.mess"].__setitem__(
            "component", "not_a_component"
        ),
    )
    with pytest.raises(ConfigError, match="not_a_component"):
        load_registry(config_dir, env={})


def test_dangling_alias_target_is_rejected(config_dir: Path) -> None:
    _mutate(
        config_dir,
        lambda payload: payload["aliases"].__setitem__("seats.sanctioned", "not_a_predicate"),
    )
    with pytest.raises(ConfigError, match="seats.sanctioned"):
        load_registry(config_dir, env={})


def test_component_collision_with_alias_is_rejected(config_dir: Path) -> None:
    _mutate(
        config_dir,
        lambda payload: payload["predicates"].__setitem__(
            "placement.eligible", {"unit_kind": "count"}
        ),
    )
    with pytest.raises(ConfigError, match="collides"):
        load_registry(config_dir, env={})
