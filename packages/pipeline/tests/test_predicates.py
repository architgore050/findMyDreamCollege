"""Wave-2 predicate registry: fee/seat/placement vocabulary maps, or config fails.

These tests pin the registry side of the wave-2 field additions: every new
predicate has a row, its unit kind agrees with the extraction schema, the
alternate spellings are declared aliases (never silent re-names), every row
carries a one-sentence description (the verifier's authority for what the
predicate means and its scope), and a dangling component, a dangling alias or a
missing description is a config error rather than a silent drop.
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

# The seat-semantics split (W2): seats.intake is the row-scoped sanctioned
# intake, seats.total the document-stated aggregate; both legacy spellings
# meant "the sanctioned seat count for one programme", so both alias to
# seats.intake now.
WAVE2_ALIASES = {
    "seats.sanctioned": "seats.intake",
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


def test_seat_vocabulary_is_split_into_row_scoped_and_aggregate(registry) -> None:
    """`seats.intake` is the row-scoped sanctioned intake; `seats.total` the aggregate.

    The Stanley case: a seat matrix prints one number per programme row per
    institution, and sometimes a printed Total row. A row value published as
    `seats.total` is exactly the semantic mismatch the verifier must reject, so
    the vocabulary itself has to separate the scopes and say which is which.
    """
    rules = registry.predicates.predicates
    intake = rules["seats.intake"]
    total = rules["seats.total"]
    assert intake.component == "admission_realism"
    assert total.component == "admission_realism"
    assert intake.unit_kind == "count" and total.unit_kind == "count"
    assert "THIS programme" in intake.description and "THIS institution" in intake.description
    assert "aggregate" in intake.description.lower()
    assert "aggregate" in total.description.lower()
    assert "never" in total.description.lower() or "must never" in total.description
    # `seats.admitted` stays a filled-count evidence input, never intake.
    assert rules["seats.admitted"].component is None
    assert rules["seats.admitted"].scoring is None
    # The predicate the extractor emits for a programme row is `seats.intake`.
    specs = predicate_specs()
    assert specs["seats.intake"].unit_kind == "count"
    assert specs["seats.total"].unit_kind == "count"


def test_every_predicate_declares_a_nonempty_description(registry) -> None:
    """The description is the authority the verifier receives; no blanks allowed."""
    for predicate, rule in registry.predicates.predicates.items():
        assert (rule.description or "").strip(), f"{predicate} needs a description"
        assert rule.description.strip().endswith("."), (
            f"{predicate}'s description should be a one-sentence statement"
        )


def test_every_predicate_declares_a_component_or_null_explicitly(registry) -> None:
    """Each row states its component (or evidence-only null), and it must exist."""
    weights = registry.weights.weights
    for predicate, rule in registry.predicates.predicates.items():
        if rule.component is not None:
            assert rule.component in weights, (
                f"{predicate}: dangling component {rule.component!r}"
            )


def test_seat_aliases_point_at_the_row_scoped_predicate(predicate_registry) -> None:
    """`seats`/`seats.sanctioned` meant one programme's sanctioned count -> seats.intake."""
    assert predicate_registry._aliases["seats"] == "seats.intake"
    assert predicate_registry._aliases["seats.sanctioned"] == "seats.intake"
    for alias in ("seats", "seats.sanctioned"):
        resolved = predicate_registry.resolve(alias)
        assert resolved is not None and resolved.predicate == "seats.intake", alias
        assert resolved.component == "admission_realism"


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
    assert resolved.predicate == "seats.intake"
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


def test_missing_description_is_rejected(config_dir: Path) -> None:
    _mutate(
        config_dir,
        lambda payload: payload["predicates"]["fee.mess"].__setitem__("description", None),
    )
    with pytest.raises(ConfigError, match="description"):
        load_registry(config_dir, env={})


def test_blank_description_is_rejected(config_dir: Path) -> None:
    _mutate(
        config_dir,
        lambda payload: payload["predicates"]["fee.mess"].__setitem__("description", "   "),
    )
    with pytest.raises(ConfigError, match="description"):
        load_registry(config_dir, env={})


def test_predicate_registry_itself_refuses_a_missing_description() -> None:
    """The scorer-side registry fails fast even without the config loader."""
    from fmd.score.predicates import PredicateRegistryError

    with pytest.raises(PredicateRegistryError, match="description"):
        PredicateRegistry.from_config(
            {"version": "test", "predicates": {"a.b": {"unit_kind": "count"}}}
        )


def test_component_collision_with_alias_is_rejected(config_dir: Path) -> None:
    _mutate(
        config_dir,
        lambda payload: payload["predicates"].__setitem__(
            "placement.eligible", {"unit_kind": "count"}
        ),
    )
    with pytest.raises(ConfigError, match="collides"):
        load_registry(config_dir, env={})
