from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmd.config import ConfigError, load_registry
from fmd.score import score_programme
from fmd.score.config import ScoringConfig
from fmd.score.evidence import ClaimIndex
from fmd.score.predicates import PredicateRegistry, PredicateRegistryError

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="module")
def registry():
    return load_registry(CONFIG_DIR, env={})


@pytest.fixture(scope="module")
def scoring(registry) -> ScoringConfig:
    return ScoringConfig.from_config(registry.weights, registry.predicates)


@pytest.fixture(scope="module")
def predicate_registry(registry) -> PredicateRegistry:
    return PredicateRegistry.from_config(registry.predicates)


def test_namespaced_predicates_map_to_components_and_units(registry) -> None:
    rules = registry.predicates.predicates
    assert rules["seats.total"].component == "admission_realism"
    assert rules["seats.total"].unit_kind == "count"
    assert rules["curriculum.internship_months"].component == "clinical_training"
    assert rules["curriculum.internship_months"].unit_kind == "months"
    assert rules["curriculum.semesters"].unit_kind == "count"
    assert rules["curriculum.semesters"].component is None
    assert rules["accreditation.naac_grade"].component == "academic_quality"
    assert rules["accreditation.naac_grade"].unit_kind == "grade"
    assert rules["salary.median"].unit_kind == "money"
    assert rules["salary.median"].requires_period is True

    # Every extraction predicate is covered.
    from fmd.extract.schemas import predicate_specs

    assert set(predicate_specs()) <= set(rules)


def test_alias_resolution_is_explicit(predicate_registry) -> None:
    resolved = predicate_registry.resolve("seats")
    assert resolved is not None
    assert resolved.predicate == "seats.total"
    assert resolved.alias_of == "seats"


def test_unit_mismatch_is_rejected(predicate_registry) -> None:
    from fmd.score.evidence import ClaimView

    rule = predicate_registry.resolve("curriculum.internship_months")
    assert rule is not None
    assert predicate_registry.rejects(
        ClaimView(predicate="curriculum.internship_months", value=6, unit="months"), rule
    ) is None
    reason = predicate_registry.rejects(
        ClaimView(predicate="curriculum.internship_months", value=6, unit="count"), rule
    )
    assert reason is not None and "unit" in reason


def test_period_is_required_for_money(predicate_registry) -> None:
    from fmd.score.evidence import ClaimView

    rule = predicate_registry.resolve("salary.median")
    assert rule is not None
    without = ClaimView(predicate="salary.median", value=30000, unit="INR")
    assert predicate_registry.rejects(without, rule) is not None
    with_period = ClaimView(
        predicate="salary.median",
        value=30000,
        unit="INR",
        qualifier={"period": "per_month"},
    )
    assert predicate_registry.rejects(with_period, rule) is None


def test_unknown_predicates_are_reported(predicate_registry) -> None:
    unknown = predicate_registry.unknown_predicates(
        [
            {"predicate": "seats.total", "value": 20, "unit": "count"},
            {"predicate": "made.up.predicate", "value": 1},
        ]
    )
    assert unknown == ("made.up.predicate",)


def test_single_component_composite_is_partial_and_wide(scoring) -> None:
    claims = [
        {
            "predicate": "seats.total",
            "value": 100,
            "unit": "count",
            "claim_id": 1,
            "source_id": 1,
            "confidence": 1.0,
        }
    ]
    result = score_programme(
        1,
        {"name": "x"},
        {},
        claims,
        config=scoring,
    )
    assert result.composite.point is not None
    assert result.composite.partial is True
    assert result.composite.components_used == ("admission_realism",)
    assert "outcomes" in result.composite.components_unknown
    # one low-weight component cannot masquerade as the whole picture
    assert result.composite.lo <= result.composite.point <= result.composite.hi
    assert result.composite.hi - result.composite.lo > 50.0
    assert result.evidence_tier in {"partial", "thin", "insufficient"}
    assert result.unknown_predicates == ()


def test_zero_components_composite_is_unknown(scoring) -> None:
    result = score_programme(1, {"name": "x"}, {}, [], config=scoring)
    assert result.composite.point is None
    assert (result.composite.lo, result.composite.hi) == (0.0, 100.0)
    assert result.composite.components_used == ()
    assert result.evidence_tier == "insufficient"


def test_two_claim_programme_is_never_well_evidenced(scoring) -> None:
    claims = [
        {
            "predicate": "seats.total",
            "value": 60,
            "unit": "count",
            "claim_id": 1,
            "source_id": 1,
        },
        {
            "predicate": "accreditation.naac_grade",
            "value": "A",
            "unit": "grade",
            "claim_id": 2,
            "source_id": 2,
        },
    ]
    result = score_programme(1, {"name": "x"}, {}, claims, config=scoring)
    assert result.evidence_tier != "well_evidenced"
    assert result.evidence_tier in {"partial", "thin", "insufficient"}


def test_score_is_reproducible_from_hash_and_versions(scoring) -> None:
    claims = [
        {
            "predicate": "seats.total",
            "value": 40,
            "unit": "count",
            "claim_id": 1,
            "source_id": 1,
        }
    ]
    first = score_programme(1, {"name": "x"}, {}, claims, config=scoring)
    second = score_programme(1, {"name": "x"}, {}, list(reversed(claims)), config=scoring)
    assert first.claims_hash == second.claims_hash
    assert (first.scorer_version, first.weights_version) == (
        second.scorer_version,
        second.weights_version,
    )
    assert first.composite.as_dict() == second.composite.as_dict()


def _mutate(config_dir: Path, mutate) -> None:
    path = config_dir / "predicates.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_missing_predicate_file_is_a_config_error(config_dir: Path) -> None:
    (config_dir / "predicates.yaml").unlink()
    with pytest.raises(ConfigError, match="predicates.yaml"):
        load_registry(config_dir, env={})


def test_dangling_component_reference_is_a_config_error(config_dir: Path) -> None:
    _mutate(
        config_dir,
        lambda payload: payload["predicates"]["seats.total"].__setitem__(
            "component", "not_a_component"
        ),
    )
    with pytest.raises(ConfigError, match="not_a_component"):
        load_registry(config_dir, env={})


def test_missing_emitted_predicate_is_a_config_error(config_dir: Path) -> None:
    _mutate(config_dir, lambda payload: payload["predicates"].pop("seats.total"))
    with pytest.raises(ConfigError, match="seats.total"):
        load_registry(config_dir, env={})


def test_registry_without_predicate_map_raises() -> None:
    with pytest.raises(PredicateRegistryError):
        PredicateRegistry.from_config({"version": "x", "predicates": {}})


def test_registry_dangling_alias_raises() -> None:
    with pytest.raises(PredicateRegistryError):
        PredicateRegistry.from_config(
            {
                "version": "x",
                "predicates": {"a.b": {"unit_kind": "count"}},
                "aliases": {"old": "missing"},
            }
        )


def test_claim_index_is_not_consulted_for_bare_guessing(scoring) -> None:
    # A bare name that is not a declared alias must stay unknown; the old
    # `seats` lookup is only tolerated because predicates.yaml declares it.
    index = ClaimIndex.build([{"predicate": "made.up", "value": 1}])
    assert scoring.predicates is not None
    assert scoring.predicates.unknown_predicates(index) == ("made.up",)
