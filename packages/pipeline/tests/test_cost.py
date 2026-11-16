from __future__ import annotations

import pytest

from fmd.cost import (
    CostInputs,
    CostItem,
    CostModelError,
    CostRules,
    LoanTerms,
    MissingFXRate,
    StudentWork,
    capitalised_debt,
    compute,
    dsr_band,
    emi,
    loan_terms_from_config,
    student_work_from_config,
)


@pytest.fixture(scope="module")
def rules(registry) -> CostRules:
    return CostRules.from_config(registry.cost_rules)


def base_items() -> list[CostItem]:
    return [
        CostItem(category="tuition", amount=100_000, period="per_year", label="tuition"),
        CostItem(category="hostel", amount=60_000, period="per_year"),
        CostItem(category="mess", amount=40_000, period="per_year"),
        CostItem(category="equipment", amount=6_000, period="one_time"),
        CostItem(
            category="transport",
            amount=10_000,
            period="one_time",
            internship_travel=True,
        ),
    ]


CITY_BASKET = {
    "hostel": 5_000,
    "mess": 3_000,
    "utilities": 1_000,
    "transport": 800,
    "rent": 8_000,
    "food": 4_000,
}


def inputs(**overrides) -> CostInputs:
    payload = {
        "items": base_items(),
        "duration_months": 48,
        "city_basket_monthly": CITY_BASKET,
        "loan": LoanTerms(
            principal=300_000,
            annual_rate=0.09,
            moratorium_months=60,
            tenure_months=120,
        ),
        "family_income_annual": 600_000,
        "p25_first_year_salary": 300_000,
    }
    payload.update(overrides)
    return CostInputs(**payload)


def test_emi_hand_computed() -> None:
    # P=300000, 9% annual, 120 months -> 3800.27
    assert emi(300_000, 0.09, 120) == pytest.approx(3_800.27, abs=0.01)
    assert emi(120_000, 0.0, 12) == pytest.approx(10_000.0)


def test_capitalised_moratorium_interest() -> None:
    # simple: 300000 * (1 + 0.09 * 60/12) = 435000
    assert capitalised_debt(300_000, 0.09, 60) == pytest.approx(435_000.0)
    compound = capitalised_debt(300_000, 0.09, 60, basis="monthly_compound")
    assert compound > 435_000.0
    assert capitalised_debt(300_000, 0.0, 60) == 300_000.0


def test_dsr25_bands_at_boundaries(rules: CostRules) -> None:
    bands = rules.dsr_bands
    assert dsr_band(0.0, bands) == "comfortable"
    assert dsr_band(0.1999, bands) == "comfortable"
    # half-open [min, max): 0.20 belongs to the next band
    assert dsr_band(0.20, bands) == "manageable"
    assert dsr_band(0.2999, bands) == "manageable"
    assert dsr_band(0.30, bands) == "stressed"
    assert dsr_band(0.3999, bands) == "stressed"
    assert dsr_band(0.40, bands) == "not_financeable"
    assert dsr_band(5.0, bands) == "not_financeable"
    # exactly one band matches any number: no overlaps and no gaps
    names = [dsr_band(value / 10_000, bands) for value in range(0, 20_001)]
    assert set(names) == {"comfortable", "manageable", "stressed", "not_financeable"}
    assert "unknown" not in names
    # missing salary is unknown, never 0
    assert dsr_band(None, bands) == "unknown"


def test_food_topup_is_a_known_cost_category(registry, rules: CostRules) -> None:
    assert "food_topup" in rules.cost_categories
    for scenario, components in rules.living_scenarios.items():
        for component in components:
            assert component in rules.cost_categories, f"{scenario}:{component}"


def test_living_scenario_component_must_be_a_known_category(registry) -> None:
    raw = registry.cost_rules.model_dump()
    raw["living_scenarios"]["hostel"]["components"].append("bribes")
    with pytest.raises(CostModelError, match="not in cost_categories"):
        CostRules.from_config(raw)


def test_overlapping_dsr_bands_are_rejected(registry) -> None:
    raw = registry.cost_rules.model_dump()
    raw["dsr25"]["bands"]["manageable"]["min"] = 0.25
    with pytest.raises(CostModelError, match="does not continue"):
        CostRules.from_config(raw)


def test_currency_is_required_not_defaulted(registry) -> None:
    raw = registry.cost_rules.model_dump()
    raw["currency"]["base"] = ""
    with pytest.raises(CostModelError, match="currency"):
        CostRules.from_config(raw)


def test_tcoa_counts_each_rupee_once(rules: CostRules) -> None:
    snapshot = compute(inputs(), rules)
    # tuition 4y = 400000; equipment 6000; internship travel 10000
    # hostel 60000 + mess 40000 + basket utilities/transport/food_topup (12000+9600+30000)
    living = snapshot.scenarios["hostel"].annual.value
    assert living == pytest.approx(151_600.0)
    assert snapshot.tcoa.value == pytest.approx(416_000 + 151_600 * 4)
    assert snapshot.tcoa_first_year.value == pytest.approx(100_000 + 6_000 + 10_000 + 151_600)
    assert "opportunity cost" in " ".join(snapshot.tcoa.assumptions)


def test_living_scenarios_are_side_by_side_not_summed(rules: CostRules) -> None:
    snapshot = compute(inputs(), rules)
    assert set(snapshot.scenarios) == {"hostel", "independent"}
    hostel = snapshot.scenarios["hostel"].total.value
    independent = snapshot.scenarios["independent"].total.value
    assert snapshot.tcoa.value == pytest.approx(416_000 + hostel)
    assert snapshot.tcoa.value != pytest.approx(416_000 + hostel + independent)
    assert snapshot.display["living_cards"][0]["scenario"] == "hostel"


def test_student_work_reduces_out_of_pocket_only(rules: CostRules) -> None:
    work = StudentWork(
        legal_hours_per_week=0,
        realistic_monthly=3_000,
        cap_pct_of_tcoa=10,
        label="informal_earnings",
    )
    without = compute(inputs(), rules)
    with_work = compute(inputs(student_work=work), rules)
    assert with_work.tcoa.value == pytest.approx(without.tcoa.value)
    assert with_work.student_work_offset.value == pytest.approx(0.10 * without.tcoa.value)
    assert with_work.family_out_of_pocket.value == pytest.approx(
        without.family_out_of_pocket.value - with_work.student_work_offset.value
    )


def test_student_work_cap_never_exceeds_realistic_earnings(registry, rules: CostRules) -> None:
    work = student_work_from_config(registry.cost_rules, "IN")
    snapshot = compute(inputs(student_work=work), rules)
    years = 4
    assert snapshot.student_work_offset.value <= work.realistic_monthly * 12 * years
    assert snapshot.student_work_offset.value <= 0.10 * snapshot.tcoa.value


def test_net_price_and_family_out_of_pocket(rules: CostRules) -> None:
    aid = [CostItem(category="scholarship", amount=50_000, period="per_year")]
    snapshot = compute(inputs(aids=aid), rules)
    assert snapshot.aid_total.value == pytest.approx(200_000.0)
    assert snapshot.net_price.value == pytest.approx(snapshot.tcoa.value - 200_000.0)
    assert snapshot.loan_principal.value == pytest.approx(300_000.0)
    assert snapshot.family_out_of_pocket.value == pytest.approx(
        max(0.0, snapshot.net_price.value - 300_000.0)
    )
    # loan principal never exceeds net price
    small = compute(
        inputs(aids=[CostItem(category="aid", amount=1_000_000, period="one_time")]), rules
    )
    assert small.loan_principal.value <= small.net_price.value


def test_dsr_and_affordability_gap(rules: CostRules) -> None:
    snapshot = compute(inputs(), rules)
    assert snapshot.dsr25.value == pytest.approx(12 * snapshot.emi.value / 300_000.0)
    assert snapshot.dsr25.band in {"comfortable", "manageable", "stressed", "not_financeable"}
    assert snapshot.dsr25.first_year_emi == pytest.approx(12 * snapshot.emi.value)
    assert snapshot.affordability_gap.value == pytest.approx(
        snapshot.family_out_of_pocket.value / 600_000.0
    )
    assert snapshot.affordability_gap.flagged is True

    no_salary = compute(inputs(p25_first_year_salary=None), rules)
    assert no_salary.dsr25.value is None
    assert no_salary.dsr25.band == "unknown"
    no_income = compute(inputs(family_income_annual=None), rules)
    assert no_income.affordability_gap.value is None
    assert no_income.affordability_gap.flagged is None


def test_fx_band_and_missing_rate(rules: CostRules) -> None:
    items = base_items() + [
        CostItem(category="insurance", amount=1_000, currency="USD", period="per_year")
    ]
    snapshot = compute(inputs(items=items, fx_rates={"USD": 83.0}), rules)
    assert snapshot.fx_band_pct == rules.fx_sensitivity_pct
    assert any(
        scenario.annual.method.startswith("modelled") for scenario in snapshot.scenarios.values()
    )

    with pytest.raises(MissingFXRate):
        compute(inputs(items=items, fx_rates={}), rules)


def test_snapshot_render_is_json_ready(rules: CostRules) -> None:
    snapshot = compute(inputs(), rules)
    payload = snapshot.render()
    assert payload["currency"] == "INR"
    assert payload["methods"]["dsr25"]
    assert payload["display"]["tcoa"]
    assert set(payload["scenarios"]) == {"hostel", "independent"}
    assert payload["dsr25"]["bands"]


def test_unknown_living_scenario_is_rejected(rules: CostRules) -> None:
    with pytest.raises((CostModelError, ValueError)):
        CostInputs(items=base_items(), living="commute")


def test_loan_defaults_expose_assumed_terms(registry) -> None:
    terms = loan_terms_from_config(registry.cost_rules)
    assert terms.annual_rate is not None
    assert 0.0 <= terms.annual_rate < 1.0
    assert terms.tenure_months > 0
    assert terms.assumed_rate is True
    assert terms.rate_source == "assumption"
    defaults = registry.cost_rules.loan_defaults
    assert all(spec.get("assumed") is True for spec in defaults.values())
    assert all(spec.get("assumed_rate_pct") is not None for spec in defaults.values())


def test_dsr25_is_unknown_without_a_rate(registry, rules: CostRules) -> None:
    no_rate = LoanTerms(principal=300_000, annual_rate=None, tenure_months=120)
    snapshot = compute(inputs(loan=no_rate), rules)
    assert snapshot.dsr25.value is None
    assert snapshot.dsr25.band == "unknown"
    assert snapshot.emi.value is None
    assert snapshot.emi.method == "unknown_rate"
    assert snapshot.debt_at_repayment.value is None


def test_assumed_rate_is_flagged_not_presented_as_published(registry, rules: CostRules) -> None:
    terms = loan_terms_from_config(registry.cost_rules)
    snapshot = compute(inputs(loan=terms), rules)
    assert snapshot.assumptions_flags["loan_rate"]["assumed"] is True
    assert any("assumed: true" in note for note in snapshot.emi.assumptions)
    assert snapshot.dsr25.inputs["rate_assumed"] is True
