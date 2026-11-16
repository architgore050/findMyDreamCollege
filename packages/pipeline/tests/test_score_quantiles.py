from __future__ import annotations

import math

import pytest

from fmd.score.quantiles import (
    QuantileRules,
    estimate,
    from_mean_median,
    from_mean_only,
    from_median_iqr,
    from_median_range,
    hozo_mean,
    sigma_from_iqr,
    sigma_from_mean_median,
    wan_range_sd,
)


@pytest.fixture(scope="module")
def rules(registry) -> QuantileRules:
    return QuantileRules.from_config(registry.weights.salary)


def test_mean_only_refuses_point_estimate() -> None:
    result = from_mean_only(300_000, n=20)
    assert result.q25 is None
    assert result.lo == 0.0
    assert result.hi == pytest.approx(400_000.0)
    assert result.point_refused is True
    assert result.method == "cantelli_markov_bound"
    assert result.n == 20
    assert any("point estimate refused" in note for note in result.assumptions)


def test_mean_and_median_lognormal(rules: QuantileRules) -> None:
    # sigma^2 = 2 ln(360000/300000) = 0.364643; sigma = 0.603857
    # Q1 = 300000 * exp(-0.6745 * 0.603857) = 199634.3
    result = from_mean_median(360_000, 300_000, 40, rules)
    assert result.method == "mean_median_lognormal"
    assert result.sigma == pytest.approx(0.603857, abs=1e-6)
    assert result.q25 == pytest.approx(199_634.30, abs=1.0)
    assert result.lo < result.q25 < result.hi
    assert result.lo == pytest.approx(168_975.76, abs=1.0)
    assert result.hi == pytest.approx(235_855.46, abs=1.0)
    assert result.n == 40


def test_mean_below_median_is_rejected(rules: QuantileRules) -> None:
    with pytest.raises(ValueError, match="log-normal mean must exceed"):
        from_mean_median(250_000, 300_000, 40, rules)


def test_median_iqr_gives_q1_directly(rules: QuantileRules) -> None:
    # sigma = ln(380000/250000)/1.349 = 0.31039
    result = from_median_iqr(300_000, 250_000, 380_000, 100, rules)
    assert result.method == "median_iqr_direct"
    assert result.q25 == 250_000.0
    assert result.sigma == pytest.approx(0.31039, abs=1e-4)
    assert result.lo == pytest.approx(236_809.95, abs=1.0)
    assert result.hi == pytest.approx(263_924.72, abs=1.0)


def test_median_range_uses_hozo_and_wan(rules: QuantileRules) -> None:
    # n=40 -> Hozo mean = (180000 + 2*300000 + 600000)/4 = 345000
    assert hozo_mean(180_000, 300_000, 600_000, 40) == 345_000.0
    # n=10 and n=100 use the median per Hozo/Wan
    assert hozo_mean(180_000, 300_000, 600_000, 10) == 300_000.0
    assert hozo_mean(180_000, 300_000, 600_000, 100) == 300_000.0
    # Wan SD = 420000 / (2 * Phi^-1((40-0.375)/(40+0.25))) ~= 97440
    assert wan_range_sd(180_000, 600_000, 40) == pytest.approx(97_440, abs=200)

    result = from_median_range(300_000, 180_000, 600_000, 40, rules)
    assert result.method == "median_range_hozo_wan"
    assert result.q25 == pytest.approx(248_893.47, abs=1.0)
    assert result.sigma == pytest.approx(0.27689, abs=1e-4)
    assert result.lo < result.q25 < result.hi


def test_small_sample_uses_sector_prior_and_widens(rules: QuantileRules) -> None:
    small = from_median_iqr(300_000, 250_000, 380_000, 6, rules, sector="government")
    assert small.method == "median_iqr_sector_prior"
    assert small.sigma == pytest.approx(0.35)
    assert small.q25 == pytest.approx(300_000 * math.exp(-0.6745 * 0.35), abs=1.0)
    assert any("no empirical quantile" in note for note in small.assumptions)
    assert any("interval_widened" in note for note in small.assumptions)

    large = from_median_iqr(300_000, 250_000, 380_000, 100, rules)
    # widening makes the small-sample interval strictly wider in log space
    assert (small.hi / small.lo) > (large.hi / large.lo)


def test_unknown_sector_falls_back_to_unknown_prior(rules: QuantileRules) -> None:
    result = from_median_range(300_000, 180_000, 600_000, 5, rules)
    assert result.sigma == pytest.approx(0.55)


def test_sigma_helpers() -> None:
    assert sigma_from_iqr(250_000, 380_000) == pytest.approx(0.31039, abs=1e-4)
    assert sigma_from_mean_median(360_000, 300_000) == pytest.approx(0.603857, abs=1e-6)
    with pytest.raises(ValueError):
        sigma_from_iqr(380_000, 250_000)


def test_estimator_dispatch_and_empty(rules: QuantileRules) -> None:
    assert estimate({}, rules) is None
    assert estimate({"mean": 300_000}, rules).method == "cantelli_markov_bound"
    assert (
        estimate({"mean": 360_000, "median": 300_000, "n": 40}, rules).method
        == "mean_median_lognormal"
    )
    assert (
        estimate({"median": 300_000, "q1": 250_000, "q3": 380_000, "n": 40}, rules).method
        == "median_iqr_direct"
    )
