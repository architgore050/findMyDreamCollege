"""Quantile estimation for salary evidence (brief §11).

Every estimator is a pure function returning an interval, the sample size and
the method that produced it. No naked point estimates: a mean-only input is
refused a point estimate and reported as a Cantelli/Markov bound.

Citations
---------
* Hozo, Djulbegovic & Hozo (2005), *BMC Medical Research Methodology* 5:13 —
  estimating the mean from median, range and sample size.
* Wan, Wang, Tong & Liu (2014), *BMC Medical Research Methodology* 14:135 —
  improved range/interquartile-range based standard-deviation estimators.
* Cantelli's inequality (= one-sided Chebyshev); for a non-negative variable
  with mean ``mu``, Markov gives ``P(X >= 4mu/3) <= 3/4``, hence
  ``Q25 <= 4mu/3``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

# z for the two-sided 80% interval the eval harness scores (§19).
DEFAULT_CONFIDENCE = 0.80


@dataclass(frozen=True)
class QuantileRules:
    """Thresholds and priors that must come from ``config/weights.yaml``."""

    min_n_for_empirical_quantile: int
    sigma_prior_by_sector: Mapping[str, float]
    fallback_sigma: float
    confidence: float = DEFAULT_CONFIDENCE

    @classmethod
    def from_config(cls, salary: Any, *, confidence: float = DEFAULT_CONFIDENCE) -> QuantileRules:
        """Build rules from ``WeightsConfig.salary`` (or a plain mapping).

        ``unknown_sigma_prior_by_sector`` is an extra field on the typed config
        model, so it is read from either the attribute or ``model_extra``.
        """

        def read(name: str, default: Any = None) -> Any:
            if isinstance(salary, Mapping):
                return salary.get(name, default)
            direct = getattr(salary, name, None)
            if direct is not None:
                return direct
            extra = getattr(salary, "model_extra", None) or {}
            return extra.get(name, default)

        priors = dict(read("unknown_sigma_prior_by_sector", {}) or {})
        fallback = float(priors.get("unknown", 0.55))
        return cls(
            min_n_for_empirical_quantile=int(read("min_n_for_empirical_quantile", 10)),
            sigma_prior_by_sector=priors,
            fallback_sigma=fallback,
            confidence=confidence,
        )

    def sigma_for(self, sector: str | None) -> float:
        if sector is None:
            return self.fallback_sigma
        return float(self.sigma_prior_by_sector.get(sector, self.fallback_sigma))


@dataclass(frozen=True)
class QuantileEstimate:
    """A quantile with its interval, sample size and method."""

    q25: float | None
    lo: float
    hi: float
    n: int
    method: str
    assumptions: tuple[str, ...] = ()
    point_refused: bool = False
    sigma: float | None = None
    interval_label: str = field(default="", kw_only=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "q25": None if self.q25 is None else round(self.q25, 2),
            "lo": round(self.lo, 2),
            "hi": round(self.hi, 2),
            "n": self.n,
            "method": self.method,
            "assumptions": list(self.assumptions),
            "point_refused": self.point_refused,
            "sigma": None if self.sigma is None else round(self.sigma, 6),
            "interval_label": self.interval_label,
        }


def _z(p: float) -> float:
    return NormalDist().inv_cdf(p)


Z25 = _z(0.25)
Z75 = _z(0.75)


def _positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be > 0 for a log-normal quantile, got {value!r}")
    return float(value)


def sigma_from_iqr(q1: float, q3: float) -> float:
    """Log-normal sigma implied by the interquartile range.

    ``sigma = (ln Q3 - ln Q1) / (2 * 0.6745)`` because the log-normal
    interquartile range is ``exp(mu) * (e^{0.6745s} - e^{-0.6745s})``.
    """

    _positive("q1", q1)
    _positive("q3", q3)
    if q3 <= q1:
        raise ValueError(f"q3 ({q3}) must be greater than q1 ({q1})")
    return math.log(q3 / q1) / (Z75 - Z25)


def sigma_from_mean_median(mean: float, median: float) -> float:
    """``sigma^2 = 2 ln(mean/median)`` (brief §11). Requires ``mean > median``."""

    _positive("mean", mean)
    _positive("median", median)
    if mean <= median:
        raise ValueError(
            "log-normal mean must exceed the median; "
            f"got mean={mean!r} median={median!r} (distribution is not log-normal)"
        )
    return math.sqrt(2.0 * math.log(mean / median))


def hozo_mean(minimum: float, median: float, maximum: float, n: int) -> float:
    """Hozo (2005) mean estimate, with the Wan (2014) ``n > 70`` simplification."""

    if n <= 0:
        raise ValueError("n must be >= 1")
    if n <= 25 or n > 70:
        return float(median)
    return (float(minimum) + 2.0 * float(median) + float(maximum)) / 4.0


def wan_range_sd(minimum: float, maximum: float, n: int) -> float:
    """Wan (2014) range-based standard deviation.

    ``SD ~ (max - min) / (2 * Phi^-1((n - 0.375) / (n + 0.25)))``. It reproduces
    the usual ``range/4`` for ``25 <= n <= 70`` and ``range/6`` for large n.
    """

    if n < 2:
        raise ValueError("range-based SD needs n >= 2")
    span = float(maximum) - float(minimum)
    if span <= 0:
        raise ValueError("range must be positive")
    quantile = (n - 0.375) / (n + 0.25)
    return span / (2.0 * _z(quantile))


def _interval(
    q: float,
    sigma: float,
    n: int,
    *,
    rules: QuantileRules,
    p: float = 0.25,
    widen: float = 1.0,
) -> tuple[float, float]:
    """Log-normal interval for ``q`` from the asymptotic SE of a sample quantile.

    ``SE(q_p) / q_p = sigma * sqrt(p(1-p)/n) / phi(z_p)``.
    """

    z_p = _z(p)
    density = NormalDist().pdf(z_p)
    effective_n = max(n, 1)
    se_log = sigma * math.sqrt(p * (1.0 - p) / effective_n) / density
    z_conf = _z(1.0 - (1.0 - rules.confidence) / 2.0)
    spread = z_conf * se_log * widen
    return q * math.exp(-spread), q * math.exp(spread)


def _prior_estimate(
    *,
    location: float,
    n: int,
    rules: QuantileRules,
    sector: str | None,
    method: str,
    reason: str,
    assumptions: tuple[str, ...],
) -> QuantileEstimate:
    """Small-sample fallback: sector prior sigma, widened interval."""

    sigma = rules.sigma_for(sector)
    q25 = _positive("location", location) * math.exp(Z25 * sigma)
    widen = math.sqrt(rules.min_n_for_empirical_quantile / max(n, 1))
    lo, hi = _interval(q25, sigma, n, rules=rules, widen=widen)
    return QuantileEstimate(
        q25=q25,
        lo=max(0.0, lo),
        hi=hi,
        n=n,
        method=f"{method}_sector_prior",
        assumptions=(
            *assumptions,
            reason,
            f"sector_prior_sigma={sigma:g}",
            f"interval_widened x{widen:.3f}",
        ),
        sigma=sigma,
        interval_label=f"{int(rules.confidence * 100)}% log-normal (widened)",
    )


def from_median_iqr(
    median: float,
    q1: float,
    q3: float,
    n: int,
    rules: QuantileRules,
    *,
    sector: str | None = None,
) -> QuantileEstimate:
    """median + IQR => Q1 is directly observed (brief §11)."""

    _positive("median", median)
    if n < 1:
        raise ValueError("n must be >= 1")
    if n < rules.min_n_for_empirical_quantile:
        return _prior_estimate(
            location=median,
            n=n,
            rules=rules,
            sector=sector,
            method="median_iqr",
            reason=f"n={n} < {rules.min_n_for_empirical_quantile}: no empirical quantile",
            assumptions=("published median and IQR",),
        )
    sigma = sigma_from_iqr(q1, q3)
    q25 = float(q1)
    lo, hi = _interval(q25, sigma, n, rules=rules)
    return QuantileEstimate(
        q25=q25,
        lo=max(0.0, lo),
        hi=hi,
        n=n,
        method="median_iqr_direct",
        assumptions=(
            "published median and interquartile range",
            "log-normal shape fitted to the IQR for the interval only; Q1 is observed",
        ),
        sigma=sigma,
        interval_label=f"{int(rules.confidence * 100)}% log-normal",
    )


def from_median_range(
    median: float,
    minimum: float,
    maximum: float,
    n: int,
    rules: QuantileRules,
    *,
    sector: str | None = None,
) -> QuantileEstimate:
    """median + range + n => Hozo (2005) mean, Wan (2014) SD, log-normal Q1."""

    _positive("median", median)
    if n < 1:
        raise ValueError("n must be >= 1")
    if maximum <= minimum:
        raise ValueError(f"maximum ({maximum}) must exceed minimum ({minimum})")
    if n < rules.min_n_for_empirical_quantile:
        return _prior_estimate(
            location=median,
            n=n,
            rules=rules,
            sector=sector,
            method="median_range_hozo_wan",
            reason=f"n={n} < {rules.min_n_for_empirical_quantile}: no empirical quantile",
            assumptions=("published median, minimum and maximum",),
        )
    mean = hozo_mean(minimum, median, maximum, n)
    sd = wan_range_sd(minimum, maximum, n)
    cv = sd / mean
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    q25 = median * math.exp(Z25 * sigma)
    lo, hi = _interval(q25, sigma, n, rules=rules)
    return QuantileEstimate(
        q25=q25,
        lo=max(0.0, lo),
        hi=hi,
        n=n,
        method="median_range_hozo_wan",
        assumptions=(
            "Hozo (2005) mean from median+range",
            "Wan (2014) range-based SD",
            "log-normal shape: sigma = sqrt(ln(1 + CV^2))",
        ),
        sigma=sigma,
        interval_label=f"{int(rules.confidence * 100)}% log-normal",
    )


def from_mean_median(
    mean: float,
    median: float,
    n: int,
    rules: QuantileRules,
    *,
    sector: str | None = None,
) -> QuantileEstimate:
    """mean + median => ``Q1 = m * exp(-0.6745 * sigma)``, ``sigma^2 = 2 ln(mu/m)``."""

    _positive("median", median)
    if n < 1:
        raise ValueError("n must be >= 1")
    sigma = sigma_from_mean_median(mean, median)
    if n < rules.min_n_for_empirical_quantile:
        return _prior_estimate(
            location=median,
            n=n,
            rules=rules,
            sector=sector,
            method="mean_median_lognormal",
            reason=f"n={n} < {rules.min_n_for_empirical_quantile}: prior sigma used",
            assumptions=("published mean and median",),
        )
    q25 = median * math.exp(Z25 * sigma)
    lo, hi = _interval(q25, sigma, n, rules=rules)
    return QuantileEstimate(
        q25=q25,
        lo=max(0.0, lo),
        hi=hi,
        n=n,
        method="mean_median_lognormal",
        assumptions=(
            "lognormal: sigma^2 = 2 ln(mean/median)",
            "Q1 = median * exp(-0.6745 * sigma)",
        ),
        sigma=sigma,
        interval_label=f"{int(rules.confidence * 100)}% log-normal",
    )


def from_mean_only(mean: float, n: int = 0) -> QuantileEstimate:
    """mean only => Cantelli/Markov bound ``[0, 4mu/3]``; no point estimate."""

    _positive("mean", mean)
    upper = (4.0 / 3.0) * mean
    return QuantileEstimate(
        q25=None,
        lo=0.0,
        hi=upper,
        n=n,
        method="cantelli_markov_bound",
        assumptions=(
            "mean-only input: distribution unknown, variance unknown",
            "non-negative variable: Markov gives P(X >= 4mu/3) <= 3/4, so Q25 <= 4mu/3",
            "point estimate refused by policy (brief §11)",
        ),
        point_refused=True,
        sigma=None,
        interval_label="Cantelli/Markov bound (no point estimate)",
    )


def _number(stats: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        raw = stats.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def estimate(
    stats: Mapping[str, Any] | Any,
    rules: QuantileRules,
    *,
    sector: str | None = None,
) -> QuantileEstimate | None:
    """Pick the strongest estimator the published statistics support.

    Returns ``None`` when no statistic is present (the caller must propagate
    ``UNKNOWN``, never a zero).
    """

    if not isinstance(stats, Mapping):
        stats = {
            name: getattr(stats, name, None)
            for name in (
                "median",
                "q1",
                "q3",
                "iqr",
                "minimum",
                "min",
                "maximum",
                "max",
                "mean",
                "n",
            )
        }
    n = int(_number(stats, "n", "sample_size") or 0)
    median = _number(stats, "median")
    q1 = _number(stats, "q1", "p25")
    q3 = _number(stats, "q3", "p75")
    minimum = _number(stats, "minimum", "min")
    maximum = _number(stats, "maximum", "max")
    mean = _number(stats, "mean", "average")

    if median is not None and q1 is not None and q3 is not None:
        return from_median_iqr(median, q1, q3, max(n, 1), rules, sector=sector)
    if median is not None and minimum is not None and maximum is not None and n > 0:
        return from_median_range(median, minimum, maximum, n, rules, sector=sector)
    if mean is not None and median is not None:
        return from_mean_median(mean, median, max(n, 1), rules, sector=sector)
    if mean is not None:
        return from_mean_only(mean, n)
    return None
