"""TCOA, financing and affordability model (brief §13).

Every rupee is counted once. Living is modelled as two complete, non-additive
scenarios (``hostel`` / ``independent``) that are presented side by side and
never summed. Student work never reduces TCOA — it reduces family
out-of-pocket, capped. Non-base-currency amounts carry an FX sensitivity band and
are never stored as a point rate.

:func:`compute` returns a :class:`CostSnapshot` whose JSON form is enough for the
API/UI to render without recomputing anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Period = Literal["one_time", "per_year", "per_month"]


class CostModelError(ValueError):
    """Raised when an input cannot be modelled without inventing data."""


class MissingFXRate(CostModelError):
    """A non-base currency has no supplied rate; refusing to guess."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class MoneyBand(_Model):
    value: float
    lo: float
    hi: float
    currency: str
    method: str
    assumptions: list[str] = Field(default_factory=list)

    @property
    def is_modelled(self) -> bool:
        return self.method.startswith("modelled")

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 2),
            "lo": round(self.lo, 2),
            "hi": round(self.hi, 2),
            "currency": self.currency,
            "method": self.method,
            "assumptions": list(self.assumptions),
        }


class CostItem(_Model):
    category: str
    label: str = ""
    amount: float
    currency: str | None = None
    period: Period = "per_year"
    method: str = "extracted"
    confidence: float = 1.0
    basis: str | None = None
    source_claim_id: int | str | None = None
    internship_travel: bool = False
    as_of: str | None = None

    @field_validator("amount")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("cost amount must be >= 0")
        return value


class LoanTerms(_Model):
    principal: float | None = None
    annual_rate: float = 0.0
    moratorium_months: int | None = None
    tenure_months: int = 120
    interest_basis: Literal["simple", "monthly_compound"] = "simple"

    @field_validator("annual_rate")
    @classmethod
    def _rate_range(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("annual_rate must be in [0, 1)")
        return value

    @field_validator("tenure_months")
    @classmethod
    def _tenure(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("tenure_months must be > 0")
        return value


class StudentWork(_Model):
    legal_hours_per_week: float = 0.0
    realistic_monthly: float = 0.0
    currency: str | None = None
    cap_pct_of_tcoa: float = 0.0
    label: str = "informal_earnings"


@dataclass(frozen=True)
class CostRules:
    base_currency: str
    fx_sensitivity_pct: float
    band_pct: float
    living_scenarios: Mapping[str, tuple[str, ...]]
    dsr_bands: Mapping[str, Mapping[str, float]]
    affordability_flag_above: float
    modelled_defaults: Mapping[str, Any] = field(default_factory=dict)
    loan_defaults: Mapping[str, Any] = field(default_factory=dict)
    as_of_required: bool = True

    @classmethod
    def from_config(cls, cost_rules: Any) -> CostRules:
        raw = _plain(cost_rules)
        return cls(
            base_currency=str((raw.get("currency") or {}).get("base", "INR")),
            fx_sensitivity_pct=float((raw.get("currency") or {}).get("fx_sensitivity_pct", 10)),
            band_pct=float((raw.get("modelled_defaults") or {}).get("band_pct", 20)),
            living_scenarios={
                name: tuple(spec.get("components") or ())
                for name, spec in (raw.get("living_scenarios") or {}).items()
            },
            dsr_bands={
                name: dict(spec)
                for name, spec in ((raw.get("dsr25") or {}).get("bands") or {}).items()
            },
            affordability_flag_above=float(
                (raw.get("affordability_gap") or {}).get("flag_above", 1.0)
            ),
            modelled_defaults=dict(raw.get("modelled_defaults") or {}),
            loan_defaults=dict(raw.get("loan_defaults") or {}),
            as_of_required=bool((raw.get("currency") or {}).get("as_of_required", True)),
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    return value


def student_work_from_config(cost_rules: Any, country: str) -> StudentWork:
    raw = _plain(cost_rules)
    entry = ((raw.get("student_work") or {}).get(country)) or {}
    informal = entry.get("realistic_informal_monthly") or {}
    return StudentWork(
        legal_hours_per_week=float(entry.get("legal_hours_per_week", 0.0)),
        realistic_monthly=float(informal.get("amount", 0.0)),
        currency=str(informal.get("currency", (raw.get("currency") or {}).get("base", "INR"))),
        cap_pct_of_tcoa=float(entry.get("cap_pct_of_tcoa", 0.0)),
        label=str(entry.get("label", "informal_earnings")),
    )


# --------------------------------------------------------------------------- inputs


class CostInputs(_Model):
    items: list[CostItem] = Field(default_factory=list)
    aids: list[CostItem] = Field(default_factory=list)
    duration_months: int = 48
    living: Literal["hostel", "independent"] = "hostel"
    city_basket_monthly: dict[str, float] = Field(default_factory=dict)
    loan: LoanTerms = Field(default_factory=LoanTerms)
    student_work: StudentWork | None = None
    family_income_annual: float | None = None
    p25_first_year_salary: float | None = None
    fx_rates: dict[str, float] = Field(default_factory=dict)
    ownership: str | None = None
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("duration_months")
    @classmethod
    def _duration(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("duration_months must be > 0")
        return value


# --------------------------------------------------------------------------- outputs


class LivingScenarioCost(_Model):
    name: str
    components: dict[str, MoneyBand]
    annual: MoneyBand
    total: MoneyBand
    modelled: bool


class DSRResult(_Model):
    value: float | None
    band: str
    formula: str
    first_year_emi: float | None
    p25_first_year_salary: float | None
    bands: dict[str, dict[str, float]]
    inputs: dict[str, Any] = Field(default_factory=dict)


class AffordabilityResult(_Model):
    value: float | None
    flagged: bool | None
    flag_above: float
    formula: str
    family_out_of_pocket: float
    annual_family_income: float | None


class CostSnapshot(_Model):
    currency: str
    duration_months: int
    living: str
    scenarios: dict[str, LivingScenarioCost]
    tcoa: MoneyBand
    tcoa_first_year: MoneyBand
    net_price: MoneyBand
    aid_total: MoneyBand
    loan_principal: MoneyBand
    family_out_of_pocket: MoneyBand
    student_work_offset: MoneyBand
    debt_at_repayment: MoneyBand
    emi: MoneyBand
    dsr25: DSRResult
    affordability_gap: AffordabilityResult
    fx_band_pct: float
    bands: dict[str, dict[str, float]]
    methods: dict[str, str]
    assumptions: list[str]
    display: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> dict[str, Any]:
        """JSON-ready representation; no recomputation required downstream."""

        return self.model_dump(mode="json")


# --------------------------------------------------------------------------- maths


def living_annual(
    items: Sequence[CostItem],
    components: Sequence[str],
    *,
    city_basket_monthly: Mapping[str, float],
    modelled_defaults: Mapping[str, Any],
    band_pct: float,
    currency: str,
) -> tuple[dict[str, MoneyBand], MoneyBand, bool]:
    """Annual living cost for one scenario; unpublished components use the basket."""

    by_category: dict[str, list[CostItem]] = {}
    for item in items:
        if item.internship_travel:
            continue
        by_category.setdefault(item.category, []).append(item)

    bands: dict[str, MoneyBand] = {}
    assumptions: list[str] = []
    modelled_any = False

    for component in components:
        matched = by_category.get(component, [])
        if matched:
            point = sum(_annualise(item.amount, item.period) for item in matched)
            low = sum(_annualise(item.amount * item.confidence, item.period) for item in matched)
            high = sum(
                _annualise(item.amount * (2.0 - item.confidence), item.period) for item in matched
            )
            assumptions.append(f"{component}: from published cost items")
            bands[component] = MoneyBand(
                value=point,
                lo=min(low, point),
                hi=max(high, point),
                currency=currency,
                method="published",
                assumptions=[f"n={len(matched)} cost items"],
            )
            continue

        monthly = city_basket_monthly.get(component)
        if monthly is None and component == "food_topup":
            default = modelled_defaults.get("food_topup_monthly") or {}
            monthly = float(default.get("amount", 0.0)) or None
        if monthly is None:
            assumptions.append(f"{component}: no published item and no city-basket entry")
            continue

        point = float(monthly) * 12.0
        modelled_any = True
        bands[component] = MoneyBand(
            value=point,
            lo=point * (1.0 - band_pct / 100.0),
            hi=point * (1.0 + band_pct / 100.0),
            currency=currency,
            method="modelled_city_basket",
            assumptions=[f"modelled from city basket, +/-{band_pct:g}%"],
        )

    annual_value = sum(band.value for band in bands.values())
    annual_lo = sum(band.lo for band in bands.values())
    annual_hi = sum(band.hi for band in bands.values())
    annual = MoneyBand(
        value=annual_value,
        lo=annual_lo,
        hi=annual_hi,
        currency=currency,
        method="modelled" if modelled_any else "published",
        assumptions=assumptions,
    )
    return bands, annual, modelled_any


def _annualise(amount: float, period: str) -> float:
    if period == "per_year":
        return float(amount)
    if period == "per_month":
        return float(amount) * 12.0
    return 0.0


def _totalise(amount: float, period: str, years: float) -> float:
    if period == "one_time":
        return float(amount)
    return _annualise(amount, period) * years


def emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """Equated monthly instalment at a contractual rate and tenure."""

    if principal <= 0:
        return 0.0
    if tenure_months <= 0:
        raise CostModelError("tenure_months must be > 0")
    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        return principal / tenure_months
    growth = (1.0 + monthly_rate) ** tenure_months
    return principal * monthly_rate * growth / (growth - 1.0)


def capitalised_debt(
    principal: float,
    annual_rate: float,
    moratorium_months: int,
    *,
    basis: str = "simple",
) -> float:
    """Principal plus interest capitalised over the moratorium."""

    if principal <= 0 or annual_rate <= 0 or moratorium_months <= 0:
        return max(principal, 0.0)
    if basis == "monthly_compound":
        return principal * (1.0 + annual_rate / 12.0) ** moratorium_months
    return principal * (1.0 + annual_rate * moratorium_months / 12.0)


def dsr_band(value: float | None, bands: Mapping[str, Mapping[str, float]]) -> str:
    """Classify DSR25 using config bands; ``None`` is ``unknown``, never 0."""

    if value is None:
        return "unknown"
    ordered = list(bands.items())
    for name, spec in ordered:
        maximum = spec.get("max")
        if maximum is not None and value <= float(maximum):
            return name
    if ordered:
        return ordered[-1][0]
    return "unknown"


def affordability_ratio(family_out_of_pocket: float, annual_income: float | None) -> float | None:
    if annual_income is None or annual_income <= 0:
        return None
    return family_out_of_pocket / annual_income


# --------------------------------------------------------------------------- compute


def _convert(
    amount: float,
    currency: str,
    *,
    base_currency: str,
    fx_rates: Mapping[str, float],
    fx_pct: float,
) -> tuple[float, float, float]:
    if currency.upper() == base_currency.upper():
        return amount, amount, amount
    rate = fx_rates.get(currency.upper()) or fx_rates.get(currency)
    if rate is None:
        raise MissingFXRate(
            f"no FX rate supplied for {currency!r}; refusing to store a point FX rate"
        )
    point = amount * rate
    band = fx_pct / 100.0
    return point, point * (1.0 - band), point * (1.0 + band)


def compute(inputs: CostInputs, rules: CostRules) -> CostSnapshot:
    """Build the complete, renderable cost snapshot."""

    currency = rules.base_currency
    years = inputs.duration_months / 12.0
    assumptions = list(inputs.assumptions)
    methods: dict[str, str] = {}

    # --- normalise every non-living item to base currency and both horizons
    non_living_total = 0.0
    non_living_total_lo = 0.0
    non_living_total_hi = 0.0
    non_living_first_year = 0.0
    non_living_first_lo = 0.0
    non_living_first_hi = 0.0
    scenario_living: dict[str, Any] = {}

    living_categories = {
        component for components in rules.living_scenarios.values() for component in components
    }

    for item in inputs.items:
        if item.internship_travel:
            pass
        elif item.category in living_categories:
            continue
        value, low, high = _convert(
            item.amount,
            item.currency or currency,
            base_currency=currency,
            fx_rates=inputs.fx_rates,
            fx_pct=rules.fx_sensitivity_pct,
        )
        confidence_band = 1.0 - item.confidence
        low *= 1.0 - confidence_band
        high *= 1.0 + confidence_band
        item_first_year = value if item.period == "one_time" else _annualise(value, item.period)
        item_first_lo = low if item.period == "one_time" else _annualise(low, item.period)
        item_first_hi = high if item.period == "one_time" else _annualise(high, item.period)
        total = _totalise(value, item.period, years)
        non_living_first_year += item_first_year
        non_living_first_lo += item_first_lo
        non_living_first_hi += item_first_hi
        non_living_total += total
        non_living_total_lo += _totalise(low, item.period, years)
        non_living_total_hi += _totalise(high, item.period, years)
        if item.period == "one_time":
            assumptions.append(f"one-time cost counted once: {item.label or item.category}")

    # --- aid (scholarships / waivers / reimbursement)
    aid_total = 0.0
    aid_lo = 0.0
    aid_hi = 0.0
    for aid in inputs.aids:
        value, low, high = _convert(
            aid.amount,
            aid.currency or currency,
            base_currency=currency,
            fx_rates=inputs.fx_rates,
            fx_pct=rules.fx_sensitivity_pct,
        )
        aid_total += _totalise(value, aid.period, years)
        aid_lo += _totalise(low, aid.period, years)
        aid_hi += _totalise(high, aid.period, years)

    # --- living scenarios, side by side
    scenario_models: dict[str, LivingScenarioCost] = {}
    for name, components in rules.living_scenarios.items():
        bands_map, annual, modelled = living_annual(
            inputs.items,
            components,
            city_basket_monthly=inputs.city_basket_monthly,
            modelled_defaults=rules.modelled_defaults,
            band_pct=rules.band_pct,
            currency=currency,
        )
        annual_lo = annual.lo
        annual_hi = annual.hi
        total_value = annual.value * years
        total_lo = annual_lo * years
        total_hi = annual_hi * years
        scenario_models[name] = LivingScenarioCost(
            name=name,
            components=bands_map,
            annual=annual,
            total=MoneyBand(
                value=total_value,
                lo=total_lo,
                hi=total_hi,
                currency=currency,
                method=annual.method,
                assumptions=list(annual.assumptions),
            ),
            modelled=modelled,
        )
        scenario_living[name] = (annual, total_value, total_lo, total_hi)

    if inputs.living not in scenario_models:
        available = ", ".join(sorted(scenario_models))
        raise CostModelError(f"living={inputs.living!r} is not a configured scenario ({available})")

    living_band, living_total, living_total_lo, living_total_hi = scenario_living[inputs.living]

    tcoa_value = non_living_total + living_total
    tcoa_lo = non_living_total_lo + living_total_lo
    tcoa_hi = non_living_total_hi + living_total_hi
    tcoa = MoneyBand(
        value=tcoa_value,
        lo=min(tcoa_lo, tcoa_value),
        hi=max(tcoa_hi, tcoa_value),
        currency=currency,
        method="modelled" if living_band.is_modelled else "computed",
        assumptions=[
            f"living={inputs.living}",
            "no opportunity cost included",
            "interval bounds are additive (comonotonic)",
        ],
    )
    first_year = non_living_first_year + living_band.value
    tcoa_first_year = MoneyBand(
        value=first_year,
        lo=min(first_year, non_living_first_lo + living_band.lo),
        hi=max(first_year, non_living_first_hi + living_band.hi),
        currency=currency,
        method=tcoa.method,
        assumptions=["first-year cash requirement (one-time costs included)"],
    )

    net_value = max(0.0, tcoa_value - aid_total)
    net_price = MoneyBand(
        value=net_value,
        lo=max(0.0, tcoa.lo - aid_hi),
        hi=max(0.0, tcoa.hi - aid_lo),
        currency=currency,
        method="computed" if aid_total == 0 else "computed_with_aid",
        assumptions=["TCOA minus scholarships/waivers/reimbursement"],
    )

    loan = inputs.loan
    principal = net_value if loan.principal is None else min(float(loan.principal), net_value)
    if loan.principal is None:
        assumptions.append("loan principal defaulted to full net price (no disbursement schedule)")
    principal_lo = min(principal, net_price.lo)
    principal_hi = min(principal, net_price.hi)
    loan_principal = MoneyBand(
        value=principal,
        lo=principal_lo,
        hi=principal_hi,
        currency=currency,
        method="computed",
        assumptions=["loan principal disbursed"],
    )

    moratorium_months = (
        loan.moratorium_months
        if loan.moratorium_months is not None
        else inputs.duration_months + 12
    )
    if loan.moratorium_months is None:
        assumptions.append("moratorium defaulted to course_plus_1y (IBA model scheme)")
    debt_value = capitalised_debt(
        principal, loan.annual_rate, moratorium_months, basis=loan.interest_basis
    )
    debt_lo = capitalised_debt(
        principal_lo, loan.annual_rate, moratorium_months, basis=loan.interest_basis
    )
    debt_hi = capitalised_debt(
        principal_hi, loan.annual_rate, moratorium_months, basis=loan.interest_basis
    )
    debt = MoneyBand(
        value=debt_value,
        lo=min(debt_lo, debt_value),
        hi=max(debt_hi, debt_value),
        currency=currency,
        method="computed_capitalised_moratorium_interest",
        assumptions=[
            f"{loan.interest_basis} interest over {moratorium_months} moratorium months",
        ],
    )

    emi_value = emi(debt_value, loan.annual_rate, loan.tenure_months)
    emi_band = MoneyBand(
        value=emi_value,
        lo=emi(debt_lo, loan.annual_rate, loan.tenure_months),
        hi=emi(debt_hi, loan.annual_rate, loan.tenure_months),
        currency=currency,
        method="computed_emi",
        assumptions=[f"tenure {loan.tenure_months} months at {loan.annual_rate:.4f}"],
    )

    # --- student work: out-of-pocket only, capped, never TCOA
    offset_value = 0.0
    offset_lo = 0.0
    offset_hi = 0.0
    work = inputs.student_work
    if work is not None and work.realistic_monthly > 0:
        work_value, work_lo, work_hi = _convert(
            work.realistic_monthly * 12.0 * years,
            work.currency or currency,
            base_currency=currency,
            fx_rates=inputs.fx_rates,
            fx_pct=rules.fx_sensitivity_pct,
        )
        cap = tcoa_value * work.cap_pct_of_tcoa / 100.0
        offset_value = max(0.0, min(work_value, cap))
        offset_lo = max(0.0, min(work_lo, cap))
        offset_hi = max(0.0, min(work_hi, cap))
        assumptions.append(
            f"{work.label}: capped at {work.cap_pct_of_tcoa:g}% of TCOA; "
            f"legal hours/week={work.legal_hours_per_week:g}"
        )
    offset = MoneyBand(
        value=offset_value,
        lo=offset_lo,
        hi=offset_hi,
        currency=currency,
        method="modelled_capped_offset",
        assumptions=["reduces family out-of-pocket only, never TCOA"],
    )

    oop_value = max(0.0, net_value - principal - offset_value)
    oop_lo = max(0.0, net_price.lo - principal_hi - offset_hi)
    oop_hi = max(0.0, net_price.hi - principal_lo - offset_lo)
    family_oop = MoneyBand(
        value=oop_value,
        lo=min(oop_lo, oop_value),
        hi=max(oop_hi, oop_value),
        currency=currency,
        method="computed",
        assumptions=["net price minus loan principal disbursed minus capped student-work offset"],
    )

    annuity = emi_value * 12.0
    dsr_value = (annuity / inputs.p25_first_year_salary) if inputs.p25_first_year_salary else None
    band_name = dsr_band(dsr_value, rules.dsr_bands)
    dsr = DSRResult(
        value=dsr_value,
        band=band_name,
        formula="first_year_emi / p25_first_year_gross_salary",
        first_year_emi=annuity if dsr_value is not None else None,
        p25_first_year_salary=inputs.p25_first_year_salary,
        bands={name: dict(spec) for name, spec in rules.dsr_bands.items()},
        inputs={
            "emi_monthly": emi_value,
            "emi_months_in_first_year": 12,
            "salary_source": "modelled_or_extracted_p25",
        },
    )

    gap_value = affordability_ratio(oop_value, inputs.family_income_annual)
    affordability = AffordabilityResult(
        value=gap_value,
        flagged=None if gap_value is None else gap_value > rules.affordability_flag_above,
        flag_above=rules.affordability_flag_above,
        formula="family_out_of_pocket / annual_family_income",
        family_out_of_pocket=oop_value,
        annual_family_income=inputs.family_income_annual,
    )
    if gap_value is not None:
        assumptions.append(
            "affordability gap compares whole-programme out-of-pocket with one year of income "
            "(brief §13.7)"
        )

    methods.update(
        {
            "tcoa": tcoa.method,
            "net_price": net_price.method,
            "family_out_of_pocket": family_oop.method,
            "debt_at_repayment": debt.method,
            "emi": emi_band.method,
            "dsr25": "computed_from_emi_and_p25",
            "affordability_gap": "computed_from_oop_and_income",
            "student_work_offset": offset.method,
        }
    )

    snapshot = CostSnapshot(
        currency=currency,
        duration_months=inputs.duration_months,
        living=inputs.living,
        scenarios=scenario_models,
        tcoa=tcoa,
        tcoa_first_year=tcoa_first_year,
        net_price=net_price,
        aid_total=MoneyBand(
            value=aid_total,
            lo=min(aid_lo, aid_total),
            hi=max(aid_hi, aid_total),
            currency=currency,
            method="published" if aid_total else "none",
            assumptions=["scholarships/waivers/reimbursement, counted once"],
        ),
        loan_principal=loan_principal,
        family_out_of_pocket=family_oop,
        student_work_offset=offset,
        debt_at_repayment=debt,
        emi=emi_band,
        dsr25=dsr,
        affordability_gap=affordability,
        fx_band_pct=rules.fx_sensitivity_pct,
        bands={name: dict(spec) for name, spec in rules.dsr_bands.items()},
        methods=methods,
        assumptions=assumptions,
    )
    snapshot.display = _display(snapshot)
    return snapshot


def _display(snapshot: CostSnapshot) -> dict[str, Any]:
    """Pre-rendered strings/cards so the UI never recomputes a number."""

    def money(band: MoneyBand) -> str:
        return f"{band.currency} {band.value:,.0f} ({band.lo:,.0f}-{band.hi:,.0f}, {band.method})"

    cards = []
    for name, scenario in sorted(snapshot.scenarios.items()):
        cards.append(
            {
                "scenario": name,
                "annual": money(scenario.annual),
                "total": money(scenario.total),
                "method": scenario.annual.method,
                "modelled": scenario.modelled,
            }
        )
    dsr_text = (
        "unknown (salary evidence missing)"
        if snapshot.dsr25.value is None
        else f"{snapshot.dsr25.value:.3f} ({snapshot.dsr25.band})"
    )
    gap_text = (
        "unknown (family income not supplied)"
        if snapshot.affordability_gap.value is None
        else f"{snapshot.affordability_gap.value:.2f}"
    )
    return {
        "tcoa": money(snapshot.tcoa),
        "first_year": money(snapshot.tcoa_first_year),
        "net_price": money(snapshot.net_price),
        "family_out_of_pocket": money(snapshot.family_out_of_pocket),
        "emi": money(snapshot.emi),
        "debt_at_repayment": money(snapshot.debt_at_repayment),
        "dsr25": dsr_text,
        "affordability_gap": gap_text,
        "living_cards": cards,
        "fx_band_pct": snapshot.fx_band_pct,
    }
