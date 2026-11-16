"""Typed models for ``config/*.yaml``.

These models are deliberately permissive about fields the brief does not fix
(``extra="allow"``) but strict about the fields the pipeline and API depend on.
Cross-file checks live in :mod:`fmd.config.loader`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fmd.er.config import ERConfig as ErConfig  # noqa: F401  (re-exported typed model)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# --------------------------------------------------------------------------- countries


class Capabilities(ConfigModel):
    programme_inventory: bool | str | None = None
    regulator: bool | str | None = None
    recognition_chain: bool | str | None = None
    admissions_calendar: bool | str | None = None
    fee_disclosure: bool | str | None = None
    col_basket: bool | str | None = None
    licensure_gate: bool | str | None = None
    student_work_rights: bool | str | None = None
    outcome_data: bool | str | None = None


class Country(ConfigModel):
    name: str
    enabled: bool = False
    currency: str
    languages: list[str] = Field(default_factory=list)
    tier: str = "C"
    capabilities: Capabilities | None = None


class CountriesConfig(ConfigModel):
    countries: dict[str, Country]

    @property
    def enabled(self) -> dict[str, Country]:
        return {code: c for code, c in self.countries.items() if c.enabled}


# --------------------------------------------------------------------------- sources

Trust = Literal["high", "medium", "low"]


class DiscoveryParams(ConfigModel):
    """Structured discovery parameters (registry data, never Python constants).

    ``domain_suffixes`` and ``pdf_url_patterns`` drive the Common Crawl URL-index
    sweep; ``query_templates`` drive SearXNG. Suffixes are bare eTLDs without a
    leading dot; PDF patterns are SQL ``LIKE`` patterns.

    The remaining fields are the per-source parameters adapters read instead of
    hardcoding URLs, query-parameter names or field lists (invariant 7). They are
    optional so a source only declares the block it needs.
    """

    index_glob: str | None = None
    domain_suffixes: list[str] = Field(default_factory=list)
    pdf_url_patterns: list[str] = Field(default_factory=list)
    query_templates: list[str] = Field(default_factory=list)

    # Listing / index pages and the API endpoints behind them.
    api_endpoint: str | None = None
    listing_urls: list[str] = Field(default_factory=list)
    listing_url: str | None = None
    ajax_url: str | None = None
    asset_url_template: str | None = None
    static_xlsx_url: str | None = None

    # UGC JSON XHR.
    unitype_param: str | None = None
    unitype_ids: list[int] = Field(default_factory=list)
    field_whitelist: list[str] = Field(default_factory=list)

    # Listing filters (regex). Anchors are kept only when both match.
    pdf_href_pattern: str | None = None
    link_text_pattern: str | None = None
    report_link_pattern: str | None = None
    date_pattern: str | None = None

    # NABH admin-ajax request shape.
    action_hospitals: str | None = None
    action_states: str | None = None
    state_param: str | None = None
    page_param: str | None = None
    request_method: str | None = None
    states: list[str] = Field(default_factory=list)
    max_pages: int | None = None

    # AISHE is disabled until a human confirms the static asset path.
    verified: bool = False

    @field_validator("domain_suffixes")
    @classmethod
    def _normalise_suffixes(cls, values: list[str]) -> list[str]:
        normalised: list[str] = []
        for value in values:
            suffix = value.strip().lower().lstrip(".")
            if not suffix:
                raise ValueError("domain suffix must not be blank")
            if "/" in suffix or ":" in suffix:
                raise ValueError(f"domain suffix {value!r} must be a bare eTLD, not a host")
            normalised.append(suffix)
        return normalised

    @field_validator("pdf_url_patterns", "query_templates")
    @classmethod
    def _non_blank(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.strip():
                raise ValueError("must not contain a blank entry")
        return values


# Params each discovery adapter needs before it can run. A discovery source
# missing them is a config error, not a silent empty crawl.
DISCOVERY_PARAMS_REQUIRED: dict[str, tuple[str, ...]] = {
    "common_crawl": ("index_glob", "domain_suffixes", "pdf_url_patterns"),
    "searxng": ("query_templates",),
}


class Source(ConfigModel):
    id: str
    name: str
    base_url: str | None = None
    kind: str
    format: str
    cadence_days: int = Field(gt=0)
    trust: Trust
    adapter: str | None = None
    enabled: bool = False
    notes: str | None = None
    requires_manual_review: bool = False
    seeds: list[str] = Field(default_factory=list)
    params: DiscoveryParams | None = None

    # Audit fields for source-level traps (spec §3, §4).
    pii_fields: list[str] = Field(default_factory=list)
    verify_first: bool = False
    abstain: bool = False
    scanned: bool = False
    level: int | None = None

    @field_validator("id", "name", "kind", "format", "adapter")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("pii_fields")
    @classmethod
    def _pii_fields_not_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("must not contain a blank field name")
        return values


class DenylistedDomain(ConfigModel):
    """A host that must never originate a recognition claim (spec §4.2.2)."""

    domain: str
    reason: str

    @field_validator("domain", "reason")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip().lower() if value is not None else value


class SourceExclusion(ConfigModel):
    id: str
    name: str
    reason: str


class SourcesConfig(ConfigModel):
    sources: list[Source] = Field(default_factory=list)
    exclusions: list[SourceExclusion] = Field(default_factory=list)
    denylisted_domains: list[DenylistedDomain] = Field(default_factory=list)

    @property
    def enabled(self) -> list[Source]:
        return [s for s in self.sources if s.enabled]

    def by_id(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def denied_domains(self) -> list[str]:
        return [entry.domain for entry in self.denylisted_domains]


# --------------------------------------------------------------------------- weights


class PeerGroups(ConfigModel):
    keys: list[str]
    min_group_size: int = 0
    fallback: str = "country_prior"


class EvidenceTier(ConfigModel):
    min_sources: int
    min_claims: int
    max_modelled_share: float


class FilterPolicy(ConfigModel):
    unknown_blocks_ranking: bool = False
    stale_exam_window_ttl_days: int = 3
    budget_uses: str = "tcoa_lo"


class SalaryRules(ConfigModel):
    preferred_statistic: str = "median"
    target_quantile: float = 0.25
    min_n_for_empirical_quantile: int = 10


class InsightsRules(ConfigModel):
    method: str = "greedy_max_coverage"
    max_insights: int = 5
    require_predicate_diversity: bool = True
    always_include: list[str] = Field(default_factory=list)
    min_marginal_gain: float = 0.0


class WeightsConfig(ConfigModel):
    weights_version: str
    scorer_version: str
    weights: dict[str, float]
    min_claims_for_score: int = 1
    min_claims_for_confidence: int = 1
    peer_groups: PeerGroups
    evidence_tiers: dict[str, EvidenceTier]
    transforms: dict[str, Any] = Field(default_factory=dict)
    filters: list[str]
    filter_policy: FilterPolicy
    gaming_flags: dict[str, float] = Field(default_factory=dict)
    salary: SalaryRules = Field(default_factory=SalaryRules)
    col: dict[str, Any] = Field(default_factory=dict)
    insights: InsightsRules = Field(default_factory=InsightsRules)


# --------------------------------------------------------------------------- models


class ProviderConfig(ConfigModel):
    kind: Literal["openai_compatible", "replay"]
    base_url: str | None = None
    api_key: str | None = None
    fixtures_dir: str | None = None


class Pricing(ConfigModel):
    """Optional per-million-token pricing.

    ``config/models.yaml`` does not publish prices (v1 forbids paid APIs), so
    absent pricing means cost 0.0. The budget code still runs and is exercised
    with explicit pricing in tests.
    """

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class ProfileConfig(ConfigModel):
    provider: str
    model: str | None = None
    temperature: float = 0.0
    guided_json: bool = False
    max_tokens: int = 4096
    pricing: Pricing | None = None

    @property
    def resolved(self) -> bool:
        return bool(self.model)


class BudgetsConfig(ConfigModel):
    daily_usd: float = 0.0
    per_run_usd: float = 0.0
    hard_stop: bool = True
    cache: bool = True

    @field_validator("daily_usd", "per_run_usd")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("budget must be >= 0")
        return value


class FallbackRule(ConfigModel):
    from_provider: str = Field(alias="from")
    to_provider: str = Field(alias="to")
    on: list[str] = Field(default_factory=list)


class ModelsConfig(ConfigModel):
    providers: dict[str, ProviderConfig]
    profiles: dict[str, ProfileConfig]
    budgets: BudgetsConfig
    routing_rules: list[dict[str, Any]] = Field(default_factory=list)
    fallbacks: list[FallbackRule] = Field(default_factory=list)


# --------------------------------------------------------------------------- exams


class Registration(ConfigModel):
    opens: str | None = None
    closes: str | None = None
    sitting_dates: Any | None = None


class Exam(ConfigModel):
    id: str
    name: str
    scope: str
    country: str
    provider: str
    state: str | list[str] | None = None
    used_for_mlt: bool | str = False
    requires_exam: bool = True
    requires_travel: bool = True
    remote_options: bool = False
    fee_amount: float | None = None
    currency: str | None = None
    notes: str | None = None
    registration: Registration | None = None


class LanguageTest(ConfigModel):
    id: str
    name: str
    requires_travel: bool = True
    remote_options: bool = False


class ExamsConfig(ConfigModel):
    exams: list[Exam]
    language_tests: list[LanguageTest] = Field(default_factory=list)
    past_papers_policy: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- cost rules


class CurrencyRules(ConfigModel):
    base: str = "INR"
    fx_sensitivity_pct: float = 10.0
    as_of_required: bool = True


class LivingScenario(ConfigModel):
    components: list[str] = Field(default_factory=list)
    notes: str | None = None


class StudentWorkRule(ConfigModel):
    legal_hours_per_week: float = 0.0
    cap_pct_of_tcoa: float = 0.0
    label: str = "informal_earnings"


class DSR25Rules(ConfigModel):
    formula: str
    horizon_years: int = 1
    bands: dict[str, dict[str, float]] = Field(default_factory=dict)
    always_publish_inputs: bool = True


class AffordabilityGap(ConfigModel):
    formula: str
    flag_above: float = 1.0


class FinancingScope(ConfigModel):
    schemes: list[str] = Field(default_factory=list)
    notes: str | None = None
    fx_band_required: bool = False


class CostRulesConfig(ConfigModel):
    currency: CurrencyRules
    cost_categories: list[str]
    living_scenarios: dict[str, LivingScenario]
    modelled_defaults: dict[str, Any] = Field(default_factory=dict)
    student_work: dict[str, Any] = Field(default_factory=dict)
    dsr25: DSR25Rules
    affordability_gap: AffordabilityGap
    financing_scopes: dict[str, FinancingScope] = Field(default_factory=dict)
    loan_defaults: dict[str, Any] = Field(default_factory=dict)
    financing_labeling: dict[str, Any] = Field(default_factory=dict)
