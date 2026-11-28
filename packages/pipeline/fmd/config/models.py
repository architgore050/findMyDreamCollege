"""Typed models for ``config/*.yaml``.

These models are deliberately permissive about fields the brief does not fix
(``extra="allow"``) but strict about the fields the pipeline and API depend on.
Cross-file checks live in :mod:`fmd.config.loader`.
"""

from __future__ import annotations

import re
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
    min_components_for_composite: int = 2
    single_claim_interval_half_width: float = 25.0
    peer_groups: PeerGroups
    evidence_tiers: dict[str, EvidenceTier]
    transforms: dict[str, Any] = Field(default_factory=dict)
    filters: list[str]
    filter_policy: FilterPolicy
    gaming_flags: dict[str, float] = Field(default_factory=dict)
    salary: SalaryRules = Field(default_factory=SalaryRules)
    col: dict[str, Any] = Field(default_factory=dict)
    insights: InsightsRules = Field(default_factory=InsightsRules)


# --------------------------------------------------------------------------- predicates

PREDICATE_UNIT_KINDS = frozenset(
    {
        "money",
        "count",
        "hours",
        "months",
        "years",
        "percent",
        "ratio",
        "grade",
        "boolean",
        "date",
        "text",
    }
)

PREDICATE_SCORING_KINDS = frozenset(
    {
        "percent",
        "presence",
        "transform",
        "saturating",
        "decay",
        "cutoff_margin",
        "eligibility_margin",
        "cost_band",
    }
)


class PredicateScoring(ConfigModel):
    """How one predicate becomes a 0..100 component.

    The parameters are registry data, never inferred from the claim value.
    """

    kind: str
    transform: str | None = None
    scale: float = 1.0
    factor: float = 1.0

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in PREDICATE_SCORING_KINDS:
            raise ValueError(
                f"unknown scoring kind {value!r}; known kinds: "
                + ", ".join(sorted(PREDICATE_SCORING_KINDS))
            )
        return value

    @field_validator("scale")
    @classmethod
    def _positive_scale(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("scale must be > 0")
        return value


class PredicateRule(ConfigModel):
    """One row of ``config/predicates.yaml``."""

    component: str | None = None
    label: str | None = None
    unit_kind: str | None = None
    requires_denominator: bool = False
    requires_period: bool = False
    scoring: PredicateScoring | None = None
    notes: str | None = None
    description: str | None = None

    @field_validator("unit_kind")
    @classmethod
    def _known_unit_kind(cls, value: str | None) -> str | None:
        if value is not None and value not in PREDICATE_UNIT_KINDS:
            raise ValueError(
                f"unknown unit_kind {value!r}; known kinds: "
                + ", ".join(sorted(PREDICATE_UNIT_KINDS))
            )
        return value


class PredicatesConfig(ConfigModel):
    """Typed view of the ninth registry file, ``config/predicates.yaml``."""

    version: str
    predicates: dict[str, PredicateRule]
    aliases: dict[str, str] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _version_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("version must be non-empty")
        return value


# ------------------------------------------------------------------------ programmes


class ProgrammeDefinition(ConfigModel):
    """The conceptual target-field definition the classifier receives.

    Deliberately prose + domain list, never a name list: this is the authority
    the semantic tier judges against (14 §B2). ``version`` is part of the
    semantic cache key, so editing the definition invalidates cached decisions.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: str
    target_field: str
    statement: str
    domains: list[str] = Field(default_factory=list)

    @field_validator("version", "target_field", "statement")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value


class ProgrammeClassifierConfig(ConfigModel):
    """Tier-2 semantic classifier settings (router purpose, batching, cache)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    purpose: str
    prompt_version: str
    bundle_size: int = 20
    cache: bool = True

    @field_validator("purpose", "prompt_version")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value

    @field_validator("bundle_size")
    @classmethod
    def _positive_bundle(cls, value: int) -> int:
        if value < 1:
            raise ValueError("bundle_size must be >= 1")
        return value


class CountryTaxonomy(ConfigModel):
    """One country's Tier-1 rule block.

    Patterns are regexes matched case-insensitively against the normalised
    programme name. ``exclude`` beats ``include``. An empty block is a valid
    stub: it declares the country and defers every name to the semantic tier.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    aliases: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def _declared_patterns_are_safe(cls, values: list[str]) -> list[str]:
        """Reject empty declared lists and patterns that admit everything.

        An *absent* key (an empty country stub) is valid and keeps its default
        empty list; only a key the file actually declares is checked. A pattern
        that matches both the empty string and ``MBBS`` is a typo in the
        taxonomy (``.*``, ``^.*$``) and would silently admit every programme,
        so it is a config error rather than a rule.
        """
        if not values:
            raise ValueError(
                "must not be empty when declared; omit the key for an empty country "
                "stub (e.g. `AU: {}`)"
            )
        for pattern in values:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error:
                # The loader reports invalid regexes with the country key.
                continue
            if compiled.search("") and compiled.search("MBBS"):
                raise ValueError(
                    f"catch-all pattern {pattern!r} matches every name (including an "
                    "empty string and 'MBBS'); write an explicit rule instead"
                )
        return values


class ProgrammesConfig(ConfigModel):
    """Typed view of the tenth registry file, ``config/programmes.yaml``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: str
    definition: ProgrammeDefinition
    classifier: ProgrammeClassifierConfig
    countries: dict[str, CountryTaxonomy]
    default: CountryTaxonomy

    @field_validator("version")
    @classmethod
    def _version_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("version must be non-empty")
        return value

    def taxonomy_for(self, country: str | None) -> tuple[CountryTaxonomy, bool]:
        """The rule block for ``country`` plus whether it fell back to default.

        A declared key (even an empty stub) returns its own block; an unknown
        code or a NULL country returns ``default`` and reports the fallback so
        the decision can carry the ``country_taxonomy_missing`` flag.
        """
        if country and (taxonomy := self.countries.get(country.strip().upper())) is not None:
            return taxonomy, False
        return self.default, True


# --------------------------------------------------------------------------- models


class ProviderConfig(ConfigModel):
    kind: Literal["openai_compatible", "anthropic", "gemini", "replay"]
    base_url: str | None = None
    api_key: str | None = None
    fixtures_dir: str | None = None

    # Self-documenting diagnostics. ``fmd llm doctor`` names these exact
    # variables instead of guessing which ``${...}`` produced a value.
    base_url_env: str | None = None
    api_key_env: str | None = None
    requires_api_key: bool = False
    supports_guided_json: bool = True
    base_url_example: str | None = None


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

    # Diagnostics: the env vars this profile resolves from, and whether a
    # missing/broken profile blocks a run or degrades to ``degrade_reason``.
    provider_env: str | None = None
    model_env: str | None = None
    model_example: str | None = None
    required: bool = True
    degrade_reason: str | None = None
    # Registry-driven escape hatch for profiles that must never silently change
    # provider (for example the free/local-only verdict_live eval). When false,
    # the router re-raises the original provider error instead of consulting the
    # global fallback table.
    allow_fallback: bool = True

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


# --------------------------------------------------------------------------- acquisition

AcquireGranularity = Literal["glyph", "span", "block"]


class AcquireTierThresholds(ConfigModel):
    """Routing thresholds for the acquisition tiers. No Python defaults."""

    text_min_chars_per_page: int
    table_min_rows: int
    table_min_numeric_columns: int
    devanagari_forces_t2: bool
    t1_table_confidence_min: float
    t2_indic_self_check_min_ratio: float
    adjudication_sample_rate: float


class AcquireAnomalyThresholds(ConfigModel):
    """Repetition-loop / entropy-collapse thresholds (never constants in code)."""

    block_length_multiplier: float
    entropy_ratio_floor: float
    min_block_chars: int
    min_page_chars: int
    repeated_block_min_occurrences: int


class AcquireBackend(ConfigModel):
    tool: str
    model: str
    granularity: AcquireGranularity
    endpoint_env: str | None = None


class AcquireBBoxPolicy(ConfigModel):
    page_epsilon: float
    gate_tiers: list[str] = Field(default_factory=list)


class AcquireBatchConfig(ConfigModel):
    """Batched acquisition (W10): one CUDA page-batch per server restart.

    Validated strictly (unknown keys rejected in the loader): a typo here must
    fail config load, not silently change how a 1,000-page backlog is cut.
    """

    model_config = ConfigDict(extra="forbid")

    #: Target converted pages per cycle. Sizing is evaluated only at document
    #: boundaries: the document that crosses the limit is always converted in
    #: full and extracted in the same cycle, and no further document starts.
    #: Default 100 — break-even measured at ~20–25 pages/cycle (CUDA
    #: 0.166–0.9 s/page vs CPU 1.0–3.8 s/page, ~20–25 s per server restart);
    #: 100 leaves margin.
    pages: int = 100
    #: Docling device for the acquisition phase (cpu|auto|cuda). The batched
    #: runner stops the model server first, so cuda is the measured default;
    #: `auto` defers to Docling's own detection.
    device: str = "cuda"
    #: Server-control script (start|stop|status|probe contract). Unset or
    #: missing on the host ⇒ degraded mode: CPU acquisition + inline
    #: extraction, recorded in the summary — never a failed run by itself.
    server_script: str | None = None
    #: Leave the model server running after the batched run ends. Default off:
    #: every exit path stops the server (try/finally).
    keep_server: bool = False
    #: /health readiness poll ceiling for `wait_ready` after a server start.
    poll_timeout_s: int = 600
    #: Double buffering (wave v1-w4): while a cycle's extraction phase keeps
    #: the GPU busy, this many next-batch documents' bytes are read off-thread
    #: (object store only — never a converter or the model server). 0 disables
    #: the prefetch worker; the acquire loop then reads live as before.
    prefetch_documents: int = 4
    #: Automatic phase control for ``fmd acquire`` (FMD-CLOUD-030): when a
    #: server controller is available, acquisition runs inside an automatic
    #: stop-convert-restart GPU window (still zero LLM calls); without one it
    #: degrades to CPU with a recorded reason. Disable for a strictly
    #: no-server-touching acquire run.
    phased_acquire: bool = True
    #: Automatic phased route for the worker's extract stage: when the batch
    #: config and a server controller exist, the worker drives the batched
    #: stop→acquire→start→extract cycles instead of the fused CPU loop — no
    #: manual server stop/start. ``FMD_PHASED_EXTRACT=off`` overrides per shell.
    phased_extract: bool = True


class AcquireCacheConfig(ConfigModel):
    """Parse-cache payload schema version (W10).

    Must agree with ``fmd.acquire.parse_cache.PARSE_SCHEMA_VERSION``; the
    loader cross-checks the two so the YAML and the code cannot drift apart
    silently. Bumping it invalidates every stored parse.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1


class AcquireConfig(ConfigModel):
    """Typed view of ``config/acquire.yaml`` (brief §9 text layer)."""

    version: str
    tiers: AcquireTierThresholds
    anomaly: AcquireAnomalyThresholds
    backends: dict[str, AcquireBackend]
    bbox: AcquireBBoxPolicy
    batch: AcquireBatchConfig | None = None
    cache: AcquireCacheConfig | None = None


# --------------------------------------------------------------------------- planning


class PlanPhaseAConfig(ConfigModel):
    """Pre-fetch acceptance/abstention thresholds (URL/anchor/extension only)."""

    min_score: float
    margin: float
    acquire_floor: float


class PlanPhaseBConfig(ConfigModel):
    """Post-acquisition text sampling and acceptance thresholds."""

    text_sample_chars: int
    max_headings: int
    min_score: float
    margin: float


class PlanConfig(ConfigModel):
    """Typed view of ``config/plan.yaml`` (two-phase planning)."""

    version: str
    phase_a: PlanPhaseAConfig
    phase_b: PlanPhaseBConfig


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
