import { z } from "zod";

/**
 * Shared domain types.
 *
 * Two rules from the brief are encoded here rather than left to components:
 * - UNKNOWN is a value, never 0 and never 50 (brief §3.5).
 * - Profile fields are all optional; nothing is pre-filled (brief §3.1, §16).
 */

export const UNKNOWN = "unknown" as const;
export type Unknown = typeof UNKNOWN;

/** Wrap a schema so `"unknown"` is always an accepted, first-class value. */
const maybeUnknown = <T extends z.ZodTypeAny>(schema: T) => z.union([schema, z.literal(UNKNOWN)]);

// --------------------------------------------------------------------------- profile

export const credentialLevelSchema = z.enum(["degree", "diploma", "lateral_entry", "certificate"]);
export type CredentialLevel = z.infer<typeof credentialLevelSchema>;

export const englishTestTypeSchema = maybeUnknown(
  z.enum(["IELTS", "TOEFL", "PTE", "Duolingo", "other"]),
);
export const testStatusSchema = z.enum(["taken", "planned", "not_taken", "unknown"]);
export const travelPreferenceSchema = z.enum(["yes", "no", "unknown"]);
export const livingPreferenceSchema = z.enum(["hostel", "independent", "unknown"]);
export const riskAppetiteSchema = z.enum([
  "must_be_safe",
  "estimated_ok",
  "speculative_ok",
  "unknown",
]);
export const incomeBandSchema = z.enum(["below_4.5L", "4.5-8L", "8-15L", "15L_plus", "unknown"]);

export const marksSchema = z.object({
  board: z.string().min(1).optional(),
  class12_aggregate_pct: maybeUnknown(z.number().min(0).max(100)).optional(),
  pcb_pct: maybeUnknown(z.number().min(0).max(100)).optional(),
});

export const englishTestSchema = z.object({
  type: englishTestTypeSchema.optional(),
  score: maybeUnknown(z.number().nonnegative()).optional(),
  status: testStatusSchema.optional(),
});

export const entranceExamSchema = z.object({
  exam: z.string().min(1),
  score: maybeUnknown(z.number().nonnegative()).optional(),
  status: testStatusSchema.optional(),
});

export const budgetSchema = z.object({
  currency: z.string().min(1).optional(),
  annual_ceiling: maybeUnknown(z.number().nonnegative()).optional(),
  total_ceiling: maybeUnknown(z.number().nonnegative()).optional(),
});

export const financingSchema = z.object({
  loan_required: maybeUnknown(z.boolean()).optional(),
  family_income_band: incomeBandSchema.optional(),
  collateral_available: maybeUnknown(z.boolean()).optional(),
});

const selfRating = maybeUnknown(z.number().int().min(1).max(5));

export const qualitativeSchema = z.object({
  extracurriculars: selfRating.optional(),
  leadership: selfRating.optional(),
  clinical_exposure: selfRating.optional(),
});

/**
 * Brief §16. Every field is optional and every scalar accepts `"unknown"`
 * (rendered as "prefer not to say"). No defaults are supplied anywhere: an
 * empty object is a valid profile.
 */
export const profileSchema = z.object({
  nationality: maybeUnknown(z.string().length(2)).optional(),
  target_intake_year: maybeUnknown(z.number().int().min(2000).max(2100)).optional(),
  target_field: maybeUnknown(z.string().min(1)).optional(),
  credential_levels: z.array(credentialLevelSchema).optional(),
  marks: marksSchema.optional(),
  english_test: englishTestSchema.optional(),
  entrance_exams: z.array(entranceExamSchema).optional(),
  budget: budgetSchema.optional(),
  financing: financingSchema.optional(),
  living_preference: z.array(livingPreferenceSchema).optional(),
  travel_for_exams: travelPreferenceSchema.optional(),
  regions: z.array(z.string().min(1)).optional(),
  qualitative: qualitativeSchema.optional(),
  risk_appetite: riskAppetiteSchema.optional(),
});

export type Profile = z.infer<typeof profileSchema>;
export type Marks = z.infer<typeof marksSchema>;
export type EntranceExam = z.infer<typeof entranceExamSchema>;

export const PROFILE_STORAGE_KEY = "fmd.profile.v1";

/** Parse untrusted localStorage content. Invalid data degrades to an empty profile. */
export function parseStoredProfile(raw: string | null): Profile {
  if (!raw) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    const result = profileSchema.safeParse(parsed);
    return result.success ? result.data : {};
  } catch {
    return {};
  }
}

// --------------------------------------------------------------------------- API envelope

export const apiMetaSchema = z.object({
  scorer_version: z.string(),
  weights_version: z.string(),
  claims_hash: z.string(),
  generated_at: z.string(),
});
export type ApiMeta = z.infer<typeof apiMetaSchema>;

export const apiErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  workstream: z.string().optional(),
});
export type ApiErrorBody = z.infer<typeof apiErrorSchema>;

// --------------------------------------------------------------------------- programme

export const recognitionStatusSchema = z.enum([
  "recognised",
  "conditional",
  "unrecognised",
  "unknown",
]);
export type RecognitionStatus = z.infer<typeof recognitionStatusSchema>;

export const programmeSchema = z.object({
  id: z.number(),
  institution_id: z.number(),
  name: z.string(),
  credential_level: credentialLevelSchema,
  duration_months: z.number().nullable().optional(),
  internship_months: z.number().nullable().optional(),
  affiliated_university_id: z.number().nullable().optional(),
  recognition_status: recognitionStatusSchema,
  seats: z.number().nullable().optional(),
  medium: z.string().nullable().optional(),
  created_at: z.string().nullable().optional(),
});
export type Programme = z.infer<typeof programmeSchema>;

export const entitySchema = z.object({
  id: z.number(),
  kind: z.enum(["institution", "programme", "hospital", "exam", "lender", "city"]),
  canonical_name: z.string(),
  country: z.string().nullable().optional(),
  state: z.string().nullable().optional(),
  city: z.string().nullable().optional(),
  status: z.string(),
  created_at: z.string().nullable().optional(),
});
export type Entity = z.infer<typeof entitySchema>;

// --------------------------------------------------------------------------- claims

export const claimMethodSchema = z.enum(["extracted", "derived", "modelled", "user_provided"]);
export type ClaimMethod = z.infer<typeof claimMethodSchema>;

export const claimSchema = z.object({
  id: z.number(),
  extraction_id: z.number(),
  entity_id: z.number(),
  predicate: z.string(),
  value: z.unknown(),
  unit: z.string().nullable().optional(),
  qualifier: z.record(z.string(), z.unknown()).nullable().optional(),
  quote: z.string(),
  char_start: z.number(),
  char_end: z.number(),
  confidence: z.number(),
  observed_at: z.string().nullable().optional(),
  method: claimMethodSchema,
  superseded_by: z.number().nullable().optional(),
  // Present on the /evidence endpoint only.
  url: z.string().nullable().optional(),
  fetched_at: z.string().nullable().optional(),
  sha256: z.string().nullable().optional(),
  snapshot_id: z.number().optional(),
});
export type Claim = z.infer<typeof claimSchema>;

export const claimEvidencePageSchema = z.object({
  items: z.array(claimSchema),
  total: z.number(),
  limit: z.number(),
  offset: z.number(),
});
export type ClaimEvidencePage = z.infer<typeof claimEvidencePageSchema>;

export const evidenceTierSchema = z.enum(["well_evidenced", "partial", "thin", "insufficient"]);
export type EvidenceTier = z.infer<typeof evidenceTierSchema>;

export const componentScoreSchema = z.object({
  name: z.string(),
  label: z.string(),
  value: z.unknown(),
  score: z.number(),
  lo: z.number(),
  hi: z.number(),
  method: z.string(),
  claim_id: z.union([z.number(), z.string()]).nullable().optional(),
  source_id: z.union([z.number(), z.string()]).nullable().optional(),
});
export type ComponentScore = z.infer<typeof componentScoreSchema>;

export const subScoreSchema = z.object({
  point: z.number().nullable().optional(),
  lo: z.number().nullable().optional(),
  hi: z.number().nullable().optional(),
  n_claims: z.number().nullable().optional(),
  n_sources: z.number().nullable().optional(),
  evidence_tier: evidenceTierSchema.optional(),
  measured: z.number().nullable().optional(),
  prior_adjusted: z.number().nullable().optional(),
  components: z.array(componentScoreSchema).optional(),
  unknown_reason: z.string().nullable().optional(),
  modelled_share: z.number().optional(),
  widened: z.boolean().optional(),
  shrinkage_inactive: z.boolean().optional(),
});
export type SubScore = z.infer<typeof subScoreSchema>;

export const compositeSchema = z.object({
  point: z.number().nullable(),
  lo: z.number(),
  hi: z.number(),
  bound: z.string(),
  lo_independent: z.number().optional(),
  hi_independent: z.number().optional(),
  lo_comonotonic: z.number().optional(),
  hi_comonotonic: z.number().optional(),
  evidence_tier: evidenceTierSchema.optional(),
  known_weight: z.number().optional(),
  total_weight: z.number().optional(),
  unknown_sub_scores: z.array(z.string()).optional(),
});
export type Composite = z.infer<typeof compositeSchema>;

/** `/api/rankings` composite band. A band, never a decimal rank (brief §11). */
export const compositeBandSchema = z.object({
  point: z.number().nullable(),
  lo: z.number(),
  hi: z.number(),
  method: z.string(),
  bound: z.string(),
});
export type CompositeBand = z.infer<typeof compositeBandSchema>;

export const filterResultSchema = z.enum(["pass", "fail", "unknown"]);
export type FilterResult = z.infer<typeof filterResultSchema>;

export const filterDecisionSchema = z.object({
  filter_id: z.string(),
  result: filterResultSchema,
  claim_id: z.union([z.number(), z.string()]).nullable(),
  reason: z.string(),
  observed_at: z.string().nullable().optional(),
  stale: z.boolean().optional(),
  detail: z.record(z.string(), z.unknown()).optional(),
});
export type FilterDecision = z.infer<typeof filterDecisionSchema>;

export const filterSummarySchema = z.object({
  order: z.array(z.string()),
  counts: z.object({
    pass: z.number(),
    fail: z.number(),
    unknown: z.number(),
  }),
  excluded: z.array(z.string()),
  unverified: z.array(z.string()),
  decisions: z.record(z.string(), filterDecisionSchema),
});
export type FilterSummary = z.infer<typeof filterSummarySchema>;

export const insightSchema = z.object({
  id: z.string(),
  predicate: z.string(),
  text: z.string(),
  coverage: z.array(z.string()),
  claim_ids: z.array(z.union([z.number(), z.string()])),
  kind: z.string(),
  detail: z.record(z.string(), z.unknown()),
});
export type Insight = z.infer<typeof insightSchema>;

export const insightSelectionSchema = z.object({
  selected: z.array(insightSchema),
  mandatory: z.array(insightSchema),
  coverage: z.number(),
  covered_units: z.array(z.string()),
  uncovered_units: z.array(z.string()),
  stopped_reason: z.string(),
});
export type InsightSelection = z.infer<typeof insightSelectionSchema>;

export const scoreResultSchema = z.object({
  entity_id: z.number().nullable(),
  scorer_version: z.string(),
  weights_version: z.string(),
  claims_hash: z.string(),
  sub_scores: z.record(z.string(), subScoreSchema),
  composite: compositeSchema,
  evidence_tier: evidenceTierSchema,
  computed_at: z.string(),
  insights: insightSelectionSchema.nullable().optional(),
  gaming_flags: z.array(z.record(z.string(), z.unknown())).optional(),
  assumptions: z.array(z.string()).optional(),
});
export type ScoreResult = z.infer<typeof scoreResultSchema>;

export const scoreSchema = z.object({
  id: z.number(),
  entity_id: z.number(),
  scorer_version: z.string(),
  weights_version: z.string(),
  claims_hash: z.string(),
  sub_scores: z.record(z.string(), z.unknown()),
  evidence_tier: evidenceTierSchema,
  composite: z.number().nullable().optional(),
  composite_lo: z.number().nullable().optional(),
  composite_hi: z.number().nullable().optional(),
  computed_at: z.string().nullable().optional(),
});
export type Score = z.infer<typeof scoreSchema>;

// --------------------------------------------------------------------------- cost snapshot

/** `MoneyBand` from the cost model. `render()` keeps model fields only. */
export const moneyBandSchema = z.object({
  value: z.number().nullable(),
  lo: z.number().nullable(),
  hi: z.number().nullable(),
  currency: z.string(),
  method: z.string(),
  assumptions: z.array(z.string()),
});
export type MoneyBand = z.infer<typeof moneyBandSchema>;

export const livingScenarioCostSchema = z.object({
  name: z.string(),
  components: z.record(z.string(), moneyBandSchema),
  annual: moneyBandSchema,
  total: moneyBandSchema,
  modelled: z.boolean(),
});
export type LivingScenarioCost = z.infer<typeof livingScenarioCostSchema>;

export const dsr25Schema = z.object({
  value: z.number().nullable(),
  band: z.string(),
  formula: z.string(),
  first_year_emi: z.number().nullable(),
  p25_first_year_salary: z.number().nullable(),
  bands: z.record(z.string(), z.record(z.string(), z.number())),
  inputs: z.record(z.string(), z.unknown()).optional(),
});
export type DSR25 = z.infer<typeof dsr25Schema>;

export const affordabilitySchema = z.object({
  value: z.number().nullable(),
  flagged: z.boolean().nullable(),
  flag_above: z.number(),
  formula: z.string(),
  family_out_of_pocket: z.number(),
  annual_family_income: z.number().nullable(),
});
export type Affordability = z.infer<typeof affordabilitySchema>;

/**
 * Per-living-scenario financing block from `POST /api/programmes/{id}/cost`.
 * Each scenario carries its own TCOA, EMI, DSR₂₅ and affordability gap; the
 * blocks are complete alternatives and are never summed.
 */
export const scenarioTotalsSchema = z.object({
  living: z.enum(["hostel", "independent"]),
  living_cost: livingScenarioCostSchema,
  tcoa: moneyBandSchema,
  tcoa_first_year: moneyBandSchema,
  net_price: moneyBandSchema,
  aid_total: moneyBandSchema.optional(),
  loan_principal: moneyBandSchema,
  family_out_of_pocket: moneyBandSchema,
  student_work_offset: moneyBandSchema.optional(),
  debt_at_repayment: moneyBandSchema,
  emi: moneyBandSchema,
  dsr25: dsr25Schema,
  affordability_gap: affordabilitySchema,
  methods: z.record(z.string(), z.string()).optional(),
  assumptions: z.array(z.string()).optional(),
  assumed: z.record(z.string(), z.unknown()).optional(),
  display: z.record(z.string(), z.unknown()).optional(),
});
export type ScenarioTotals = z.infer<typeof scenarioTotalsSchema>;

/**
 * `POST /api/programmes/{id}/cost`. Both living scenarios are returned in one
 * payload and are complete alternatives: never summed (brief §13).
 */
export const costPayloadSchema = z.object({
  programme_id: z.number(),
  living: z.enum(["hostel", "independent"]),
  available: z.boolean(),
  needs_profile: z.boolean().optional(),
  profile_persisted: z.boolean().optional(),
  reason: z.string().optional(),
  note: z.string().optional(),
  non_additive_scenarios: z.boolean().optional(),
  /** True flags for values the model assumed rather than extracted. */
  assumed: z.record(z.string(), z.unknown()).optional(),
  /** Machine-readable reasons a metric stayed UNKNOWN instead of defaulting. */
  unknown_reasons: z.array(z.string()).optional(),
  currency: z.string().optional(),
  duration_months: z.number().optional(),
  scenarios: z.record(z.string(), livingScenarioCostSchema).default({}),
  /** Full financing metrics per scenario, including DSR₂₅ each. */
  scenario_totals: z.record(z.string(), scenarioTotalsSchema).default({}),
  tcoa: moneyBandSchema.optional(),
  tcoa_first_year: moneyBandSchema.optional(),
  net_price: moneyBandSchema.optional(),
  aid_total: moneyBandSchema.optional(),
  loan_principal: moneyBandSchema.optional(),
  family_out_of_pocket: moneyBandSchema.optional(),
  student_work_offset: moneyBandSchema.optional(),
  debt_at_repayment: moneyBandSchema.optional(),
  emi: moneyBandSchema.optional(),
  dsr25: dsr25Schema.optional(),
  affordability_gap: affordabilitySchema.optional(),
  fx_band_pct: z.number().optional(),
  bands: z.record(z.string(), z.record(z.string(), z.number())).optional(),
  dsr25_bands: z.record(z.string(), z.record(z.string(), z.number())).optional(),
  methods: z.record(z.string(), z.string()).optional(),
  assumptions: z.array(z.string()).optional(),
  assumptions_flags: z.record(z.string(), z.unknown()).optional(),
  display: z.record(z.string(), z.unknown()).optional(),
});
export type CostPayload = z.infer<typeof costPayloadSchema>;

/** `POST /api/programmes/{id}/filters`: all eight hard filters, profile in body. */
export const programmeFiltersPayloadSchema = z.object({
  programme_id: z.number().optional(),
  needs_profile: z.boolean().optional(),
  profile_persisted: z.boolean().optional(),
  order: z.array(z.string()).default([]),
  counts: z.object({ pass: z.number(), fail: z.number(), unknown: z.number() }).optional(),
  filters: z.array(filterDecisionSchema).default([]),
  would_exclude: z.boolean().default(false),
  excluded: z.array(z.string()).optional(),
  unverified: z.array(z.string()).optional(),
  reason: z.string().optional(),
});
export type ProgrammeFiltersPayload = z.infer<typeof programmeFiltersPayloadSchema>;

// --------------------------------------------------------------------------- costs

export const costCategorySchema = z.enum([
  "hostel",
  "mess",
  "rent",
  "utilities",
  "food",
  "transport",
  "equipment",
  "exam_fee",
  "visa",
  "insurance",
  "misc",
]);
export type CostCategory = z.infer<typeof costCategorySchema>;

/** Postgres `numeric` arrives as a JSON string from the API. */
const decimalLike = z.union([z.number(), z.string()]);

export const costItemSchema = z.object({
  id: z.number(),
  programme_id: z.number(),
  category: costCategorySchema,
  label: z.string(),
  amount: decimalLike,
  currency: z.string(),
  period: z.enum(["one_time", "per_year", "per_month"]),
  basis: z.string().nullable().optional(),
  source_claim_id: z.number().nullable().optional(),
  confidence: z.number().nullable().optional(),
});
export type CostItem = z.infer<typeof costItemSchema>;

export const costScenarioSchema = z.object({
  living: z.enum(["hostel", "independent"]),
  items: z.array(costItemSchema),
  total: decimalLike.nullable(),
  currency: z.string().nullable(),
  modelled: z.boolean(),
});
export type CostScenario = z.infer<typeof costScenarioSchema>;

// --------------------------------------------------------------------------- exams

export const examSchema = z.object({
  id: z.string(),
  name: z.string(),
  country: z.string(),
  scope: z.enum(["national", "state", "institution"]),
  provider: z.string(),
  registration_opens: z.string().nullable().optional(),
  registration_closes: z.string().nullable().optional(),
  earliest_sitting: z.string().nullable().optional(),
  latest_sitting: z.string().nullable().optional(),
  requires_travel: z.boolean(),
  remote_options: z.boolean(),
  fee_amount: decimalLike.nullable().optional(),
  currency: z.string().nullable().optional(),
  source_claim_id: z.number().nullable().optional(),
});
export type Exam = z.infer<typeof examSchema>;

// --------------------------------------------------------------------------- sources

export const sourceSchema = z.object({
  id: z.string(),
  name: z.string(),
  base_url: z.string().nullable().optional(),
  kind: z.string(),
  format: z.string(),
  cadence_days: z.number(),
  trust: z.string(),
  adapter: z.string(),
  enabled: z.boolean(),
  requires_manual_review: z.boolean().optional(),
  notes: z.string().nullable().optional(),
  last_crawl: z.string().nullable().optional(),
});
export type Source = z.infer<typeof sourceSchema>;

export const sourceExclusionSchema = z.object({
  id: z.string(),
  name: z.string(),
  reason: z.string(),
});
export type SourceExclusion = z.infer<typeof sourceExclusionSchema>;

// --------------------------------------------------------------------------- response payloads

export const healthSchema = z.object({
  status: z.string(),
  version: z.string(),
  database: z.string(),
  config: z
    .object({
      config_dir: z.string(),
      enabled_countries: z.array(z.string()),
      sources_enabled: z.number(),
    })
    .optional(),
  config_error: z.string().optional(),
});

export const programmeListSchema = z.object({
  items: z.array(programmeSchema),
  total: z.number(),
  limit: z.number(),
  offset: z.number(),
  facets: z.record(z.string(), z.record(z.string(), z.number())).optional(),
  exclusions: z
    .object({
      available: z.boolean(),
      owner: z.string().optional(),
    })
    .optional(),
});
export type ProgrammeList = z.infer<typeof programmeListSchema>;

export const programmeDetailSchema = z.object({
  programme: programmeSchema,
  institution: entitySchema.nullable(),
  claims: z.array(claimSchema),
  claims_by_predicate: z.record(z.string(), z.array(claimSchema)).optional(),
  costs: z.array(costItemSchema),
  exams: z.array(examSchema),
  score: scoreSchema.nullable(),
});
export type ProgrammeDetail = z.infer<typeof programmeDetailSchema>;

// --------------------------------------------------------------------------- WS-F requests

/** Profile travels in the body; nothing else identifies the user. */
export const profileRequestSchema = z.object({
  profile: profileSchema,
  limit: z.number().int().min(1).max(500).optional(),
});
export type ProfileRequest = z.infer<typeof profileRequestSchema>;

export const rankingsRequestSchema = z.object({
  profile: profileSchema,
  limit: z.number().int().min(1).max(500).optional(),
});
export type RankingsRequest = z.infer<typeof rankingsRequestSchema>;

/**
 * Exact `POST /api/programmes/{id}/cost` body. `living` selects which scenario
 * the top-level totals mirror; both scenarios always come back.
 */
export const programmeCostRequestSchema = z.object({
  profile: profileSchema,
  living: z.enum(["hostel", "independent"]),
});
export type ProgrammeCostRequest = z.infer<typeof programmeCostRequestSchema>;

/** Exact `POST /api/programmes/{id}/filters` body. Profile only, never in URL. */
export const programmeFiltersRequestSchema = z.object({
  profile: profileSchema,
});
export type ProgrammeFiltersRequest = z.infer<typeof programmeFiltersRequestSchema>;

// --------------------------------------------------------------------------- WS-F payloads

export const rankingRowSchema = z.object({
  programme: programmeSchema,
  score: scoreResultSchema,
  composite_band: compositeBandSchema,
  evidence_tier: evidenceTierSchema,
  filters: filterSummarySchema,
  excluded: z.boolean(),
  insights: insightSelectionSchema.nullable(),
  cost: z.record(z.string(), z.unknown()).nullable().optional(),
});
export type RankingRow = z.infer<typeof rankingRowSchema>;

export const rankingsPayloadSchema = z.object({
  items: z.array(rankingRowSchema),
  total: z.number(),
  weights: z.record(z.string(), z.number()),
  profile_persisted: z.boolean(),
});
export type RankingsPayload = z.infer<typeof rankingsPayloadSchema>;

export const examStatusSchema = z.enum(["open", "closed", "unverified"]);
export type ExamStatus = z.infer<typeof examStatusSchema>;

export const examWindowSchema = z.object({
  exam_id: z.string(),
  name: z.string(),
  country: z.string(),
  status: examStatusSchema,
  observed_at: z.string().nullable(),
  stale: z.boolean(),
  ttl_days: z.number(),
  requires_travel: z.boolean(),
  remote_options: z.boolean(),
  travel_blocked: z.boolean(),
  deciding_claim_id: z.union([z.number(), z.string()]).nullable(),
  reason: z.string(),
});
export type ExamWindow = z.infer<typeof examWindowSchema>;

export const examWindowsPayloadSchema = z.object({
  items: z.array(examWindowSchema),
  counts: z.object({
    open: z.number(),
    closed: z.number(),
    unverified: z.number(),
  }),
  profile_persisted: z.boolean(),
});
export type ExamWindowsPayload = z.infer<typeof examWindowsPayloadSchema>;

export const excludedProgrammeSchema = z.object({
  programme: programmeSchema,
  deciding_filters: z.array(filterDecisionSchema),
  deciding_claim_ids: z.array(z.union([z.number(), z.string()])),
});
export type ExcludedProgramme = z.infer<typeof excludedProgrammeSchema>;

export const exclusionsPayloadSchema = z.object({
  histogram: z.record(z.string(), z.number()),
  unknown_histogram: z.record(z.string(), z.number()),
  excluded: z.array(excludedProgrammeSchema),
  excluded_total: z.number(),
  evaluated_total: z.number(),
  profile_persisted: z.boolean(),
});
export type ExclusionsPayload = z.infer<typeof exclusionsPayloadSchema>;

export const dominanceVerdictSchema = z.enum(["better", "worse", "not_comparable"]);
export type DominanceVerdict = z.infer<typeof dominanceVerdictSchema>;

export const paretoCellSchema = z.object({
  undominated: z.boolean(),
  better_than: z.array(z.string()),
  worse_than: z.array(z.string()),
  not_comparable: z.array(z.string()),
});
export type ParetoCell = z.infer<typeof paretoCellSchema>;

const claimIdMapSchema = z.record(z.string(), z.array(z.union([z.number(), z.string()])));

export const comparisonSchema = z.object({
  verdict: dominanceVerdictSchema,
  differing_sub_scores: z.array(z.string()),
  differing_claims: z.object({
    left: claimIdMapSchema,
    right: claimIdMapSchema,
  }),
});
export type Comparison = z.infer<typeof comparisonSchema>;

export const comparePayloadSchema = z.object({
  ids: z.array(z.number()),
  matrix: z.record(z.string(), paretoCellSchema),
  comparisons: z.record(z.string(), comparisonSchema),
  not_comparable_pairs: z.array(z.string()),
});
export type ComparePayload = z.infer<typeof comparePayloadSchema>;

export const sourcesPayloadSchema = z.object({
  items: z.array(sourceSchema),
  exclusions: z.array(sourceExclusionSchema),
});
export type SourcesPayload = z.infer<typeof sourcesPayloadSchema>;

export const footnotes = {
  provenance: "Every published number must resolve to claim → extraction → snapshot → URL.",
} as const;
