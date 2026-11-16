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

export const subScoreSchema = z.object({
  point: z.number().nullable().optional(),
  lo: z.number().nullable().optional(),
  hi: z.number().nullable().optional(),
  n_claims: z.number().nullable().optional(),
  n_sources: z.number().nullable().optional(),
  evidence_tier: evidenceTierSchema.optional(),
});
export type SubScore = z.infer<typeof subScoreSchema>;

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

// --------------------------------------------------------------------------- costs

export const costCategorySchema = z.enum([
  "tuition",
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

export const sourcesPayloadSchema = z.object({
  items: z.array(sourceSchema),
  exclusions: z.array(sourceExclusionSchema),
});
export type SourcesPayload = z.infer<typeof sourcesPayloadSchema>;

export const footnotes = {
  provenance: "Every published number must resolve to claim → extraction → snapshot → URL.",
} as const;
