import type { Profile } from "@/lib/types";

/**
 * Form draft: every value is a string so "not set" and "unknown" are distinct
 * and typed entry never fights the schema. Conversion to `Profile` happens only
 * on save, where zod validates it.
 */
export interface ExamDraft {
  exam: string;
  score: string;
  status: string;
}

export interface ProfileDraft {
  nationality: string;
  target_intake_year: string;
  target_field: string;
  credential_levels: string[];
  marks_board: string;
  marks_class12: string;
  marks_pcb: string;
  english_type: string;
  english_score: string;
  english_status: string;
  entrance_exams: ExamDraft[];
  budget_currency: string;
  budget_annual: string;
  budget_total: string;
  loan_required: string;
  family_income_band: string;
  collateral_available: string;
  living_preference: string[];
  travel_for_exams: string;
  regions: string;
  extracurriculars: string;
  leadership: string;
  clinical_exposure: string;
  risk_appetite: string;
}

export function emptyDraft(): ProfileDraft {
  return {
    nationality: "",
    target_intake_year: "",
    target_field: "",
    credential_levels: [],
    marks_board: "",
    marks_class12: "",
    marks_pcb: "",
    english_type: "",
    english_score: "",
    english_status: "",
    entrance_exams: [],
    budget_currency: "",
    budget_annual: "",
    budget_total: "",
    loan_required: "",
    family_income_band: "",
    collateral_available: "",
    living_preference: [],
    travel_for_exams: "",
    regions: "",
    extracurriculars: "",
    leadership: "",
    clinical_exposure: "",
    risk_appetite: "",
  };
}

function asDraftValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

function num(value: string): number | "unknown" | undefined {
  if (value === "") return undefined;
  if (value === "unknown") return "unknown";
  return Number(value);
}

function text(value: string): string | "unknown" | undefined {
  const trimmed = value.trim();
  if (trimmed === "") return undefined;
  if (trimmed === "unknown") return "unknown";
  return trimmed;
}

export function fromProfile(profile: Profile): ProfileDraft {
  const base = emptyDraft();
  return {
    ...base,
    nationality: asDraftValue(profile.nationality),
    target_intake_year: asDraftValue(profile.target_intake_year),
    target_field: asDraftValue(profile.target_field),
    credential_levels: profile.credential_levels ?? [],
    marks_board: profile.marks?.board ?? "",
    marks_class12: asDraftValue(profile.marks?.class12_aggregate_pct),
    marks_pcb: asDraftValue(profile.marks?.pcb_pct),
    english_type: asDraftValue(profile.english_test?.type),
    english_score: asDraftValue(profile.english_test?.score),
    english_status: profile.english_test?.status ?? "",
    entrance_exams: (profile.entrance_exams ?? []).map((exam) => ({
      exam: exam.exam,
      score: asDraftValue(exam.score),
      status: exam.status ?? "",
    })),
    budget_currency: profile.budget?.currency ?? "",
    budget_annual: asDraftValue(profile.budget?.annual_ceiling),
    budget_total: asDraftValue(profile.budget?.total_ceiling),
    loan_required: asDraftValue(profile.financing?.loan_required),
    family_income_band: profile.financing?.family_income_band ?? "",
    collateral_available: asDraftValue(profile.financing?.collateral_available),
    living_preference: profile.living_preference ?? [],
    travel_for_exams: profile.travel_for_exams ?? "",
    regions: (profile.regions ?? []).join(", "),
    extracurriculars: asDraftValue(profile.qualitative?.extracurriculars),
    leadership: asDraftValue(profile.qualitative?.leadership),
    clinical_exposure: asDraftValue(profile.qualitative?.clinical_exposure),
    risk_appetite: profile.risk_appetite ?? "",
  };
}

/** Build a candidate profile. zod decides whether it is valid. */
export function toProfile(draft: ProfileDraft): Profile {
  const marksSet = draft.marks_board !== "" || draft.marks_class12 !== "" || draft.marks_pcb !== "";
  const englishSet =
    draft.english_type !== "" || draft.english_score !== "" || draft.english_status !== "";
  const budgetSet =
    draft.budget_currency !== "" || draft.budget_annual !== "" || draft.budget_total !== "";
  const financingSet =
    draft.loan_required !== "" ||
    draft.family_income_band !== "" ||
    draft.collateral_available !== "";
  const qualitativeSet =
    draft.extracurriculars !== "" || draft.leadership !== "" || draft.clinical_exposure !== "";

  const profile: Profile = {
    ...(draft.nationality === ""
      ? {}
      : {
          nationality:
            draft.nationality === "unknown" ? "unknown" : draft.nationality.toUpperCase(),
        }),
    ...(draft.target_intake_year === ""
      ? {}
      : { target_intake_year: num(draft.target_intake_year) }),
    ...(draft.target_field === "" ? {} : { target_field: text(draft.target_field) }),
    ...(draft.credential_levels.length > 0
      ? {
          credential_levels: draft.credential_levels.filter(
            (value): value is "degree" | "diploma" | "lateral_entry" | "certificate" =>
              value === "degree" ||
              value === "diploma" ||
              value === "lateral_entry" ||
              value === "certificate",
          ),
        }
      : {}),
    ...(marksSet
      ? {
          marks: {
            ...(draft.marks_board.trim() === "" ? {} : { board: draft.marks_board.trim() }),
            ...(draft.marks_class12 === ""
              ? {}
              : { class12_aggregate_pct: num(draft.marks_class12) }),
            ...(draft.marks_pcb === "" ? {} : { pcb_pct: num(draft.marks_pcb) }),
          },
        }
      : {}),
    ...(englishSet
      ? {
          english_test: {
            ...(draft.english_type === ""
              ? {}
              : {
                  type: draft.english_type as
                    "IELTS" | "TOEFL" | "PTE" | "Duolingo" | "other" | "unknown",
                }),
            ...(draft.english_score === "" ? {} : { score: num(draft.english_score) }),
            ...(draft.english_status === ""
              ? {}
              : {
                  status: draft.english_status as "taken" | "planned" | "not_taken" | "unknown",
                }),
          },
        }
      : {}),
    ...(draft.entrance_exams.some((exam) => exam.exam.trim() !== "")
      ? {
          entrance_exams: draft.entrance_exams
            .filter((exam) => exam.exam.trim() !== "")
            .map((exam) => ({
              exam: exam.exam.trim(),
              ...(exam.score === "" ? {} : { score: num(exam.score) }),
              ...(exam.status === ""
                ? {}
                : { status: exam.status as "taken" | "planned" | "not_taken" | "unknown" }),
            })),
        }
      : {}),
    ...(budgetSet
      ? {
          budget: {
            ...(draft.budget_currency.trim() === ""
              ? {}
              : { currency: draft.budget_currency.trim().toUpperCase() }),
            ...(draft.budget_annual === "" ? {} : { annual_ceiling: num(draft.budget_annual) }),
            ...(draft.budget_total === "" ? {} : { total_ceiling: num(draft.budget_total) }),
          },
        }
      : {}),
    ...(financingSet
      ? {
          financing: {
            ...(draft.loan_required === ""
              ? {}
              : {
                  loan_required:
                    draft.loan_required === "unknown"
                      ? ("unknown" as const)
                      : draft.loan_required === "yes",
                }),
            ...(draft.family_income_band === ""
              ? {}
              : {
                  family_income_band: draft.family_income_band as
                    "below_4.5L" | "4.5-8L" | "8-15L" | "15L_plus" | "unknown",
                }),
            ...(draft.collateral_available === ""
              ? {}
              : {
                  collateral_available:
                    draft.collateral_available === "unknown"
                      ? ("unknown" as const)
                      : draft.collateral_available === "yes",
                }),
          },
        }
      : {}),
    ...(draft.living_preference.length > 0
      ? {
          living_preference: draft.living_preference.filter(
            (value): value is "hostel" | "independent" | "unknown" =>
              value === "hostel" || value === "independent" || value === "unknown",
          ),
        }
      : {}),
    ...(draft.travel_for_exams === ""
      ? {}
      : {
          travel_for_exams: draft.travel_for_exams as "yes" | "no" | "unknown",
        }),
    ...(draft.regions.trim() === ""
      ? {}
      : {
          regions: draft.regions
            .split(",")
            .map((region) => region.trim())
            .filter((region) => region.length > 0),
        }),
    ...(qualitativeSet
      ? {
          qualitative: {
            ...(draft.extracurriculars === ""
              ? {}
              : { extracurriculars: num(draft.extracurriculars) }),
            ...(draft.leadership === "" ? {} : { leadership: num(draft.leadership) }),
            ...(draft.clinical_exposure === ""
              ? {}
              : { clinical_exposure: num(draft.clinical_exposure) }),
          },
        }
      : {}),
    ...(draft.risk_appetite === ""
      ? {}
      : {
          risk_appetite: draft.risk_appetite as
            "must_be_safe" | "estimated_ok" | "speculative_ok" | "unknown",
        }),
  };
  return profile;
}

export function profileIsEmpty(profile: Profile): boolean {
  return Object.keys(profile).length === 0;
}
