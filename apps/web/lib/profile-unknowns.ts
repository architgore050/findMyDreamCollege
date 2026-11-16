import { UNKNOWN, type Profile } from "./types";

/**
 * Which ranking inputs the profile cannot answer.
 *
 * A ranking still runs with an incomplete profile — the missing inputs become
 * UNKNOWN intervals, never 0 and never 50 (brief §3.5). This list exists so the
 * UI can say *which* inputs widened the bands instead of pretending the profile
 * was complete.
 */

export interface UnknownInput {
  /** Stable key, e.g. `marks.pcb_pct`. */
  key: string;
  /** Human label for the screen. */
  label: string;
}

function unanswered(value: unknown): boolean {
  if (value === undefined || value === null) return true;
  if (value === UNKNOWN) return true;
  if (typeof value === "string") return value.trim() === "";
  if (Array.isArray(value)) return value.length === 0;
  return false;
}

export function unknownProfileInputs(profile: Profile): UnknownInput[] {
  const unknown: UnknownInput[] = [];

  const check = (key: string, label: string, value: unknown) => {
    if (unanswered(value)) unknown.push({ key, label });
  };

  check("target_intake_year", "target intake year", profile.target_intake_year);
  check("target_field", "target field", profile.target_field);
  check("credential_levels", "credential level", profile.credential_levels);
  check("marks.class12_aggregate_pct", "class 12 aggregate", profile.marks?.class12_aggregate_pct);
  check("marks.pcb_pct", "PCB aggregate", profile.marks?.pcb_pct);
  check("english_test", "English test", profile.english_test?.status ?? profile.english_test);
  check("entrance_exams", "entrance exams", profile.entrance_exams);
  check(
    "budget",
    "budget ceiling",
    profile.budget?.annual_ceiling ?? profile.budget?.total_ceiling,
  );
  check("travel_for_exams", "travel tolerance", profile.travel_for_exams);
  check("regions", "preferred regions", profile.regions);
  check("living_preference", "living preference", profile.living_preference);
  check("risk_appetite", "risk appetite", profile.risk_appetite);

  return unknown;
}

export function hasUnknownInputs(profile: Profile): boolean {
  return unknownProfileInputs(profile).length > 0;
}
