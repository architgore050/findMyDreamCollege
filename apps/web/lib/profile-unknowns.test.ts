import { describe, expect, it } from "vitest";

import { unknownProfileInputs } from "@/lib/profile-unknowns";
import type { Profile } from "@/lib/types";

describe("unknownProfileInputs", () => {
  it("reports every ranking input as unknown for an empty profile", () => {
    const unknowns = unknownProfileInputs({});
    const keys = unknowns.map((input) => input.key);

    expect(keys).toContain("target_intake_year");
    expect(keys).toContain("marks.pcb_pct");
    expect(keys).toContain("budget");
    expect(keys).toContain("travel_for_exams");
    expect(unknowns.every((input) => input.label.length > 0)).toBe(true);
  });

  it("treats the literal 'unknown' and empty arrays as unanswered", () => {
    const profile: Profile = {
      target_intake_year: "unknown",
      credential_levels: [],
      marks: { pcb_pct: "unknown" },
      regions: ["Delhi"],
    };

    const keys = unknownProfileInputs(profile).map((input) => input.key);

    expect(keys).toContain("target_intake_year");
    expect(keys).toContain("credential_levels");
    expect(keys).toContain("marks.pcb_pct");
    expect(keys).not.toContain("regions");
  });

  it("returns nothing when the ranking inputs are answered", () => {
    const profile: Profile = {
      target_intake_year: 2027,
      target_field: "Medical Laboratory Technology",
      credential_levels: ["degree"],
      marks: { class12_aggregate_pct: 81, pcb_pct: 78 },
      english_test: { type: "IELTS", status: "taken", score: 7 },
      entrance_exams: [{ exam: "NEET-UG", status: "taken", score: 540 }],
      budget: { currency: "INR", annual_ceiling: 250000 },
      travel_for_exams: "yes",
      regions: ["Maharashtra"],
      living_preference: ["hostel"],
      risk_appetite: "estimated_ok",
    };

    expect(unknownProfileInputs(profile)).toEqual([]);
  });
});
