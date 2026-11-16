import { describe, expect, it } from "vitest";

import { fromProfile, toProfile } from "@/lib/profile-draft";
import { parseStoredProfile, profileSchema, PROFILE_STORAGE_KEY } from "@/lib/types";

describe("profile schema", () => {
  it("accepts an empty profile and inserts no defaults", () => {
    const parsed = profileSchema.parse({});
    expect(parsed).toEqual({});
    expect(PROFILE_STORAGE_KEY).toBe("fmd.profile.v1");
    expect("target_intake_year" in parsed).toBe(false);
    expect("budget" in parsed).toBe(false);
    expect("nationality" in parsed).toBe(false);
  });

  it("accepts 'unknown' as a first-class value everywhere it applies", () => {
    const parsed = profileSchema.parse({
      nationality: "unknown",
      target_intake_year: "unknown",
      target_field: "unknown",
      marks: { class12_aggregate_pct: "unknown", pcb_pct: "unknown" },
      english_test: { type: "unknown", score: "unknown", status: "unknown" },
      entrance_exams: [{ exam: "MHT_CET", score: "unknown", status: "unknown" }],
      budget: { annual_ceiling: "unknown", total_ceiling: "unknown" },
      financing: { loan_required: "unknown", collateral_available: "unknown" },
      living_preference: ["unknown"],
      travel_for_exams: "unknown",
      qualitative: { leadership: "unknown" },
      risk_appetite: "unknown",
    });

    expect(parsed.target_intake_year).toBe("unknown");
    expect(parsed.marks?.class12_aggregate_pct).toBe("unknown");
    expect(parsed.travel_for_exams).toBe("unknown");
  });

  it("rejects impossible marks rather than clamping them", () => {
    const result = profileSchema.safeParse({ marks: { pcb_pct: 150 } });
    expect(result.success).toBe(false);
  });

  it("degrades corrupt localStorage content to an empty profile", () => {
    expect(parseStoredProfile(null)).toEqual({});
    expect(parseStoredProfile("{not json")).toEqual({});
    expect(parseStoredProfile(JSON.stringify({ target_intake_year: "soon" }))).toEqual({});
  });
});

describe("profile draft round-trip", () => {
  it("keeps unset distinct from unknown", () => {
    const empty = toProfile(fromProfile({}));
    expect(empty).toEqual({});

    const withUnknown = toProfile({
      ...fromProfile({}),
      target_intake_year: "unknown",
      marks_pcb: "unknown",
    });
    expect(withUnknown.target_intake_year).toBe("unknown");
    expect(withUnknown.marks?.pcb_pct).toBe("unknown");
    expect(withUnknown.marks?.class12_aggregate_pct).toBeUndefined();
  });

  it("builds a profile that the schema accepts", () => {
    const draft = fromProfile({
      target_intake_year: 2027,
      target_field: "mlt",
      credential_levels: ["degree"],
      marks: { class12_aggregate_pct: 78 },
      regions: ["MH", "KA"],
    });
    const profile = toProfile(draft);
    expect(profileSchema.safeParse(profile).success).toBe(true);
    expect(profile.regions).toEqual(["MH", "KA"]);
  });
});
