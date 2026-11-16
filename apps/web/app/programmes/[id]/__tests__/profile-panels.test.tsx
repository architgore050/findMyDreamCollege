import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ProgrammeProfilePanels } from "../profile-panels";
import { PROFILE_STORAGE_KEY, type Profile } from "@/lib/types";

/**
 * The API is always mocked. These tests pin the exact POST bodies the UI sends
 * (profile in the body only) and check that assumed/unknown values are shown
 * honestly rather than replaced by a client-side default.
 */

const meta = {
  scorer_version: "score-1",
  weights_version: "weights-1",
  claims_hash: "a".repeat(64),
  generated_at: "2026-11-16T09:00:00Z",
};

const profile: Profile = {
  target_intake_year: 2027,
  credential_levels: ["degree"],
  marks: { class12_aggregate_pct: 78 },
  financing: { loan_required: true, family_income_band: "4.5-8L" },
};

function band(value: number | null, method = "computed") {
  return {
    value,
    lo: value === null ? null : value * 0.9,
    hi: value === null ? null : value * 1.1,
    currency: "INR",
    method,
    assumptions: [],
  };
}

function livingCost(name: string) {
  return {
    name,
    components: {},
    annual: band(120000, "modelled_city_basket"),
    total: band(480000, "modelled_city_basket"),
    modelled: true,
  };
}

function totals(living: "hostel" | "independent") {
  const unknownRate = living === "independent";
  return {
    living,
    living_cost: livingCost(living),
    tcoa: band(living === "hostel" ? 420000 : 480000),
    tcoa_first_year: band(180000),
    net_price: band(420000),
    loan_principal: band(420000),
    family_out_of_pocket: band(210000),
    debt_at_repayment: band(unknownRate ? null : 460000, unknownRate ? "unknown_rate" : "computed"),
    emi: band(unknownRate ? null : 4200, unknownRate ? "unknown_rate" : "computed_emi"),
    dsr25: {
      value: unknownRate ? null : 0.168,
      band: unknownRate ? "unknown" : "comfortable",
      formula: "first_year_emi / p25_first_year_gross_salary",
      first_year_emi: unknownRate ? null : 50400,
      p25_first_year_salary: 300000,
      bands: {},
      inputs: {},
    },
    affordability_gap: {
      value: null,
      flagged: null,
      flag_above: 1,
      formula: "family_out_of_pocket / annual_family_income",
      family_out_of_pocket: 210000,
      annual_family_income: null,
    },
    methods: {},
    assumptions: [],
    assumed: {},
  };
}

function costPayload() {
  return {
    programme_id: 7,
    living: "hostel",
    available: true,
    needs_profile: false,
    profile_persisted: false,
    non_additive_scenarios: true,
    note: "hostel and independent are complete alternatives; never sum them",
    currency: "INR",
    duration_months: 48,
    scenarios: { hostel: livingCost("hostel"), independent: livingCost("independent") },
    scenario_totals: { hostel: totals("hostel"), independent: totals("independent") },
    assumed: {
      loan_rate: {
        assumed: true,
        source: "assumption",
        note: "modelled assumption, not a published scheme rate",
      },
      loan_terms: {
        assumed: true,
        source: "assumption",
        note: "tenure and moratorium are modelled defaults, not published sanction terms",
      },
    },
    unknown_reasons: [
      "income: the profile states a family-income band, not a point value; affordability_gap stays UNKNOWN rather than inventing a midpoint",
    ],
    tcoa: band(420000),
    emi: band(4200),
    dsr25: {
      value: 0.168,
      band: "comfortable",
      formula: "first_year_emi / p25_first_year_gross_salary",
      first_year_emi: 50400,
      p25_first_year_salary: 300000,
      bands: {},
      inputs: {},
    },
    affordability_gap: {
      value: null,
      flagged: null,
      flag_above: 1,
      formula: "family_out_of_pocket / annual_family_income",
      family_out_of_pocket: 210000,
      annual_family_income: null,
    },
  };
}

const FILTER_IDS = [
  "credential_level",
  "recognition_status",
  "eligibility",
  "language",
  "exam_registration_open",
  "exam_travel",
  "intake_year_available",
  "budget_ceiling",
] as const;

function filtersPayload() {
  return {
    programme_id: 7,
    needs_profile: false,
    profile_persisted: false,
    order: [...FILTER_IDS],
    counts: { pass: 5, fail: 1, unknown: 2 },
    filters: FILTER_IDS.map((filterId) =>
      filterId === "budget_ceiling"
        ? {
            filter_id: filterId,
            result: "fail" as const,
            claim_id: 42,
            reason: "TCOA lower bound 420000 exceeds ceiling 400000",
            observed_at: "2026-11-01T00:00:00Z",
            stale: false,
            detail: {},
          }
        : {
            filter_id: filterId,
            result: "unknown" as const,
            claim_id: null,
            reason: "profile does not state this input",
            observed_at: null,
            stale: false,
            detail: {},
          },
    ),
    would_exclude: true,
    excluded: ["budget_ceiling"],
    unverified: ["language"],
  };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify({ ...body, ...meta }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function installFetch() {
  const fetchImpl = vi.fn<typeof fetch>(async (input) => {
    const url = String(input);
    if (url.endsWith("/cost")) return jsonResponse(costPayload());
    if (url.endsWith("/filters")) return jsonResponse(filtersPayload());
    throw new Error(`unexpected request: ${url}`);
  });
  vi.stubGlobal("fetch", fetchImpl);
  return fetchImpl;
}

beforeEach(() => {
  process.env.NEXT_PUBLIC_API_URL = "http://api.test";
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ProgrammeProfilePanels", () => {
  it("POSTs the exact profile bodies and renders decided filters and per-scenario DSR", async () => {
    window.localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profile));
    const fetchImpl = installFetch();

    render(<ProgrammeProfilePanels programmeId={7} />);

    expect(await screen.findByText(/is excluded for the stored profile/)).toBeInTheDocument();
    expect(screen.getByText(/TCOA lower bound 420000 exceeds ceiling 400000/)).toBeInTheDocument();
    expect(screen.getByText(/deciding claim #42/)).toBeInTheDocument();
    expect(screen.getByText("0.168 (comfortable)")).toBeInTheDocument();
    expect(screen.getByText(/UNKNOWN — not a zero/)).toBeInTheDocument();

    expect(fetchImpl).toHaveBeenCalledTimes(2);
    const calls = fetchImpl.mock.calls.map(([input, init]) => ({
      url: String(input),
      method: init?.method,
      body: JSON.parse(String(init?.body)) as unknown,
    }));
    expect(calls).toContainEqual({
      url: "http://api.test/api/programmes/7/cost",
      method: "POST",
      body: { profile, living: "hostel" },
    });
    expect(calls).toContainEqual({
      url: "http://api.test/api/programmes/7/filters",
      method: "POST",
      body: { profile },
    });
    for (const call of calls) {
      expect(call.url).not.toContain("?");
      expect(call.url).not.toContain("profile");
    }
  });

  it("labels assumed loan terms and keeps unknown reasons visible", async () => {
    window.localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profile));
    installFetch();

    render(<ProgrammeProfilePanels programmeId={7} />);

    expect(
      await screen.findByRole("heading", {
        name: /Assumed values and terms \(not extracted evidence\)/,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("loan_terms")).toBeInTheDocument();
    expect(screen.getByText(/not published sanction terms/)).toBeInTheDocument();
    expect(
      screen.getByText(/the profile states a family-income band, not a point value/),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/UNKNOWN — income not supplied/)).toHaveLength(2);
    expect(screen.getByText(/Still UNKNOWN by design/)).toBeInTheDocument();
  });

  it("still sends an empty profile and never invents a typical family income", async () => {
    const fetchImpl = installFetch();

    render(<ProgrammeProfilePanels programmeId={7} />);

    expect(await screen.findByText(/Your stored profile is empty/)).toBeInTheDocument();

    await waitFor(() => expect(fetchImpl).toHaveBeenCalledTimes(2));
    const bodies = fetchImpl.mock.calls.map(([, init]) => JSON.parse(String(init?.body)));
    expect(bodies).toContainEqual({ profile: {}, living: "hostel" });
    expect(bodies).toContainEqual({ profile: {} });
  });
});
