import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CostPanel } from "@/components/cost-panel";
import type { CostPayload, MoneyBand } from "@/lib/types";

function band(value: number | null, method = "modelled_city_basket"): MoneyBand {
  return {
    value,
    lo: value === null ? null : value * 0.8,
    hi: value === null ? null : value * 1.2,
    currency: "INR",
    method,
    assumptions: [],
  };
}

function payload(): CostPayload {
  return {
    programme_id: 7,
    living: "hostel",
    available: true,
    non_additive_scenarios: true,
    note: "hostel and independent are complete alternatives; never sum them",
    currency: "INR",
    duration_months: 48,
    assumed: {
      loan_rate: {
        assumed: true,
        source: "assumption",
        note: "modelled assumption, not a published scheme rate",
      },
    },
    scenarios: {
      hostel: {
        name: "hostel",
        modelled: true,
        components: {
          hostel: band(60000),
          mess: band(null, "unknown"),
        },
        annual: band(60000),
        total: band(240000),
      },
      independent: {
        name: "independent",
        modelled: true,
        components: {
          rent: band(96000),
          food: band(48000),
        },
        annual: band(144000),
        total: band(576000),
      },
    },
    tcoa: band(420000, "computed"),
    tcoa_first_year: band(180000, "computed"),
    net_price: band(420000, "computed"),
    loan_principal: band(420000, "computed"),
    family_out_of_pocket: band(300000, "computed"),
    student_work_offset: band(0, "computed"),
    debt_at_repayment: band(null, "unknown_rate"),
    emi: band(null, "unknown_rate"),
    dsr25: {
      value: null,
      band: "unknown",
      formula: "first_year_emi / p25_first_year_gross_salary",
      first_year_emi: null,
      p25_first_year_salary: null,
      bands: {},
      inputs: {},
    },
    affordability_gap: {
      value: null,
      flagged: null,
      flag_above: 0.4,
      formula: "family income not supplied",
      family_out_of_pocket: 300000,
      annual_family_income: null,
    },
    bands: {},
    methods: {},
    assumptions: ["living=hostel", "no opportunity cost included"],
    assumptions_flags: {},
    display: {},
  };
}

describe("CostPanel", () => {
  it("shows both non-additive living scenarios side by side", () => {
    render(<CostPanel cost={payload()} />);

    expect(screen.getByRole("heading", { name: "Hostel scenario" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Independent scenario" })).toBeInTheDocument();
    expect(screen.getByText(/never sum them/)).toBeInTheDocument();
  });

  it("labels values the model assumed rather than extracted", () => {
    render(<CostPanel cost={payload()} />);

    expect(
      screen.getByRole("heading", { name: /Assumed values \(not extracted evidence\)/ }),
    ).toBeInTheDocument();
    expect(screen.getByText("loan_rate")).toBeInTheDocument();
    expect(screen.getByText(/not a published scheme rate/)).toBeInTheDocument();
    expect(screen.getAllByText("modelled_city_basket").length).toBeGreaterThan(0);
  });

  it("does not fabricate a DSR25 when salary evidence is missing", () => {
    render(<CostPanel cost={payload()} />);

    expect(
      screen.getByText(/UNKNOWN — salary or loan-rate evidence is missing/),
    ).toBeInTheDocument();
  });

  it("renders an explicit unavailable state instead of a zero total", () => {
    render(
      <CostPanel
        cost={{
          programme_id: 7,
          living: "hostel",
          available: false,
          reason: "no complete cost evidence; refusing to model a TCOA from nothing",
          scenarios: {},
        }}
      />,
    );

    expect(
      screen.getByText(/no complete cost evidence; refusing to model a TCOA from nothing/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Hostel scenario/)).not.toBeInTheDocument();
  });
});
