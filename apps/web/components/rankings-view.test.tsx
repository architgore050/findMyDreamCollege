import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RankingsView } from "@/components/rankings-view";
import type { RankingsPayload } from "@/lib/types";

const meta = {
  scorer_version: "score-1",
  weights_version: "weights-1",
  claims_hash: "a".repeat(64),
  generated_at: "2026-11-16T09:00:00Z",
};

function payload(): RankingsPayload {
  return {
    total: 2,
    weights: { affordability_roi: 50, admission_realism: 50 },
    profile_persisted: false,
    items: [
      {
        programme: {
          id: 7,
          institution_id: 3,
          name: "BSc Medical Laboratory Technology",
          credential_level: "degree",
          recognition_status: "recognised",
          duration_months: 48,
          internship_months: 12,
          seats: 60,
          medium: "English",
        },
        score: {
          entity_id: 3,
          scorer_version: "score-1",
          weights_version: "weights-1",
          claims_hash: "a".repeat(64),
          sub_scores: {},
          composite: {
            point: null,
            lo: 0,
            hi: 100,
            bound: "comonotonic",
            unknown_sub_scores: ["affordability_roi", "admission_realism"],
          },
          evidence_tier: "insufficient",
          computed_at: "2026-11-16T09:00:00Z",
          insights: null,
          gaming_flags: [],
          assumptions: [],
        },
        composite_band: {
          point: null,
          lo: 0,
          hi: 100,
          method: "weighted_composite",
          bound: "comonotonic",
        },
        evidence_tier: "insufficient",
        filters: {
          order: ["credential_level", "recognition_status"],
          counts: { pass: 1, fail: 0, unknown: 1 },
          excluded: [],
          unverified: ["recognition_status"],
          decisions: {
            credential_level: {
              filter_id: "credential_level",
              result: "pass",
              claim_id: 12,
              reason: "degree credential",
            },
          },
        },
        excluded: false,
        insights: {
          selected: [
            {
              id: "salary",
              predicate: "p25_first_year_salary",
              text: "No p25 first-year salary claim resolves for this programme.",
              coverage: ["salary"],
              claim_ids: [],
              kind: "evidence",
              detail: {},
            },
          ],
          mandatory: [],
          coverage: 0.25,
          covered_units: ["salary"],
          uncovered_units: ["placement"],
          stopped_reason: "evidence_exhausted",
        },
        cost: null,
      },
      {
        programme: {
          id: 9,
          institution_id: 4,
          name: "BSc MLT (unrecognised institute)",
          credential_level: "degree",
          recognition_status: "unrecognised",
          duration_months: 48,
          internship_months: 6,
          seats: 40,
          medium: "English",
        },
        score: {
          entity_id: 4,
          scorer_version: "score-1",
          weights_version: "weights-1",
          claims_hash: "a".repeat(64),
          sub_scores: {},
          composite: {
            point: null,
            lo: 0,
            hi: 100,
            bound: "comonotonic",
            unknown_sub_scores: [],
          },
          evidence_tier: "thin",
          computed_at: "2026-11-16T09:00:00Z",
          insights: null,
          gaming_flags: [],
          assumptions: [],
        },
        composite_band: {
          point: null,
          lo: 0,
          hi: 100,
          method: "weighted_composite",
          bound: "comonotonic",
        },
        evidence_tier: "thin",
        filters: {
          order: ["recognition_status"],
          counts: { pass: 0, fail: 1, unknown: 0 },
          excluded: ["recognition_status"],
          unverified: [],
          decisions: {
            recognition_status: {
              filter_id: "recognition_status",
              result: "fail",
              claim_id: 41,
              reason: "institute not on the NCAHP register",
              observed_at: "2026-02-01T00:00:00Z",
              stale: false,
              detail: {},
            },
          },
        },
        excluded: true,
        insights: null,
        cost: null,
      },
    ],
  };
}

describe("RankingsView", () => {
  it("renders a composite band and evidence tier, never a decimal point score", () => {
    render(<RankingsView payload={payload()} meta={meta} />);

    const first = screen.getByRole("article", { name: /BSc Medical Laboratory Technology/i });
    expect(within(first).getByText("0–100")).toBeInTheDocument();
    expect(within(first).getAllByText(/insufficient evidence/i).length).toBeGreaterThan(0);
    expect(within(first).queryByText(/\b\d{1,3}\.\d+\b/)).not.toBeInTheDocument();
  });

  it("names which sub-scores are UNKNOWN instead of scoring them as zero", () => {
    render(<RankingsView payload={payload()} meta={meta} />);

    expect(
      screen.getByText(/UNKNOWN sub-scores: Affordability Roi, Admission Realism/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/not scored as zero/i)).toBeInTheDocument();
  });

  it("renders the insight set with the claim that supports it", () => {
    render(<RankingsView payload={payload()} meta={meta} />);

    expect(
      screen.getByText(/No p25 first-year salary claim resolves for this programme\./),
    ).toBeInTheDocument();
    expect(screen.getByText(/no deciding claim published/i)).toBeInTheDocument();
  });

  it("marks a hard-filter exclusion with its deciding claim", () => {
    render(<RankingsView payload={payload()} meta={meta} />);

    const second = screen.getByRole("article", { name: /unrecognised institute/i });
    expect(within(second).getByText("EXCLUDED")).toBeInTheDocument();
    expect(within(second).getByText(/institute not on the NCAHP register/)).toBeInTheDocument();
    expect(within(second).getByText(/claim #41/)).toBeInTheDocument();
  });

  it("treats a profile_persisted=true response as a bug to surface", () => {
    render(<RankingsView payload={{ ...payload(), profile_persisted: true }} meta={meta} />);

    expect(screen.getByText(/profile_persisted: true/)).toBeInTheDocument();
  });
});
