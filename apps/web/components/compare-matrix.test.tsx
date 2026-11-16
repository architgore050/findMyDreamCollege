import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CompareMatrix } from "@/components/compare-matrix";
import type { ComparePayload } from "@/lib/types";

const names = { 7: "BSc MLT, Government", 9: "BSc MLT, Private" };

function payload(notComparable = true): ComparePayload {
  return {
    ids: [7, 9],
    matrix: notComparable
      ? {
          "7": { undominated: true, better_than: [], worse_than: [], not_comparable: ["9"] },
          "9": { undominated: true, better_than: [], worse_than: [], not_comparable: ["7"] },
        }
      : {
          "7": { undominated: true, better_than: ["9"], worse_than: [], not_comparable: [] },
          "9": { undominated: false, better_than: [], worse_than: ["7"], not_comparable: [] },
        },
    comparisons: {
      "7|9": {
        verdict: notComparable ? "not_comparable" : "better",
        differing_sub_scores: ["affordability_roi"],
        differing_claims: {
          left: { affordability_roi: [11, 12] },
          right: { affordability_roi: [] },
        },
      },
    },
    not_comparable_pairs: notComparable ? ["7|9"] : [],
  };
}

describe("CompareMatrix", () => {
  it("makes not_comparable pairs unmissable", () => {
    render(<CompareMatrix payload={payload()} names={names} />);

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("NOT COMPARABLE — 1 pair cannot be ordered");
    expect(alert).toHaveTextContent("BSc MLT, Government vs BSc MLT, Private");
    expect(alert).toHaveTextContent(/no winner is implied/i);
    expect(screen.getAllByText("NOT COMPARABLE").length).toBeGreaterThan(1);
  });

  it("renders dominance verdicts and differing claims when pairs resolve", () => {
    render(<CompareMatrix payload={payload(false)} names={names} />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    const table = screen.getByRole("table", { name: /Pairwise Pareto dominance/i });
    expect(within(table).getByText("undominated")).toBeInTheDocument();
    expect(screen.getAllByText("better").length).toBeGreaterThan(0);
    expect(screen.getByText(/claims #11, #12 vs none/)).toBeInTheDocument();
  });
});
