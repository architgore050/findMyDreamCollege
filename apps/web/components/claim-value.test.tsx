import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ClaimValue } from "@/components/claim-value";

describe("ClaimValue", () => {
  it("renders UNKNOWN instead of zero when there is no value", () => {
    render(<ClaimValue label="P25 first-year salary" value={undefined} />);

    const root = screen.getByTestId("claim-value");
    expect(root).toHaveAttribute("data-state", "unknown");
    expect(screen.getByText("UNKNOWN")).toBeInTheDocument();
    expect(root).toHaveTextContent("interval not reported");
    expect(root).toHaveTextContent("n=—");
    expect(root).toHaveTextContent("not zero and not a midpoint");
    expect(screen.queryByText("0")).not.toBeInTheDocument();
  });

  it("treats the literal string 'unknown' as missing, not as a value", () => {
    render(<ClaimValue label="Placement rate" value="unknown" />);

    expect(screen.getByTestId("claim-value")).toHaveAttribute("data-state", "unknown");
    expect(screen.getByText("UNKNOWN")).toBeInTheDocument();
  });

  it("renders point, interval, n and evidence tier together", () => {
    render(
      <ClaimValue
        label="P25 first-year salary"
        value={25000}
        unit="INR"
        lo={18000}
        hi={31000}
        n={42}
        tier="thin"
        method="extracted"
        sourceUrl="https://example.ac.in/placement-report.pdf"
        observedAt="2026-03-01T00:00:00Z"
        confidence={0.82}
      />,
    );

    const root = screen.getByTestId("claim-value");
    expect(root).toHaveAttribute("data-state", "known");
    expect(root).toHaveTextContent("25,000 INR");
    expect(root).toHaveTextContent("[18,000, 31,000]");
    expect(root).toHaveTextContent("n=42");
    expect(root).toHaveTextContent("thin evidence");
    expect(root).toHaveTextContent("extracted");
    expect(root).toHaveTextContent("observed 2026-03-01");
    expect(root).toHaveTextContent("claim confidence 0.82");

    const link = screen.getByRole("link", { name: /source: example\.ac\.in/i });
    expect(link).toHaveAttribute("href", "https://example.ac.in/placement-report.pdf");
    expect(link).toHaveAttribute("target", "_blank");
  });

  it("flags a number with no source in a warning style", () => {
    render(<ClaimValue label="Seats" value={60} n={1} />);

    expect(screen.getByTestId("claim-value")).toHaveAttribute("data-state", "known");
    expect(screen.getByText(/no source/i)).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("renders a known interval even when the point estimate is unknown", () => {
    render(<ClaimValue label="TCOA" value={null} lo={400000} hi={520000} n={3} />);

    const root = screen.getByTestId("claim-value");
    expect(root).toHaveTextContent("UNKNOWN");
    expect(root).toHaveTextContent("interval [4,00,000, 5,20,000]");
    expect(root).toHaveTextContent("n=3");
  });
});
