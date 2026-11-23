/**
 * Wave v1-w9: the methodology trust surface.
 * The page must render what the API generated (weights, versions, exclusions,
 * dispute path) and degrade to a named failure state — never invent content.
 */

import { render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MethodologyClient } from "@/components/methodology-client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const meta = {
  scorer_version: "0.3.0",
  weights_version: "2026-11-18.1",
  claims_hash: "a".repeat(64),
  generated_at: "2026-11-22T00:00:00Z",
};

const methodology = {
  weights: {
    outcomes: 30,
    affordability_roi: 20,
    clinical_training: 15,
    academic_quality: 10,
    student_life_safety: 10,
    admission_realism: 10,
    financing_feasibility: 5,
  },
  weights_sum: 100,
  component_policy: {
    min_claims_for_score: 1,
    min_claims_for_confidence: 5,
    min_components_for_composite: 2,
    single_claim_interval_half_width: 25,
  },
  evidence_tiers: { well_evidenced: { min_claims: 2 } },
  predicates_version: "pred-1",
  predicates_count: 42,
  predicate_aliases: 7,
  sources: {
    total: 30,
    enabled: 28,
    by_trust: { high: 20, medium: 8, low: 2 },
    enabled_by_trust: { high: 19, medium: 7, low: 2 },
  },
  source_exclusions: [
    { id: "job_portals", name: "Job portals", reason: "PII-heavy, not evidence" },
  ],
  value_policy: [
    "UNKNOWN is a value: a component with no usable claim is reported UNKNOWN.",
    "Every published number carries the claim id that produced it.",
  ],
  dispute_path: [
    "Follow the claim link to the source document and compare the quote.",
    "Open an issue with the claim id; corrections are appended, never rewritten.",
  ],
};

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function expectCleanRender(container: HTMLElement) {
  const html = container.innerHTML;
  expect(html).not.toContain("NaN");
  expect(html).not.toContain("undefined");
  expect(html).not.toMatch(/\d+\.\d{3,}/);
}

describe("MethodologyClient (/methodology)", () => {
  it("renders the generated weights, versions, exclusions and dispute path", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({ ...meta, ...methodology }))));
    const { container } = render(<MethodologyClient />);
    await waitFor(() => expect(container.textContent).toMatch(/affordability_roi/));
    expect(container.textContent).toMatch(/2026-11-18\.1/); // weights version from the envelope
    expect(container.textContent).toMatch(/job_portals/);
    expect(container.textContent).toMatch(/Disputing a number/i);
    expect(container.textContent).toMatch(/append-only/i);
    expect(container.textContent).toMatch(/pred-1/);
    expectCleanRender(container);
  });

  it("renders a named failure state instead of invented content", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          jsonResponse(
            { error: { code: "not_implemented", message: "/api/methodology is not implemented" } },
            501,
          ),
        ),
      ),
    );
    const { container } = render(<MethodologyClient />);
    await waitFor(() => expect(container.textContent).toMatch(/not implemented/i));
    expect(container.textContent).not.toMatch(/affordability_roi/);
    expectCleanRender(container);
  });
});
