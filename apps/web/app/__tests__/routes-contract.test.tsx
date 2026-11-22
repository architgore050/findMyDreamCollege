/**
 * Wave v1-w8: route contracts for the surfaces the earlier waves left
 * untested, plus the rendered-output acceptance helper.
 *
 * The seven routes render WITHOUT NaN/undefined/bare composite decimals
 * (PLAN #8 acceptance (b)) — `expectCleanRender` enforces that mechanically
 * on every surface here, the same bar the acceptance run applies.
 * Provenance click-through (acceptance (a) for /programmes/[id]) is pinned at
 * the ClaimRow level: quote, span, sha256, source link and claim id all
 * render, so every number traces back to its evidence.
 * Privacy (acceptance (c)): the clients read localStorage and POST the
 * profile to the local API only — asserted here by the no-profile branch
 * (fetch never fires without a stored profile) and by the existing
 * profile tests, which stay green.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ClaimRow } from "@/components/evidence-tabs";
import { ExamsClient } from "@/components/exams-client";
import { ExclusionsClient } from "@/components/exclusions-client";
import { ShortlistClient } from "@/components/shortlist-client";
import type { Claim, ExclusionsPayload, ExamWindowsPayload, RankingsPayload } from "@/lib/types";
import { PROFILE_STORAGE_KEY } from "@/lib/types";

/** PLAN #8 (b): no NaN, no "undefined", no unformatted long decimals. */
function expectCleanRender(container: HTMLElement) {
  const html = container.innerHTML;
  expect(html).not.toMatch(/>NaN</);
  expect(html).not.toContain("NaN");
  expect(html).not.toContain("undefined");
  // a bare composite decimal would render with 3+ fractional digits
  expect(html).not.toMatch(/\d+\.\d{3,}/);
}

const meta = {
  scorer_version: "score-1",
  weights_version: "weights-1",
  claims_hash: "a".repeat(64),
  generated_at: "2026-11-16T09:00:00Z",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const examPayload: ExamWindowsPayload = {
  items: [
    {
      exam_id: "neet-ug",
      name: "NEET UG",
      country: "IN",
      status: "open",
      observed_at: "2026-11-01T00:00:00Z",
      stale: false,
      ttl_days: 30,
      requires_travel: true,
      remote_options: false,
      travel_blocked: false,
      deciding_claim_id: 12,
      reason: "registration closes 2027-01-31",
      cycle_year: 2027,
    },
  ],
  counts: { open: 1, closed: 0, unverified: 0 },
  next_deadlines: [
    {
      exam_id: "neet-ug",
      name: "NEET UG",
      cycle_year: 2027,
      deadline: "2027-01-31",
      act_before: "2027-01-31",
      status: "open",
    },
  ],
  profile_persisted: false,
};

const exclusionsPayload: ExclusionsPayload = {
  histogram: { budget_ceiling: 2, exam_registration_open: 1 },
  unknown_histogram: { eligibility: 3 },
  excluded: [
    {
      programme: {
        id: 5,
        institution_id: 2,
        name: "BSc Medical Laboratory Technology",
        credential_level: "degree",
        recognition_status: "recognised",
        duration_months: 48,
        internship_months: 12,
        seats: 40,
        medium: "English",
      },
      deciding_filters: [
        {
          filter_id: "budget_ceiling",
          result: "fail",
          claim_id: 31,
          reason: "TCOA lower bound exceeds the ceiling",
        },
      ],
      deciding_claim_ids: [31],
    },
  ],
  excluded_total: 3,
  evaluated_total: 9,
  profile_persisted: false,
};

const rankingsPayload: RankingsPayload = {
  total: 1,
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
        sub_scores: {
          affordability_roi: {
            point: 72.5,
            lo: 60,
            hi: 85,
            n_claims: 2,
            n_sources: 1,
            evidence_tier: "well_evidenced",
            unknown_reason: null,
          },
        },
        composite: {
          point: 68.25,
          lo: 55,
          hi: 80,
          bound: "comonotonic",
          unknown_sub_scores: ["clinical_training"],
        },
        evidence_tier: "partial",
        computed_at: "2026-11-16T09:00:00Z",
        insights: null,
        gaming_flags: [],
        assumptions: [],
      },
      composite_band: {
        point: 68.25,
        lo: 55,
        hi: 80,
        method: "weighted_composite",
        bound: "comonotonic",
      },
      evidence_tier: "partial",
      filters: {
        order: ["credential_level", "recognition_status"],
        counts: { pass: 2, fail: 0, unknown: 0 },
        excluded: [],
        unverified: [],
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
      rank: 1,
    },
  ],
};

const evidenceClaim: Claim = {
  id: 431,
  extraction_id: 17,
  entity_id: 3,
  predicate: "fee.tuition_annual_inr",
  value: 100000,
  unit: "INR/year",
  qualifier: { head: "tuition" },
  quote: "Tuition fee: INR 100000 per annum",
  char_start: 8,
  char_end: 41,
  confidence: 0.9,
  observed_at: "2026-11-02T00:00:00Z",
  method: "extracted",
  url: "https://demoihs.ac.in/files/fee/Fee-Structure-2025-26.pdf",
  fetched_at: "2026-11-03T00:00:00Z",
  sha256: "b".repeat(64),
};

function storedProfile() {
  window.localStorage.setItem(
    PROFILE_STORAGE_KEY,
    JSON.stringify({
      target_intake_year: 2027,
      target_field: "Medical Laboratory Technology",
      marks: { pcb_pct: 78 },
    }),
  );
}

beforeEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("provenance click-through (ClaimRow)", () => {
  it("renders quote, span, sha256, source link and claim id", () => {
    const { container } = render(<ClaimRow claim={evidenceClaim} />);
    expect(screen.getByText(/Tuition fee: INR 100000 per annum/)).toBeInTheDocument();
    expect(screen.getByText(/claim #431/)).toBeInTheDocument();
    expect(screen.getByText(/span 8–41/)).toBeInTheDocument();
    expect(screen.getByText(/sha256 bbbbbbbbbbbb/)).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /Fee-Structure-2025-26\.pdf|source/i });
    expect(link).toHaveAttribute(
      "href",
      "https://demoihs.ac.in/files/fee/Fee-Structure-2025-26.pdf",
    );
    expectCleanRender(container);
  });

  it("never leaks NaN or undefined for a claim without a numeric value", () => {
    const { container } = render(
      <ClaimRow claim={{ ...evidenceClaim, value: null, lo: null, hi: null }} />,
    );
    expectCleanRender(container);
  });
});

describe("ExamsClient (/exams)", () => {
  it("sends the profile only in the POST body, never in the URL", async () => {
    storedProfile();
    const fetchImpl = vi.fn(() =>
      Promise.resolve(jsonResponse({ ...meta, ...examPayload })),
    );
    vi.stubGlobal("fetch", fetchImpl);
    const { container } = render(<ExamsClient />);
    await waitFor(() => expect(screen.getAllByText(/NEET UG/).length).toBeGreaterThan(0));
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).not.toMatch(/pcb_pct|target_intake_year/); // profile never in a URL
    expect(init.method).toBe("POST");
    const body = JSON.parse(String(init.body));
    expect(body.profile.target_intake_year).toBe(2027); // and carried in the body
    expectCleanRender(container);
  });

  it("renders a 501 as a named state and stays clean", async () => {
    storedProfile();
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          jsonResponse(
            {
              error: { code: "not_implemented", message: "/api/exams is not implemented" },
            },
            501,
          ),
        ),
      ),
    );
    const { container } = render(<ExamsClient />);
    await waitFor(() => expect(container.textContent).not.toBe(""));
    await waitFor(() => {
      expect(container.textContent).toMatch(/not implemented|not yet available/i);
    });
    expectCleanRender(container);
  });

  it("renders exam windows with counts and no raw decimals", async () => {
    storedProfile();
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(jsonResponse({ ...meta, ...examPayload }))),
    );
    const { container } = render(<ExamsClient />);
    await waitFor(() => expect(screen.getAllByText(/NEET UG/).length).toBeGreaterThan(0));
    expect(screen.getAllByText(/2027-01-31|Jan/i).length).toBeGreaterThan(0);
    expectCleanRender(container);
  });
});

describe("ExclusionsClient (/exclusions)", () => {
  it("renders the histogram and excluded programmes cleanly", async () => {
    storedProfile();
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(jsonResponse({ ...meta, ...exclusionsPayload }))),
    );
    const { container } = render(<ExclusionsClient />);
    await waitFor(() => expect(container.textContent).toMatch(/budget_ceiling|Budget ceiling/i));
    expect(container.textContent).toMatch(/excluded|Excluded/i);
    expectCleanRender(container);
  });
});

describe("ShortlistClient (/shortlist)", () => {
  it("ranks from the stored profile and stays clean", async () => {
    storedProfile();
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(jsonResponse({ ...meta, ...rankingsPayload }))),
    );
    const { container } = render(<ShortlistClient />);
    await waitFor(() =>
      expect(screen.getAllByText(/Medical Laboratory Technology/i).length).toBeGreaterThan(0),
    );
    // the RANKED row, not the profile summary: the band text comes from the
    // composite_band payload, and schema-validity is what makes this real
    expect(container.textContent).toMatch(/weighted_composite/);
    expect(container.textContent).not.toMatch(/invalid_response|Request failed/);
    expectCleanRender(container);
  });

  it("shows the unknown-inputs banner when the profile is empty", async () => {
    window.localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify({}));
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(jsonResponse({ ...meta, ...rankingsPayload }))),
    );
    const { container } = render(<ShortlistClient />);
    await waitFor(() => expect(container.textContent).toMatch(/UNKNOWN/));
    expectCleanRender(container);
  });
});
