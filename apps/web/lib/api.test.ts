import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  apiCompare,
  apiExams,
  apiExclusions,
  apiProgrammeCost,
  apiProgrammes,
  apiRankings,
  apiReviewQueue,
} from "@/lib/api";
import type { Profile } from "@/lib/types";

/**
 * All four result states are exercised with a mocked fetch. Nothing here touches
 * the network: the base URL is a literal test origin and no request is issued.
 */

const meta = {
  scorer_version: "score-1",
  weights_version: "weights-1",
  claims_hash: "a".repeat(64),
  generated_at: "2026-11-16T09:00:00Z",
};

const profile: Profile = {
  target_intake_year: 2027,
  target_field: "Medical Laboratory Technology",
  marks: { pcb_pct: 78 },
};

function envelope(payload: Record<string, unknown>): Record<string, unknown> {
  return { ...payload, ...meta };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function fetchReturning(body: unknown, status = 200) {
  return vi.fn<typeof fetch>(async () => jsonResponse(body, status));
}

function programmePayload() {
  return {
    id: 7,
    institution_id: 3,
    name: "BSc MLT",
    credential_level: "degree",
    recognition_status: "recognised",
    duration_months: 48,
    internship_months: 6,
    seats: 60,
    medium: "English",
  };
}

function rankingRow() {
  return {
    programme: programmePayload(),
    score: {
      entity_id: 3,
      scorer_version: "score-1",
      weights_version: "weights-1",
      claims_hash: "a".repeat(64),
      sub_scores: {
        affordability_roi: {
          point: null,
          lo: 0,
          hi: 100,
          n_claims: 0,
          n_sources: 0,
          evidence_tier: "insufficient",
          components: [],
          unknown_reason: "no cost evidence",
        },
      },
      composite: {
        point: null,
        lo: 0,
        hi: 100,
        bound: "comonotonic",
        unknown_sub_scores: ["affordability_roi"],
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
      order: ["credential_level"],
      counts: { pass: 1, fail: 0, unknown: 7 },
      excluded: [],
      unverified: ["language"],
      decisions: {
        credential_level: {
          filter_id: "credential_level",
          result: "pass",
          claim_id: null,
          reason: "degree required",
        },
      },
    },
    excluded: false,
    insights: null,
    cost: null,
  };
}

beforeEach(() => {
  process.env.NEXT_PUBLIC_API_URL = "http://api.test";
});

describe("api client", () => {
  it("returns ok with a validated envelope", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        items: [programmePayload()],
        total: 1,
        limit: 50,
        offset: 0,
        facets: {},
        exclusions: { available: false, owner: "WS-F" },
      }),
    );

    const result = await apiProgrammes({}, { fetchImpl });

    expect(result.state).toBe("ok");
    if (result.state !== "ok") throw new Error("expected ok");
    expect(result.data.items).toHaveLength(1);
    expect(result.data.items[0]?.name).toBe("BSc MLT");
    expect(result.meta.scorer_version).toBe("score-1");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(fetchImpl.mock.calls[0]?.[0]).toBe("http://api.test/api/programmes");
  });

  it("maps HTTP 501 to a not_implemented state naming the workstream", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        error: {
          code: "not_implemented",
          message: "/api/review/queue is not implemented (WS-G, see brief §18.7)",
          workstream: "WS-G",
        },
      }),
      501,
    );

    const result = await apiReviewQueue({ fetchImpl });

    expect(result.state).toBe("not_implemented");
    if (result.state !== "not_implemented") throw new Error("expected not_implemented");
    expect(result.workstream).toBe("WS-G");
    expect(result.endpoint).toBe("/api/review/queue");
    expect(result.meta?.claims_hash).toHaveLength(64);
  });

  it("maps HTTP 503 with database_unavailable to its own state", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        error: {
          code: "database_unavailable",
          message: "database unavailable: connection refused",
        },
      }),
      503,
    );

    const result = await apiProgrammes({}, { fetchImpl });

    expect(result.state).toBe("database_unavailable");
    if (result.state !== "database_unavailable") throw new Error("expected database_unavailable");
    expect(result.message).toContain("connection refused");
  });

  it("maps other failures to error and keeps the code", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        error: { code: "validation_error", message: "limit must be <= 200" },
      }),
      422,
    );

    const result = await apiProgrammes({}, { fetchImpl });

    expect(result.state).toBe("error");
    if (result.state !== "error") throw new Error("expected error");
    expect(result.code).toBe("validation_error");
    expect(result.status).toBe(422);
  });

  it("reports a transport failure without inventing an envelope", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => {
      throw new Error("connection refused");
    });

    const result = await apiProgrammes({}, { fetchImpl });

    expect(result.state).toBe("error");
    if (result.state !== "error") throw new Error("expected error");
    expect(result.code).toBe("network_error");
    expect(result.status).toBe(0);
    expect(result.meta).toBeNull();
  });

  it("rejects a response that lacks the evidence envelope", async () => {
    const fetchImpl = fetchReturning({ items: [], total: 0, limit: 50, offset: 0 });

    const result = await apiProgrammes({}, { fetchImpl });

    expect(result.state).toBe("error");
    if (result.state !== "error") throw new Error("expected error");
    expect(result.code).toBe("invalid_response");
    expect(result.message).toContain("evidence envelope");
  });

  it("rejects a malformed payload even when the envelope is valid", async () => {
    const fetchImpl = fetchReturning(envelope({ items: "not-an-array" }));

    const result = await apiProgrammes({}, { fetchImpl });

    expect(result.state).toBe("error");
    if (result.state !== "error") throw new Error("expected error");
    expect(result.code).toBe("invalid_response");
  });
});

describe("profile-bearing POST endpoints", () => {
  it("POSTs the rankings profile in the body and never in the URL", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        items: [rankingRow()],
        total: 1,
        weights: { affordability_roi: 100 },
        profile_persisted: false,
      }),
    );

    const result = await apiRankings(profile, { fetchImpl });

    expect(result.state).toBe("ok");
    if (result.state !== "ok") throw new Error("expected ok");
    expect(result.data.items[0]?.composite_band.point).toBeNull();
    expect(result.data.profile_persisted).toBe(false);

    const [url, init] = fetchImpl.mock.calls[0] ?? [];
    expect(url).toBe("http://api.test/api/rankings");
    expect(String(url)).not.toContain("profile");
    expect(String(url)).not.toContain("2027");
    expect(String(url)).not.toContain("Medical+Laboratory");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ profile });
  });

  it("refuses to send a profile that fails the request schema", async () => {
    const fetchImpl = fetchReturning(envelope({}));

    const result = await apiRankings({ target_intake_year: 1800 }, { fetchImpl });

    expect(result.state).toBe("error");
    if (result.state !== "error") throw new Error("expected error");
    expect(result.code).toBe("invalid_request");
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("POSTs the exam profile in the body and validates the window payload", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        items: [
          {
            exam_id: "neet-ug",
            name: "NEET-UG",
            country: "IN",
            status: "unverified",
            observed_at: null,
            stale: true,
            ttl_days: 120,
            requires_travel: true,
            remote_options: false,
            travel_blocked: false,
            deciding_claim_id: null,
            reason: "window has no observed_at; cannot verify freshness",
          },
        ],
        counts: { open: 0, closed: 0, unverified: 1 },
        profile_persisted: false,
      }),
    );

    const result = await apiExams(profile, { fetchImpl });

    expect(result.state).toBe("ok");
    if (result.state !== "ok") throw new Error("expected ok");
    expect(result.data.items[0]?.status).toBe("unverified");

    const [url, init] = fetchImpl.mock.calls[0] ?? [];
    expect(url).toBe("http://api.test/api/exams");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ profile });
  });

  it("POSTs the exclusions profile and validates histogram + deciding claim", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        histogram: { recognition_status: 2 },
        unknown_histogram: { language: 1 },
        excluded: [
          {
            programme: programmePayload(),
            deciding_filters: [
              {
                filter_id: "recognition_status",
                result: "fail",
                claim_id: 41,
                reason: "institute not on the NCAHP register",
                observed_at: "2026-02-01T00:00:00Z",
                stale: false,
                detail: {},
              },
            ],
            deciding_claim_ids: [41],
          },
        ],
        excluded_total: 1,
        evaluated_total: 25,
        profile_persisted: false,
      }),
    );

    const result = await apiExclusions(profile, { fetchImpl });

    expect(result.state).toBe("ok");
    if (result.state !== "ok") throw new Error("expected ok");
    expect(result.data.histogram["recognition_status"]).toBe(2);
    expect(result.data.excluded[0]?.deciding_claim_ids).toEqual([41]);

    const [url, init] = fetchImpl.mock.calls[0] ?? [];
    expect(url).toBe("http://api.test/api/exclusions");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ profile });
  });

  it("keeps a 501 on the review queue and does not treat it as a ranking", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        error: {
          code: "not_implemented",
          message: "/api/review/queue is not implemented (WS-G, see brief §18.7)",
          workstream: "WS-G",
        },
      }),
      501,
    );

    const result = await apiReviewQueue({ fetchImpl });

    expect(result.state).toBe("not_implemented");
  });
});

describe("GET endpoints", () => {
  it("requests the cost scenario with living in the query and parses assumed flags", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        programme_id: 7,
        living: "hostel",
        available: true,
        non_additive_scenarios: true,
        assumed: {
          loan_rate: { assumed: true, source: "assumption", note: "modelled assumption" },
        },
        currency: "INR",
        duration_months: 48,
        scenarios: {
          hostel: {
            name: "hostel",
            components: {},
            annual: {
              value: 60000,
              lo: 48000,
              hi: 72000,
              currency: "INR",
              method: "modelled_city_basket",
              assumptions: [],
            },
            total: {
              value: 240000,
              lo: 192000,
              hi: 288000,
              currency: "INR",
              method: "modelled_city_basket",
              assumptions: [],
            },
            modelled: true,
          },
          independent: {
            name: "independent",
            components: {},
            annual: {
              value: 90000,
              lo: 72000,
              hi: 108000,
              currency: "INR",
              method: "modelled_city_basket",
              assumptions: [],
            },
            total: {
              value: 360000,
              lo: 288000,
              hi: 432000,
              currency: "INR",
              method: "modelled_city_basket",
              assumptions: [],
            },
            modelled: true,
          },
        },
        dsr25: {
          value: null,
          band: "unknown",
          formula: "first_year_emi / p25_first_year_gross_salary",
          first_year_emi: null,
          p25_first_year_salary: null,
          bands: {},
          inputs: {},
        },
        assumptions: ["living=hostel"],
      }),
    );

    const result = await apiProgrammeCost(7, "hostel", { fetchImpl });

    expect(result.state).toBe("ok");
    if (result.state !== "ok") throw new Error("expected ok");
    expect(Object.keys(result.data.scenarios)).toEqual(["hostel", "independent"]);
    expect(result.data.dsr25?.value).toBeNull();
    expect(fetchImpl.mock.calls[0]?.[0]).toBe(
      "http://api.test/api/programmes/7/cost?living=hostel",
    );
  });

  it("parses the compare Pareto matrix and its not_comparable pairs", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        ids: [7, 9],
        matrix: {
          "7": {
            undominated: true,
            better_than: [],
            worse_than: [],
            not_comparable: ["9"],
          },
          "9": {
            undominated: true,
            better_than: [],
            worse_than: [],
            not_comparable: ["7"],
          },
        },
        comparisons: {
          "7|9": {
            verdict: "not_comparable",
            differing_sub_scores: ["affordability_roi"],
            differing_claims: {
              left: { affordability_roi: [1] },
              right: { affordability_roi: [] },
            },
          },
        },
        not_comparable_pairs: ["7|9"],
      }),
    );

    const result = await apiCompare([7, 9], { fetchImpl });

    expect(result.state).toBe("ok");
    if (result.state !== "ok") throw new Error("expected ok");
    expect(result.data.not_comparable_pairs).toEqual(["7|9"]);
    expect(result.data.comparisons["7|9"]?.verdict).toBe("not_comparable");
    expect(fetchImpl.mock.calls[0]?.[0]).toBe("http://api.test/api/compare?ids=7%2C9");
  });
});
