import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiProgrammes, apiRankings } from "@/lib/api";

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

beforeEach(() => {
  process.env.NEXT_PUBLIC_API_URL = "http://api.test";
});

describe("api client", () => {
  it("returns ok with a validated envelope", async () => {
    const fetchImpl = fetchReturning(
      envelope({
        items: [
          {
            id: 7,
            institution_id: 3,
            name: "BSc MLT",
            credential_level: "degree",
            recognition_status: "recognised",
            duration_months: 48,
            internship_months: 6,
            seats: 60,
            medium: "English",
          },
        ],
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
          message: "/api/rankings is not implemented (WS-F, see brief §11)",
          workstream: "WS-F",
        },
      }),
      501,
    );

    const result = await apiRankings({}, { fetchImpl });

    expect(result.state).toBe("not_implemented");
    if (result.state !== "not_implemented") throw new Error("expected not_implemented");
    expect(result.workstream).toBe("WS-F");
    expect(result.endpoint).toBe("/api/rankings");
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
