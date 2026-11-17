import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  API_GET_RETRIES,
  API_TIMEOUT_MS,
  apiHealth,
  apiRankings,
  isNetworkUnavailable,
} from "@/lib/api";
import type { Profile } from "@/lib/types";

/**
 * Client-side network policy: timeout, idempotent-GET retries, no POST retries,
 * and a distinct `network_unavailable` state. Fake timers keep every test
 * bounded; nothing touches the network.
 */

const meta = {
  scorer_version: "score-1",
  weights_version: "weights-1",
  claims_hash: "a".repeat(64),
  generated_at: "2026-11-16T09:00:00Z",
};

const health = { ...meta, status: "ok", version: "0.1.0", database: "up" };

const profile: Profile = {
  target_intake_year: 2027,
  target_field: "Medical Laboratory Technology",
  marks: { pcb_pct: 78 },
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function neverResolvingFetch(): ReturnType<typeof vi.fn<typeof fetch>> {
  return vi.fn<typeof fetch>(
    (_url, init) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        });
      }),
  );
}

async function drainTimers(): Promise<void> {
  for (let i = 0; i < 8; i += 1) {
    await vi.advanceTimersByTimeAsync(API_TIMEOUT_MS * 2);
  }
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("request timeout", () => {
  it("aborts a hanging GET and reports network_timeout within a bound", async () => {
    const fetchImpl = neverResolvingFetch();
    const pending = apiHealth({ fetchImpl });

    await drainTimers();
    const result = await pending;

    expect(result.state).toBe("error");
    if (result.state === "error") {
      expect(result.code).toBe("network_timeout");
      expect(result.status).toBe(0);
    }
    expect(isNetworkUnavailable(result)).toBe(true);
    // One attempt + the configured retries; never an unbounded loop.
    expect(fetchImpl).toHaveBeenCalledTimes(1 + API_GET_RETRIES);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("retry policy", () => {
  it("retries an idempotent GET after a 503 and then succeeds", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ error: { code: "unavailable", message: "down" } }, 503))
      .mockResolvedValueOnce(jsonResponse(health));

    const pending = apiHealth({ fetchImpl });
    await drainTimers();
    const result = await pending;

    expect(result.state).toBe("ok");
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("gives up after the GET retry budget", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => jsonResponse({}, 502));

    const pending = apiHealth({ fetchImpl });
    await drainTimers();
    const result = await pending;

    expect(result.state).toBe("error");
    if (result.state === "error") expect(result.status).toBe(502);
    expect(fetchImpl).toHaveBeenCalledTimes(1 + API_GET_RETRIES);
  });

  it("never retries a POST that carries a profile", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => jsonResponse({}, 500));

    const pending = apiRankings(profile, { fetchImpl });
    await drainTimers();
    const result = await pending;

    expect(result.state).toBe("error");
    if (result.state === "error") {
      expect(result.code).toBe("http_500");
      expect(result.status).toBe(500);
    }
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});

describe("offline vs 5xx", () => {
  it("reports network_unavailable when the browser is offline", async () => {
    Object.defineProperty(window.navigator, "onLine", {
      value: false,
      configurable: true,
    });
    const fetchImpl = vi.fn<typeof fetch>(async () => {
      throw new TypeError("Failed to fetch");
    });

    const pending = apiHealth({ fetchImpl });
    await drainTimers();
    const result = await pending;

    expect(result.state).toBe("error");
    if (result.state === "error") expect(result.code).toBe("network_unavailable");
    expect(isNetworkUnavailable(result)).toBe(true);
    expect(fetchImpl).toHaveBeenCalledTimes(1);

    Object.defineProperty(window.navigator, "onLine", {
      value: true,
      configurable: true,
    });
  });

  it("keeps a 5xx distinct from a network failure", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => jsonResponse({}, 500));

    const pending = apiRankings(profile, { fetchImpl });
    await drainTimers();
    const result = await pending;

    expect(result.state).toBe("error");
    if (result.state === "error") {
      expect(result.status).toBe(500);
      expect(result.code).toBe("http_500");
    }
    expect(isNetworkUnavailable(result)).toBe(false);
  });
});
