import { describe, expect, it } from "vitest";

import { costCategorySchema, costItemSchema, sourceSchema } from "../types";

/**
 * Contract tests for the API-client schemas against shapes the backend really
 * emits (packages/pipeline/fmd/api.py + db/repositories.py). The tuition
 * regression is the acceptance-path break: every seeded programme carries a
 * tuition cost row, and the missing enum member rejected the whole programme
 * detail payload in the browser.
 */

describe("costCategorySchema", () => {
  it("accepts tuition — the first category the backend seeds", () => {
    expect(costCategorySchema.safeParse("tuition").success).toBe(true);
  });

  it("accepts every backend cost category", () => {
    for (const category of [
      "tuition",
      "hostel",
      "mess",
      "rent",
      "utilities",
      "food",
      "transport",
      "equipment",
      "exam_fee",
      "visa",
      "insurance",
      "misc",
    ]) {
      expect(costCategorySchema.safeParse(category).success).toBe(true);
    }
  });

  it("rejects a category the registry does not define", () => {
    expect(costCategorySchema.safeParse("donation").success).toBe(false);
  });
});

describe("costItemSchema", () => {
  it("parses a tuition cost row as the API serialises it", () => {
    const parsed = costItemSchema.safeParse({
      id: 1,
      programme_id: 7,
      category: "tuition",
      label: "tuition",
      amount: "100000.00", // Postgres numeric arrives as a JSON string
      currency: "INR",
      period: "per_year",
      basis: null,
      source_claim_id: null,
      confidence: 1.0,
    });
    expect(parsed.success).toBe(true);
  });
});

describe("sourceSchema", () => {
  it("accepts a source registered without an adapter (backend emits null)", () => {
    const parsed = sourceSchema.safeParse({
      id: "example-source",
      name: "Example Source",
      base_url: "https://example.gov.in",
      kind: "inventory",
      format: "html",
      cadence_days: 30,
      trust: "medium",
      adapter: null,
      enabled: true,
      requires_manual_review: false,
      notes: null,
      last_crawl: null,
    });
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.adapter).toBeNull();
    }
  });
});
