const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
});

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "not finite";
  return Number.isInteger(value) ? integerFormatter.format(value) : numberFormatter.format(value);
}

export function withUnit(value: string, unit?: string | null): string {
  if (!unit) return value;
  return unit === "%" ? `${value}%` : `${value} ${unit}`;
}

export function formatAmount(amount: number | string, currency?: string | null): string {
  const numeric = typeof amount === "string" ? Number(amount) : amount;
  if (!Number.isFinite(numeric)) return `${amount}${currency ? ` ${currency}` : ""}`;
  if (currency) {
    try {
      return new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency,
        maximumFractionDigits: 0,
      }).format(numeric);
    } catch {
      // Unknown currency code: fall back to a plain amount plus the literal code.
    }
  }
  return `${formatNumber(numeric)}${currency ? ` ${currency}` : ""}`;
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "not recorded";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return `${parsed.toISOString().slice(0, 10)} ${parsed.toISOString().slice(11, 16)}Z`;
}

export function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export function titleFromPredicate(predicate: string): string {
  return predicate.replace(/[_.]+/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

/** Narrow `unknown` JSON values without `any`. */
export function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

export interface ClaimPoint {
  point: number | null;
  lo: number | null;
  hi: number | null;
  n: number | null;
  unit: string | null;
  display: string | null;
}

/**
 * Pull a point/interval/n out of a claim without inventing values.
 * A claim with no numeric value is UNKNOWN, never 0.
 */
export function readClaimPoint(claim: {
  value?: unknown;
  unit?: string | null;
  qualifier?: Record<string, unknown> | null;
}): ClaimPoint {
  const qualifier = claim.qualifier ?? null;
  const numeric = asNumber(claim.value);
  const lo = asNumber(qualifier?.["lo"] ?? qualifier?.["interval_lo"]) ?? null;
  const hi = asNumber(qualifier?.["hi"] ?? qualifier?.["interval_hi"]) ?? null;
  const n =
    asNumber(qualifier?.["n"] ?? qualifier?.["sample_size"] ?? qualifier?.["n_claims"]) ?? null;
  const unit = claim.unit ?? null;

  if (numeric !== null) {
    return { point: numeric, lo, hi, n, unit, display: null };
  }
  if (typeof claim.value === "string" && claim.value.trim() !== "") {
    return { point: null, lo, hi, n, unit, display: claim.value };
  }
  if (typeof claim.value === "boolean") {
    return { point: null, lo, hi, n, unit, display: claim.value ? "yes" : "no" };
  }
  return { point: null, lo, hi, n, unit, display: null };
}
