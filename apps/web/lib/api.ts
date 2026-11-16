import { z } from "zod";

import {
  apiErrorSchema,
  apiMetaSchema,
  claimEvidencePageSchema,
  comparePayloadSchema,
  costPayloadSchema,
  examWindowsPayloadSchema,
  exclusionsPayloadSchema,
  healthSchema,
  programmeCostRequestSchema,
  programmeDetailSchema,
  programmeFiltersPayloadSchema,
  programmeFiltersRequestSchema,
  programmeListSchema,
  profileRequestSchema,
  rankingsPayloadSchema,
  rankingsRequestSchema,
  sourcesPayloadSchema,
  type ApiMeta,
  type Profile,
} from "./types";

/**
 * Typed API client.
 *
 * Every response is checked against zod schemas. Endpoints owned by other
 * workstreams answer 501 with `{error: {code, message, workstream}}`; that is a
 * distinct result state so screens can render "not yet available" instead of a
 * crash or an endless spinner. Responses for which the API is unreachable,
 * malformed, or reports a down database are likewise first-class states.
 */

export type ApiResult<T> =
  | { state: "ok"; data: T; meta: ApiMeta }
  | {
      state: "not_implemented";
      endpoint: string;
      message: string;
      workstream: string;
      meta: ApiMeta | null;
    }
  | {
      state: "error";
      endpoint: string;
      code: string;
      message: string;
      status: number;
      meta: ApiMeta | null;
    }
  | {
      state: "database_unavailable";
      endpoint: string;
      message: string;
      meta: ApiMeta | null;
    };

export type ApiOk<T> = Extract<ApiResult<T>, { state: "ok" }>;

export function isOk<T>(result: ApiResult<T>): result is ApiOk<T> {
  return result.state === "ok";
}

export function apiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_URL?.trim();
  const base = configured && configured.length > 0 ? configured : "http://localhost:8000";
  return base.replace(/\/+$/, "");
}

const unknownPayloadSchema = z.unknown();

type ZodSchema<T> = z.ZodType<T, z.ZodTypeDef, unknown>;

interface RequestOptions {
  /** Test seam only. Production paths never pass a fetch implementation. */
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
  method?: "GET" | "POST";
  body?: unknown;
  /** Validated before the request leaves the browser (strictly no `any`). */
  bodySchema?: ZodSchema<unknown>;
}

function readMeta(body: unknown): ApiMeta | null {
  const parsed = apiMetaSchema.safeParse(body);
  return parsed.success ? parsed.data : null;
}

function readError(body: unknown): { code: string; message: string; workstream?: string } | null {
  if (typeof body !== "object" || body === null || !("error" in body)) return null;
  const candidate = (body as { error?: unknown }).error;
  const parsed = apiErrorSchema.safeParse(candidate);
  return parsed.success ? parsed.data : null;
}

function failure<T>(
  endpoint: string,
  status: number,
  body: unknown,
  transport?: { code: string; message: string },
): ApiResult<T> {
  const meta = readMeta(body);
  const apiError = readError(body);

  if (transport !== undefined) {
    return {
      state: "error",
      endpoint,
      code: transport.code,
      message: transport.message,
      status,
      meta,
    };
  }

  if (status === 501 || apiError?.code === "not_implemented") {
    return {
      state: "not_implemented",
      endpoint,
      message: apiError?.message ?? `${endpoint} is not implemented`,
      workstream: apiError?.workstream ?? "unassigned",
      meta,
    };
  }

  if (status === 503 || apiError?.code === "database_unavailable") {
    return {
      state: "database_unavailable",
      endpoint,
      message:
        apiError?.message ??
        "The API reports its database is unavailable. Start Postgres, run migrations, then reload.",
      meta,
    };
  }

  return {
    state: "error",
    endpoint,
    code: apiError?.code ?? `http_${status}`,
    message: apiError?.message ?? `Request failed with HTTP ${status}.`,
    status,
    meta,
  };
}

async function request<T>(
  path: string,
  schema: ZodSchema<T>,
  options: RequestOptions = {},
): Promise<ApiResult<T>> {
  const endpoint = path.split("?")[0] ?? path;
  const fetchImpl = options.fetchImpl ?? fetch;

  if (options.bodySchema !== undefined) {
    const checked = options.bodySchema.safeParse(options.body);
    if (!checked.success) {
      return {
        state: "error",
        endpoint,
        code: "invalid_request",
        message: `Refusing to send a request that fails its schema: ${checked.error.issues
          .slice(0, 3)
          .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
          .join("; ")}`,
        status: 0,
        meta: null,
      };
    }
  }

  let response: Response;
  try {
    response = await fetchImpl(`${apiBaseUrl()}${path}`, {
      method: options.method ?? "GET",
      headers: {
        accept: "application/json",
        ...(options.body === undefined ? {} : { "content-type": "application/json" }),
      },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      cache: "no-store",
      signal: options.signal,
    });
  } catch (error) {
    return failure<T>(endpoint, 0, null, {
      code: "network_error",
      message: `Could not reach the API at ${apiBaseUrl()}: ${
        error instanceof Error ? error.message : "network error"
      }`,
    });
  }

  const raw = await response.text();
  let body: unknown = null;
  if (raw.length > 0) {
    try {
      body = JSON.parse(raw) as unknown;
    } catch {
      return failure<T>(endpoint, response.status, null, {
        code: "invalid_response",
        message: "API returned a non-JSON response.",
      });
    }
  }

  if (!response.ok) {
    return failure<T>(endpoint, response.status, body);
  }

  const meta = readMeta(body);
  if (meta === null) {
    return failure<T>(endpoint, response.status, body, {
      code: "invalid_response",
      message:
        "API response is missing the evidence envelope (scorer_version, weights_version, claims_hash, generated_at).",
    });
  }

  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return {
      state: "error",
      endpoint,
      code: "invalid_response",
      message: `API response did not match the expected shape: ${parsed.error.issues
        .slice(0, 3)
        .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
        .join("; ")}`,
      status: response.status,
      meta,
    };
  }

  return { state: "ok", data: parsed.data, meta };
}

// --------------------------------------------------------------------------- endpoints

export interface ProgrammeQuery {
  credential_level?: string;
  recognition_status?: string;
  state?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

function queryString(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const rendered = search.toString();
  return rendered.length > 0 ? `?${rendered}` : "";
}

export function apiHealth(options?: RequestOptions) {
  return request("/api/health", healthSchema, options);
}

export function apiSources(options?: RequestOptions) {
  return request("/api/sources", sourcesPayloadSchema, options);
}

export function apiProgrammes(query: ProgrammeQuery = {}, options?: RequestOptions) {
  return request(
    `/api/programmes${queryString({
      credential_level: query.credential_level,
      recognition_status: query.recognition_status,
      state: query.state,
      q: query.q,
      limit: query.limit,
      offset: query.offset,
    })}`,
    programmeListSchema,
    options,
  );
}

export function apiProgramme(programmeId: number, options?: RequestOptions) {
  return request(`/api/programmes/${programmeId}`, programmeDetailSchema, options);
}

export function apiProgrammeEvidence(
  programmeId: number,
  params: { predicate?: string; limit?: number; offset?: number } = {},
  options?: RequestOptions,
) {
  return request(
    `/api/programmes/${programmeId}/evidence${queryString({
      predicate: params.predicate,
      limit: params.limit,
      offset: params.offset,
    })}`,
    claimEvidencePageSchema,
    options,
  );
}

/** WS-F (brief §13). Both living scenarios arrive in one non-additive payload. */
export function apiProgrammeCost(
  programmeId: number,
  profile: Profile,
  living: "hostel" | "independent" = "hostel",
  options?: RequestOptions,
) {
  return request(`/api/programmes/${programmeId}/cost`, costPayloadSchema, {
    ...options,
    method: "POST",
    body: { profile, living },
    bodySchema: programmeCostRequestSchema,
  });
}

/**
 * WS-F (brief §12). All eight hard filters for one programme, evaluated against
 * the local profile. Profile in the body: never a URL, never a log, never stored.
 */
export function apiProgrammeFilters(
  programmeId: number,
  profile: Profile,
  options?: RequestOptions,
) {
  return request(`/api/programmes/${programmeId}/filters`, programmeFiltersPayloadSchema, {
    ...options,
    method: "POST",
    body: { profile },
    bodySchema: programmeFiltersRequestSchema,
  });
}

/**
 * WS-F (brief §11). The profile travels in the JSON body only: never in a URL,
 * never in a log line, never persisted by this client or the API.
 */
export function apiRankings(profile: Profile, options?: RequestOptions) {
  return request("/api/rankings", rankingsPayloadSchema, {
    ...options,
    method: "POST",
    body: { profile },
    bodySchema: rankingsRequestSchema,
  });
}

/** WS-F (brief §11). Pareto matrix + differing claims + not_comparable pairs. */
export function apiCompare(ids: number[], options?: RequestOptions) {
  return request(
    `/api/compare${queryString({ ids: ids.join(",") })}`,
    comparePayloadSchema,
    options,
  );
}

/** WS-F (brief §12.5). Profile in the body so travel preference can cut exams. */
export function apiExams(profile: Profile, options?: RequestOptions) {
  return request("/api/exams", examWindowsPayloadSchema, {
    ...options,
    method: "POST",
    body: { profile },
    bodySchema: profileRequestSchema,
  });
}

/** WS-F (brief §12). Histogram + per-programme deciding claim; profile in body. */
export function apiExclusions(profile: Profile, options?: RequestOptions) {
  return request("/api/exclusions", exclusionsPayloadSchema, {
    ...options,
    method: "POST",
    body: { profile },
    bodySchema: profileRequestSchema,
  });
}

/** WS-G (brief §18.7). Queue is served by the API workstream, not the browser. */
export function apiReviewQueue(options?: RequestOptions) {
  return request("/api/review/queue", unknownPayloadSchema, options);
}
