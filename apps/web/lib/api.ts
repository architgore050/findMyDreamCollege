import { z } from "zod";

import {
  aliasReviewActionRequestSchema,
  methodologyPayloadSchema,
  apiErrorSchema,
  apiMetaSchema,
  claimEvidencePageSchema,
  claimReviewActionRequestSchema,
  comparePayloadSchema,
  costPayloadSchema,
  documentReviewActionRequestSchema,
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
  reviewActionResultSchema,
  reviewAuditPayloadSchema,
  reviewQueuePayloadSchema,
  sourcesPayloadSchema,
  type ApiMeta,
  type Profile,
  type ReviewQueueKind,
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

type ZodSchema<T> = z.ZodType<T, z.ZodTypeDef, unknown>;

/** Wall-clock ceiling for one API attempt (no infinite spinner). */
export const API_TIMEOUT_MS = 20_000;
/** Extra attempts for idempotent GETs only. POSTs (profile payloads) never retry. */
export const API_GET_RETRIES = 2;
const RETRY_BASE_MS = 300;
const RETRY_MAX_DELAY_MS = 3_000;

function isRetryableStatus(status: number): boolean {
  return status === 429 || status === 502 || status === 503 || status === 504;
}

function retryAfterMs(headers: Headers): number | null {
  const raw = headers.get("retry-after");
  if (!raw) return null;
  const seconds = Number(raw);
  if (Number.isFinite(seconds)) return Math.min(RETRY_MAX_DELAY_MS, Math.max(0, seconds * 1000));
  const when = Date.parse(raw);
  if (Number.isNaN(when)) return null;
  return Math.min(RETRY_MAX_DELAY_MS, Math.max(0, when - Date.now()));
}

function backoffMs(attempt: number): number {
  const ceiling = Math.min(RETRY_MAX_DELAY_MS, RETRY_BASE_MS * 2 ** (attempt - 1));
  return Math.round(ceiling / 2 + Math.random() * (ceiling / 2));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function offline(): boolean {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

interface TransportFailure {
  code: string;
  message: string;
  retryable: boolean;
}

function classifyFetchError(error: unknown, external: AbortSignal | undefined, timedOut: boolean): TransportFailure {
  if (timedOut) {
    return {
      code: "network_timeout",
      message: `The API at ${apiBaseUrl()} did not answer within ${Math.round(API_TIMEOUT_MS / 1000)} s. The connection is slow or the API is down.`,
      retryable: true,
    };
  }
  if (external?.aborted) {
    return { code: "request_cancelled", message: "The request was cancelled.", retryable: false };
  }
  if (offline()) {
    return {
      code: "network_unavailable",
      message: `This device is offline. Reconnect and retry; nothing was sent to ${apiBaseUrl()}.`,
      retryable: false,
    };
  }
  return {
    code: "network_error",
    message: `Could not reach the API at ${apiBaseUrl()}: ${
      error instanceof Error ? error.message : "network error"
    }`,
    retryable: true,
  };
}

interface RequestOptions {
  /** Test seam only. Production paths never pass a fetch implementation. */
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
  method?: "GET" | "POST";
  body?: unknown;
  /** Validated before the request leaves the browser (strictly no `any`). */
  bodySchema?: ZodSchema<unknown>;
  /** Extra request headers. Only the review console uses this, for the
   * operator-supplied admin token (X-Admin-Token); it is never persisted. */
  headers?: Record<string, string>;
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
  const method = options.method ?? "GET";
  // Only GETs are retried. A profile POST may be expensive and is not proven
  // idempotent; a blind retry could double-submit if the API ever grows effects.
  const idempotent = method === "GET";
  const maxAttempts = idempotent ? 1 + API_GET_RETRIES : 1;

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

  let response: Response | null = null;
  let transportFailure: TransportFailure | null = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, API_TIMEOUT_MS);
    const onExternalAbort = () => controller.abort();
    options.signal?.addEventListener("abort", onExternalAbort);

    try {
      const attemptResponse = await fetchImpl(`${apiBaseUrl()}${path}`, {
        method,
        headers: {
          accept: "application/json",
          ...(options.body === undefined ? {} : { "content-type": "application/json" }),
          ...(options.headers ?? {}),
        },
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        cache: "no-store",
        signal: controller.signal,
      });

      if (
        idempotent &&
        attempt < maxAttempts &&
        isRetryableStatus(attemptResponse.status)
      ) {
        const delay = retryAfterMs(attemptResponse.headers) ?? backoffMs(attempt);
        await attemptResponse.body?.cancel().catch(() => undefined);
        response = null;
        transportFailure = null;
        await sleep(delay);
        continue;
      }

      response = attemptResponse;
      transportFailure = null;
      break;
    } catch (error) {
      const failure = classifyFetchError(error, options.signal, timedOut);
      transportFailure = failure;
      if (!idempotent || !failure.retryable || attempt >= maxAttempts) break;
      await sleep(backoffMs(attempt));
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", onExternalAbort);
    }
  }

  if (response === null) {
    return failure<T>(endpoint, 0, null, {
      code: transportFailure?.code ?? "network_error",
      message:
        transportFailure?.message ?? `Could not reach the API at ${apiBaseUrl()}.`,
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

/**
 * True when the result means the network/API was unreachable (status 0), as
 * distinct from a 4xx/5xx that actually reached the server. Screens should
 * offer "retry when online" for this state and "something went wrong" for 5xx.
 */
export function isNetworkUnavailable(
  result: ApiResult<unknown>,
): result is Extract<ApiResult<unknown>, { state: "error" }> {
  return (
    result.state === "error" &&
    result.status === 0 &&
    (result.code === "network_unavailable" ||
      result.code === "network_timeout" ||
      result.code === "network_error")
  );
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

/** PLAN v1-w9: the trust surface, generated from `config/` by the API. */
export function apiMethodology(options?: RequestOptions) {
  return request("/api/methodology", methodologyPayloadSchema, options);
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

// --------------------------------------------------------------------------- review console (WS-G)

/**
 * WS-G (brief §18.7). All review endpoints are admin-token gated; the token is
 * supplied per call by the console (in memory, never stored) via
 * `options.headers["X-Admin-Token"]`.
 */
export function apiReviewQueue(
  kind: ReviewQueueKind = "claim",
  params: { limit?: number; offset?: number } = {},
  options?: RequestOptions,
) {
  return request(
    `/api/review/queue${queryString({ kind, limit: params.limit, offset: params.offset })}`,
    reviewQueuePayloadSchema,
    options,
  );
}

/** One queue item's decision trail, newest first. */
export function apiReviewAudit(
  kind: ReviewQueueKind,
  subjectId: number,
  options?: RequestOptions,
) {
  return request(
    `/api/review/audit${queryString({ kind, subject_id: subjectId })}`,
    reviewAuditPayloadSchema,
    options,
  );
}

/**
 * Accept / reject / correct a claim. A correction carries `corrected_quote`
 * and inserts a superseding claim — the original is never mutated.
 */
export function apiReviewClaimAction(
  claimId: number,
  body: { action: "accept" | "reject" | "correct"; corrected_quote?: string; note?: string },
  options?: RequestOptions,
) {
  return request(`/api/review/claims/${claimId}`, reviewActionResultSchema, {
    ...options,
    method: "POST",
    body,
    bodySchema: claimReviewActionRequestSchema,
  });
}

/** Confirm / reject an alias suggestion. Confirming a `manual` alias arms the ER lock. */
export function apiReviewAliasAction(
  aliasId: number,
  body: { action: "confirm" | "reject"; note?: string },
  options?: RequestOptions,
) {
  return request(`/api/review/aliases/${aliasId}`, reviewActionResultSchema, {
    ...options,
    method: "POST",
    body,
    bodySchema: aliasReviewActionRequestSchema,
  });
}

/** Resolve an ambiguous document: link the entity or mark it not applicable. */
export function apiReviewDocumentAction(
  documentId: number,
  body: { action: "link_entity" | "mark_not_applicable"; entity_id?: number; note?: string },
  options?: RequestOptions,
) {
  return request(`/api/review/documents/${documentId}`, reviewActionResultSchema, {
    ...options,
    method: "POST",
    body,
    bodySchema: documentReviewActionRequestSchema,
  });
}
