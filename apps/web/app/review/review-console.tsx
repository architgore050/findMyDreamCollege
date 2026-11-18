"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiFailurePanel } from "@/components/api-state";
import {
  apiReviewAliasAction,
  apiReviewAudit,
  apiReviewClaimAction,
  apiReviewDocumentAction,
  apiReviewQueue,
  isOk,
  type ApiResult,
} from "@/lib/api";
import type {
  ReviewAliasItem,
  ReviewAuditRecord,
  ReviewClaimItem,
  ReviewDocumentItem,
  ReviewQueuePayload,
  ReviewQueueKind,
} from "@/lib/types";

/**
 * The review console (brief §18.7): the last human gate before evidence is
 * published. The admin token lives in React state only — never localStorage,
 * never a cookie — and travels only in the X-Admin-Token header to the API.
 * Every action appends a server-side audit row; the trail is fetched and
 * rendered after each decision.
 */

const QUEUE_ORDER: ReviewQueueKind[] = ["claim", "alias", "document"];

const KIND_LABEL: Record<ReviewQueueKind, string> = {
  claim: "Claims",
  alias: "Alias suggestions",
  document: "Ambiguous documents",
};

export type ReviewQueueItem = ReviewQueuePayload["items"][number];

function shortHash(sha: string): string {
  return sha.length > 12 ? `${sha.slice(0, 12)}…` : sha;
}

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "supported" || status === "confirmed" || status === "linked"
      ? "border-emerald-800 bg-emerald-50 text-emerald-950"
      : status === "rejected" || status === "not_applicable"
        ? "border-red-800 bg-red-50 text-red-950"
        : "border-amber-700 bg-amber-50 text-amber-950";
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ${tone}`}
    >
      {status.replace(/_/g, " ")}
    </span>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap gap-x-1 text-[11px] text-stone-700">
      <dt className="font-semibold text-stone-800">{label}</dt>
      <dd className="num">{children}</dd>
    </div>
  );
}

function AuditTrail({ records }: { records: ReviewAuditRecord[] }) {
  if (records.length === 0) return null;
  return (
    <div className="mt-2 border-t border-stone-200 pt-2" aria-label="Decision trail">
      <h4 className="text-[11px] font-bold uppercase tracking-wide text-stone-600">
        Decision trail ({records.length})
      </h4>
      <ul className="mt-1 space-y-1">
        {records.map((record) => (
          <li key={record.id} className="text-[11px] text-stone-700">
            <span className="num font-semibold">{record.action}</span> by{" "}
            <span className="num">{record.decided_by}</span> at{" "}
            <span className="num">{new Date(record.created_at).toISOString()}</span> — previous{" "}
            <code className="num">{JSON.stringify(record.previous_state)}</code>
            {typeof record.details?.note === "string" && record.details.note ? (
              <span className="block text-stone-600">note: {record.details.note}</span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ClaimCard({
  item,
  busy,
  onAction,
}: {
  item: ReviewClaimItem;
  busy: boolean;
  onAction: (id: number, body: { action: "accept" | "reject" | "correct"; corrected_quote?: string; note?: string }) => void;
}) {
  const [mode, setMode] = useState<"reject" | "correct" | null>(null);
  const [note, setNote] = useState("");
  const [correctedQuote, setCorrectedQuote] = useState("");
  const qualifier = (item.qualifier ?? {}) as { page_no?: unknown; bbox?: unknown };
  const needsNoteWarning = mode === "reject" && item.verification_status === "supported";

  return (
    <article className="panel px-3 py-3" data-testid={`review-claim-${item.id}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-bold">
          <span className="num">#{item.id}</span> · <code className="num">{item.predicate}</code>
        </h3>
        <StatusBadge status={item.verification_status} />
      </div>
      <blockquote className="mt-2 border-l-2 border-stone-400 bg-stone-50 px-2 py-1 text-sm text-stone-900">
        “{item.quote}”
      </blockquote>
      <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <Field label="value">{String(item.value)}{item.unit ? ` ${item.unit}` : ""}</Field>
        <Field label="span">
          {item.char_start}–{item.char_end}
        </Field>
        <Field label="page">
          {typeof qualifier.page_no === "number" ? qualifier.page_no : "—"}
        </Field>
        <Field label="bbox">
          {Array.isArray(qualifier.bbox) ? JSON.stringify(qualifier.bbox) : "—"}
        </Field>
        <Field label="confidence">{item.confidence}</Field>
        <Field label="evidence tier">{item.fetcher_tier ?? "unknown"}</Field>
        <Field label="deciding claim">
          #{item.id} ({item.method})
        </Field>
      </dl>
      <p className="mt-1 text-[11px] text-stone-600">
        Source:{" "}
        {item.url ? (
          <a className="link break-all" href={item.url} rel="noreferrer" target="_blank">
            {item.url}
          </a>
        ) : (
          "unknown"
        )}
        {item.fetched_at ? ` · fetched ${new Date(item.fetched_at).toISOString()}` : ""}
        {item.sha256 ? ` · sha256 ${shortHash(item.sha256)}` : ""}
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          className="rounded border border-emerald-800 bg-emerald-50 px-2 py-1 text-xs font-bold text-emerald-950 disabled:opacity-50"
          disabled={busy}
          onClick={() => onAction(item.id, { action: "accept" })}
          type="button"
        >
          Accept
        </button>
        <button
          className="rounded border border-red-800 bg-red-50 px-2 py-1 text-xs font-bold text-red-950 disabled:opacity-50"
          disabled={busy}
          onClick={() => setMode(mode === "reject" ? null : "reject")}
          type="button"
        >
          Reject
        </button>
        <button
          className="rounded border border-stone-700 bg-white px-2 py-1 text-xs font-bold text-stone-800 disabled:opacity-50"
          disabled={busy}
          onClick={() => setMode(mode === "correct" ? null : "correct")}
          type="button"
        >
          Correct
        </button>
      </div>
      {needsNoteWarning ? (
        <p className="mt-1 text-[11px] font-semibold text-amber-950" role="note">
          Rejecting a supported claim requires a note — the verifier&apos;s verdict must not be
          overturned silently.
        </p>
      ) : null}
      {mode === "reject" ? (
        <div className="mt-2 space-y-1">
          <textarea
            aria-label={`note for claim ${item.id}`}
            className="w-full rounded border border-stone-400 px-2 py-1 text-xs"
            onChange={(event) => setNote(event.target.value)}
            placeholder="Why is this claim wrong? (required when rejecting a supported claim)"
            rows={2}
            value={note}
          />
          <button
            className="rounded border border-red-800 bg-red-50 px-2 py-1 text-xs font-bold text-red-950 disabled:opacity-50"
            disabled={busy}
            onClick={() => {
              onAction(item.id, { action: "reject", note: note || undefined });
              setMode(null);
              setNote("");
            }}
            type="button"
          >
            Submit rejection
          </button>
        </div>
      ) : null}
      {mode === "correct" ? (
        <div className="mt-2 space-y-1">
          <textarea
            aria-label={`corrected quote for claim ${item.id}`}
            className="w-full rounded border border-stone-400 px-2 py-1 text-xs"
            onChange={(event) => setCorrectedQuote(event.target.value)}
            placeholder="Corrected verbatim quote — becomes a new superseding claim; this claim is never edited"
            rows={2}
            value={correctedQuote}
          />
          <textarea
            aria-label={`correction note for claim ${item.id}`}
            className="w-full rounded border border-stone-400 px-2 py-1 text-xs"
            onChange={(event) => setNote(event.target.value)}
            placeholder="What was wrong? (required)"
            rows={2}
            value={note}
          />
          <button
            className="rounded border border-stone-700 bg-white px-2 py-1 text-xs font-bold text-stone-800 disabled:opacity-50"
            disabled={busy}
            onClick={() => {
              onAction(item.id, {
                action: "correct",
                corrected_quote: correctedQuote,
                note: note || undefined,
              });
              setMode(null);
              setNote("");
              setCorrectedQuote("");
            }}
            type="button"
          >
            Submit correction
          </button>
        </div>
      ) : null}
    </article>
  );
}

export function AliasCard({
  item,
  busy,
  onAction,
}: {
  item: ReviewAliasItem;
  busy: boolean;
  onAction: (id: number, action: "confirm" | "reject") => void;
}) {
  return (
    <article className="panel px-3 py-3" data-testid={`review-alias-${item.id}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-bold">
          <span className="num">#{item.id}</span> · “{item.alias}”
        </h3>
        <StatusBadge status={item.review_status} />
      </div>
      <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <Field label="type">{item.alias_type}</Field>
        <Field label="entity">
          {item.entity_name ?? `#${item.entity_id}`}
        </Field>
        <Field label="confidence">{item.confidence ?? "—"}</Field>
      </dl>
      {item.quote ? (
        <blockquote className="mt-1 border-l-2 border-stone-400 bg-stone-50 px-2 py-1 text-xs text-stone-900">
          “{item.quote}”
        </blockquote>
      ) : null}
      {item.alias_type === "manual" ? (
        <p className="mt-1 text-[11px] font-semibold text-amber-950" role="note">
          A confirmed manual alias is authoritative: entity resolution will stop auto-merging this
          pair and resolve the alias to this entity unattended.
        </p>
      ) : null}
      <div className="mt-2 flex flex-wrap gap-2">
        <button
          className="rounded border border-emerald-800 bg-emerald-50 px-2 py-1 text-xs font-bold text-emerald-950 disabled:opacity-50"
          disabled={busy}
          onClick={() => onAction(item.id, "confirm")}
          type="button"
        >
          Confirm
        </button>
        <button
          className="rounded border border-red-800 bg-red-50 px-2 py-1 text-xs font-bold text-red-950 disabled:opacity-50"
          disabled={busy}
          onClick={() => onAction(item.id, "reject")}
          type="button"
        >
          Reject
        </button>
      </div>
    </article>
  );
}

export function DocumentCard({
  item,
  busy,
  onAction,
}: {
  item: ReviewDocumentItem;
  busy: boolean;
  onAction: (id: number, body: { action: "link_entity" | "mark_not_applicable"; entity_id?: number; note?: string }) => void;
}) {
  const [entityId, setEntityId] = useState("");
  const parsedEntityId = Number.parseInt(entityId, 10);
  return (
    <article className="panel px-3 py-3" data-testid={`review-document-${item.id}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-bold">
          <span className="num">#{item.id}</span> ·{" "}
          <a className="link break-all" href={item.url} rel="noreferrer" target="_blank">
            {item.url}
          </a>
        </h3>
        <StatusBadge status={item.plan_state ?? "unknown"} />
      </div>
      <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <Field label="family">{item.extraction_family ?? "—"}</Field>
        <Field label="plan confidence">{item.plan_confidence ?? "—"}</Field>
        <Field label="linked entity">{item.entity_id ?? "none"}</Field>
      </dl>
      {item.plan_reason ? (
        <p className="mt-1 text-[11px] text-stone-700">Planner: {item.plan_reason}</p>
      ) : null}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input
          aria-label={`entity id for document ${item.id}`}
          className="w-28 rounded border border-stone-400 px-2 py-1 text-xs"
          inputMode="numeric"
          onChange={(event) => setEntityId(event.target.value)}
          placeholder="entity id"
          value={entityId}
        />
        <button
          className="rounded border border-emerald-800 bg-emerald-50 px-2 py-1 text-xs font-bold text-emerald-950 disabled:opacity-50"
          disabled={busy || Number.isNaN(parsedEntityId)}
          onClick={() => {
            onAction(item.id, { action: "link_entity", entity_id: parsedEntityId });
            setEntityId("");
          }}
          type="button"
        >
          Link entity
        </button>
        <button
          className="rounded border border-stone-700 bg-white px-2 py-1 text-xs font-bold text-stone-800 disabled:opacity-50"
          disabled={busy}
          onClick={() => onAction(item.id, { action: "mark_not_applicable" })}
          type="button"
        >
          Not applicable
        </button>
      </div>
    </article>
  );
}

export function ReviewQueueView({
  kind,
  payload,
  busyId,
  actionError,
  auditsBySubject,
  onPage,
  onClaimAction,
  onAliasAction,
  onDocumentAction,
}: {
  kind: ReviewQueueKind;
  payload: ReviewQueuePayload;
  busyId: number | null;
  actionError: string | null;
  auditsBySubject: Record<number, ReviewAuditRecord[]>;
  onPage: (delta: 1 | -1) => void;
  onClaimAction: (id: number, body: { action: "accept" | "reject" | "correct"; corrected_quote?: string; note?: string }) => void;
  onAliasAction: (id: number, action: "confirm" | "reject") => void;
  onDocumentAction: (id: number, body: { action: "link_entity" | "mark_not_applicable"; entity_id?: number; note?: string }) => void;
}) {
  const hasPrev = payload.offset > 0;
  const hasNext = payload.offset + payload.items.length < payload.total;
  return (
    <section className="space-y-3" aria-labelledby="review-queue-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="review-queue-title" className="text-lg font-bold">
          {KIND_LABEL[kind]} awaiting a decision
        </h2>
        <p className="num text-[11px] text-stone-600">
          {payload.total} queued · showing {payload.offset}–
          {payload.offset + payload.items.length} · order: {payload.ordering ?? "queue order"}
        </p>
      </div>

      {actionError ? (
        <p className="border border-red-700 bg-red-50 px-3 py-2 text-xs font-semibold text-red-950" role="alert">
          {actionError}
        </p>
      ) : null}

      {payload.items.length === 0 ? (
        <p className="panel border-dashed px-3 py-3 text-sm text-stone-700">
          The {kind} queue is empty. Nothing is awaiting a human decision right now — either the
          machine work is done or nothing has been extracted yet.
        </p>
      ) : (
        <div className="space-y-3">
          {payload.items.map((item) => {
            const audit = auditsBySubject[item.id] ?? [];
            const trail = <AuditTrail records={audit} />;
            if (kind === "claim") {
              return (
                <div key={`claim-${item.id}`}>
                  <ClaimCard
                    busy={busyId === item.id}
                    item={item as ReviewClaimItem}
                    onAction={onClaimAction}
                  />
                  {trail}
                </div>
              );
            }
            if (kind === "alias") {
              return (
                <div key={`alias-${item.id}`}>
                  <AliasCard
                    busy={busyId === item.id}
                    item={item as ReviewAliasItem}
                    onAction={onAliasAction}
                  />
                  {trail}
                </div>
              );
            }
            return (
              <div key={`document-${item.id}`}>
                <DocumentCard
                  busy={busyId === item.id}
                  item={item as ReviewDocumentItem}
                  onAction={onDocumentAction}
                />
                {trail}
              </div>
            );
          })}
        </div>
      )}

      <nav aria-label="Queue pagination" className="flex flex-wrap gap-2">
        <button
          className="rounded border border-stone-700 bg-white px-2 py-1 text-xs font-bold text-stone-800 disabled:opacity-50"
          disabled={!hasPrev || busyId !== null}
          onClick={() => onPage(-1)}
          type="button"
        >
          ← Previous page
        </button>
        <button
          className="rounded border border-stone-700 bg-white px-2 py-1 text-xs font-bold text-stone-800 disabled:opacity-50"
          disabled={!hasNext || busyId !== null}
          onClick={() => onPage(1)}
          type="button"
        >
          Next page →
        </button>
      </nav>
    </section>
  );
}

export function ReviewConsole() {
  const [token, setToken] = useState("");
  const [tokenSubmitted, setTokenSubmitted] = useState(false);
  const [kind, setKind] = useState<ReviewQueueKind>("claim");
  const [offset, setOffset] = useState(0);
  const [result, setResult] = useState<ApiResult<ReviewQueuePayload> | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [auditsBySubject, setAuditsBySubject] = useState<Record<number, ReviewAuditRecord[]>>({});

  const load = useCallback(
    async (nextKind: ReviewQueueKind, nextOffset: number, adminToken: string) => {
      const value = await apiReviewQueue(
        nextKind,
        { limit: 20, offset: nextOffset },
        { headers: { "X-Admin-Token": adminToken } },
      );
      setResult(value);
      if (isOk(value)) {
        const audits: Record<number, ReviewAuditRecord[]> = {};
        await Promise.all(
          value.data.items.map(async (item) => {
            const trail = await apiReviewAudit(nextKind, item.id, {
              headers: { "X-Admin-Token": adminToken },
            });
            if (isOk(trail)) audits[item.id] = trail.data.items;
          }),
        );
        setAuditsBySubject(audits);
      }
    },
    [],
  );

  useEffect(() => {
    if (!tokenSubmitted || token.length === 0) return;
    const controller = new AbortController();
    setResult(null);
    void load(kind, offset, token);
    return () => {
      controller.abort();
    };
  }, [kind, load, offset, token, tokenSubmitted]);

  const afterAction = useCallback(
    async (subjectId: number) => {
      await load(kind, offset, token);
      const trail = await apiReviewAudit(kind, subjectId, {
        headers: { "X-Admin-Token": token },
      });
      if (isOk(trail)) {
        setAuditsBySubject((previous) => ({ ...previous, [subjectId]: trail.data.items }));
      }
    },
    [kind, load, offset, token],
  );

  const handleClaimAction = useCallback(
    async (
      id: number,
      body: { action: "accept" | "reject" | "correct"; corrected_quote?: string; note?: string },
    ) => {
      setBusyId(id);
      setActionError(null);
      const value = await apiReviewClaimAction(id, body, {
        headers: { "X-Admin-Token": token },
      });
      setBusyId(null);
      if (!isOk(value)) {
        setActionError(value.message);
        return;
      }
      // Idempotent repeats refresh the trail without appending decisions.
      await afterAction(id);
    },
    [afterAction, token],
  );

  const handleAliasAction = useCallback(
    async (id: number, action: "confirm" | "reject") => {
      setBusyId(id);
      setActionError(null);
      const value = await apiReviewAliasAction(id, { action }, {
        headers: { "X-Admin-Token": token },
      });
      setBusyId(null);
      if (!isOk(value)) {
        setActionError(value.message);
        return;
      }
      if (!value.data.already_done) {
        // Confirmed/rejected items leave the pending queue; the trail shows why.
        await afterAction(id);
      }
    },
    [afterAction, token],
  );

  const handleDocumentAction = useCallback(
    async (
      id: number,
      body: { action: "link_entity" | "mark_not_applicable"; entity_id?: number; note?: string },
    ) => {
      setBusyId(id);
      setActionError(null);
      const value = await apiReviewDocumentAction(id, body, {
        headers: { "X-Admin-Token": token },
      });
      setBusyId(null);
      if (!isOk(value)) {
        setActionError(value.message);
        return;
      }
      if (!value.data.already_done) await afterAction(id);
    },
    [afterAction, token],
  );

  if (!tokenSubmitted) {
    return (
      <section className="panel px-3 py-3" aria-label="Admin token gate">
        <h2 className="text-sm font-bold">Admin token required</h2>
        <p className="mt-1 text-xs text-stone-700">
          Every review endpoint is gated by the API&apos;s admin token (no user accounts in v1).
          Paste it once; it stays in this page&apos;s memory only — never stored, never logged —
          and is sent as <code className="num">X-Admin-Token</code> to the API alone.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <input
            aria-label="admin token"
            className="w-72 max-w-full rounded border border-stone-400 px-2 py-1 text-xs"
            onChange={(event) => setToken(event.target.value)}
            type="password"
            value={token}
          />
          <button
            className="rounded border border-stone-700 bg-white px-2 py-1 text-xs font-bold text-stone-800 disabled:opacity-50"
            disabled={token.length === 0}
            onClick={() => setTokenSubmitted(true)}
            type="button"
          >
            Load queue
          </button>
        </div>
        <p className="mt-1 text-[11px] text-stone-600">
          A wrong token earns HTTP 403 — the console has no retry-with-credentials flow by design.
        </p>
      </section>
    );
  }

  return (
    <div className="space-y-4">
      <nav aria-label="Queue kind" className="flex flex-wrap gap-2">
        {QUEUE_ORDER.map((candidate) => (
          <button
            key={candidate}
            className={`rounded border px-2 py-1 text-xs font-bold ${
              candidate === kind
                ? "border-stone-900 bg-stone-900 text-white"
                : "border-stone-700 bg-white text-stone-800"
            }`}
            onClick={() => {
              setKind(candidate);
              setOffset(0);
              setActionError(null);
            }}
            type="button"
          >
            {KIND_LABEL[candidate]}
          </button>
        ))}
        <button
          className="rounded border border-stone-400 bg-white px-2 py-1 text-xs font-bold text-stone-700"
          onClick={() => setTokenSubmitted(false)}
          type="button"
        >
          Forget token
        </button>
      </nav>

      {result === null ? (
        <p role="status" className="text-xs text-stone-600">
          Loading the {kind} queue…
        </p>
      ) : isOk(result) ? (
        <ReviewQueueView
          actionError={actionError}
          auditsBySubject={auditsBySubject}
          busyId={busyId}
          kind={kind}
          onAliasAction={(id, action) => void handleAliasAction(id, action)}
          onClaimAction={(id, body) => void handleClaimAction(id, body)}
          onDocumentAction={(id, body) => void handleDocumentAction(id, body)}
          onPage={(delta) => setOffset((value) => Math.max(0, value + delta * 20))}
          payload={result.data}
        />
      ) : (
        <ApiFailurePanel result={result} />
      )}
    </div>
  );
}
