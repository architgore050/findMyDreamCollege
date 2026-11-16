import Link from "next/link";

import { formatDate } from "@/lib/format";
import type { ApiResult, ApiOk } from "@/lib/api";
import type { ApiMeta } from "@/lib/types";

type Failure = Exclude<ApiResult<never>, { state: "ok" }>;

export function EnvelopeStamp({ meta }: { meta: ApiMeta | null }) {
  if (!meta) {
    return (
      <p className="text-[11px] text-stone-600">
        No evidence envelope was returned. Treat every number on this page as unverified.
      </p>
    );
  }
  return (
    <dl className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-stone-600">
      <div className="flex gap-1">
        <dt className="font-semibold">scorer</dt>
        <dd className="num">{meta.scorer_version}</dd>
      </div>
      <div className="flex gap-1">
        <dt className="font-semibold">weights</dt>
        <dd className="num">{meta.weights_version}</dd>
      </div>
      <div className="flex gap-1">
        <dt className="font-semibold">claims</dt>
        <dd className="num" title={meta.claims_hash}>
          {meta.claims_hash.slice(0, 12)}…
        </dd>
      </div>
      <div className="flex gap-1">
        <dt className="font-semibold">generated</dt>
        <dd className="num">{formatDate(meta.generated_at)}</dd>
      </div>
    </dl>
  );
}

export function OkStamp<T>({ result }: { result: ApiOk<T> }) {
  return <EnvelopeStamp meta={result.meta} />;
}

/**
 * Renders any non-ok API result as an explicit, readable state. A 501 is never
 * a spinner and never a crash: it names the owning workstream.
 */
export function ApiFailurePanel({ result }: { result: Failure }) {
  if (result.state === "not_implemented") {
    return (
      <section className="panel border-amber-700" aria-label="Not yet available">
        <div className="panel-head bg-amber-50">
          <h2 className="text-sm font-bold text-amber-950">
            Not yet available — owned by {result.workstream}
          </h2>
          <span className="rounded border border-amber-700 bg-white px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-amber-900">
            HTTP 501
          </span>
        </div>
        <div className="space-y-2 px-3 py-3 text-sm text-stone-800">
          <p>{result.message}</p>
          <p className="text-xs text-stone-600">
            Endpoint <code className="num">{result.endpoint}</code> is declared but not implemented
            in this build. This screen will fill in when that workstream lands; no placeholder
            numbers are shown meanwhile.
          </p>
          <p className="text-xs">
            <Link className="link" href="/sources">
              Check the source registry and crawl coverage
            </Link>
          </p>
        </div>
      </section>
    );
  }

  if (result.state === "database_unavailable") {
    return (
      <section className="panel border-amber-700" aria-label="Database unavailable">
        <div className="panel-head bg-amber-50">
          <h2 className="text-sm font-bold text-amber-950">Database unavailable</h2>
          <span className="rounded border border-amber-700 bg-white px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-amber-900">
            HTTP 503
          </span>
        </div>
        <div className="space-y-2 px-3 py-3 text-sm text-stone-800">
          <p>{result.message}</p>
          <ol className="list-decimal space-y-0.5 pl-5 text-xs text-stone-700">
            <li>
              <code className="num">docker compose -f infra/docker-compose.yml up -d db</code>
            </li>
            <li>
              <code className="num">make migrate</code>
            </li>
            <li>
              <code className="num">make seed</code>, then reload this page.
            </li>
          </ol>
          <p className="text-xs">
            Inventory{" "}
            <Link className="link" href="/sources">
              source status
            </Link>{" "}
            remains readable while the database is down.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="panel border-red-700" aria-label="Request failed">
      <div className="panel-head bg-red-50">
        <h2 className="text-sm font-bold text-red-950">Request failed</h2>
        <span className="num rounded border border-red-700 bg-white px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-red-900">
          {result.status === 0 ? "network" : `HTTP ${result.status}`} · {result.code}
        </span>
      </div>
      <div className="space-y-2 px-3 py-3 text-sm text-stone-800">
        <p>{result.message}</p>
        <p className="text-xs text-stone-600">
          Endpoint <code className="num">{result.endpoint}</code>. Check that the API at{" "}
          <code className="num">{process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}</code>{" "}
          is running.
        </p>
      </div>
    </section>
  );
}

export function EmptyPanel({
  title,
  children,
  showSourcesLink = true,
}: {
  title: string;
  children: React.ReactNode;
  showSourcesLink?: boolean;
}) {
  return (
    <section className="panel border-dashed" aria-label={title}>
      <div className="panel-head">
        <h2 className="text-sm font-bold">{title}</h2>
      </div>
      <div className="space-y-2 px-3 py-3 text-sm text-stone-800">
        {children}
        {showSourcesLink ? (
          <p className="text-xs">
            What is missing is evidence, not a programme.{" "}
            <Link className="link" href="/sources">
              Inspect the source registry
            </Link>{" "}
            to see which families are enabled and where crawl coverage stops.
          </p>
        ) : null}
      </div>
    </section>
  );
}
