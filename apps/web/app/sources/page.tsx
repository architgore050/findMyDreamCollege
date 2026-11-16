import Link from "next/link";

import { ApiFailurePanel, EmptyPanel, EnvelopeStamp } from "@/components/api-state";
import { apiSources, isOk } from "@/lib/api";
import { formatNumber } from "@/lib/format";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Sources — findMyDreamCollege",
};

function trustClass(trust: string): string {
  if (trust === "high") return "border-emerald-700 bg-emerald-50 text-emerald-950";
  if (trust === "medium") return "border-sky-700 bg-sky-50 text-sky-950";
  return "border-amber-700 bg-amber-50 text-amber-950";
}

export default async function SourcesPage() {
  const sources = await apiSources();

  if (!isOk(sources)) {
    return (
      <div className="space-y-4">
        <header className="space-y-2">
          <h1 className="text-2xl font-bold tracking-tight">Sources</h1>
          <p className="max-w-3xl text-sm text-stone-700">
            The registry is the only legitimate starting point for a crawl. Nothing on this page is
            evidence by itself; it records where evidence may come from and how much it can be
            trusted.
          </p>
        </header>
        <ApiFailurePanel result={sources} />
      </div>
    );
  }

  const { items, exclusions } = sources.data;
  const enabled = items.filter((source) => source.enabled).length;

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Registry · trust · last crawl</p>
        <h1 className="text-2xl font-bold tracking-tight">Sources</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          Every crawled URL must be attributable to a registry entry or a query template. Politeness
          beats throughput: robots.txt is honoured and rate limits are per registrable domain.
        </p>
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
          <span className="num text-[11px] text-stone-600">
            {formatNumber(enabled)} of {formatNumber(items.length)} enabled ·{" "}
            {formatNumber(exclusions.length)} documented exclusions
          </span>
          <EnvelopeStamp meta={sources.meta} />
        </div>
      </header>

      {items.length === 0 ? (
        <EmptyPanel title="No sources are registered">
          <p>
            The API returned an empty registry. Check{" "}
            <code className="num">config/sources.yaml</code> and the seed/registry loader — an empty
            registry cannot produce evidence.
          </p>
        </EmptyPanel>
      ) : (
        <section className="panel overflow-x-auto" aria-labelledby="registry-title">
          <div className="panel-head">
            <h2 id="registry-title" className="text-sm font-bold">
              Source registry
            </h2>
            <span className="text-[11px] text-stone-600">last crawl: telemetry owned by WS-B</span>
          </div>
          <table>
            <caption className="sr-only">
              Registered sources with trust, adapter, cadence and crawl status
            </caption>
            <thead>
              <tr>
                <th scope="col">Source</th>
                <th scope="col">Kind / format</th>
                <th scope="col">Trust</th>
                <th scope="col">Cadence</th>
                <th scope="col">Adapter</th>
                <th scope="col">Enabled</th>
                <th scope="col">Last crawl</th>
                <th scope="col">Notes</th>
              </tr>
            </thead>
            <tbody>
              {items.map((source) => (
                <tr key={source.id}>
                  <th scope="row" className="min-w-56 font-normal">
                    <span className="block font-semibold">{source.name}</span>
                    <span className="num block text-[10px] text-stone-500">{source.id}</span>
                    {source.base_url ? (
                      <a
                        className="link block break-all text-[10px]"
                        href={source.base_url}
                        target="_blank"
                        rel="noreferrer noopener"
                      >
                        {source.base_url}
                        <span className="sr-only"> (opens in a new tab)</span>
                      </a>
                    ) : (
                      <span className="block text-[10px] text-amber-950">
                        no base URL — per-state discovery / manual review
                      </span>
                    )}
                  </th>
                  <td className="text-[11px]">
                    {source.kind}
                    <span className="block text-stone-500">{source.format}</span>
                  </td>
                  <td>
                    <span
                      className={`inline-flex rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${trustClass(source.trust)}`}
                    >
                      {source.trust}
                    </span>
                  </td>
                  <td className="num text-[11px]">{source.cadence_days} d</td>
                  <td className="num text-[11px]">
                    {source.adapter}
                    {source.requires_manual_review ? (
                      <span className="mt-1 block font-sans text-[10px] font-semibold text-amber-950">
                        requires manual review
                      </span>
                    ) : null}
                  </td>
                  <td className="text-[11px] font-semibold">
                    {source.enabled ? "enabled" : "disabled"}
                  </td>
                  <td className="text-[11px] text-stone-600">
                    {source.last_crawl ?? "no crawl telemetry"}
                  </td>
                  <td className="max-w-72 text-[11px] text-stone-600">{source.notes ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="panel" aria-labelledby="exclusions-title">
        <div className="panel-head">
          <h2 id="exclusions-title" className="text-sm font-bold">
            Documented exclusions
          </h2>
          <span className="text-[11px] text-stone-600">
            “Do not scrape” decisions, with reasons
          </span>
        </div>
        {exclusions.length === 0 ? (
          <p className="px-3 py-3 text-xs text-stone-700">
            No exclusions are recorded. That is unusual: paywalled, login-gated or robots.txt-denied
            sources should be listed here with a reason.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th scope="col">Excluded source</th>
                <th scope="col">Reason</th>
              </tr>
            </thead>
            <tbody>
              {exclusions.map((exclusion) => (
                <tr key={exclusion.id}>
                  <th scope="row" className="font-normal">
                    <span className="block font-semibold">{exclusion.name}</span>
                    <span className="num block text-[10px] text-stone-500">{exclusion.id}</span>
                  </th>
                  <td className="text-[11px]">{exclusion.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <p className="text-xs text-stone-600">
        Trust is not a score. Low-trust sources may inform <code className="num">modelled</code>{" "}
        claims only; they can never be cited as extracted evidence.{" "}
        <Link className="link" href="/shortlist">
          Back to the shortlist
        </Link>
        .
      </p>
    </div>
  );
}
