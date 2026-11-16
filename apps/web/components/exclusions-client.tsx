"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiFailurePanel } from "@/components/api-state";
import { ProfileStorageNote } from "@/components/profile-storage-note";
import { apiExclusions, type ApiResult } from "@/lib/api";
import { formatNumber, titleFromPredicate } from "@/lib/format";
import { useStoredProfile } from "@/lib/use-profile";
import type { ExclusionsPayload } from "@/lib/types";

function Histogram({
  counts,
  label,
  empty,
}: {
  counts: Record<string, number>;
  label: string;
  empty: string;
}) {
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (rows.length === 0) {
    return <p className="px-3 py-2 text-xs text-stone-700">{empty}</p>;
  }
  return (
    <table>
      <caption className="sr-only">{label}</caption>
      <thead>
        <tr>
          <th scope="col">Filter</th>
          <th scope="col">Programmes</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([filterId, count]) => (
          <tr key={filterId}>
            <th scope="row" className="font-normal">
              {titleFromPredicate(filterId)}
              <span className="num block text-[10px] text-stone-500">{filterId}</span>
            </th>
            <td className="num font-semibold">{formatNumber(count)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ExclusionsClient() {
  const profile = useStoredProfile();
  const [result, setResult] = useState<ApiResult<ExclusionsPayload> | null>(null);

  useEffect(() => {
    if (profile === null) return;
    const controller = new AbortController();
    let cancelled = false;
    setResult(null);
    void apiExclusions(profile, { signal: controller.signal }).then((value) => {
      if (!cancelled) setResult(value);
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [profile]);

  return (
    <div className="space-y-4">
      <ProfileStorageNote />

      {result === null ? (
        profile === null ? null : (
          <p role="status" className="text-xs text-stone-600">
            Evaluating the eight hard filters against your profile…
          </p>
        )
      ) : result.state === "ok" ? (
        <>
          <p className="num text-xs text-stone-700">
            {formatNumber(result.data.excluded_total)} excluded of{" "}
            {formatNumber(result.data.evaluated_total)} evaluated. Unknown is neither pass nor fail.
          </p>

          <div className="grid gap-3 lg:grid-cols-2">
            <section className="panel" aria-labelledby="histogram-title">
              <div className="panel-head">
                <h2 id="histogram-title" className="text-sm font-bold">
                  Excluded by filter
                </h2>
                <span className="text-[11px] text-stone-600">fail counts</span>
              </div>
              <Histogram
                counts={result.data.histogram}
                label="Programmes excluded per hard filter"
                empty="No programme failed a hard filter in this run. The histogram is empty, not zero-filled."
              />
            </section>

            <section className="panel" aria-labelledby="unknown-title">
              <div className="panel-head">
                <h2 id="unknown-title" className="text-sm font-bold">
                  Unverified by filter
                </h2>
                <span className="text-[11px] text-stone-600">unknown counts</span>
              </div>
              <Histogram
                counts={result.data.unknown_histogram}
                label="Programmes with an unknown hard-filter outcome"
                empty="Every filter resolved for every programme in this run."
              />
            </section>
          </div>

          <section className="space-y-2" aria-labelledby="excluded-list-title">
            <div>
              <h2 id="excluded-list-title" className="text-lg font-bold">
                Excluded programmes and the claim that decided it
              </h2>
              <p className="text-xs text-stone-600">
                Each row names a failing filter, its deciding claim (when one exists) and the
                observation date. A missing claim is a gap in the exclusion, not a pass.
              </p>
            </div>

            {result.data.excluded.length === 0 ? (
              <section
                className="panel border-dashed px-3 py-3"
                aria-label="No excluded programmes"
              >
                <h2 className="text-sm font-bold">Nothing was excluded in this run</h2>
                <p className="mt-1 text-xs text-stone-700">
                  That means no hard filter failed for the evaluated inventory. It does not mean
                  every programme is a good match; unknown outcomes remain unknown.
                </p>
              </section>
            ) : (
              <ul className="space-y-2">
                {result.data.excluded.map((item) => (
                  <li key={item.programme.id}>
                    <article
                      className="panel border-red-700"
                      aria-labelledby={`excluded-${item.programme.id}`}
                    >
                      <div className="panel-head bg-red-50">
                        <h3 id={`excluded-${item.programme.id}`} className="text-sm font-bold">
                          <Link className="link" href={`/programmes/${item.programme.id}`}>
                            {item.programme.name}
                          </Link>
                        </h3>
                        <span className="num text-[10px] text-red-950">
                          #{item.programme.id} · {item.deciding_filters.length} failing filter
                          {item.deciding_filters.length === 1 ? "" : "s"}
                        </span>
                      </div>
                      <table>
                        <caption className="sr-only">
                          Failing hard filters and their deciding claims
                        </caption>
                        <thead>
                          <tr>
                            <th scope="col">Filter</th>
                            <th scope="col">Deciding claim</th>
                            <th scope="col">Reason</th>
                          </tr>
                        </thead>
                        <tbody>
                          {item.deciding_filters.map((decision) => (
                            <tr key={decision.filter_id}>
                              <th scope="row" className="font-mono text-[11px] font-normal">
                                {decision.filter_id}
                              </th>
                              <td className="num text-[11px]">
                                {decision.claim_id === null
                                  ? "none published"
                                  : `#${String(decision.claim_id)}`}
                              </td>
                              <td className="text-[11px] text-stone-800">
                                {decision.reason}
                                {decision.observed_at ? (
                                  <span className="num block text-[10px] text-stone-500">
                                    observed {decision.observed_at}
                                    {decision.stale ? " · stale" : ""}
                                  </span>
                                ) : null}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </article>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      ) : (
        <ApiFailurePanel result={result} />
      )}
    </div>
  );
}
