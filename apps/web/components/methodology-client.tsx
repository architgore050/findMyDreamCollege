"use client";

import { useEffect, useState } from "react";

import { ApiFailurePanel } from "@/components/api-state";
import { EnvelopeStamp } from "@/components/api-state";
import { apiMethodology, type ApiResult } from "@/lib/api";
import type { MethodologyPayload } from "@/lib/types";

/**
 * `/methodology` body (PLAN v1-w9). Every number is fetched from
 * `/api/methodology`, which generates the payload from `config/` at request
 * time — the page hard-codes nothing. UNKNOWN is a value here too: a failed
 * fetch renders a named failure state, never a partial page with invented
 * weights.
 */
export function MethodologyClient() {
  const [result, setResult] = useState<ApiResult<MethodologyPayload> | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    void apiMethodology({ signal: controller.signal }).then((value) => {
      if (!cancelled) setResult(value);
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  if (result === null) {
    return (
      <p role="status" className="text-xs text-stone-600">
        Reading the scoring configuration…
      </p>
    );
  }
  if (result.state !== "ok") {
    return <ApiFailurePanel result={result} />;
  }
  const data = result.data;

  return (
    <div className="space-y-5">
      <EnvelopeStamp meta={result.meta} />
      <section className="panel px-3 py-3" aria-labelledby="weights-title">
        <h2 id="weights-title" className="text-lg font-bold">
          Sub-score weights
        </h2>
        <p className="mt-1 text-xs text-stone-600">
          Generated from <code className="num">config/weights.yaml</code> at request
          time (weights version {result.meta.weights_version}, scorer{" "}
          {result.meta.scorer_version}). The sum is {String(data.weights_sum)}.
        </p>
        <table className="mt-2 w-full max-w-xl">
          <caption className="sr-only">Sub-score weights from the registry</caption>
          <thead>
            <tr>
              <th scope="col">Sub-score</th>
              <th scope="col">Weight</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.weights).map(([name, weight]) => (
              <tr key={name}>
                <th scope="row" className="font-normal">
                  {name}
                </th>
                <td className="num">{String(weight)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <dl className="mt-3 grid grid-cols-2 gap-2 text-xs text-stone-700 sm:grid-cols-4">
          <div>
            <dt className="font-semibold">min claims for a score</dt>
            <dd className="num">{String(data.component_policy.min_claims_for_score)}</dd>
          </div>
          <div>
            <dt className="font-semibold">min components for a composite</dt>
            <dd className="num">{String(data.component_policy.min_components_for_composite)}</dd>
          </div>
          <div>
            <dt className="font-semibold">confidence threshold (claims)</dt>
            <dd className="num">{String(data.component_policy.min_claims_for_confidence)}</dd>
          </div>
          <div>
            <dt className="font-semibold">single-claim half-width</dt>
            <dd className="num">
              {String(data.component_policy.single_claim_interval_half_width)}
            </dd>
          </div>
        </dl>
        <p className="mt-2 text-xs text-stone-600">
          Predicate registry version {data.predicates_version}: {String(data.predicates_count)}{" "}
          predicates, {String(data.predicate_aliases)} aliases. Sources:{" "}
          {String(data.sources.enabled)} enabled of {String(data.sources.total)}.
        </p>
      </section>

      <section className="panel px-3 py-3" aria-labelledby="policy-title">
        <h2 id="policy-title" className="text-lg font-bold">
          How a number becomes a number
        </h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-stone-800">
          {data.value_policy.map((line) => (
            <li key={line.slice(0, 40)}>{line}</li>
          ))}
        </ul>
      </section>

      <section className="panel px-3 py-3" aria-labelledby="dispute-title">
        <h2 id="dispute-title" className="text-lg font-bold">
          Disputing a number
        </h2>
        <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-stone-800">
          {data.dispute_path.map((line) => (
            <li key={line.slice(0, 40)}>{line}</li>
          ))}
        </ol>
        <p className="mt-2 text-xs text-stone-600">
          See also{" "}
          <a className="underline" href="/sources">
            the source registry
          </a>{" "}
          and the{" "}
          <a className="underline" href="/exclusions">
            exclusion registry
          </a>
          .
        </p>
      </section>

      <section className="panel px-3 py-3" aria-labelledby="exclusions-title">
        <h2 id="exclusions-title" className="text-lg font-bold">
          Excluded sources (append-only registry)
        </h2>
        {data.source_exclusions.length === 0 ? (
          <p className="mt-1 text-xs text-stone-600">No source is excluded.</p>
        ) : (
          <ul className="mt-2 space-y-1 text-sm text-stone-800">
            {data.source_exclusions.map((item) => (
              <li key={item.id}>
                <span className="font-semibold">{item.name}</span>{" "}
                <span className="text-xs text-stone-500">({item.id})</span> — {item.reason}
              </li>
            ))}
          </ul>
        )}
        <p className="mt-2 text-xs text-stone-600">
          This registry is append-only: corrections are appended under a new id, and
          history is never rewritten.
        </p>
      </section>
    </div>
  );
}
