"use client";

import { useEffect, useState } from "react";

import { ApiFailurePanel, EnvelopeStamp } from "@/components/api-state";
import { ProfileStorageNote } from "@/components/profile-storage-note";
import { ProfileSummary } from "@/components/profile-summary";
import { RankingsView } from "@/components/rankings-view";
import { apiRankings, type ApiResult } from "@/lib/api";
import { unknownProfileInputs } from "@/lib/profile-unknowns";
import { useStoredProfile } from "@/lib/use-profile";
import type { RankingsPayload } from "@/lib/types";

/**
 * `/shortlist` body. The profile is read from this browser's localStorage and
 * sent as a POST body; it is never placed in the URL. Ranking runs even for an
 * empty profile — the response is honest about which inputs were UNKNOWN.
 */
export function ShortlistClient() {
  const profile = useStoredProfile();
  const [result, setResult] = useState<ApiResult<RankingsPayload> | null>(null);

  useEffect(() => {
    if (profile === null) return;
    const controller = new AbortController();
    let cancelled = false;
    setResult(null);
    void apiRankings(profile, { signal: controller.signal }).then((value) => {
      if (!cancelled) setResult(value);
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [profile]);

  const unknownInputs = profile === null ? [] : unknownProfileInputs(profile);

  return (
    <div className="space-y-4">
      <ProfileStorageNote />
      <ProfileSummary profile={profile} />

      {profile === null ? (
        <p className="text-xs text-stone-600">Ranking will run once the stored profile is read.</p>
      ) : (
        <section
          className={`rounded border px-3 py-2 text-xs ${
            unknownInputs.length === 0
              ? "border-emerald-700 bg-emerald-50 text-emerald-950"
              : "border-amber-700 bg-amber-50 text-amber-950"
          }`}
          aria-labelledby="unknown-inputs-title"
        >
          <h2 id="unknown-inputs-title" className="text-xs font-bold">
            {unknownInputs.length === 0
              ? "Every ranking input is answered"
              : `Ranking anyway — ${unknownInputs.length} input${
                  unknownInputs.length === 1 ? " is" : "s are"
                } UNKNOWN`}
          </h2>
          {unknownInputs.length === 0 ? (
            <p className="mt-0.5">
              Each sub-score still carries its own interval and claim count. A filled profile does
              not make thin evidence thick.
            </p>
          ) : (
            <>
              <p className="mt-0.5">
                UNKNOWN is a value here, never 0 and never 50. These inputs widen the affected
                sub-score intervals instead of silently defaulting:
              </p>
              <ul className="num mt-1 flex flex-wrap gap-x-3 gap-y-1">
                {unknownInputs.map((input) => (
                  <li key={input.key} data-testid={`unknown-input-${input.key}`}>
                    {input.label}
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}

      {result === null ? (
        profile === null ? null : (
          <p role="status" className="text-xs text-stone-600">
            Computing the ranked shortlist from the evidence currently stored…
          </p>
        )
      ) : result.state === "ok" ? (
        <>
          <EnvelopeStamp meta={result.meta} />
          <RankingsView payload={result.data} meta={result.meta} />
        </>
      ) : (
        <ApiFailurePanel result={result} />
      )}
    </div>
  );
}
