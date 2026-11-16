"use client";

import { useEffect, useState } from "react";

import { ApiFailurePanel } from "@/components/api-state";
import { HardFilterRow, type FilterState } from "@/components/hard-filter";
import { apiProgrammeCost, apiProgrammeFilters, type ApiResult } from "@/lib/api";
import { formatAmount } from "@/lib/format";
import { useStoredProfile } from "@/lib/use-profile";
import {
  scenarioTotalsSchema,
  type CostPayload,
  type MoneyBand,
  type ProgrammeFiltersPayload,
  type ScenarioTotals,
} from "@/lib/types";

const SCENARIO_ORDER = ["hostel", "independent"] as const;

function bandText(band: MoneyBand | undefined): string {
  if (!band || band.value === null) return "UNKNOWN";
  const point = formatAmount(band.value, band.currency);
  if (band.lo === null || band.hi === null) return point;
  return `${point} (${formatAmount(band.lo, band.currency)}–${formatAmount(band.hi, band.currency)})`;
}

function assumedFlags(flags: Record<string, unknown> | undefined): { key: string; note: string }[] {
  if (!flags) return [];
  const entries: { key: string; note: string }[] = [];
  for (const [key, raw] of Object.entries(flags)) {
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) continue;
    const record = raw as Record<string, unknown>;
    if (record["assumed"] !== true) continue;
    entries.push({ key, note: typeof record["note"] === "string" ? record["note"] : "" });
  }
  return entries;
}

function ScenarioBlock({ totals }: { totals: ScenarioTotals }) {
  const label = totals.living === "hostel" ? "Hostel scenario" : "Independent scenario";
  const dsr = totals.dsr25;
  const gap = totals.affordability_gap;
  return (
    <section
      className="rounded border border-stone-300 bg-white"
      aria-labelledby={`scenario-totals-${totals.living}`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-stone-200 px-3 py-2">
        <h3 id={`scenario-totals-${totals.living}`} className="text-sm font-bold">
          {label}
        </h3>
        <span className="text-[10px] font-semibold uppercase tracking-wide text-stone-600">
          complete alternative
        </span>
      </div>
      <table>
        <caption className="sr-only">
          Financing metrics for the {totals.living} scenario. Never added to the other scenario.
        </caption>
        <tbody>
          <tr>
            <th scope="row" className="font-normal">
              TCOA (whole programme)
            </th>
            <td className="num font-semibold">{bandText(totals.tcoa)}</td>
          </tr>
          <tr>
            <th scope="row" className="font-normal">
              Family out of pocket
            </th>
            <td className="num font-semibold">{bandText(totals.family_out_of_pocket)}</td>
          </tr>
          <tr>
            <th scope="row" className="font-normal">
              EMI (monthly)
            </th>
            <td className="num font-semibold">{bandText(totals.emi)}</td>
          </tr>
          <tr>
            <th scope="row" className="font-normal">
              DSR₂₅
            </th>
            <td className="num font-semibold">
              {dsr.value === null ? (
                <span className="text-amber-950">UNKNOWN — not a zero</span>
              ) : (
                `${dsr.value.toFixed(3)} (${dsr.band})`
              )}
            </td>
          </tr>
          <tr>
            <th scope="row" className="font-normal">
              Affordability gap
            </th>
            <td className="num font-semibold">
              {gap.value === null ? (
                <span className="text-amber-950">UNKNOWN — income not supplied</span>
              ) : (
                <>
                  {gap.value.toFixed(2)}
                  {gap.flagged ? " · flagged" : ""}
                </>
              )}
            </td>
          </tr>
        </tbody>
      </table>
      <p className="border-t border-stone-200 px-3 py-2 text-[11px] text-stone-600">
        {totals.emi.method === "unknown_rate"
          ? "EMI and DSR₂₅ need a loan rate; none was available, so both stay UNKNOWN."
          : `EMI method: ${totals.emi.method}. ${dsr.formula}.`}
      </p>
    </section>
  );
}

function CostSection({ result }: { result: ApiResult<CostPayload> }) {
  if (result.state !== "ok") return <ApiFailurePanel result={result} />;
  const cost = result.data;

  if (cost.needs_profile || !cost.available) {
    return (
      <section className="panel border-dashed px-3 py-3" aria-label="Cost needs a profile">
        <h3 className="text-sm font-bold">No family-specific cost was published</h3>
        <p className="mt-1 text-xs text-stone-700">
          {cost.reason ?? "The API did not compute a total cost of attendance for this programme."}{" "}
          No number has been invented to fill the gap.
        </p>
        {cost.unknown_reasons && cost.unknown_reasons.length > 0 ? (
          <ul className="mt-2 list-disc space-y-0.5 pl-5 text-[11px] text-amber-950">
            {cost.unknown_reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        ) : null}
      </section>
    );
  }

  const totals = SCENARIO_ORDER.flatMap((name) => {
    const raw = cost.scenario_totals[name];
    if (raw === undefined) return [];
    const parsed = scenarioTotalsSchema.safeParse(raw);
    return parsed.success ? [parsed.data] : [];
  });
  const assumed = assumedFlags(cost.assumed);

  return (
    <div className="space-y-3">
      <p className="border border-stone-400 bg-white px-3 py-1.5 text-[11px] font-semibold text-stone-800">
        {cost.note ?? "Hostel and independent are complete alternatives; never sum them."}
      </p>

      {cost.unknown_reasons && cost.unknown_reasons.length > 0 ? (
        <section
          className="border border-amber-700 bg-amber-50 px-3 py-2"
          aria-label="Values that stay unknown"
        >
          <h3 className="text-[11px] font-bold uppercase tracking-wide text-amber-950">
            Still UNKNOWN by design
          </h3>
          <ul className="mt-1 space-y-0.5">
            {cost.unknown_reasons.map((reason) => (
              <li key={reason} className="text-[11px] text-amber-950">
                {reason}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {assumed.length > 0 ? (
        <section
          className="border border-amber-700 bg-amber-50 px-3 py-2"
          aria-label="Assumed values"
        >
          <h3 className="text-[11px] font-bold uppercase tracking-wide text-amber-950">
            Assumed values and terms (not extracted evidence)
          </h3>
          <ul className="mt-1 space-y-0.5">
            {assumed.map((flag) => (
              <li key={flag.key} className="text-[11px] text-amber-950">
                <code className="num">{flag.key}</code>
                {flag.note ? ` — ${flag.note}` : ""}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <div className="grid gap-3 lg:grid-cols-2">
        {totals.map((scenario) => (
          <ScenarioBlock key={scenario.living} totals={scenario} />
        ))}
      </div>
    </div>
  );
}

function FiltersSection({ result }: { result: ApiResult<ProgrammeFiltersPayload> }) {
  if (result.state !== "ok") return <ApiFailurePanel result={result} />;
  const payload = result.data;

  if (payload.needs_profile) {
    return (
      <section className="panel border-dashed px-3 py-3" aria-label="Filters need a profile">
        <h3 className="text-sm font-bold">No filter was evaluated</h3>
        <p className="mt-1 text-xs text-stone-700">
          {payload.reason ??
            "Hard filters compare this programme with a profile. Without one, every state would be a guess."}
        </p>
      </section>
    );
  }

  return (
    <div className="space-y-2">
      <p
        className={`border px-3 py-1.5 text-[11px] font-semibold ${
          payload.would_exclude
            ? "border-red-800 bg-red-50 text-red-950"
            : "border-stone-400 bg-white text-stone-800"
        }`}
      >
        {payload.would_exclude
          ? `This programme is excluded for the stored profile: ${(payload.excluded ?? []).join(", ")}.`
          : "No hard filter fails for the stored profile. Unknown outcomes are not passes."}
        {payload.counts
          ? ` (${payload.counts.pass} pass · ${payload.counts.fail} fail · ${payload.counts.unknown} unknown)`
          : ""}
      </p>
      <div className="panel overflow-x-auto">
        <table>
          <caption className="sr-only">
            Hard filter outcomes for this programme against the stored profile
          </caption>
          <thead>
            <tr>
              <th scope="col">Filter</th>
              <th scope="col">State</th>
              <th scope="col">Deciding claim / reason</th>
            </tr>
          </thead>
          <tbody>
            {payload.filters.map((decision) => (
              <HardFilterRow
                key={decision.filter_id}
                name={decision.filter_id}
                state={decision.result as FilterState}
                reason={decision.reason}
                claimLink={
                  decision.claim_id === null ? undefined : (
                    <span className="num text-[10px] text-stone-600">
                      deciding claim #{String(decision.claim_id)}
                      {decision.observed_at ? ` · observed ${decision.observed_at}` : ""}
                      {decision.stale ? " · stale" : ""}
                    </span>
                  )
                }
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * Profile-aware programme drill-down panels.
 *
 * The profile is read from this browser's localStorage and sent only as a POST
 * body to the local API. It never appears in the URL, a log line or analytics.
 * Numbers come from the API; this component formats them and never recomputes a
 * TCOA, EMI or DSR.
 */
export function ProgrammeProfilePanels({ programmeId }: { programmeId: number }) {
  const profile = useStoredProfile();
  const [cost, setCost] = useState<ApiResult<CostPayload> | null>(null);
  const [filters, setFilters] = useState<ApiResult<ProgrammeFiltersPayload> | null>(null);

  useEffect(() => {
    if (profile === null) return;
    const controller = new AbortController();
    let cancelled = false;
    setCost(null);
    setFilters(null);
    void apiProgrammeCost(programmeId, profile, "hostel", {
      signal: controller.signal,
    }).then((value) => {
      if (!cancelled) setCost(value);
    });
    void apiProgrammeFilters(programmeId, profile, { signal: controller.signal }).then((value) => {
      if (!cancelled) setFilters(value);
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [profile, programmeId]);

  const answered = profile === null ? 0 : Object.keys(profile).length;

  return (
    <>
      <section className="space-y-2" aria-labelledby="filters-title">
        <div>
          <h2 id="filters-title" className="text-lg font-bold">
            Hard filters
          </h2>
          <p className="text-xs text-stone-600">
            Binary cutoffs evaluated against your stored profile. A failed filter excludes the
            programme; it never lowers a score inside a weighted sum. Unknown is not pass.
          </p>
        </div>
        {profile === null ? (
          <p className="text-xs text-stone-600">Reading the profile stored in this browser…</p>
        ) : answered === 0 ? (
          <p className="border border-amber-700 bg-amber-50 px-3 py-2 text-[11px] text-amber-950">
            Your stored profile is empty. The eight filters are still evaluated, but with nothing to
            compare they return UNKNOWN. No “typical” profile is invented on your behalf.
          </p>
        ) : null}
        {filters === null ? (
          profile === null ? null : (
            <p role="status" className="text-xs text-stone-600">
              Evaluating the eight hard filters against your profile…
            </p>
          )
        ) : (
          <FiltersSection result={filters} />
        )}
      </section>

      <section className="space-y-2" aria-labelledby="cost-title">
        <div>
          <h2 id="cost-title" className="text-lg font-bold">
            Cost scenarios
          </h2>
          <p className="text-xs text-stone-600">
            Both living scenarios are computed from the same cost evidence. They are complete
            alternatives, never added together. DSR₂₅ and the affordability gap use only the family
            financing you supplied; assumed loan terms are labelled.
          </p>
        </div>
        {cost === null ? (
          profile === null ? null : (
            <p role="status" className="text-xs text-stone-600">
              Computing TCOA, EMI and DSR₂₅ for both living scenarios…
            </p>
          )
        ) : (
          <CostSection result={cost} />
        )}
      </section>
    </>
  );
}
