import Link from "next/link";

import { ApiFailurePanel, EmptyPanel } from "@/components/api-state";
import { HardFilterRow, type FilterState } from "@/components/hard-filter";
import { apiProgrammes, isOk } from "@/lib/api";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Exclusions — findMyDreamCollege",
};

const FILTERS = [
  {
    name: "credential_level",
    rule: "degree required; lateral entry only when the profile allows it",
  },
  { name: "recognition_status", rule: "NCAHP / state council / affiliation chain must resolve" },
  { name: "eligibility", rule: "10+2 PCB aggregate and age against the profile" },
  { name: "language", rule: "required test minimum, or medium-of-instruction exemption" },
  {
    name: "exam_registration_open",
    rule: "now < registration_closes on the observed window + TTL",
  },
  { name: "exam_travel", rule: "travel-required exam when the profile forbids travel" },
  { name: "intake_year_available", rule: "profile's target intake must exist for the programme" },
  { name: "budget_ceiling", rule: "fail only when the TCOA lower bound exceeds the ceiling" },
] as const;

export default async function ExclusionsPage() {
  const programmes = await apiProgrammes({ limit: 1 });
  const exclusions = isOk(programmes) ? programmes.data.exclusions : undefined;

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Binary cutoffs · deciding claim required</p>
        <h1 className="text-2xl font-bold tracking-tight">Exclusions</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          A programme disappears from the shortlist only because a hard filter failed. Every
          exclusion must name the filter and the claim that decided it. Unknown is not silently
          treated as pass, and nothing is excluded for a low score.
        </p>
      </header>

      {!isOk(programmes) ? <ApiFailurePanel result={programmes} /> : null}

      <section className="panel" aria-labelledby="histogram-title">
        <div className="panel-head">
          <h2 id="histogram-title" className="text-sm font-bold">
            Exclusion histogram
          </h2>
          <span className="rounded border border-amber-700 bg-amber-50 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-amber-950">
            not published
          </span>
        </div>
        <div className="space-y-2 px-3 py-3 text-sm text-stone-800">
          <p>
            The API reports{" "}
            <code className="num">
              exclusions:{" "}
              {exclusions
                ? JSON.stringify({ available: exclusions.available, owner: exclusions.owner })
                : "unavailable"}
            </code>
            . No programme has a published exclusion decision yet, so this page cannot name a count,
            a reason histogram or a deciding claim.
          </p>
          <p className="text-xs text-stone-600">
            N programmes excluded: <strong>not reported</strong>. That is an evidence gap, not zero
            exclusions. Ownership: {exclusions?.owner ?? "WS-F"}.
          </p>
          <p className="text-xs">
            <Link className="link" href="/shortlist">
              Return to the unranked inventory
            </Link>{" "}
            or{" "}
            <Link className="link" href="/sources">
              inspect source coverage
            </Link>
            .
          </p>
        </div>
      </section>

      <section className="space-y-2" aria-labelledby="filter-catalogue-title">
        <div>
          <h2 id="filter-catalogue-title" className="text-lg font-bold">
            Filter catalogue
          </h2>
          <p className="text-xs text-stone-600">
            All eight cutoffs, in evaluation order. Current state is unknown for every programme
            because WS-F has not published filter decisions.
          </p>
        </div>
        <div className="panel overflow-x-auto">
          <table>
            <caption className="sr-only">
              Hard filter catalogue and current evaluation state
            </caption>
            <thead>
              <tr>
                <th scope="col">Filter</th>
                <th scope="col">State</th>
                <th scope="col">Deciding rule</th>
              </tr>
            </thead>
            <tbody>
              {FILTERS.map((filter) => (
                <HardFilterRow
                  key={filter.name}
                  name={filter.name}
                  state={"unknown" as FilterState}
                  reason={`${filter.rule}. No deciding claim has been published.`}
                />
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {isOk(programmes) && programmes.data.items.length === 0 ? (
        <EmptyPanel title="No programmes in the database">
          <p>
            Exclusions require an inventory to exclude from. Run{" "}
            <code className="num">make seed</code> and the inventory adapters first.
          </p>
        </EmptyPanel>
      ) : null}
    </div>
  );
}
