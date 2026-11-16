import Link from "next/link";

import { ExclusionsClient } from "@/components/exclusions-client";

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

export default function ExclusionsPage() {
  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Binary cutoffs · deciding claim required</p>
        <h1 className="text-2xl font-bold tracking-tight">Exclusions</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          A programme disappears from the shortlist only because a hard filter failed. Every
          exclusion names the filter and, where one exists, the claim that decided it. Unknown is
          not silently treated as pass, and nothing is excluded for a low score.
        </p>
      </header>

      <ExclusionsClient />

      <section className="space-y-2" aria-labelledby="filter-catalogue-title">
        <div>
          <h2 id="filter-catalogue-title" className="text-lg font-bold">
            Filter catalogue
          </h2>
          <p className="text-xs text-stone-600">
            All eight cutoffs, in evaluation order. The counts above come from the current run
            against the profile stored in this browser.
          </p>
        </div>
        <div className="panel overflow-x-auto">
          <table>
            <caption className="sr-only">Hard filter catalogue and deciding rule</caption>
            <thead>
              <tr>
                <th scope="col">Filter</th>
                <th scope="col">Deciding rule</th>
              </tr>
            </thead>
            <tbody>
              {FILTERS.map((filter) => (
                <tr key={filter.name}>
                  <th scope="row" className="font-mono text-[11px] font-normal text-stone-700">
                    {filter.name}
                  </th>
                  <td>{filter.rule}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs">
          <Link className="link" href="/shortlist">
            Return to the ranked shortlist
          </Link>{" "}
          or{" "}
          <Link className="link" href="/sources">
            inspect source coverage
          </Link>
          .
        </p>
      </section>
    </div>
  );
}
