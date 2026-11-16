import Link from "next/link";

import { ApiFailurePanel, EmptyPanel } from "@/components/api-state";
import { ProfileSummary } from "@/components/profile-summary";
import { apiProgrammes, apiRankings, isOk } from "@/lib/api";
import { formatNumber } from "@/lib/format";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Shortlist — findMyDreamCollege",
};

export default async function ShortlistPage() {
  // The profile never leaves the browser: the endpoint is probed with an empty
  // profile to report availability, while ranking inputs are supplied later by
  // the client once WS-F ships `/api/rankings`.
  const [rankings, programmes] = await Promise.all([apiRankings({}), apiProgrammes({ limit: 25 })]);

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Step 2 of 3 · ranked shortlist</p>
        <h1 className="text-2xl font-bold tracking-tight">Shortlist</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          A programme appears here only after hard filters pass. Composite scores are bands, never
          decimals, and every published number links to the claim that proves it.
        </p>
      </header>

      <ProfileSummary />

      {isOk(rankings) ? (
        <section className="panel px-3 py-3" aria-labelledby="rankings-ok-title">
          <h2 id="rankings-ok-title" className="text-sm font-bold">
            Ranking endpoint responded
          </h2>
          <p className="mt-1 text-xs text-stone-700">
            The API returned a ranked payload, but this build only renders the documented programme
            records below. Ranked card fields (dominance, DSR₂₅, insight sets) are not interpreted
            here until the payload contract is frozen by WS-F.
          </p>
        </section>
      ) : (
        <ApiFailurePanel result={rankings} />
      )}

      <section className="space-y-2" aria-labelledby="inventory-title">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 id="inventory-title" className="text-lg font-bold">
              Unranked inventory
            </h2>
            <p className="text-xs text-stone-600">
              Database order, not a ranking. No programme here has been scored or filtered yet.
            </p>
          </div>
          <Link className="link text-xs" href="/sources">
            Where does inventory come from?
          </Link>
        </div>

        {isOk(programmes) ? (
          programmes.data.items.length === 0 ? (
            <EmptyPanel title="No programmes are seeded">
              <p>
                The programmes table is empty. Load the registry and inventory before expecting a
                shortlist: <code className="num">make migrate</code>, then{" "}
                <code className="num">make seed</code>. An empty table is not evidence that no
                programmes exist.
              </p>
            </EmptyPanel>
          ) : (
            <>
              <p className="num text-[11px] text-stone-600">
                Showing {programmes.data.items.length} of {formatNumber(programmes.data.total)}{" "}
                programmes
              </p>
              <div className="panel overflow-x-auto">
                <table>
                  <caption className="sr-only">
                    Unranked programme inventory from the database
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">id</th>
                      <th scope="col">Programme</th>
                      <th scope="col">Credential</th>
                      <th scope="col">Recognition</th>
                      <th scope="col">Duration</th>
                      <th scope="col">Internship</th>
                      <th scope="col">Seats</th>
                    </tr>
                  </thead>
                  <tbody>
                    {programmes.data.items.map((programme) => (
                      <tr key={programme.id}>
                        <td className="num">{programme.id}</td>
                        <th scope="row" className="font-normal">
                          <Link className="link" href={`/programmes/${programme.id}`}>
                            {programme.name}
                          </Link>
                        </th>
                        <td>{programme.credential_level}</td>
                        <td>
                          <span
                            className={
                              programme.recognition_status === "recognised"
                                ? "font-semibold text-emerald-900"
                                : programme.recognition_status === "unrecognised"
                                  ? "font-semibold text-red-900"
                                  : "font-semibold text-amber-900"
                            }
                          >
                            {programme.recognition_status}
                          </span>
                        </td>
                        <td className="num">
                          {programme.duration_months === null ||
                          programme.duration_months === undefined
                            ? "unknown"
                            : `${programme.duration_months} mo`}
                        </td>
                        <td className="num">
                          {programme.internship_months === null ||
                          programme.internship_months === undefined
                            ? "unknown"
                            : `${programme.internship_months} mo`}
                        </td>
                        <td className="num">
                          {programme.seats === null || programme.seats === undefined
                            ? "unknown"
                            : formatNumber(programme.seats)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-stone-600">
                Internship months are part of the academic programme, never counted as work
                experience (brief §4).
              </p>
            </>
          )
        ) : (
          <ApiFailurePanel result={programmes} />
        )}
      </section>
    </div>
  );
}
