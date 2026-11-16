import Link from "next/link";

import { ApiFailurePanel, EmptyPanel } from "@/components/api-state";
import { apiExams, isOk } from "@/lib/api";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Exams — findMyDreamCollege",
};

const STATES = [
  {
    state: "open",
    meaning: "Registration close date is in the future and its observation is within TTL.",
  },
  {
    state: "closed",
    meaning: "Registration close date has passed, using the observed window and its observed_at.",
  },
  {
    state: "unverified",
    meaning: "The window is missing or stale. Never rendered as open and never treated as pass.",
  },
] as const;

export default async function ExamsPage() {
  const exams = await apiExams();

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Registration windows · travel requirement</p>
        <h1 className="text-2xl font-bold tracking-tight">Exams</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          Open, closed and unverified are three different states. A stale calendar is unverified,
          never “probably open”, and a travel-required exam is a hard cutoff when your profile
          forbids travel.
        </p>
      </header>

      {isOk(exams) ? (
        <p className="panel px-3 py-3 text-sm text-stone-700">
          The exams endpoint responded, but this build does not interpret its payload until the WS-F
          response contract is frozen. No exam rows are rendered.
        </p>
      ) : (
        <ApiFailurePanel result={exams} />
      )}

      <section className="panel" aria-labelledby="exam-states-title">
        <div className="panel-head">
          <h2 id="exam-states-title" className="text-sm font-bold">
            How exam state is decided
          </h2>
          <span className="text-[11px] text-stone-600">brief §12, filter 5–6</span>
        </div>
        <table>
          <thead>
            <tr>
              <th scope="col">State</th>
              <th scope="col">Rule</th>
            </tr>
          </thead>
          <tbody>
            {STATES.map((row) => (
              <tr key={row.state}>
                <th scope="row" className="w-32 font-mono text-[11px] font-semibold">
                  {row.state}
                </th>
                <td>{row.meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <EmptyPanel title="No exam calendar is being served">
        <p>
          Registration windows require an observed source claim with{" "}
          <code className="num">observed_at</code> and a TTL. The exam registry exists in{" "}
          <code className="num">config/exams.yaml</code>, but the API endpoint that would serve
          per-profile open/closed status (with travel requirement) is owned by WS-F.
        </p>
        <p className="text-xs">
          When it lands, this table will link each sitting to a past-paper or notice URL only where
          the source is public.{" "}
          <Link className="link" href="/sources">
            Check whether an admissions source is enabled
          </Link>
          .
        </p>
      </EmptyPanel>
    </div>
  );
}
