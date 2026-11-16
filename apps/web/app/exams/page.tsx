import { ExamsClient } from "@/components/exams-client";

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

export default function ExamsPage() {
  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Registration windows · travel requirement</p>
        <h1 className="text-2xl font-bold tracking-tight">Exams</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          Open, closed and unverified are three different states. A stale calendar is unverified,
          never “probably open”, and a travel-required exam is a hard cutoff when your profile
          forbids travel. Your travel preference is read from this browser and sent only in the
          request body.
        </p>
      </header>

      <ExamsClient />

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
    </div>
  );
}
