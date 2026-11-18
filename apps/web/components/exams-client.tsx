"use client";

import { useEffect, useState } from "react";

import { ApiFailurePanel } from "@/components/api-state";
import { ProfileStorageNote } from "@/components/profile-storage-note";
import { formatDate } from "@/lib/format";
import { apiExams, type ApiResult } from "@/lib/api";
import { useStoredProfile } from "@/lib/use-profile";
import type { ExamCycle, ExamStatus, ExamWindowsPayload, NextExamDeadline } from "@/lib/types";

const STATUS_CLASSES: Record<ExamStatus, string> = {
  open: "border-emerald-800 bg-emerald-50 text-emerald-950",
  closed: "border-red-800 bg-red-50 text-red-950",
  unverified: "border-amber-700 bg-amber-50 text-amber-950",
};

function StatusBadge({ status }: { status: ExamStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ${STATUS_CLASSES[status]}`}
    >
      {status}
    </span>
  );
}

/** Cycles coexist per exam (spec 17 §2); the deciding one carries the badge. */
function CycleChips({ cycles }: { cycles: ExamCycle[] | undefined }) {
  if (!cycles || cycles.length === 0) {
    return <span className="text-[11px] text-stone-500">no observed cycle</span>;
  }
  return (
    <ul className="space-y-1">
      {cycles.map((cycle) => (
        <li key={cycle.cycle_year} className="num text-[11px]">
          <span className="font-semibold">{cycle.cycle_year}</span>
          {cycle.status ? (
            <span
              className={`ml-1 font-semibold ${
                cycle.status === "open"
                  ? "text-emerald-900"
                  : cycle.status === "closed"
                    ? "text-red-900"
                    : "text-amber-900"
              }`}
            >
              {cycle.status}
            </span>
          ) : null}
          {cycle.registration_closes ? (
            <span className="text-stone-700">
              {" "}
              · closes {cycle.date_text ?? cycle.registration_closes.slice(0, 10)}
            </span>
          ) : null}
          {cycle.precision === "month" ? (
            <span className="ml-1 rounded border border-amber-700 bg-amber-50 px-1 text-[10px] font-semibold text-amber-950">
              month precision — verify
            </span>
          ) : null}
          {cycle.stale ? (
            <span className="ml-1 text-[10px] font-semibold text-amber-950">stale</span>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

function NextDeadlines({ deadlines }: { deadlines: NextExamDeadline[] }) {
  return (
    <section className="panel overflow-x-auto" aria-labelledby="next-deadlines-title">
      <div className="panel-head">
        <h2 id="next-deadlines-title" className="text-sm font-bold">
          Next actionable deadlines (for your profile)
        </h2>
        <span className="text-[11px] text-stone-600">sorted ascending · spec 17</span>
      </div>
      <table>
        <caption className="sr-only">
          Exam registration deadlines still open to this profile, earliest first
        </caption>
        <thead>
          <tr>
            <th scope="col">Exam</th>
            <th scope="col">Cycle</th>
            <th scope="col">Act before</th>
            <th scope="col">Status</th>
            <th scope="col">Claim</th>
          </tr>
        </thead>
        <tbody>
          {deadlines.map((entry) => (
            <tr
              key={`${entry.exam_id}-${entry.cycle_year}`}
              data-testid={`deadline-${entry.exam_id}`}
            >
              <th scope="row" className="font-normal">
                {entry.name}
                <span className="block text-[10px] text-stone-500">{entry.exam_id}</span>
              </th>
              <td className="num text-[11px]">{entry.cycle_year}</td>
              <td className="num whitespace-nowrap text-[11px]">
                {entry.deadline_month
                  ? `within ${entry.deadline_month} (no day printed)`
                  : formatDate(entry.deadline)}
                {entry.stale ? (
                  <span className="block text-[10px] font-semibold text-amber-950">
                    window unverified — verify first
                  </span>
                ) : null}
              </td>
              <td>
                <StatusBadge status={entry.status} />
              </td>
              <td className="num text-[11px]">
                {entry.claim_id === null ? "none published" : `#${String(entry.claim_id)}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function ExamsClient() {
  const profile = useStoredProfile();
  const [result, setResult] = useState<ApiResult<ExamWindowsPayload> | null>(null);

  useEffect(() => {
    if (profile === null) return;
    const controller = new AbortController();
    let cancelled = false;
    setResult(null);
    void apiExams(profile, { signal: controller.signal }).then((value) => {
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
            Checking registration windows against your travel preference…
          </p>
        )
      ) : result.state === "ok" ? (
        result.data.items.length === 0 ? (
          <section className="panel border-dashed px-3 py-3" aria-label="No exam calendar">
            <h2 className="text-sm font-bold">No exam windows are being served</h2>
            <p className="mt-1 text-xs text-stone-700">
              The registry has no enabled exam with an observable window. That is missing evidence,
              not an open window.
            </p>
          </section>
        ) : (
          <>
            {result.data.next_deadlines && result.data.next_deadlines.length > 0 ? (
              <NextDeadlines deadlines={result.data.next_deadlines} />
            ) : null}
            <section className="space-y-2" aria-labelledby="exam-table-title">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <div>
                  <h2 id="exam-table-title" className="text-lg font-bold">
                    Registration windows
                  </h2>
                  <p className="text-xs text-stone-600">
                    Decided from an observed window plus its TTL. A travel-required exam is closed
                    to a profile that forbids travel, regardless of the date. A month-precision
                    window is never completed into a day — it renders “verify” inside that month.
                  </p>
                </div>
                <p className="num text-[11px] text-stone-600">
                  {result.data.counts.open} open · {result.data.counts.closed} closed ·{" "}
                  {result.data.counts.unverified} unverified
                </p>
              </div>

              <div className="panel overflow-x-auto">
                <table>
                  <caption className="sr-only">
                    Exam registration status, observation freshness, travel requirement and the
                    claim that decided it
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Exam</th>
                      <th scope="col">Status</th>
                      <th scope="col">Observed</th>
                      <th scope="col">Cycles</th>
                      <th scope="col">Travel</th>
                      <th scope="col">Deciding claim</th>
                      <th scope="col">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.data.items.map((exam) => (
                      <tr key={exam.exam_id} data-testid={`exam-${exam.exam_id}`}>
                        <th scope="row" className="font-normal">
                          {exam.name}
                          <span className="block text-[10px] text-stone-500">
                            {exam.exam_id} · {exam.country}
                            {exam.scope ? ` · ${exam.scope}` : ""}
                          </span>
                        </th>
                        <td>
                          <StatusBadge status={exam.status} />
                          {exam.stale ? (
                            <span className="mt-1 block text-[10px] font-semibold text-amber-950">
                              stale (&gt; {exam.ttl_days}d)
                            </span>
                          ) : null}
                        </td>
                        <td className="num whitespace-nowrap text-[11px]">
                          {formatDate(exam.observed_at)}
                        </td>
                        <td>
                          <CycleChips cycles={exam.cycles} />
                        </td>
                        <td className="text-[11px]">
                          {exam.requires_travel ? "required" : "not required"}
                          {exam.remote_options ? " (remote available)" : ""}
                          {exam.travel_blocked ? (
                            <span className="mt-1 block font-semibold text-red-950">
                              blocked: profile forbids travel
                            </span>
                          ) : null}
                        </td>
                        <td className="num text-[11px]">
                          {exam.deciding_claim_id === null
                            ? "none published"
                            : `#${String(exam.deciding_claim_id)}`}
                        </td>
                        <td className="text-[11px] text-stone-700">{exam.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {result.data.profile_persisted ? (
                <p className="border border-red-700 bg-red-50 px-3 py-2 text-xs font-semibold text-red-950">
                  The API reports <code className="num">profile_persisted: true</code>; this build
                  expects false.
                </p>
              ) : null}
            </section>
          </>
        )
      ) : (
        <ApiFailurePanel result={result} />
      )}
    </div>
  );
}
