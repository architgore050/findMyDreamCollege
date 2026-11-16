"use client";

import { useEffect, useState } from "react";

import { ApiFailurePanel } from "@/components/api-state";
import { ProfileStorageNote } from "@/components/profile-storage-note";
import { formatDate } from "@/lib/format";
import { apiExams, type ApiResult } from "@/lib/api";
import { useStoredProfile } from "@/lib/use-profile";
import type { ExamStatus, ExamWindowsPayload } from "@/lib/types";

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
          <section className="space-y-2" aria-labelledby="exam-table-title">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div>
                <h2 id="exam-table-title" className="text-lg font-bold">
                  Registration windows
                </h2>
                <p className="text-xs text-stone-600">
                  Decided from an observed window plus its TTL. A travel-required exam is closed to
                  a profile that forbids travel, regardless of the date.
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
                  Exam registration status, observation freshness, travel requirement and the claim
                  that decided it
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Exam</th>
                    <th scope="col">Status</th>
                    <th scope="col">Observed</th>
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
        )
      ) : (
        <ApiFailurePanel result={result} />
      )}
    </div>
  );
}
