import Link from "next/link";

import { ApiFailurePanel, EmptyPanel } from "@/components/api-state";
import { CompareForm } from "@/components/compare-form";
import { ScoreBand } from "@/components/score-band";
import { apiCompare, apiProgramme, isOk, type ApiResult } from "@/lib/api";
import type { ProgrammeDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Compare — findMyDreamCollege",
};

function parseIds(raw: string | string[] | undefined): number[] {
  const text = Array.isArray(raw) ? raw.join(",") : (raw ?? "");
  const ids = text
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);
  return Array.from(new Set(ids)).slice(0, 6);
}

export default async function ComparePage({
  searchParams,
}: {
  searchParams: Promise<{ ids?: string | string[] }>;
}) {
  const { ids: rawIds } = await searchParams;
  const ids = parseIds(rawIds);
  const initialIds = ids.join(", ");

  const [compare, details] = await Promise.all([
    ids.length > 0 ? apiCompare(ids) : Promise.resolve(null),
    Promise.all(ids.map((id) => apiProgramme(id))),
  ]);

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Evidence comparison</p>
        <h1 className="text-2xl font-bold tracking-tight">Compare</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          Comparison is Pareto dominance, not a total order: A is shown as better than B only when A
          is at least as good on every known sub-score and strictly better on one. Undominated pairs
          are “not comparable on current evidence”.
        </p>
      </header>

      <CompareForm initialIds={initialIds} />

      {ids.length === 0 ? (
        <EmptyPanel title="No programmes selected" showSourcesLink={false}>
          <p>
            Add programme ids above. Ids are shown on each{" "}
            <Link className="link" href="/shortlist">
              inventory row
            </Link>{" "}
            and in the drill-down breadcrumb.
          </p>
        </EmptyPanel>
      ) : null}

      {compare ? (
        <>
          <section className="space-y-2" aria-labelledby="pareto-title">
            <div>
              <h2 id="pareto-title" className="text-lg font-bold">
                Pareto matrix
              </h2>
              <p className="text-xs text-stone-600">
                Dominance, differing claims and confidence are computed by the API. No local
                substitute is fabricated when the endpoint is absent.
              </p>
            </div>
            {!isOk(compare) ? (
              <ApiFailurePanel result={compare} />
            ) : (
              <p className="panel px-3 py-3 text-sm text-stone-700">
                The compare endpoint responded successfully, but this build does not interpret its
                payload until the WS-F response contract is frozen. No dominance claim is rendered.
              </p>
            )}
          </section>

          <section className="space-y-2" aria-labelledby="side-by-side-title">
            <div>
              <h2 id="side-by-side-title" className="text-lg font-bold">
                Stored records, side by side
              </h2>
              <p className="text-xs text-stone-600">
                These are programme fields and evidence volumes, not scores. “unknown” is a value.
              </p>
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {details.map((detail, index) => (
                <ComparisonCard
                  key={ids[index] ?? index}
                  programmeId={ids[index] ?? 0}
                  result={detail}
                />
              ))}
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}

function ComparisonCard({
  programmeId,
  result,
}: {
  programmeId: number;
  result: ApiResult<ProgrammeDetail>;
}) {
  if (!isOk(result)) {
    if (result.state === "error" && result.code === "not_found") {
      return (
        <section className="panel px-3 py-3" aria-label={`Programme ${programmeId}`}>
          <h3 className="num text-sm font-bold">Programme {programmeId}</h3>
          <p className="mt-1 text-xs text-stone-700">No record exists for this id.</p>
        </section>
      );
    }
    return (
      <div>
        <h3 className="num mb-1 text-sm font-bold">Programme {programmeId}</h3>
        <ApiFailurePanel result={result} />
      </div>
    );
  }

  const detail = result.data;
  const programme = detail.programme;
  const institution = detail.institution;
  return (
    <section className="panel" aria-labelledby={`compare-${programme.id}`}>
      <div className="panel-head">
        <h3 id={`compare-${programme.id}`} className="text-sm font-bold">
          <Link className="link" href={`/programmes/${programme.id}`}>
            {programme.name}
          </Link>
        </h3>
        <span className="num text-[10px] text-stone-500">#{programme.id}</span>
      </div>
      <div className="px-3 py-3">
        <ScoreBand
          lo={detail.score?.composite_lo}
          hi={detail.score?.composite_hi}
          tier={detail.score?.evidence_tier}
          label="composite band"
          scorerVersion={detail.score?.scorer_version ?? result.meta.scorer_version}
          weightsVersion={detail.score?.weights_version ?? result.meta.weights_version}
        />
        <table className="mt-2">
          <tbody>
            <tr>
              <th scope="row">Institution</th>
              <td>{institution?.canonical_name ?? "unresolved"}</td>
            </tr>
            <tr>
              <th scope="row">Location</th>
              <td>
                {institution
                  ? [institution.city, institution.state].filter(Boolean).join(", ") || "unknown"
                  : "unknown"}
              </td>
            </tr>
            <tr>
              <th scope="row">Credential</th>
              <td>{programme.credential_level}</td>
            </tr>
            <tr>
              <th scope="row">Recognition</th>
              <td>{programme.recognition_status}</td>
            </tr>
            <tr>
              <th scope="row">Duration</th>
              <td className="num">
                {programme.duration_months == null ? "unknown" : `${programme.duration_months} mo`}
              </td>
            </tr>
            <tr>
              <th scope="row">Internship</th>
              <td className="num">
                {programme.internship_months == null
                  ? "unknown"
                  : `${programme.internship_months} mo`}
              </td>
            </tr>
            <tr>
              <th scope="row">Claims</th>
              <td className="num">{detail.claims.length}</td>
            </tr>
            <tr>
              <th scope="row">Cost items</th>
              <td className="num">{detail.costs.length}</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-2 text-[11px] text-amber-950">
          Not comparable on current evidence: Pareto dominance is computed by WS-F and is not
          available. No band overlap across cards implies a winner.
        </p>
      </div>
    </section>
  );
}
