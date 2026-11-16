import Link from "next/link";

import { ApiFailurePanel, EmptyPanel, EnvelopeStamp } from "@/components/api-state";
import { EvidenceTierBadge } from "@/components/badges";
import { ClaimValue } from "@/components/claim-value";
import { CostPanel } from "@/components/cost-panel";
import { EvidenceTabs } from "@/components/evidence-tabs";
import { HardFilterRow, type FilterState } from "@/components/hard-filter";
import { ScoreBand } from "@/components/score-band";
import { apiProgramme, apiProgrammeCost, apiProgrammeEvidence, isOk } from "@/lib/api";
import { formatNumber, titleFromPredicate } from "@/lib/format";
import { subScoreSchema, type Claim, type ProgrammeDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

const HARD_FILTERS = [
  "credential_level",
  "recognition_status",
  "eligibility",
  "language",
  "exam_registration_open",
  "exam_travel",
  "intake_year_available",
  "budget_ceiling",
] as const;

function mergeClaims(detail: ProgrammeDetail, evidence: Claim[]): Claim[] {
  const byId = new Map<number, Claim>();
  for (const claim of detail.claims) byId.set(claim.id, claim);
  for (const claim of evidence) {
    const existing = byId.get(claim.id);
    byId.set(claim.id, { ...existing, ...claim });
  }
  return Array.from(byId.values());
}

export default async function ProgrammePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const programmeId = Number(id);
  if (!Number.isInteger(programmeId) || programmeId <= 0) {
    return (
      <EmptyPanel title="Invalid programme id" showSourcesLink={false}>
        <p>
          Programme ids are positive integers. Go back to the{" "}
          <Link className="link" href="/shortlist">
            shortlist
          </Link>
          .
        </p>
      </EmptyPanel>
    );
  }

  const [detail, evidence, cost] = await Promise.all([
    apiProgramme(programmeId),
    apiProgrammeEvidence(programmeId, { limit: 200 }),
    apiProgrammeCost(programmeId, "hostel"),
  ]);

  if (!isOk(detail)) {
    if (detail.state === "error" && detail.code === "not_found") {
      return (
        <EmptyPanel title={`Programme ${programmeId} not found`}>
          <p>The API has no record with this id. It may not be seeded, or the id may be wrong.</p>
        </EmptyPanel>
      );
    }
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold tracking-tight">Programme {programmeId}</h1>
        <ApiFailurePanel result={detail} />
      </div>
    );
  }

  const data = detail.data;
  const claims = mergeClaims(data, isOk(evidence) ? evidence.data.items : []);
  const institution = data.institution;
  const score = data.score;
  const subScores = score
    ? Object.entries(score.sub_scores).flatMap(([name, raw]) => {
        const parsed = subScoreSchema.safeParse(raw);
        return parsed.success ? [{ name, ...parsed.data }] : [];
      })
    : [];
  const knownCount = data.claims.length;
  const evidenceTotal = isOk(evidence) ? evidence.data.total : knownCount;

  return (
    <div className="space-y-5">
      <nav aria-label="Breadcrumb" className="text-xs text-stone-600">
        <Link className="link" href="/shortlist">
          Shortlist
        </Link>{" "}
        / <span aria-current="page">Programme {data.programme.id}</span>
      </nav>

      <header className="space-y-2">
        <p className="kicker">
          {institution?.city ?? "city unknown"} · {institution?.state ?? "state unknown"} ·{" "}
          {institution?.country ?? "country unknown"}
        </p>
        <h1 className="text-2xl font-bold leading-tight tracking-tight">{data.programme.name}</h1>
        <p className="text-sm text-stone-700">
          {institution ? (
            <Link className="link" href={`/programmes/${programmeId}`} aria-current="page">
              {institution.canonical_name}
            </Link>
          ) : (
            "Awarding institution not resolved"
          )}
        </p>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <span className="rounded border border-stone-400 bg-white px-1.5 py-0.5 font-semibold uppercase tracking-wide">
            {data.programme.credential_level}
          </span>
          <span
            className={`rounded border px-1.5 py-0.5 font-semibold uppercase tracking-wide ${
              data.programme.recognition_status === "recognised"
                ? "border-emerald-700 bg-emerald-50 text-emerald-950"
                : data.programme.recognition_status === "unrecognised"
                  ? "border-red-700 bg-red-50 text-red-950"
                  : "border-amber-700 bg-amber-50 text-amber-950"
            }`}
          >
            recognition: {data.programme.recognition_status}
          </span>
          <span className="num text-stone-600">
            {formatNumber(claims.length)} claims · {formatNumber(evidenceTotal)} in evidence index
          </span>
        </div>
        <EnvelopeStamp meta={detail.meta} />
      </header>

      <div className="grid gap-4 lg:grid-cols-3">
        <section className="panel lg:col-span-2" aria-labelledby="facts-title">
          <div className="panel-head">
            <h2 id="facts-title" className="text-sm font-bold">
              Programme facts
            </h2>
            <span className="text-[11px] text-stone-600">
              stored fields, not claim-backed scores
            </span>
          </div>
          <p className="border-b border-stone-200 px-3 py-2 text-[11px] text-amber-950">
            These values come from the programme record, not from a resolved claim. They are shown
            verbatim for orientation and are not treated as evidence for any score. Claim-backed
            numbers appear under Evidence with a source link or an explicit “no source” warning.
          </p>
          <table>
            <caption className="sr-only">Programme record fields and their values</caption>
            <tbody>
              <tr>
                <th scope="row" className="w-1/3">
                  Duration
                </th>
                <td className="num">
                  {data.programme.duration_months === null ||
                  data.programme.duration_months === undefined
                    ? "unknown"
                    : `${data.programme.duration_months} months`}
                </td>
              </tr>
              <tr>
                <th scope="row">Internship</th>
                <td className="num">
                  {data.programme.internship_months === null ||
                  data.programme.internship_months === undefined
                    ? "unknown"
                    : `${data.programme.internship_months} months`}
                  <span className="ml-2 text-[10px] font-normal text-stone-600">
                    academic component; never employment
                  </span>
                </td>
              </tr>
              <tr>
                <th scope="row">Seats</th>
                <td className="num">
                  {data.programme.seats === null || data.programme.seats === undefined
                    ? "unknown"
                    : formatNumber(data.programme.seats)}
                </td>
              </tr>
              <tr>
                <th scope="row">Medium</th>
                <td>{data.programme.medium ?? "unknown"}</td>
              </tr>
              <tr>
                <th scope="row">Affiliated university id</th>
                <td className="num">
                  {data.programme.affiliated_university_id === null ||
                  data.programme.affiliated_university_id === undefined
                    ? "not linked"
                    : data.programme.affiliated_university_id}
                </td>
              </tr>
              <tr>
                <th scope="row">Recognition chain</th>
                <td>
                  {data.programme.recognition_status}
                  <span className="block text-[11px] text-stone-600">
                    NCAHP / state council / university affiliation chain resolution is evaluated by
                    WS-F. The stored status is shown verbatim.
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="panel" aria-labelledby="band-title">
          <div className="panel-head">
            <h2 id="band-title" className="text-sm font-bold">
              Composite
            </h2>
            {score ? <EvidenceTierBadge tier={score.evidence_tier} /> : null}
          </div>
          <div className="space-y-2 px-3 py-3">
            <ScoreBand
              lo={score?.composite_lo}
              hi={score?.composite_hi}
              tier={score?.evidence_tier}
              scorerVersion={score?.scorer_version ?? detail.meta.scorer_version}
              weightsVersion={score?.weights_version ?? detail.meta.weights_version}
            />
            {score ? (
              <p className="num text-[10px] text-stone-500">
                computed {score.computed_at ?? "unknown"} · claims {score.claims_hash.slice(0, 12)}…
              </p>
            ) : (
              <p className="text-[11px] text-stone-600">
                No score row exists for this institution. Ranking is not available, and a blank
                composite is not a low score.
              </p>
            )}
          </div>
        </section>
      </div>

      <section className="space-y-2" aria-labelledby="filters-title">
        <div>
          <h2 id="filters-title" className="text-lg font-bold">
            Hard filters
          </h2>
          <p className="text-xs text-stone-600">
            Binary cutoffs. A failed filter excludes the programme; it never lowers a score inside a
            weighted sum.
          </p>
        </div>
        <div className="panel overflow-x-auto">
          <table>
            <caption className="sr-only">Hard filter outcomes for this programme</caption>
            <thead>
              <tr>
                <th scope="col">Filter</th>
                <th scope="col">State</th>
                <th scope="col">Deciding claim / reason</th>
              </tr>
            </thead>
            <tbody>
              {HARD_FILTERS.map((filter) => (
                <HardFilterRow
                  key={filter}
                  name={filter}
                  state={"unknown" as FilterState}
                  reason="Filter outcomes depend on a profile. This record view carries no profile, so the state is UNKNOWN here; the ranked shortlist and exclusions screens publish the decided outcomes. Unknown is not pass."
                />
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-2" aria-labelledby="cost-title">
        <div>
          <h2 id="cost-title" className="text-lg font-bold">
            Cost scenarios
          </h2>
          <p className="text-xs text-stone-600">
            Hostel and independent are complete alternatives, returned together and never added
            together. Published components carry their method; modelled or assumed values are
            labelled as such.
          </p>
        </div>
        {isOk(cost) ? <CostPanel cost={cost.data} /> : <ApiFailurePanel result={cost} />}
      </section>

      <section className="space-y-2" aria-labelledby="subscores-title">
        <div>
          <h2 id="subscores-title" className="text-lg font-bold">
            Sub-scores
          </h2>
          <p className="text-xs text-stone-600">
            Each sub-score carries its own interval and claim count. A sub-score with no evidence is
            UNKNOWN and propagates as a wide interval, not a midpoint.
          </p>
        </div>
        {subScores.length === 0 ? (
          <EmptyPanel title="No sub-scores computed">
            <p>
              No score row with sub-scores exists for this institution yet. The weighted breakdown
              appears after a scoring run; a missing breakdown is not a zero score.
            </p>
          </EmptyPanel>
        ) : (
          <div className="panel divide-y divide-stone-200">
            {subScores.map((subScore) => (
              <div key={subScore.name} className="px-3 py-3">
                <ClaimValue
                  label={titleFromPredicate(subScore.name)}
                  value={subScore.point}
                  lo={subScore.lo}
                  hi={subScore.hi}
                  n={subScore.n_claims}
                  tier={subScore.evidence_tier ?? score?.evidence_tier}
                  sourceUrl={null}
                  note={
                    subScore.n_sources === null || subScore.n_sources === undefined
                      ? "Aggregate of claims; no single source URL. See evidence below."
                      : `Aggregate of claims from ${subScore.n_sources} sources; see evidence below.`
                  }
                />
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-2" aria-labelledby="redflags-title">
        <div>
          <h2 id="redflags-title" className="text-lg font-bold">
            Claim-integrity flags
          </h2>
          <p className="text-xs text-stone-600">
            Flags attach to a claim, not to the institution. They lower claim confidence and are
            displayed verbatim when the API publishes them.
          </p>
        </div>
        <EmptyPanel title="No red-flag records published" showSourcesLink={false}>
          <p>
            The API currently exposes no red-flag field on claims. Absence of flags means the check
            has not run, not that the evidence is clean.
          </p>
        </EmptyPanel>
      </section>

      <section className="space-y-2" aria-labelledby="evidence-title">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 id="evidence-title" className="text-lg font-bold">
              Evidence
            </h2>
            <p className="text-xs text-stone-600">
              Quotes are verbatim and spans index into the stored extraction. Modelled claims are
              kept separate from measured ones.
            </p>
          </div>
          {isOk(evidence) && evidence.data.total > evidence.data.items.length ? (
            <span className="num text-[11px] text-amber-950">
              showing {evidence.data.items.length} of {evidence.data.total} claims
            </span>
          ) : null}
        </div>
        {!isOk(evidence) ? <ApiFailurePanel result={evidence} /> : null}
        {claims.length === 0 ? (
          <EmptyPanel title="No claims resolve for this programme">
            <p>
              No extracted, derived or modelled claim exists. Every dependent number must therefore
              render UNKNOWN. Run the crawl and extraction pipeline (
              <code className="num">make crawl</code>, <code className="num">make extract</code>) or
              inspect source coverage.
            </p>
          </EmptyPanel>
        ) : (
          <EvidenceTabs claims={claims} />
        )}
      </section>

      <section className="panel" aria-labelledby="exams-on-page-title">
        <div className="panel-head">
          <h2 id="exams-on-page-title" className="text-sm font-bold">
            Exams linked to this country
          </h2>
          <Link className="link text-[11px]" href="/exams">
            exam calendar
          </Link>
        </div>
        {data.exams.length === 0 ? (
          <p className="px-3 py-3 text-xs text-stone-700">
            No exam rows are attached to this country. Registration windows and travel requirements
            are therefore unverified.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th scope="col">Exam</th>
                <th scope="col">Scope</th>
                <th scope="col">Registration closes</th>
                <th scope="col">Travel</th>
              </tr>
            </thead>
            <tbody>
              {data.exams.map((exam) => (
                <tr key={exam.id}>
                  <th scope="row" className="font-normal">
                    {exam.name}
                  </th>
                  <td>{exam.scope}</td>
                  <td className="num">{exam.registration_closes ?? "not observed"}</td>
                  <td>{exam.requires_travel ? "required" : "not required"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
