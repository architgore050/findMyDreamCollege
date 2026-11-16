import Link from "next/link";

import { EvidenceTierBadge } from "@/components/badges";
import { HardFilterBadge } from "@/components/hard-filter";
import { ScoreBand } from "@/components/score-band";
import { formatNumber, titleFromPredicate } from "@/lib/format";
import type { ApiMeta, Insight, RankingsPayload } from "@/lib/types";

function InsightItem({ insight, mandatory }: { insight: Insight; mandatory: boolean }) {
  return (
    <li className="pl-3">
      <span className="kicker block">
        {mandatory ? "must show · " : ""}
        {insight.kind}
      </span>
      <p className="text-xs text-stone-800">{insight.text}</p>
      {insight.claim_ids.length > 0 ? (
        <p className="num mt-0.5 text-[10px] text-stone-500">
          claims {insight.claim_ids.map((id) => `#${String(id)}`).join(", ")}
        </p>
      ) : (
        <p className="text-[10px] text-amber-950">no deciding claim published</p>
      )}
    </li>
  );
}

/**
 * Ranked shortlist rendering.
 *
 * The ranked order is the API's sort (excluded last, then band upper bound). It
 * is not a "best university" list: each row publishes a band, an evidence tier
 * and the hard-filter decisions behind it. A composite point is never printed.
 */
export function RankingsView({
  payload,
  meta,
}: {
  payload: RankingsPayload;
  meta: ApiMeta | null;
}) {
  if (payload.items.length === 0) {
    return (
      <section className="panel border-dashed px-3 py-3" aria-label="No ranked programmes">
        <h2 className="text-sm font-bold">No programmes to rank</h2>
        <p className="mt-1 text-xs text-stone-700">
          The ranking ran and matched zero programme records. That is an inventory gap, not a score
          of zero.
        </p>
      </section>
    );
  }

  const excludedCount = payload.items.filter((row) => row.excluded).length;

  return (
    <section className="space-y-2" aria-labelledby="rankings-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 id="rankings-title" className="text-lg font-bold">
            Ranked shortlist
          </h2>
          <p className="text-xs text-stone-600">
            Sorted by composite band upper bound, excluded programmes last. A band is not a rank and
            a wider band is not a weaker programme: it means less evidence.
          </p>
        </div>
        <p className="num text-[11px] text-stone-600" data-testid="rankings-count">
          {formatNumber(payload.items.length)} of {formatNumber(payload.total)} ranked ·{" "}
          {formatNumber(excludedCount)} excluded by a hard filter
        </p>
      </div>

      {payload.profile_persisted ? (
        <p className="border border-red-700 bg-red-50 px-3 py-2 text-xs font-semibold text-red-950">
          The API reports <code className="num">profile_persisted: true</code>. This build expects
          false — profile data must never be stored server-side.
        </p>
      ) : null}

      <ol className="space-y-2">
        {payload.items.map((row, index) => {
          const programme = row.programme;
          const band = row.composite_band;
          const failed = row.filters.excluded;
          const insights = row.insights;

          return (
            <li key={programme.id}>
              <article
                className={`panel ${row.excluded ? "border-red-700" : ""}`}
                aria-labelledby={`ranked-${programme.id}`}
              >
                <div className="panel-head">
                  <h3 id={`ranked-${programme.id}`} className="text-sm font-bold">
                    <span className="num mr-2 text-stone-500">{index + 1}.</span>
                    <Link className="link" href={`/programmes/${programme.id}`}>
                      {programme.name}
                    </Link>
                  </h3>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {row.excluded ? (
                      <HardFilterBadge state="fail" />
                    ) : (
                      <EvidenceTierBadge tier={row.evidence_tier} />
                    )}
                    <span className="num text-[10px] text-stone-500">#{programme.id}</span>
                  </div>
                </div>

                <div className="space-y-3 px-3 py-3">
                  <ScoreBand
                    lo={band.lo}
                    hi={band.hi}
                    tier={row.evidence_tier}
                    label={`composite band (${band.method})`}
                    scorerVersion={row.score.scorer_version ?? meta?.scorer_version}
                    weightsVersion={row.score.weights_version ?? meta?.weights_version}
                  />

                  <dl className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-stone-700">
                    <div className="flex gap-1">
                      <dt className="font-semibold">credential</dt>
                      <dd>{programme.credential_level}</dd>
                    </div>
                    <div className="flex gap-1">
                      <dt className="font-semibold">recognition</dt>
                      <dd>{programme.recognition_status}</dd>
                    </div>
                    <div className="flex gap-1">
                      <dt className="font-semibold">hard filters</dt>
                      <dd className="num" data-testid={`filter-counts-${programme.id}`}>
                        {row.filters.counts.pass} pass · {row.filters.counts.fail} excluded ·{" "}
                        {row.filters.counts.unknown} unknown
                      </dd>
                    </div>
                  </dl>

                  {failed.length > 0 ? (
                    <div className="border border-red-700 bg-red-50 px-2 py-1.5">
                      <p className="text-[11px] font-bold uppercase tracking-wide text-red-950">
                        Excluded by hard filter
                      </p>
                      <ul className="mt-1 space-y-1">
                        {failed.map((filterId) => {
                          const decision = row.filters.decisions[filterId];
                          return (
                            <li key={filterId} className="text-[11px] text-red-950">
                              <code className="num">{filterId}</code>
                              {" — "}
                              {decision?.reason ?? "reason not published"}
                              {decision?.claim_id !== null && decision?.claim_id !== undefined ? (
                                <span className="num"> (claim #{String(decision.claim_id)})</span>
                              ) : (
                                <span> (no deciding claim)</span>
                              )}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  ) : null}

                  {insights !== null && insights !== undefined ? (
                    <div>
                      <div className="flex flex-wrap items-baseline justify-between gap-2">
                        <p className="kicker">insights</p>
                        <p className="num text-[10px] text-stone-500">
                          coverage {(insights.coverage * 100).toFixed(0)}% ·{" "}
                          {insights.stopped_reason.replace(/_/g, " ")}
                        </p>
                      </div>
                      {insights.selected.length + insights.mandatory.length === 0 ? (
                        <p className="text-xs text-amber-950">
                          No insight was selectable: none of the expected aspects has a resolving
                          claim.
                        </p>
                      ) : (
                        <ul className="mt-1 space-y-1.5">
                          {insights.mandatory.map((insight) => (
                            <InsightItem key={`m-${insight.id}`} insight={insight} mandatory />
                          ))}
                          {insights.selected.map((insight) => (
                            <InsightItem
                              key={`s-${insight.id}`}
                              insight={insight}
                              mandatory={false}
                            />
                          ))}
                        </ul>
                      )}
                    </div>
                  ) : (
                    <p className="text-[11px] text-amber-950">
                      No insight set was returned for this programme.
                    </p>
                  )}

                  {(row.score.composite.unknown_sub_scores ?? []).length > 0 ? (
                    <p className="text-[11px] text-amber-950">
                      UNKNOWN sub-scores:{" "}
                      {(row.score.composite.unknown_sub_scores ?? [])
                        .map((name) => titleFromPredicate(name))
                        .join(", ")}
                      . They widen the band; they are not scored as zero.
                    </p>
                  ) : null}
                </div>
              </article>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
