import { formatNumber, titleFromPredicate } from "@/lib/format";
import type { ComparePayload, DominanceVerdict } from "@/lib/types";

const VERDICT_STYLE: Record<DominanceVerdict, string> = {
  better: "border-emerald-800 bg-emerald-50 text-emerald-950",
  worse: "border-stone-500 bg-stone-100 text-stone-800",
  not_comparable: "border-red-800 bg-red-100 text-red-950",
};

const VERDICT_LABEL: Record<DominanceVerdict, string> = {
  better: "better",
  worse: "worse",
  not_comparable: "NOT COMPARABLE",
};

function pairLabel(pair: string, names: Record<number, string>): string {
  return pair
    .split("|")
    .map((raw) => {
      const id = Number(raw);
      return names[id] ?? `#${raw}`;
    })
    .join(" vs ");
}

function relation(payload: ComparePayload, left: string, right: string): DominanceVerdict | "self" {
  const cell = payload.matrix[left];
  if (left === right) return "self";
  if (!cell) return "not_comparable";
  if (cell.better_than.includes(right)) return "better";
  if (cell.worse_than.includes(right)) return "worse";
  return "not_comparable";
}

/**
 * Pareto matrix. `not_comparable` is not a tidy "tie": it means at least one
 * dimension is unknown on one side or the two rows trade wins. It is rendered
 * as a loud, red state rather than smoothed into an ordering.
 */
export function CompareMatrix({
  payload,
  names,
}: {
  payload: ComparePayload;
  names: Record<number, string>;
}) {
  const ids = payload.ids.map(String);

  return (
    <div className="space-y-3">
      {payload.not_comparable_pairs.length > 0 ? (
        <section
          role="alert"
          className="border-2 border-red-800 bg-red-50 px-3 py-2"
          aria-labelledby="not-comparable-title"
        >
          <h3 id="not-comparable-title" className="text-sm font-bold text-red-950">
            NOT COMPARABLE — {payload.not_comparable_pairs.length} pair
            {payload.not_comparable_pairs.length === 1 ? "" : "s"} cannot be ordered
          </h3>
          <p className="mt-0.5 text-[11px] text-red-950">
            A pair is not comparable when a sub-score is UNKNOWN on either side, or when each
            programme wins a different dimension. No winner is implied and no band overlap decides
            it.
          </p>
          <ul className="num mt-1 space-y-0.5 text-[11px] font-semibold text-red-950">
            {payload.not_comparable_pairs.map((pair) => (
              <li key={pair}>{pairLabel(pair, names)}</li>
            ))}
          </ul>
        </section>
      ) : (
        <p className="border border-stone-400 bg-white px-3 py-2 text-xs text-stone-700">
          Every pair resolved to a dominance relation on the known sub-scores. That is still not a
          total order: check which dimensions differ before acting on it.
        </p>
      )}

      <div className="panel overflow-x-auto">
        <table>
          <caption className="sr-only">
            Pairwise Pareto dominance. A row is better than a column only when it is at least as
            good on every known sub-score and strictly better on one.
          </caption>
          <thead>
            <tr>
              <th scope="col">Programme</th>
              <th scope="col">Undominated</th>
              {ids.map((id) => (
                <th key={id} scope="col" className="num">
                  {names[Number(id)] ? `${names[Number(id)]} (#${id})` : `#${id}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ids.map((left) => {
              const cell = payload.matrix[left];
              return (
                <tr key={left}>
                  <th scope="row" className="font-normal">
                    {names[Number(left)] ?? `#${left}`}
                    <span className="num block text-[10px] text-stone-500">#{left}</span>
                  </th>
                  <td>
                    {cell?.undominated ? (
                      <span className="rounded border border-emerald-800 bg-emerald-50 px-1 py-0.5 text-[10px] font-bold uppercase text-emerald-950">
                        undominated
                      </span>
                    ) : (
                      <span className="text-[10px] uppercase text-stone-600">dominated</span>
                    )}
                  </td>
                  {ids.map((right) => {
                    const verdict = relation(payload, left, right);
                    return (
                      <td key={right}>
                        {verdict === "self" ? (
                          <span className="text-[10px] text-stone-400">—</span>
                        ) : (
                          <span
                            className={`inline-flex items-center rounded border px-1 py-0.5 text-[10px] font-bold uppercase tracking-wide ${VERDICT_STYLE[verdict]}`}
                          >
                            {VERDICT_LABEL[verdict]}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <section className="panel" aria-labelledby="pair-detail-title">
        <div className="panel-head">
          <h2 id="pair-detail-title" className="text-sm font-bold">
            Pair detail
          </h2>
          <span className="text-[11px] text-stone-600">
            {formatNumber(Object.keys(payload.comparisons).length)} pairs
          </span>
        </div>
        <ul className="divide-y divide-stone-200">
          {Object.entries(payload.comparisons).map(([pair, comparison]) => (
            <li
              key={pair}
              className={`px-3 py-2 ${comparison.verdict === "not_comparable" ? "bg-red-50" : ""}`}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <p className="text-xs font-semibold text-stone-900">{pairLabel(pair, names)}</p>
                <span
                  className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ${VERDICT_STYLE[comparison.verdict]}`}
                >
                  {VERDICT_LABEL[comparison.verdict]}
                </span>
              </div>
              {comparison.differing_sub_scores.length === 0 ? (
                <p className="mt-0.5 text-[11px] text-stone-700">
                  No sub-score differs on known evidence — that is not a tie in quality.
                </p>
              ) : (
                <div className="mt-1 space-y-0.5">
                  {comparison.differing_sub_scores.map((dimension) => {
                    const leftClaims = comparison.differing_claims.left[dimension] ?? [];
                    const rightClaims = comparison.differing_claims.right[dimension] ?? [];
                    return (
                      <p key={dimension} className="num text-[11px] text-stone-700">
                        <span className="font-semibold">{titleFromPredicate(dimension)}</span>
                        {" — claims "}
                        {leftClaims.length > 0
                          ? leftClaims.map((id) => `#${String(id)}`).join(", ")
                          : "none"}
                        {" vs "}
                        {rightClaims.length > 0
                          ? rightClaims.map((id) => `#${String(id)}`).join(", ")
                          : "none"}
                      </p>
                    );
                  })}
                </div>
              )}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
