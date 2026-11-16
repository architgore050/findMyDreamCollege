import type { ReactNode } from "react";

export type FilterState = "pass" | "fail" | "unknown";

const styles: Record<FilterState, string> = {
  pass: "border-emerald-800 bg-emerald-50 text-emerald-950",
  fail: "border-red-800 bg-red-50 text-red-950",
  unknown: "border-amber-700 bg-amber-50 text-amber-950",
};

const glyphs: Record<FilterState, string> = {
  pass: "✓",
  fail: "✕",
  unknown: "?",
};

const labels: Record<FilterState, string> = {
  pass: "PASS",
  fail: "EXCLUDED",
  unknown: "UNKNOWN",
};

/** Hard filters are binary cutoffs, never penalties (brief §3.4, §12). */
export function HardFilterBadge({ state }: { state: FilterState }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ${styles[state]}`}
    >
      <span aria-hidden="true">{glyphs[state]}</span>
      {labels[state]}
    </span>
  );
}

export function HardFilterRow({
  name,
  state,
  reason,
  claimLink,
}: {
  name: string;
  state: FilterState;
  reason: string;
  claimLink?: ReactNode;
}) {
  return (
    <tr>
      <th scope="row" className="font-mono text-[11px] font-normal text-stone-700">
        {name}
      </th>
      <td>
        <HardFilterBadge state={state} />
      </td>
      <td className="text-stone-800">
        {reason}
        {claimLink ? <div className="mt-1 text-[11px]">{claimLink}</div> : null}
      </td>
    </tr>
  );
}
