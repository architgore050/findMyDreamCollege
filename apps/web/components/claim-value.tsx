import { EvidenceTierBadge, MethodBadge, SourceLink } from "@/components/badges";
import { formatNumber, withUnit } from "@/lib/format";
import type { ClaimMethod, EvidenceTier } from "@/lib/types";

export interface ClaimValueProps {
  /** Human label, e.g. "P25 first-year salary". */
  label?: string;
  value: number | string | null | undefined;
  unit?: string | null;
  /** Interval bounds. Never derived from the point estimate. */
  lo?: number | null;
  hi?: number | null;
  /** Denominator / sample size behind the number. */
  n?: number | null;
  tier?: EvidenceTier | null;
  method?: ClaimMethod | null;
  currency?: string | null;
  sourceUrl?: string | null;
  observedAt?: string | null;
  confidence?: number | null;
  /** Extra qualifier text shown verbatim (e.g. "denominator: 214 placement-eligible"). */
  note?: string | null;
  className?: string;
}

function isMissing(value: ClaimValueProps["value"]): boolean {
  return (
    value === null ||
    value === undefined ||
    (typeof value === "string" && (value.trim() === "" || value.trim().toLowerCase() === "unknown"))
  );
}

/**
 * The only component allowed to render a published number.
 *
 * Shows point + interval + n + evidence tier + source. Missing evidence is
 * rendered as UNKNOWN with no point estimate (brief §3.3, §3.5), and a numeric
 * value without a source is flagged rather than shown as settled.
 */
export function ClaimValue({
  label,
  value,
  unit,
  lo,
  hi,
  n,
  tier,
  method,
  currency,
  sourceUrl,
  observedAt,
  confidence,
  note,
  className = "",
}: ClaimValueProps) {
  const missing = isMissing(value);
  const interval =
    lo !== null && lo !== undefined && hi !== null && hi !== undefined
      ? `[${formatNumber(lo)}, ${formatNumber(hi)}]`
      : null;

  const renderedValue = (() => {
    if (missing) return null;
    if (typeof value === "number") {
      return currency
        ? new Intl.NumberFormat("en-IN", {
            style: "currency",
            currency,
            maximumFractionDigits: 0,
          }).format(value)
        : withUnit(formatNumber(value), unit);
    }
    return String(value);
  })();

  return (
    <div
      className={`flex flex-col gap-1 ${className}`}
      data-testid="claim-value"
      data-state={missing ? "unknown" : "known"}
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        {label ? <span className="kicker">{label}</span> : null}
        {missing ? (
          <span className="inline-flex items-center rounded border border-amber-700 bg-amber-50 px-1.5 py-0.5 text-xs font-bold tracking-wide text-amber-950">
            UNKNOWN
          </span>
        ) : (
          <span className="num text-base font-semibold text-stone-900">{renderedValue}</span>
        )}
        <span className="num text-xs text-stone-700">
          {interval ? (
            <>
              {missing ? "interval " : ""}
              {interval}
            </>
          ) : (
            <span title="No interval was published for this claim.">interval not reported</span>
          )}
        </span>
        <span className="num text-xs text-stone-600">
          n={n === null || n === undefined ? "—" : formatNumber(n)}
          {n === null || n === undefined ? (
            <span className="sr-only"> (sample size not reported)</span>
          ) : null}
        </span>
        <EvidenceTierBadge tier={tier} />
        {method ? <MethodBadge method={method} /> : null}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-stone-600">
        <SourceLink url={sourceUrl} />
        {observedAt ? <span className="num">observed {observedAt.slice(0, 10)}</span> : null}
        {typeof confidence === "number" ? (
          <span className="num">claim confidence {confidence.toFixed(2)}</span>
        ) : null}
      </div>
      {missing ? (
        <p className="text-[11px] text-amber-950">
          No observed value. This is not zero and not a midpoint; treat the underlying quantity as
          unknown until evidence resolves.
        </p>
      ) : null}
      {note ? <p className="text-[11px] text-stone-600">{note}</p> : null}
    </div>
  );
}
