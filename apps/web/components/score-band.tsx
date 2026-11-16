import { EvidenceTierBadge } from "@/components/badges";
import type { EvidenceTier } from "@/lib/types";

export interface BandProps {
  lo: number | null | undefined;
  hi: number | null | undefined;
  tier?: EvidenceTier | null;
  label?: string;
  /** Version stamps from the response envelope when available. */
  scorerVersion?: string | null;
  weightsVersion?: string | null;
}

/**
 * Composite scores are intervals. There is deliberately no point rendering:
 * a fake decimal hides how thin the evidence is.
 */
export function ScoreBand({
  lo,
  hi,
  tier,
  label = "composite band",
  scorerVersion,
  weightsVersion,
}: BandProps) {
  const hasBand =
    typeof lo === "number" && typeof hi === "number" && Number.isFinite(lo) && Number.isFinite(hi);
  const left = hasBand ? Math.max(0, Math.min(100, lo)) : 0;
  const right = hasBand ? Math.max(0, Math.min(100, hi)) : 0;
  const width = hasBand ? Math.max(1, right - left) : 0;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="kicker">{label}</span>
        <EvidenceTierBadge tier={tier} />
      </div>
      {hasBand ? (
        <>
          <div className="flex items-center gap-2">
            <div
              className="relative h-3 w-full overflow-hidden rounded-sm border border-stone-400 bg-stone-100"
              role="img"
              aria-label={`Composite score band from ${lo} to ${hi} on a 0 to 100 scale; no point estimate is published.`}
            >
              <div
                className="absolute inset-y-0 bg-stone-700"
                style={{ left: `${left}%`, width: `${width}%` }}
              />
            </div>
            <span className="num shrink-0 text-sm font-semibold">
              {lo.toFixed(0)}–{hi.toFixed(0)}
            </span>
          </div>
          <p className="text-[11px] text-stone-600">
            Interval, not a rank. Width {Math.round(right - left)} points on a 0–100 scale. Compare
            only when bands do not overlap, and only with Pareto dominance (brief §11).
            {scorerVersion ? ` scorer ${scorerVersion}` : ""}
            {weightsVersion ? ` · weights ${weightsVersion}` : ""}
          </p>
        </>
      ) : (
        <p className="text-xs text-amber-950">
          <span className="inline-flex items-center rounded border border-amber-700 bg-amber-50 px-1.5 py-0.5 font-bold tracking-wide">
            NOT SCORED
          </span>{" "}
          No composite interval exists for this programme yet. Not zero, not a placeholder rank.
        </p>
      )}
    </div>
  );
}
