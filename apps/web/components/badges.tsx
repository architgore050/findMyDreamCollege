import type { ClaimMethod, EvidenceTier } from "@/lib/types";
import { hostOf } from "@/lib/format";

const tierLabels: Record<EvidenceTier, string> = {
  well_evidenced: "well evidenced",
  partial: "partial evidence",
  thin: "thin evidence",
  insufficient: "insufficient evidence",
};

const tierClasses: Record<EvidenceTier, string> = {
  well_evidenced: "border-emerald-700 bg-emerald-50 text-emerald-900",
  partial: "border-sky-700 bg-sky-50 text-sky-900",
  thin: "border-amber-700 bg-amber-50 text-amber-900",
  insufficient: "border-stone-500 bg-stone-100 text-stone-800",
};

export function EvidenceTierBadge({ tier }: { tier: EvidenceTier | null | undefined }) {
  if (!tier) {
    return (
      <span className="inline-flex items-center rounded border border-stone-400 bg-white px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-stone-700">
        evidence tier: unknown
      </span>
    );
  }
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${tierClasses[tier]}`}
    >
      {tierLabels[tier]}
    </span>
  );
}

const methodLabels: Record<ClaimMethod, string> = {
  extracted: "extracted",
  derived: "derived",
  modelled: "modelled",
  user_provided: "user provided",
};

export function MethodBadge({ method }: { method: ClaimMethod | null | undefined }) {
  if (!method) return null;
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
        method === "modelled"
          ? "border-violet-700 bg-violet-50 text-violet-900"
          : method === "user_provided"
            ? "border-stone-500 bg-stone-100 text-stone-800"
            : "border-stone-400 bg-white text-stone-700"
      }`}
    >
      {methodLabels[method]}
    </span>
  );
}

export function SourceLink({ url }: { url: string | null | undefined }) {
  if (!url) {
    return (
      <span className="inline-flex items-center gap-1 rounded border border-amber-700 bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-900">
        <span aria-hidden="true">⚠</span> no source
      </span>
    );
  }
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer noopener"
      className="link inline-flex items-center gap-1 text-[11px]"
      title={url}
    >
      source: {hostOf(url)}
      <span aria-hidden="true">↗</span>
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  );
}
