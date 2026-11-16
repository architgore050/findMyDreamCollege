"use client";

import { useState } from "react";

import { ClaimValue } from "@/components/claim-value";
import { readClaimPoint, titleFromPredicate } from "@/lib/format";
import type { Claim } from "@/lib/types";

function qualifierNote(qualifier: Record<string, unknown> | null | undefined): string | null {
  if (!qualifier) return null;
  const parts = Object.entries(qualifier)
    .filter(
      ([, value]) =>
        typeof value === "string" || typeof value === "number" || typeof value === "boolean",
    )
    .slice(0, 4)
    .map(([key, value]) => `${key}=${String(value)}`);
  return parts.length > 0 ? `qualifier: ${parts.join(" · ")}` : null;
}

export function ClaimRow({
  claim,
  showPredicate = true,
}: {
  claim: Claim;
  showPredicate?: boolean;
}) {
  const point = readClaimPoint(claim);
  return (
    <div className="border-b border-stone-200 px-3 py-3 last:border-b-0">
      <ClaimValue
        label={showPredicate ? titleFromPredicate(claim.predicate) : undefined}
        value={point.display ?? point.point}
        unit={point.unit}
        lo={point.lo}
        hi={point.hi}
        n={point.n}
        method={claim.method}
        sourceUrl={claim.url}
        observedAt={claim.observed_at}
        confidence={claim.confidence}
        note={qualifierNote(claim.qualifier)}
      />
      <blockquote className="mt-2 border-l-2 border-stone-400 pl-2 text-xs italic text-stone-700">
        “{claim.quote}”
      </blockquote>
      <p className="num mt-1 text-[10px] text-stone-500">
        claim #{claim.id} · span {claim.char_start}–{claim.char_end}
        {claim.fetched_at ? ` · fetched ${claim.fetched_at.slice(0, 10)}` : ""}
        {claim.sha256 ? ` · sha256 ${claim.sha256.slice(0, 12)}…` : ""}
      </p>
    </div>
  );
}

export function ClaimList({
  claims,
  showPredicate = true,
}: {
  claims: Claim[];
  showPredicate?: boolean;
}) {
  if (claims.length === 0) return null;
  return (
    <div className="panel">
      {claims.map((claim) => (
        <ClaimRow key={claim.id} claim={claim} showPredicate={showPredicate} />
      ))}
    </div>
  );
}

/**
 * Tabs exist because "measured" and "modelled" evidence must never be read as
 * one body of facts (brief §3.3). Keyboard: arrows, Home, End.
 */
export function EvidenceTabs({ claims }: { claims: Claim[] }) {
  const measured = claims.filter(
    (claim) => claim.method === "extracted" || claim.method === "derived",
  );
  const modelled = claims.filter((claim) => claim.method === "modelled");
  const tabs = [
    { id: "measured", label: `Measured (${measured.length})`, claims: measured },
    { id: "modelled", label: `Modelled (${modelled.length})`, claims: modelled },
  ] as const;
  const [active, setActive] = useState<"measured" | "modelled">("measured");

  function move(direction: number) {
    const index = tabs.findIndex((tab) => tab.id === active);
    const next = tabs[(index + direction + tabs.length) % tabs.length];
    if (next) setActive(next.id);
  }

  return (
    <div>
      <div
        role="tablist"
        aria-label="Evidence method"
        className="flex gap-1 border-b border-stone-300"
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            role="tab"
            type="button"
            aria-selected={active === tab.id}
            aria-controls={`panel-${tab.id}`}
            tabIndex={active === tab.id ? 0 : -1}
            className={`-mb-px border-b-2 px-3 py-1.5 text-xs font-semibold ${
              active === tab.id
                ? "border-stone-900 text-stone-900"
                : "border-transparent text-stone-600 hover:text-stone-900"
            }`}
            onClick={() => setActive(tab.id)}
            onKeyDown={(event) => {
              if (event.key === "ArrowRight") {
                event.preventDefault();
                move(1);
              } else if (event.key === "ArrowLeft") {
                event.preventDefault();
                move(-1);
              } else if (event.key === "Home") {
                event.preventDefault();
                setActive("measured");
              } else if (event.key === "End") {
                event.preventDefault();
                setActive("modelled");
              }
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          id={`panel-${tab.id}`}
          role="tabpanel"
          aria-labelledby={`tab-${tab.id}`}
          hidden={active !== tab.id}
          tabIndex={0}
        >
          {tab.claims.length === 0 ? (
            <p className="px-3 py-4 text-xs text-stone-600">
              {tab.id === "measured"
                ? "No extracted or derived claims exist for this programme yet. That is an evidence gap, not a zero."
                : "No modelled claims were produced. Nothing has been imputed in its place."}
            </p>
          ) : (
            <ClaimList claims={tab.claims} />
          )}
        </div>
      ))}
    </div>
  );
}
