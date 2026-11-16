"use client";

import { useState } from "react";

import { formatAmount } from "@/lib/format";
import type { CostCategory, CostItem } from "@/lib/types";

const hostelCategories: readonly CostCategory[] = [
  "tuition",
  "hostel",
  "mess",
  "equipment",
  "exam_fee",
  "insurance",
  "misc",
];

const independentCategories: readonly CostCategory[] = [
  "tuition",
  "rent",
  "food",
  "utilities",
  "transport",
  "equipment",
  "exam_fee",
  "insurance",
  "misc",
];

function ScenarioColumn({
  living,
  items,
  categories,
  selected,
}: {
  living: "hostel" | "independent";
  items: CostItem[];
  categories: readonly CostCategory[];
  selected: boolean;
}) {
  const published = categories
    .map((category) => ({ category, items: items.filter((item) => item.category === category) }))
    .filter((group) => group.items.length > 0);
  const missing = categories.filter((category) =>
    items.every((item) => item.category !== category),
  );

  return (
    <section
      aria-labelledby={`scenario-${living}`}
      className={`rounded border ${selected ? "border-stone-900" : "border-stone-300"} bg-white`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-stone-200 px-3 py-2">
        <h3 id={`scenario-${living}`} className="text-sm font-bold">
          {living === "hostel" ? "Hostel scenario" : "Independent scenario"}
        </h3>
        <span className="text-[10px] font-semibold uppercase tracking-wide text-stone-600">
          {selected ? "selected view" : "parallel view"}
        </span>
      </div>
      {published.length === 0 ? (
        <p className="px-3 py-3 text-xs text-stone-700">
          No published {living} cost items. The cost model (WS-F, brief §13) will substitute a city
          basket marked <code className="num">modelled</code> with a ±20% band. This page does not
          invent those numbers.
        </p>
      ) : (
        <table className="border-0">
          <caption className="sr-only">
            Published {living} scenario cost items. Rows are not summed with the other scenario.
          </caption>
          <thead>
            <tr>
              <th scope="col">Component</th>
              <th scope="col">Amount</th>
              <th scope="col">Basis</th>
            </tr>
          </thead>
          <tbody>
            {published.map((group) =>
              group.items.map((item) => (
                <tr key={item.id}>
                  <th scope="row" className="font-normal">
                    <span className="kicker block">{item.category}</span>
                    {item.label}
                  </th>
                  <td className="num whitespace-nowrap font-semibold">
                    {formatAmount(item.amount, item.currency)}
                    <span className="block text-[10px] font-normal text-stone-600">
                      {item.period.replace("_", " ")}
                    </span>
                  </td>
                  <td className="text-[11px] text-stone-600">
                    {item.basis ?? "basis not published"}
                    {item.confidence !== null && item.confidence !== undefined ? (
                      <span className="num block">confidence {item.confidence.toFixed(2)}</span>
                    ) : null}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      )}
      {missing.length > 0 && published.length > 0 ? (
        <p className="border-t border-stone-200 px-3 py-2 text-[11px] text-amber-950">
          Unpublished in this scenario: {missing.join(", ")}. Missing cost components are not
          treated as zero-cost.
        </p>
      ) : null}
      <p className="border-t border-stone-200 px-3 py-2 text-[11px] text-stone-600">
        Scenario components above are not summed with the other column. TCOA and DSR₂₅ are computed
        by WS-F and will appear as intervals with n; nothing here is a total.
      </p>
    </section>
  );
}

/**
 * Two complete, non-additive living scenarios (brief §13). The toggle changes
 * emphasis only; both columns stay visible so comparison is never hidden behind
 * a click.
 */
export function CostScenarios({ items }: { items: CostItem[] }) {
  const [selected, setSelected] = useState<"hostel" | "independent">("hostel");

  return (
    <div className="space-y-2">
      <fieldset className="flex flex-wrap items-center gap-3">
        <legend className="text-xs font-semibold text-stone-800">
          Living cost scenario (side by side, never summed)
        </legend>
        <label className="flex items-center gap-1.5 text-xs">
          <input
            type="radio"
            name="living"
            value="hostel"
            checked={selected === "hostel"}
            onChange={() => setSelected("hostel")}
          />
          hostel (institution hostel + mess)
        </label>
        <label className="flex items-center gap-1.5 text-xs">
          <input
            type="radio"
            name="living"
            value="independent"
            checked={selected === "independent"}
            onChange={() => setSelected("independent")}
          />
          independent (PG rent + food + utilities + transport)
        </label>
      </fieldset>
      <div className="grid gap-3 lg:grid-cols-2">
        <ScenarioColumn
          living="hostel"
          items={items}
          categories={hostelCategories}
          selected={selected === "hostel"}
        />
        <ScenarioColumn
          living="independent"
          items={items}
          categories={independentCategories}
          selected={selected === "independent"}
        />
      </div>
    </div>
  );
}
