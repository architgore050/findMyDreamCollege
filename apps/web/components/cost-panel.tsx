import { formatAmount } from "@/lib/format";
import type { CostPayload, LivingScenarioCost, MoneyBand } from "@/lib/types";

const SCENARIO_ORDER = ["hostel", "independent"] as const;

function bandText(band: MoneyBand | undefined): string {
  if (!band || band.value === null) return "UNKNOWN";
  const point = formatAmount(band.value, band.currency);
  if (band.lo === null || band.hi === null) return point;
  return `${point} (${formatAmount(band.lo, band.currency)}–${formatAmount(band.hi, band.currency)})`;
}

function MethodTag({ method }: { method: string }) {
  const modelled = method.startsWith("modelled");
  return (
    <span
      className={`inline-flex items-center rounded border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide ${
        modelled
          ? "border-violet-700 bg-violet-50 text-violet-900"
          : "border-stone-400 bg-white text-stone-700"
      }`}
    >
      {method}
    </span>
  );
}

interface AssumedFlag {
  key: string;
  note: string;
}

function assumedFlags(flags: Record<string, unknown> | undefined): AssumedFlag[] {
  if (!flags) return [];
  const entries: AssumedFlag[] = [];
  for (const [key, raw] of Object.entries(flags)) {
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) continue;
    const record = raw as Record<string, unknown>;
    if (record["assumed"] !== true) continue;
    entries.push({
      key,
      note: typeof record["note"] === "string" ? record["note"] : "",
    });
  }
  return entries;
}

function ScenarioColumn({
  scenario,
  selected,
}: {
  scenario: LivingScenarioCost;
  selected: boolean;
}) {
  const components = Object.entries(scenario.components).sort(([a], [b]) => a.localeCompare(b));

  return (
    <section
      aria-labelledby={`scenario-${scenario.name}`}
      className={`rounded border ${selected ? "border-stone-900" : "border-stone-300"} bg-white`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-stone-200 px-3 py-2">
        <h3 id={`scenario-${scenario.name}`} className="text-sm font-bold">
          {scenario.name === "hostel" ? "Hostel scenario" : "Independent scenario"}
        </h3>
        <div className="flex items-center gap-1">
          {scenario.modelled ? (
            <span className="rounded border border-violet-700 bg-violet-50 px-1 py-0.5 text-[9px] font-bold uppercase tracking-wide text-violet-900">
              modelled
            </span>
          ) : null}
          <span className="text-[10px] font-semibold uppercase tracking-wide text-stone-600">
            {selected ? "default view" : "parallel view"}
          </span>
        </div>
      </div>

      {components.length === 0 ? (
        <p className="px-3 py-3 text-xs text-amber-950">
          No living component could be modelled for this scenario. Missing is not zero.
        </p>
      ) : (
        <table className="border-0">
          <caption className="sr-only">
            {scenario.name} living components. These are complete alternatives; rows from the other
            column are never added to these.
          </caption>
          <thead>
            <tr>
              <th scope="col">Component</th>
              <th scope="col">Annual</th>
              <th scope="col">Method</th>
            </tr>
          </thead>
          <tbody>
            {components.map(([name, band]) => (
              <tr key={name}>
                <th scope="row" className="font-normal capitalize">
                  {name.replace(/_/g, " ")}
                </th>
                <td className="num whitespace-nowrap font-semibold">{bandText(band)}</td>
                <td>
                  <MethodTag method={band.method} />
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th scope="row">Annual living</th>
              <td className="num whitespace-nowrap font-semibold">{bandText(scenario.annual)}</td>
              <td>
                <MethodTag method={scenario.annual.method} />
              </td>
            </tr>
            <tr>
              <th scope="row">Total over programme</th>
              <td className="num whitespace-nowrap font-semibold">{bandText(scenario.total)}</td>
              <td>
                <MethodTag method={scenario.total.method} />
              </td>
            </tr>
          </tfoot>
        </table>
      )}
      <p className="border-t border-stone-200 px-3 py-2 text-[11px] text-stone-600">
        {scenario.name} is a complete scenario. It is never summed with the other column.
      </p>
    </section>
  );
}

/**
 * The real cost model payload (brief §13): two complete, non-additive living
 * scenarios, TCOA, financing and DSR₂₅. Values the model assumed rather than
 * extracted are labelled, never blended in silently.
 */
export function CostPanel({ cost }: { cost: CostPayload }) {
  if (!cost.available) {
    return (
      <section className="panel border-dashed px-3 py-3" aria-label="Cost cannot be modelled">
        <h2 className="text-sm font-bold">Cost cannot be modelled from current evidence</h2>
        <p className="mt-1 text-xs text-stone-700">
          {cost.reason ?? "The API declined to build a total cost of attendance."} No number has
          been invented to fill the gap.
        </p>
      </section>
    );
  }

  const assumed = assumedFlags(cost.assumed);
  const scenarios = SCENARIO_ORDER.map((name) => cost.scenarios[name]).filter(
    (scenario): scenario is LivingScenarioCost => scenario !== undefined,
  );

  return (
    <div className="space-y-3">
      {cost.non_additive_scenarios ? (
        <p className="border border-stone-400 bg-white px-3 py-1.5 text-[11px] font-semibold text-stone-800">
          {cost.note ?? "Hostel and independent are complete alternatives; never sum them."}
        </p>
      ) : null}

      {assumed.length > 0 ? (
        <section
          className="border border-amber-700 bg-amber-50 px-3 py-2"
          aria-label="Assumed values"
        >
          <h3 className="text-[11px] font-bold uppercase tracking-wide text-amber-950">
            Assumed values (not extracted evidence)
          </h3>
          <ul className="mt-1 space-y-0.5">
            {assumed.map((flag) => (
              <li key={flag.key} className="text-[11px] text-amber-950">
                <code className="num">{flag.key}</code>
                {flag.note ? ` — ${flag.note}` : ""}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <div className="grid gap-3 lg:grid-cols-2">
        {scenarios.map((scenario) => (
          <ScenarioColumn
            key={scenario.name}
            scenario={scenario}
            selected={scenario.name === cost.living}
          />
        ))}
      </div>

      <section className="panel" aria-labelledby="tcoa-title">
        <div className="panel-head">
          <h2 id="tcoa-title" className="text-sm font-bold">
            Totals for the {cost.living} scenario
          </h2>
          <span className="text-[11px] text-stone-600">
            {cost.duration_months ? `${cost.duration_months} months` : "duration unknown"}
          </span>
        </div>
        <table>
          <caption className="sr-only">
            Cost of attendance, financing and affordability for the selected living scenario
          </caption>
          <thead>
            <tr>
              <th scope="col">Measure</th>
              <th scope="col">Band</th>
              <th scope="col">Method</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["TCOA (whole programme)", cost.tcoa],
              ["TCOA (first year)", cost.tcoa_first_year],
              ["Net price after aid", cost.net_price],
              ["Loan principal", cost.loan_principal],
              ["Family out of pocket", cost.family_out_of_pocket],
              ["Student-work offset", cost.student_work_offset],
              ["Debt at repayment", cost.debt_at_repayment],
              ["EMI", cost.emi],
            ].map(([label, band]) => {
              const money = band as MoneyBand | undefined;
              return (
                <tr key={String(label)}>
                  <th scope="row" className="font-normal">
                    {String(label)}
                  </th>
                  <td className="num font-semibold">{bandText(money)}</td>
                  <td>{money ? <MethodTag method={money.method} /> : "unknown"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="border-t border-stone-200 px-3 py-2">
          <p className="text-[11px] font-semibold text-stone-800">
            DSR₂₅ —{" "}
            {cost.dsr25 && cost.dsr25.value !== null ? (
              <span className="num">
                {cost.dsr25.value.toFixed(3)} ({cost.dsr25.band})
              </span>
            ) : (
              <span className="text-amber-950">
                UNKNOWN — salary or loan-rate evidence is missing
              </span>
            )}
          </p>
          <p className="text-[10px] text-stone-600">
            {cost.dsr25?.formula ?? "first_year_emi / p25_first_year_gross_salary"}
          </p>
          {cost.affordability_gap ? (
            <p className="mt-1 text-[11px] text-stone-800">
              Affordability gap:{" "}
              {cost.affordability_gap.value === null ? (
                <span className="text-amber-950">
                  UNKNOWN — {cost.affordability_gap.formula || "family income not supplied"}
                </span>
              ) : (
                <span className="num">{cost.affordability_gap.value.toFixed(2)}</span>
              )}
              {cost.affordability_gap.flagged ? " · flagged" : ""}
            </p>
          ) : null}
        </div>
        {cost.assumptions && cost.assumptions.length > 0 ? (
          <details className="border-t border-stone-200 px-3 py-2">
            <summary className="cursor-pointer text-[11px] font-semibold text-stone-800">
              Model assumptions ({cost.assumptions.length})
            </summary>
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-[11px] text-stone-700">
              {cost.assumptions.map((assumption) => (
                <li key={assumption}>{assumption}</li>
              ))}
            </ul>
          </details>
        ) : null}
      </section>
    </div>
  );
}
