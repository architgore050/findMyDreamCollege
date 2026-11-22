import { MethodologyClient } from "@/components/methodology-client";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Methodology — findMyDreamCollege",
};

export default function MethodologyPage() {
  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Trust surface · generated from the registry</p>
        <h1 className="text-2xl font-bold tracking-tight">Methodology</h1>
        <p className="max-w-2xl text-sm text-stone-700">
          How this site decides what to show. The weights, thresholds and source
          counts below are generated from the scoring configuration at request
          time — this page never hard-copies them. Every number on every page
          links back to the claim and the source document that produced it.
        </p>
      </header>
      <MethodologyClient />
    </div>
  );
}
