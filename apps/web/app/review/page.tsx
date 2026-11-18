import { ReviewConsole } from "./review-console";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Review console — findMyDreamCollege",
};

export default function ReviewPage() {
  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Human adjudication · low-confidence work first</p>
        <h1 className="text-2xl font-bold tracking-tight">Review console</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          A claim is accepted, rejected or corrected against the snapshot span that produced it; an
          alias suggestion is confirmed or refused; an ambiguous document is linked to its entity or
          taken out of the plan. Corrections are appended as new claims — no reviewed claim is
          edited in place — and every decision is written to an append-only audit trail with the
          previous state.
        </p>
        <p className="max-w-3xl text-xs text-stone-600">
          No profile data is shown or stored here: this surface decides evidence, never applicants.
        </p>
      </header>

      <ReviewConsole />

      <section className="panel" aria-labelledby="review-contract-title">
        <div className="panel-head">
          <h2 id="review-contract-title" className="text-sm font-bold">
            How a decision is recorded
          </h2>
          <span className="text-[11px] text-stone-600">brief §18.7</span>
        </div>
        <div className="space-y-2 px-3 py-3 text-sm text-stone-800">
          <ol className="list-decimal space-y-1 pl-5">
            <li>
              Queues are served by <code className="num">/api/review/queue?kind=claim|alias|document</code>,
              lowest confidence first, paginated.
            </li>
            <li>
              <strong>Accept / reject</strong> write only the verification columns
              (<code className="num">verification_status</code>, <code className="num">verified_by=human</code>,
              notes). Rejecting a machine-supported claim requires a note — the console warns.
            </li>
            <li>
              <strong>Correct</strong> inserts a superseding claim with the corrected quote and sets
              <code className="num"> superseded_by</code>; the original row is never mutated.
            </li>
            <li>
              <strong>Alias confirm</strong> makes a <code className="num">manual</code> alias
              authoritative: entity resolution resolves it unattended and stops auto-merging the
              pair. <strong>Reject</strong> hides the suggestion from resolution.
            </li>
            <li>
              <strong>Documents</strong> resolve a planner abstention
              (<code className="num">plan_state=needs_review</code>) by linking the entity or
              marking the document not applicable.
            </li>
            <li>
              Every applied action appends one <code className="num">review_audit</code> row with
              <code className="num"> decided_by=&apos;human&apos;</code>, the timestamp and the previous
              state; the trail renders under each item.
            </li>
          </ol>
        </div>
      </section>
    </div>
  );
}
