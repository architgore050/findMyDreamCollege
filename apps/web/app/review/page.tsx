import Link from "next/link";

import { ApiFailurePanel, EmptyPanel } from "@/components/api-state";
import { apiReviewQueue, isOk } from "@/lib/api";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Review console — findMyDreamCollege",
};

export default async function ReviewPage() {
  const queue = await apiReviewQueue();

  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="kicker">Human adjudication · low-confidence claims</p>
        <h1 className="text-2xl font-bold tracking-tight">Review console</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          A claim is accepted, rejected or corrected against the snapshot span that produced it.
          Corrections are appended as new claims; no reviewed claim is edited in place.
        </p>
      </header>

      {isOk(queue) ? (
        <p className="panel px-3 py-3 text-sm text-stone-700">
          The review queue responded, but this build does not interpret its payload until the
          response contract is frozen. No claims are rendered and no labels can be submitted.
        </p>
      ) : (
        <ApiFailurePanel result={queue} />
      )}

      <section className="panel" aria-labelledby="review-workflow-title">
        <div className="panel-head">
          <h2 id="review-workflow-title" className="text-sm font-bold">
            Review contract (not yet interactive)
          </h2>
          <span className="text-[11px] text-stone-600">brief §18.7</span>
        </div>
        <div className="space-y-2 px-3 py-3 text-sm text-stone-800">
          <ol className="list-decimal space-y-1 pl-5">
            <li>Queue low-confidence claims, grouped by source family and predicate.</li>
            <li>
              Show the snapshot with the claim&apos;s <code className="num">char_start</code>–
              <code className="num">char_end</code> span highlighted, plus URL, fetched_at and
              sha256.
            </li>
            <li>Accept, reject or correct. Every action is recorded as a human label.</li>
            <li>
              A correction inserts a claim and sets <code className="num">superseded_by</code>; the
              old claim remains readable.
            </li>
          </ol>
          <p className="text-xs text-amber-950">
            <strong>Actions are disabled:</strong> submitting a label needs the admin-token API,
            which is 501 in this build. This browser will not hold a shared admin token — no
            client-side secrets.
          </p>
          <p className="text-xs">
            While waiting, use{" "}
            <Link className="link" href="/sources">
              the source registry
            </Link>{" "}
            to see which sources drive the queue.
          </p>
        </div>
      </section>

      <EmptyPanel title="No review queue is being served" showSourcesLink={false}>
        <p>
          The queue endpoint, <code className="num">/api/review/queue</code>, is owned by WS-G but
          sits behind the API layer and the admin token. Nothing is stubbed with fake claims; the
          console will list real low-confidence rows when the endpoint lands.
        </p>
      </EmptyPanel>
    </div>
  );
}
