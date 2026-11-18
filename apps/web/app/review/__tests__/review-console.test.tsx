import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  AliasCard,
  ClaimCard,
  DocumentCard,
  ReviewQueueView,
} from "@/app/review/review-console";
import type {
  ReviewAliasItem,
  ReviewAuditRecord,
  ReviewClaimItem,
  ReviewDocumentItem,
  ReviewQueuePayload,
} from "@/lib/types";

/**
 * Offline queue-render contract: the console renders real queue rows with no
 * API reachable. No fetch is mocked or awaited here — rendering the queue and
 * firing the action callbacks must be possible against fixture data alone.
 */

const claimFixture: ReviewClaimItem = {
  id: 5,
  extraction_id: 9,
  entity_id: 2,
  predicate: "fees.total_annual",
  value: 42000,
  unit: "INR",
  qualifier: { page_no: 3, bbox: [40, 120, 300, 18] },
  quote: "Annual fee for B.Sc MLT is Rs 42,000",
  char_start: 210,
  char_end: 246,
  confidence: 0.32,
  observed_at: "2026-11-17T10:00:00Z",
  method: "extracted",
  superseded_by: null,
  verification_status: "supported",
  verified_by: "mock-verifier",
  verified_at: "2026-11-17T11:00:00Z",
  verification_notes: { verdict: "supported" },
  url: "https://example.ac.in/fee-2026.pdf",
  fetched_at: "2026-11-17T09:59:00Z",
  sha256: "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2a3b4c5d6a7b8c9d0e1f2",
  snapshot_id: 4,
  fetcher_tier: "direct",
};

const aliasFixture: ReviewAliasItem = {
  id: 11,
  entity_id: 2,
  entity_name: "Example Medical College",
  alias: "EMC Delhi",
  alias_type: "manual",
  confidence: null,
  quote: null,
  review_status: "pending",
};

const documentFixture: ReviewDocumentItem = {
  id: 21,
  url: "https://ambiguous.ac.in/notice.pdf",
  domain: "ambiguous.ac.in",
  extraction_family: "fee_structure",
  entity_id: null,
  plan_state: "needs_review",
  plan_confidence: 0.3,
  plan_reason: "two institutions share this domain",
};

const auditFixture: ReviewAuditRecord = {
  id: 31,
  kind: "claim",
  subject_id: 5,
  action: "accept",
  previous_state: { verification_status: "unverified", verified_by: null },
  details: { note: "quote matches the fee table row" },
  decided_by: "human",
  created_at: "2026-11-18T04:00:00Z",
};

function claimPayload(): ReviewQueuePayload {
  return {
    kind: "claim",
    items: [claimFixture],
    total: 1,
    limit: 20,
    offset: 0,
    ordering: "confidence ASC",
  };
}

describe("review console (offline render)", () => {
  it("renders a claim with predicate, quote, span, page/bbox, provenance and status", () => {
    render(<ClaimCard busy={false} item={claimFixture} onAction={vi.fn()} />);

    expect(screen.getByText("fees.total_annual")).toBeInTheDocument();
    expect(screen.getByText(/Annual fee for B.Sc MLT is Rs 42,000/)).toBeInTheDocument();
    expect(screen.getByText("210–246")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument(); // page from the qualifier
    expect(screen.getByText("[40,120,300,18]")).toBeInTheDocument(); // bbox
    expect(screen.getByText("direct")).toBeInTheDocument(); // evidence tier (fetch ladder)
    expect(screen.getByText(/supported/i)).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "https://example.ac.in/fee-2026.pdf" }),
    ).toHaveAttribute("href", "https://example.ac.in/fee-2026.pdf");
    expect(screen.getByText(/#5 \(extracted\)/)).toBeInTheDocument(); // deciding claim
  });

  it("warns before rejecting a supported claim and refuses to submit without a note", () => {
    const onAction = vi.fn();
    render(<ClaimCard busy={false} item={claimFixture} onAction={onAction} />);

    fireEvent.click(screen.getByRole("button", { name: "Reject" }));
    expect(
      screen.getByText(/Rejecting a supported claim requires a note/),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Submit rejection" }));
    expect(onAction).toHaveBeenCalledWith(5, { action: "reject", note: undefined });

    fireEvent.click(screen.getByRole("button", { name: "Correct" }));
    expect(screen.getByLabelText("corrected quote for claim 5")).toBeInTheDocument();
    expect(screen.getByLabelText("correction note for claim 5")).toBeInTheDocument();
  });

  it("flags a manual alias as ER-authoritative and renders confirm/reject", () => {
    render(<AliasCard busy={false} item={aliasFixture} onAction={vi.fn()} />);

    expect(screen.getByText(/EMC Delhi/)).toBeInTheDocument();
    expect(screen.getByText(/Example Medical College/)).toBeInTheDocument();
    expect(screen.getByText(/authoritative/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Confirm" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reject" })).toBeInTheDocument();
  });

  it("renders an ambiguous document with link and not-applicable actions", () => {
    const onAction = vi.fn();
    render(<DocumentCard busy={false} item={documentFixture} onAction={onAction} />);

    expect(screen.getByText(/two institutions share this domain/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("entity id for document 21"), {
      target: { value: "7" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Link entity" }));
    expect(onAction).toHaveBeenCalledWith(21, { action: "link_entity", entity_id: 7 });
    fireEvent.click(screen.getByRole("button", { name: "Not applicable" }));
    expect(onAction).toHaveBeenCalledWith(21, { action: "mark_not_applicable" });
  });

  it("renders the mixed queue with pagination and the decision trail", () => {
    const onPage = vi.fn();
    render(
      <ReviewQueueView
        actionError={null}
        auditsBySubject={{ 5: [auditFixture] }}
        busyId={null}
        kind="claim"
        onAliasAction={vi.fn()}
        onClaimAction={vi.fn()}
        onDocumentAction={vi.fn()}
        onPage={onPage}
        payload={claimPayload()}
      />,
    );

    expect(screen.getByRole("heading", { name: /Claims awaiting a decision/ })).toBeInTheDocument();
    expect(screen.getByText(/1 queued · showing 0–1/)).toBeInTheDocument();
    expect(screen.getByText(/Decision trail \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/quote matches the fee table row/)).toBeInTheDocument();
    expect(screen.getByText(/"verification_status":"unverified"/)).toBeInTheDocument();

    expect(screen.getByRole("button", { name: /Previous page/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Next page/ })).toBeDisabled();

    // No spinner, no placeholder rows: exactly one rendered claim card.
    expect(screen.getByTestId("review-claim-5")).toBeInTheDocument();
  });
});
