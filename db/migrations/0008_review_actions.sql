-- 0008_review_actions.sql — persistence for the human review console (WS-G).
--
-- Forward-only: 0001–0007 are applied and never edited.
--
-- The review console is the correction path for everything the machine decided
-- on its own. It needs three things the schema did not have yet:
--
--   1. A review disposition per alias. Claims already carry
--      verification_status/verified_by/verified_at/notes (0005) and documents
--      carry plan_state (0003); entity_aliases had only the evidence columns,
--      so a mined alias suggestion could not be accepted or rejected without
--      reusing `confirmed_by` ambiguously. `review_status` is the queue state
--      machine: pending -> confirmed | rejected. Confirm additionally writes
--      confirmed_by/confirmed_at (0002), which is what makes the alias a
--      durable human assertion. Rejected aliases stay readable — the audit
--      trail must survive — but they stop resolving mentions (the ER-facing
--      read filters them).
--
--   2. An append-only audit table for every human action, with the previous
--      state recorded. Decisions are always `decided_by='human'`: machine
--      verdicts live in er_audit and claims.verification_*, never here.
--
--   3. An index for the review queue ordering (low-confidence claims first).
--      Partial, like 0005's, so it stays small as queues drain.
--
-- No existing column changes meaning. Nothing here mutates claim content:
-- corrections remain INSERT + superseded_by (AGENTS.md data rules).

-- --------------------------------------------------------------------------- alias review state
ALTER TABLE entity_aliases
    ADD COLUMN IF NOT EXISTS review_status text NOT NULL DEFAULT 'pending';

ALTER TABLE entity_aliases
    DROP CONSTRAINT IF EXISTS entity_aliases_review_status_check;
ALTER TABLE entity_aliases
    ADD CONSTRAINT entity_aliases_review_status_check
    CHECK (review_status IN ('pending', 'confirmed', 'rejected'));

-- The queue reads pending rows; the partial index keeps that cheap as the
-- backlog drains into confirmed/rejected.
CREATE INDEX IF NOT EXISTS entity_aliases_review_queue_idx
    ON entity_aliases (id)
    WHERE review_status = 'pending';

-- --------------------------------------------------------------------------- review audit
-- subject_id deliberately has no foreign key: the audit must remain readable
-- even if a claim/entity/document row is later removed, and one table serves
-- three subject kinds.
CREATE TABLE IF NOT EXISTS review_audit (
    id              bigserial PRIMARY KEY,
    kind            text NOT NULL CHECK (kind IN ('claim', 'alias', 'document')),
    subject_id      bigint NOT NULL,
    action          text NOT NULL,
    previous_state  jsonb NOT NULL DEFAULT '{}'::jsonb,
    details_json    jsonb,
    decided_by      text NOT NULL DEFAULT 'human' CHECK (decided_by = 'human'),
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS review_audit_subject_idx
    ON review_audit (kind, subject_id, id);
CREATE INDEX IF NOT EXISTS review_audit_created_at_idx
    ON review_audit (created_at DESC);

-- --------------------------------------------------------------------------- claim queue ordering
-- Queue order is confidence ASC (lowest first), id ASC for determinism.
-- Partial: supported claims never queue and superseded ones never resolve.
CREATE INDEX IF NOT EXISTS claims_review_confidence_idx
    ON claims (confidence, id)
    WHERE verification_status <> 'supported' AND superseded_by IS NULL;
