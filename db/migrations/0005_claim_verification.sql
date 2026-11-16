-- 0005_claim_verification.sql — record the second-pass verdict on a claim.
--
-- Forward-only: 0001–0004 are applied and never edited.
--
-- The verifier is a structurally different call from extraction: it sees the
-- bounded source window and the *existing* quote, never the claimed value, and
-- returns a judgement only (`supported | contradicted | absent | wrong_field`).
-- Its result belongs on the claim so the review console and downstream
-- consumers can exclude a claim without mutating claim content.
--
-- Content columns (value/quote/span/confidence) are never touched by
-- verification; only the four columns below are written. A claim that fails
-- verification stays in the append-only evidence table with
-- `verification_status = 'rejected'` and a machine-readable reason in
-- `verification_notes`, so the rejection itself is auditable.
--
-- `unverified` is the default: a run that runs out of verifier budget, or a
-- claim extracted before this migration, is honestly unknown — not supported.

ALTER TABLE claims
    ADD COLUMN IF NOT EXISTS verification_status text NOT NULL DEFAULT 'unverified';

ALTER TABLE claims
    DROP CONSTRAINT IF EXISTS claims_verification_status_check;
ALTER TABLE claims
    ADD CONSTRAINT claims_verification_status_check
    CHECK (verification_status IN ('unverified', 'supported', 'rejected'));

ALTER TABLE claims
    ADD COLUMN IF NOT EXISTS verified_by text;

ALTER TABLE claims
    ADD COLUMN IF NOT EXISTS verified_at timestamptz;

ALTER TABLE claims
    ADD COLUMN IF NOT EXISTS verification_notes jsonb;

-- Review queue: everything that is not positively supported (unverified,
-- rejected) plus the index needed to page it by claim id. Partial so the index
-- stays small as most claims become supported.
CREATE INDEX IF NOT EXISTS claims_review_queue_idx
    ON claims (verification_status, id)
    WHERE verification_status <> 'supported';

-- A verifier run reads pending claims for one extraction in order.
CREATE INDEX IF NOT EXISTS claims_extraction_verification_idx
    ON claims (extraction_id, verification_status);
