-- 0007_dedupe_basis.sql — make every dedupe decision auditable.
--
-- Forward-only: 0001–0006 are applied and never edited.
--
-- The 2026-11-18 audit found 73 distinct PDFs dropped as
-- `near_duplicate_simhash` because the fingerprint was computed over the fetched
-- *container bytes* decoded as UTF-8 (`%PDF` headers, object streams, fonts),
-- not over the document's text. The dropped documents have no snapshot row, so
-- the only trace of the decision was an in-memory run-summary entry.
--
-- These columns persist the decision next to the document that was judged:
--
--   text_sha256                    sha256 of normalised extracted/decoded text
--                                  (NULL when no text layer could be read)
--   dedupe_mode                    exact | simhash | identity
--   dedupe_basis                   raw_bytes | extracted_text | identity
--   dedupe_similarity              [0,1] similarity to compared_document_id
--   dedupe_reason                  unchanged_content | duplicate_sha256 |
--                                  duplicate_text_sha256 |
--                                  near_duplicate_simhash |
--                                  near_duplicate_identity
--   dedupe_compared_document_id    the document this one was compared against
--   dedupe_simhash                 64-bit fingerprint as 16 hex chars, so a
--                                  later review can recompute the distance
--
-- A stored snapshot also records `text_sha256`/`dedupe_basis`: a snapshot with
-- no basis is not evidence of what it was compared against.
--
-- The FK is deliberately omitted: a decision may point at a document row that is
-- later removed by an operator purge, and losing the justification for a drop
-- must not cascade into losing the drop itself.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS text_sha256 text;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS dedupe_mode text;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS dedupe_basis text;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS dedupe_similarity double precision;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS dedupe_reason text;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS dedupe_compared_document_id bigint;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS dedupe_simhash text;

ALTER TABLE documents
    ADD CONSTRAINT documents_text_sha256_check
    CHECK (text_sha256 IS NULL OR text_sha256 ~ '^[0-9a-f]{64}$');

ALTER TABLE documents
    ADD CONSTRAINT documents_dedupe_mode_check
    CHECK (dedupe_mode IS NULL OR dedupe_mode IN ('exact', 'simhash', 'identity'));

ALTER TABLE documents
    ADD CONSTRAINT documents_dedupe_basis_check
    CHECK (dedupe_basis IS NULL OR dedupe_basis IN ('raw_bytes', 'extracted_text', 'identity'));

ALTER TABLE documents
    ADD CONSTRAINT documents_dedupe_similarity_check
    CHECK (dedupe_similarity IS NULL OR (dedupe_similarity >= 0 AND dedupe_similarity <= 1));

ALTER TABLE snapshots
    ADD COLUMN IF NOT EXISTS text_sha256 text;

ALTER TABLE snapshots
    ADD COLUMN IF NOT EXISTS dedupe_basis text;

ALTER TABLE snapshots
    ADD CONSTRAINT snapshots_text_sha256_check
    CHECK (text_sha256 IS NULL OR text_sha256 ~ '^[0-9a-f]{64}$');

ALTER TABLE snapshots
    ADD CONSTRAINT snapshots_dedupe_basis_check
    CHECK (dedupe_basis IS NULL OR dedupe_basis IN ('raw_bytes', 'extracted_text', 'identity'));

CREATE INDEX IF NOT EXISTS documents_text_sha256_idx ON documents (text_sha256);
CREATE INDEX IF NOT EXISTS snapshots_text_sha256_idx ON snapshots (text_sha256);
