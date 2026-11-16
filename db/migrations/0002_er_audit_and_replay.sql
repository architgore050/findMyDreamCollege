-- 0002_er_audit_and_replay.sql — make entity resolution auditable and POST
-- sources replayable (brief §6, §10; WS-E).
--
-- Forward-only: 0001_init.sql is already applied and is never edited.
--
--   * er_audit is append-only: one row per merge / non-merge decision, tied to
--     the thresholds version and (optionally) the claims hash that was visible
--     when the decision was taken.
--   * entity_aliases gains the evidence span that produced the alias, so an
--     alias is as traceable as a claim, and `manual` becomes a persisted type.
--   * entities gain the derived blocking columns and a reversible merge link.
--   * documents record the request shape so admin-ajax POST sources replay.

-- --------------------------------------------------------------------------- merge audit
CREATE TABLE IF NOT EXISTS er_audit (
    id                   bigserial PRIMARY KEY,
    subject_entity_id    bigint NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    candidate_entity_id  bigint NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    band                 text NOT NULL
                         CHECK (band IN ('accept', 'reject', 'adjudicate')),
    score                double precision NOT NULL
                         CHECK (score >= 0 AND score <= 1),
    features_json        jsonb NOT NULL DEFAULT '{}'::jsonb,
    thresholds_version   text NOT NULL,
    model                text,
    model_verdict        text CHECK (model_verdict IS NULL OR
                                     model_verdict IN ('same', 'different', 'unsure')),
    model_reason         text,
    decided_by           text NOT NULL CHECK (decided_by IN ('auto', 'model', 'human')),
    claims_hash          text,
    created_at           timestamptz NOT NULL DEFAULT now(),
    CHECK (subject_entity_id <> candidate_entity_id)
);

CREATE INDEX IF NOT EXISTS er_audit_pair_idx
    ON er_audit (subject_entity_id, candidate_entity_id);
CREATE INDEX IF NOT EXISTS er_audit_created_at_idx
    ON er_audit (created_at DESC);

-- --------------------------------------------------------------------------- aliases
ALTER TABLE entity_aliases
    ADD COLUMN IF NOT EXISTS span_start integer,
    ADD COLUMN IF NOT EXISTS span_end integer,
    ADD COLUMN IF NOT EXISTS quote text,
    ADD COLUMN IF NOT EXISTS source_document_id bigint
        REFERENCES documents (id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS source_extraction_id bigint
        REFERENCES extractions (id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS confirmed_by text,
    ADD COLUMN IF NOT EXISTS confirmed_at timestamptz;

-- Extend the closed alias_type set with the human-confirmed marker.
ALTER TABLE entity_aliases
    DROP CONSTRAINT IF EXISTS entity_aliases_alias_type_check;
ALTER TABLE entity_aliases
    ADD CONSTRAINT entity_aliases_alias_type_check
    CHECK (alias_type IN ('acronym', 'former_name', 'transliteration',
                          'misspelling', 'local_name', 'manual'));

-- A span must be a well-formed interval when either endpoint is present.
ALTER TABLE entity_aliases
    DROP CONSTRAINT IF EXISTS entity_aliases_span_check;
ALTER TABLE entity_aliases
    ADD CONSTRAINT entity_aliases_span_check
    CHECK (span_start IS NULL OR span_end IS NULL
           OR (span_start >= 0 AND span_end > span_start));

CREATE INDEX IF NOT EXISTS entity_aliases_manual_idx
    ON entity_aliases (entity_id) WHERE alias_type = 'manual';

-- --------------------------------------------------------------------------- entities
ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS normalised_name text,
    ADD COLUMN IF NOT EXISTS blocking_key text,
    ADD COLUMN IF NOT EXISTS merged_into_entity_id bigint
        REFERENCES entities (id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS merged_at timestamptz;

CREATE INDEX IF NOT EXISTS entities_normalised_name_idx
    ON entities (normalised_name);
CREATE INDEX IF NOT EXISTS entities_blocking_key_idx
    ON entities (blocking_key);
CREATE INDEX IF NOT EXISTS entities_merged_into_idx
    ON entities (merged_into_entity_id);

-- --------------------------------------------------------------------------- documents
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS http_method text NOT NULL DEFAULT 'GET',
    ADD COLUMN IF NOT EXISTS http_body jsonb,
    ADD COLUMN IF NOT EXISTS request_headers jsonb;

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_http_method_check;
ALTER TABLE documents
    ADD CONSTRAINT documents_http_method_check
    CHECK (http_method IN ('GET', 'POST'));

-- The frontier row must carry the same request shape, otherwise a queued POST
-- page loses its body between enqueue and claim.
ALTER TABLE crawl_frontier
    ADD COLUMN IF NOT EXISTS http_method text NOT NULL DEFAULT 'GET',
    ADD COLUMN IF NOT EXISTS http_body jsonb,
    ADD COLUMN IF NOT EXISTS request_headers jsonb;

ALTER TABLE crawl_frontier
    DROP CONSTRAINT IF EXISTS crawl_frontier_http_method_check;
ALTER TABLE crawl_frontier
    ADD CONSTRAINT crawl_frontier_http_method_check
    CHECK (http_method IN ('GET', 'POST'));

