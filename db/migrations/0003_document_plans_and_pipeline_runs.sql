-- 0003_document_plans_and_pipeline_runs.sql — persist the crawl/extract plan on
-- the evidence rows and the run ledger in SQL (brief §6, §7, §9, §21; WS-S3).
--
-- Forward-only: 0001_init.sql and 0002_er_audit_and_replay.sql are already
-- applied and are never edited.
--
--   * ``documents.source_id`` / ``documents.extraction_family`` /
--     ``documents.entity_id`` replace the ``request_headers['fmd_plan']`` hack.
--     A document now states, in columns, which registry source produced it,
--     which extraction family its text belongs to and which entity owns it, so
--     ``fmd extract`` can select and build a target with SQL instead of
--     re-parsing a JSON blob out of an HTTP header.
--   * ``plan_state`` / ``plan_confidence`` / ``plan_reason`` keep abstentions
--     reviewable: an ambiguous entity link is stored as ``needs_review`` with a
--     NULL entity, never guessed.
--   * ``crawl_frontier.source_id`` survives a restart, so the frontier can say
--     which registry entry a queued URL belongs to.
--   * ``pipeline_runs`` makes the run ledger queryable (counts + errors as JSONB,
--     status as a closed set). The stdout JSON summary is unchanged.
--
-- ``documents.source_id`` / ``crawl_frontier.source_id`` are intentionally not
-- foreign keys: a discovered ref may name a source that has not been seeded yet
-- and a rejected document must still be recorded. ``documents.entity_id`` is a
-- real FK because an entity link is a claim about a row that must exist.

-- --------------------------------------------------------------------------- documents
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS source_id text,
    ADD COLUMN IF NOT EXISTS extraction_family text,
    ADD COLUMN IF NOT EXISTS entity_id bigint
        REFERENCES entities (id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS plan_state text,
    ADD COLUMN IF NOT EXISTS plan_confidence double precision,
    ADD COLUMN IF NOT EXISTS plan_reason text;

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_plan_state_check;
ALTER TABLE documents
    ADD CONSTRAINT documents_plan_state_check
    CHECK (plan_state IS NULL OR plan_state IN ('planned', 'linked', 'needs_review',
                                                'abstained'));

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_plan_confidence_check;
ALTER TABLE documents
    ADD CONSTRAINT documents_plan_confidence_check
    CHECK (plan_confidence IS NULL OR (plan_confidence >= 0 AND plan_confidence <= 1));

CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source_id);
CREATE INDEX IF NOT EXISTS documents_family_idx ON documents (extraction_family);
CREATE INDEX IF NOT EXISTS documents_entity_idx ON documents (entity_id);
-- The extract queue is "planned documents only"; a partial index keeps that scan
-- cheap as the corpus grows.
CREATE INDEX IF NOT EXISTS documents_extractable_idx
    ON documents (id) WHERE extraction_family IS NOT NULL AND entity_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS documents_review_idx
    ON documents (plan_state) WHERE plan_state = 'needs_review';

-- --------------------------------------------------------------------------- frontier
ALTER TABLE crawl_frontier
    ADD COLUMN IF NOT EXISTS source_id text,
    ADD COLUMN IF NOT EXISTS extraction_family text,
    ADD COLUMN IF NOT EXISTS entity_id bigint;

CREATE INDEX IF NOT EXISTS crawl_frontier_source_idx ON crawl_frontier (source_id);
CREATE INDEX IF NOT EXISTS crawl_frontier_source_state_idx
    ON crawl_frontier (source_id, state);
CREATE INDEX IF NOT EXISTS crawl_frontier_entity_idx ON crawl_frontier (entity_id);

-- --------------------------------------------------------------------------- run ledger
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          text PRIMARY KEY,
    stage           text NOT NULL
                    CHECK (stage IN ('crawl', 'extract', 'score', 'worker')),
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz,
    counts_json     jsonb NOT NULL DEFAULT '{}'::jsonb,
    errors_json     jsonb NOT NULL DEFAULT '[]'::jsonb,
    status          text NOT NULL
                    CHECK (status IN ('running', 'succeeded', 'failed', 'dry_run')),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

CREATE INDEX IF NOT EXISTS pipeline_runs_stage_started_idx
    ON pipeline_runs (stage, started_at DESC);
CREATE INDEX IF NOT EXISTS pipeline_runs_status_idx ON pipeline_runs (status);
