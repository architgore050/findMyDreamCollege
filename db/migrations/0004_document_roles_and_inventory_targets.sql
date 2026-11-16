-- 0004_document_roles_and_inventory_targets.sql — give every stored document a
-- *role* so extraction is only asked to read evidence, and persist the
-- institution domains discovered from inventory payloads (WS-P2).
--
-- Forward-only: 0001–0003 are applied and never edited.
--
-- The first live crawl stored inventory payloads, navigation pages and
-- administrative PDFs with no distinction. The planner then abstained with
-- ``family_undetected`` on all of them, which read as an extraction failure when
-- it was really "we fetched the wrong kind of page". The role column makes that
-- difference explicit:
--
--   inventory   structured payloads that describe institutions (UGC JSON, AISHE
--               workbooks) — ingested into entities, never extracted;
--   navigation  listing / nav HTML that only exists to be followed — not
--               extractable, not an abstention;
--   evidence    PDFs and report pages (placement, annual report, AQAR/SSR,
--               NIRF, audit report, prospectus, fee structure) — the only role
--               the planner and extractor act on.
--
-- ``not_applicable`` joins the plan states: a document whose role rules it out
-- is *answered*, not a failed extraction. ``institution_domains`` is the missing
-- link between an inventory row ("this university publishes at example.ac.in")
-- and an institution-site crawl: a bounded, prioritised, restartable work queue.

-- --------------------------------------------------------------------------- documents
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS role text NOT NULL DEFAULT 'evidence';

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_role_check;
ALTER TABLE documents
    ADD CONSTRAINT documents_role_check
    CHECK (role IN ('inventory', 'navigation', 'evidence'));

-- The extract queue becomes "evidence only"; a partial index keeps that cheap.
CREATE INDEX IF NOT EXISTS documents_role_idx ON documents (role);
CREATE INDEX IF NOT EXISTS documents_evidence_role_idx
    ON documents (id) WHERE role = 'evidence';

-- ``not_applicable`` is a terminal, honest answer: the document was never a
-- candidate for extraction. It must never be counted as ``family_undetected``.
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_plan_state_check;
ALTER TABLE documents
    ADD CONSTRAINT documents_plan_state_check
    CHECK (plan_state IS NULL OR plan_state IN ('planned', 'linked', 'needs_review',
                                                'abstained', 'not_applicable'));

-- --------------------------------------------------------------------------- entities
-- Inventory rows carry a website and an institution type. Persisting them keeps
-- ingest idempotent and lets targeting priority use facts a human can check.
ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS website_url text,
    ADD COLUMN IF NOT EXISTS institution_type text;

CREATE INDEX IF NOT EXISTS entities_website_idx
    ON entities (website_url) WHERE website_url IS NOT NULL;

-- --------------------------------------------------------------------------- institution domains
-- One row per registrable domain discovered from an inventory website field.
-- ``state`` is the crawl work queue: pending -> queued (selected for this run)
-- -> planned (handed to the institution-site adapter) -> skipped.
CREATE TABLE IF NOT EXISTS institution_domains (
    domain          text PRIMARY KEY,
    seed_url        text NOT NULL,
    entity_id       bigint REFERENCES entities (id) ON DELETE SET NULL,
    discovered_by   text,
    state           text NOT NULL DEFAULT 'pending'
                    CHECK (state IN ('pending', 'queued', 'planned', 'skipped')),
    priority        integer NOT NULL DEFAULT 0,
    reason          text,
    first_seen      timestamptz NOT NULL DEFAULT now(),
    last_seen       timestamptz NOT NULL DEFAULT now(),
    last_queued_at  timestamptz
);

CREATE INDEX IF NOT EXISTS institution_domains_queue_idx
    ON institution_domains (state, priority DESC, domain);
CREATE INDEX IF NOT EXISTS institution_domains_entity_idx ON institution_domains (entity_id);

-- --------------------------------------------------------------------------- run ledger
-- ``ingest`` is a first-class stage: it reads inventory documents and writes
-- entities + domain seeds, and its counts belong in the same ledger.
ALTER TABLE pipeline_runs
    DROP CONSTRAINT IF EXISTS pipeline_runs_stage_check;
ALTER TABLE pipeline_runs
    ADD CONSTRAINT pipeline_runs_stage_check
    CHECK (stage IN ('crawl', 'extract', 'score', 'worker', 'ingest'));
