-- 0010_programme_scoping.sql — three-tier programme scoping (W9).
--
-- Forward-only: 0001–0009 are applied and never edited.
--
-- The W8 materialisation put every named programme into the pool, so MBBS and
-- Agriculture rows reached the ranked set and a row-numbered duplicate
-- ("3. B.Sc. Medical Laboratory Technology") became a second row. Scoping
-- decides whether a programme IS the product's target field (Medical
-- Laboratory Technology / Science) and records HOW it was decided:
--
--   in_scope            false rows are kept (audit) but never scored or ranked
--   scope_reason        machine-readable why (`rule_exclude:mbbs`, ...)
--   scope_tier          rule | semantic | manual — the tier that decided
--   scope_evidence      evidence list: claim ids and quoted strings the
--                       decision relied on (a semantic `yes` MUST cite some)
--   scope_model         model that produced a semantic decision (audit)
--   scope_prompt_version prompt version for a semantic decision (audit)
--   scope_cached        true when the decision came from the cache, not a call
--   scope_decided_at    when the current scope state was written
--
-- Existing rows default to in_scope=true: they keep the pre-W9 behaviour
-- until `fmd programmes requalify` (or the next materialise) scopes them.
--
-- programme_scope_decisions is the durable classifier cache AND the learning
-- loop store. One row per decision key — sha256 over (normalised name,
-- institution type, definition version) — so one call is made per new name,
-- not per row. Human decisions (`fmd programmes decide`) are written here
-- with tier='manual' and decided_by='human'; Tier 1 reads them first, so a
-- human decision always wins over the classifier and is consumed as a manual
-- alias on every later run. Rows are never deleted: superseded decisions are
-- recorded, the audit trail survives.

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS in_scope boolean NOT NULL DEFAULT true;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS scope_reason text;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS scope_tier text;

ALTER TABLE programmes
    DROP CONSTRAINT IF EXISTS programmes_scope_tier_check;
ALTER TABLE programmes
    ADD CONSTRAINT programmes_scope_tier_check
    CHECK (scope_tier IS NULL OR scope_tier IN ('rule', 'semantic', 'manual'));

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS scope_evidence jsonb NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS scope_model text;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS scope_prompt_version text;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS scope_cached boolean NOT NULL DEFAULT false;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS scope_decided_at timestamptz;

CREATE INDEX IF NOT EXISTS programmes_scope_idx
    ON programmes (in_scope)
    WHERE NOT in_scope;

CREATE TABLE IF NOT EXISTS programme_scope_decisions (
    id                bigserial PRIMARY KEY,
    cache_key         text NOT NULL UNIQUE,
    normalised_name   text NOT NULL,
    raw_name          text,
    institution_type  text,
    country           char(2),
    decision          text NOT NULL CHECK (decision IN ('in_scope', 'out_of_scope', 'uncertain')),
    tier              text NOT NULL CHECK (tier IN ('rule', 'semantic', 'manual')),
    confidence        numeric,
    rationale         text,
    evidence          jsonb NOT NULL DEFAULT '[]'::jsonb,
    model             text,
    prompt_version    text,
    cached            boolean NOT NULL DEFAULT false,
    needs_review      boolean NOT NULL DEFAULT false,
    flags             jsonb NOT NULL DEFAULT '[]'::jsonb,
    definition_version text NOT NULL,
    decided_by        text NOT NULL DEFAULT 'classifier' CHECK (decided_by IN ('classifier', 'human')),
    note              text,
    created_at        timestamptz NOT NULL DEFAULT now()
);

-- The review queue reads uncertain decisions; partial, like 0005/0008 queues.
CREATE INDEX IF NOT EXISTS programme_scope_decisions_review_idx
    ON programme_scope_decisions (id)
    WHERE needs_review;
