-- 0006_programme_materialisation.sql — evidence-backed programme rows.
--
-- Forward-only: 0001–0005 are applied and never edited.
--
-- A programme is materialised from *verified* claims, one row per
-- (institution, ER-normalised programme name). The columns below make the
-- provenance of every attribute explicit:
--
--   normalised_name   stable grouping key (ER normalisation + registry alias map)
--   field_sources     {"seats": "42", "duration_months": "17", ...} claim id per
--                     published attribute; a field with no claim stays NULL and
--                     is absent from this map
--   needs_review      true when grouping/attribute confidence is not provable
--                     (name collision, unresolved affiliation, defaulted level)
--   review_reasons    machine-readable reasons, display verbatim in the UI
--   source_family     predicate family that supplied the name
--                     (curriculum | seats | accreditation | ...)
--
-- `name` and every nullable attribute stay NULL/absent rather than guessed:
-- a programme row may legitimately carry only a name and institution.

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS normalised_name text;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS field_sources jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS needs_review boolean NOT NULL DEFAULT false;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS review_reasons jsonb NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE programmes
    ADD COLUMN IF NOT EXISTS source_family text;

-- Existing rows predate materialisation; their name is the only stable key.
UPDATE programmes
SET normalised_name = lower(regexp_replace(btrim(name), '\s+', ' ', 'g'))
WHERE normalised_name IS NULL;

ALTER TABLE programmes
    ALTER COLUMN normalised_name SET NOT NULL;

-- Idempotent re-ingest: one row per institution + normalised programme name.
CREATE UNIQUE INDEX IF NOT EXISTS programmes_institution_normalised_key
    ON programmes (institution_id, normalised_name);

CREATE INDEX IF NOT EXISTS programmes_review_idx
    ON programmes (needs_review)
    WHERE needs_review;
