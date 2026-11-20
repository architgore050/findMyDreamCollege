-- 0009_document_sections.sql — the document section tree (W8 section-structure).
--
-- Forward-only: 0001–0008 are applied and never edited.
--
-- A parsed PDF has a natural outline; before this patch it was discarded and
-- the span resolver had to be "unique" inside a heuristic ±200-char window or
-- a flat heading block. This table persists the heading tree extracted at
-- acquisition time (Docling labels first, text heuristics otherwise) so the
-- resolver can scope quotes to real sections, prompts can carry a breadcrumb,
-- and chunking can cut at section boundaries.
--
-- One row per heading node. The tree shape is carried by parent_id (adjacency
-- list); level is denormalised for the narrowest-containing lookup and the
-- no-gap normalisation check, and the char ranges index into the canonical
-- text of the document's snapshot (the same text extractions.text carries).
--
-- Sections are DERIVED structure, never evidence: they are rebuilt wholesale
-- from a fresh acquisition run (delete + insert in one transaction), so a
-- re-extraction cannot leave a stale tree behind. Claims stay append-only;
-- nothing here touches claim rows.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS, and the
-- migration runner records the checksum so a re-apply is a no-op.

CREATE TABLE IF NOT EXISTS document_sections (
    id                 bigserial PRIMARY KEY,
    document_id        bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    parent_id          bigint REFERENCES document_sections (id) ON DELETE CASCADE,
    node_key           integer NOT NULL CHECK (node_key >= 1),
    level              integer NOT NULL CHECK (level >= 1),
    title              text NOT NULL,
    char_start         integer NOT NULL CHECK (char_start >= 0),
    char_end           integer NOT NULL CHECK (char_end >= char_start),
    heading_char_start integer NOT NULL CHECK (heading_char_start >= 0),
    heading_char_end   integer NOT NULL CHECK (heading_char_end >= heading_char_start),
    page_no            integer,
    bbox_json          jsonb,
    extractor          text NOT NULL,
    extractor_version  text,
    created_at         timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, node_key)
);

-- A parent must belong to the same document; a same-table CHECK cannot see
-- the parent row, so the adjacency list is guarded by a trigger instead.
CREATE OR REPLACE FUNCTION fmd_document_sections_parent_same_document()
RETURNS trigger AS $$
BEGIN
    IF NEW.parent_id IS NOT NULL THEN
        IF EXISTS (
            SELECT 1 FROM document_sections parent
            WHERE parent.id = NEW.parent_id AND parent.document_id <> NEW.document_id
        ) THEN
            RAISE EXCEPTION 'document_sections.parent_id % crosses documents',
                NEW.parent_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS document_sections_parent_guard ON document_sections;
CREATE TRIGGER document_sections_parent_guard
    BEFORE INSERT OR UPDATE ON document_sections
    FOR EACH ROW EXECUTE FUNCTION fmd_document_sections_parent_same_document();

-- Lookup path for the resolver and the review UI: the deepest section
-- containing a span is found on the (document_id, char_start, level DESC)
-- btree and the breadcrumb walks parent_id, so a per-claim lookup is an
-- index scan plus a depth-bounded walk — never a linear scan.
CREATE INDEX IF NOT EXISTS document_sections_document_span_idx
    ON document_sections (document_id, char_start, level DESC);
