-- 0011_document_parses.sql — persistent T1 parse cache (W10 batched acquire).
--
-- Forward-only: 0001–0010 are applied and never edited.
--
-- Why: extraction was a single fused loop — Docling T1 conversion on CPU
-- immediately followed by the LLM call — so the GPU idled during every
-- conversion and a re-run re-converted every document. This table is the
-- lookup half of the parse cache; the payload itself is content-addressed in
-- the snapshot object store (objects/<aa>/<sha256>), written with the same
-- atomic temp→fsync→rename scheme as evidence bytes.
--
-- Key = sha256 over a canonical JSON of
--   (snapshot_sha256, tier, policy_digest, schema_version)
-- where:
--   snapshot_sha256  sha256 of the stored document bytes (the snapshot object);
--   tier             the acquisition tier whose parse is cached — "T1" only in
--                    this patch (T0 is cheap to redo; T2/OCR is out of scope);
--   policy_digest    sha256 over the device-independent, parse-shaping subset
--                    of config/acquire.yaml (tiers, outline, anomaly,
--                    backends.T1 tool/model/granularity). The Docling device
--                    is deliberately NOT part of the key: a CUDA parse is
--                    reusable by a CPU run and vice versa. A policy or schema
--                    change invalidates; a device change must not.
--   schema_version   payload schema version (acquire.yaml:cache.schema_version).
--
-- A row is written only after the payload object exists in the store, so an
-- interrupted write (object without row) is never read back: the next run sees
-- a miss and reconverts. A cache entry represents a complete document only —
-- a failed conversion writes nothing.
--
-- Docling's library version is recorded (provenance) but is NOT part of the
-- key: upgrading docling does not invalidate stored parses. That is a known,
-- documented gap (runbook §6), not an accident.

CREATE TABLE IF NOT EXISTS document_parses (
    key_digest       text PRIMARY KEY,
    snapshot_sha256  text NOT NULL,
    tier             text NOT NULL,
    policy_digest    text NOT NULL,
    schema_version   integer NOT NULL CHECK (schema_version >= 1),
    object_sha256    text NOT NULL,
    pages            integer NOT NULL CHECK (pages >= 0),
    parser           text NOT NULL,
    parser_version   text,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS document_parses_snapshot_idx
    ON document_parses (snapshot_sha256);
