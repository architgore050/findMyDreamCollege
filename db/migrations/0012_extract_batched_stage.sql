-- 0012_extract_batched_stage — allow the batched-extract stage in the run ledger.
--
-- Forward-only: 0001–0011 are applied and never edited.
--
-- `fmd extract --batched` records its runs as stage 'extract_batched' (the
-- pipeline_runs check constraint allowed only crawl/extract/score/worker/
-- ingest), so every batched run failed to persist ("could not persist run …
-- stage 'extract_batched'"). Extend the constraint; the old values stay allowed.

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_stage_check;
ALTER TABLE pipeline_runs
    ADD CONSTRAINT pipeline_runs_stage_check
    CHECK (stage IN ('crawl', 'extract', 'score', 'worker', 'ingest', 'extract_batched'));
