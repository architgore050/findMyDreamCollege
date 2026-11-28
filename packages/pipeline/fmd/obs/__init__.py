"""Observability: compute metrics (FMD-CLOUD-030).

Records enough per stage/phase/task to answer "what is the bottleneck?" —
append-only JSONL under ``data/compute-metrics/`` (gitignored except a README),
one file per ``run_id``, never containing applicant or document user data.
"""
