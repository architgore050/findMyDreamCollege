"""``fmd`` command line entrypoint.

Only workstream-A commands do work. Commands owned by other workstreams fail
with a non-zero exit and a pointer to the brief rather than pretending to run.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from fmd import __version__
from fmd.config import ConfigError, load_registry
from fmd.db import Database, DatabaseError, MigrationError, run_migrations
from fmd.seed import SeedError, seed_database


class NotImplementedYet(RuntimeError):
    def __init__(self, command: str, workstream: str, section: str) -> None:
        self.command = command
        self.workstream = workstream
        self.section = section
        super().__init__(f"not implemented ({workstream}, see brief §{section})")


# Commands owned by other workstreams (brief §21). Keeping this table explicit
# means a stub cannot silently succeed before its workstream lands. `eval` is
# handled in main(): no-argument eval remains a stub, --golden is implemented.
UNIMPLEMENTED: dict[str, tuple[str, str]] = {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmd",
        description="findMyDreamCollege evidence pipeline (workstream A foundation).",
    )
    parser.add_argument("--version", action="version", version=f"fmd {__version__}")
    parser.add_argument(
        "--database-url",
        default=None,
        help="override DATABASE_URL (defaults to DATABASE_URL or the local compose port)",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="directory containing config/*.yaml (defaults to FMD_CONFIG_DIR or repo config/)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("migrate", help="apply pending SQL migrations")
    subparsers.add_parser("seed", help="load countries/sources/exams config into the database")

    crawl_parser = subparsers.add_parser("crawl", help="run the crawl frontier (WS-O glue)")
    crawl_parser.add_argument(
        "--limit", type=int, default=None, help="maximum documents to plan and fetch"
    )
    crawl_parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="ID",
        help="only this registry source id (repeatable)",
    )
    crawl_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned URLs and write nothing",
    )

    extract_parser = subparsers.add_parser(
        "extract", help="snapshots -> extractions -> claims (WS-O glue)"
    )
    extract_parser.add_argument(
        "--limit", type=int, default=None, help="maximum snapshots to process"
    )
    extract_parser.add_argument(
        "--force",
        action="store_true",
        help="re-extract snapshots already processed at this tool version",
    )
    # ---- W10 phase split: parse cache, stage flags, batched orchestrator ----
    extract_parser.add_argument(
        "--from-cache",
        action="store_true",
        help="extract only from cached parses; guaranteed to invoke no Docling "
        "(a document without a cached parse is recorded missing_parse_cache)",
    )
    extract_parser.add_argument(
        "--no-acquire",
        action="store_true",
        help="skip inline conversion; a document without a cached parse is "
        "recorded missing_parse_cache and the run continues",
    )
    extract_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan (documents, known pages, expected cache hits), "
        "write nothing, contact no model URL",
    )
    extract_parser.add_argument(
        "--batched",
        action="store_true",
        help="batched orchestrator: per cycle stop server -> acquire one "
        "page-batch on CUDA -> start server (/health ready) -> extract the "
        "batch from cache; the server is stopped on every exit path unless "
        "--keep-server",
    )
    extract_parser.add_argument(
        "--batch-pages",
        type=int,
        default=None,
        help="target converted pages per cycle (default: acquire.yaml batch.pages); "
        "sizing is evaluated only at document boundaries and the document that "
        "crosses the limit always finishes",
    )
    extract_parser.add_argument(
        "--keep-server",
        action="store_true",
        help="leave the model server running after the batched run",
    )
    extract_parser.add_argument(
        "--server-script",
        default=None,
        help="server-control script (start|stop|status|probe); default: "
        "acquire.yaml batch.server_script",
    )
    extract_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Docling device for acquisition (default: acquire.yaml batch.device "
        "for --batched, else FMD_DOCLING_DEVICE/config)",
    )

    acquire_parser = subparsers.add_parser(
        "acquire",
        help="convert documents into cached parses (W10); makes zero LLM calls",
    )
    acquire_parser.add_argument(
        "--limit", type=int, default=None, help="maximum documents to acquire"
    )
    acquire_parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="page budget: evaluated only at document boundaries — the document "
        "that crosses it is converted in full, then acquisition stops",
    )
    acquire_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Docling device for this run (overrides FMD_DOCLING_DEVICE/config)",
    )
    acquire_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan (documents, known pages, expected cache hits), "
        "convert nothing, write nothing",
    )

    plan_parser = subparsers.add_parser(
        "plan",
        help="plan documents that should be extractable but are not (recovery/backfill)",
    )
    plan_parser.add_argument(
        "--limit", type=int, default=None, help="maximum documents to decide this run"
    )
    plan_parser.add_argument(
        "--force",
        action="store_true",
        help="also re-decide decided documents that are stale at the current planner version",
    )
    plan_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute every decision and print the counters, write nothing",
    )
    plan_parser.add_argument(
        "--admit-evidence",
        type=Path,
        default=None,
        help=(
            "apply a validated semantic-admissions file (JSON list of "
            "{document_url, family, quote}) instead of planning: every admission "
            "must cite a verbatim quote from the document's text sample; "
            "ambiguous documents are refused, never guessed"
        ),
    )

    report_parser = subparsers.add_parser(
        "classify-report",
        help="read-only: why each evidence document is missing a family (wave v1-w5)",
    )
    report_parser.add_argument(
        "--limit", type=int, default=2000, help="maximum documents to diagnose"
    )
    report_parser.add_argument(
        "--with-text",
        action="store_true",
        help="fetch the bounded T0 sample per document (CPU-only) and replay the text chain",
    )
    report_parser.add_argument(
        "--json-out", type=Path, default=None, help="write the full report JSON here"
    )

    coverage_parser = subparsers.add_parser(
        "coverage-report",
        help=(
            "read-only: per-programme scoring coverage — which of the 7 components "
            "are populated and which claims feed them (wave v1-w7)"
        ),
    )
    coverage_parser.add_argument(
        "--limit", type=int, default=500, help="maximum in-scope programmes to diagnose"
    )
    coverage_parser.add_argument(
        "--json-out", type=Path, default=None, help="write the full report JSON here"
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="second pass: judge unverified claims against their source window",
    )
    verify_parser.add_argument(
        "--limit", type=int, default=500, help="maximum pending claims (default 500)"
    )
    verify_parser.add_argument(
        "--mock",
        action="store_true",
        help="replay recorded verdicts instead of calling a provider (offline)",
    )
    verify_parser.add_argument(
        "--fixtures",
        default=None,
        help="fixture directory for --mock (defaults to providers.mock.fixtures_dir)",
    )

    worker_parser = subparsers.add_parser(
        "worker", help="long-running crawl -> extract -> score loop (WS-O glue)"
    )
    worker_parser.add_argument(
        "--limit", type=int, default=None, help="maximum items per stage per tick"
    )
    worker_parser.add_argument(
        "--idle-seconds",
        type=float,
        default=5.0,
        help="sleep between ticks (default 5)",
    )
    worker_parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="stop after N ticks (default: run until SIGTERM/SIGINT)",
    )

    eval_parser = subparsers.add_parser(
        "eval", help="golden set, provenance coverage, drift (offline, WS-F)"
    )
    eval_parser.add_argument(
        "--golden",
        default=None,
        help="directory of golden YAML files (required; e.g. data/golden)",
    )
    eval_parser.add_argument(
        "--report-out",
        default=None,
        help="write the JSON report here (created if missing)",
    )
    eval_parser.add_argument(
        "--baseline",
        default=None,
        help="previous JSON report to diff against (drift); omitted means no drift",
    )
    eval_parser.add_argument(
        "--fail-under",
        type=float,
        default=1.0,
        help="hard gate: minimum provenance coverage (default 1.0)",
    )
    eval_parser.add_argument(
        "--verify-fixtures",
        default=None,
        help=(
            "optional committed verification case directory; adds verified_rate, "
            "fabrication_caught and correct_rejected to the report (does not "
            "change the provenance gate)"
        ),
    )
    eval_parser.add_argument(
        "--eval-kind",
        choices=["pipeline", "product", "verdict"],
        default="pipeline",
        help=(
            "pipeline (default) = golden-set extraction/provenance eval; "
            "product = proxy/self-drift/cv-stability streams; "
            "verdict = verdict-fast research-agent canary or expert-panel scoring"
        ),
    )
    eval_parser.add_argument(
        "--stream",
        choices=[
            "golden",
            "nirf-proxy",
            "naac-proxy",
            "self-drift",
            "cv-stability",
            "prompt-effectiveness",
            "verdict-fast",
            "verdict-live",
            "expert-panel",
        ],
        default=None,
        help=(
            "product/verdict stream to run: nirf-proxy|naac-proxy|self-drift|"
            "cv-stability|prompt-effectiveness|expert-panel (--eval-kind "
            "product) or verdict-fast|verdict-live (--eval-kind verdict)"
        ),
    )
    eval_parser.add_argument(
        "--fixture-path",
        default=None,
        help=(
            "stream input: shortlist JSON (proxy/self-drift/cv-stability/"
            "prompt-effectiveness), directory of profile fixtures "
            "(verdict-fast), or directory of panel-rating YAMLs + "
            "product_ranking.json (expert-panel)"
        ),
    )
    eval_parser.add_argument(
        "--previous-snapshot",
        default=None,
        help="self-drift only: the previous snapshot JSON to diff against",
    )
    eval_parser.add_argument(
        "--truth-dir",
        default=None,
        help="proxy streams: truth table directory (default: repo data/truth)",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="cv-stability: k for top-k survival (default 5)",
    )
    eval_parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help=(
            "cv-stability: +-epsilon ABSOLUTE weight perturbation in weight "
            "units (config/weights.yaml uses a 0-100 scale; default 2.0)"
        ),
    )
    eval_parser.add_argument(
        "--holdout-n",
        type=int,
        default=3,
        help="cv-stability: maximum N for leave-N-out (default 3)",
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="product/verdict kinds: compute and print a summary without writing a report",
    )
    eval_parser.add_argument(
        "--audit-sample",
        type=float,
        default=None,
        help=(
            "verdict-live: sample a seeded random 10-20 percent of verdict cases "
            "(fraction is clamped into that band) and write an audit worksheet JSON "
            "next to the report; every incorrect/partially_correct verdict is always "
            "sampled (docs/v1/eval/04-verdict-audit-protocol.md section 2.1)"
        ),
    )
    eval_parser.add_argument(
        "--audit-seed",
        type=int,
        default=13,
        help="seed for --audit-sample (recorded in the worksheet; default 13)",
    )
    eval_parser.add_argument(
        "--audit-score",
        default=None,
        metavar="WORKSHEET_JSON",
        help=(
            "verdict: score a completed audit worksheet against the >=70 percent "
            "human-agreement gate and print PASS/FAIL/INSUFFICIENT (exit 0/1/3); "
            "the gate is reported, never enforced in CI"
        ),
    )
    eval_parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="verdict-live: hard cap on model turns per profile (default 8)",
    )
    eval_parser.add_argument(
        "--max-fetches",
        type=int,
        default=6,
        help="verdict-live: hard cap on fetches per profile (default 6)",
    )

    score_parser = subparsers.add_parser(
        "score", help="claims -> scores + insight sets (WS-F, idempotent)"
    )
    score_parser.add_argument(
        "--profile",
        default=None,
        help=("profile JSON as a path or inline object (runtime input only; never persisted)"),
    )
    score_parser.add_argument(
        "--dry-run", action="store_true", help="score and print a summary without writing"
    )
    score_parser.add_argument(
        "--limit", type=int, default=500, help="maximum programmes to score (default 500)"
    )
    score_parser.add_argument(
        "--force", action="store_true", help="write even when the latest score is identical"
    )

    llm_parser = subparsers.add_parser(
        "llm", help="inspect model configuration (never prints secrets)"
    )
    llm_sub = llm_parser.add_subparsers(dest="llm_command", required=True)
    doctor_parser = llm_sub.add_parser(
        "doctor",
        help="per-profile resolution table with exact env-var remediation",
    )
    doctor_parser.add_argument(
        "--live",
        action="store_true",
        help="also probe reachability with a cheap /models call (network; off by default)",
    )
    doctor_parser.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )
    llm_ping = llm_sub.add_parser("ping", help="alias for `doctor` without --live")
    llm_ping.add_argument("--json", action="store_true", help="machine-readable output")

    golden_parser = subparsers.add_parser(
        "golden",
        help="golden-set curation tool (WS-F): add rows from a source PDF, check, list",
    )
    golden_sub = golden_parser.add_subparsers(dest="golden_command", required=True)

    golden_add = golden_sub.add_parser(
        "add",
        help="curate one expected row: extract the quote from the source PDF, prove it, write it",
    )
    golden_add.add_argument(
        "--source",
        default=None,
        help="source PDF path or URL (the quote is extracted from it, never typed)",
    )
    golden_add.add_argument(
        "--page", type=int, default=None, help="1-based page of the PDF the quote is on"
    )
    golden_add.add_argument(
        "--predicate", default=None, help="expected field name (e.g. seats.total)"
    )
    golden_add.add_argument(
        "--value", default=None, help="expected normalised value (int/float/bool/text)"
    )
    golden_add.add_argument("--unit", default=None, help="normalised unit (count, INR, ratio, ...)")
    golden_add.add_argument(
        "--printed-value",
        default=None,
        help="the form the document prints (also used to locate candidate lines)",
    )
    golden_add.add_argument("--printed-unit", default=None, help="the unit the document prints")
    golden_add.add_argument("--institution", default=None, help="institution named in the source")
    golden_add.add_argument("--programme", default=None, help="programme the row is about")
    golden_add.add_argument("--state", default=None, help="state (optional)")
    golden_add.add_argument(
        "--entry-id", default=None, help="golden entry id (defaults to a slug of --institution)"
    )
    golden_add.add_argument(
        "--context",
        default=None,
        help="table-row anchor cell (institution/heading); must occur exactly once on the page",
    )
    golden_add.add_argument("--row-label", default=None, help="table-row claim: the row label cell")
    golden_add.add_argument(
        "--column-header", default=None, help="table-row claim: the column header cell"
    )
    golden_add.add_argument(
        "--value-cell", default=None, help="table-row claim: the printed value cell"
    )
    golden_add.add_argument(
        "--table-page", type=int, default=None, help="table-row claim: 1-based table page"
    )
    golden_add.add_argument(
        "--source-url", default=None, help="public URL for provenance when --source is a local file"
    )
    golden_add.add_argument("--notes", default=None, help="free-text curator notes")
    golden_add.add_argument(
        "--pick", type=int, default=None, help="1-based candidate line to quote (see add output)"
    )
    golden_add.add_argument(
        "--trim-left",
        type=int,
        default=0,
        help="drop N leading words from the extracted line (e.g. a serial number)",
    )
    golden_add.add_argument(
        "--trim-right", type=int, default=0, help="drop N trailing words from the extracted line"
    )
    golden_add.add_argument(
        "--interactive",
        action="store_true",
        help="prompt for missing inputs and the candidate pick (TTY)",
    )
    golden_add.add_argument(
        "--dry-run", action="store_true", help="run every proof and print the result; write nothing"
    )
    golden_add.add_argument(
        "--force",
        action="store_true",
        help="replace an existing predicate on a tool-generated entry (never promotes)",
    )
    golden_add.add_argument(
        "--quote",
        default=None,
        help=argparse.SUPPRESS,  # accepted-and-refused: a typed quote can never be verified
    )
    golden_add.add_argument(
        "--golden-dir", default=None, help="golden directory (default data/golden)"
    )
    golden_add.add_argument(
        "--root",
        default=None,
        help="repo root fixtures/entries are recorded against (default: cwd)",
    )

    golden_check = golden_sub.add_parser(
        "check",
        help="offline re-validation: fixtures, sha256, contexts, duplicates, per-family status",
    )
    golden_check.add_argument(
        "--golden", default=None, help="golden directory (default data/golden)"
    )
    golden_check.add_argument(
        "--root", default=None, help="repo root fixture paths resolve against (default: cwd)"
    )
    golden_check.add_argument("--json", action="store_true", help="machine-readable report")

    golden_list = golden_sub.add_parser("list", help="entries with status and what each row gates")
    golden_list.add_argument(
        "--golden", default=None, help="golden directory (default data/golden)"
    )
    golden_list.add_argument(
        "--root", default=None, help="repo root fixture paths resolve against (default: cwd)"
    )
    golden_list.add_argument("--json", action="store_true", help="machine-readable output")

    # Backwards-compatible top-level alias; prefer `fmd llm doctor`.
    ping = subparsers.add_parser(
        "llm-ping", help="alias for `fmd llm doctor` (never prints secrets)"
    )
    ping.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )

    # Scrape-failure log (append-only evidence): which institutions the scraper
    # lost out on, and why. Storage + taxonomy: docs/product-development-cycle/18.
    fails_parser = subparsers.add_parser(
        "failures",
        help="aggregate the append-only scrape-failure log by institution",
        description=(
            "Summarise <repo>/data/scrape-failures/failures-*.jsonl (or"
            " $FMD_SCRAPE_FAILURES_DIR): which institutions lost fetches and"
            " why (robots blocks, 404/410 gone pages, transport errors,"
            " all_tiers_blocked)."
        ),
    )
    fails_parser.add_argument(
        "--since",
        default=None,
        help="only failures at/after this ISO8601 UTC timestamp",
    )
    fails_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="maximum institutions listed (default: 10)",
    )
    fails_parser.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )

    programmes_parser = subparsers.add_parser(
        "programmes",
        help="materialise / list / inspect / requalify / decide scoped programme rows",
        description=(
            "Programme materialisation without a crawl: builds/refreshes"
            " `programmes` rows from claims already in the database"
            " (verified, unsuperseded, non-modelled, role=evidence), scoped"
            " in three tiers (config/programmes.yaml rules, semantic"
            " equivalence, manual decisions) so out-of-scope rows are kept"
            " for audit but never scored."
        ),
    )
    programmes_sub = programmes_parser.add_subparsers(dest="programmes_command", required=True)

    materialise = programmes_sub.add_parser(
        "materialise",
        help="build/refresh scoped programme rows from existing verified claims (no crawl)",
    )
    materialise.add_argument(
        "--force",
        action="store_true",
        help="rewrite rows even when nothing changed (default: only changed rows are written)",
    )
    materialise.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan (creates, updates, refused aggregates); write nothing",
    )
    materialise.add_argument(
        "--no-classifier",
        action="store_true",
        help=(
            "skip the Tier-2 semantic classifier (rules + manual decisions only;"
            " undecided names become needs_review). Default: classifier on, degrading"
            " to uncertain when the model profile is unavailable."
        ),
    )
    materialise.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )

    programmes_list = programmes_sub.add_parser(
        "list", help="programmes with seats + source claim, evidence completeness, review reasons"
    )
    programmes_list.add_argument("--limit", type=int, default=100, help="maximum rows to print")
    programmes_list.add_argument(
        "--institution-id", type=int, default=None, help="only this institution's programmes"
    )
    programmes_list.add_argument("--state", default=None, help="only this state")
    programmes_list.add_argument(
        "--all",
        action="store_true",
        help="include out-of-scope rows (default: in-scope only)",
    )
    programmes_list.add_argument(
        "--in-scope",
        dest="in_scope",
        action="store_true",
        default=True,
        help="only in-scope rows (the default)",
    )
    programmes_list.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )

    inspect = programmes_sub.add_parser(
        "inspect", help="every field of one programme with its claim id, quote and page"
    )
    inspect.add_argument("id", type=int, help="programme row id")
    inspect.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )

    requalify = programmes_sub.add_parser(
        "requalify",
        help=(
            "re-scope every existing row under the current taxonomy + decisions:"
            " fold row-numbering duplicates, mark MBBS/Agriculture out, apply"
            " manual verdicts"
        ),
    )
    requalify.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan (rescopes, merges, unchanged); write nothing",
    )
    requalify.add_argument(
        "--no-classifier",
        action="store_true",
        help="skip the Tier-2 semantic classifier (rules + manual decisions only)",
    )
    requalify.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )

    decide = programmes_sub.add_parser(
        "decide",
        help="record a human equivalence decision for one programme (the Tier-3 learning loop)",
    )
    decide.add_argument("id", type=int, help="programme row id")
    group = decide.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--equivalent",
        action="store_true",
        help="the programme IS Medical Laboratory Technology/Science (enters the pool)",
    )
    group.add_argument(
        "--not-equivalent",
        action="store_true",
        help="the programme is NOT the target field (stays out of every ranked set)",
    )
    decide.add_argument(
        "--note",
        default="",
        help="why (stored verbatim as the decision's evidence and shown on inspect)",
    )
    decide.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )
    return parser


def _database(args: argparse.Namespace) -> Database:
    return Database(args.database_url)


def resolve_snapshot_store() -> Any:
    """The default snapshot store (FMD_SNAPSHOT_DIR / configured path)."""
    from fmd.paths import resolve_snapshot_dir
    from fmd.store import SnapshotStore

    return SnapshotStore(resolve_snapshot_dir())


def _cmd_migrate(args: argparse.Namespace) -> int:
    with _database(args) as database:
        applied = run_migrations(database)
    if not applied:
        print("database is up to date; no migrations applied")
    else:
        print("applied migrations: " + ", ".join(applied))
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    registry = load_registry(args.config_dir)
    with _database(args) as database:
        counts = seed_database(database, registry)
    print("seeded registry: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


def _load_profile(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    candidate = Path(value)
    text = candidate.read_text(encoding="utf-8") if candidate.exists() else value
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("--profile must be a JSON object")
    return parsed


def _cmd_score(args: argparse.Namespace) -> int:
    from fmd.orchestrate.score import ScoreError

    registry = load_registry(args.config_dir)
    profile = _load_profile(args.profile)
    database = _database(args)
    try:
        from fmd.orchestrate import Orchestrator

        orchestrator = Orchestrator(registry=registry, database=database)
        summary = orchestrator.score(
            profile=profile,
            dry_run=args.dry_run,
            limit=args.limit,
            force=args.force,
        )
    except ScoreError as exc:
        print(f"fmd score: {exc}", file=sys.stderr)
        return 1
    finally:
        database.close()

    for entry in summary.outcomes:
        if "line" in entry:
            print(entry["line"])
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"fmd score: {verb} {summary.counts.get('written', 0)}, "
        f"skipped {summary.counts.get('skipped', 0)}, "
        f"unknown {summary.counts.get('unknown', 0)}, "
        f"programmes {summary.counts.get('programmes', 0)} "
        f"(scorer={registry.scorer_version}, weights={registry.weights_version})"
    )
    print(summary.as_json())
    return 1 if summary.failed else 0


def _orchestrator(args: argparse.Namespace):
    from fmd.orchestrate import Orchestrator

    registry = load_registry(args.config_dir)
    database = _database(args)
    return Orchestrator(registry=registry, database=database), database


def _cmd_crawl(args: argparse.Namespace) -> int:
    orchestrator, database = _orchestrator(args)
    try:
        summary = orchestrator.crawl(
            sources=args.source,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    finally:
        database.close()
    print(summary.as_json())
    return 1 if summary.failed else 0


def _cmd_extract(args: argparse.Namespace) -> int:
    if args.batched:
        return _cmd_extract_batched(args)
    orchestrator, database = _orchestrator(args)
    try:
        summary = orchestrator.extract(
            limit=args.limit,
            force=args.force,
            from_cache=args.from_cache,
            no_acquire=args.no_acquire,
            dry_run=args.dry_run,
        )
    finally:
        database.close()
    print(summary.as_json())
    return 1 if summary.failed else 0


def _batch_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Batched-run settings: acquire.yaml ``batch`` block with CLI overrides."""
    registry = load_registry(args.config_dir)
    batch = getattr(registry.acquire, "batch", None)
    defaults = {
        "pages": 100,
        "device": "cuda",
        "server_script": None,
        "keep_server": False,
        "poll_timeout_s": 600,
        "prefetch_documents": 4,
    }
    if batch is not None:
        defaults.update(
            {
                "pages": int(batch.pages),
                "device": str(batch.device),
                "server_script": batch.server_script,
                "keep_server": bool(batch.keep_server),
                "poll_timeout_s": int(batch.poll_timeout_s),
                "prefetch_documents": int(batch.prefetch_documents),
            }
        )
    if getattr(args, "batch_pages", None) is not None:
        defaults["pages"] = int(args.batch_pages)
    if getattr(args, "device", None):
        defaults["device"] = str(args.device)
    if getattr(args, "keep_server", False):
        defaults["keep_server"] = True
    if getattr(args, "server_script", None):
        defaults["server_script"] = str(args.server_script)
    return defaults


def _cmd_extract_batched(args: argparse.Namespace) -> int:
    from fmd.orchestrate.batched import BatchedRunner

    settings = _batch_settings(args)
    orchestrator, database = _orchestrator(args)
    try:
        runner = BatchedRunner(
            orchestrator=orchestrator,
            batch_pages=settings["pages"],
            device=settings["device"],
            keep_server=settings["keep_server"],
            poll_timeout_s=settings["poll_timeout_s"],
            prefetch_depth=settings["prefetch_documents"],
            server_script=settings["server_script"],
        )
        summary = runner.run(limit=args.limit, dry_run=args.dry_run)
    finally:
        database.close()
    print(summary.as_json())
    return 1 if summary.failed else 0


def _cmd_acquire(args: argparse.Namespace) -> int:
    orchestrator, database = _orchestrator(args)
    try:
        summary = orchestrator.acquire(
            limit=args.limit,
            pages=args.pages,
            device=args.device,
            dry_run=args.dry_run,
        )
    finally:
        database.close()
    print(summary.as_json())
    return 1 if summary.failed else 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from fmd.orchestrate.plan import run_planning_pass

    if getattr(args, "admit_evidence", None) is not None:
        return _cmd_plan_admit_evidence(args)
    registry = load_registry(args.config_dir)
    database = _database(args)
    try:
        summary = run_planning_pass(
            registry,
            database,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
        )
    finally:
        database.close()
    print(summary.as_json())
    return 1 if summary.failed else 0


def _cmd_classify_report(args: argparse.Namespace) -> int:
    """Wave v1-w5: replay the deterministic family chain per evidence document.

    Read-only. Buckets explain every missing_family row; below_floor /
    ambiguous / no_signal rows are the semantic-tier queue (LOCAL-QUEUE LQ-7)
    — this command never admits anything.
    """
    from fmd.db.repositories import DocumentsRepo, SnapshotsRepo, SourcesRepo
    from fmd.orchestrate.plan import _cheap_text
    from fmd.plan.planner import DocumentPlanner, declared_family
    from fmd.plan.report import build_report, render_report

    registry = load_registry(args.config_dir)
    database = _database(args)
    try:
        planner = DocumentPlanner(plan_config=getattr(registry, "plan", None))
        documents = DocumentsRepo(database)
        sources_repo = SourcesRepo(database)
        # The object store is only needed when the text chain is replayed:
        # without --with-text the report never touches snapshot bytes (and no
        # snapshot directory has to exist on the host).
        store = resolve_snapshot_store() if args.with_text else None
        snapshots = SnapshotsRepo(database) if store is not None else None
        rows = documents.plan_evidence_rows(limit=args.limit)
        sources_by_id = {str(getattr(source, "id", "")): source for source in sources_repo.list()}
        enriched: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            source = sources_by_id.get(str(row.get("source_id") or ""))
            record["declared_family"] = declared_family(source)
            enriched.append(record)
        sample_cache: dict[str, str | None] = {}

        def sample_for(row: dict[str, Any]) -> str | None:
            key = str(row["id"])
            if key in sample_cache:
                return sample_cache[key]
            if store is None or snapshots is None:
                sample_cache[key] = None
                return None
            snapshot = snapshots.latest_for_document(int(key))
            if snapshot is None:
                sample_cache[key] = None
                return None
            try:
                doc = _cheap_text(
                    store.get(snapshot.sha256),
                    content_type=getattr(snapshot, "content_type", None),
                    policy=None,
                )
            except Exception:
                sample_cache[key] = None
                return None
            sample = str(getattr(doc, "text", "") or "")[: planner.thresholds.text_sample_chars]
            sample_cache[key] = sample
            return sample

        thresholds = planner.thresholds
        report = build_report(
            enriched,
            rules=planner.rules,
            phase_b_min_score=thresholds.phase_b_min_score,
            phase_b_margin=thresholds.phase_b_margin,
            phase_a_min_score=thresholds.phase_a_min_score,
            phase_a_margin=thresholds.phase_a_margin,
            recurring_min=getattr(thresholds, "recurring_min", None),
            table_min_rows=getattr(thresholds, "table_min_rows", None),
            sample_for=sample_for if args.with_text else None,
        )
    finally:
        database.close()
    print(render_report(report))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written: {args.json_out}")
    return 0


def _cmd_plan_admit_evidence(args: argparse.Namespace) -> int:
    """Apply a semantic-admissions file through the quoted-evidence contract."""
    from collections import Counter

    from fmd.orchestrate.plan import _cheap_text, build_planning_pass
    from fmd.plan.semantic import SemanticAdmission, apply_admissions, validate_admissions

    registry = load_registry(args.config_dir)
    try:
        payload = json.loads(Path(args.admit_evidence).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"fmd plan: cannot read admissions file: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, list):
        print("fmd plan: admissions file must be a JSON list", file=sys.stderr)
        return 1
    admissions = [
        SemanticAdmission(
            document_url=str(item.get("document_url") or ""),
            family=str(item.get("family") or ""),
            quote=str(item.get("quote") or ""),
            score=(float(item["score"]) if item.get("score") is not None else None),
        )
        for item in payload
        if isinstance(item, dict)
    ]
    database = _database(args)
    try:
        stage, _ledger = build_planning_pass(registry, database)
        documents = stage.documents
        sample_cache: dict[str, str] = {}

        def sample_for(url: str) -> str | None:
            if url in sample_cache:
                return sample_cache[url]
            document = documents.get_by_url(url)
            if document is None:
                return None
            snapshot = stage.snapshots.latest_for_document(int(document.id))
            if snapshot is None:
                return None
            try:
                doc = _cheap_text(
                    stage.store.get(snapshot.sha256),
                    content_type=getattr(snapshot, "content_type", None),
                    policy=stage.policy,
                )
            except Exception:
                return None
            sample = str(getattr(doc, "text", "") or "")[: stage.text_sample_chars]
            sample_cache[url] = sample
            return sample

        samples = {}
        for url in {admission.document_url for admission in admissions}:
            sample = sample_for(url)
            if sample is not None:
                samples[url] = sample
        validated, refused = validate_admissions(admissions, samples)
        outcome = apply_admissions(validated, documents=documents, sample_for=sample_for)
    finally:
        database.close()
    print(
        json.dumps(
            {
                "proposed": len(admissions),
                "admitted": outcome["applied"],
                "missing_document": outcome["missing_document"],
                "refused": len(refused),
                "refusal_reasons": dict(Counter(refusal.reason for refusal in refused)),
                "refused_detail": [
                    {
                        "document_url": refusal.document_url,
                        "family": refusal.family,
                        "reason": refusal.reason,
                    }
                    for refusal in refused[:100]
                ],
                # Audit trail for the admitted evidence (LQ-7 output): the
                # quote digest is what makes the admission reproducible.
                "admitted_detail": [
                    {
                        "document_url": admission.document_url,
                        "family": admission.family,
                        "quote_sha256": admission.quote_sha256,
                    }
                    for admission in validated[:200]
                ],
            },
            indent=2,
        )
    )
    return 0


def _cmd_coverage_report(args: argparse.Namespace, database: Any = None) -> int:
    """Wave v1-w7: why is a programme 1/7? Read-only scoring diagnostics.

    Runs the exact scoring computation per in-scope programme (same
    evaluate_all → score_programme path the score command uses), reports which
    sub-scores are populated vs unknown and which claims feed them — and
    persists nothing. No weight changes, no model calls.
    """
    from fmd.db.repositories import ClaimsRepo, ProgrammeFilter, ProgrammesRepo
    from fmd.filters import evaluate_all
    from fmd.score import score_programme
    from fmd.score.config import ScoringConfig
    from fmd.score.coverage import build_coverage_report, programme_coverage, render_coverage

    registry = load_registry(args.config_dir)
    profile = _load_profile(getattr(args, "profile", None))
    config = ScoringConfig.from_config(registry.weights, registry.predicates)
    problems = config.validate()
    if problems:
        print(
            "fmd coverage-report: invalid scoring config: " + "; ".join(problems), file=sys.stderr
        )
        return 1
    # ``database`` is a testability seam: the CLI opens its own connection,
    # tests pass the fixture's schema-scoped one. Closed only when opened here.
    owned = database is None
    database = database or _database(args)
    try:
        programmes = ProgrammesRepo(database)
        claims = ClaimsRepo(database)
        rows = programmes.list(ProgrammeFilter(limit=max(1, args.limit), offset=0))
        scoped = [row for row in rows if getattr(row, "in_scope", True)]
        report_rows: list[dict[str, Any]] = []
        skipped_errors = 0
        for programme in scoped:
            try:
                entity_claims = claims.list_for_entity(programme.institution_id)
                view = programme.model_dump(mode="json")
                view["claims"] = entity_claims
                summary = evaluate_all(view, profile)
                result = score_programme(
                    programme.id,
                    view,
                    profile,
                    entity_claims,
                    config=config,
                    cost_rules=registry.cost_rules,
                    filter_decisions=list(summary.decisions.values()),
                )
                report_rows.append(
                    programme_coverage(
                        programme.id,
                        programme.institution_id,
                        str(getattr(programme, "name", "") or ""),
                        result.sub_scores,
                    )
                )
            except Exception:  # one bad programme must not abort the report
                skipped_errors += 1
                continue
    finally:
        if owned:
            database.close()
    report = build_coverage_report(report_rows)
    report["skipped_errors"] = skipped_errors
    print(render_coverage(report))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written: {args.json_out}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from fmd.db.repositories import ClaimsRepo, ExtractionsRepo
    from fmd.extract.pipeline import build_router
    from fmd.orchestrate.ledger import RunSummary, utcnow
    from fmd.verify import Verifier, VerifyPolicy
    from fmd.verify.prompts import PromptNotFound
    from fmd.verify.runner import record_summary, verify_pending
    from fmd.verify.verifier import VerifierUnavailable

    registry = load_registry(args.config_dir)
    policy = VerifyPolicy.load(registry.models)
    if policy is None:
        print(
            "fmd verify: no `verify` profile in config/models.yaml; nothing was judged. "
            "Claims stay unverified.",
            file=sys.stderr,
        )
        return 1
    summary = RunSummary(run_id=uuid.uuid4().hex, stage="verify", started_at=utcnow())
    database = _database(args)
    try:
        claims = ClaimsRepo(database)
        router = build_router(registry.models, mock=args.mock, fixtures_dir=args.fixtures)
        verifier = Verifier(
            router=router, claims=claims, policy=policy, predicates=registry.predicates
        )
        run = verify_pending(
            claims=claims,
            extractions=ExtractionsRepo(database),
            verifier=verifier,
            limit=args.limit,
        )
        record_summary(summary, run)
    except (VerifierUnavailable, PromptNotFound) as exc:
        print(f"fmd verify: {exc}", file=sys.stderr)
        return 1
    finally:
        database.close()
    summary.finish()
    print(summary.as_json())
    return 1 if summary.failed else 0


def _cmd_worker(args: argparse.Namespace) -> int:
    orchestrator, database = _orchestrator(args)
    summaries: list[Any] = []
    try:
        summaries = orchestrator.worker(
            limit=args.limit,
            idle_seconds=args.idle_seconds,
            max_ticks=args.max_ticks,
            on_summary=lambda summary: print(summary.as_json(), flush=True),
        )
    finally:
        database.close()
    return 1 if any(summary.failed for summary in summaries) else 0


def _cmd_failures(args: argparse.Namespace) -> int:
    """`fmd failures`: aggregate the scrape-failure log BY INSTITUTION.

    The institution is the thing we lose — the table shows which good
    universities the scraper lost out on and why (docs §scrape-failures).
    """
    from fmd.fetch.failure_log import aggregate, read_failures

    failures = read_failures(since=args.since)
    agg = aggregate(failures, top_n=args.top)
    if args.json:
        print(json.dumps(agg, indent=2))
        return 0

    rows: list[tuple[str, dict]] = [
        (name, entry) for name, entry in agg["by_institution"].items()
    ]
    unresolved = agg["unresolved"]
    if unresolved["n"] > 0:
        rows.append(("(unresolved)", unresolved))
    rows.sort(key=lambda row: (-row[1]["n"], row[0]))

    print(f"scrape failures: total={agg['total']} since={args.since or 'all'}")
    for name, entry in rows:
        kinds = ", ".join(f"{kind}:{count}" for kind, count in entry["kinds"].items())
        sample = ", ".join(entry.get("sample_urls", []))
        suffix = f"   sample: {sample}" if sample else ""
        print(f"{name}   {entry['n']}   kinds: {kinds}{suffix}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    eval_kind = getattr(args, "eval_kind", "pipeline")
    if eval_kind == "product":
        return _cmd_eval_product(args)
    if eval_kind == "verdict":
        return _cmd_eval_verdict(args)
    return _cmd_eval_pipeline(args)


def _cmd_eval_pipeline(args: argparse.Namespace) -> int:
    from fmd.eval import (
        EvalError,
        GoldenError,
        apply_gate,
        render_summary,
        run_evaluation,
        write_report,
    )

    try:
        outcome = run_evaluation(args.golden, baseline_path=args.baseline)
    except (EvalError, GoldenError) as exc:
        print(f"fmd eval: {exc}", file=sys.stderr)
        return 1
    report = outcome.report
    verification = None
    if getattr(args, "verify_fixtures", None):
        from fmd.verify.casefile import CaseError, evaluate_fixture_dir

        try:
            verification = evaluate_fixture_dir(
                args.verify_fixtures, load_registry(args.config_dir)
            )
        except (CaseError, ConfigError) as exc:
            print(f"fmd eval: verification fixture failed: {exc}", file=sys.stderr)
            return 1
        report["verification"] = verification
    failures = apply_gate(report, args.fail_under)
    report["ok"] = not failures
    report_path = None
    if args.report_out:
        report_path = write_report(report, args.report_out)
    print(render_summary(report, report_path=report_path))
    if verification is not None:
        print(
            "  verification: verified_rate {rate} fabrication_caught "
            "{caught}/{total} correct_rejected {correct}".format(
                rate=verification.get("verified_rate"),
                caught=verification.get("fabrication_caught", 0),
                total=verification.get("fabricated_total", 0),
                correct=verification.get("correct_rejected", 0),
            )
        )
    for failure in failures:
        print(f"fmd eval: FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


def _eval_report_target(args: argparse.Namespace, stream: str) -> Path | None:
    """Report path: explicit --report-out, else reports/eval-<stream>-<ts>.json."""

    if getattr(args, "dry_run", False):
        return None
    if args.report_out:
        return Path(args.report_out)
    reports_dir = Path(args.report_out) if args.report_out else None
    from fmd.eval.product import default_report_path

    repo_root = Path(__file__).resolve().parents[3]
    return default_report_path(reports_dir or repo_root / "reports", stream)


def _cmd_eval_product(args: argparse.Namespace) -> int:
    """Product streams: proxy ranking, prompt-effectiveness, self-drift, cv-stability."""

    from fmd.eval.product import (
        ShortlistError,
        build_product_report,
        load_shortlist,
        load_truth_for_stream,
        prompt_effectiveness_aggregate,
        render_product_summary,
        run_cv_stability,
        run_naac_proxy,
        run_nirf_proxy,
        run_prompt_effectiveness,
        run_self_drift,
        truth_summary,
    )
    from fmd.eval.product import write_report as write_product_report
    from fmd.eval.truth import TruthError

    stream = args.stream
    if stream is None:
        print("fmd eval: --stream is required for --eval-kind product", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parents[3]

    def _fixture_default(name: str) -> str:
        """Committed offline canary input; --fixture-path overrides it."""

        base = repo_root / "packages" / "pipeline" / "tests" / "fixtures" / "eval_product"
        return str(Path(args.fixture_path) if args.fixture_path else base / name)

    try:
        if stream in {"nirf-proxy", "naac-proxy"}:
            shortlist = load_shortlist(_fixture_default("shortlist_fixture.json"))
            truth_dir = Path(args.truth_dir) if args.truth_dir else repo_root / "data" / "truth"
            truth = load_truth_for_stream(stream, truth_dir, root=repo_root)
            if stream == "nirf-proxy":
                case = run_nirf_proxy(shortlist, truth)
            else:
                case = run_naac_proxy(shortlist, truth)
            report = build_product_report(stream, [case], truth_header=truth_summary(truth))
            report["proxy_label"] = case["proxy_label"]
        elif stream == "self-drift":
            if args.fixture_path and not args.previous_snapshot:
                print(
                    "fmd eval: self-drift needs --previous-snapshot when --fixture-path "
                    "names the current snapshot",
                    file=sys.stderr,
                )
                return 2
            fixture_dir = (
                Path(args.fixture_path)
                if args.fixture_path and Path(args.fixture_path).is_dir()
                else repo_root / "packages" / "pipeline" / "tests" / "fixtures" / "eval_product"
            )
            current_path = (
                Path(args.fixture_path)
                if args.fixture_path and Path(args.fixture_path).is_file()
                else fixture_dir / "snapshot_current.json"
            )
            previous_path = (
                Path(args.previous_snapshot)
                if args.previous_snapshot
                else fixture_dir / "snapshot_previous.json"
            )
            current = load_shortlist(current_path)
            previous = load_shortlist(previous_path)
            case = run_self_drift(current, previous)
            report = build_product_report(stream, [case])
            report["proxy_label"] = None
        elif stream == "cv-stability":
            shortlist = load_shortlist(_fixture_default("cv_candidates.json"))
            weights = _load_weights_for_cv(args.config_dir)
            case = run_cv_stability(
                shortlist,
                weights,
                k=args.top_k,
                epsilon=args.epsilon,
                holdout_n=args.holdout_n,
            )
            report = build_product_report(
                stream,
                [case],
                extra={"weights_version": shortlist.weights_version},
            )
            report["proxy_label"] = None
        elif stream == "prompt-effectiveness":
            shortlist = load_shortlist(_fixture_default("prompt_effectiveness_fixture.json"))
            truth_dir = Path(args.truth_dir) if args.truth_dir else repo_root / "data" / "truth"
            nirf_truth = load_truth_for_stream("nirf-proxy", truth_dir, root=repo_root)
            naac_truth = load_truth_for_stream("naac-proxy", truth_dir, root=repo_root)
            cases = [run_prompt_effectiveness(shortlist, nirf_truth, naac_truth)]
            report = build_product_report("prompt-effectiveness", cases)
            report["aggregate"].update(prompt_effectiveness_aggregate(cases))
            report["proxy_label"] = None
        elif stream == "expert-panel":
            return _cmd_eval_expert_panel(args)
        else:
            print(f"fmd eval: unknown product stream {stream!r}", file=sys.stderr)
            return 2
    except (ShortlistError, TruthError) as exc:
        print(f"fmd eval: {exc}", file=sys.stderr)
        return 1

    report_path = _eval_report_target(args, stream)
    if report_path is not None:
        write_product_report(report, report_path)
    print(render_product_summary(report))
    if report_path is not None:
        print(f"  report: {report_path}")
    return 0


def _load_weights_for_cv(config_dir: str | None) -> dict[str, float]:
    """Load the configured sub-score weights (registry-driven, invariant 7)."""

    from fmd.eval.product import ShortlistError
    from fmd.paths import PathResolutionError, resolve_config_dir

    try:
        config_path = Path(config_dir) if config_dir else resolve_config_dir()
    except PathResolutionError as exc:
        raise ShortlistError(f"cannot locate config dir for weights.yaml: {exc}") from exc
    import yaml

    data = yaml.safe_load((config_path / "weights.yaml").read_text(encoding="utf-8")) or {}
    weights = data.get("weights") or {}
    if not weights:
        raise ShortlistError("config/weights.yaml has no weights block; cv-stability cannot run")
    return {str(name): float(value) for name, value in weights.items()}


def _cmd_eval_expert_panel(args: argparse.Namespace) -> int:
    """Expert-panel stream (slow stream, fixture-driven; see docs/v1/eval docs)."""

    from fmd.eval.panel import (
        PanelError,
        load_panel_ratings,
        load_product_ranking,
        render_panel_summary,
        run_expert_panel,
    )
    from fmd.eval.product import build_product_report
    from fmd.eval.product import write_report as write_product_report

    repo_root = Path(__file__).resolve().parents[3]
    fixture_dir = (
        Path(args.fixture_path)
        if args.fixture_path
        else repo_root / "packages" / "pipeline" / "tests" / "fixtures" / "eval_panel"
    )
    try:
        ratings = load_panel_ratings(fixture_dir)
        ranking = load_product_ranking(fixture_dir)
        case = run_expert_panel(ratings, ranking)
        report = build_product_report("expert-panel", [case])
        report["proxy_label"] = None
    except PanelError as exc:
        print(f"fmd eval: {exc}", file=sys.stderr)
        return 1
    report_path = _eval_report_target(args, "expert-panel")
    if report_path is not None:
        write_product_report(report, report_path)
    print(render_panel_summary(case))
    if report_path is not None:
        print(f"  report: {report_path}")
    return 0


def _cmd_eval_verdict(args: argparse.Namespace) -> int:
    """Verdict streams: the verdict-fast canary and the verdict-live backend."""

    audit_score = getattr(args, "audit_score", None)
    if audit_score:
        return _cmd_eval_audit_score(args)
    stream = args.stream
    if stream == "verdict-live":
        return _cmd_eval_verdict_live(args)

    from fmd.eval.product import build_product_report
    from fmd.eval.product import write_report as write_product_report
    from fmd.eval.verdict import (
        VerdictError,
        load_profile_fixtures,
        render_verdict_summary,
        run_verdict_fast,
        verdict_aggregate,
    )

    if stream != "verdict-fast":
        print(
            "fmd eval: --eval-kind verdict supports --stream verdict-fast|verdict-live",
            file=sys.stderr,
        )
        return 2
    repo_root = Path(__file__).resolve().parents[3]
    fixture_dir = Path(args.fixture_path) if args.fixture_path else (
        repo_root / "packages" / "pipeline" / "tests" / "fixtures" / "eval_verdict" / "profiles"
    )
    try:
        profiles = load_profile_fixtures(fixture_dir)
        cases = run_verdict_fast(profiles)
    except VerdictError as exc:
        print(f"fmd eval: {exc}", file=sys.stderr)
        return 1
    report = build_product_report("verdict-fast", cases)
    report["proxy_label"] = None
    report["aggregate"].update(verdict_aggregate(cases))
    report_path = _eval_report_target(args, "verdict-fast")
    if report_path is not None:
        write_product_report(report, report_path)
    print(render_verdict_summary(report))
    if report_path is not None:
        print(f"  report: {report_path}")
    return 0


def _cmd_eval_audit_score(args: argparse.Namespace) -> int:
    """--audit-score: harness-computed agreement rate vs the >=70% gate (doc 04).

    The gate is REPORTED, never enforced in CI (no operator exists in CI).
    Exit codes: 0 PASS, 1 FAIL, 3 INSUFFICIENT.
    """

    from pydantic import ValidationError

    from fmd.eval.verdict_live import AuditWorksheet, score_audit_worksheet

    path = Path(args.audit_score)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"fmd eval: cannot read audit worksheet {path}: {exc}", file=sys.stderr)
        return 1
    try:
        worksheet = AuditWorksheet.model_validate(payload)
    except ValidationError as exc:
        print(f"fmd eval: audit worksheet {path} is not schema-valid: {exc}", file=sys.stderr)
        return 1
    outcome = score_audit_worksheet(worksheet)
    print(f"fmd eval (audit-score): worksheet {path}")
    print(
        "  sampled {n} verdict(s), completed {m} operator audit(s), "
        "agreement_rate {rate}, gate {gate}".format(
            n=outcome["n_sampled"],
            m=outcome["n_completed"],
            rate=outcome["agreement_rate"],
            gate=outcome["gate"],
        )
    )
    for disagreement in outcome["disagreements"]:
        print(
            "  disagreement: {profile_id} agent={agent_verdict} — {reason}".format(
                **disagreement
            )
        )
    if outcome["fabricated_citations"]:
        print(
            "  fabricated citation finding(s): "
            + ", ".join(outcome["fabricated_citations"])
            + " — whole run quarantined (doc 04 section 2.3)"
        )
    print(f"  result: {outcome['result']} — {outcome['reason']}")
    print("  note: the gate is reported, never enforced in CI (no operator in CI)")
    if outcome["result"] == "PASS":
        return 0
    return 3 if outcome["result"] == "INSUFFICIENT" else 1


def _cmd_eval_verdict_live(args: argparse.Namespace) -> int:
    """verdict-live: the bounded live-research backend (replay in CI, live on host).

    Router resolution: the ``verdict_live`` profile in config/models.yaml. In
    tests/CI (FMD_VERDICT_LIVE_PROVIDER=mock, a replay provider) the stream
    replays committed scripted sessions and never touches the network; on the
    operator's host it resolves to the configured local profile and fetches go
    through the fetch ladder (robots.txt, rate limits, truthful UA).
    """

    from datetime import UTC, datetime

    from fmd.eval.product import build_product_report
    from fmd.eval.product import write_report as write_product_report
    from fmd.eval.verdict import VerdictError, load_profile_fixtures
    from fmd.eval.verdict_live import (
        VERDICT_LIVE_PURPOSE,
        LiveVerdictBackend,
        ScriptedModelRouter,
        ScriptedPageFetcher,
        build_audit_worksheet,
        clamp_audit_fraction,
        load_live_sessions,
        render_verdict_live_summary,
        run_verdict_live,
        verdict_live_aggregate,
    )
    from fmd.llm.doctor import profile_blockers, profile_remediation
    from fmd.llm.providers import LLMError
    from fmd.llm.router import ModelRouter

    repo_root = Path(__file__).resolve().parents[3]
    fixture_dir = Path(args.fixture_path) if args.fixture_path else (
        repo_root / "packages" / "pipeline" / "tests" / "fixtures" / "eval_verdict" / "profiles"
    )
    sessions_dir = fixture_dir.parent / "live" / "sessions"
    try:
        profiles = load_profile_fixtures(fixture_dir)
    except VerdictError as exc:
        print(f"fmd eval: {exc}", file=sys.stderr)
        return 1

    router = ModelRouter()
    profile_cfg = router.models.profiles.get(VERDICT_LIVE_PURPOSE)
    if profile_cfg is None:
        print(
            f"fmd eval: config/models.yaml has no {VERDICT_LIVE_PURPOSE!r} profile; "
            "the verdict-live stream cannot resolve its model",
            file=sys.stderr,
        )
        return 1
    provider_name = profile_cfg.provider
    provider_cfg = router.models.providers.get(provider_name)
    if provider_cfg is None:
        print(
            f"fmd eval: provider {provider_name!r} for {VERDICT_LIVE_PURPOSE!r} is not "
            "declared in config/models.yaml",
            file=sys.stderr,
        )
        return 1
    replay_mode = provider_cfg.kind == "replay"

    try:
        if replay_mode:
            sessions = load_live_sessions(sessions_dir)
            missing = [p.profile_id for p in profiles if p.profile_id not in sessions]
            if missing:
                print(
                    "fmd eval: no scripted live session for profile(s): "
                    + ", ".join(missing)
                    + f" (expected under {sessions_dir})",
                    file=sys.stderr,
                )
                return 1
            fetcher = ScriptedPageFetcher(
                [page for session in sessions.values() for page in session.pages]
            )
            backends = [
                LiveVerdictBackend(
                    ScriptedModelRouter(sessions[profile.profile_id]),
                    fetcher,
                    max_turns=args.max_turns,
                    max_fetches=args.max_fetches,
                )
                for profile in profiles
            ]
            mode = "replay"
        else:
            _providers, blockers = profile_blockers(router.models, profile_cfg)
            if blockers:
                print(
                    "fmd eval: the verdict_live model profile does not resolve on this "
                    "host: " + "; ".join(blockers) + ". " + profile_remediation(
                        router.models, VERDICT_LIVE_PURPOSE
                    ),
                    file=sys.stderr,
                )
                return 1
            from fmd.fetch.ladder import FetchLadder

            backend = LiveVerdictBackend(
                router,
                FetchLadder(),
                max_turns=args.max_turns,
                max_fetches=args.max_fetches,
            )
            backends = [backend] * len(profiles)
            mode = "live"

        cases = []
        for profile, backend in zip(profiles, backends, strict=True):
            cases.extend(run_verdict_live([profile], backend))
    except (VerdictError, LLMError, RuntimeError) as exc:
        print(f"fmd eval: verdict-live failed: {exc}", file=sys.stderr)
        return 1

    report = build_product_report("verdict-live", cases)
    report["proxy_label"] = None
    report["aggregate"].update(verdict_live_aggregate(cases))
    report["backend"] = {
        "mode": mode,
        "provider": provider_name,
        "purpose": VERDICT_LIVE_PURPOSE,
        "max_turns": args.max_turns,
        "max_fetches": args.max_fetches,
        "sessions_dir": str(sessions_dir) if replay_mode else None,
    }
    if replay_mode:
        report["backend"]["note"] = (
            "replay path (committed scripted sessions, offline); the live stream has "
            "NEVER run against a real model yet — the first real run is the "
            "operator's (LQ-9)"
        )
    report_path = _eval_report_target(args, "verdict-live")
    if report_path is not None:
        write_product_report(report, report_path)

    worksheet_preview = None
    if getattr(args, "audit_sample", None) is not None:
        worksheet = build_audit_worksheet(
            cases,
            fraction=args.audit_sample,
            seed=args.audit_seed,
            generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if report_path is not None and not getattr(args, "dry_run", False):
            worksheet_path = report_path.with_name(
                report_path.name.replace("eval-verdict-live", "audit-verdict-live", 1)
            )
            worksheet_path.write_text(
                json.dumps(worksheet.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            worksheet_preview = f"  audit worksheet: {worksheet_path}"
        else:
            worksheet_preview = (
                "  audit worksheet (dry-run preview): seed {seed} fraction {frac} "
                "sampled {n}/{total}: {ids}".format(
                    seed=worksheet.seed,
                    frac=clamp_audit_fraction(args.audit_sample),
                    n=len(worksheet.entries),
                    total=worksheet.n_cases,
                    ids=", ".join(e.profile_id for e in worksheet.entries) or "(none)",
                )
            )

    print(render_verdict_live_summary(report))
    if report_path is not None:
        print(f"  report: {report_path}")
    if worksheet_preview is not None:
        print(worksheet_preview)
    return 0


def _cmd_llm_doctor(args: argparse.Namespace) -> int:
    from fmd.acquire.ocr_doctor import diagnose_ocr
    from fmd.llm.doctor import doctor_report

    registry = load_registry(args.config_dir)
    live = bool(getattr(args, "live", False))
    ocr = diagnose_ocr(registry.acquire.backends["T2"])
    report = doctor_report(registry.models, live=live, ocr=ocr)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "profiles": report["profiles"],
                    "budgets": registry.models.budgets.model_dump(),
                    "required_ok": report["required_ok"],
                    "ok": report["ok"],
                    "live": report["live"],
                    "remediation": report["remediation"],
                    "ocr": report["ocr"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["required_ok"] else 1

    print(f"model profiles in {Path(registry.config_dir) / 'models.yaml'}:")
    header = (
        f"{'PROFILE':<20} {'PROVIDER':<10} {'KIND':<18} {'BASE_URL':<34} "
        f"{'KEY':<12} {'MODEL':<26} {'GUIDED':<13} {'REQ':<4} {'REACHABLE'}"
    )
    print(header)
    for row in report["profiles"]:
        if row["base_url_resolved"]:
            base = row["base_url"]
        else:
            base = f"MISSING ({row['base_url_env'] or 'base_url'})"
        if row["key_present"]:
            key = "present"
        elif row["key_required"]:
            key = "MISSING"
        else:
            key = "not-required"
        model = row["model"] or f"MISSING ({row['model_env'] or 'model'})"
        guided = "yes" if row["guided_json_expected"] else "json-instruct"
        if row["reachable"] is None:
            reachable = "not-checked"
        elif row["reachable"]:
            reachable = "ok"
        else:
            reachable = f"FAIL: {row['reachability_note']}"
        print(
            f"{row['profile']:<20} {row['provider']:<10} {row['kind']:<18} "
            f"{base:<34} {key:<12} {model:<26} {guided:<13} "
            f"{'yes' if row['required'] else 'no':<4} {reachable}"
        )
        if not row["usable"] and row["degrade_reason"]:
            print(
                f"    degrades to {row['degrade_reason']} per document; text extraction still runs."
            )
    print(report["ocr"]["line"])
    if report["remediation"]:
        print("remediation:")
        for line in report["remediation"]:
            print(f"  {line}")
    else:
        print("all required profiles resolved.")
    budgets = registry.models.budgets
    print(
        f"budgets: daily=${budgets.daily_usd:.2f} per_run=${budgets.per_run_usd:.2f} "
        f"hard_stop={budgets.hard_stop} cache={budgets.cache}"
    )
    if not live:
        print("reachability not checked; re-run with --live for a cheap /models probe.")
    print("secrets are never printed; only presence is reported.")
    return 0 if report["required_ok"] else 1


def _cmd_llm_ping(args: argparse.Namespace) -> int:
    return _cmd_llm_doctor(args)


def _default_golden_dir() -> Path:
    from fmd.eval.golden import DEFAULT_GOLDEN_DIR

    return DEFAULT_GOLDEN_DIR


def _golden_root(args: argparse.Namespace) -> Path:
    if getattr(args, "root", None):
        return Path(args.root)
    from fmd.eval.golden import REPO_ROOT

    return REPO_ROOT


def _prompt(text: str, default: str | None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{text}{suffix}: ").strip()
    except EOFError as exc:
        raise SystemExit(2) from exc
    return answer or (default or "")


def _cmd_golden_add(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from fmd.eval import curator
    from fmd.eval.curator import AddInputs, CuratorError
    from fmd.eval.pdfpage import extract_page

    if args.quote is not None:
        print(
            "fmd golden add: a typed quote is rejected; quotes are extracted from the source "
            "PDF itself so the span can be proven. Pick a candidate line instead "
            "(--pick N, optional --trim-left/--trim-right).",
            file=sys.stderr,
        )
        return 2

    interactive = args.interactive or sys.stdin.isatty()
    source = args.source or (_prompt("source PDF path or URL", None) if interactive else None)
    page = args.page
    if page is None and interactive:
        page = int(_prompt("page (1-based)", "1"))
    predicate = args.predicate or (_prompt("predicate", None) if interactive else None)
    value = args.value or (_prompt("expected value", None) if interactive else None)
    if args.institution:
        institution = args.institution
    elif interactive:
        institution = _prompt("institution", None)
    else:
        institution = None
    if args.programme:
        programme = args.programme
    elif interactive:
        programme = _prompt("programme", "BSc Medical Laboratory Technology")
    else:
        programme = None
    if not args.unit and interactive:
        args.unit = _prompt("unit (empty for none)", "") or None

    inputs = AddInputs(
        source=source or "",
        page=page or 0,
        predicate=predicate or "",
        value_raw=value or "",
        unit=curator.parse_optional(args.unit),
        printed_value=curator.parse_optional(args.printed_value),
        printed_unit=curator.parse_optional(args.printed_unit),
        institution=institution or "",
        programme=programme or "",
        state=curator.parse_optional(args.state),
        entry_id=args.entry_id,
        context=curator.parse_optional(args.context),
        row_label=curator.parse_optional(args.row_label),
        column_header=curator.parse_optional(args.column_header),
        value_cell=curator.parse_optional(args.value_cell),
        table_page=args.table_page,
        source_url=curator.parse_optional(args.source_url),
        notes=curator.parse_optional(args.notes),
        trim_left=args.trim_left,
        trim_right=args.trim_right,
        golden_dir=Path(args.golden_dir) if args.golden_dir else _default_golden_dir(),
        root=_golden_root(args),
        dry_run=args.dry_run,
        force=args.force,
    )

    try:
        inputs.validate()  # report every missing flag before touching the source
        page_obj, meta = extract_page(inputs.source, inputs.page)
        plan = curator.plan_add(inputs, page_obj)
        pick = args.pick
        if pick is None and interactive and len(plan.candidates) > 1:
            print(
                f"candidates for {plan.needle!r} on page {inputs.page}:",
                "\n".join(f"  {i}: {c.display()}" for i, c in enumerate(plan.candidates, 1)),
                sep="\n",
            )
            pick = int(_prompt("pick a candidate (1-based)", "1"))
        candidate = curator.resolve_candidate(plan, pick)
        outcome = curator.execute_add(plan, candidate, meta, now=datetime.now(UTC).date())
    except CuratorError as exc:
        print(f"fmd golden add: {exc}", file=sys.stderr)
        return 2

    _print_add_outcome(outcome, plan, meta, page_obj)
    if outcome.status == "verified":
        return 0
    for reason in outcome.reasons:
        print(f"fmd golden add: check failed: {reason}", file=sys.stderr)
    return 1


def _print_add_outcome(outcome, plan, meta, page_obj) -> None:
    noun = "would write" if outcome.dry_run else "wrote"
    print(
        f"fmd golden add: source {plan.inputs.source} (sha256 {meta.sha256}, "
        f"{meta.page_count} pages); page {page_obj.page} extracted with pdftext "
        f"{meta.tool_version}: {len(page_obj.lines)} lines"
    )
    quote = outcome.entry["expected"][outcome.predicate].get("quote")
    print(f"  needle {plan.needle!r}: {len(plan.candidates)} candidate line(s)")
    print(f"  selected line {candidate_line(outcome, plan)}: quote {quote!r}")
    print(f"  proof: {outcome.proof.summary}")
    if outcome.status == "verified":
        print(f"  status: verified ({noun} a fixture-checked row)")
    else:
        print("  status: unverified (value stays null; excluded from accuracy metrics)")
    for path_name, path in (
        ("entry", outcome.entry_path),
        ("fixture", outcome.fixture_path),
        ("metadata", outcome.sidecar_path),
    ):
        if path is not None:
            print(f"  {path_name}: {path}")
    if outcome.dry_run:
        print("  dry run: nothing was written")


def candidate_line(outcome, plan) -> str:
    quote = outcome.entry["expected"][outcome.predicate].get("quote")
    for candidate in plan.candidates:
        if candidate.quote == quote:
            return str(candidate.line_no)
    return "?"


def _cmd_golden_check(args: argparse.Namespace) -> int:
    from fmd.eval.curator import check_golden

    golden_dir = args.golden or _default_golden_dir()
    report = check_golden(golden_dir, root=_golden_root(args))
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        _print_check_report(report.as_dict())
    return 0 if report.ok else 1


def _print_check_report(payload: dict) -> None:
    print(
        "fmd golden check: {} entries, verified rows {}, unverified {}".format(
            payload.get("entries", 0),
            payload.get("verified_rows", 0),
            payload.get("unverified_rows", 0),
        )
    )
    for family, counts in (payload.get("by_family") or {}).items():
        print(
            "  family {}: entries {entries} verified {verified} unverified {unverified}".format(
                family, **counts
            )
        )
    for fixture in payload.get("fixtures") or []:
        if fixture.get("present") is False:
            print(f"  fixture MISSING: {fixture['fixture']}")
        elif fixture.get("match") is False:
            print(f"  fixture TAMPERED: {fixture['fixture']} (sha256 mismatch)")
        elif fixture.get("match") is None:
            print(f"  fixture unrecorded (pre-tool): {fixture['fixture']}")
        else:
            print(f"  fixture ok: {fixture['fixture']}")
    context_checks = payload.get("context_checks") or []
    failed_contexts = [check for check in context_checks if check.get("ok") is False]
    if context_checks:
        checked = len(context_checks)
        failed = len(failed_contexts)
        print(f"  row contexts: {checked} checked, {failed} failed")
    duplicates = payload.get("duplicates") or {}
    for warning in duplicates.get("warnings") or []:
        print(f"  duplicate warning: {warning}")
    for problem in payload.get("problems") or []:
        print(f"fmd golden check: FAIL {problem}", file=sys.stderr)
    if not payload.get("problems"):
        print("  all offline checks passed")


def _cmd_golden_list(args: argparse.Namespace) -> int:
    from fmd.eval.curator import list_golden

    golden_dir = args.golden or _default_golden_dir()
    payload = list_golden(golden_dir, root=_golden_root(args))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(
        "fmd golden list: {} entries, verified rows {}, unverified {}".format(
            payload["totals"]["entries"],
            payload["totals"]["verified_rows"],
            payload["totals"]["unverified_rows"],
        )
    )
    for entry in payload["entries"]:
        source = entry["fixture"] or entry["source_url"] or "(no source)"
        print(
            "{}  {}  family={family}  measured={measured}  "
            "verified={verified} unverified={unverified}  source={source}".format(
                entry["id"],
                entry["institution"],
                family=entry["family"] or "none",
                measured="yes" if entry["measured"] else "no",
                verified=entry["verified_rows"],
                unverified=entry["unverified_rows"],
                source=source,
            )
        )
        print(f"    gates: {entry['entry_gate']}")
        for row in entry["rows"]:
            context = f" context={row['context']!r}" if row["context"] else ""
            value = row["value"]
            print(
                "    {predicate}: {status}/{expectation} value={value} unit={unit}{context}".format(
                    predicate=row["predicate"],
                    status=row["status"],
                    expectation=row["expectation"],
                    value=value if value is not None else "null",
                    unit=row["unit"] or "none",
                    context=context,
                )
            )
            print(f"      {row['gate']}")
    print("legend:")
    for line in payload["legend"]:
        print(f"  - {line}")
    return 0


def _cmd_golden(args: argparse.Namespace) -> int:
    command = getattr(args, "golden_command", None)
    if command == "add":
        return _cmd_golden_add(args)
    if command == "check":
        return _cmd_golden_check(args)
    if command == "list":
        return _cmd_golden_list(args)
    return 2


# --------------------------------------------------------------- programmes (W8)


def _programme_mapping(registry: Any, database: Any):
    """The registry's programme mapping (institution-site params or defaults)."""
    from fmd.orchestrate.ingest import ProgrammeStage

    stage = ProgrammeStage.from_database(database, registry)
    return stage.mapping()


def _programme_router(registry: Any, database: Any) -> Any | None:
    """The provider-agnostic router for the Tier-2 classifier (or None).

    Every call is audited through ``llm_calls``; the profile is optional, so
    an unconfigured model degrades to ``uncertain`` → ``needs_review`` and
    the run never guesses.
    """
    try:
        from fmd.db.repositories import LLMCallsRepo
        from fmd.llm.router import ModelRouter

        return ModelRouter(registry=registry, sink=LLMCallsRepo(database))
    except Exception as exc:  # pragma: no cover - misconfiguration surfaces in doctor
        print(f"programmes: classifier unavailable ({exc}); names stay uncertain", file=sys.stderr)
        return None


def _scope_tally(drafts: Any) -> dict[str, int]:
    """Dry-run scope counts: what the rules could decide and what waits.

    ``would_classify`` is the number of names no rule or manual decision
    settled; with the classifier skipped in a dry run they are also
    ``would_defer`` (uncertain → needs_review), never silently admitted.
    """
    tally = {"would_admit": 0, "would_reject": 0, "would_defer": 0, "would_classify": 0}
    for draft in drafts or ():
        reason = getattr(draft, "scope_reason", None) or ""
        if reason == "semantic_unavailable_uncertain":
            tally["would_classify"] += 1
            tally["would_defer"] += 1
        elif getattr(draft, "in_scope", False):
            tally["would_admit"] += 1
        elif getattr(draft, "needs_review", False):
            tally["would_defer"] += 1
        else:
            tally["would_reject"] += 1
    return tally


def _cmd_programmes_materialise(args: argparse.Namespace) -> int:
    from fmd.db.repositories import EntitiesRepo, ProgrammesRepo
    from fmd.ingest.materialise import materialise_from_database

    registry = load_registry(args.config_dir)
    database = _database(args)
    try:
        # Lazily build the router: a dry run never needs a model server, so it
        # skips Tier 2 entirely and reports how many names *would* be
        # classified instead of failing on an unreachable endpoint.
        skip_classifier = args.no_classifier or args.dry_run
        router = None if skip_classifier else _programme_router(registry, database)
        result = materialise_from_database(
            database,
            registry,
            router=router,
            dry_run=args.dry_run,
            force=args.force,
        )
        institution_names = {}
        entities = EntitiesRepo(database)
        for entry in result.entries:
            if entry.institution_id not in institution_names:
                row = entities.get(entry.institution_id)
                institution_names[entry.institution_id] = (
                    getattr(row, "canonical_name", None) or f"#{entry.institution_id}"
                )
        programmes = ProgrammesRepo(database)
        counts = dict(programmes.counts())
        scope_counts = programmes.scope_counts()
    finally:
        database.close()

    if args.json:
        payload = result.as_dict()
        payload["institution_names"] = {str(key): value for key, value in institution_names.items()}
        payload["table"] = counts
        payload["scope"] = scope_counts
        if args.dry_run:
            payload["would_scope"] = _scope_tally(result.drafts)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    verb = "would_write" if args.dry_run else "wrote"
    print(
        f"fmd programmes materialise: claims={result.claims} rows={len(result.entries)} "
        f"{verb}={result.written} (create={result.created} update={result.updated} "
        f"unchanged={result.unchanged}) refused_aggregates={len(result.aggregates_refused)} "
        f"classifier={'off' if skip_classifier else 'on'}"
    )
    if args.dry_run:
        tally = _scope_tally(result.drafts)
        print(
            "dry-run scope: would_admit={would_admit} would_reject={would_reject} "
            "would_defer={would_defer} would_classify={would_classify}".format(**tally)
        )
    print(
        "scope: in={in_scope} out={out_of_scope} (rule={tier_rule} semantic={tier_semantic} "
        "manual={tier_manual}) review_queue={review_queue}".format(**scope_counts)
    )
    for entry in result.entries:
        name = institution_names.get(entry.institution_id, f"#{entry.institution_id}")
        prefix = "[dry-run] " if args.dry_run else ""
        marker = f"{prefix}[{entry.action}]"
        changed = ""
        if entry.changed_fields:
            parts = []
            for change in entry.changed_fields:
                parts.append(
                    f"{change.field} (claim {change.old_claim_id or '—'} -> "
                    f"{change.new_claim_id or '—'})"
                )
            changed = " changed: " + ", ".join(parts)
        forced = " [forced]" if entry.forced else ""
        seats = f" seats={entry.seats}" if entry.seats is not None else " seats=NULL"
        review = " review:" + ",".join(entry.review_reasons) if entry.review_reasons else ""
        print(
            f"  {marker} {name} ({entry.institution_id}) {entry.name!r}{seats}"
            f"{changed}{forced}{review}"
        )
    for refusal in result.aggregates_refused:
        print(
            f"  [refused] seats.total claim {refusal.claim_id} value={refusal.value} "
            f"institution={refusal.institution_id}: {refusal.reason}"
        )
    for rejection in result.rejected:
        if rejection.get("reason") == "seats_total_aggregate_not_programme_intake":
            continue  # already reported as [refused]
        print("  [rejected] " + json.dumps(rejection, default=str))
    if args.dry_run:
        print("dry run: nothing written")
    return 0


def _cmd_programmes_list(args: argparse.Namespace) -> int:
    from fmd.db.repositories import EntitiesRepo, ProgrammeFilter, ProgrammesRepo
    from fmd.ingest.materialise import evidence_completeness

    registry = load_registry(args.config_dir)
    database = _database(args)
    try:
        mapping = _programme_mapping(registry, database)
        programmes = ProgrammesRepo(database)
        rows = programmes.list(
            ProgrammeFilter(
                limit=args.limit,
                institution_id=args.institution_id,
                state=args.state,
                in_scope=None if args.all else True,
            )
        )
        out_of_scope_total = programmes.count(ProgrammeFilter(in_scope=False, limit=1))
        # Eligible claims per institution: the same filtered set materialisation
        # and scoring consume (supported, unsuperseded, non-modelled, evidence).
        claims_by_entity: dict[int, list[dict[str, Any]]] = {}
        for claim in ProgrammesRepo(database).materialisation_claims():
            claims_by_entity.setdefault(int(claim["entity_id"]), []).append(claim)
        names: dict[int, str] = {}
        entities = EntitiesRepo(database)
        for row in rows:
            if row.institution_id not in names:
                entity = entities.get(row.institution_id)
                names[row.institution_id] = (
                    getattr(entity, "canonical_name", None) or f"#{row.institution_id}"
                )
    finally:
        database.close()

    if args.json:
        payload = []
        for row in rows:
            completeness = evidence_completeness(
                claims_by_entity.get(row.institution_id, []), mapping, registry.predicates
            )
            payload.append(
                {
                    "id": row.id,
                    "institution_id": row.institution_id,
                    "institution": names.get(row.institution_id),
                    "name": row.name,
                    "seats": row.seats,
                    "seats_source_claim_id": _seats_source_claim_id(row),
                    "completeness": completeness.as_dict(),
                    "needs_review": row.needs_review,
                    "review_reasons": list(row.review_reasons),
                    "in_scope": row.in_scope,
                    "scope_tier": row.scope_tier,
                    "scope_reason": row.scope_reason,
                    "scope_evidence": list(row.scope_evidence or []),
                }
            )
        print(json.dumps({"programmes": payload}, indent=2, default=str))
        return 0

    hidden = "" if args.all else f" ({out_of_scope_total} out-of-scope hidden)"
    print(f"programmes: {len(rows)} row(s){hidden}")
    for row in rows:
        completeness = evidence_completeness(
            claims_by_entity.get(row.institution_id, []), mapping, registry.predicates
        )
        components = (
            f"components {len(completeness.components_backed)}/{completeness.components_total}"
        )
        if completeness.components_backed:
            components += f" ({', '.join(completeness.components_backed)})"
        seats_claim = _seats_source_claim_id(row)
        seats = (
            f"seats {row.seats} (claim {seats_claim})"
            if row.seats is not None and seats_claim
            else f"seats {row.seats if row.seats is not None else '—'} (no claim)"
        )
        review = "review: " + ",".join(row.review_reasons) if row.review_reasons else "review: —"
        scope = f"scope={row.scope_tier or '—'}:{row.scope_reason or '—'}" + (
            "" if row.in_scope else " [OUT]"
        )
        print(
            f"  {row.id} | {names.get(row.institution_id)} ({row.institution_id}) | "
            f"{row.name} | {seats} | "
            f"fields {completeness.fields_backed}/{completeness.fields_total} | "
            f"{components} | {scope} | needs_review={str(row.needs_review).lower()} | {review}"
        )
    return 0


def _seats_source_claim_id(row: Any) -> str | None:
    """The claim id behind the seats field, from ``field_sources``."""
    sources = getattr(row, "field_sources", None) or {}
    for predicate in sorted(sources):
        if predicate in {"seats.intake", "seats.sanctioned"}:
            return str(sources[predicate])
    return None


def _cmd_programmes_inspect(args: argparse.Namespace) -> int:
    from fmd.db.repositories import EntitiesRepo, ProgrammesRepo
    from fmd.ingest.programmes import field_for_predicate_map

    registry = load_registry(args.config_dir)
    database = _database(args)
    try:
        programmes = ProgrammesRepo(database)
        record = programmes.get(args.id)
        if record is None:
            print(f"fmd programmes inspect: no programme with id {args.id}", file=sys.stderr)
            return 1
        mapping = _programme_mapping(registry, database)
        evidence = programmes.evidence_for_programme(args.id)
        entity = EntitiesRepo(database).get(record.institution_id)
    finally:
        database.close()

    bridge = field_for_predicate_map(mapping)
    claim_by_field: dict[str, tuple[str, str]] = {}
    for predicate, claim_id in (record.field_sources or {}).items():
        attribute = bridge.get(predicate, predicate)
        claim_by_field.setdefault(attribute, (predicate, str(claim_id)))
    # credential_level is classified from the name claim, never claimed on its
    # own: cite the name claim so the audit chain stays complete.
    if "credential_level" not in claim_by_field and "name" in claim_by_field:
        claim_by_field["credential_level"] = claim_by_field["name"]

    if args.json:
        print(
            json.dumps(
                {
                    "programme": record.model_dump(mode="json"),
                    "institution": getattr(entity, "canonical_name", None),
                    "fields": {
                        attribute: {"predicate": predicate, "claim_id": claim_id}
                        for attribute, (predicate, claim_id) in claim_by_field.items()
                    },
                    "claims": [row.model_dump(mode="json") for row in evidence],
                },
                indent=2,
                default=str,
            )
        )
        return 0
    print(f"programme {record.id} — {record.name}")
    if entity is not None:
        print(f"institution: {record.institution_id} {getattr(entity, 'canonical_name', '')}")
    print(f"normalised_name: {record.normalised_name}  source_family: {record.source_family}")
    scope_tier = record.scope_tier or "—"
    scope_state = "IN" if record.in_scope else "OUT"
    print(
        f"scope: {scope_state}  tier: {scope_tier}"
        + (f"  reason: {record.scope_reason}" if record.scope_reason else "")
        + ("  [cached]" if record.scope_cached else "")
    )
    if record.scope_evidence:
        for entry in record.scope_evidence:
            print(f"  scope evidence: {entry}")
    if record.scope_model or record.scope_prompt_version:
        print(
            f"  decided by {record.scope_model or '—'}"
            f" (prompt {record.scope_prompt_version or '—'})"
        )
    print(
        f"needs_review: {str(record.needs_review).lower()}"
        + (f"  reasons: {', '.join(record.review_reasons)}" if record.review_reasons else "")
    )
    print("fields (value <- proving claim):")
    attribute_values = [
        ("name", record.name),
        ("credential_level", record.credential_level),
        ("seats", record.seats),
        ("seats_total_aggregate", None),
        ("duration_months", record.duration_months),
        ("internship_months", record.internship_months),
        ("medium", record.medium),
        ("affiliated_university", record.affiliated_university_id),
        ("recognition_status", record.recognition_status),
    ]
    for attribute, value in attribute_values:
        source = claim_by_field.get(attribute)
        if attribute == "seats_total_aggregate":
            note = (
                "not stored — an aggregate (seats.total) is never a per-programme intake"
                " (a refused claim is reported by `fmd programmes materialise`)"
            )
            print(f"  {attribute:22} = {note}")
            continue
        rendered = "NULL" if value is None else str(value)
        if source is None:
            print(f"  {attribute:22} = {rendered}   [no claim]")
        else:
            predicate, claim_id = source
            suffix = " (derived from the name claim)" if attribute == "credential_level" else ""
            print(f"  {attribute:22} = {rendered}   [claim {claim_id} · {predicate}]{suffix}")
    print("claims (field_sources, quote and page):")
    if not evidence:
        print("  (none)")
    for row in evidence:
        print(
            f"  #{row.claim_id} {row.predicate} value={row.value} unit={row.unit or '—'} "
            f"confidence={row.confidence} verification={row.verification_status or '—'}"
        )
        print(f"    page={row.page_range or '—'}")
        print(f"    quote: {row.quote or '—'}")
        print(
            f"    source: {row.url or '—'} (fetched {row.fetched_at or '—'}; "
            f"sha256 {row.sha256 or '—'}; chars {row.char_start}–{row.char_end})"
        )
    return 0


def _cmd_programmes_requalify(args: argparse.Namespace) -> int:
    from fmd.db.repositories import EntitiesRepo, ProgrammesRepo
    from fmd.ingest.programmes import ProgrammeMapping  # noqa: F401 - re-export clarity
    from fmd.programmes.service import requalify_programmes, scoper_for_database

    registry = load_registry(args.config_dir)
    database = _database(args)
    result = None
    try:
        programmes = ProgrammesRepo(database)
        entities = EntitiesRepo(database)
        mapping = _programme_mapping(registry, database)
        # A dry run never needs the model: skip Tier 2 and report the names
        # that would be classified instead of dialling a dead endpoint.
        skip_classifier = args.no_classifier or args.dry_run
        router = None if skip_classifier else _programme_router(registry, database)
        scoper = scoper_for_database(database, registry, mapping, router=router)
        result = requalify_programmes(
            programmes,
            entities,
            mapping,
            scoper,
            dry_run=args.dry_run,
        )
        scope_counts = programmes.scope_counts()
    finally:
        database.close()
    assert result is not None

    if args.json:
        payload = result.as_dict()
        payload["scope"] = scope_counts
        print(json.dumps(payload, indent=2, default=str))
        return 0

    verb = "would_rescope" if args.dry_run else "rescoped"
    print(
        f"fmd programmes requalify: rows={result.rows} {verb}={result.rescoped} "
        f"merged={result.merged} unchanged={result.unchanged} "
        f"uncertain={result.uncertain} classifier={'off' if skip_classifier else 'on'}"
    )
    if args.dry_run:
        print(
            f"dry-run scope: would_admit={result.would_admit} "
            f"would_reject={result.would_reject} would_defer={result.would_defer} "
            f"would_classify={result.would_classify}"
        )
    for note in result.notes:
        print(f"  note: {note}")
    for entry in result.entries:
        prefix = "[dry-run] " if args.dry_run else ""
        if entry.action == "merge":
            print(
                f"  {prefix}[merge] #{entry.programme_id} {entry.name!r} -> "
                f"#{entry.merged_into}" + (f" {entry.merged_rows}" if entry.merged_rows else "")
            )
            continue
        if entry.action == "unchanged":
            print(f"  {prefix}[unchanged] #{entry.programme_id} {entry.name!r}")
            continue
        flags = f" flags={','.join(entry.flags)}" if entry.flags else ""
        review = " review" if entry.needs_review else ""
        print(
            f"  {prefix}[rescope] #{entry.programme_id} {entry.name!r} -> "
            f"{'IN' if entry.in_scope else 'OUT'} tier={entry.scope_tier} "
            f"reason={entry.scope_reason}{review}{flags}"
        )
    if args.dry_run:
        print("dry run: nothing written")
    return 0


def _cmd_programmes_decide(args: argparse.Namespace) -> int:
    from fmd.db.repositories import EntitiesRepo, ProgrammesRepo
    from fmd.programmes.service import record_manual_decision, scoper_for_database

    registry = load_registry(args.config_dir)
    database = _database(args)
    outcome = None
    try:
        programmes = ProgrammesRepo(database)
        entities = EntitiesRepo(database)
        mapping = _programme_mapping(registry, database)
        taxonomy = scoper_for_database(database, registry, mapping, router=None).taxonomy
        outcome = record_manual_decision(
            programmes,
            entities,
            taxonomy,
            registry.programmes,
            args.id,
            equivalent=bool(args.equivalent),
            note=args.note,
        )
        scope_counts = programmes.scope_counts()
    finally:
        database.close()

    if args.json:
        print(json.dumps({**outcome.as_dict(), "scope": scope_counts}, indent=2, default=str))
        return 0

    verdict = "IN scope (equivalent)" if outcome.equivalent else "OUT of scope (not equivalent)"
    print(f"fmd programmes decide: programme {outcome.programme_id} {outcome.name!r} -> {verdict}")
    if outcome.previous:
        previous = outcome.previous
        print(
            f"  replaced: {previous.get('tier')}/{previous.get('decision')} "
            f"by {previous.get('decided_by')}"
            + (f" — {previous.get('rationale')}" if previous.get("rationale") else "")
        )
    print(f"  decision key: {outcome.cache_key}")
    if outcome.note:
        print(f"  note: {outcome.note}")
    print(
        "  this manual decision wins over rules and the classifier from the next "
        "materialise/requalify run (and was applied to the row now)."
    )
    return 0


def _cmd_programmes(args: argparse.Namespace) -> int:
    command = getattr(args, "programmes_command", None)
    if command == "materialise":
        return _cmd_programmes_materialise(args)
    if command == "list":
        return _cmd_programmes_list(args)
    if command == "inspect":
        return _cmd_programmes_inspect(args)
    if command == "requalify":
        return _cmd_programmes_requalify(args)
    if command == "decide":
        return _cmd_programmes_decide(args)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in UNIMPLEMENTED:
        workstream, section = UNIMPLEMENTED[args.command]
        print(
            f"{args.command}: not implemented ({workstream}, see brief §{section})",
            file=sys.stderr,
        )
        return 2

    if (
        args.command == "eval"
        and getattr(args, "eval_kind", "pipeline") == "pipeline"
        and args.golden is None
    ):
        # The legacy no-argument invocation is still a stub; a real run must name
        # its golden set so it can never silently evaluate an empty directory.
        # product/verdict kinds take --fixture-path instead.
        print(
            "eval: --golden is required for --eval-kind pipeline; the no-argument "
            "form is still not implemented (WS-F, see brief §19)",
            file=sys.stderr,
        )
        return 2

    try:
        if args.command == "migrate":
            return _cmd_migrate(args)
        if args.command == "seed":
            return _cmd_seed(args)
        if args.command == "score":
            return _cmd_score(args)
        if args.command == "crawl":
            return _cmd_crawl(args)
        if args.command == "extract":
            return _cmd_extract(args)
        if args.command == "acquire":
            return _cmd_acquire(args)
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "classify-report":
            return _cmd_classify_report(args)
        if args.command == "coverage-report":
            return _cmd_coverage_report(args)
        if args.command == "verify":
            return _cmd_verify(args)
        if args.command == "worker":
            return _cmd_worker(args)
        if args.command == "failures":
            return _cmd_failures(args)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "llm":
            command = getattr(args, "llm_command", None)
            if command in {"doctor", "ping"}:
                return _cmd_llm_doctor(args)
            parser.error("llm: choose `doctor` or `ping`")
        if args.command == "golden":
            return _cmd_golden(args)
        if args.command == "programmes":
            return _cmd_programmes(args)
        if args.command == "llm-ping":
            return _cmd_llm_ping(args)
    except (ConfigError, DatabaseError, MigrationError, SeedError) as exc:
        print(f"fmd {args.command}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        from fmd.orchestrate import OrchestrationError

        if isinstance(exc, OrchestrationError):
            print(f"fmd {args.command}: {exc}", file=sys.stderr)
            return 1
        raise
    except NotImplementedYet as exc:  # pragma: no cover - defensive
        print(f"fmd {args.command}: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
