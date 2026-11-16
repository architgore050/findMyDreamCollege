"""``fmd`` command line entrypoint.

Only workstream-A commands do work. Commands owned by other workstreams fail
with a non-zero exit and a pointer to the brief rather than pretending to run.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    llm_ping = llm_sub.add_parser(
        "ping", help="alias for `doctor` without --live"
    )
    llm_ping.add_argument("--json", action="store_true", help="machine-readable output")

    # Backwards-compatible top-level alias; prefer `fmd llm doctor`.
    ping = subparsers.add_parser(
        "llm-ping", help="alias for `fmd llm doctor` (never prints secrets)"
    )
    ping.add_argument(
        "--json", action="store_true", help="machine-readable output (default: table)"
    )
    return parser


def _database(args: argparse.Namespace) -> Database:
    return Database(args.database_url)


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
    orchestrator, database = _orchestrator(args)
    try:
        summary = orchestrator.extract(limit=args.limit, force=args.force)
    finally:
        database.close()
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


def _cmd_eval(args: argparse.Namespace) -> int:
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
    failures = apply_gate(report, args.fail_under)
    report["ok"] = not failures
    report_path = None
    if args.report_out:
        report_path = write_report(report, args.report_out)
    print(render_summary(report, report_path=report_path))
    for failure in failures:
        print(f"fmd eval: FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


def _cmd_llm_doctor(args: argparse.Namespace) -> int:
    from fmd.llm.doctor import doctor_report

    registry = load_registry(args.config_dir)
    live = bool(getattr(args, "live", False))
    report = doctor_report(registry.models, live=live)
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
                f"    degrades to {row['degrade_reason']} per document; "
                "text extraction still runs."
            )
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

    if args.command == "eval" and args.golden is None:
        # The legacy no-argument invocation is still a stub; a real run must name
        # its golden set so it can never silently evaluate an empty directory.
        print(
            "eval: --golden is required; the no-argument form is still "
            "not implemented (WS-F, see brief §19)",
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
        if args.command == "worker":
            return _cmd_worker(args)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "llm":
            command = getattr(args, "llm_command", None)
            if command in {"doctor", "ping"}:
                return _cmd_llm_doctor(args)
            parser.error("llm: choose `doctor` or `ping`")
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
