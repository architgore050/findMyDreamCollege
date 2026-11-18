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


def _cmd_plan(args: argparse.Namespace) -> int:
    from fmd.orchestrate.plan import run_planning_pass

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
                f"    degrades to {row['degrade_reason']} per document; text extraction still runs."
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
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "verify":
            return _cmd_verify(args)
        if args.command == "worker":
            return _cmd_worker(args)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "llm":
            command = getattr(args, "llm_command", None)
            if command in {"doctor", "ping"}:
                return _cmd_llm_doctor(args)
            parser.error("llm: choose `doctor` or `ping`")
        if args.command == "golden":
            return _cmd_golden(args)
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
