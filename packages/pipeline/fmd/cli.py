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
from fmd.llm import ModelRouter
from fmd.seed import SeedError, seed_database


class NotImplementedYet(RuntimeError):
    def __init__(self, command: str, workstream: str, section: str) -> None:
        self.command = command
        self.workstream = workstream
        self.section = section
        super().__init__(f"not implemented ({workstream}, see brief §{section})")


# Commands owned by other workstreams (brief §21). Keeping this table explicit
# means `make crawl` cannot silently succeed before WS-B lands.
UNIMPLEMENTED: dict[str, tuple[str, str]] = {
    "crawl": ("WS-B", "7"),
    "extract": ("WS-D", "9"),
    "eval": ("WS-F", "19"),
    "worker": ("WS-B", "7"),
}


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

    for name, help_text in (
        ("crawl", "run the crawl frontier (WS-B)"),
        ("extract", "snapshots -> extractions -> claims (WS-D)"),
        ("eval", "golden set, provenance coverage, drift (WS-F)"),
        ("worker", "run the background worker (WS-B)"),
    ):
        subparsers.add_parser(name, help=help_text)

    score_parser = subparsers.add_parser(
        "score", help="claims -> scores + insight sets (WS-F, idempotent)"
    )
    score_parser.add_argument(
        "--profile",
        default=None,
        help=(
            "profile JSON as a path or inline object (runtime input only; never persisted)"
        ),
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

    ping = subparsers.add_parser(
        "llm-ping", help="report model profile resolution without printing secrets"
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
    from fmd.db.repositories import (
        ClaimsRepo,
        ProgrammeFilter,
        ProgrammesRepo,
        ScoresRepo,
    )
    from fmd.filters import evaluate_all
    from fmd.score import score_programme
    from fmd.score.config import ScoringConfig
    from fmd.score.persist import persist_score

    registry = load_registry(args.config_dir)
    profile = _load_profile(args.profile)
    config = ScoringConfig.from_config(registry.weights)
    problems = config.validate()
    if problems:
        print("fmd score: invalid scoring config: " + "; ".join(problems), file=sys.stderr)
        return 1

    written = skipped = unknown = 0
    programmes_scored = 0
    with _database(args) as database:
        programmes = ProgrammesRepo(database).list(
            ProgrammeFilter(limit=max(1, args.limit), offset=0)
        )
        for programme in programmes:
            claims = ClaimsRepo(database).list_for_entity(programme.institution_id)
            view = programme.model_dump(mode="json")
            view["claims"] = claims
            summary = evaluate_all(view, profile)
            result = score_programme(
                programme.id,
                view,
                profile,
                claims,
                config=config,
                cost_rules=registry.cost_rules,
                filter_decisions=list(summary.decisions.values()),
            )
            programmes_scored += 1
            if result.composite.point is None:
                unknown += 1
            if args.dry_run:
                point = result.composite.point
                rendered = "UNKNOWN" if point is None else f"{point:.2f}"
                print(
                    f"[dry-run] programme={programme.id} composite={rendered} "
                    f"tier={result.evidence_tier} claims={len(claims)}"
                )
                continue
            latest = ScoresRepo(database).latest_for_entity(programme.institution_id)
            if (
                not args.force
                and latest is not None
                and latest.claims_hash == result.claims_hash
                and latest.scorer_version == result.scorer_version
                and latest.weights_version == result.weights_version
            ):
                skipped += 1
                continue
            persist_score(database, result)
            written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"fmd score: {verb} {written}, skipped {skipped}, unknown {unknown}, "
        f"programmes {programmes_scored} "
        f"(scorer={config.scorer_version}, weights={config.weights_version})"
    )
    return 0


def _cmd_llm_ping(args: argparse.Namespace) -> int:
    registry = load_registry(args.config_dir)
    router = ModelRouter(registry.models)
    report = router.ping()
    if args.json:
        print(
            json.dumps(
                {
                    "profiles": report,
                    "budgets": registry.models.budgets.model_dump(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"profiles in {Path(registry.config_dir) / 'models.yaml'}:")
    for row in report:
        status = "resolved" if row["resolved"] else "UNRESOLVED"
        key = (
            "n/a"
            if row["key_present"] is None
            else ("present" if row["key_present"] else "MISSING")
        )
        print(
            f"  {row['profile']:<14} {status:<11} provider={row['provider']:<6} "
            f"model={row['model'] or '-'} key={key}"
        )
        if not row["resolved"]:
            print(f"      reason: {row['reason']}")
    budgets = registry.models.budgets
    print(
        f"budgets: daily=${budgets.daily_usd:.2f} per_run=${budgets.per_run_usd:.2f} "
        f"hard_stop={budgets.hard_stop} cache={budgets.cache}"
    )
    print("secrets are never printed; only presence is reported.")
    return 0


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

    try:
        if args.command == "migrate":
            return _cmd_migrate(args)
        if args.command == "seed":
            return _cmd_seed(args)
        if args.command == "score":
            return _cmd_score(args)
        if args.command == "llm-ping":
            return _cmd_llm_ping(args)
    except (ConfigError, DatabaseError, MigrationError, SeedError) as exc:
        print(f"fmd {args.command}: {exc}", file=sys.stderr)
        return 1
    except NotImplementedYet as exc:  # pragma: no cover - defensive
        print(f"fmd {args.command}: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
