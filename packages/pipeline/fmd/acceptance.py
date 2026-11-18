"""Acceptance-path assertion helpers (scripts/acceptance.sh).

Pure functions over parsed JSON / HTML text so the acceptance script's checks
can be unit-tested with no database, no API server and no model (wave w1,
workstream e2e-acceptance). The shell script shells out to the CLI at the
bottom of this module; ``tests/test_api_acceptance.py`` imports the same
functions, so a check can only drift from its test if the contract itself
moves.

Enforced here (brief §17 envelope, AGENTS.md invariant 5, brief §11/§13
money rules):

- every response carries the evidence envelope (scorer_version,
  weights_version, claims_hash, generated_at);
- UNKNOWN is a value: a sub-score that reports itself unknown never renders a
  point estimate of 0 or 50;
- every computed money/salary block carries ``method`` and, when it is a
  range, both ``lo`` and ``hi`` with ``lo <= hi``;
- hard-filter ``fail``/``unknown`` decisions always carry a reason;
- rendered HTML never leaks raw NaN/undefined or a bare composite decimal;
- a synthetic-profile sentinel never reaches a response body, a server log or
  a database row.

Standard library only: the script must be able to run this module with a bare
``python3`` (``PYTHONPATH=packages/pipeline``) on a host without a synced
virtual environment.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from typing import Any

ENVELOPE_KEYS = ("scorer_version", "weights_version", "claims_hash", "generated_at")
CLAIMS_HASH_LENGTH = 64
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

# Computed money/salary blocks (cost model outputs and ranking cost payloads).
# Raw evidence rows (``costs`` items on programme detail) are provenance-tagged
# via source_claim_id and are deliberately not subject to the method rule.
MONEY_KEYS = frozenset(
    {
        "tcoa",
        "tcoa_first_year",
        "net_price",
        "aid_total",
        "loan_principal",
        "family_out_of_pocket",
        "student_work_offset",
        "debt_at_repayment",
        "emi",
        "dsr25",
        "affordability_gap",
        "p25_first_year_salary",
    }
)

VALID_FILTER_RESULTS = ("pass", "fail", "unknown")


# ----------------------------------------------------------------- envelope


def envelope_failures(payload: Any) -> list[str]:
    """The response must carry the four envelope keys with sane values."""

    if not isinstance(payload, dict):
        return ["response is not a JSON object"]
    problems: list[str] = []
    for key in ENVELOPE_KEYS:
        if key not in payload:
            problems.append(f"missing envelope key: {key}")
    if problems:
        return problems
    for key in ("scorer_version", "weights_version"):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            problems.append(f"envelope {key} must be a non-empty string, got {value!r}")
    claims_hash = payload["claims_hash"]
    if not isinstance(claims_hash, str) or not _HEX_RE.match(claims_hash):
        problems.append(
            f"envelope claims_hash must be {CLAIMS_HASH_LENGTH} hex chars, got {claims_hash!r}"
        )
    generated_at = payload["generated_at"]
    if not isinstance(generated_at, str):
        problems.append(f"envelope generated_at must be an ISO-8601 string, got {generated_at!r}")
    else:
        try:
            datetime.fromisoformat(generated_at)
        except ValueError:
            problems.append(f"envelope generated_at is not ISO-8601: {generated_at!r}")
    return problems


# ----------------------------------------------------------------- unknown


def _is_unknown_marked(node: dict[str, Any]) -> bool:
    """True when the dict itself says its value is unknown/not measured."""

    if node.get("unknown_reason"):
        return True
    if node.get("measured") is False:
        return True
    return node.get("n_claims") == 0


def unknown_render_failures(payload: Any, path: str = "$") -> list[str]:
    """UNKNOWN must never be rendered as a point estimate of 0 or 50.

    Walks every ``point`` field: a non-null point of exactly 0 or 50 on a dict
    that marks itself unknown (unknown_reason, measured=false or n_claims=0)
    is an invariant violation (AGENTS.md invariant 5). A null point with the
    honest [0, 100] interval is the correct render and passes.
    """

    problems: list[str] = []
    if isinstance(payload, dict):
        point = payload.get("point")
        if (
            isinstance(point, (int, float))
            and not isinstance(point, bool)
            and point in (0, 50)
            and _is_unknown_marked(payload)
        ):
            problems.append(
                f"{path}: unknown value rendered as point {point!r} "
                f"(must stay UNKNOWN, never 0 or 50)"
            )
        for key, value in payload.items():
            problems.extend(unknown_render_failures(value, f"{path}.{key}"))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            problems.extend(unknown_render_failures(item, f"{path}[{index}]"))
    return problems


# ----------------------------------------------------------------- money


def _money_block_problems(name: str, node: Any, path: str) -> list[str]:
    if not isinstance(node, dict):
        if node is None:
            return []
        if isinstance(node, bool) or not isinstance(node, (int, float, str)):
            return []
        return [
            f"{path}: money field {name!r} is a bare scalar {node!r}; "
            "computed money must be a band object with method and lo/hi"
        ]

    problems: list[str] = []
    value = node.get("value")
    method = node.get("method")
    lo = node.get("lo")
    hi = node.get("hi")

    if value is not None:
        if method is None or (isinstance(method, str) and not method.strip()):
            problems.append(f"{path}: money block {name!r} has value {value!r} but no method")
        # A range must carry both bounds and order them.
        if lo is not None or hi is not None:
            if lo is None or hi is None:
                problems.append(
                    f"{path}: money block {name!r} declares a partial range "
                    f"(lo={lo!r}, hi={hi!r}); a range needs both lo and hi"
                )
            elif (
                isinstance(lo, (int, float))
                and not isinstance(lo, bool)
                and isinstance(hi, (int, float))
                and not isinstance(hi, bool)
                and lo > hi
            ):
                problems.append(
                    f"{path}: money block {name!r} has inverted bounds lo={lo!r} > hi={hi!r}"
                )
    return problems


def _is_ratio_block(node: Any) -> bool:
    """True for computed-ratio blocks (DSR₂₅, affordability gap).

    These publish their method on the block itself and their scalar children
    (first_year_emi, p25_first_year_salary, family_out_of_pocket, …) are
    contract-defined anchors — the web client's dsr25/affordability schemas
    pin them as numbers — so the per-child money rules do not apply inside
    them.
    """

    return isinstance(node, dict) and "formula" in node and ("band" in node or "flag_above" in node)


def money_method_failures(payload: Any, path: str = "$") -> list[str]:
    """Every computed money/salary block carries method; ranges carry lo and hi."""

    problems: list[str] = []
    if isinstance(payload, dict):
        if _is_ratio_block(payload) and not str(payload.get("method") or "").strip():
            problems.append(
                f"{path}: ratio block with formula {payload.get('formula')!r} carries no method"
            )
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in ("methods", "display"):
                # Snapshot-level maps, not money values: `methods` is
                # name -> method-string, `display` holds pre-rendered strings
                # that already embed the method (e.g. "INR 920,000 (…, modelled)").
                continue
            if (
                not _is_ratio_block(payload)
                and not _is_ratio_block(value)
                and (key in MONEY_KEYS or key.endswith("_salary"))
            ):
                problems.extend(_money_block_problems(key, value, child_path))
                # A salary block that itself contains named money children
                # (rare) still gets walked below.
            problems.extend(money_method_failures(value, child_path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            problems.extend(money_method_failures(item, f"{path}[{index}]"))
    return problems


# ----------------------------------------------------------------- filters


def filter_reason_failures(payload: Any, path: str = "$") -> list[str]:
    """``fail``/``unknown`` filter decisions always carry a reason."""

    problems: list[str] = []
    if isinstance(payload, dict):
        if "filter_id" in payload and "result" in payload:
            result = payload["result"]
            if result not in VALID_FILTER_RESULTS:
                problems.append(
                    f"{path}: filter {payload['filter_id']!r} has result {result!r}; "
                    f"expected one of {VALID_FILTER_RESULTS}"
                )
            elif result in ("fail", "unknown"):
                reason = payload.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    problems.append(
                        f"{path}: filter {payload['filter_id']!r} returned {result} "
                        "without a reason"
                    )
        for key, value in payload.items():
            problems.extend(filter_reason_failures(value, f"{path}.{key}"))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            problems.extend(filter_reason_failures(item, f"{path}[{index}]"))
    return problems


# ----------------------------------------------------------------- cost/profile


def cost_respects_profile_failures(
    payload: Any,
    *,
    loan_required: bool | None = None,
    family_income: float | None = None,
) -> list[str]:
    """The cost endpoint must not ignore the profile it was POSTed.

    The acceptance sweep posts a synthetic profile with ``loan_required=false``
    and an explicit ``annual_family_income``. Assertions that hold by contract:

    - a loan-free profile must never produce a positive loan principal or
      debt-at-repayment;
    - when the snapshot is available and the profile supplies a family income
      point, the affordability gap is computed (never silently UNKNOWN).
    """

    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["cost payload is not a JSON object"]
    if payload.get("available") is not True:
        # Unavailable snapshots must say why instead of inventing numbers.
        if not str(payload.get("reason", "")).strip():
            problems.append("cost snapshot unavailable but no reason given")
        return problems

    totals = payload.get("scenario_totals")
    if not isinstance(totals, dict) or not totals:
        return ["available cost payload carries no scenario_totals"]

    for scenario, block in totals.items():
        if not isinstance(block, dict):
            problems.append(f"scenario_totals.{scenario} is not an object")
            continue
        if loan_required is False:
            principal = block.get("loan_principal") or {}
            debt = block.get("debt_at_repayment") or {}
            for name, band in (("loan_principal", principal), ("debt_at_repayment", debt)):
                value = band.get("value") if isinstance(band, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    problems.append(
                        f"scenario_totals.{scenario}.{name}.value is {value!r} but the "
                        "profile states loan_required=false"
                    )
        if family_income is not None:
            gap = block.get("affordability_gap") or {}
            gap_value = gap.get("value") if isinstance(gap, dict) else None
            if gap_value is None:
                problems.append(
                    f"scenario_totals.{scenario}.affordability_gap.value is None although "
                    "the profile supplied annual_family_income; the gap must be computed "
                    "when the snapshot is available"
                )
    return problems


# ----------------------------------------------------------------- html


_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BARE_DECIMAL_RE = re.compile(r"composite[^0-9<]{0,24}\d+\.\d+", re.IGNORECASE)


def strip_script_style(html: str) -> str:
    """Rendered markup only: framework bundles legitimately contain 'undefined'
    inside <script> payloads, so the artifact scan must not read them."""

    without = _SCRIPT_RE.sub(" ", html)
    without = _STYLE_RE.sub(" ", without)
    return _COMMENT_RE.sub(" ", without)


def html_artifact_failures(html: str) -> list[str]:
    """Smoke-rendered HTML must not leak raw NaN/undefined/Infinity, a bare
    composite decimal, or a JavaScript artifact into visible markup."""

    problems: list[str] = []
    markup = strip_script_style(html)
    for artifact in ("NaN", "undefined", "Infinity", "[object Object]"):
        if artifact in markup:
            problems.append(f"rendered HTML contains raw {artifact!r}")
    match = _BARE_DECIMAL_RE.search(markup)
    if match:
        problems.append(
            f"rendered HTML shows a bare composite decimal: {match.group(0)[:60]!r} "
            "(composite must render as a band, not a fake decimal)"
        )
    return problems


# ----------------------------------------------------------------- privacy


def privacy_failures(text: str, sentinel: str) -> list[str]:
    """The synthetic-profile sentinel must never appear in a response body,
    a server log line, or a persisted row."""

    if not sentinel or len(sentinel) < 8:
        return ["privacy sentinel is missing or too short to be meaningful"]
    if sentinel in text:
        return [f"profile sentinel {sentinel!r} leaked into output that must not persist it"]
    return []


# ----------------------------------------------------------------- CLI


def _run(
    check: str,
    sentinel: str | None,
    raw: str,
    *,
    loan_required: bool | None = None,
    family_income: float | None = None,
) -> list[str]:
    if check == "html":
        return html_artifact_failures(raw)
    if check == "privacy":
        return privacy_failures(raw, sentinel or "")
    payload = json.loads(raw)
    if check == "envelope":
        return envelope_failures(payload)
    if check == "unknown":
        return unknown_render_failures(payload)
    if check == "money":
        return money_method_failures(payload)
    if check == "filters":
        return filter_reason_failures(payload)
    if check == "cost-profile":
        return cost_respects_profile_failures(
            payload, loan_required=loan_required, family_income=family_income
        )
    raise SystemExit(f"unknown acceptance check: {check}")


KNOWN_CHECKS = ("envelope", "unknown", "money", "filters", "cost-profile", "html", "privacy")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m fmd.acceptance <check> [options] [FILE]", file=sys.stderr)
        print(
            f"checks: {' '.join(KNOWN_CHECKS)} (FILE omitted or '-' reads stdin)", file=sys.stderr
        )
        print(
            "options: --sentinel S (privacy) | --loan-required true|false and "
            "--family-income N (cost-profile)",
            file=sys.stderr,
        )
        return 2
    check = args.pop(0)
    if check not in KNOWN_CHECKS:
        print(f"unknown acceptance check: {check}", file=sys.stderr)
        return 2
    sentinel: str | None = None
    loan_required: bool | None = None
    family_income: float | None = None
    target: str | None = None
    while args:
        arg = args.pop(0)
        if arg == "--sentinel":
            if not args:
                print("--sentinel requires a value", file=sys.stderr)
                return 2
            sentinel = args.pop(0)
        elif arg == "--loan-required":
            if not args:
                print("--loan-required requires true|false", file=sys.stderr)
                return 2
            loan_required = args.pop(0).strip().lower() == "true"
        elif arg == "--family-income":
            if not args:
                print("--family-income requires a number", file=sys.stderr)
                return 2
            family_income = float(args.pop(0))
        elif target is None:
            target = arg
        else:
            print(f"unexpected argument: {arg}", file=sys.stderr)
            return 2

    if target in (None, "-"):
        raw = sys.stdin.read()
    else:
        with open(target, encoding="utf-8") as handle:
            raw = handle.read()

    try:
        problems = _run(
            check,
            sentinel,
            raw,
            loan_required=loan_required,
            family_income=family_income,
        )
    except json.JSONDecodeError as exc:
        print(f"FAIL: input is not valid JSON: {exc}")
        return 1
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print(f"ok: {check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
