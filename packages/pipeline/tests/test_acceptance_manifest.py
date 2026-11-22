"""Pins for the acceptance harness manifest (wave v1-w2).

Brief item 2: "every one of the acceptance checks either PASS or SKIP with a
stated reason; no check silently omitted. Add a `--json` mode for CI."

scripts/acceptance.sh carries a canonical EXPECTED_CHECKS manifest and a
safety net that records a SKIP for any manifest id that reached no verdict.
These tests pin the manifest itself: complete (no omitted check id), exact
(only ids the script actually records), and surfaced in the --json-out report
(missing[] computed, verdict field present).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "acceptance.sh"

CANONICAL_MANIFEST = [
    "compose/smoke",
    "preflight/db-reachable",
    "preflight/migrations",
    "preflight/registry-seeded",
    "preflight/model-reachable",
    "preflight/serve-local-probe",
    "api/server",
    "api/health",
    "api/sources",
    "api/programmes",
    "api/programme-detail",
    "api/programme-cost",
    "api/programme-filters",
    "api/rankings",
    "api/compare",
    "api/exams",
    "api/exclusions",
    "web/pnpm",
    "web/lint",
    "web/build",
    "web/test",
    "web/server",
    "web/render /",
    "web/render /shortlist",
    "web/render /programmes/<id>",
    "web/render /compare",
    "web/render /exams",
    "web/render /exclusions",
    "web/render /sources",
    "privacy/response-bodies",
    "privacy/api-logs",
    "privacy/db-rows",
]


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def extract_expected_checks() -> list[str]:
    """Pull the EXPECTED_CHECKS array out of the script (order-sensitive)."""
    text = script_text()
    match = re.search(r"^EXPECTED_CHECKS=\(\n(.*?)^\)$", text, re.S | re.M)
    assert match, "scripts/acceptance.sh must define EXPECTED_CHECKS=( ... )"
    ids = re.findall(r'"([^"]+)"', match.group(1))
    assert ids, "EXPECTED_CHECKS must not be empty"
    return ids


def literal_record_sites() -> set[str]:
    """First arguments of every literal `record "id"` call in the script."""
    text = script_text()
    # record may take a line-continuation; capture only the quoted first arg.
    ids = set(re.findall(r'record\s+"([^"$]+)"', text))
    return ids


def test_manifest_matches_the_canonical_inventory() -> None:
    assert extract_expected_checks() == CANONICAL_MANIFEST, (
        "EXPECTED_CHECKS drifted from the canonical manifest — a check was "
        "added/removed/renamed without updating the pinned inventory"
    )


def variable_covered_ids(manifest: set[str], text: str) -> set[str]:
    """Ids covered through variable record sites (api helpers + render loop)."""
    api_labels = set(re.findall(r'api_(?:get|post)\s+"([^"$]+)"', text))
    api_labels |= {"health", "sources", "programmes"}  # the API-down skip loop
    api_ids = {f"api/{lbl}" for lbl in api_labels}
    render_ids = {m for m in manifest if m.startswith("web/render ")}
    assert 'record "api/$label"' in text, (
        "api_get/api_post helpers must record 'api/$label'"
    )
    assert "record \"web/render $route\"" in text, (
        "the render loop must record `web/render $route` (variable) so live "
        "routes land in the verdict table"
    )
    return api_ids | render_ids


def test_every_literal_record_site_is_in_the_manifest() -> None:
    manifest = set(extract_expected_checks())
    variable = variable_covered_ids(manifest, script_text())
    sites = literal_record_sites() - variable
    # `web/pnpm-install` is an auxiliary setup step that only records failures
    # and is intentionally outside the manifest.
    sites -= {"web/pnpm-install"}
    unmanifested = sites - manifest
    assert not unmanifested, (
        f"record() sites not present in EXPECTED_CHECKS: {sorted(unmanifested)} — "
        "add them to the manifest or mark them auxiliary"
    )


def test_every_manifest_id_has_at_least_one_record_site() -> None:
    text = script_text()
    manifest = set(extract_expected_checks())
    variable = variable_covered_ids(manifest, text)
    covered = set(literal_record_sites()) | variable
    covered -= {"web/pnpm-install"}  # auxiliary setup step, only records failures
    assert covered == manifest, (
        f"coverage mismatch: unmanifested={sorted(covered - manifest)} "
        f"never-recorded={sorted(manifest - covered)}"
    )


def test_safety_net_exists_and_covers_manifest_gaps() -> None:
    text = script_text()
    assert "harness safety net" in text, (
        "the safety net (records SKIP for any manifest id that reached no "
        "verdict) is missing — checks can be silently omitted again"
    )
    assert re.search(r"for expected in \"\$\{EXPECTED_CHECKS\[@\]\}\"; do", text), (
        "the safety net must iterate EXPECTED_CHECKS, not a hardcoded list"
    )


def test_json_report_carries_manifest_completeness_and_verdict() -> None:
    text = script_text()
    for needle in ('"manifest"', '"missing"', '"complete"', '"verdict"'):
        assert needle in text, (
            f"the --json-out report must expose {needle} for CI to consume"
        )


def test_json_writer_python_parses() -> None:
    # The report writer is an inline heredoc; parse it so a syntax error cannot
    # surface only at acceptance time. The script has several PYEOF heredocs —
    # pick the one that builds the JSON report.
    text = script_text()
    bodies = re.findall(r"<<'PYEOF'\n(.*?)\nPYEOF", text, re.S)
    writers = [b for b in bodies if '"manifest"' in b]
    assert len(writers) == 1, "exactly one heredoc must build the JSON report"
    ast.parse(writers[0])
