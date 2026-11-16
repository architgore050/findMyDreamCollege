"""Cross-adapter contract: fetch layer only.

An adapter yields ``DocumentRef`` values. It never constructs a claim, never
parses a number into one and never touches the database (AGENTS.md invariants
2 and 3). This module enforces that mechanically for the WS-C3 adapters so a
future edit cannot quietly grow an extraction path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fmd.discover.base import DocumentRef

ADAPTERS = Path(__file__).resolve().parents[1] / "fmd" / "adapters"

WS_C3_MODULES = (
    "institution_site",
    "health_university",
    "fee_regulator",
    "state_council",
    "nirf",
    "pay_matrix",
)

FORBIDDEN_IMPORT_PREFIXES = (
    "fmd.db",
    "fmd.extract",
    "fmd.score",
    "fmd.claims",
    "fmd.er",
    "psycopg",
    "sqlalchemy",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", WS_C3_MODULES)
def test_adapter_imports_no_extraction_or_database_module(module: str) -> None:
    imports = _imports(ADAPTERS / f"{module}.py")
    for name in imports:
        assert not name.startswith(FORBIDDEN_IMPORT_PREFIXES), f"{module} imports {name}"


@pytest.mark.parametrize("module", WS_C3_MODULES)
def test_adapter_has_no_write_surface(module: str) -> None:
    tree = ast.parse((ADAPTERS / f"{module}.py").read_text(encoding="utf-8"))
    methods = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for forbidden in ("write", "insert", "persist", "save", "commit", "to_claim"):
        assert forbidden not in methods, f"{module}.{forbidden}() must not exist in the fetch layer"


def test_every_ref_from_every_adapter_is_a_lead(registry) -> None:
    sources = [
        "seventh_cpc_pay_matrix",
        "state_paramedical_councils",
        "state_health_universities",
        "nirf_rankings",
        "fee_regulators",
        "institution_sites",
    ]
    from fmd.adapters.fee_regulator import FeeRegulatorAdapter
    from fmd.adapters.health_university import HealthUniversityAdapter
    from fmd.adapters.institution_site import InstitutionSiteAdapter
    from fmd.adapters.nirf import NirfAdapter
    from fmd.adapters.pay_matrix import PayMatrixAdapter
    from fmd.adapters.state_council import StateCouncilAdapter

    fixtures = {
        "state_paramedical_councils": (
            "state_council_ncahp_statecouncil_snippet.html",
            "https://ncahp.abdm.gov.in/StateCouncil",
        ),
        "state_health_universities": (
            "health_university_ruhs_listing_snippet.html",
            "https://ruhsraj.org/affiliated-colleges/",
        ),
        "nirf_rankings": (
            "nirf_2025_medical_snippet.html",
            "https://www.nirfindia.org/Rankings/2025/MedicalRanking.html",
        ),
        "fee_regulators": (
            "fee_regulator_order_listing_structural.html",
            "https://fee.example.gov.in/orders/",
        ),
        "institution_sites": (
            "institution_site_nirf_listing_snippet.html",
            "https://mtct.ac.in/nirf/",
        ),
    }
    builders = {
        "state_paramedical_councils": lambda source: StateCouncilAdapter(
            source, denylisted_domains=registry.sources.denied_domains()
        ),
        "state_health_universities": HealthUniversityAdapter,
        "nirf_rankings": NirfAdapter,
        "fee_regulators": FeeRegulatorAdapter,
        "institution_sites": InstitutionSiteAdapter,
        "seventh_cpc_pay_matrix": PayMatrixAdapter,
    }

    seen: set[str] = set()
    for source_id in sources:
        adapter = builders[source_id](registry.sources.by_id(source_id))
        assert source_id in adapter.source_id or adapter.source_id == source_id
        if source_id in fixtures:
            filename, base = fixtures[source_id]
            text = (Path(__file__).parent / "fixtures" / "adapters" / filename).read_text(
                encoding="utf-8"
            )
            refs = adapter.parse_listing(text, base_url=base)
        else:
            refs = list(adapter.fetch())
        assert refs, source_id
        for ref in refs:
            assert isinstance(ref, DocumentRef)
            assert ref.is_lead is True
            assert ref.source_id == source_id
            assert isinstance(ref.metadata, dict)
        seen.add(source_id)
    assert seen == set(sources)
