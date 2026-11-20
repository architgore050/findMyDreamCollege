"""Bake-off harness tests (WS-MODELDOC).

Everything here is offline: a fake provider replays fixed guided JSON, and the
production `validate_document` derives values, units, and spans from the
quote-first payload. These tests pin the bake-off's table, gates, and
ground-truth exclusions without touching a network or a real model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from fmd.config.loader import interpolate_env
from fmd.config.models import ModelsConfig
from fmd.llm.doctor import doctor_report
from fmd.llm.providers import ProviderError, ProviderResult

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bakeoff.py"


def _load_bakeoff() -> Any:
    spec = importlib.util.spec_from_file_location("fmd_bakeoff_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bakeoff = _load_bakeoff()

TEXT = (
    "Batch 2024-25. Eligible students: 120. Students placed: 96. "
    "Placement rate: 80%. Median salary: INR 420000 per annum. "
    "Internships are included in the placement count."
)

CONTENT = {
    "cohort_year": {"quote": "2024-25", "confidence": 0.95},
    "eligible_count": {"quote": "Eligible students: 120", "confidence": 0.95},
    "placed_count": {"quote": "Students placed: 96", "confidence": 0.95},
    "placement_rate": {"quote": "Placement rate: 80%", "confidence": 0.9},
    "median_salary": {"quote": "Median salary: INR 420000 per annum", "confidence": 0.93},
    "highest_package": {"quote": None, "confidence": None},
    "internship_counted_as_employment": {
        "quote": "Internships are included in the placement count.",
        "confidence": 0.88,
    },
    "outcome_basis": {
        "quote": "Internships are included in the placement count.",
        "confidence": 0.7,
    },
}

class FakeProvider:
    kind = "replay"
    supports_guided_json = False

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.config = SimpleNamespace(supports_guided_json=False)
        self.name = "mock"
        self.requests = 0

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type,
        profile: Any,
        prompt_sha256: str,
        purpose: str,
    ) -> ProviderResult:
        self.requests += 1
        if self.fail:
            raise ProviderError("json_schema_violation", "fake schema failure")
        return ProviderResult(
            content=CONTENT,
            raw=json.dumps(CONTENT),
            model=model,
            provider="mock",
            tokens_in=1000,
            tokens_out=250,
        )

    def probe(self) -> tuple[bool, str]:
        return True, "fake"


def _models() -> ModelsConfig:
    raw = yaml.safe_load((REPO_ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    return ModelsConfig.model_validate(interpolate_env(raw, {}))


def _ground_truth(
    tmp_path: Path, *, only_unverified: bool = False, eligible_value: int = 120
) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc-1.txt").write_text(TEXT, encoding="utf-8")
    expected: dict[str, Any] = {
        "placement.cohort_year": {
            "value": "2024-25",
            "quote": "2024-25",
            "status": "verified",
        },
        "placement.eligible_count": {
            "value": eligible_value,
            "unit": "count",
            "quote": "Eligible students: 120",
            "status": "verified",
        },
        "placement.placed_count": {
            "value": 96,
            "unit": "count",
            "quote": "Students placed: 96",
            "status": "verified",
        },
        "placement.rate": {
            "value": 0.8,
            "unit": "ratio",
            "quote": "Placement rate: 80%",
            "status": "verified",
        },
        "salary.median": {
            "value": 420000,
            "unit": "INR",
            "quote": "Median salary: INR 420000 per annum",
            "status": "verified",
        },
        "placement.internship_counted_as_employment": {
            "value": True,
            "quote": "Internships are included in the placement count.",
            "status": "verified",
        },
        "placement.basis": {
            "value": "included in the placement count",
            "quote": "Internships are included in the placement count.",
            "status": "verified",
        },
        "salary.highest_package": {
            "value": None,
            "quote": None,
            "status": "verified",
            "expectation": "abstain",
            "absent_terms": ["highest package", "highest salary"],
        },
        "fee.tuition": {
            "value": None,
            "unit": "INR",
            "quote": None,
            "status": "unverified",
            "note": "No fee is printed in the fixture. A human must find a dated source.",
        },
    }
    if only_unverified:
        expected = {"fee.tuition": expected["fee.tuition"]}
    entry = {
        "schema_version": 1,
        "id": "doc-1",
        "institution": "Fixture Institute",
        "programme": "BSc Medical Laboratory Technology",
        "fixture": "docs/doc-1.txt",
        "family": "placement_report",
        "expected": expected,
    }
    path = tmp_path / "golden" / "doc-1.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(entry, sort_keys=False), encoding="utf-8")
    return path


def _run(
    tmp_path: Path, *, eligible_value: int = 120, fail: bool = False
) -> Any:
    ground_truth = _ground_truth(tmp_path, eligible_value=eligible_value)
    candidate = bakeoff.Candidate(
        name="fake-9b",
        provider="mock",
        model="fake-9b",
        input_per_mtok=0.10,
        output_per_mtok=0.15,
        currency="USD",
        price_as_of="2026-11-16",
    )
    reports = bakeoff.run_bakeoff(
        tmp_path / "docs",
        ground_truth,
        [candidate],
        models=_models(),
        providers={"mock": FakeProvider(fail=fail)},
        root=tmp_path,
    )
    return reports[0]


def test_bakeoff_table_renders_fake_candidate_and_fixture_ground_truth(tmp_path: Path) -> None:
    report = _run(tmp_path)

    assert report.candidate.name == "fake-9b"
    assert report.entries == 1
    assert report.attempted == 1
    assert report.schema_valid == 1
    assert report.pages == 1
    assert report.metrics.evaluated == 8
    assert report.metrics.precision == 1.0
    assert report.fuzzy_precision == 1.0
    assert report.catastrophic_rate == 0.0
    assert report.null_precision == 1.0
    assert report.null_recall == 1.0
    assert report.schema_valid_rate == 1.0
    assert report.cost_per_page is not None and report.cost_per_page > 0
    assert report.p50_latency_ms is not None
    assert report.p95_latency_ms is not None
    assert report.gates == {"span_exact": True, "catastrophic": True, "null_precision": True}
    assert report.passed is True

    table = bakeoff.render_table([report])
    assert "candidate" in table
    assert "span-exact" in table
    assert "100.00%" in table
    assert "PASS" in table
    assert "USD 0.000" in table
    assert "n/a" not in table.splitlines()[2]


def test_hard_gate_fail_path_is_reported_not_hidden(tmp_path: Path) -> None:
    report = _run(tmp_path, eligible_value=999)

    assert report.metrics.precision is not None
    assert report.metrics.precision < 0.98
    assert report.catastrophic_rate is not None
    assert report.catastrophic_rate > 0.005
    assert report.gates["span_exact"] is False
    assert report.gates["catastrophic"] is False
    assert report.passed is False
    assert "FAIL" in bakeoff.render_table([report])


def test_unverified_rows_are_counted_and_never_scored(tmp_path: Path) -> None:
    report = _run(tmp_path)

    assert report.unverified_excluded == 1
    assert report.metrics.evaluated == 8
    scored_fields = {result.field for result in report.metrics.per_field}
    assert "fee.tuition" not in scored_fields


def test_refuses_ground_truth_with_no_verified_rows(tmp_path: Path) -> None:
    ground_truth = _ground_truth(tmp_path, only_unverified=True)
    candidate = bakeoff.Candidate(name="fake", provider="mock", model="fake")
    with pytest.raises(bakeoff.BakeoffError, match="zero verified rows"):
        bakeoff.run_bakeoff(
            tmp_path / "docs",
            ground_truth,
            [candidate],
            models=_models(),
            providers={"mock": FakeProvider()},
            root=tmp_path,
        )


def test_schema_violation_is_reported_as_a_failed_attempt(tmp_path: Path) -> None:
    report = _run(tmp_path, fail=True)

    assert report.attempted == 1
    assert report.schema_valid == 0
    assert report.schema_valid_rate == 0.0
    assert report.passed is False
    assert any("json_schema_violation" in error for error in report.errors)


def test_models_config_defaults_and_every_profile_is_env_overridable() -> None:
    raw = yaml.safe_load((REPO_ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    defaults = ModelsConfig.model_validate(interpolate_env(raw, {}))

    assert defaults.profiles["bulk_extract"].provider == "local"
    assert defaults.profiles["bulk_extract"].model == "gemma-4-26b-a4b-gguf"
    assert defaults.profiles["bulk_extract"].context_tokens == 65536
    assert defaults.profiles["bulk_extract"].quantization == "int4"
    assert defaults.profiles["bulk_extract"].thinking is False
    assert defaults.profiles["bulk_extract"].guided_json is True
    # OCR has no router profile: the `ocr_vlm` profile was retired 2026-11-16 and
    # OCR now lives in config/acquire.yaml (T2/T2b backends, an HTTP stage).
    assert "ocr_vlm" not in defaults.profiles
    # `programme_equivalence` is the W9 Tier-2 semantic equivalence classifier
    # (config/programmes.yaml is its taxonomy of record).
    assert set(defaults.profiles) == {
        "bulk_extract",
        "hard_pdf",
        "adjudicate",
        "entity_adjudication",
        "nlp_rubric",
        "verify",
        "programme_equivalence",
        "embed",
    }
    # `hard_pdf` is the hosted fallback: provider `api` plus the shared
    # FMD_API_MODEL. Unset interpolates to None, so it degrades per document.
    assert defaults.profiles["hard_pdf"].provider == "api"
    assert defaults.profiles["hard_pdf"].model is None
    assert defaults.profiles["hard_pdf"].model_env == "FMD_API_MODEL"
    assert defaults.profiles["adjudicate"].provider == "gemini"
    assert defaults.profiles["adjudicate"].model == "gemini-2.5-flash"
    assert defaults.profiles["nlp_rubric"].model == "gemini-2.5-flash"
    assert defaults.profiles["embed"].model == "BAAI/bge-m3"

    for name, profile in defaults.profiles.items():
        assert profile.provider_env, f"{name} has no provider_env"
        assert profile.model_env, f"{name} has no model_env"
        assert profile.provider_env in raw["profiles"][name]["provider"]
        assert profile.model_env in raw["profiles"][name]["model"]

    env = {
        "FMD_EXTRACT_PROVIDER": "api",
        "FMD_EXTRACT_MODEL": "custom-bulk",
        "FMD_ADJUDICATE_PROVIDER": "anthropic",
        "FMD_ADJUDICATE_MODEL": "custom-adjudicate",
        "FMD_NLP_PROVIDER": "api",
        "FMD_NLP_MODEL": "custom-nlp",
    }
    overridden = ModelsConfig.model_validate(interpolate_env(raw, env))
    assert overridden.profiles["bulk_extract"].provider == "api"
    assert overridden.profiles["bulk_extract"].model == "custom-bulk"
    assert overridden.profiles["adjudicate"].provider == "anthropic"
    assert overridden.profiles["adjudicate"].model == "custom-adjudicate"
    assert overridden.profiles["nlp_rubric"].model == "custom-nlp"
    assert set(overridden.profiles) == set(defaults.profiles)


def test_doctor_never_prints_secret_values() -> None:
    raw = yaml.safe_load((REPO_ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    secret = "sk-test-do-not-print"
    models = ModelsConfig.model_validate(
        interpolate_env(
            raw,
            {
                "FMD_API_LLM_URL": "https://api.example.test/v1",
                "FMD_API_LLM_KEY": secret,
                "FMD_GEMINI_KEY": secret,
            },
        )
    )
    payload = json.dumps(doctor_report(models), sort_keys=True)
    assert secret not in payload
    assert "key_present" in payload
