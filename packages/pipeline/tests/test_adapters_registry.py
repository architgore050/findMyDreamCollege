from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmd.config import (
    ConfigError,
    RecognitionOriginError,
    load_registry,
    validate_recognition_origin,
)


def test_ugc_registry_matches_the_verified_json_xhr(registry) -> None:
    source = registry.sources.by_id("ugc_university_list")
    assert source.format == "api"
    assert source.seeds == []
    assert source.params.api_endpoint.endswith("/Getuniversity_details")
    assert source.params.unitype_param == "unitypeID"
    assert source.params.unitype_ids == [0, 1, 2, 3, 4]
    assert "uni_name" in source.params.field_whitelist
    assert "AIshe_code" in source.params.field_whitelist
    assert {"user_password", "user_username", "Nodal_Mob"} <= set(source.pii_fields)


def test_aishe_is_disabled_until_a_fetch_succeeds(registry) -> None:
    source = registry.sources.by_id("aishe_reports")
    assert source.enabled is False
    assert source.verify_first is True
    assert source.params.verified is False
    assert source.params.static_xlsx_url.endswith("/assets/excel/2023-24.xlsx")


def test_nabl_is_an_exclusion_not_a_source(registry) -> None:
    assert registry.sources.by_id("nabl_labs") is None
    exclusion = next(item for item in registry.sources.exclusions if item.id == "nabl_labs")
    assert "Disallow" in exclusion.reason


def test_dead_or_misleading_urls_are_corrected(registry) -> None:
    assert registry.sources.by_id("naac_results").base_url.startswith("https://naac.gov.in/")
    assert registry.sources.by_id("nabh_hospitals").base_url == "https://nabh.co/"
    assert registry.sources.by_id("nabh_hospitals").format == "json"
    assert registry.sources.by_id("ugc_consolidated_lists").enabled is False


def test_state_counselling_corrections(registry) -> None:
    mp = registry.sources.by_id("counselling_madhya_pradesh")
    up = registry.sources.by_id("counselling_uttar_pradesh")
    assert mp.base_url == "https://paramedical.mponline.gov.in/"
    assert up.base_url == "https://abvmuup.edu.in/admission-paramedical.php"
    assert "CAHET" in up.name
    assert registry.sources.by_id("counselling_telangana") is not None
    assert registry.sources.by_id("counselling_andhra_pradesh").enabled is False


def test_pay_anchor_is_level_4_and_scanned(registry) -> None:
    source = registry.sources.by_id("seventh_cpc_pay_matrix")
    assert source.level == 4
    assert source.scanned is True
    assert source.verify_first is True
    assert source.format == "pdf_scan"


def test_financing_sources_abstain(registry) -> None:
    for source_id in ("vidya_lakshmi", "csis_subsidy", "pm_vidyalaxmi"):
        source = registry.sources.by_id(source_id)
        assert source.abstain is True
        assert source.enabled is True  # attempt once, then abstain


def test_nirf_notes_record_the_ua_gate(registry) -> None:
    notes = registry.sources.by_id("nirf_rankings").notes or ""
    assert "CloudFront 403" in notes
    assert "institution-level" in notes


def test_impersonating_councils_are_denylisted(registry) -> None:
    domains = registry.sources.denied_domains()
    assert "statealliedandhealthcarecouncil.com" in domains
    assert "sahstc.com" in domains
    for url in (
        "https://statealliedandhealthcarecouncil.com/",
        "https://www.sahstc.com/about",
    ):
        with pytest.raises(RecognitionOriginError):
            validate_recognition_origin(url, domains)
    # A statutory NCAHP host is unaffected.
    validate_recognition_origin("https://ncahp.abdm.gov.in/StateCouncil", domains)


def test_loader_rejects_a_source_on_a_denylisted_domain(config_dir: Path) -> None:
    path = config_dir / "sources.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["sources"][0]["base_url"] = "https://www.sahstc.com/"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_registry(config_dir, env={})
    assert "denylisted_domains" in str(excinfo.value)


def test_ws_c3_adapters_are_registry_driven(registry) -> None:
    from fmd.adapters import KNOWN_ADAPTERS

    for source_id in (
        "institution_sites",
        "state_health_universities",
        "fee_regulators",
        "state_paramedical_councils",
        "nirf_rankings",
        "seventh_cpc_pay_matrix",
    ):
        source = registry.sources.by_id(source_id)
        assert source is not None, source_id
        assert source.adapter in KNOWN_ADAPTERS

    institutions = registry.sources.by_id("institution_sites")
    assert institutions.params.max_depth == 2
    assert institutions.params.max_pages_per_domain == 20
    assert institutions.params.document_filename_patterns

    nirf = registry.sources.by_id("nirf_rankings")
    assert len(nirf.params.listing_urls) == 3
    assert nirf.params.pdf_href_pattern

    pay = registry.sources.by_id("seventh_cpc_pay_matrix")
    assert pay.params.document_url == pay.base_url
    assert pay.params.verified is True
