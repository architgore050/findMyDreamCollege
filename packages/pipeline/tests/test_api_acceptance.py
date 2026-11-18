"""Unit tests for scripts/acceptance.sh assertion helpers (fmd.acceptance).

These run with no database, no API server and no model: the checks are pure
functions over parsed JSON / HTML text, plus the CLI exit-code contract the
shell script relies on.
"""

from __future__ import annotations

import json

import pytest

from fmd.acceptance import (
    cost_respects_profile_failures,
    envelope_failures,
    filter_reason_failures,
    html_artifact_failures,
    main,
    money_method_failures,
    privacy_failures,
    strip_script_style,
    unknown_render_failures,
)

GOOD_ENVELOPE = {
    "scorer_version": "scorer-2026.11",
    "weights_version": "weights-2026.11",
    "claims_hash": "a" * 64,
    "generated_at": "2026-11-18T04:00:00+00:00",
}


# ----------------------------------------------------------------- envelope


def test_complete_envelope_passes() -> None:
    assert envelope_failures(dict(GOOD_ENVELOPE, payload=True)) == []


def test_missing_envelope_keys_are_named() -> None:
    problems = envelope_failures({"scorer_version": "x"})
    assert any("weights_version" in problem for problem in problems)
    assert any("claims_hash" in problem for problem in problems)
    assert any("generated_at" in problem for problem in problems)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"claims_hash": "short"}, "claims_hash"),
        ({"claims_hash": "Z" * 64}, "claims_hash"),
        ({"generated_at": "not-a-date"}, "generated_at"),
        ({"scorer_version": ""}, "scorer_version"),
    ],
)
def test_malformed_envelope_values_fail(override: dict, fragment: str) -> None:
    payload = {**GOOD_ENVELOPE, **override}
    problems = envelope_failures(payload)
    assert any(fragment in problem for problem in problems)


def test_non_object_response_fails() -> None:
    assert envelope_failures([1, 2, 3]) == ["response is not a JSON object"]


# ----------------------------------------------------------------- unknown


def test_unknown_point_zero_fails() -> None:
    payload = {
        "sub_scores": {"outcomes": {"point": 0, "unknown_reason": "no claims", "lo": 0, "hi": 100}}
    }
    problems = unknown_render_failures(payload)
    assert len(problems) == 1
    assert "outcomes" in problems[0]
    assert "never 0 or 50" in problems[0]


def test_unknown_point_fifty_fails() -> None:
    payload = {"point": 50, "measured": False}
    assert unknown_render_failures(payload)


def test_unknown_with_null_point_and_full_interval_passes() -> None:
    payload = {"point": None, "lo": 0, "hi": 100, "unknown_reason": "no claims"}
    assert unknown_render_failures(payload) == []


def test_measured_zero_with_claims_passes() -> None:
    payload = {"point": 0, "n_claims": 3, "measured": True, "unknown_reason": None}
    assert unknown_render_failures(payload) == []


def test_unknown_scan_walks_nested_lists() -> None:
    payload = {"items": [{"score": {"point": 50, "n_claims": 0}}]}
    problems = unknown_render_failures(payload)
    assert len(problems) == 1
    assert "items[0].score" in problems[0]


# ----------------------------------------------------------------- money


def _tcoa(**overrides: object) -> dict:
    block = {
        "value": 400000,
        "lo": 380000,
        "hi": 420000,
        "method": "sum_of_items",
        "currency": "INR",
    }
    block.update(overrides)  # type: ignore[arg-type]
    return {"scenario_totals": {"hostel": {"tcoa": block}}}


def test_complete_money_block_passes() -> None:
    assert money_method_failures(_tcoa()) == []


def test_money_value_without_method_fails() -> None:
    problems = money_method_failures(_tcoa(method=None))
    assert any("no method" in problem for problem in problems)


def test_partial_range_fails() -> None:
    problems = money_method_failures(_tcoa(hi=None))
    assert any("needs both lo and hi" in problem for problem in problems)


def test_inverted_bounds_fail() -> None:
    problems = money_method_failures(_tcoa(lo=500000, hi=400000))
    assert any("inverted bounds" in problem for problem in problems)


def test_bare_scalar_money_fails() -> None:
    problems = money_method_failures({"tcoa": 400000})
    assert any("bare scalar" in problem for problem in problems)


def test_unknown_money_without_method_is_honest() -> None:
    # UNKNOWN money is a null value with a reason — not a method violation.
    assert money_method_failures(_tcoa(value=None, lo=None, hi=None, method=None)) == []


def test_salary_key_requires_method() -> None:
    problems = money_method_failures({"p25_first_year_salary": {"value": 300000}})
    assert any("no method" in problem for problem in problems)


def test_raw_cost_evidence_rows_are_not_method_gated() -> None:
    # cost_items rows carry provenance (source_claim_id), not a method; the
    # rule targets computed money blocks only.
    payload = {"costs": [{"category": "tuition", "amount": "100000.00", "currency": "INR"}]}
    assert money_method_failures(payload) == []


def test_methods_map_values_are_not_money_blocks() -> None:
    # CostSnapshot.methods is a name -> method-string map; its members are the
    # method names the blocks themselves must carry, never money values.
    payload = {
        "scenario_totals": {"hostel": {"methods": {"tcoa": "modelled", "net_price": "computed"}}}
    }
    assert money_method_failures(payload) == []


def test_display_map_strings_are_not_money_blocks() -> None:
    # CostSnapshot.display holds pre-rendered strings that already embed the
    # method; the structured blocks beside them carry the machine-readable one.
    payload = {
        "scenario_totals": {
            "hostel": {
                "display": {"tcoa": "INR 920,000 (896,000-944,000, modelled)"},
                "tcoa": {"value": 920000, "lo": 896000, "hi": 944000, "method": "modelled"},
            }
        }
    }
    assert money_method_failures(payload) == []


def test_ratio_block_without_method_fails() -> None:
    # DSR₂₅/affordability blocks publish method on themselves (the cost model
    # sets it); a block that loses it is a contract violation.
    payload = {
        "scenario_totals": {
            "hostel": {
                "dsr25": {
                    "value": 0.57,
                    "band": "manageable",
                    "formula": "first_year_emi / p25_first_year_gross_salary",
                    "first_year_emi": 170000.0,
                    "p25_first_year_salary": 300000.0,
                    "bands": {"manageable": {"min": 0.2, "max": 0.3}},
                }
            }
        }
    }
    problems = money_method_failures(payload)
    assert len(problems) == 1
    assert "carries no method" in problems[0]


def test_ratio_block_with_method_and_anchor_scalars_passes() -> None:
    # The scalar anchors inside a ratio block are contract-defined numbers
    # (dsr25Schema pins them); only the block-level method is enforced.
    payload = {
        "scenario_totals": {
            "hostel": {
                "dsr25": {
                    "value": 0.57,
                    "band": "manageable",
                    "formula": "first_year_emi / p25_first_year_gross_salary",
                    "first_year_emi": 170000.0,
                    "p25_first_year_salary": 300000.0,
                    "bands": {"manageable": {"min": 0.2, "max": 0.3}},
                    "method": "computed_from_emi_and_p25",
                },
                "affordability_gap": {
                    "value": 1.2,
                    "flagged": True,
                    "flag_above": 1.0,
                    "formula": "family_out_of_pocket / annual_family_income",
                    "family_out_of_pocket": 720000.0,
                    "annual_family_income": 600000.0,
                    "method": "computed_from_oop_and_income",
                },
            }
        }
    }
    assert money_method_failures(payload) == []


# ----------------------------------------------------------------- filters


def test_unknown_filter_without_reason_fails() -> None:
    payload = {
        "filters": [{"filter_id": "exam_registration_open", "result": "unknown", "reason": ""}]
    }
    problems = filter_reason_failures(payload)
    assert len(problems) == 1
    assert "exam_registration_open" in problems[0]


def test_fail_filter_without_reason_fails() -> None:
    payload = {"filters": [{"filter_id": "budget_ceiling", "result": "fail"}]}
    assert any("without a reason" in problem for problem in filter_reason_failures(payload))


def test_pass_filter_needs_no_reason() -> None:
    payload = {"filters": [{"filter_id": "budget_ceiling", "result": "pass", "reason": ""}]}
    assert filter_reason_failures(payload) == []


def test_invalid_filter_result_fails() -> None:
    payload = {"filters": [{"filter_id": "exam_travel", "result": "maybe"}]}
    assert any("result" in problem for problem in filter_reason_failures(payload))


def test_rankings_decision_map_is_scanned() -> None:
    payload = {
        "items": [
            {
                "filters": {
                    "decisions": {
                        "credential_level": {
                            "filter_id": "credential_level",
                            "result": "unknown",
                            "reason": "no claims",
                        },
                        "eligibility": {
                            "filter_id": "eligibility",
                            "result": "unknown",
                            "reason": "  ",
                        },
                    }
                }
            }
        ]
    }
    problems = filter_reason_failures(payload)
    assert len(problems) == 1
    assert "eligibility" in problems[0]


# ----------------------------------------------------------------- cost/profile


def test_loan_free_profile_with_positive_principal_fails() -> None:
    payload = {
        "available": True,
        "scenario_totals": {
            "hostel": {
                "loan_principal": {
                    "value": 1200000,
                    "lo": 1200000,
                    "hi": 1200000,
                    "method": "modelled",
                },
                "debt_at_repayment": {
                    "value": 1300000,
                    "lo": 1300000,
                    "hi": 1300000,
                    "method": "modelled",
                },
            }
        },
    }
    problems = cost_respects_profile_failures(payload, loan_required=False, family_income=None)
    assert len(problems) == 2
    assert all("loan_required=false" in problem for problem in problems)


def test_loan_free_profile_with_zero_principal_passes() -> None:
    payload = {
        "available": True,
        "scenario_totals": {
            "hostel": {
                "loan_principal": {"value": 0, "method": "not_required"},
                "debt_at_repayment": {"value": 0, "method": "not_required"},
            }
        },
    }
    assert cost_respects_profile_failures(payload, loan_required=False, family_income=None) == []


def test_family_income_without_gap_fails() -> None:
    payload = {
        "available": True,
        "scenario_totals": {"hostel": {"affordability_gap": {"value": None, "lo": 0, "hi": 0}}},
    }
    problems = cost_respects_profile_failures(payload, loan_required=None, family_income=600000.0)
    assert len(problems) == 1
    assert "annual_family_income" in problems[0]


def test_family_income_with_gap_passes() -> None:
    payload = {
        "available": True,
        "scenario_totals": {"hostel": {"affordability_gap": {"value": 1.2, "lo": 0.8, "hi": 2.0}}},
    }
    assert cost_respects_profile_failures(payload, loan_required=None, family_income=600000.0) == []


def test_unavailable_snapshot_must_say_why() -> None:
    assert cost_respects_profile_failures({"available": False, "reason": "no cost evidence"}) == []
    problems = cost_respects_profile_failures({"available": False})
    assert problems == ["cost snapshot unavailable but no reason given"]


def test_available_snapshot_without_totals_fails() -> None:
    assert cost_respects_profile_failures({"available": True, "scenario_totals": {}}) == [
        "available cost payload carries no scenario_totals"
    ]


# ----------------------------------------------------------------- html


def test_clean_html_passes() -> None:
    html = "<html><body><p>composite band 40–60 · fee ₹1,20,000</p></body></html>"
    assert html_artifact_failures(html) == []


def test_nan_in_markup_fails() -> None:
    assert any("NaN" in problem for problem in html_artifact_failures("<p>fee NaN</p>"))


def test_undefined_in_markup_fails() -> None:
    assert any("undefined" in problem for problem in html_artifact_failures("<p>undefined</p>"))


def test_artifacts_inside_script_are_ignored() -> None:
    html = "<p>ok</p><script>var x = NaN; if (typeof y === 'undefined') {}</script>"
    assert html_artifact_failures(html) == []


def test_strip_script_style_removes_only_those_blocks() -> None:
    html = "<style>p{color:red}</style><script>NaN</script><p>kept</p>"
    assert "kept" in strip_script_style(html)
    assert "NaN" not in strip_script_style(html)
    assert "color" not in strip_script_style(html)


def test_bare_composite_decimal_fails() -> None:
    html = "<p>composite 62.5</p>"
    problems = html_artifact_failures(html)
    assert len(problems) == 1
    assert "band" in problems[0]


def test_composite_word_without_decimal_passes() -> None:
    assert html_artifact_failures("<p>composite band 40–60</p>") == []


def test_html_comments_are_stripped() -> None:
    assert html_artifact_failures("<!-- NaN in a comment -->") == []


def test_object_object_artifact_fails() -> None:
    assert html_artifact_failures("<p>[object Object]</p>")


# ----------------------------------------------------------------- privacy


def test_sentinel_leak_fails() -> None:
    sentinel = "fmd-acceptance-sentinel-12345"
    problems = privacy_failures(f"{'x' * 10}{sentinel}{'y' * 10}", sentinel)
    assert len(problems) == 1
    assert "leaked" in problems[0]


def test_clean_output_passes() -> None:
    assert privacy_failures("no profile data here", "fmd-acceptance-sentinel-12345") == []


def test_too_short_sentinel_is_refused() -> None:
    assert privacy_failures("anything", "abc") == [
        "privacy sentinel is missing or too short to be meaningful"
    ]


# ----------------------------------------------------------------- CLI


@pytest.fixture()
def json_file(tmp_path):
    def _write(payload: dict) -> str:
        path = tmp_path / "body.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    return _write


def test_cli_envelope_ok_and_fail(json_file, capsys) -> None:
    assert main(["envelope", json_file(GOOD_ENVELOPE)]) == 0
    assert "ok: envelope" in capsys.readouterr().out

    bad = dict(GOOD_ENVELOPE)
    del bad["claims_hash"]
    assert main(["envelope", json_file(bad)]) == 1
    assert "FAIL:" in capsys.readouterr().out


def test_cli_money_check_exit_codes(json_file, capsys) -> None:
    assert main(["money", json_file({"tcoa": {"value": 1, "method": "m", "lo": 0, "hi": 1}})]) == 0
    assert main(["money", json_file({"tcoa": {"value": 1}})]) == 1
    assert "no method" in capsys.readouterr().out


def test_cli_cost_profile_flags(json_file, capsys) -> None:
    payload = {
        "available": True,
        "scenario_totals": {"hostel": {"loan_principal": {"value": 999, "method": "m"}}},
    }
    assert main(["cost-profile", "--loan-required", "false", json_file(payload)]) == 1
    assert "loan_required=false" in capsys.readouterr().out


def test_cli_html_reads_file(tmp_path, capsys) -> None:
    page = tmp_path / "page.html"
    page.write_text("<p>composite 62.5</p>", encoding="utf-8")
    assert main(["html", str(page)]) == 1
    assert "composite" in capsys.readouterr().out


def test_cli_privacy_sentinel_flag(tmp_path, capsys) -> None:
    body = tmp_path / "body.txt"
    body.write_text("leak: fmd-acceptance-sentinel-777", encoding="utf-8")
    assert main(["privacy", "--sentinel", "fmd-acceptance-sentinel-777", str(body)]) == 1
    assert "leaked" in capsys.readouterr().out


def test_cli_invalid_json_fails(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["envelope", str(bad)]) == 1
    assert "not valid JSON" in capsys.readouterr().out


def test_cli_unknown_check_exits_two(capsys) -> None:
    assert main(["no-such-check"]) == 2
    assert "unknown acceptance check" in capsys.readouterr().err


def test_cli_usage_without_args(capsys) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err
