"""Stage 6 tests: end-to-end resolution, audit trail, manual aliases, determinism."""

from __future__ import annotations

import json
from pathlib import Path

from er_support import candidates, er_config, mentions

from fmd.config.models import ModelsConfig
from fmd.er.config import ERConfig
from fmd.er.decide import AdjudicationPair, build_prompt, pair_hash
from fmd.er.service import resolve, resolve_with_audit
from fmd.er.types import Alias, EntityCandidate, Mention
from fmd.llm import MockProvider, ModelRouter, prompt_hash


def _config() -> ERConfig:
    return er_config(
        blocking={"backend": "deterministic"},
        similarity={"use_embeddings": False},
    )


def _models(tmp_path: Path) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "providers": {"mock": {"kind": "replay", "fixtures_dir": str(tmp_path)}},
            "profiles": {
                "entity_adjudication": {
                    "provider": "mock",
                    "model": "recorded-model",
                    "temperature": 0,
                    "guided_json": True,
                    "max_tokens": 1024,
                }
            },
            "budgets": {"daily_usd": 0, "per_run_usd": 0, "hard_stop": True, "cache": True},
            "fallbacks": [],
        }
    )


def _router(tmp_path: Path) -> ModelRouter:
    models = _models(tmp_path)
    return ModelRouter(models, providers={"mock": MockProvider("mock", models.providers["mock"])})


def _record_prompt(tmp_path: Path, decisions: list[dict], config: ERConfig) -> None:
    pairs = [
        AdjudicationPair(
            pair_id=item["pair_id"],
            mention_name=item["mention_name"],
            candidate_name=item["candidate_name"],
            score=item["score"],
            mention_state=item.get("mention_state"),
            mention_city=item.get("mention_city"),
            mention_pincode=item.get("mention_pincode"),
            candidate_state=item.get("candidate_state"),
            candidate_city=item.get("candidate_city"),
            candidate_pincode=item.get("candidate_pincode"),
        )
        for item in decisions
    ]
    system, user = build_prompt(pairs, config.adjudication)
    payload = {"content": {"decisions": decisions}, "model": "recorded-model"}
    (tmp_path / f"{prompt_hash(system, user)}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _adjudication_decisions(batch, config: ERConfig) -> list[dict]:
    decisions: list[dict] = []
    for resolution in batch.resolutions:
        audit = resolution.audit
        for item in resolution.candidates:
            if item.band != "adjudicate":
                continue
            pair_id = pair_hash(
                resolution.normalised.text,
                item.candidate.canonical_name,
                salt=config.version,
                state=item.candidate.state or resolution.mention.state,
                pincode=item.candidate.pincode or resolution.mention.pincode,
            )
            if pair_id not in audit.adjudication_pairs:
                continue
            verdict = "unsure"
            if "Medcal" in resolution.mention.text:
                verdict = "same"
            elif item.candidate.canonical_name.endswith("Medical Technology"):
                verdict = "different"
            decisions.append(
                {
                    "pair_id": pair_id,
                    "mention_name": resolution.mention.text,
                    "candidate_name": item.candidate.canonical_name,
                    "score": item.score,
                    "mention_state": resolution.mention.state,
                    "mention_city": resolution.mention.city,
                    "mention_pincode": resolution.mention.pincode,
                    "candidate_state": item.candidate.state,
                    "candidate_city": item.candidate.city,
                    "candidate_pincode": item.candidate.pincode,
                    "verdict": verdict,
                    "reason": "fixture verdict",
                    "evidence": resolution.normalised.text.split(),
                }
            )
    return decisions


def test_resolution_accepts_near_duplicates_and_keeps_distinct_institutions(tmp_path: Path) -> None:
    config = _config()
    first_pass = resolve_with_audit(mentions(), candidates(), config=config, embedder=None)
    decisions = _adjudication_decisions(first_pass, config)
    assert decisions, "expected at least one adjudicated pair"
    _record_prompt(tmp_path, decisions, config)

    batch = resolve_with_audit(mentions(), candidates(), config=config, router=_router(tmp_path))
    by_text = {resolution.mention.text: resolution for resolution in batch.resolutions}
    assert len(batch.resolutions) == len(mentions())

    expected = {
        "King George's Medical University": 1,
        "King George Medical University": 1,
        "KGMU": 1,
        "University of Delhi": 2,
        "King George Medcal University": 1,
        "Government Medical College, Patna": 4,
        "Rajendra Institute of Medical Sciences": 6,
        "Sarojini Naidu Medical College": 8,
    }
    for text, entity_id in expected.items():
        resolution = by_text[text]
        assert resolution.chosen is not None, (text, resolution.audit.notes)
        assert resolution.chosen.entity_id == entity_id, text

    # Distinct same-state institution: present as a candidate, never chosen.
    rajendra = by_text["Rajendra Institute of Medical Sciences"]
    assert rajendra.chosen is not None
    assert rajendra.chosen.entity_id != 7
    assert any(item.candidate.entity_id == 7 for item in rajendra.candidates)
    patna = by_text["Government Medical College, Patna"]
    assert patna.chosen is not None
    assert patna.chosen.entity_id == 4
    assert all(item.candidate.entity_id != 5 for item in patna.candidates if item.band == "accept")

    # Unknown place: no merge.
    bhopal = by_text["All India Institute of Medical Sciences, Bhopal"]
    assert bhopal.chosen is None
    assert bhopal.band == "reject"

    # The typo needed the model, and the audit says so.
    typo = by_text["King George Medcal University"]
    assert typo.audit.decided_by == "model"
    assert typo.audit.model_verdict is not None
    assert typo.audit.model_verdict.verdict == "same"


def test_audit_records_features_band_and_blocking(tmp_path: Path) -> None:
    config = _config()
    first_pass = resolve_with_audit(mentions(), candidates(), config=config)
    _record_prompt(tmp_path, _adjudication_decisions(first_pass, config), config)
    batch = resolve_with_audit(mentions(), candidates(), config=config, router=_router(tmp_path))
    resolution = next(item for item in batch.resolutions if item.mention.text == "KGMU")
    audit = resolution.audit
    assert audit.blocking_backend == "deterministic"
    assert audit.blocking_keys
    assert audit.thresholds == {"high": config.thresholds.high, "low": config.thresholds.low}
    assert audit.config_version == config.version
    assert audit.band == "accept"
    assert audit.decided_by == "score"
    assert audit.features["token_jaccard"] == 1.0
    assert resolution.candidates[0].score == resolution.score


def test_resolution_without_a_router_leaves_ambiguous_pairs_for_review() -> None:
    config = _config()
    resolutions = resolve(mentions(), candidates(), config=config)
    typo = next(
        item for item in resolutions if item.mention.text == "King George Medcal University"
    )
    assert typo.band == "adjudicate"
    assert typo.chosen is None
    assert typo.audit.decided_by == "none"
    assert "awaiting_adjudication" in typo.audit.notes


def test_resolution_is_deterministic(tmp_path: Path) -> None:
    config = _config()
    first_pass = resolve_with_audit(mentions(), candidates(), config=config)
    _record_prompt(tmp_path, _adjudication_decisions(first_pass, config), config)
    first = resolve_with_audit(mentions(), candidates(), config=config, router=_router(tmp_path))
    second = resolve_with_audit(mentions(), candidates(), config=config, router=_router(tmp_path))
    assert first.model_dump() == second.model_dump()


def _locked_candidate() -> list[EntityCandidate]:
    return [
        EntityCandidate(
            entity_id=50,
            canonical_name="Delhi University",
            state="Delhi",
            aliases=[Alias(alias="Delhi Vishwavidyalaya", alias_type="manual")],
        )
    ]


def test_manual_alias_is_authoritative() -> None:
    config = _config()
    resolutions = resolve(
        [
            Mention(text="Delhi Vishwavidyalaya", state="Delhi"),
            Mention(text="Delhi University", state="Delhi"),
        ],
        _locked_candidate(),
        config=config,
    )
    manual_match, locked_exact = resolutions
    assert manual_match.chosen is not None
    assert manual_match.audit.decided_by == "manual"
    assert "manual_alias_authoritative" in manual_match.audit.notes
    # A manually confirmed canonical name is never auto-merged, even on an exact
    # name match; a human must confirm.
    assert locked_exact.chosen is None
    assert locked_exact.band == "adjudicate"
    assert "manual_canonical_lock" in locked_exact.audit.notes


def test_manual_lock_declines_a_model_accept(tmp_path: Path) -> None:
    config = _config()
    mention = Mention(text="Delhi Vishwavidyalaya Society", state="Delhi")
    locked = _locked_candidate()
    first_pass = resolve_with_audit([mention], locked, config=config)
    pair_id = first_pass.resolutions[0].audit.adjudication_pairs[0]
    _record_prompt(
        tmp_path,
        [
            {
                "pair_id": pair_id,
                "mention_name": mention.text,
                "candidate_name": locked[0].canonical_name,
                "score": first_pass.resolutions[0].score,
                "mention_state": mention.state,
                "candidate_state": locked[0].state,
                "verdict": "same",
                "reason": "fixture",
                "evidence": ["delhi"],
            }
        ],
        config,
    )
    batch = resolve_with_audit([mention], locked, config=config, router=_router(tmp_path))
    resolution = batch.resolutions[0]
    assert resolution.chosen is None
    assert resolution.band == "adjudicate"
    assert resolution.audit.model_verdict is not None
    assert any(note.startswith("manual_canonical_lock") for note in resolution.audit.notes)
