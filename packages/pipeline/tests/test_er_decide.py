"""Stage 4 tests: bands, bundled adjudication, MockProvider, caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fmd.config.models import ModelsConfig
from fmd.er.config import ERConfig, Thresholds
from fmd.er.decide import (
    AdjudicationCache,
    AdjudicationPair,
    adjudicate,
    band_for,
    build_prompt,
    pair_hash,
)
from fmd.llm import MockProvider, ModelRouter, prompt_hash


def _pair(pair_id: str, mention: str, candidate: str, score: float = 0.6) -> AdjudicationPair:
    return AdjudicationPair(
        pair_id=pair_id, mention_name=mention, candidate_name=candidate, score=score
    )


def _models(
    tmp_path: Path, *, purpose: str = "entity_adjudication", model: str | None = "recorded-model"
):
    return ModelsConfig.model_validate(
        {
            "providers": {"mock": {"kind": "replay", "fixtures_dir": str(tmp_path)}},
            "profiles": {
                purpose: {
                    "provider": "mock",
                    "model": model,
                    "temperature": 0,
                    "guided_json": True,
                    "max_tokens": 512,
                }
            },
            "budgets": {"daily_usd": 0, "per_run_usd": 0, "hard_stop": True, "cache": True},
            "fallbacks": [],
        }
    )


def _router(models: ModelsConfig) -> ModelRouter:
    return ModelRouter(models, providers={"mock": MockProvider("mock", models.providers["mock"])})


def _record(
    tmp_path: Path,
    pairs: list[AdjudicationPair],
    decisions: list[dict[str, Any]],
    *,
    config: ERConfig | None = None,
) -> str:
    cfg = (config or ERConfig()).adjudication
    system, user = build_prompt(pairs, cfg)
    digest = prompt_hash(system, user)
    payload = {
        "content": {"decisions": decisions},
        "tokens_in": 10,
        "tokens_out": 5,
        "model": "recorded-model",
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{digest}.json").write_text(json.dumps(payload), encoding="utf-8")
    return digest


def test_bands_follow_injected_thresholds() -> None:
    thresholds = Thresholds(high=0.8, low=0.3)
    assert band_for(0.8, thresholds) == "accept"
    assert band_for(0.95, thresholds) == "accept"
    assert band_for(0.3, thresholds) == "reject"
    assert band_for(0.2, thresholds) == "reject"
    assert band_for(0.5, thresholds) == "adjudicate"


def test_pair_hash_is_order_insensitive_and_context_aware() -> None:
    first = pair_hash("Delhi University", "University of Delhi", salt="er-test")
    second = pair_hash("University of Delhi", "Delhi University", salt="er-test")
    assert first == second
    assert first != pair_hash("Delhi University", "University of Delhi", salt="er-v2")
    assert first != pair_hash("Delhi University", "University of Delhi", state="Bihar")


def test_adjudication_uses_mock_provider_and_caches(tmp_path: Path) -> None:
    config = ERConfig()
    pairs = [
        _pair("pair-a", "King George Medcal University", "King George's Medical University"),
        _pair(
            "pair-b",
            "Rajendra Institute of Medical Sciences",
            "Rajendra Institute of Medical Technology",
        ),
    ]
    _record(
        tmp_path,
        pairs,
        [
            {
                "pair_id": "pair-a",
                "verdict": "same",
                "reason": "single-letter typo",
                "evidence": ["king", "george", "medical"],
            },
            {
                "pair_id": "pair-b",
                "verdict": "different",
                "reason": "sciences vs technology",
                "evidence": ["rajendra", "medical"],
            },
        ],
        config=config,
    )
    cache = AdjudicationCache()
    outcome = adjudicate(pairs, config=config, router=_router(_models(tmp_path)), cache=cache)
    assert outcome.verdicts["pair-a"].verdict == "same"
    assert outcome.verdicts["pair-b"].verdict == "different"
    assert outcome.bundles == 1
    assert len(cache) == 2
    assert outcome.verdicts["pair-a"].cached is False

    again = adjudicate(pairs, config=config, router=_router(_models(tmp_path)), cache=cache)
    assert again.cache_hits == 2
    assert again.bundles == 0
    assert again.verdicts["pair-a"].cached is True


def test_bundling_respects_config_size(tmp_path: Path) -> None:
    config = ERConfig.from_mapping(
        {
            "adjudication": {"bundle_size": 1},
            "blocking": {"backend": "deterministic"},
        }
    )
    pairs = [_pair("a", "Alpha", "Alfa"), _pair("b", "Beta", "Betta")]
    _record(tmp_path, [pairs[0]], [{"pair_id": "a", "verdict": "same"}], config=config)
    _record(tmp_path, [pairs[1]], [{"pair_id": "b", "verdict": "different"}], config=config)
    outcome = adjudicate(pairs, config=config, router=_router(_models(tmp_path)))
    assert outcome.bundles == 2
    assert outcome.verdicts["a"].verdict == "same"
    assert outcome.verdicts["b"].verdict == "different"


def test_omitted_pair_becomes_unsure(tmp_path: Path) -> None:
    config = ERConfig()
    pairs = [_pair("only", "Alpha", "Alfa")]
    _record(tmp_path, pairs, [], config=config)
    outcome = adjudicate(pairs, config=config, router=_router(_models(tmp_path)))
    assert outcome.verdicts["only"].verdict == "unsure"
    assert outcome.verdicts["only"].reason


def test_no_router_leaves_pairs_for_manual_review() -> None:
    outcome = adjudicate([_pair("x", "Alpha", "Alfa")], config=ERConfig(), router=None)
    assert outcome.verdicts == {}
    assert outcome.notes


def test_unresolved_model_profile_degrades_instead_of_raising() -> None:
    config = ERConfig()
    outcome = adjudicate(
        [_pair("x", "Alpha", "Alfa")],
        config=config,
        router=_router(_models(Path("/nonexistent"), model=None)),
    )
    assert outcome.verdicts == {}
    assert any("model unavailable" in note for note in outcome.notes)
