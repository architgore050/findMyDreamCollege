"""Wave v1-w9: the trust surface — generated methodology + append-only exclusions.

PLAN #9 acceptance: (a) the methodology payload is GENERATED from
``config/`` at request time — the tests pin it against the loaded registry
so a hand-copied number fails; (b) the dispute path is part of the payload;
(c) the source-exclusion registry is append-only — an attempt to change an
existing row raises and the correction must carry a new id; (d) everything
here runs offline (TestClient + the committed registry) except the DB-gated
append tests.
"""

from __future__ import annotations

import pytest

from fmd.config import load_registry
from tests.conftest import CONFIG_DIR

REGISTRY = load_registry(CONFIG_DIR, env={})


# ----------------------------------------------------------- (a) + (b) methodology


def test_methodology_weights_are_generated_not_hand_copied() -> None:
    """The payload's weights equal the registry's — byte for byte."""
    from fastapi.testclient import TestClient

    from fmd.api import app

    client = TestClient(app)
    response = client.get("/api/methodology")
    assert response.status_code == 200
    body = response.json()
    methodology = body["methodology"]
    assert methodology["weights"] == {
        name: float(value) for name, value in REGISTRY.weights.weights.items()
    }
    assert methodology["weights_sum"] == pytest.approx(
        sum(float(v) for v in REGISTRY.weights.weights.values())
    )
    assert body["scorer_version"] == REGISTRY.scorer_version
    assert body["weights_version"] == REGISTRY.weights_version


def test_methodology_surfaces_policy_sources_and_dispute_path() -> None:
    from fastapi.testclient import TestClient

    from fmd.api import app

    client = TestClient(app)
    body = client.get("/api/methodology").json()
    methodology = body["methodology"]
    # policy numbers come from the same config object the scorer reads
    assert methodology["component_policy"]["min_components_for_composite"] == (
        REGISTRY.weights.min_components_for_composite
    )
    assert methodology["component_policy"]["min_claims_for_confidence"] == (
        REGISTRY.weights.min_claims_for_confidence
    )
    assert methodology["evidence_tiers"].keys() == REGISTRY.weights.evidence_tiers.keys()
    assert methodology["predicates_count"] == len(REGISTRY.predicates.predicates)
    assert methodology["predicates_version"] == REGISTRY.predicates.version
    # sources: counts match the registry listing, split by trust
    sources = list(REGISTRY.sources.sources)
    assert methodology["sources"]["total"] == len(sources)
    assert methodology["sources"]["enabled"] == sum(
        1 for s in sources if getattr(s, "enabled", False)
    )
    # the exclusion registry is surfaced with id, name and reason
    expected = [
        {"id": item.id, "name": item.name, "reason": item.reason}
        for item in REGISTRY.sources.exclusions
    ]
    assert methodology["source_exclusions"] == expected
    # (b) the dispute path ships in the payload: verbatim claim links, the
    # issue tracker, and the never-rewritten rule
    dispute = " ".join(methodology["dispute_path"])
    assert "claim" in dispute and "issue" in dispute
    assert "never" in dispute and "rewritten" in dispute
    # UNKNOWN is a value — stated, not implied
    assert "UNKNOWN" in " ".join(methodology["value_policy"])


# --------------------------------------------------- (c) append-only exclusions


def test_exclusion_append_is_idempotent_and_refuses_rewrites(pg_database) -> None:
    from fmd.db.repositories import (
        AppendOnlyViolation,
        SourceExclusionSeed,
        SourceExclusionsRepo,
    )

    repo = SourceExclusionsRepo(pg_database)
    seed = SourceExclusionSeed(
        id="w9_append_probe", name="Probe Portal", reason="not evidence"
    )
    assert repo.append(seed) == "appended"
    # an identical re-seed is a no-op, not an error and not a rewrite
    assert repo.append(seed) == "unchanged"
    # a CHANGED reason for the same id must fail: corrections get a new id
    changed = SourceExclusionSeed(
        id="w9_append_probe", name="Probe Portal", reason="PII-heavy"
    )
    with pytest.raises(AppendOnlyViolation) as excinfo:
        repo.append(changed)
    assert "new id" in str(excinfo.value)
    # the append-only guarantee means the row still carries the original reason
    with pg_database.connection() as conn:
        stored = conn.execute(
            "SELECT reason FROM source_exclusions WHERE id = %s", (seed.id,)
        ).fetchone()
    assert stored is not None and stored["reason"] == "not evidence"
