"""Offline /api/exams contract tests: cycles, staleness, precision, deadlines.

No database and no network: window rows and claims come from in-memory fakes,
so the response semantics (per-exam cycles, staleness, month precision, travel
requirement, dependent programmes, next actionable deadlines sorted ascending)
are pinned without Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import fmd.api as api_module
from fmd.api import app
from fmd.db.repositories import (
    ClaimRecord,
    ExamWindowRecord,
    ProgrammeFilter,
)

client = TestClient(app)
ENVELOPE_KEYS = {"scorer_version", "weights_version", "claims_hash", "generated_at"}

NOW = datetime.now(UTC)


def window(
    exam_id: str,
    cycle_year: int,
    closes: datetime,
    claim_id: int,
    *,
    opens: datetime | None = None,
    observed: datetime | None = None,
    sitting_dates: list[str] | None = None,
) -> ExamWindowRecord:
    return ExamWindowRecord(
        id=claim_id,
        exam_id=exam_id,
        cycle_year=cycle_year,
        registration_opens=opens,
        registration_closes=closes,
        sitting_dates=sitting_dates,
        source_claim_id=claim_id,
        observed_at=observed if observed is not None else NOW - timedelta(days=1),
    )


def closing_claim(
    claim_id: int,
    value: str,
    *,
    precision: str | None = None,
    date_text: str | None = None,
    cycle_year: int | None = None,
) -> ClaimRecord:
    qualifier: dict[str, object] = {"exam_id": "fake"}
    if precision is not None:
        qualifier["precision"] = precision
    if date_text is not None:
        qualifier["date_text"] = date_text
    if cycle_year is not None:
        qualifier["cycle_year"] = cycle_year
    return ClaimRecord(
        id=claim_id,
        extraction_id=1,
        entity_id=11,
        predicate="exam_registration_closes",
        value=value,
        qualifier=qualifier,
        quote="printed window",
        char_start=0,
        char_end=14,
        observed_at=NOW - timedelta(days=1),
    )


class FakeExamsRepo:
    """In-memory stand-in for ExamsRepo over the seeded registry exams."""

    def __init__(self, _database: object) -> None:
        pass

    def list_windows(self, exam_id: str) -> list[ExamWindowRecord]:
        return {
            "mht_cet_pcb": [
                window(
                    "mht_cet_pcb",
                    2026,
                    datetime(2026, 8, 1, 15, 29, tzinfo=UTC),
                    901,
                ),
                window(
                    "mht_cet_pcb",
                    2027,
                    datetime(2027, 3, 12, 11, 30, tzinfo=UTC),
                    902,
                    opens=datetime(2027, 1, 12, tzinfo=UTC),
                    sitting_dates=["2027-04-22", "2027-04-23"],
                ),
            ],
            "kcet": [
                window(
                    "kcet",
                    2027,
                    datetime(2027, 2, 15, tzinfo=UTC),
                    903,
                    observed=NOW - timedelta(days=10),
                ),
            ],
            "jenpas_ug": [
                window(
                    "jenpas_ug",
                    2027,
                    datetime(2027, 3, 1, 0, 0, tzinfo=UTC),
                    904,
                ),
            ],
            "neet_ug": [
                window(
                    "neet_ug",
                    2026,
                    datetime(2026, 11, 1, tzinfo=UTC),
                    905,
                ),
            ],
        }.get(exam_id, [])


CLAIMS: dict[int, ClaimRecord] = {
    901: closing_claim(901, "2026-08-01", precision="day", date_text="01/08/2026"),
    902: closing_claim(
        902,
        "2027-03-12T11:30:00+00:00",
        precision="day",
        date_text="12/03/2027 5:00 pm",
    ),
    903: closing_claim(903, "2027-02-15", precision="day", date_text="15/02/2027"),
    904: closing_claim(904, "2027-03", precision="month", date_text="March 2027"),
    905: closing_claim(905, "2026-11-01", precision="day", date_text="01/11/2026"),
}


class FakeClaimsRepo:
    def __init__(self, _database: object) -> None:
        pass

    def get(self, claim_id: int) -> ClaimRecord | None:
        return CLAIMS.get(claim_id)

    def list_for_entity(self, entity_id: int) -> list[SimpleNamespace]:
        if entity_id != 11:
            return []
        return [
            SimpleNamespace(
                predicate="exam_required",
                value="mht_cet_pcb",
                qualifier={"exam_id": "mht_cet_pcb"},
            ),
            SimpleNamespace(
                predicate="exam_required",
                value="kcet",
                qualifier={"exam_id": "kcet"},
            ),
        ]


class FakeProgrammesRepo:
    def __init__(self, _database: object) -> None:
        pass

    def list(self, _filters: ProgrammeFilter) -> list[SimpleNamespace]:
        return [SimpleNamespace(id=7, institution_id=11)]


@pytest.fixture()
def fake_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "get_database", lambda: object())
    monkeypatch.setattr(api_module, "ExamsRepo", FakeExamsRepo)
    monkeypatch.setattr(api_module, "ClaimsRepo", FakeClaimsRepo)
    monkeypatch.setattr(api_module, "ProgrammesRepo", FakeProgrammesRepo)


PROFILE = {"target_intake_year": 2027, "travel_for_exams": "yes"}


def test_exams_reports_cycles_staleness_and_programme_dependencies(fake_db) -> None:
    response = client.post("/api/exams", json={"profile": PROFILE})
    assert response.status_code == 200
    body = response.json()
    assert body.keys() >= ENVELOPE_KEYS
    assert body["profile_persisted"] is False
    assert body["counts"]["open"] + body["counts"]["closed"] + body["counts"]["unverified"] == len(
        body["items"]
    )

    mht = next(item for item in body["items"] if item["exam_id"] == "mht_cet_pcb")
    assert mht["status"] == "open"
    # The profile's target cycle decides: 2027 (open), not 2026 (closed).
    assert mht["cycle_year"] == 2027
    assert mht["closes"] == "2027-03-12T11:30:00+00:00"
    assert mht["closes_text"] == "12/03/2027 5:00 pm"
    assert mht["precision"] == "day"
    assert mht["deciding_claim_id"] == 902
    assert mht["scope"] == "state"
    assert mht["requires_travel"] is True
    assert mht["travel_blocked"] is False
    assert mht["programme_ids"] == [7]
    assert {cycle["cycle_year"] for cycle in mht["cycles"]} == {2026, 2027}
    cycle_2027 = next(c for c in mht["cycles"] if c["cycle_year"] == 2027)
    assert cycle_2027["sitting_dates"] == ["2027-04-22", "2027-04-23"]
    assert cycle_2027["status"] == "open"
    cycle_2026 = next(c for c in mht["cycles"] if c["cycle_year"] == 2026)
    assert cycle_2026["status"] == "closed"

    neet = next(item for item in body["items"] if item["exam_id"] == "neet_ug")
    assert neet["status"] == "closed"
    assert neet["deciding_claim_id"] == 905
    # No extracted requirement claims point at NEET-UG in this fixture.
    assert neet["programme_ids"] == []

    merit = next(item for item in body["items"] if item["exam_id"] == "merit_class12")
    assert merit["status"] == "unverified"
    assert merit["cycles"] == []


def test_stale_window_is_unverified_never_open(fake_db) -> None:
    body = client.post("/api/exams", json={"profile": PROFILE}).json()

    kcet = next(item for item in body["items"] if item["exam_id"] == "kcet")
    assert kcet["status"] == "unverified"
    assert kcet["stale"] is True
    assert kcet["deciding_claim_id"] == 903


def test_month_precision_never_publishes_a_fabricated_day(fake_db) -> None:
    body = client.post("/api/exams", json={"profile": PROFILE}).json()

    jenpas = next(item for item in body["items"] if item["exam_id"] == "jenpas_ug")
    assert jenpas["precision"] == "month"
    assert jenpas["closes_text"] == "March 2027"
    assert jenpas["closes"] is None  # no day exists; none may be invented
    # now (2026-11) is before the printed month, so the window is open...
    assert jenpas["status"] == "open"
    # ...but the deadline stays month-precision, sorted by the month's start.
    entry = next(d for d in body["next_deadlines"] if d["exam_id"] == "jenpas_ug")
    assert entry["deadline"] is None
    assert entry["deadline_month"] == "2027-03"
    assert entry["claim_id"] == 904


def test_next_deadlines_are_sorted_ascending_and_exclude_closed(fake_db) -> None:
    body = client.post("/api/exams", json={"profile": PROFILE}).json()

    deadlines = body["next_deadlines"]
    assert deadlines, "at least one actionable deadline must be listed"
    assert [d["act_before"] for d in deadlines] == sorted(d["act_before"] for d in deadlines)
    assert {d["exam_id"] for d in deadlines}.isdisjoint({"neet_ug"})
    ids = [d["exam_id"] for d in deadlines]
    assert ids.index("kcet") < ids.index("jenpas_ug") < ids.index("mht_cet_pcb")
    kcet_entry = next(d for d in deadlines if d["exam_id"] == "kcet")
    assert kcet_entry["status"] == "unverified"  # stale, but verification is actionable


def test_travel_blocked_exam_is_closed_for_a_no_travel_profile(fake_db) -> None:
    body = client.post("/api/exams", json={"profile": {**PROFILE, "travel_for_exams": "no"}}).json()

    mht = next(item for item in body["items"] if item["exam_id"] == "mht_cet_pcb")
    assert mht["travel_blocked"] is True
    assert mht["status"] == "closed"
    assert "travel" in mht["reason"]
    # A travel-blocked exam is not an actionable deadline.
    assert {d["exam_id"] for d in body["next_deadlines"]}.isdisjoint({"mht_cet_pcb"})


def test_exams_without_a_database_serve_an_honest_empty_state(monkeypatch) -> None:
    def boom() -> None:
        raise api_module.DatabaseError("cannot connect")

    monkeypatch.setattr(api_module, "get_database", boom)
    response = client.post("/api/exams", json={"profile": PROFILE})
    assert response.status_code == 200
    body = response.json()
    assert body.keys() >= ENVELOPE_KEYS
    assert body["items"]
    for item in body["items"]:
        assert {"status", "observed_at", "stale", "requires_travel", "deciding_claim_id"} <= (
            item.keys()
        )
        assert item["status"] == "unverified"
        assert item["cycles"] == []
        assert item["programme_ids"] == []
        assert item["deciding_claim_id"] is None
    assert body["next_deadlines"] == []
    assert client.get("/api/exams").status_code == 405
