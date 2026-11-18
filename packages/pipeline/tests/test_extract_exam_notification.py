"""exam_notification@v3: window fields, precision contract, prompt history.

The v3 prompt keeps v1/v2 (they remain valid against the same schema: the new
fields default to null). v3 is where the extractor learns the rules the
workstream depends on: registration close is never a sitting date, a
month-only print is never completed into a day, and scope/state are quoted
from the notification instead of inferred.
"""

from __future__ import annotations

from fmd.extract.prompts import load_prompt
from fmd.extract.schemas import ExamNotification, predicate_specs

V3_CONTRACT_MARKERS = (
    "registration close is not a sitting date",
    "must never be quoted as a registration close",
    "never invent, complete or normalise a day",
    "month precision",
    "quote a printed time of day together with its date",
    "the time is part of the deadline",
    "never infer scope from",
    "never write",
)


def test_v3_prompt_exists_and_older_versions_are_kept() -> None:
    v1 = load_prompt("exam_notification", "v1")
    v2 = load_prompt("exam_notification", "v2")
    v3 = load_prompt("exam_notification", "v3")

    assert v1.ref == "exam_notification@v1"
    assert v2.ref == "exam_notification@v2"
    assert v3.ref == "exam_notification@v3"
    assert load_prompt("exam_notification").ref == "exam_notification@v3"


def test_v3_prompt_states_the_window_contract() -> None:
    prompt = load_prompt("exam_notification", "v3")
    flattened = " ".join(prompt.text.split()).lower()

    for marker in V3_CONTRACT_MARKERS:
        assert " ".join(marker.split()).lower() in flattened, (
            f"exam_notification@v3 is missing the contract statement {marker!r}"
        )


def test_v3_schema_declares_the_window_fields() -> None:
    specs = ExamNotification.SPECS

    assert set(specs) == {
        "exam_name",
        "cycle_year",
        "scope",
        "state",
        "registration_opens",
        "registration_closes",
        "exam_date",
        "sitting_dates",
        "requires_travel",
        "fee_amount",
    }
    assert specs["sitting_dates"].unit_kind == "text"
    assert specs["scope"].unit_kind == "text"
    assert specs["state"].unit_kind == "text"
    ExamNotification.validate_schema()


def test_new_predicates_are_registry_rows() -> None:
    specs = predicate_specs()

    assert specs["exam.scope"].unit_kind == "text"
    assert specs["exam.state"].unit_kind == "text"
    assert specs["exam.sitting_dates"].unit_kind == "text"


def test_old_prompt_payloads_still_validate_against_the_v3_schema() -> None:
    """A v1/v2 payload (without the new keys) is still a valid document."""

    document = ExamNotification(
        exam_name=None,
        cycle_year=None,
        scope=None,
        state=None,
        registration_opens=None,
        registration_closes=None,
        exam_date=None,
        sitting_dates=None,
        requires_travel=None,
        fee_amount=None,
    )
    assert document.scope is None
    assert document.sitting_dates is None
