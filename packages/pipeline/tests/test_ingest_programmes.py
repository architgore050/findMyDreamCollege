"""Programme materialisation from verified claims (pure, offline).

The contract under test: numbers come only from a claim, every set attribute
cites its claim id, and an unproven name is flagged rather than merged.
"""

from __future__ import annotations

from fmd.ingest.dedupe_audit import (
    audit,
    audit_from_run,
    classify,
    document_stem,
    is_same_document,
)
from fmd.ingest.programmes import (
    ProgrammeMapping,
    names_are_similar,
    normalise_programme_name,
    programme_drafts,
)


def claim(
    claim_id: int,
    entity_id: int,
    predicate: str,
    value,
    *,
    unit: str | None = None,
    status: str = "supported",
    method: str = "extracted",
    confidence: float = 0.9,
    document_role: str = "evidence",
) -> dict:
    return {
        "id": claim_id,
        "entity_id": entity_id,
        "predicate": predicate,
        "value": value,
        "unit": unit,
        "confidence": confidence,
        "method": method,
        "verification_status": status,
        "document_role": document_role,
        "superseded_by": None,
    }


def mapping() -> ProgrammeMapping:
    return ProgrammeMapping.from_params({})


def test_claims_become_one_programme_with_sourced_fields() -> None:
    claims = [
        claim(10, 1, "curriculum.programme_name", "Bachelor of Medical Laboratory Sciences(B.MLS)"),
        claim(11, 1, "curriculum.duration_months", 48, unit="months"),
        claim(12, 1, "curriculum.internship_months", 6, unit="months"),
        claim(13, 1, "seats.total", 60, unit="count"),
        claim(14, 1, "accreditation.recognition_status", "recognised"),
    ]
    drafts, rejected = programme_drafts(claims, mapping())

    assert rejected == []
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.institution_id == 1
    assert draft.normalised_name == "bsc mls"
    assert draft.credential_level == "degree"
    assert draft.source_family == "curriculum"
    assert (draft.duration_months, draft.internship_months, draft.seats) == (48, 6.0, 60)
    assert draft.recognition_status == "recognised"
    assert draft.field_sources == {
        "curriculum.programme_name": "10",
        "curriculum.duration_months": "11",
        "curriculum.internship_months": "12",
        "seats.total": "13",
        "accreditation.recognition_status": "14",
    }
    assert draft.needs_review is False


def test_missing_attributes_stay_null_and_are_not_invented() -> None:
    """A programme may be only a name and an institution; nothing is backfilled."""
    drafts, _ = programme_drafts(
        [claim(1, 5, "seats.programme_name", "B.Sc. MLT")],
        mapping(),
    )
    draft = drafts[0]
    assert draft.duration_months is None
    assert draft.internship_months is None
    assert draft.seats is None
    assert draft.medium is None
    assert draft.affiliated_university_id is None
    assert draft.recognition_status == "unknown"
    assert draft.field_sources == {"seats.programme_name": "1"}


def test_unverified_or_modelled_claims_never_materialise() -> None:
    claims = [
        claim(1, 2, "curriculum.programme_name", "B.Sc. MLT", status="unverified"),
        claim(2, 2, "curriculum.programme_name", "B.Sc. MLT", method="modelled"),
        claim(3, 3, "curriculum.programme_name", "B.Sc. MLT", status="rejected"),
    ]
    drafts, rejected = programme_drafts(claims, mapping())
    assert drafts == []
    assert rejected == []


def test_navigation_documents_are_not_evidence() -> None:
    drafts, rejected = programme_drafts(
        [
            claim(
                1,
                4,
                "curriculum.programme_name",
                "B.Sc. MLT",
                document_role="navigation",
            )
        ],
        mapping(),
    )
    assert drafts == []
    assert rejected == []


def test_unresolvable_institution_is_refused() -> None:
    drafts, rejected = programme_drafts(
        [claim(1, 99, "curriculum.programme_name", "B.Sc. MLT")],
        mapping(),
        resolve_institution=lambda entity_id: None,
    )
    assert drafts == []
    assert rejected == [{"claim_id": 1, "reason": "entity_unresolvable"}]


def test_er_normalisation_collapses_aliases_but_not_mlt_into_mls() -> None:
    m = mapping()
    mlt = normalise_programme_name("B.Sc. MLT", m)
    long_form = normalise_programme_name("Bachelor of Medical Laboratory Technology (BMLT)", m)
    mls = normalise_programme_name("Bachelor of Medical Laboratory Sciences(B.MLS)", m)

    assert mlt.key == long_form.key == "bsc mlt"
    assert mlt.matched_alias and long_form.matched_alias
    assert mls.key == "bsc mls"
    # Uncertain collapse stays separate and is only flagged.
    assert names_are_similar(mlt.key, mls.key) is True


def test_similar_names_at_one_institution_are_flagged_not_merged() -> None:
    drafts, _ = programme_drafts(
        [
            claim(1, 7, "curriculum.programme_name", "B.Sc. MLT"),
            claim(2, 7, "seats.programme_name", "Bachelor of Medical Laboratory Sciences"),
        ],
        mapping(),
    )
    assert {draft.normalised_name for draft in drafts} == {"bsc mlt", "bsc mls"}
    assert all(draft.needs_review for draft in drafts)
    assert any("possible_duplicate_programme_name" in r for d in drafts for r in d.review_reasons)


def test_ambiguous_recognition_is_unknown_and_reviewed() -> None:
    drafts, _ = programme_drafts(
        [
            claim(1, 8, "curriculum.programme_name", "B.Sc. MLT"),
            claim(2, 8, "accreditation.recognition_status", "Accredited"),
        ],
        mapping(),
    )
    draft = drafts[0]
    assert draft.recognition_status == "unknown"
    assert "recognition_value_ambiguous" in draft.review_reasons


def test_dedupe_audit_calls_distinct_pdfs_false_positives() -> None:
    entries = [
        {
            "url": "https://a.ac.in/Annual-Report-2021-22.pdf",
            "compared_url": "https://a.ac.in/Annual-Report-2022-23.pdf",
            "similarity": 1.0,
            "hamming_distance": 0,
            "mode": "simhash",
            "source_id": "institution_sites",
        },
        {
            "url": "https://b.ac.in/prospectus.pdf",
            "compared_url": "https://mirror.example/prospectus.pdf",
            "similarity": 1.0,
            "hamming_distance": 0,
            "mode": "simhash",
            "source_id": "institution_sites",
        },
    ]
    report = audit(entries)
    assert report.total == 2
    assert report.false_positives == 1
    assert report.boilerplate == 1
    assert report.false_positive_ratio == 0.5
    assert report.by_source["institution_sites"]["false_positive"] == 1


def test_dedupe_audit_reads_run_metadata_and_never_invents_urls() -> None:
    summary = {
        "metadata": {
            "dedupe": [
                {
                    "url": "https://x.ac.in/a.pdf",
                    "compared_url": "https://x.ac.in/b.pdf",
                    "similarity": 0.98,
                    "mode": "simhash",
                }
            ]
        }
    }
    report = audit_from_run(summary)
    assert report.total == 1
    assert report.false_positives == 1
    assert report.examples[0].url == "https://x.ac.in/a.pdf"


def test_dedupe_url_identity_helpers() -> None:
    assert document_stem("https://a.ac.in/x/Annual-Report-2021-22.pdf") == "annual report 2021 22"
    assert is_same_document("https://a.ac.in/a.pdf", "https://b.ac.in/a.pdf") is True
    assert is_same_document("https://a.ac.in/list?page=1", "https://a.ac.in/list?page=2") is True
    assert is_same_document("https://a.ac.in/a.pdf", "https://a.ac.in/b.pdf") is False
    assert classify({"url": "https://a.ac.in/a.pdf", "mode": "simhash"}).verdict == "unknown"
