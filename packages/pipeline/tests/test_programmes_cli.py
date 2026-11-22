"""`fmd programmes` CLI: materialise / list / inspect against Postgres.

Skipped cleanly without DATABASE_URL (see ``pg_database`` in conftest). The
handlers open their own database connection, so each test points the CLI at
the fixture's throw-away schema via a conninfo option (same trick as
``test_cli.py``).
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest

from fmd.cli import main
from fmd.db import (
    ClaimCreate,
    ClaimsRepo,
    Database,
    DocumentsRepo,
    DocumentUpsert,
    EntitiesRepo,
    EntityCreate,
    ExtractionCreate,
    ExtractionsRepo,
    SnapshotCreate,
    SnapshotsRepo,
    run_migrations,
)
from fmd.db.repositories import ClaimVerification

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

pytestmark = pytest.mark.usefixtures("pg_url")


@pytest.fixture
def cli_database(pg_url: str):
    schema = "test_cli_prg_" + uuid.uuid4().hex[:10]
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    database = Database(pg_url, search_path=schema, min_size=1, max_size=2)
    run_migrations(database)
    try:
        yield database
    finally:
        database.close()
        with psycopg.connect(pg_url, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


def _dsn(database) -> str:
    option = quote(f"-c search_path={database.search_path}")
    separator = "&" if "?" in database.url else "?"
    return f"{database.url}{separator}options={option}"


def _seed_claim(database, entity_id: int, predicate: str, value, *, unit=None, quote="q"):
    suffix = uuid.uuid4().hex[:8]
    document = DocumentsRepo(database).upsert(
        DocumentUpsert(
            url=f"https://example.ac.in/{suffix}.pdf",
            domain="example.ac.in",
            kind="curriculum",
            role="evidence",
        )
    )
    text = f"evidence text for {predicate} {suffix}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    snapshot = SnapshotsRepo(database).create(
        SnapshotCreate(document_id=document.id, sha256=digest, bytes_path=f"o/{digest}")
    )
    extraction = ExtractionsRepo(database).create(
        ExtractionCreate(snapshot_id=snapshot.id, tool="test", text=text, page_range="2-3")
    )
    claim = ClaimsRepo(database).create(
        ClaimCreate(
            extraction_id=extraction.id,
            entity_id=entity_id,
            predicate=predicate,
            value=value,
            unit=unit,
            quote=quote,
            char_start=0,
            char_end=len(text),
            confidence=0.9,
        )
    )
    ClaimsRepo(database).record_verification(
        claim.id, ClaimVerification(status="supported", verified_by="cli-test")
    )
    return claim


@pytest.fixture
def seeded(cli_database):
    entity = EntitiesRepo(cli_database).create(
        EntityCreate(
            kind="institution",
            canonical_name="Stanley College of Engineering",
            country="IN",
            state="Telangana",
        )
    )
    name_claim = _seed_claim(cli_database, entity.id, "seats.programme_name", "B.Sc. MLT")
    intake_claim = _seed_claim(
        cli_database, entity.id, "seats.intake", 20, unit="count", quote="B.Sc. MLT | 20"
    )
    total_claim = _seed_claim(
        cli_database, entity.id, "seats.total", 195, unit="count", quote="Total | 195"
    )
    return {
        "database": cli_database,
        "entity": entity,
        "name_claim": name_claim,
        "intake_claim": intake_claim,
        "total_claim": total_claim,
    }


def test_materialise_dry_run_on_empty_database(cli_database, capsys) -> None:
    argv = [
        "--database-url",
        _dsn(cli_database),
        "--config-dir",
        str(CONFIG_DIR),
        "programmes",
        "materialise",
        "--dry-run",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "claims=0" in out
    assert "dry run: nothing written" in out


def test_materialise_writes_rows_and_rerun_is_unchanged(seeded, capsys) -> None:
    database = seeded["database"]
    base = ["--database-url", _dsn(database), "--config-dir", str(CONFIG_DIR)]

    assert main([*base, "programmes", "materialise"]) == 0
    out = capsys.readouterr().out
    assert "wrote=1 (create=1 update=0" in out
    assert "B.Sc. Medical Laboratory Technology" in out
    # The aggregate guard is visible in the output, with its reason.
    assert "[refused] seats.total claim" in out
    assert str(seeded["total_claim"].id) in out
    assert "seats_total_aggregate_not_programme_intake" in out

    assert main([*base, "programmes", "materialise"]) == 0
    out = capsys.readouterr().out
    assert "wrote=0 (create=0 update=0 unchanged=1)" in out

    assert main([*base, "programmes", "materialise", "--json"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["created"] == 0 and payload["unchanged"] == 1
    assert payload["aggregates_refused"][0]["value"] == 195


def test_list_prints_seats_claim_and_completeness(seeded, capsys) -> None:
    database = seeded["database"]
    base = ["--database-url", _dsn(database), "--config-dir", str(CONFIG_DIR)]
    assert main([*base, "programmes", "materialise"]) == 0
    capsys.readouterr()

    assert main([*base, "programmes", "list"]) == 0
    out = capsys.readouterr().out
    assert "Stanley College of Engineering" in out
    assert "B.Sc. Medical Laboratory Technology" in out
    assert f"seats 20 (claim {seeded['intake_claim'].id})" in out
    assert "fields 4/6" in out  # name, credential, seats, aggregate evidence
    assert "components 1/7 (admission_realism)" in out
    assert "needs_review=false" in out

    assert main([*base, "programmes", "list", "--json"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    row = payload["programmes"][0]
    assert row["seats_source_claim_id"] == str(seeded["intake_claim"].id)
    assert row["completeness"]["components_total"] == 7
    assert row["completeness"]["backed_fields"] == [
        "name",
        "credential_level",
        "seats",
        "seats_total_aggregate",
    ]


def test_inspect_cites_every_field_with_claim_quote_and_page(seeded, capsys) -> None:
    database = seeded["database"]
    base = ["--database-url", _dsn(database), "--config-dir", str(CONFIG_DIR)]
    assert main([*base, "programmes", "materialise"]) == 0
    capsys.readouterr()

    assert main([*base, "programmes", "list", "--json"]) == 0
    import json

    programme_id = json.loads(capsys.readouterr().out)["programmes"][0]["id"]

    assert main([*base, "programmes", "inspect", str(programme_id)]) == 0
    out = capsys.readouterr().out
    # Every field line cites its claim; a reviewer can audit end to end.
    assert "[claim " in out and "· seats.intake]" in out
    assert "· seats.programme_name]" in out
    assert "(derived from the name claim)" in out
    assert "duration_months        = NULL   [no claim]" in out
    assert "seats_total_aggregate" in out and "never a per-programme intake" in out
    # The claim block: id, quote, page.
    assert f"#{seeded['intake_claim'].id} seats.intake" in out
    assert "quote: B.Sc. MLT | 20" in out
    assert "page=2-3" in out
    assert "sha256" in out

    assert main([*base, "programmes", "inspect", str(programme_id), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["programme"]["id"] == programme_id
    predicates = {row["predicate"] for row in payload["claims"]}
    assert predicates == {"seats.programme_name", "seats.intake"}


def test_inspect_unknown_programme_exits_non_zero(cli_database, capsys) -> None:
    argv = [
        "--database-url",
        _dsn(cli_database),
        "--config-dir",
        str(CONFIG_DIR),
        "programmes",
        "inspect",
        "9999",
    ]
    assert main(argv) == 1
    assert "no programme with id 9999" in capsys.readouterr().err


def test_materialise_dry_run_does_not_create_rows(seeded, capsys) -> None:
    """A fresh schema: --dry-run plans the row but the table stays empty."""
    database = seeded["database"]
    argv = [
        "--database-url",
        _dsn(database),
        "--config-dir",
        str(CONFIG_DIR),
        "programmes",
        "materialise",
        "--dry-run",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "would_write=1 (create=1" in out
    assert "dry run: nothing written" in out
    from fmd.db.repositories import ProgrammesRepo

    assert ProgrammesRepo(database).counts()["total"] == 0


def test_dry_run_succeeds_with_an_unreachable_model_url(seeded, capsys, monkeypatch) -> None:
    """A dry run must never dial the model: it reports what it would do.

    The endpoint is a closed port, so any real router call would fail. Both
    commands still exit 0 and print the scope tallies, with ``would_classify``
    standing in for the semantic names the real run would send.
    """
    monkeypatch.setenv("FMD_LOCAL_LLM_URL", "http://127.0.0.1:9/v1")
    database = seeded["database"]
    base = ["--database-url", _dsn(database), "--config-dir", str(CONFIG_DIR)]

    assert main([*base, "programmes", "materialise", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry-run scope: would_admit=1 would_reject=0" in out
    assert "would_classify=0" in out
    from fmd.db.repositories import ProgrammesRepo

    assert ProgrammesRepo(database).counts()["total"] == 0

    assert main([*base, "programmes", "requalify", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry-run scope: would_admit=" in out
    assert "would_reject=" in out and "would_defer=" in out and "would_classify=" in out
