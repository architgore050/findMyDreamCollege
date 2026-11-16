from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fmd.store import SnapshotCorrupt, SnapshotNotFound, SnapshotStore


def test_put_get_roundtrip_and_layout(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    data = b"<html>MLT programme</html>"
    digest = store.put(data)

    assert digest == hashlib.sha256(data).hexdigest()
    assert store.get(digest) == data
    assert store.verify(digest) is True

    expected = tmp_path / "objects" / digest[:2] / digest
    assert store.path_for(digest) == expected
    assert expected.is_file()


def test_put_is_idempotent_and_content_addressed(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    first = store.put(b"same bytes")
    second = store.put(b"same bytes")
    assert first == second
    assert len([p for p in (tmp_path / "objects").rglob("*") if p.is_file()]) == 1

    other = store.put(b"different bytes")
    assert other != first
    assert store.get(other) == b"different bytes"


def test_missing_snapshot_raises(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    missing = "0" * 64
    with pytest.raises(SnapshotNotFound):
        store.get(missing)
    assert store.verify(missing) is False
    assert store.exists(missing) is False


def test_corruption_is_detected(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    digest = store.put(b"evidence")
    store.path_for(digest).write_bytes(b"tampered")
    assert store.verify(digest) is False
    with pytest.raises(SnapshotCorrupt):
        store.get(digest)
    with pytest.raises(SnapshotCorrupt):
        store.put(b"evidence")


def test_path_for_rejects_non_sha256(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    with pytest.raises(ValueError):
        store.path_for("../../etc/passwd")
