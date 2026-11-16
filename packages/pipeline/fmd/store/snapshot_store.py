"""Content-addressed snapshot store (brief §7).

Bytes are the evidence: a snapshot is addressed by the sha256 of the bytes that
were fetched. The database stores the hash and logical path; this module stores
and verifies the bytes. Writes are atomic (temp file + fsync + rename) and
idempotent. Sharding keeps directories small: ``objects/<first two hex>/<sha256>``.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024


class SnapshotStoreError(RuntimeError):
    """Base class for store failures."""


class SnapshotNotFound(SnapshotStoreError):
    pass


class SnapshotCorrupt(SnapshotStoreError):
    pass


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _validate(digest: str) -> str:
        normalised = digest.lower()
        if not _HASH_RE.fullmatch(normalised):
            raise ValueError(f"not a sha256 hex digest: {digest!r}")
        return normalised

    def path_for(self, sha256: str) -> Path:
        digest = self._validate(sha256)
        return self.objects / digest[:2] / digest

    def put(self, data: bytes) -> str:
        digest = self.sha256(data)
        target = self.path_for(digest)
        if target.is_file():
            if self.verify(digest):
                return digest
            raise SnapshotCorrupt(
                f"existing object {digest} is corrupt; refusing to overwrite evidence"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{digest}.", suffix=".tmp")
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
            self._fsync_dir(target.parent)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return digest

    def get(self, sha256: str) -> bytes:
        digest = self._validate(sha256)
        path = self.path_for(digest)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise SnapshotNotFound(f"snapshot {digest} is not in the store at {path}") from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise SnapshotCorrupt(f"snapshot {digest} failed integrity check (actual {actual})")
        return data

    def verify(self, sha256: str) -> bool:
        digest = self._validate(sha256)
        path = self.path_for(digest)
        if not path.is_file():
            return False
        hasher = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(_CHUNK):
                    hasher.update(chunk)
        except OSError:
            return False
        return hasher.hexdigest() == digest

    def exists(self, sha256: str) -> bool:
        return self.path_for(sha256).is_file()

    @staticmethod
    def _fsync_dir(directory: Path) -> None:
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
