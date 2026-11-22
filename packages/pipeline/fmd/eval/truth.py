"""Truth tables and proxy-ranking metrics (docs/v1/eval/03-truth-ingestion.md).

``data/truth/`` holds external reference tables (NIRF rankings, NAAC grades)
that the *product* streams of ``fmd eval`` rank shortlists against. The loader
enforces the ingestion contract mechanically:

* the provenance block is complete and self-consistent;
* ``source_sha256`` matches the bytes of ``raw_file`` (re-verified offline on
  every load — truth that cannot prove its own provenance never reaches a
  metric);
* NIRF ranks form a non-decreasing series starting at 1 (ties are legal NIRF
  output); every entry carries all five parameter scores;
* NAAC grades come from the published vocabulary and CGPA stays in [0, 4];
* ``institution_key`` equals ``normalise_institution_key(name)`` recomputed
  here, never trusted from the file.

The metrics are pure functions over rankings: top-k overlap, Spearman's rho
and Kendall's tau-b on the overlapping subset, and the "best available" hit
rate. UNKNOWN is a value (invariant 5): a shortlist row whose institution has
no truth entry never enters a correlation; it is counted as UNKNOWN coverage.
Correlations return ``None`` when fewer than two comparable pairs exist —
never 0, and never a fabricated number for a proxy stream.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRUTH_DIR = REPO_ROOT / "data" / "truth"

NIRF_PARAM_KEYS = ("tlr", "rpc", "go", "oi", "pr")
NAAC_GRADES = ("A++", "A+", "A", "B++", "B+", "B", "C", "D")
VALID_TRUTH_KINDS = ("nirf_ranking", "naac_grades")


class TruthError(ValueError):
    """A truth file violates the ingestion contract."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        joined = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__("truth file is invalid. Fix the following and re-run:\n" + joined)


def normalise_institution_key(name: str) -> str:
    """Shared join key rule (contract §4): fold, lowercase, '&'->and, strip punct."""

    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NirfEntry(BaseModel):
    """One institution on one NIRF ranking page (institution-level only)."""

    model_config = ConfigDict(extra="forbid")

    institution_id: str
    institution_key: str
    name: str
    city: str
    state: str
    rank: int
    score: float
    params: dict[str, float]
    source_pdf: str | None = None

    @model_validator(mode="after")
    def _entry_rules(self) -> NirfEntry:
        if not self.institution_key.strip():
            raise ValueError(f"{self.institution_id}: empty institution_key")
        if self.rank < 1:
            raise ValueError(f"{self.institution_id}: rank must be >= 1")
        if not 0.0 <= self.score <= 100.0:
            raise ValueError(f"{self.institution_id}: score {self.score} outside 0..100")
        missing = [key for key in NIRF_PARAM_KEYS if key not in self.params]
        if missing:
            raise ValueError(f"{self.institution_id}: missing parameter scores {missing}")
        expected = normalise_institution_key(self.name)
        if self.institution_key != expected:
            raise ValueError(
                f"{self.institution_id}: institution_key {self.institution_key!r} does not "
                f"match normalise_institution_key(name) = {expected!r}"
            )
        return self


class NaacEntry(BaseModel):
    """One accredited institution (5-year cycle; institutional, never programme)."""

    model_config = ConfigDict(extra="forbid")

    institution_key: str
    aishe_code: str | None = None
    grade: str
    cgpa: float | None = None
    accreditation_cycle: int | None = None
    valid_until: str | None = None
    regime: Literal["graded", "binary"] = "graded"
    source_pdf: str | None = None

    @model_validator(mode="after")
    def _entry_rules(self) -> NaacEntry:
        if not self.institution_key.strip():
            raise ValueError("NAAC entry with empty institution_key")
        if self.grade not in NAAC_GRADES:
            raise ValueError(
                f"{self.institution_key}: grade {self.grade!r} not in published set "
                f"{NAAC_GRADES} (binary-regime labels need regime: binary)"
            )
        if self.cgpa is not None and not 0.0 <= self.cgpa <= 4.0:
            raise ValueError(f"{self.institution_key}: cgpa {self.cgpa} outside 0..4")
        return self


class TruthFile(BaseModel):
    """Header block shared by every truth file (contract §2)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    truth_kind: Literal["nirf_ranking", "naac_grades"]
    truth_version: int
    scope: Literal["institution"]
    as_of: str | None = None
    source_url: str
    source_sha256: str | None = None
    fetched_at: str | None = None
    raw_file: str | None = None
    acquisition_status: str | None = None
    ingest_tool: str | None = None
    notes: str | None = None
    # kind-specific
    category: str | None = None
    year: int | None = None
    entries: list[dict[str, Any]] = Field(default_factory=list)


class NirfTruthFile(BaseModel):
    """A validated NIRF ranking edition (one category, one year)."""

    model_config = ConfigDict(extra="forbid")

    header: TruthFile
    category: str
    year: int
    entries: list[NirfEntry]
    source_sha256: str
    raw_file: Path
    fetched_at: str

    @property
    def as_of(self) -> str:
        return str(self.year)

    def ranked_keys(self) -> list[str]:
        """Institution keys in reference order (rank asc, score desc for ties)."""

        ordered = sorted(self.entries, key=lambda e: (e.rank, -e.score))
        return [entry.institution_key for entry in ordered]

    def by_key(self) -> dict[str, NirfEntry]:
        return {entry.institution_key: entry for entry in self.entries}


class NaacTruthFile(BaseModel):
    """A validated NAAC grades table (possibly empty: not ingested yet)."""

    model_config = ConfigDict(extra="forbid")

    header: TruthFile
    entries: list[NaacEntry]
    source_sha256: str | None
    raw_file: Path | None
    fetched_at: str | None

    @property
    def as_of(self) -> str | None:
        dates = [entry.valid_until for entry in self.entries if entry.valid_until]
        return max(dates) if dates else None

    def ranked_keys(self) -> list[str]:
        """Institution keys in reference order (CGPA desc, grade as tiebreak)."""

        grade_rank = {grade: index for index, grade in enumerate(NAAC_GRADES)}
        ordered = sorted(
            self.entries,
            key=lambda e: (-(e.cgpa if e.cgpa is not None else -1.0), grade_rank[e.grade]),
        )
        return [entry.institution_key for entry in ordered]

    def by_key(self) -> dict[str, NaacEntry]:
        return {entry.institution_key: entry for entry in self.entries}


def _load_header(path: Path) -> TruthFile:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TruthError([f"cannot read truth file {path}: {exc}"]) from exc
    if not isinstance(payload, dict):
        raise TruthError([f"{path}: truth file must be a YAML mapping"])
    try:
        return TruthFile.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError + ValueError wraps
        raise TruthError([f"{path}: header block invalid: {exc}"]) from exc


def load_truth_file(path: str | Path, *, root: Path | None = None) -> NirfTruthFile | NaacTruthFile:
    """Load and validate one truth file, re-verifying its provenance offline."""

    base = Path(root) if root is not None else REPO_ROOT
    file_path = Path(path)
    if not file_path.is_file():
        raise TruthError([f"truth file not found: {file_path}"])
    header = _load_header(file_path)
    problems: list[str] = []

    if header.truth_kind == "naac_grades" and header.truth_version == 0:
        # The explicitly-unfetched table: entries must be empty, hashes null.
        if header.entries:
            problems.append("truth_version 0 means 'not ingested'; entries must be empty")
        if header.source_sha256 or header.fetched_at:
            problems.append("truth_version 0 forbids a source_sha256/fetched_at")
        if problems:
            raise TruthError(problems)
        return NaacTruthFile(
            header=header, entries=[], source_sha256=None, raw_file=None, fetched_at=None
        )

    if header.raw_file is None:
        problems.append("raw_file is required for an ingested truth table")
    if header.source_sha256 is None:
        problems.append("source_sha256 is required for an ingested truth table")
    if header.fetched_at is None:
        problems.append("fetched_at is required for an ingested truth table")
    if problems:
        raise TruthError(problems)

    raw_path = base / header.raw_file  # type: ignore[arg-type]
    if not raw_path.is_file():
        raise TruthError([f"raw page copy missing: {header.raw_file}"])
    actual_hash = _sha256_of(raw_path)
    if actual_hash != header.source_sha256:
        problems.append(
            f"source_sha256 mismatch: header says {header.source_sha256}, "
            f"raw file hashes to {actual_hash}"
        )
        raise TruthError(problems)

    if header.truth_kind == "nirf_ranking":
        entries: list[NirfEntry] = []
        seen_ids: set[str] = set()
        for index, row in enumerate(header.entries, start=1):
            try:
                entry = NirfEntry.model_validate(row)
            except Exception as exc:
                problems.append(f"entry #{index} invalid: {exc}")
                continue
            if entry.institution_id in seen_ids:
                problems.append(f"duplicate institution_id {entry.institution_id}")
            seen_ids.add(entry.institution_id)
            entries.append(entry)
        ranks = [entry.rank for entry in entries]
        if not ranks:
            problems.append("nirf_ranking with no entries; ingest a page or mark version 0")
        elif ranks[0] != 1 or any(b < a for a, b in zip(ranks, ranks[1:], strict=False)):
            problems.append(f"ranks must be a non-decreasing series starting at 1: {ranks[:10]}...")
        if header.category is None or header.year is None:
            problems.append("nirf_ranking requires category and year in the header")
        if header.as_of != str(header.year):
            problems.append(f"as_of must be the edition year ({header.year}) for NIRF files")
        if problems:
            raise TruthError(problems)
        return NirfTruthFile(
            header=header,
            category=header.category,  # type: ignore[arg-type]
            year=header.year,  # type: ignore[arg-type]
            entries=entries,
            source_sha256=header.source_sha256,  # type: ignore[arg-type]
            raw_file=raw_path,
            fetched_at=header.fetched_at,  # type: ignore[arg-type]
        )

    if header.truth_kind == "naac_grades":
        entries = []
        seen_keys: set[str] = set()
        for index, row in enumerate(header.entries, start=1):
            try:
                entry = NaacEntry.model_validate(row)
            except Exception as exc:
                problems.append(f"entry #{index} invalid: {exc}")
                continue
            if entry.institution_key in seen_keys:
                problems.append(f"duplicate institution_key {entry.institution_key}")
            seen_keys.add(entry.institution_key)
            entries.append(entry)
        if problems:
            raise TruthError(problems)
        return NaacTruthFile(
            header=header,
            entries=entries,
            source_sha256=header.source_sha256,
            raw_file=raw_path,
            fetched_at=header.fetched_at,
        )

    raise TruthError(
        [f"unknown truth_kind {header.truth_kind!r}; valid kinds: {VALID_TRUTH_KINDS}"]
    )


# --------------------------------------------------------------------------- metrics


def top_k_overlap(product_order: list[str], reference_order: list[str], k: int) -> int:
    """Count of institutions shared by the two top-k prefixes."""

    return len(set(product_order[:k]) & set(reference_order[:k]))


def _average_ranks(values: list[float]) -> list[float]:
    """1-based ranks with average ranks for ties (Spearman tie handling)."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    """Spearman's rho with tie-aware average ranks; None when degenerate."""

    if len(x) != len(y) or len(x) < 2:
        return None
    rank_x = _average_ranks(x)
    rank_y = _average_ranks(y)
    if len(set(rank_x)) == 1 or len(set(rank_y)) == 1:
        return None  # all-tied: no ordering information exists
    return round(_pearson(rank_x, rank_y), 6)


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    """Kendall's tau-b with full tie correction; None when degenerate."""

    if len(x) != len(y) or len(x) < 2:
        return None
    n = len(x)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom_x = n * (n - 1) / 2 - ties_x
    denom_y = n * (n - 1) / 2 - ties_y
    if denom_x <= 0 or denom_y <= 0:
        return None
    return round((concordant - discordant) / math.sqrt(denom_x * denom_y), 6)


def best_available_hit(product_order: list[str], reference_order: list[str]) -> bool:
    """Does the product's #1 appear in the reference top-3 (contract §1.2)?"""

    if not product_order:
        return False
    return product_order[0] in set(reference_order[:3])


def rank_correlation_on_overlap(
    product_order: list[str],
    reference_order: list[str],
    *,
    product_positions: dict[str, float],
    reference_positions: dict[str, float],
) -> dict[str, Any]:
    """Spearman/Kendall between the two orders on their shared institutions.

    ``product_positions`` / ``reference_positions`` map an institution key to
    its rank position (1-based, lower is better) in each order. Institutions
    missing from either side are UNKNOWN for this comparison and are excluded
    (never coerced to a numeric rank).
    """

    shared = [key for key in product_order if key in reference_order and key in reference_positions]
    if len(shared) < 2:
        return {"spearman_rho": None, "kendall_tau_b": None, "n_pairs": len(shared)}
    x = [product_positions[key] for key in shared]
    y = [reference_positions[key] for key in shared]
    return {
        "spearman_rho": spearman_rho(x, y),
        "kendall_tau_b": kendall_tau_b(x, y),
        "n_pairs": len(shared),
    }


__all__ = [
    "DEFAULT_TRUTH_DIR",
    "NAAC_GRADES",
    "NaacEntry",
    "NaacTruthFile",
    "NIRF_PARAM_KEYS",
    "NirfEntry",
    "NirfTruthFile",
    "TruthError",
    "TruthFile",
    "best_available_hit",
    "kendall_tau_b",
    "load_truth_file",
    "normalise_institution_key",
    "rank_correlation_on_overlap",
    "spearman_rho",
    "top_k_overlap",
]
