"""Fetch a NIRF ranking page and ingest it as a versioned truth file.

Run with the pipeline venv:

    cd packages/pipeline && uv run python ../../scripts/eval/fetch_nirf_truth.py \
        --year 2025 --category medical --out-dir ../../data/truth/nirf

What it does:

1. Fetches ``https://www.nirfindia.org/Rankings/<year>/<Category>Ranking.html``
   with a truthful UA first. ``nirfindia.org`` answers the default crawler UA
   with a CloudFront 403 (see ``fmd/adapters/nirf.py``), so the fetch escalates
   to a browser UA on that exact block signal — the same ladder the pipeline
   uses, and no auth/CAPTCHA bypass (invariant 9).
2. Parses the embedded DataTable rows: each institution row carries
   Institute ID, Name (+ hidden parameter table TLR/RPC/GO/OI/PERCEPTION and
   the per-institute PDF link), City, State, Score, Rank.
3. Writes the raw HTML bytes to ``<out-dir>/raw/<year>_<category>Ranking.html``
   and the parsed truth YAML to ``<out-dir>/nirf_<year>_<category>.yaml`` with
   full provenance (truth_version, scope, as_of, source_url, source_sha256,
   fetched_at). ``source_sha256`` is the hash of the committed raw file, so a
   reader can re-verify every row offline.

Nothing here invents a rank: rows that do not parse cleanly abort the run and
write nothing. See docs/v1/eval/03-truth-ingestion.md for the contract the
loader (fmd/eval/truth.py) enforces on the produced file.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import sys
import unicodedata
from html import unescape
from pathlib import Path
from urllib import request as _urlrequest

TRUTHFUL_UA = "findMyDreamCollege-truth-ingest/0.1 (public rank lists; contact: research@findmydreamcollege.invalid)"
ESCALATED_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
PARAM_KEYS = ("tlr", "rpc", "go", "oi", "pr")

CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
INST_START_RE = re.compile(r"<td>\s*(IR-[A-Z0-9-]+)\s*</td>")


def http_get(url: str) -> bytes:
    """Fetch bytes with the truthful UA, escalating to a browser UA on a 403."""

    for ua in (TRUTHFUL_UA, ESCALATED_UA):
        req = _urlrequest.Request(url, headers={"User-Agent": ua})
        try:
            with _urlrequest.urlopen(req, timeout=120) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - ingestion script, report and continue
            status = getattr(exc, "code", None)
            if status == 403 and ua == TRUTHFUL_UA:
                continue  # documented CloudFront block; escalate once
            raise SystemExit(f"fetch failed for {url}: {exc}") from exc
    raise SystemExit(f"fetch failed for {url}: unreachable")


def strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("More Details", " ").replace("Close", " ")
    # The name cell appends "|" separators for the hidden More Details / PDF
    # / Close anchors; a real institution name never ends in pipes.
    text = re.sub(r"\|", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_hidden_block(row_html: str) -> tuple[str, dict[str, float], str | None]:
    """Split the hidden parameter table out of the row before cell matching.

    The Name cell embeds a ``tbl_hidden`` <div> holding the parameter table
    (TLR/RPC/GO/OI/PERCEPTION) and the per-institute PDF link. Its nested
    <td>s would otherwise confuse the main-cell regex, so it is removed first.
    """

    hidden = re.search(r'<div class="tbl_hidden"[^>]*>(.*?)</div>', row_html, re.S)
    params: dict[str, float] = {}
    pdf: str | None = None
    # The per-institute PDF link lives in the "More Details" button div, before
    # the hidden table, so it is searched in the full segment, not the block.
    pdf_found = re.search(r'href="(https://www\.nirfindia\.org/nirfpdfcdn/[^"]+\.pdf)"', row_html)
    pdf = pdf_found.group(1) if pdf_found else None
    if hidden:
        block = hidden.group(1)
        values = re.findall(r"<td[^>]*>\s*([0-9]+(?:\.[0-9]+)?)\s*</td>", block)
        if len(values) < len(PARAM_KEYS):
            raise ValueError(
                f"hidden parameter table has {len(values)} numeric cells, expected 5"
            )
        params = {key: float(value) for key, value in zip(PARAM_KEYS, values, strict=True)}
        row_html = row_html.replace(hidden.group(0), "")
    return row_html, params, pdf


def normalise_institution_key(name: str) -> str:
    """Shared join key rule (mirrors fmd.eval.truth.normalise_institution_key).

    Lowercase, ASCII-fold, '&' -> 'and', strip punctuation, collapse spaces.
    Documented in docs/v1/eval/03-truth-ingestion.md; the loader recomputes and
    re-checks this key, so a divergence here fails ingestion, not the eval.
    """

    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_ranking(html: str) -> list[dict[str, object]]:
    """Slice the page between institution-ID anchors and parse each segment.

    Row-level <tr> regexes cannot be used because each institution row embeds a
    hidden parameter table with its own <tr>s; an anchor-to-anchor slice is the
    only cut that always contains exactly one institution.
    """

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    anchors = list(INST_START_RE.finditer(html))
    if not anchors:
        raise ValueError("no Institute ID anchors found; page structure may have changed")
    for index, anchor in enumerate(anchors):
        end = anchors[index + 1].start() if index + 1 < len(anchors) else len(html)
        inst_id = anchor.group(1)
        if inst_id in seen_ids:
            raise ValueError(f"duplicate Institute ID in page: {inst_id}")
        seen_ids.add(inst_id)
        segment, params, pdf = _strip_hidden_block(html[anchor.start() : end])
        cells = CELL_RE.findall(segment)
        if len(cells) < 6:
            raise ValueError(f"{inst_id}: only {len(cells)} cells parsed, expected >= 6")
        name = strip_tags(cells[1])
        if not name:
            raise ValueError(f"{inst_id}: empty institution name")
        city = strip_tags(cells[2])
        state = strip_tags(cells[3])
        score_text = strip_tags(cells[4])
        rank_text = strip_tags(cells[5])
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", score_text):
            raise ValueError(f"{inst_id}: unparsable score {score_text!r}")
        if not re.fullmatch(r"[0-9]+", rank_text):
            raise ValueError(f"{inst_id}: unparsable rank {rank_text!r}")
        if len(params) != len(PARAM_KEYS):
            raise ValueError(f"{inst_id}: hidden parameter table missing")
        rows.append(
            {
                "institution_id": inst_id,
                "institution_key": normalise_institution_key(name),
                "name": name,
                "city": city,
                "state": state,
                "rank": int(rank_text),
                "score": float(score_text),
                "params": params,
                "source_pdf": pdf,
            }
        )
    if not rows:
        raise ValueError("no institution rows parsed; page structure may have changed")
    ranks = [int(row["rank"]) for row in rows]  # type: ignore[arg-type]
    # NIRF ties rank institutions with equal scores (e.g. two at rank 27), so
    # the sequence is non-decreasing with repeats, not strictly 1..N.
    if ranks[0] != 1 or any(b < a for a, b in zip(ranks, ranks[1:])):
        raise ValueError(f"rank sequence is not a non-decreasing series starting at 1: {ranks}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="ranking edition year")
    parser.add_argument(
        "--category", choices=["medical", "overall", "university", "college"], required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="e.g. data/truth/nirf")
    parser.add_argument("--url", default=None, help="override the page URL")
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=None,
        help="parse an already-fetched page copy instead of re-fetching (rate-limit safety)",
    )
    parser.add_argument(
        "--truth-version", type=int, default=1, help="bump when the same edition is re-ingested"
    )
    args = parser.parse_args(argv)

    url = args.url or (
        f"https://www.nirfindia.org/Rankings/{args.year}/"
        f"{args.category.capitalize()}Ranking.html"
    )
    if args.raw_file is not None:
        raw = args.raw_file.read_bytes()
        if not raw:
            raise SystemExit(f"--raw-file is empty: {args.raw_file}")
    else:
        raw = http_get(url)
    text = raw.decode("utf-8", errors="replace")
    entries = parse_ranking(text)

    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_name = f"{args.year}_{args.category.capitalize()}Ranking.html"
    raw_path = raw_dir / raw_name
    raw_path.write_bytes(raw)
    sha256 = hashlib.sha256(raw).hexdigest()

    fetched_at = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    lines: list[str] = [
        "# NIRF ranking truth file — ingested by scripts/eval/fetch_nirf_truth.py.",
        "# Institution-level proxy ONLY: NIRF publishes no paramedical/MLT category,",
        "# so a rank may never be attributed to a programme (invariant 3/6).",
        "schema_version: 1",
        "truth_kind: nirf_ranking",
        f"truth_version: {args.truth_version}",
        f"category: {args.category}",
        f"year: {args.year}",
        "scope: institution",
        f"as_of: \"{args.year}\"",
        f"source_url: {url}",
        f"fetched_at: \"{fetched_at}\"",
        f"source_sha256: {sha256}",
        f"raw_file: {raw_path.as_posix()}",
        "ingest_tool: scripts/eval/fetch_nirf_truth.py",
        "notes: >-",
        "  Rows parsed from the embedded DataTable of the ranking page. The raw HTML",
        "  is committed beside this file; source_sha256 hashes those exact bytes.",
        "entries:",
    ]
    for entry in entries:
        params = entry["params"]
        params_yaml = ", ".join(f"{key}: {value}" for key, value in params.items())  # type: ignore[attr-defined]
        pdf = entry["source_pdf"]
        lines.append(f"  - institution_id: {entry['institution_id']}")
        lines.append(f"    institution_key: {entry['institution_key']}")
        lines.append(f"    name: {entry['name']}")
        lines.append(f"    city: {entry['city']}")
        lines.append(f"    state: {entry['state']}")
        lines.append(f"    rank: {entry['rank']}")
        lines.append(f"    score: {entry['score']}")
        lines.append(f"    params: {{{params_yaml}}}")
        lines.append(f"    source_pdf: {pdf if pdf else 'null'}")

    out_path = args.out_dir / f"nirf_{args.year}_{args.category}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ingested {len(entries)} institutions from {url}")
    print(f"  raw:  {raw_path} (sha256 {sha256[:12]}...)")
    print(f"  truth: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
