import os
import sys
import time
import hashlib
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import quote

import requests


# Hugging Face Hub uses these for its own metadata calls. Defaults are sized for
# a slow link: a dead endpoint reports in 10 s, not the 10-minute default.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ============================================================
# CONFIG
# ============================================================

# Originals were hardcoded; they are now overridable so this file needs no
# edit per model. Defaults below are the values the original script shipped.
REPO_ID = os.environ.get(
    "HF_REPO_ID",
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
)
REPO_SUBDIR = os.environ.get("HF_REPO_SUBDIR", "IQ3_XXS")
REVISION = os.environ.get("HF_REVISION", "main")

# Optional include filter, applied *after* the REPO_SUBDIR prefix is stripped.
# Comma- or whitespace-separated globs (fnmatch syntax, case-insensitive) that
# must match a file name for it to be downloaded:
#
#   HF_INCLUDE="gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
#   HF_INCLUDE="*IQ4_XS.gguf, *UD-Q4_K_XL.gguf"
#
# Empty (the default) means "no filter", i.e. every file inside REPO_SUBDIR,
# which is the original behaviour. A repository that also publishes other
# quants (~65 GB for a single 26B model) needs this to fetch one file.
#
# Globs are matched against the path relative to OUTPUT_DIR first and, failing
# that, against the full repository path, so both `*.gguf` and
# `MTF/mtp-*.gguf` spellings work.
HF_INCLUDE = os.environ.get("HF_INCLUDE", "")


def include_globs():
    """Parse HF_INCLUDE into a list of lowercased fnmatch patterns."""
    return [
        pattern.lower()
        for pattern in HF_INCLUDE.replace(",", " ").split()
        if pattern
    ]

OUTPUT_DIR = Path(
    os.environ.get(
        "HF_OUTPUT_DIR",
        r"/media/archit/Windows/AI/Models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS",
    )
)

# Network behavior
CHUNK_SIZE = 8 * 1024 * 1024          # 8 MiB
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 120

# Stall detection. A body that trickles below this rate for this long is not a
# slow link, it is a dead one: abort the attempt so the resume logic runs again
# instead of sleeping on a socket for hours. Mirrors curl's --speed-limit /
# --speed-time pair used by scripts/fetch_model.sh.
STALL_SPEED_BPS = 8 * 1024            # 8 KiB/s
STALL_TIME_S = 60

# Retry behavior for individual download attempts.
MAX_RETRIES_PER_FILE = 20
INITIAL_RETRY_DELAY = 5
MAX_RETRY_DELAY = 120

# A server Retry-After is honoured, but capped so a bad header cannot park the
# downloader for an hour.
MAX_RETRY_AFTER = 300

# Metadata retries.
MAX_METADATA_RETRIES = 10

# Print progress roughly every N bytes.
PROGRESS_INTERVAL = 256 * 1024 * 1024  # 256 MiB

# Optional integrity verification.
#
# False:
#   Only exact byte size is checked.
#   Fastest because no second full-file read is needed.
#
# True:
#   If HF exposes an LFS SHA256, hash the entire completed file
#   and compare it.
#
# For a 132+ GB model this can take significant additional time,
# so leave False unless you specifically want post-download
# cryptographic verification.
VERIFY_SHA256 = False


# ============================================================
# IMPORT HUGGING FACE API
# ============================================================

try:
    from huggingface_hub import HfApi
except ImportError:
    print()
    print("ERROR: huggingface_hub is not installed.")
    print()
    print("Install it with:")
    print()
    print("    python -m pip install -U huggingface_hub")
    print()
    sys.exit(1)


# ============================================================
# DISPLAY HELPERS
# ============================================================

class RetryAfterError(RuntimeError):
    """The server asked us to wait (429/503). The outer loop sleeps and retries."""

    def __init__(self, delay):
        self.delay = delay
        super().__init__(f"server asked to retry after {delay:.0f}s")


def parse_retry_after(value):
    """``Retry-After`` as delta-seconds or an HTTP date, capped."""
    if not value:
        return None
    value = value.strip()
    try:
        return min(MAX_RETRY_AFTER, max(0.0, float(value)))
    except ValueError:
        pass
    from email.utils import parsedate_to_datetime

    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    from datetime import UTC, datetime

    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return min(
        MAX_RETRY_AFTER,
        max(0.0, (when - datetime.now(UTC)).total_seconds()),
    )


def parse_content_range(value):
    """Return ``(start, end, total)`` from ``bytes start-end/total``."""
    if not value:
        return None
    value = value.strip()
    if not value.lower().startswith("bytes "):
        return None
    try:
        span, total = value[6:].split("/", 1)
        start, end = span.split("-", 1)
        return int(start), int(end), (None if total == "*" else int(total))
    except (ValueError, TypeError):
        return None


def plan_resume(existing, status, content_range, expected_size):
    """Decide what to do with a ranged response. Never append blind.

    Returns ``(action, reason)`` where action is ``append``, ``restart`` or
    ``error``. A 206 whose ``Content-Range`` does not start at the local size,
    or names a different total, is an error: assembling those chunks would
    corrupt the file.
    """
    if status == 206:
        if existing <= 0:
            return "error", "HTTP 206 without a local partial file"
        parsed = parse_content_range(content_range)
        if parsed is None:
            return "error", "HTTP 206 without a usable Content-Range"
        start, _end, total = parsed
        if start != existing:
            return "error", (
                f"Content-Range starts at {start}, local file is {existing} bytes"
            )
        if expected_size and total is not None and total != expected_size:
            return "error", (
                f"Content-Range total {total} != expected {expected_size}"
            )
        return "append", "server accepted resume"
    if status == 200:
        if existing > 0:
            return "restart", "server ignored Range (HTTP 200)"
        return "append", "fresh download"
    if status == 416:
        return "restart", "HTTP 416 Range Not Satisfiable"
    return "error", f"unexpected HTTP {status}"


def format_bytes(num_bytes):
    """Return a human-readable byte size."""

    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]

    value = float(num_bytes)

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PiB"


def format_duration(seconds):
    """Return a human-readable duration."""

    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{minutes}m {seconds}s"

    hours, minutes = divmod(minutes, 60)

    return f"{hours}h {minutes}m"


# ============================================================
# TOKEN
# ============================================================

def get_token():
    """
    Obtain the HF token.

    Priority:
        1. HF_TOKEN environment variable
        2. huggingface_hub's locally stored authentication
        3. interactive prompt
    """

    token = os.environ.get("HF_TOKEN")

    if token:
        return token

    # Try the normal HF login/token storage.
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()

        if token:
            return token

    except Exception:
        pass

    # Newer huggingface_hub versions use get_token().
    try:
        from huggingface_hub import get_token

        token = get_token()

        if token:
            return token

    except Exception:
        pass

    print()
    print("No Hugging Face token was found.")
    print()

    # Headless/detached runs have no tty to prompt at. Public repos need no
    # token, so proceed anonymously instead of blocking on getpass().
    if not sys.stdin.isatty():
        print("No tty: continuing anonymously (public repos need no token).")
        print()
        return ""

    print("Enter your Hugging Face token.")
    print("The token will not be printed.")
    print()

    import getpass

    token = getpass.getpass("HF token: ")

    if not token:
        print()
        print("ERROR: No token supplied.")
        sys.exit(1)

    return token


# ============================================================
# HF METADATA
# ============================================================

def get_repository_files(token):
    """
    Retrieve the authoritative repository file list and sizes.

    IMPORTANT:

    This does NOT use `hf download --dry-run`.

    The purpose of this function is only to answer:

        What files exist in the repository?
        What is the exact byte size of each file?

    Local completion/resume decisions are handled separately.
    """

    print()
    print("=" * 70)
    print("RETRIEVING HUGGING FACE REPOSITORY METADATA")
    print("=" * 70)
    print()
    print(f"Repository : {REPO_ID}")
    print(f"Subfolder  : {REPO_SUBDIR}/")
    print(f"Revision   : {REVISION}")
    if HF_INCLUDE.strip():
        print(f"Include    : {', '.join(include_globs())}")
    print()

    api = HfApi(token=token or None)

    last_error = None

    for attempt in range(1, MAX_METADATA_RETRIES + 1):

        print(
            f"Metadata request "
            f"(attempt {attempt}/{MAX_METADATA_RETRIES})..."
        )

        try:

            # model_info(..., files_metadata=True) gives repository
            # siblings with exact file sizes and LFS information.
            info = api.model_info(
                repo_id=REPO_ID,
                revision=REVISION,
                files_metadata=True,
                token=token or None,
            )

            siblings = getattr(info, "siblings", None)

            if not siblings:
                raise RuntimeError(
                    "Hugging Face returned no repository files."
                )

            files = []

            for sibling in siblings:

                filename = getattr(
                    sibling,
                    "rfilename",
                    None,
                )

                if filename is None:
                    filename = getattr(
                        sibling,
                        "path",
                        None,
                    )

                size = getattr(
                    sibling,
                    "size",
                    None,
                )

                if not filename:
                    continue

                # Only download files inside the requested repository folder.
                # An empty REPO_SUBDIR means "the repository root": no prefix.
                prefix = REPO_SUBDIR.rstrip("/") + "/" if REPO_SUBDIR else ""
                if prefix and not filename.startswith(prefix):
                    continue

                # Keep the full repository path for the download URL,
                # but save the file relative to OUTPUT_DIR.
                repo_path = filename
                filename = filename[len(prefix):]

                if not filename:
                    continue

                # Optional include filter (HF_INCLUDE). Empty = keep everything.
                globs = include_globs()
                if globs:
                    lowered = filename.lower()
                    repo_lowered = repo_path.lower()
                    if not any(
                        fnmatch(lowered, pattern)
                        or fnmatch(repo_lowered, pattern)
                        for pattern in globs
                    ):
                        continue

                if size is None:
                    print(
                        f"WARNING: No size metadata for "
                        f"{filename}; skipping."
                    )
                    continue

                lfs = getattr(
                    sibling,
                    "lfs",
                    None,
                )

                sha256 = None

                if lfs is not None:

                    if isinstance(lfs, dict):
                        sha256 = lfs.get("sha256")
                    else:
                        sha256 = getattr(
                            lfs,
                            "sha256",
                            None,
                        )

                files.append(
                    {
                        "path": filename,
                        "repo_path": repo_path,
                        "size": int(size),
                        "sha256": sha256,
                    }
                )

            if not files:
                raise RuntimeError(
                    f"No files with size metadata were found inside "
                    f"the repository folder '{REPO_SUBDIR}/'. "
                    + (
                        f"HF_INCLUDE={HF_INCLUDE!r} filtered every remaining "
                        f"file out. " if HF_INCLUDE.strip() else ""
                    )
                    + "Check that the folder name exists and is spelled correctly."
                )

            # Stable ordering makes the download order predictable.
            files.sort(
                key=lambda item: item["path"].lower()
            )

            print()
            print(
                f"Metadata successfully retrieved: "
                f"{len(files)} files."
            )

            return files

        except Exception as e:

            last_error = e

            print()
            print(
                f"Metadata request failed:"
                f" {type(e).__name__}: {e}"
            )

            # Auth / missing repo will never improve: fail now rather than
            # backing off ten times on a dead link.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403, 404):
                print()
                print(
                    f"HTTP {status} is not retryable: check HF_REPO_ID, "
                    "HF_REVISION and HF_TOKEN."
                )
                sys.exit(1)

            if attempt < MAX_METADATA_RETRIES:

                delay = min(
                    INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),
                    MAX_RETRY_DELAY,
                )
                if status == 429:
                    hinted = parse_retry_after(
                        getattr(getattr(e, "response", None), "headers", {}).get(
                            "Retry-After"
                        )
                    )
                    if hinted is not None:
                        delay = max(delay, hinted)

                print(
                    f"Retrying in {delay} seconds..."
                )
                print()

                time.sleep(delay)

    print()
    print("=" * 70)
    print("ERROR: COULD NOT RETRIEVE HF REPOSITORY METADATA")
    print("=" * 70)
    print()
    print(
        "The model files already on disk have NOT been touched."
    )
    print()
    print(
        "Last error:"
    )
    print(
        f"  {type(last_error).__name__}: {last_error}"
    )
    print()

    sys.exit(1)


# ============================================================
# LOCAL FILE CLASSIFICATION
# ============================================================

def check_local_files(files):
    """
    Classify every expected repository file.

    COMPLETE:
        local size == expected remote size

    PARTIAL:
        local size < expected remote size

    MISSING:
        local file does not exist

    INVALID:
        local size > expected remote size

    This is deliberately independent of Hugging Face's downloader.
    """

    complete = []
    partial = []
    missing = []
    invalid = []

    for file_info in files:

        filename = file_info["path"]
        expected_size = file_info["size"]

        local_path = OUTPUT_DIR / filename

        if not local_path.exists():

            missing.append(filename)
            continue

        actual_size = local_path.stat().st_size

        if actual_size == expected_size:

            complete.append(filename)

        elif actual_size < expected_size:

            partial.append(
                {
                    "path": filename,
                    "actual_size": actual_size,
                    "expected_size": expected_size,
                }
            )

        else:

            invalid.append(
                {
                    "path": filename,
                    "actual_size": actual_size,
                    "expected_size": expected_size,
                }
            )

    return (
        complete,
        partial,
        missing,
        invalid,
    )


# ============================================================
# LOCAL STATUS DISPLAY
# ============================================================

def print_local_status(files):
    """
    Print the current state of the model directory.
    """

    (
        complete,
        partial,
        missing,
        invalid,
    ) = check_local_files(files)

    print()
    print("=" * 70)
    print("LOCAL FILE CHECK")
    print("=" * 70)

    print(f"Expected files : {len(files)}")
    print(f"Complete       : {len(complete)}")
    print(f"Partial        : {len(partial)}")
    print(f"Missing        : {len(missing)}")
    print(f"Invalid        : {len(invalid)}")

    if partial:

        print()
        print("PARTIAL FILES:")

        for item in partial:

            print(
                f"  {item['path']}"
            )

            print(
                f"      "
                f"{format_bytes(item['actual_size'])}"
                f" / "
                f"{format_bytes(item['expected_size'])}"
            )

    if missing:

        print()
        print("MISSING FILES:")

        for filename in missing:
            print(f"  {filename}")

    if invalid:

        print()
        print("INVALID FILES:")
        print(
            "These are larger than the expected remote files "
            "and will be deleted."
        )

        for item in invalid:

            print(
                f"  {item['path']}: "
                f"{format_bytes(item['actual_size'])} "
                f"> "
                f"{format_bytes(item['expected_size'])}"
            )

            path = OUTPUT_DIR / item["path"]

            try:

                path.unlink()

                print(
                    "      Deleted oversized file."
                )

            except Exception as e:

                print(
                    f"      Could not delete: {e}"
                )

    print("=" * 70)

    return (
        complete,
        partial,
        missing,
        invalid,
    )


# ============================================================
# SHA256
# ============================================================

def calculate_sha256(path):
    """
    Calculate SHA256 for a completed file.

    This intentionally happens only when VERIFY_SHA256=True.
    """

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                32 * 1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def verify_file(file_info):
    """
    Verify size and optionally SHA256.
    """

    filename = file_info["path"]
    expected_size = file_info["size"]
    expected_sha256 = file_info.get("sha256")

    path = OUTPUT_DIR / filename

    if not path.exists():
        return False

    actual_size = path.stat().st_size

    if actual_size != expected_size:

        print()
        print(
            f"VERIFY FAILED: {filename}"
        )

        print(
            f"Expected size: "
            f"{format_bytes(expected_size)}"
        )

        print(
            f"Actual size:   "
            f"{format_bytes(actual_size)}"
        )

        return False

    if VERIFY_SHA256 and expected_sha256:

        print(
            "Calculating SHA256..."
        )

        actual_sha256 = calculate_sha256(path)

        if actual_sha256.lower() != expected_sha256.lower():

            print()
            print(
                "SHA256 MISMATCH!"
            )

            print(
                f"Expected: {expected_sha256}"
            )

            print(
                f"Actual:   {actual_sha256}"
            )

            return False

        print(
            "SHA256 verified."
        )

    return True


# ============================================================
# DIRECT HTTPS DOWNLOAD
# ============================================================

def download_with_https_once(
    token,
    file_info,
):
    """
    Perform ONE HTTP download attempt.

    This function is deliberately resume-aware.

    If the local file is:

        0 bytes:
            download from byte 0

        10 GB:
            request Range: bytes=10737418240-

    If the server honors the Range request:

        HTTP 206

    then data is appended.

    If the server ignores Range:

        HTTP 200

    the existing partial file is discarded and the file
    is restarted from byte 0.

    The function intentionally does NOT retry itself.
    The outer function handles retries.
    """

    filename = file_info["path"]
    expected_size = file_info["size"]

    output_path = OUTPUT_DIR / filename

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Check current local state
    # --------------------------------------------------------

    if output_path.exists():

        existing = output_path.stat().st_size

    else:

        existing = 0

    # Already complete.
    if existing == expected_size:

        print(
            f"Already complete: {filename}"
        )

        return True

    # Oversized file.
    if existing > expected_size:

        print(
            f"Local file is larger than expected."
        )

        print(
            "Deleting it and restarting."
        )

        output_path.unlink()

        existing = 0

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    url = (
        f"https://huggingface.co/"
        f"{REPO_ID}/resolve/"
        f"{REVISION}/"
        f"{quote(file_info.get('repo_path', filename), safe='/')}"
    )

    headers = {
        "Accept-Encoding": "identity",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if existing > 0:

        headers["Range"] = (
            f"bytes={existing}-"
        )

        print()
        print(
            f"Resuming {filename}"
        )

        print(
            f"Local data: "
            f"{format_bytes(existing)}"
        )

        print(
            f"Remaining:  "
            f"{format_bytes(expected_size - existing)}"
        )

    else:

        print()
        print(
            f"Starting {filename}"
        )

        print(
            f"Size: "
            f"{format_bytes(expected_size)}"
        )

    # --------------------------------------------------------
    # HTTP request
    # --------------------------------------------------------

    with requests.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
        ),
    ) as response:

        # ----------------------------------------------------
        # RATE LIMIT / RETRY-AFTER
        # ----------------------------------------------------

        if response.status_code in (429, 503):
            raise RetryAfterError(
                parse_retry_after(response.headers.get("Retry-After"))
                or INITIAL_RETRY_DELAY
            )

        # ----------------------------------------------------
        # RANGE RESPONSE HANDLING
        # ----------------------------------------------------

        action, reason = plan_resume(
            existing,
            response.status_code,
            response.headers.get("Content-Range"),
            expected_size,
        )

        if action == "error":
            response.raise_for_status()
            raise RuntimeError(
                f"Unsafe resume for {filename}: {reason}"
            )

        if action == "restart":
            print()
            print(f"{reason}.")
            print("Restarting this file from byte 0.")
            response.close()
            if response.status_code == 416 and output_path.exists():
                if output_path.stat().st_size == expected_size:
                    print(
                        "HTTP 416 received, but local file "
                        "is already complete."
                    )
                    return True
            if output_path.exists():
                output_path.unlink()
            # Retry the same file from scratch; existing is now 0.
            return download_with_https_once(token, file_info)

        if existing > 0 and response.status_code == 206:
            mode = "ab"
            received = existing
            print(f"Server accepted resume: HTTP 206 ({reason})")
            print(
                f"Content-Range: {response.headers.get('Content-Range')}"
            )
        else:
            mode = "wb"
            received = 0
            print(f"Server response: HTTP {response.status_code}")

        # ----------------------------------------------------
        # SIZE AGREEMENT BEFORE WRITING A BYTE
        # ----------------------------------------------------

        declared = response.headers.get("Content-Length")
        if declared is not None and expected_size:
            try:
                declared_int = int(declared)
            except ValueError:
                declared_int = None
            if declared_int is not None:
                want = expected_size - received
                if declared_int != want:
                    raise RuntimeError(
                        f"Refusing to write {filename}: server declares "
                        f"{declared_int} bytes, expected {want}. "
                        "The file changed upstream or the range is wrong."
                    )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        start_time = time.time()

        last_progress = received

        stall_started = time.monotonic()
        stall_bytes = 0

        with open(output_path, mode) as f:

            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):

                if not chunk:
                    continue

                f.write(chunk)

                received += len(chunk)
                stall_bytes += len(chunk)

                # --------------------------------------------------
                # STALL DETECTION (curl --speed-limit/--speed-time)
                # --------------------------------------------------
                stall_elapsed = time.monotonic() - stall_started
                if stall_elapsed >= STALL_TIME_S:
                    rate = stall_bytes / stall_elapsed
                    if rate < STALL_SPEED_BPS:
                        raise ConnectionError(
                            f"Transfer stalled at {format_bytes(rate)}/s "
                            f"for {stall_elapsed:.0f}s; resuming from "
                            f"{format_bytes(received)}."
                        )
                    stall_started = time.monotonic()
                    stall_bytes = 0

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        start_time = time.time()

        last_progress = received

        with open(output_path, mode) as f:

            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):

                if not chunk:
                    continue

                f.write(chunk)

                received += len(chunk)

                # Print progress periodically.
                if (
                    received - last_progress
                    >= PROGRESS_INTERVAL
                ):

                    elapsed = (
                        time.time() - start_time
                    )

                    transferred = (
                        received - existing
                    )

                    if elapsed > 0:

                        speed = (
                            transferred / elapsed
                        )

                        remaining = (
                            expected_size
                            - received
                        )

                        if speed > 0:

                            eta = (
                                remaining / speed
                            )

                            eta_string = (
                                format_duration(eta)
                            )

                        else:

                            eta_string = "unknown"

                        print(
                            f"  Progress: "
                            f"{format_bytes(received)}"
                            f" / "
                            f"{format_bytes(expected_size)}"
                            f" | "
                            f"{format_bytes(speed)}/s"
                            f" | ETA "
                            f"{eta_string}"
                        )

                    last_progress = received

        # ----------------------------------------------------
        # POST-DOWNLOAD SIZE CHECK
        # ----------------------------------------------------

        final_size = output_path.stat().st_size

        print()

        print(
            f"Connection ended."
        )

        print(
            f"Local file now: "
            f"{format_bytes(final_size)}"
            f" / "
            f"{format_bytes(expected_size)}"
        )

        if final_size == expected_size:

            return True

        if final_size < expected_size:

            raise ConnectionError(
                f"Download interrupted at "
                f"{format_bytes(final_size)} "
                f"of "
                f"{format_bytes(expected_size)}."
            )

        raise RuntimeError(
            f"Downloaded file is larger than expected: "
            f"{final_size} > {expected_size}."
        )


# ============================================================
# ROBUST FILE DOWNLOADER
# ============================================================

def download_file(token, file_info):
    """
    Keep retrying ONE file until it is complete or the retry
    budget is exhausted.

    Crucially, every retry re-checks the actual local file
    size. Therefore an interrupted transfer resumes from
    exactly where the previous attempt stopped.
    """

    filename = file_info["path"]
    expected_size = file_info["size"]

    print()
    print("#" * 70)
    print(
        f"FILE: {filename}"
    )
    print(
        f"EXPECTED SIZE: {format_bytes(expected_size)}"
    )
    print("#" * 70)

    for attempt in range(
        1,
        MAX_RETRIES_PER_FILE + 1,
    ):

        print()
        print(
            f"Download attempt "
            f"{attempt}/{MAX_RETRIES_PER_FILE}"
        )

        try:

            success = download_with_https_once(
                token,
                file_info,
            )

            if not success:
                raise RuntimeError(
                    "Download function returned failure."
                )

            # ------------------------------------------------
            # Final local verification
            # ------------------------------------------------

            if not verify_file(file_info):

                raise RuntimeError(
                    "Downloaded file failed verification."
                )

            print()
            print(
                f"SUCCESS: {filename}"
            )

            print(
                f"Verified size: "
                f"{format_bytes(expected_size)}"
            )

            return True

        except KeyboardInterrupt:

            print()
            print(
                "Download interrupted by user."
            )

            print(
                "The partial file has been preserved."
            )

            raise

        except RetryAfterError as e:

            # Honor the server's own pacing instead of guessing at backoff.
            if attempt >= MAX_RETRIES_PER_FILE:
                print()
                print("Maximum retry count reached.")
                return False
            delay = min(MAX_RETRY_DELAY, max(e.delay, INITIAL_RETRY_DELAY))
            print()
            print(f"Rate limited: {e}.")
            print(f"Waiting {delay:.0f} seconds before retry...")
            time.sleep(delay)
            continue

        except Exception as e:

            print()
            print(
                f"DOWNLOAD ATTEMPT FAILED:"
            )

            print(
                f"  {type(e).__name__}: {e}"
            )

            # Show exactly what survived the failure.
            if (
                OUTPUT_DIR / filename
            ).exists():

                current_size = (
                    OUTPUT_DIR / filename
                ).stat().st_size

                print()
                print(
                    f"Partial data preserved: "
                    f"{format_bytes(current_size)}"
                    f" / "
                    f"{format_bytes(expected_size)}"
                )

            if attempt >= MAX_RETRIES_PER_FILE:

                print()
                print(
                    "Maximum retry count reached."
                )

                return False

            delay = min(
                INITIAL_RETRY_DELAY
                * (2 ** (attempt - 1)),
                MAX_RETRY_DELAY,
            )

            print()
            print(
                f"Waiting {delay} seconds before retry..."
            )

            time.sleep(delay)

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("HF MODEL DOWNLOAD - RESUMABLE")
    print("=" * 70)
    print()

    print(
        f"Repository : {REPO_ID}"
    )

    print(
        f"Revision   : {REVISION}"
    )

    print(
        f"Output     : {OUTPUT_DIR}"
    )

    print()

    print(
        "Download strategy:"
    )

    print(
        "  - HF API metadata"
    )

    print(
        "  - Local byte-size verification"
    )

    print(
        "  - HTTP Range resume"
    )

    print(
        "  - Automatic per-file retries"
    )

    print(
        "  - Partial files are preserved"
    )

    print()

    # --------------------------------------------------------
    # TOKEN
    # --------------------------------------------------------

    token = get_token()

    # --------------------------------------------------------
    # REPOSITORY METADATA
    # --------------------------------------------------------

    files = get_repository_files(token)

    # --------------------------------------------------------
    # LOCAL STATUS
    # --------------------------------------------------------

    (
        complete,
        partial,
        missing,
        invalid,
    ) = print_local_status(files)

    # --------------------------------------------------------
    # DETERMINE WORK
    # --------------------------------------------------------

    incomplete = []

    complete_set = set(complete)

    for file_info in files:

        filename = file_info["path"]

        if filename not in complete_set:

            incomplete.append(file_info)

    print()

    print("=" * 70)
    print("DOWNLOAD PLAN")
    print("=" * 70)

    print(
        f"Total repository files : {len(files)}"
    )

    print(
        f"Already complete       : {len(complete)}"
    )

    print(
        f"Need downloading       : {len(incomplete)}"
    )

    print("=" * 70)

    if not incomplete:

        print()
        print(
            "ALL FILES ARE ALREADY COMPLETE."
        )

        print()

        return

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    failed = []

    for index, file_info in enumerate(
        incomplete,
        start=1,
    ):

        filename = file_info["path"]

        print()
        print()
        print(
            "=" * 70
        )

        print(
            f"DOWNLOAD "
            f"{index}/{len(incomplete)}"
        )

        print(
            filename
        )

        print(
            "=" * 70
        )

        success = download_file(
            token,
            file_info,
        )

        if not success:

            failed.append(filename)

            print()
            print(
                f"FAILED: {filename}"
            )

            print(
                "Moving to the next file."
            )

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    final_complete = []
    final_partial = []
    final_missing = []
    final_invalid = []

    (
        final_complete,
        final_partial,
        final_missing,
        final_invalid,
    ) = check_local_files(files)

    print(
        f"Complete : {len(final_complete)}"
    )

    print(
        f"Partial  : {len(final_partial)}"
    )

    print(
        f"Missing  : {len(final_missing)}"
    )

    print(
        f"Invalid  : {len(final_invalid)}"
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if (
        len(final_complete)
        == len(files)
    ):

        print()
        print("=" * 70)
        print(
            "ALL MODEL FILES ARE COMPLETE."
        )
        print("=" * 70)
        print()

        print(
            f"Repository : {REPO_ID}"
        )

        print(
            f"Files      : {len(files)}"
        )

        print(
            f"Location   : {OUTPUT_DIR}"
        )

        print()

        return

    # --------------------------------------------------------
    # FAILURE REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "DOWNLOAD INCOMPLETE"
    )
    print("=" * 70)

    if failed:

        print()
        print(
            "Files that exhausted their retry budget:"
        )

        for filename in failed:

            print(
                f"  - {filename}"
            )

    if final_partial:

        print()
        print(
            "PARTIAL FILES:"
        )

        for item in final_partial:

            print(
                f"  - {item['path']}: "
                f"{format_bytes(item['actual_size'])}"
                f" / "
                f"{format_bytes(item['expected_size'])}"
            )

    if final_missing:

        print()
        print(
            "MISSING FILES:"
        )

        for filename in final_missing:

            print(
                f"  - {filename}"
            )

    print()
    print(
        "Run the script again to continue."
    )

    print(
        "Existing partial files will be resumed "
        "from their current byte offsets."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
